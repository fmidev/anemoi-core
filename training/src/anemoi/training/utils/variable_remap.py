# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Carry a checkpoint's variable-indexed tensors over to a model with a different variable set.

Adding or removing a variable changes the shape of exactly the tensors that have one slot
per variable: the input embeddings of the encoder and decoder (one column per input variable
per input step, followed by a variable-free tail of node attributes, the forecast step and,
with ``condition_on_residual``, one column per prognostic variable), the final layer of the
decoder (one row per output variable per output step) and the normaliser's per-variable
statistics. Stock transfer learning drops every tensor whose shape changed, which
re-initialises those layers completely.

:func:`remap_variable_tensors` instead rebuilds each of them in the new layout, copying the
checkpoint's slot for every variable the two share - matched by NAME, so variables that sort
into the middle of the list are handled - and zero-initialising the slots of new variables.
Zero input columns make a new variable invisible to the model until it is trained, so the
warm-started model reproduces the checkpoint exactly for the shared outputs; zero output rows
start a new prognostic variable at persistence (the residual connection adds the input).
Normaliser statistics of shared variables are kept from the checkpoint, the statistics the
model was trained against; new variables take the ones the model computed from its dataset.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from dataclasses import field
from typing import Any

import torch
from torch import nn

LOGGER = logging.getLogger(__name__)

NORMALISER_STATISTICS = ("_norm_mul", "_norm_add")
PROCESSOR_CONTAINERS = ("pre_processors", "post_processors", "pre_processors_tendencies", "post_processors_tendencies")


class VariableRemapError(ValueError):
    """The checkpoint cannot be remapped onto the model's variable layout."""


@dataclass
class DatasetRemap:
    """What the remap did for one dataset."""

    added: list[str]
    removed: list[str]
    remapped_keys: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class _Layout:
    full: list[str]
    model_input: list[str]
    model_output: list[str]
    prognostic_input: list[str]

    @classmethod
    def from_indices(cls, indices: Any) -> _Layout:
        full = sorted(indices.name_to_index, key=indices.name_to_index.get)
        model_input = [full[i] for i in _as_list(indices.data.input.full)]
        model_output = [full[i] for i in _as_list(indices.data.output.full)]
        prognostic_input = [model_input[i] for i in _as_list(indices.model.input.prognostic)]
        return cls(full, model_input, model_output, prognostic_input)

    def role(self, name: str) -> str:
        is_input, is_output = name in self.model_input, name in self.model_output
        if is_input and is_output:
            return "prognostic"
        if is_input:
            return "forcing"
        if is_output:
            return "diagnostic"
        return "target"


def _as_list(index: Any) -> list[int]:
    return [int(i) for i in (index.tolist() if hasattr(index, "tolist") else index)]


def _copy_block(
    old: torch.Tensor,
    new: torch.Tensor,
    dim: int,
    old_start: int,
    old_names: list[str],
    new_start: int,
    new_names: list[str],
) -> None:
    """Copy the slots of shared names from ``old`` into ``new`` along ``dim``."""
    old_position = {name: i for i, name in enumerate(old_names)}
    for new_i, name in enumerate(new_names):
        if name in old_position:
            new.select(dim, new_start + new_i).copy_(old.select(dim, old_start + old_position[name]))


def _remap_input_columns(
    old: torch.Tensor,
    shape: torch.Size,
    old_layout: _Layout,
    new_layout: _Layout,
    n_step_input: int,
    tail: int,
    condition_on_residual: bool,
) -> torch.Tensor:
    """Rebuild a (out, input_dim) weight in the new input layout."""
    new = torch.zeros(shape, dtype=old.dtype)
    n_old, n_new = len(old_layout.model_input), len(new_layout.model_input)
    for step in range(n_step_input):
        _copy_block(old, new, 1, step * n_old, old_layout.model_input, step * n_new, new_layout.model_input)
    # node attributes and the forecast step: no variable dimension, copied as they are
    old_tail, new_tail = n_step_input * n_old, n_step_input * n_new
    new[:, new_tail : new_tail + tail] = old[:, old_tail : old_tail + tail]
    if condition_on_residual:
        _copy_block(
            old,
            new,
            1,
            old_tail + tail,
            old_layout.prognostic_input,
            new_tail + tail,
            new_layout.prognostic_input,
        )
    return new


def _remap_output_rows(
    old: torch.Tensor,
    shape: torch.Size,
    old_layout: _Layout,
    new_layout: _Layout,
    n_step_output: int,
) -> torch.Tensor:
    """Rebuild an (output_dim, ...) weight or bias in the new output layout."""
    new = torch.zeros(shape, dtype=old.dtype)
    n_old, n_new = len(old_layout.model_output), len(new_layout.model_output)
    for step in range(n_step_output):
        _copy_block(old, new, 0, step * n_old, old_layout.model_output, step * n_new, new_layout.model_output)
    return new


def _linears(module: nn.Module) -> list[tuple[str, nn.Module]]:
    return [
        (name, sub)
        for name, sub in module.named_modules()
        if hasattr(sub, "in_features") and hasattr(sub, "out_features") and getattr(sub, "weight", None) is not None
    ]


def _check_roles(dataset_name: str, old_layout: _Layout, new_layout: _Layout) -> None:
    changed = {
        name: (old_layout.role(name), new_layout.role(name))
        for name in old_layout.full
        if name in new_layout.full and old_layout.role(name) != new_layout.role(name)
    }
    if changed:
        details = ", ".join(f"{name}: {old} -> {new}" for name, (old, new) in sorted(changed.items()))
        msg = (
            f"Dataset {dataset_name!r}: variables changed role between checkpoint and config ({details}). "
            "Their trained weights mean something different in the new role (a prognostic output is a "
            "tendency on top of the residual, a diagnostic output is the full value), so they are not "
            "remapped. Keep their roles, or drop them from one side."
        )
        raise VariableRemapError(msg)


def _remap_dataset(
    state_dict: dict[str, torch.Tensor],
    model_state: dict[str, torch.Tensor],
    inner: nn.Module,
    inner_prefix: str,
    interface_prefix: str,
    dataset_name: str,
    old_layout: _Layout,
    new_layout: _Layout,
) -> list[str]:
    n_step_input = int(inner.n_step_input)
    n_step_output = int(inner.n_step_output)
    condition_on_residual = bool(getattr(inner, "condition_on_residual", False)) and dataset_name not in (
        getattr(inner, "residuals_to_skip", None) or []
    )
    n_prognostic_new = len(new_layout.prognostic_input) if condition_on_residual else 0
    n_prognostic_old = len(old_layout.prognostic_input) if condition_on_residual else 0

    input_dim_new = int(inner.input_dim[dataset_name])
    tail = input_dim_new - n_step_input * len(new_layout.model_input) - n_prognostic_new
    input_dim_old = n_step_input * len(old_layout.model_input) + tail + n_prognostic_old
    output_dim_new = n_step_output * len(new_layout.model_output)
    output_dim_old = n_step_output * len(old_layout.model_output)

    remapped: list[str] = []
    found = {"input": 0, "output": 0}
    for part in ("encoder", "decoder"):
        container = getattr(inner, part, None)
        if container is None or dataset_name not in container:
            continue
        for name, linear in _linears(container[dataset_name]):
            prefix = f"{inner_prefix}{part}.{dataset_name}.{name}."
            weight_key, bias_key = prefix + "weight", prefix + "bias"
            if weight_key not in state_dict:
                continue
            old_weight = state_dict[weight_key]
            if linear.in_features == input_dim_new and old_weight.shape[1] == input_dim_old:
                state_dict[weight_key] = _remap_input_columns(
                    old_weight,
                    model_state[weight_key].shape,
                    old_layout,
                    new_layout,
                    n_step_input,
                    tail,
                    condition_on_residual,
                )
                remapped.append(weight_key)
                found["input"] += 1
            elif part == "decoder" and linear.out_features == output_dim_new and old_weight.shape[0] == output_dim_old:
                keys = [weight_key] + ([bias_key] if bias_key in state_dict else [])
                for key in keys:
                    state_dict[key] = _remap_output_rows(
                        state_dict[key],
                        model_state[key].shape,
                        old_layout,
                        new_layout,
                        n_step_output,
                    )
                remapped.extend(keys)
                found["output"] += 1

    if found["input"] == 0 or found["output"] == 0:
        msg = (
            f"Dataset {dataset_name!r}: could not find the variable-indexed layers "
            f"(input embeddings found: {found['input']}, output layers found: {found['output']}; "
            f"expected input_dim {input_dim_old} -> {input_dim_new}, output_dim {output_dim_old} -> {output_dim_new}). "
            "Refusing to fall back to random initialisation."
        )
        raise VariableRemapError(msg)

    processor_pattern = re.compile(
        rf"^{re.escape(interface_prefix)}(?:{'|'.join(PROCESSOR_CONTAINERS)})\.{re.escape(dataset_name)}\.(?P<rest>.+)$",
    )
    for key in list(state_dict):
        match = processor_pattern.match(key)
        if match is None or key not in model_state:
            continue
        if key.rsplit(".", 1)[-1] in NORMALISER_STATISTICS and state_dict[key].ndim == 1:
            if state_dict[key].shape[0] != len(old_layout.full):
                msg = f"{key}: expected {len(old_layout.full)} statistics, found {tuple(state_dict[key].shape)}"
                raise VariableRemapError(msg)
            new = model_state[key].detach().clone()
            _copy_block(state_dict[key], new, 0, 0, old_layout.full, 0, new_layout.full)
            state_dict[key] = new
        else:
            # index buffers and anything else describing the layout: the new model's
            state_dict[key] = model_state[key].detach().clone()
        remapped.append(key)

    return remapped


def remap_variable_tensors(
    state_dict: dict[str, torch.Tensor],
    model: nn.Module,
    ckpt_data_indices: dict[str, Any],
    data_indices: dict[str, Any],
    inner_path: str = "model.model",
) -> dict[str, DatasetRemap]:
    """Rewrite ``state_dict`` in place so its variable-indexed tensors fit ``model``.

    Parameters
    ----------
    state_dict : dict[str, torch.Tensor]
        Checkpoint state dict, keyed like ``model.state_dict()``. Modified in place.
    model : nn.Module
        The module the state dict will be loaded into (the training module).
    ckpt_data_indices : dict[str, IndexCollection]
        Per-dataset data indices recorded in the checkpoint.
    data_indices : dict[str, IndexCollection]
        Per-dataset data indices of the new configuration.
    inner_path : str
        Path from ``model`` to the encoder-processor-decoder. Its parent holds the
        pre/post-processors.

    Returns
    -------
    dict[str, DatasetRemap]
        One entry per dataset whose variables differ; empty when nothing had to change.
    """
    inner = model.get_submodule(inner_path)
    inner_prefix = f"{inner_path}."
    interface_prefix = f"{inner_path.rsplit('.', 1)[0]}." if "." in inner_path else ""
    model_state = model.state_dict()

    report: dict[str, DatasetRemap] = {}
    for dataset_name, indices in data_indices.items():
        if dataset_name not in ckpt_data_indices:
            continue
        old_layout = _Layout.from_indices(ckpt_data_indices[dataset_name])
        new_layout = _Layout.from_indices(indices)
        if old_layout == new_layout:
            continue
        _check_roles(dataset_name, old_layout, new_layout)

        remapped = _remap_dataset(
            state_dict,
            model_state,
            inner,
            inner_prefix,
            interface_prefix,
            dataset_name,
            old_layout,
            new_layout,
        )
        entry = DatasetRemap(
            added=[name for name in new_layout.full if name not in old_layout.full],
            removed=[name for name in old_layout.full if name not in new_layout.full],
            remapped_keys=remapped,
        )
        report[dataset_name] = entry
        LOGGER.info(
            "Remapped variables of dataset %r by name: added %s (zero-initialised), removed %s; %d tensors rebuilt",
            dataset_name,
            entry.added,
            entry.removed,
            len(remapped),
        )
    return report
