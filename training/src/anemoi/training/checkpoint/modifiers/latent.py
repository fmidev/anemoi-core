# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Checkpoint validation for frozen latent representations."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

import torch
from omegaconf import DictConfig
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from anemoi.training.checkpoint.base import CheckpointContext
    from anemoi.training.schemas.base_schema import BaseSchema
    from anemoi.training.train.methods.base import BaseTrainingModule

from anemoi.training.checkpoint.modifiers.base import ModelModifier

CODEC_PREFIXES = (
    "model.model.encoder.",
    "model.model.encoder_graph_provider.",
    "model.model.node_attributes.",
    "model.model.latent_aggregator.",
    "model.pre_processors.",
    "model.post_processors.",
)
CONFIG_PATHS = (
    "model.encoders",
    "model.trainable_parameters",
    "model.attributes.nodes",
    "model.model.hidden_nodes_name",
    "model.latent.normalize_state",
    "data.datasets",
)


def configuration(config: BaseSchema | DictConfig) -> DictConfig:
    return config if isinstance(config, DictConfig) else OmegaConf.create(config.model_dump(by_alias=True))


def representation_fingerprint(module: BaseTrainingModule) -> str:
    """Identity of encoder weights, preprocessing, layout and immutable geometry."""
    config = configuration(module.config)
    signature = {
        path: OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
        for path in CONFIG_PATHS
        if (value := OmegaConf.select(config, path)) is not None
    }
    for encoder in signature.get("model.encoders", {}).values():
        for setting in ("num_chunks", "gradient_checkpointing", "cpu_offload"):
            encoder["mapper"].pop(setting, None)
    signature["variables"] = {name: list(index.data_full_ordered_names) for name, index in module.data_indices.items()}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, default=str).encode())
    for key, value in sorted(module.state_dict().items()):
        if key.startswith(CODEC_PREFIXES):
            digest.update(key.encode())
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    core = module.model.model
    for name in [*core.input_datasets, core._graph_name_hidden]:
        digest.update(core._graph_data[name].x.detach().cpu().contiguous().numpy().tobytes())
    for name in core.input_datasets:
        digest.update(
            core._graph_data[name, "to", core._graph_name_hidden].edge_index.cpu().contiguous().numpy().tobytes(),
        )
    return digest.hexdigest()


class VerifyLatentRepresentation(ModelModifier):
    """Verify encoder state, preprocessing and geometry against a source checkpoint."""

    def process(self, context: CheckpointContext) -> CheckpointContext:
        source = context.checkpoint_data
        expected = source.get("latent_representation_id")
        if expected is None:
            message = (
                "Source checkpoint has no latent representation identity; use a stage-1 latent-forecaster checkpoint."
            )
            raise ValueError(message)
        state = source["state_dict"]
        for key, value in context.model.state_dict().items():
            if key.startswith(CODEC_PREFIXES) and (key not in state or not torch.equal(value.cpu(), state[key].cpu())):
                message = "Frozen representation tensor did not load exactly: " + key
                raise ValueError(message)
        actual = representation_fingerprint(context.model)
        if actual != expected:
            message = "Encoder/preprocessing/geometry configuration differs from the source representation."
            raise ValueError(message)
        context.metadata["latent_representation_id"] = actual
        return context
