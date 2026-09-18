# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime
import logging

import numpy as np
import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import TemporalDownscalerPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.time_indices import normalize_time_indices
from anemoi.utils.dates import as_timedelta

LOGGER = logging.getLogger(__name__)


class TemporalInterpolator(BaseSingleStepTask):
    """Temporal interpolation task for fraction-conditioned decoder heads.

    Every dataset receives the two bracketing states ``t+0`` and
    ``t+input_timestep`` as input. Datasets listed in ``target_datasets``
    additionally provide targets at the intermediate times given by
    ``target_fractions`` (each a fraction of the input window in [0, 1]:
    0 = t+0, 1 = t+input_timestep, 0.5 = the midpoint).

    The model decodes one output time per decoder invocation, conditioned on
    the target fraction (and optionally target-time forcings), so
    ``num_output_timesteps`` is 1 while ``num_loss_timesteps`` is the number
    of target fractions.
    """

    name: str = "temporal-interpolator"

    def __init__(
        self,
        input_timestep: str,
        target_fractions: list[float],
        target_datasets: list[str],
        target_forcing: dict[str, dict] | None = None,
        target_fractions_per_step: int | None = None,
        fraction_sampling: str = "uniform",
        **_kwargs,
    ) -> None:
        """Initialize the temporal interpolation task.

        Parameters
        ----------
        input_timestep : str
            Width of the input window as a duration string (e.g. '6H'). The
            input offsets are ``[0, input_timestep]`` for every dataset.
        target_fractions : list[float]
            Target times as fractions of the input window in [0, 1], strictly
            increasing. Each fraction is snapped to the nearest whole second;
            the resulting offset must lie on the target dataset's time axis
            (validated when relative date indices are computed).
        target_datasets : list[str]
            Datasets that provide targets at the intermediate times. All other
            datasets are input-only.
        target_forcing : dict[str, dict] | None
            Per-target-dataset conditioning config with keys ``data`` (list of
            forcing variable names sampled at the target time) and
            ``time_fraction`` (append the fraction as a scalar feature).
        target_fractions_per_step : int | None
            Stochastic fraction sampling: decode only this many randomly chosen
            target fractions per TRAINING step (an unbiased estimator of the
            full-fraction loss at proportionally lower cost). Validation always
            decodes all fractions. None (default) decodes all fractions every step.
        fraction_sampling : str
            How the per-step subset is drawn: "uniform" (without replacement,
            scattered) or "block" (contiguous run of fractions, which keeps
            adjacent hours together, e.g. for temporal-difference loss terms).
        """
        self.input_timestep = input_timestep
        self.input_timedelta = as_timedelta(input_timestep)
        self.target_datasets = list(target_datasets)
        self.target_forcing = {name: dict(config) for name, config in (target_forcing or {}).items()}

        self._validate_target_fractions(target_fractions)

        if target_fractions_per_step is not None:
            if not 1 <= target_fractions_per_step <= len(target_fractions):
                msg = (
                    f"target_fractions_per_step must be in [1, {len(target_fractions)}], "
                    f"got {target_fractions_per_step}"
                )
                raise ValueError(msg)
            if fraction_sampling not in ("uniform", "block"):
                msg = f"fraction_sampling must be 'uniform' or 'block', got {fraction_sampling!r}"
                raise ValueError(msg)
        self.target_fractions_per_step = target_fractions_per_step
        self.fraction_sampling = fraction_sampling
        # Deterministic sampling state: every rank draws the same subset per step
        # (ensemble members gather predictions across ranks, so the decoded hours
        # must agree within a group). The counter advances once per training step
        # in get_forward_kwargs, which all ranks call in lockstep.
        self._sampling_seed = get_base_seed()
        self._sampling_step = 0
        self._active_fraction_indices: list[int] | None = None

        window_seconds = self.input_timedelta.total_seconds()
        output_offsets = [
            datetime.timedelta(seconds=round(float(fraction) * window_seconds)) for fraction in target_fractions
        ]
        # Exact fractions recomputed from the snapped offsets; these are the
        # values fed to the model as the conditioning feature.
        self.fractions = [offset.total_seconds() / window_seconds for offset in output_offsets]

        unknown_forcing_datasets = set(self.target_forcing) - set(self.target_datasets)
        if unknown_forcing_datasets:
            msg = f"target_forcing configured for non-target datasets: {sorted(unknown_forcing_datasets)}"
            raise ValueError(msg)

        super().__init__(
            input_offsets=[datetime.timedelta(0), self.input_timedelta],
            output_offsets=output_offsets,
        )
        self._plot_adapter = TemporalDownscalerPlotAdapter(self)

    @staticmethod
    def _validate_target_fractions(target_fractions: list[float]) -> None:
        if not target_fractions:
            msg = "target_fractions must not be empty"
            raise ValueError(msg)
        if any(not 0.0 <= float(fraction) <= 1.0 for fraction in target_fractions):
            msg = f"target_fractions must be in [0, 1], got {list(target_fractions)}"
            raise ValueError(msg)
        if any(b <= a for a, b in zip(target_fractions, target_fractions[1:])):
            msg = f"target_fractions must be strictly increasing, got {list(target_fractions)}"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Per-dataset offsets: non-target datasets are input-only
    # ------------------------------------------------------------------

    def get_output_offsets(self, dataset_name: str | None = None, **_kwargs) -> list[datetime.timedelta]:
        if dataset_name is None or dataset_name in self.target_datasets:
            return self._output_offsets
        return []

    def get_offsets(self, dataset_name: str | None = None, **_kwargs) -> list[datetime.timedelta]:
        if dataset_name is None or dataset_name in self.target_datasets:
            return self._offsets
        return self._input_offsets

    # ------------------------------------------------------------------
    # Model contract
    # ------------------------------------------------------------------

    @property
    def num_output_timesteps(self) -> int:
        """The model decodes one output time per (fraction-conditioned) invocation."""
        return 1

    @property
    def num_loss_timesteps(self) -> int:
        """All target fractions enter the loss each task step."""
        return len(self._output_offsets)

    @property
    def tendency_delta(self) -> datetime.timedelta:
        """Delta used for tendency statistics: the smallest positive target offset."""
        positive = [offset for offset in self._output_offsets if offset > datetime.timedelta(0)]
        return min(positive) if positive else self.input_timedelta

    def _sample_fraction_indices(self) -> list[int]:
        """Draw the per-step fraction subset, deterministically in the step counter."""
        rng = np.random.default_rng((self._sampling_seed, self._sampling_step))
        self._sampling_step += 1
        n = len(self.fractions)
        k = self.target_fractions_per_step
        if self.fraction_sampling == "block":
            start = int(rng.integers(0, n - k + 1))
            return list(range(start, start + k))
        return sorted(int(i) for i in rng.choice(n, size=k, replace=False))

    def get_forward_kwargs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        validation_mode: bool = False,
        **_step_kwargs,
    ) -> dict:
        """Build the per-target-dataset conditioning passed to ``model.forward``.

        Returns ``{"target_times": {name: {"fractions": (T,), "forcings": (bs, T, ens, grid, F) | None}}}``
        where the forcings are sliced from the (normalized, possibly grid-sharded)
        batch at the target times, in DATA_FULL variable space.

        With ``target_fractions_per_step`` set, training steps decode only a
        randomly drawn subset of the fractions; ``get_targets`` (called after
        this within the same step) returns the matching target hours. Validation
        always decodes all fractions.
        """
        subsample = (
            not validation_mode
            and self.target_fractions_per_step is not None
            and self.target_fractions_per_step < len(self.fractions)
        )
        self._active_fraction_indices = self._sample_fraction_indices() if subsample else None

        target_times: dict[str, dict[str, torch.Tensor | None]] = {}
        for dataset_name in self.target_datasets:
            dataset_batch = batch[dataset_name]
            fractions = self.fractions
            if self._active_fraction_indices is not None:
                fractions = [fractions[i] for i in self._active_fraction_indices]
            fractions = torch.tensor(fractions, dtype=dataset_batch.dtype, device=dataset_batch.device)

            forcings = None
            forcing_names = self.target_forcing.get(dataset_name, {}).get("data", [])
            if forcing_names:
                name_to_index = data_indices[dataset_name].name_to_index
                forcing_idx = [name_to_index[name] for name in forcing_names]
                out_idx = self.get_batch_output_indices(dataset_name=dataset_name)
                if self._active_fraction_indices is not None:
                    out_idx = [out_idx[i] for i in self._active_fraction_indices]
                forcings = dataset_batch[:, normalize_time_indices(out_idx)][..., forcing_idx]

            target_times[dataset_name] = {"fractions": fractions, "forcings": forcings}

        return {"target_times": target_times}

    def get_targets(self, batch: dict[str, torch.Tensor], **kwargs) -> dict[str, torch.Tensor]:
        """Extract targets, restricted to the fraction subset drawn for this step (if any)."""
        y = super().get_targets(batch, **kwargs)
        if self._active_fraction_indices is not None:
            for dataset_name in self.target_datasets:
                if dataset_name in y:
                    y[dataset_name] = y[dataset_name][:, self._active_fraction_indices]
        return y

    def training_runtime_state_dict(self) -> dict:
        """Persist the sampling counter so a resumed run continues the sequence."""
        return {"fraction_sampling_step": self._sampling_step}

    def load_training_runtime_state_dict(self, state: dict) -> None:
        """Restore the sampling counter from a training checkpoint."""
        self._sampling_step = state.get("fraction_sampling_step", 0)

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return self.input_timestep
