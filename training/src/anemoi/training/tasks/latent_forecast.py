# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Temporal layout for joint reconstruction and latent rollout training."""

from datetime import timedelta

from torch import Tensor

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta


class LatentForecast(BaseSingleStepTask):
    """Temporal sampling and input masking for latent-state prediction.

    Parameters
    ----------
    timestep : str or timedelta, default "6h"
        Interval between successive latent states.
    forecast_steps : int, default 1
        Number of processor transitions.
    loss_steps : list[int], optional
        Supervised step indices. Zero denotes initial-state reconstruction.
        Defaults to all indices from zero through ``forecast_steps``.
    """

    name = "latent-forecast"

    def __init__(
        self,
        timestep: str | timedelta = "6h",
        forecast_steps: int = 1,
        loss_steps: list[int] | None = None,
    ) -> None:
        self.timestep = frequency_to_timedelta(timestep)
        if (
            self.timestep.total_seconds() <= 0
            or isinstance(forecast_steps, bool)
            or not isinstance(forecast_steps, int)
            or forecast_steps < 0
        ):
            message = "A positive timestep and nonnegative integer forecast_steps are required."
            raise ValueError(message)
        self.forecast_steps = forecast_steps
        self.loss_steps = list(range(forecast_steps + 1)) if loss_steps is None else list(loss_steps)
        if not self.loss_steps or any(isinstance(k, bool) or not isinstance(k, int) for k in self.loss_steps):
            message = "loss_steps must be a nonempty list of integer indices."
            raise ValueError(message)
        if (
            self.loss_steps != sorted(set(self.loss_steps))
            or min(self.loss_steps) < 0
            or max(self.loss_steps) > forecast_steps
        ):
            message = "loss_steps must be sorted unique indices within the rollout."
            raise ValueError(message)
        super().__init__(
            input_offsets=[k * self.timestep for k in range(forecast_steps + 1)],
            output_offsets=[k * self.timestep for k in self.loss_steps],
        )

    def get_inputs(
        self,
        batch: dict[str, Tensor],
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> dict[str, Tensor]:
        inputs = super().get_inputs(batch, data_indices, **kwargs)
        for name, value in inputs.items():
            inputs[name] = value.clone()
            inputs[name][:, 1:, ..., data_indices[name].model.input.prognostic.long()] = 0
        return inputs

    def get_encoder_targets(
        self,
        batch: dict[str, Tensor],
        data_indices: dict[str, IndexCollection],
    ) -> dict[str, Tensor]:
        """Normalised unmasked snapshots, for the no-gradient target encoder only."""
        return super().get_inputs(batch, data_indices)

    def _get_timestep_for_metadata(self) -> str:
        return frequency_to_string(self.timestep)
