# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Single-call reconstruction and free latent-rollout task."""

import logging

import torch

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.diagnostics.callbacks.plot_adapter import PredictiveAutoencoderPlotAdapter
from anemoi.training.tasks.base import BaseSingleStepTask
from anemoi.utils.dates import frequency_to_string
from anemoi.utils.dates import frequency_to_timedelta

LOGGER = logging.getLogger(__name__)


class PredictiveAutoencoder(BaseSingleStepTask):
    """Reconstruct the current state and forecast future states in one model call.

    The model receives two complete history snapshots. For each requested
    forecast valid time, only forcing fields remain visible; prognostic fields
    are zeroed after the training module has applied preprocessing.
    """

    name: str = "predictive-autoencoder"

    def __init__(self, timestep: str = "6H", forecast_steps: int = 1, **kwargs) -> None:
        if not isinstance(forecast_steps, int) or isinstance(forecast_steps, bool) or forecast_steps < 1:
            message = f"forecast_steps must be a positive integer, got {forecast_steps!r}."
            raise ValueError(message)

        self.timestep = frequency_to_timedelta(timestep)
        self.forecast_steps = forecast_steps

        if kwargs:
            LOGGER.warning(
                "The following extra parameters were provided to %s but will be ignored: %s",
                self.__class__.__name__,
                kwargs,
            )

        future_offsets = [(step + 1) * self.timestep for step in range(forecast_steps)]
        super().__init__(
            input_offsets=[-self.timestep, self.timestep * 0, *future_offsets],
            output_offsets=[self.timestep * 0, *future_offsets],
        )
        self._plot_adapter = PredictiveAutoencoderPlotAdapter(self)

    def get_inputs(
        self,
        batch: dict[str, torch.Tensor],
        data_indices: dict[str, IndexCollection],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Extract inputs and mask future prognostics while retaining forcings."""
        inputs = super().get_inputs(batch, data_indices, **kwargs)
        for dataset_name, tensor in inputs.items():
            tensor = tensor.clone()
            prognostic_indices = data_indices[dataset_name].model.input.prognostic.long()
            tensor[:, 2:, ..., prognostic_indices] = 0
            inputs[dataset_name] = tensor
        return inputs

    def _get_timestep_for_metadata(self) -> str:
        """Get the timestep string for metadata."""
        return frequency_to_string(self.timestep)
