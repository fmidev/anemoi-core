# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Annotated
from typing import Literal

from pydantic import Discriminator
from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import model_validator
from typing_extensions import Self

from anemoi.utils.schemas import BaseModel


class RolloutSchema(BaseModel):
    """Rollout configuration for task."""

    start: NonNegativeInt = Field(example=1)
    "Number of rollouts to start with."
    epoch_increment: NonNegativeInt = Field(example=0)
    "Number of epochs to increment the rollout."
    maximum: NonNegativeInt = Field(example=1)
    "Maximum number of rollouts."


class ForecasterSchema(BaseModel):
    """Configuration for forecasting tasks."""

    target_: Literal["anemoi.training.tasks.Forecaster"] = Field(..., alias="_target_")
    "Task class path for the forecasting task."
    multistep_input: PositiveInt = Field(example=2)
    "Number of input timesteps provided to the model."
    multistep_output: PositiveInt = Field(example=1)
    "Number of output timesteps the model should predict."
    timestep: str = Field(example="6H")
    "Timestep string (e.g. '6H') defining the frequency of the input and output steps."
    rollout: RolloutSchema = Field(...)
    "Rollout configuration for autoregressive training."
    validation_rollout: NonNegativeInt | None = Field(default=None, example=[None, 6, 12])
    "Number of rollouts to use for validation. If unset, validation uses the training rollout."


class AutoencoderTaskSchema(BaseModel):
    """Configuration for autoencoding tasks."""

    target_: Literal["anemoi.training.tasks.Autoencoder"] = Field(..., alias="_target_")
    "Task class path for the autoencoding task."


class TemporalDownscalerSchema(BaseModel):
    """Configuration for temporal downscaling task."""

    target_: Literal["anemoi.training.tasks.TemporalDownscaler"] = Field(..., alias="_target_")
    "Task class path for the temporal downscaling task."
    input_timestep: str = Field(example="6H")
    "Input data timestep as a duration string (e.g. '6H')."
    output_timestep: str = Field(example="1H")
    "Desired output timestep as a duration string (e.g. '1H')."
    output_left_boundary: bool = Field(example=False)
    "Whether to include the left boundary in the output."
    output_right_boundary: bool = Field(example=False)
    "Whether to include the right boundary in the output."


class TaskTargetForcingSchema(BaseModel):
    """Decoder conditioning at target times for a fraction-conditioned dataset."""

    data: list[str] = Field(default=[])
    "Forcing variables sampled at the target time and appended to the decoder input."
    time_fraction: bool = Field(default=True)
    "Append the target fraction of the input window as a scalar feature."


class TemporalInterpolatorSchema(BaseModel):
    """Configuration for the temporal interpolation task."""

    target_: Literal["anemoi.training.tasks.TemporalInterpolator"] = Field(..., alias="_target_")
    "Task class path for the temporal interpolation task."
    input_timestep: str = Field(example="6H")
    "Width of the input window as a duration string; inputs are (t+0, t+input_timestep)."
    target_fractions: list[float] = Field(example=[0.16667, 0.33333, 0.5, 0.66667, 0.83333])
    "Target times as fractions of the input window in [0, 1]: 0 = t+0, 1 = t+input_timestep."
    target_datasets: list[str] = Field(example=["interp"])
    "Datasets that provide targets at the intermediate times; all others are input-only."
    target_forcing: dict[str, TaskTargetForcingSchema] = Field(default={})
    "Per-target-dataset decoder conditioning at the target times."
    target_fractions_per_step: PositiveInt | None = Field(default=None)
    "Stochastic fraction sampling: decode only this many randomly drawn fractions per training step. None = all."
    fraction_sampling: Literal["uniform", "block"] = Field(default="uniform")
    "Subset draw mode: 'uniform' (scattered, without replacement) or 'block' (contiguous fractions)."

    @model_validator(mode="after")
    def check_target_fractions(self) -> Self:
        if not self.target_fractions:
            msg = "target_fractions must not be empty"
            raise ValueError(msg)
        if any(not 0.0 <= fraction <= 1.0 for fraction in self.target_fractions):
            msg = f"target_fractions must be in [0, 1], got {self.target_fractions}"
            raise ValueError(msg)
        if any(b <= a for a, b in zip(self.target_fractions, self.target_fractions[1:])):
            msg = f"target_fractions must be strictly increasing, got {self.target_fractions}"
            raise ValueError(msg)
        unknown = set(self.target_forcing) - set(self.target_datasets)
        if unknown:
            msg = f"target_forcing configured for non-target datasets: {sorted(unknown)}"
            raise ValueError(msg)
        if self.target_fractions_per_step is not None and self.target_fractions_per_step > len(self.target_fractions):
            msg = (
                f"target_fractions_per_step ({self.target_fractions_per_step}) cannot exceed "
                f"the number of target_fractions ({len(self.target_fractions)})"
            )
            raise ValueError(msg)
        return self


TaskSchema = Annotated[
    ForecasterSchema | AutoencoderTaskSchema | TemporalDownscalerSchema | TemporalInterpolatorSchema,
    Discriminator("target_"),
]
