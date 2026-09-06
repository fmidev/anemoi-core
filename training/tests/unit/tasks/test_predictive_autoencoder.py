# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime

import pytest
import torch
from omegaconf import DictConfig

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.tasks import PredictiveAutoencoder


def _indices() -> dict[str, IndexCollection]:
    config = DictConfig({"forcing": ["forcing"], "diagnostic": [], "target": []})
    return {"data": IndexCollection(config, {"forcing": 0, "state_a": 1, "state_b": 2})}


@pytest.mark.parametrize("forecast_steps", [0, 1, 3])
def test_offsets_and_single_training_step(forecast_steps: int) -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=forecast_steps)

    assert task.get_input_offsets() == [
        datetime.timedelta(hours=-6),
        datetime.timedelta(0),
        *[datetime.timedelta(hours=6 * step) for step in range(1, forecast_steps + 1)],
    ]
    assert task.get_output_offsets() == [
        datetime.timedelta(0),
        *[datetime.timedelta(hours=6 * step) for step in range(1, forecast_steps + 1)],
    ]
    assert list(task.steps("training")) == [{}]
    assert task.num_input_timesteps == forecast_steps + 2
    assert task.num_output_timesteps == forecast_steps + 1


def test_current_analysis_only_offsets() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=2, use_previous_state=False)

    assert task.get_input_offsets() == [
        datetime.timedelta(0),
        datetime.timedelta(hours=6),
        datetime.timedelta(hours=12),
    ]
    assert task.get_output_offsets() == task.get_input_offsets()
    assert task.num_input_timesteps == 3
    assert task.num_output_timesteps == 3


def test_future_prognostics_are_masked_but_forcings_are_retained() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=2)
    indices = _indices()
    batch = {"data": torch.arange(1 * 4 * 1 * 2 * 3, dtype=torch.float32).reshape(1, 4, 1, 2, 3)}
    original = batch["data"].clone()

    inputs = task.get_inputs(batch, indices)["data"]

    torch.testing.assert_close(inputs[:, :2], original[:, :2])
    torch.testing.assert_close(inputs[:, 2:, ..., 0], original[:, 2:, ..., 0])
    assert torch.count_nonzero(inputs[:, 2:, ..., 1:]) == 0
    torch.testing.assert_close(batch["data"], original)


def test_current_analysis_only_masks_all_future_prognostics() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=2, use_previous_state=False)
    indices = _indices()
    batch = {"data": torch.arange(1 * 3 * 1 * 2 * 3, dtype=torch.float32).reshape(1, 3, 1, 2, 3)}
    original = batch["data"].clone()

    inputs = task.get_inputs(batch, indices)["data"]

    torch.testing.assert_close(inputs[:, :1], original[:, :1])
    torch.testing.assert_close(inputs[:, 1:, ..., 0], original[:, 1:, ..., 0])
    assert torch.count_nonzero(inputs[:, 1:, ..., 1:]) == 0


def test_targets_include_reconstruction_then_forecasts() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=2)
    batch = {"data": torch.arange(4, dtype=torch.float32).reshape(1, 4, 1, 1, 1)}

    targets = task.get_targets(batch)["data"]

    assert targets[:, :, 0, 0, 0].tolist() == [[1.0, 2.0, 3.0]]


def test_metadata_records_six_hour_timestep() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=1)
    metadata = {"metadata_inference": {"dataset_names": ["data"], "data": {}}}

    task.fill_metadata(metadata)

    timesteps = metadata["metadata_inference"]["data"]["timesteps"]
    assert timesteps["timestep"] == "6h"
    assert timesteps["input_relative_date_indices"] == [0, 1, 2]
    assert timesteps["output_relative_date_indices"] == [1, 2]


def test_metadata_records_current_analysis_only_layout() -> None:
    task = PredictiveAutoencoder(timestep="6H", forecast_steps=1, use_previous_state=False)
    metadata = {"metadata_inference": {"dataset_names": ["data"], "data": {}}}

    task.fill_metadata(metadata)

    timesteps = metadata["metadata_inference"]["data"]["timesteps"]
    assert timesteps["input_relative_date_indices"] == [0, 1]
    assert timesteps["output_relative_date_indices"] == [0, 1]


@pytest.mark.parametrize("forecast_steps", [-1, True, 1.5])
def test_forecast_steps_must_be_a_non_negative_integer(forecast_steps: object) -> None:
    with pytest.raises(ValueError, match="non-negative integer"):
        PredictiveAutoencoder(forecast_steps=forecast_steps)  # type: ignore[arg-type]


def test_reconstruction_only_current_state() -> None:
    task = PredictiveAutoencoder(forecast_steps=0, use_previous_state=False)
    batch = {"data": torch.randn(2, 1, 1, 2, 3)}
    assert task.get_input_offsets() == task.get_output_offsets() == [datetime.timedelta(0)]
    torch.testing.assert_close(task.get_inputs(batch, _indices())["data"], batch["data"])
    assert task.get_targets(batch)["data"].shape[1] == 1
    sample = batch["data"][0]
    plots = list(task._plot_adapter.iter_plot_samples(sample, sample))
    assert len(plots) == 1
    assert plots[0][3] == "recon"
    torch.testing.assert_close(plots[0][0], sample[0].squeeze())


def test_use_previous_state_must_be_boolean() -> None:
    with pytest.raises(ValueError, match="must be a boolean"):
        PredictiveAutoencoder(use_previous_state=1)  # type: ignore[arg-type]
