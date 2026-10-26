# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Representation identity and checkpoint lifecycle regression tests."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

from anemoi.training.checkpoint.modifiers.latent import representation_fingerprint
from anemoi.training.train.methods.latent_forecast import LatentForecastTraining


@pytest.fixture
def module() -> SimpleNamespace:
    graph = {
        "data": SimpleNamespace(x=torch.tensor([[0.0, 0.0], [0.1, 0.2]])),
        "hidden": SimpleNamespace(x=torch.tensor([[0.0, 0.1], [0.2, 0.3]])),
        ("data", "to", "hidden"): SimpleNamespace(edge_index=torch.tensor([[0, 1], [1, 0]])),
    }
    core = SimpleNamespace(input_datasets=["data"], _graph_name_hidden="hidden", _graph_data=graph)
    state = {"model.model.encoder.weight": torch.ones(2, 2)}
    return SimpleNamespace(
        config=OmegaConf.create(
            {
                "model": {
                    "encoders": {
                        "0": {
                            "mapper": {
                                "num_layers": 3,
                                "num_chunks": 1,
                                "gradient_checkpointing": False,
                                "cpu_offload": False,
                            },
                        },
                    },
                },
            },
        ),
        data_indices={"data": SimpleNamespace(data_full_ordered_names=["2t", "sp"])},
        model=SimpleNamespace(model=core),
        state_dict=lambda: state,
    )


@pytest.mark.parametrize(
    ("setting", "value"),
    [("num_chunks", 4), ("gradient_checkpointing", True), ("cpu_offload", True)],
)
def test_execution_settings_preserve_representation_identity(
    module: SimpleNamespace,
    setting: str,
    value: int | bool,
) -> None:
    reference = representation_fingerprint(module)
    module.config.model.encoders["0"].mapper[setting] = value
    assert representation_fingerprint(module) == reference


@pytest.mark.parametrize("change", ["architecture", "weights", "coordinates", "edges", "variables"])
def test_representation_changes_are_detected(module: SimpleNamespace, change: str) -> None:
    reference = representation_fingerprint(module)
    if change == "architecture":
        module.config.model.encoders["0"].mapper.num_layers = 4
    elif change == "weights":
        module.state_dict()["model.model.encoder.weight"][0, 0] += 1
    elif change == "coordinates":
        module.model.model._graph_data["hidden"].x[0, 0] += 0.1
    elif change == "edges":
        module.model.model._graph_data["data", "to", "hidden"].edge_index[0, 0] = 1
    else:
        module.data_indices["data"].data_full_ordered_names.reverse()
    assert representation_fingerprint(module) != reference


def test_checkpoint_save_preserves_upstream_task_state() -> None:
    module = object.__new__(LatentForecastTraining)
    torch.nn.Module.__init__(module)
    module.task = SimpleNamespace(training_runtime_state_dict=lambda: {"counter": 7})
    checkpoint = {}
    with patch("anemoi.training.checkpoint.modifiers.latent.representation_fingerprint", return_value="representation"):
        module.on_save_checkpoint(checkpoint)
    assert checkpoint == {"task_state": {"counter": 7}, "latent_representation_id": "representation"}
