# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path
from unittest.mock import patch

import torch
from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.graphs.create import GraphCreator
from anemoi.graphs.describe import GraphDescriptor
from anemoi.graphs.schemas.base_graph import BaseGraphSchema
from anemoi.training.commands.config import ConfigGenerator
from anemoi.training.schemas.base_schema import BaseSchema

FORCINGS = [
    "cos_latitude",
    "cos_longitude",
    "sin_latitude",
    "sin_longitude",
    "cos_julian_day",
    "cos_local_time",
    "sin_julian_day",
    "sin_local_time",
    "insolation",
    "lsm",
    "z",
]
LEVELS = [50, 100, 150, 200, 250, 300, 400, 500, 700, 850, 925, 1000]
PROGNOSTICS = ["sp", *[f"{family}_{level}" for family in ["q", "t", "u", "v", "w", "z"] for level in LEVELS]]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def _compose_reference_config(monkeypatch) -> DictConfig:  # noqa: ANN001
    test_paths = REPOSITORY_ROOT / ".predictive-autoencoder-test-paths"
    monkeypatch.setenv("ANEMOI_ERA5_DATASET", str(test_paths / "era5-n320.zarr"))
    monkeypatch.setenv("ANEMOI_AUTOENCODER_GRAPH", str(test_paths / "graph.pt"))
    monkeypatch.setenv("ANEMOI_AUTOENCODER_OUTPUT", str(test_paths / "output"))
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        config = compose(config_name="global_predictive_autoencoder")
    OmegaConf.resolve(config)
    return config


def test_reference_config_composes_validates_and_selects_exact_fields(monkeypatch) -> None:  # noqa: ANN001
    config = _compose_reference_config(monkeypatch)

    BaseSchema(**config)

    assert config.config_validation is True
    assert list(config.data.datasets.data.forcing) == FORCINGS
    assert list(config.data.datasets.data.diagnostic) == []
    assert list(config.dataloader.dataset.select) == [*FORCINGS, *PROGNOSTICS]
    assert len(config.dataloader.dataset.select) == 84
    assert config.task.forecast_steps == 1
    assert config.task.use_previous_state is True
    assert config.graph.nodes.data.node_builder._target_.endswith("ReducedGaussianGridNodes")
    assert config.graph.nodes.data.node_builder.grid == "n320"
    assert config.graph.nodes.hidden.node_builder.resolution == 5
    assert config.graph.edges[0].edge_builders[0].max_num_neighbours == 128
    assert config.model.num_channels == 128
    assert config.model.model.expected_num_forcing_fields == 11
    assert config.model.model.expected_num_prognostic_fields == 73
    assert not config.model.processor._target_.endswith("NoOpProcessor")
    assert config.training.scalers.datasets.data.stdev_tendency.timestep == "6h"
    assert config.training.scalers.datasets.data.var_tendency.timestep == "6h"
    assert list(config.training.scalers.datasets.data.time_steps.weights) == [1.0, 1.0]
    assert list(config.training.scalers.datasets.data.reconstruction_time.weights) == [1.0, 0.0]
    assert list(config.training.scalers.datasets.data.forecast_time.weights) == [0.0, 1.0]
    assert list(config.training.validation_metrics.datasets.data.reconstruction_mse.scalers) == [
        "node_weights",
        "reconstruction_time",
    ]
    assert list(config.training.validation_metrics.datasets.data.forecast_mse.scalers) == [
        "node_weights",
        "forecast_time",
    ]


def test_existing_graph_override_is_load_only(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("ANEMOI_ERA5_DATASET", str(REPOSITORY_ROOT / "dataset.zarr"))
    monkeypatch.setenv("ANEMOI_AUTOENCODER_GRAPH", str(REPOSITORY_ROOT / "graph.pt"))
    monkeypatch.setenv("ANEMOI_AUTOENCODER_OUTPUT", str(REPOSITORY_ROOT / "output"))
    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        config = compose(config_name="global_predictive_autoencoder", overrides=["graph=existing"])

    BaseSchema(**config)
    assert config.graph.overwrite is False
    assert "nodes" not in config.graph
    assert "edges" not in config.graph


def test_config_generator_emits_reference_and_task_presets(tmp_path: Path) -> None:
    ConfigGenerator().traverse_config(tmp_path)

    assert (tmp_path / "global_predictive_autoencoder.yaml").is_file()
    assert (tmp_path / "graph" / "predictive_n320.yaml").is_file()
    assert (tmp_path / "task" / "predictive_autoencoder.yaml").is_file()


def test_standalone_graph_recipe_matches_training_topology(monkeypatch) -> None:  # noqa: ANN001
    config = _compose_reference_config(monkeypatch)
    standalone = OmegaConf.load(REPOSITORY_ROOT / "global_predictive_autoencoder_graph.yaml")
    BaseGraphSchema(**standalone)

    training_graph = OmegaConf.to_container(config.graph, resolve=True)
    standalone_graph = OmegaConf.to_container(standalone, resolve=True)
    assert isinstance(training_graph, dict)
    assert isinstance(standalone_graph, dict)

    assert standalone_graph == training_graph


def test_standalone_recipe_builds_and_describes_required_graph(tmp_path: Path) -> None:
    recipe = OmegaConf.load(REPOSITORY_ROOT / "global_predictive_autoencoder_graph.yaml")
    recipe.nodes.data.node_builder.grid = "o16"
    recipe.nodes.hidden.node_builder.resolution = 2
    recipe.edges[1].edge_builders[0].scale_resolutions = 2
    latitudes = torch.linspace(-1.2, 1.2, 8)
    longitudes = torch.arange(16) * 2 * torch.pi / 16
    coordinates = torch.cartesian_prod(latitudes, longitudes)
    graph_path = tmp_path / "predictive-autoencoder-graph.pt"

    with patch("anemoi.graphs.nodes.ReducedGaussianGridNodes.get_coordinates", return_value=coordinates):
        graph = GraphCreator(recipe).create(save_path=graph_path)

    expected_edges = {
        ("data", "to", "hidden"),
        ("hidden", "to", "hidden"),
        ("hidden", "to", "data"),
    }
    assert set(graph.node_types) == {"data", "hidden"}
    assert set(graph.edge_types) == expected_edges
    for edge_type in expected_edges:
        assert {"edge_length", "edge_dirs"} <= set(graph[edge_type].edge_attrs())

    descriptor = GraphDescriptor(graph_path)
    assert {row[0] for row in descriptor.get_node_summary()} == {"data", "hidden"}
    assert {(row[0], row[1]) for row in descriptor.get_edge_summary()} == {
        ("data", "hidden"),
        ("hidden", "hidden"),
        ("hidden", "data"),
    }
