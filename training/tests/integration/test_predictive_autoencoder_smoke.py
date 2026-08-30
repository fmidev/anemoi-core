# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

import pytest
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf

from anemoi.graphs.create import GraphCreator
from anemoi.training.train.train import AnemoiTrainer

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
VARIABLES = [*FORCINGS, *PROGNOSTICS]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _synthetic_dataset() -> dict:
    return {
        "layout": "gridded",
        "geography": {"bbox": [60, 0, -60, 315], "resolution": [30, 45]},
        "dates": {
            "start": "2000-01-01T00:00:00",
            "end": "2000-01-03T18:00:00",
            "frequency": "6h",
        },
        "variables": VARIABLES,
        "values": {"random": {"mean": 1.0, "std": 0.2}},
        "seed": 17,
    }


def _build_graph(tmp_path: Path, synthetic: dict) -> Path:
    recipe = OmegaConf.load(REPOSITORY_ROOT / "global_predictive_autoencoder_graph.yaml")
    recipe.nodes.data.node_builder = {
        "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
        "dataset": {"synthetic": synthetic},
    }
    recipe.nodes.hidden.node_builder.resolution = 2
    recipe.edges[1].edge_builders[0].scale_resolutions = 2
    graph_path = tmp_path / "predictive-autoencoder-smoke-graph.pt"
    GraphCreator(recipe).create(save_path=graph_path)
    return graph_path


def _configure_smoke(config, tmp_path: Path, graph_path: Path, synthetic: dict) -> None:  # noqa: ANN001
    config.system.input.graph = str(graph_path)
    config.system.output.root = str(tmp_path / "output")
    config.graph = {"overwrite": False}
    config.dataloader.dataset = {"synthetic": synthetic, "select": VARIABLES}

    for stage, start, end in [
        ("training", "2000-01-01 00:00:00", "2000-01-01 18:00:00"),
        ("validation", "2000-01-02 00:00:00", "2000-01-02 18:00:00"),
        ("test", "2000-01-03 00:00:00", "2000-01-03 18:00:00"),
    ]:
        config.dataloader[stage].datasets.data.start = start
        config.dataloader[stage].datasets.data.end = end
        config.dataloader.num_workers[stage] = 1
        config.dataloader.batch_size[stage] = 1
    OmegaConf.update(config, "dataloader.multiprocessing_context", "fork", force_add=True)
    config.dataloader.limit_batches.training = 1
    config.dataloader.limit_batches.validation = 1
    config.dataloader.limit_batches.test = 1
    config.dataloader.prefetch_factor = 1
    config.dataloader.pin_memory = False

    config.system.hardware.accelerator = "cpu"
    config.model.num_channels = 8
    config.model.processor.num_layers = 1
    config.model.processor.num_chunks = 1
    config.model.processor.num_heads = 4
    config.model.processor.gradient_checkpointing = False
    config.model.processor.graph_attention_backend = "pyg"
    for name in ["encoder", "decoder", "forcing_encoder"]:
        component = config.model[name]
        component.num_chunks = 1
        component.num_heads = 4
        component.gradient_checkpointing = False
        component.graph_attention_backend = "pyg"
    config.model.compile = []

    config.training.precision = "32-true"
    config.training.max_epochs = 1
    config.training.max_steps = 1
    config.training.num_sanity_val_steps = 1
    config.training.optimization.lr_scheduler.warmup_t = 0

    config.diagnostics.enable_progress_bar = False
    config.diagnostics.log.interval = 1
    config.diagnostics.checkpoint.every_n_minutes.save_frequency = None
    config.diagnostics.checkpoint.every_n_epochs.save_frequency = None
    config.diagnostics.checkpoint.every_n_train_steps.save_frequency = 1
    config.diagnostics.checkpoint.every_n_train_steps.num_models_saved = 1
    for profiler in config.diagnostics.benchmark_profiler.values():
        if hasattr(profiler, "enabled"):
            profiler.enabled = False


@pytest.mark.slow
def test_predictive_autoencoder_train_validation_smoke_writes_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic = _synthetic_dataset()
    graph_path = _build_graph(tmp_path, synthetic)
    monkeypatch.setenv("ANEMOI_ERA5_DATASET", "synthetic-placeholder")
    monkeypatch.setenv("ANEMOI_AUTOENCODER_GRAPH", str(graph_path))
    monkeypatch.setenv("ANEMOI_AUTOENCODER_OUTPUT", str(tmp_path / "output"))
    monkeypatch.setenv("ANEMOI_BASE_SEED", "42")

    with initialize_config_module(version_base=None, config_module="anemoi.training.config"):
        config = compose(config_name="global_predictive_autoencoder")
    _configure_smoke(config, tmp_path, graph_path, synthetic)
    OmegaConf.resolve(config)

    trainer = AnemoiTrainer(config)
    trainer.train()

    checkpoint_root = Path(config.system.output.root) / "checkpoint"
    training_checkpoints = [
        path for path in checkpoint_root.rglob("*.ckpt") if not path.name.startswith("inference-")
    ]
    inference_checkpoints = list(checkpoint_root.rglob("inference-*.ckpt"))
    statistics = trainer.model.model.model.latent_scalar_statistics["data"]

    assert trainer.model.trainer.global_step == 1
    assert training_checkpoints
    assert inference_checkpoints
    assert statistics["ratio"] < 1
