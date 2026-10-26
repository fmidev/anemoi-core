# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from pathlib import Path

from hydra import compose
from hydra import initialize_config_module
from omegaconf import DictConfig
from omegaconf import OmegaConf

from anemoi.graphs.create import GraphCreator
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.train.train import AnemoiTrainer


def setup(root: Path) -> DictConfig:
    with initialize_config_module(config_module="anemoi.training.config", version_base=None):
        c = compose(config_name="latent_forecaster")
    OmegaConf.set_struct(c, False)
    synthetic = {
        "layout": "gridded",
        "geography": {"bbox": [60, 0, -60, 315], "resolution": [30, 45]},
        "dates": {"start": "2000-01-01T00:00:00", "end": "2000-01-05T21:00:00", "frequency": "3h"},
        "variables": list(c.dataloader.dataset.select),
        "values": {"random": {"mean": 1.0, "std": 0.2}},
        "seed": 17,
    }
    recipe = OmegaConf.load("training/src/anemoi/training/config/graph/latent_n320_r6.yaml")
    recipe.nodes.data.node_builder = {
        "_target_": "anemoi.graphs.nodes.AnemoiDatasetNodes",
        "dataset": {"synthetic": synthetic},
    }
    recipe.nodes.hidden.node_builder.resolution = 1
    recipe.edges[1].edge_builders[0].scale_resolutions = 1
    recipe.edges[0].edge_builders = [{"_target_": "anemoi.graphs.edges.KNNEdges", "num_nearest_neighbours": 4}]
    path = root / "graph.pt"
    GraphCreator(recipe).create(save_path=path, overwrite=True)
    c.system.input.graph = str(path)
    c.system.output.root = str(root / "output")
    c.dataloader.dataset = {"synthetic": synthetic}
    for stage, start, end in [
        ("training", "2000-01-01", "2000-01-02T21:00:00"),
        ("validation", "2000-01-03", "2000-01-03T21:00:00"),
        ("test", "2000-01-04", "2000-01-05T21:00:00"),
    ]:
        c.dataloader[stage].datasets.data.start = start
        c.dataloader[stage].datasets.data.end = end
        c.dataloader.num_workers[stage] = 1
        c.dataloader.limit_batches[stage] = 1
    c.dataloader.pin_memory = False
    c.dataloader.read_group_size = 1
    c.dataloader.prefetch_factor = 1
    c.dataloader.multiprocessing_context = "fork"
    c.system.hardware.accelerator = "cpu"
    c.system.hardware.num_nodes = 1
    c.system.hardware.num_gpus_per_node = 1
    c.system.hardware.num_gpus_per_model = 1
    c.model.num_channels = 8
    c.model.processor.num_layers = 2
    c.model.processor.num_heads = 2
    c.model.processor.num_chunks = 1
    c.model.processor.graph_attention_backend = "pyg"
    for routes in [c.model.encoders, c.model.decoders]:
        for route in routes.values():
            route.mapper.num_heads = 2
            route.mapper.num_layers = 2
            route.mapper.num_chunks = 1
            route.mapper.graph_attention_backend = "pyg"
    c.training.precision = "32-true"
    c.training.max_steps = 1
    c.training.num_sanity_val_steps = 1
    c.training.optimization.lr = 1e-3
    c.diagnostics.enable_progress_bar = False
    c.diagnostics.log.mlflow.enabled = False
    c.diagnostics.log.mlflow.experiment_name = "latent-smoke"
    c.diagnostics.checkpoint.every_n_minutes.save_frequency = None
    c.diagnostics.checkpoint.every_n_epochs.save_frequency = None
    c.diagnostics.checkpoint.every_n_train_steps.save_frequency = 1
    c.diagnostics.checkpoint.every_n_train_steps.num_models_saved = 1
    OmegaConf.resolve(c)
    BaseSchema(**c)
    return c


def test_joint_and_parallel_refinement_stages(tmp_path: Path) -> None:
    from copy import deepcopy

    import torch

    torch.set_num_threads(2)
    config = setup(tmp_path)
    first = AnemoiTrainer(config)
    first.train()
    source = next((tmp_path / "output/checkpoint").rglob("last.ckpt"))
    source_state = torch.load(source, weights_only=False, map_location="cpu")["state_dict"]
    assert first.model.trainer.global_step == 1
    for mode in ["decoder", "rollout"]:
        c = deepcopy(config)
        c.system.output.root = str(tmp_path / mode)
        c.training.checkpoint = {
            "source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": str(source)},
            "loading": {
                "_target_": "anemoi.training.checkpoint.loading.strategies.ColdStartLoader",
                "strict": mode == "decoder",
            },
            "modifiers": [],
        }
        c.training.latent.mode = mode
        c.training.latent.reconstruction_weight = 1.0 if mode == "decoder" else 0.0
        c.training.latent.latent_weight = 0.0 if mode == "decoder" else 1.0
        if mode == "decoder":
            c.task.forecast_steps = 0
            c.task.loss_steps = [0]
            c.training.latent.lead_weights = []
            c.training.latent.validation_decode_steps = []
        else:
            c.task.timestep = "3h"
            c.model.latent.timestep = "3h"
            c.task.forecast_steps = 2
            c.task.loss_steps = [1, 2]
            c.training.latent.lead_weights = [1.0, 1.0]
            c.model.processor.num_layers = 4
            c.model.processor.mlp_hidden_ratio = 6
            c.training.checkpoint.loading.skip_mismatched = True
        frozen = [
            "model.model.encoder",
            "model.model.encoder_graph_provider",
            "model.model.node_attributes",
            "model.model.latent_aggregator",
            "model.pre_processors",
            "model.post_processors",
        ]
        frozen += (
            [
                "model.model.processor",
                "model.model.processor_graph_provider",
                "model.model.static_embeddings",
                "model.model.temporal_embeddings",
            ]
            if mode == "decoder"
            else ["model.model.decoder", "model.model.decoder_graph_provider"]
        )
        c.training.checkpoint.modifiers = [
            {"_target_": "anemoi.training.checkpoint.modifiers.latent.VerifyLatentRepresentation"},
            {
                "_target_": "anemoi.training.checkpoint.modifiers.freezing.FreezingModifierStage",
                "strict": True,
                "submodules_to_freeze": frozen,
            },
        ]
        OmegaConf.resolve(c)
        trainer = AnemoiTrainer(c)
        trainer.train()
        end = trainer.model.state_dict()
        for key, value in source_state.items():
            if key.startswith(
                (
                    "model.model.encoder.",
                    "model.model.encoder_graph_provider.",
                    "model.model.node_attributes.",
                    "model.pre_processors.",
                ),
            ):
                torch.testing.assert_close(end[key], value, rtol=0, atol=0)
            if mode == "rollout" and key.startswith("model.model.decoder."):
                torch.testing.assert_close(end[key], value, rtol=0, atol=0)
        assert trainer.model.trainer.global_step == 1
        # A training checkpoint can be restored without the source dataset/job.
        saved = next((tmp_path / mode / "checkpoint").rglob("last.ckpt"))
        restored = type(trainer.model).load_from_checkpoint(saved, map_location="cpu", weights_only=False)
        for key, value in end.items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)


def test_warm_start_matches_continuous_training(tmp_path: Path) -> None:
    from copy import deepcopy

    import torch

    torch.set_num_threads(2)
    split = tmp_path / "split"
    control = tmp_path / "control"
    split.mkdir()
    control.mkdir()
    first_config = setup(split)
    first_config.training.seed = 42
    first_config.dataloader.limit_batches.training = 2
    torch.manual_seed(1234)
    first = AnemoiTrainer(first_config)
    first.train()
    source = next((split / "output/checkpoint").rglob("last.ckpt"))

    resumed_config = deepcopy(first_config)
    resumed_config.system.output.root = str(tmp_path / "resumed")
    resumed_config.training.max_steps = 2
    resumed_config.training.checkpoint = {
        "source": {"_target_": "anemoi.training.checkpoint.sources.local.LocalSource", "path": str(source)},
        "loading": {"_target_": "anemoi.training.checkpoint.loading.strategies.WarmStartLoader"},
        "modifiers": [],
    }
    resumed = AnemoiTrainer(resumed_config)
    resumed.train()
    assert resumed.model.trainer.global_step == 2

    control_config = setup(control)
    control_config.training.seed = 42
    control_config.training.max_steps = 2
    control_config.dataloader.limit_batches.training = 2
    torch.manual_seed(1234)
    continuous = AnemoiTrainer(control_config)
    continuous.train()
    for name, value in continuous.model.state_dict().items():
        torch.testing.assert_close(resumed.model.state_dict()[name], value, rtol=0, atol=0)
    resumed_optimizer = resumed.model.trainer.optimizers[0].state_dict()
    continuous_optimizer = continuous.model.trainer.optimizers[0].state_dict()
    assert resumed_optimizer["param_groups"] == continuous_optimizer["param_groups"]
    for parameter, state in continuous_optimizer["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(resumed_optimizer["state"][parameter][name], value, rtol=0, atol=0)
