# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from collections.abc import Iterator

import pytest
import torch
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.models import AnemoiModelPredictiveAutoEncoder
from anemoi.models.schemas.models import BaseModelSchema


def _component(target: str) -> dict:
    return {
        "_target_": target,
        "cpu_offload": False,
        "gradient_checkpointing": False,
        "layer_kernels": {},
    }


def _model_config(*, latent_skip: bool = True, require_bottleneck: bool = True) -> DictConfig:
    return OmegaConf.create(
        {
            "model": {
                "num_channels": 2,
                "trainable_parameters": {"data": 0, "hidden": 0},
                "model": {
                    "_target_": "anemoi.models.models.AnemoiModelPredictiveAutoEncoder",
                    "hidden_nodes_name": "hidden",
                    "latent_skip": latent_skip,
                    "expected_num_forcing_fields": 1,
                    "expected_num_prognostic_fields": 3,
                    "require_bottleneck": require_bottleneck,
                },
                "encoder": _component("anemoi.models.layers.mapper.PointWiseForwardMapper"),
                "forcing_encoder": _component("anemoi.models.layers.mapper.PointWiseForwardMapper"),
                "decoder": {
                    **_component("anemoi.models.layers.mapper.PointWiseBackwardMapper"),
                    "initialise_data_extractor_zero": False,
                },
                "processor": {
                    **_component("anemoi.models.layers.processor.PointWiseMLPProcessor"),
                    "num_layers": 1,
                    "num_chunks": 1,
                    "mlp_hidden_ratio": 2,
                    "dropout_p": 0.0,
                },
                "state_context_mixer": {
                    "_target_": "anemoi.models.layers.temporal.LatentStateContextMixer",
                    "mlp_hidden_ratio": 2,
                    "n_extra_layers": 0,
                    "final_activation": False,
                    "layer_norm": False,
                    "layer_kernels": {},
                },
                "residual": {
                    "_target_": "anemoi.models.layers.residual.SkipConnection",
                    "step": -1,
                },
                "bounding": [],
                "output_mask": {"_target_": "anemoi.training.utils.masks.NoOutputMask"},
            },
        },
    )


def _data_indices() -> dict[str, IndexCollection]:
    config = DictConfig({"forcing": ["forcing"], "diagnostic": [], "target": []})
    names = {"forcing": 0, "state_a": 1, "state_b": 2, "state_c": 3}
    return {"data": IndexCollection(config, names)}


def _graph() -> HeteroData:
    graph = HeteroData()
    coordinates = torch.tensor(
        [
            [-1.0, 0.0],
            [-0.3, 1.0],
            [0.3, 2.0],
            [1.0, 3.0],
        ],
    )
    graph["data"].x = coordinates
    graph["data"].num_nodes = coordinates.shape[0]
    graph["hidden"].x = coordinates.clone()
    graph["hidden"].num_nodes = coordinates.shape[0]
    return graph


def _make_model(
    forecast_steps: int = 1,
    *,
    latent_skip: bool = True,
    require_bottleneck: bool = True,
    use_previous_state: bool = True,
) -> AnemoiModelPredictiveAutoEncoder:
    torch.manual_seed(7)
    return AnemoiModelPredictiveAutoEncoder(
        model_config=_model_config(latent_skip=latent_skip, require_bottleneck=require_bottleneck),
        data_indices=_data_indices(),
        statistics={"data": {}},
        n_step_input=forecast_steps + 1 + int(use_previous_state),
        n_step_output=forecast_steps + 1,
        graph_data=_graph(),
    )


def _input(
    forecast_steps: int,
    *,
    requires_grad: bool = False,
    use_previous_state: bool = True,
) -> dict[str, torch.Tensor]:
    torch.manual_seed(11)
    num_input_steps = forecast_steps + 1 + int(use_previous_state)
    return {"data": torch.randn(2, num_input_steps, 1, 4, 4, requires_grad=requires_grad)}


def _nonzero_gradient(parameters: Iterator[torch.nn.Parameter]) -> bool:
    return any(parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0 for parameter in parameters)


@pytest.mark.parametrize("forecast_steps", [0, 1, 2])
@pytest.mark.parametrize("use_previous_state", [True, False])
def test_output_shape_and_time_order(forecast_steps: int, use_previous_state: bool) -> None:
    model = _make_model(forecast_steps, use_previous_state=use_previous_state)
    inputs = _input(forecast_steps, use_previous_state=use_previous_state)

    output = model(inputs)["data"]

    assert output.shape == (2, forecast_steps + 1, 1, 4, 3)


@pytest.mark.parametrize("use_previous_state", [True, False])
def test_decoder_finetuning_skips_dynamics_and_preserves_checkpoint_weights(monkeypatch, use_previous_state) -> None:
    pretrained = _make_model(use_previous_state=use_previous_state)
    model = _make_model(forecast_steps=0, use_previous_state=use_previous_state)
    model.load_state_dict(pretrained.state_dict(), strict=True)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith(("decoder.", "decoder_graph_provider.")))

    def unexpected_call(*args, **kwargs):
        pytest.fail("Reconstruction-only mode must not execute forecast networks")

    for method in ("encode_static_forcing_context", "encode_forcing_context", "transition_latent"):
        monkeypatch.setattr(model, method, unexpected_call)

    inputs = _input(1, use_previous_state=use_previous_state)
    with torch.no_grad():
        expected = pretrained(inputs)["data"][:, :1]
    output = model({"data": inputs["data"][:, : 1 + int(use_previous_state)]})["data"]
    torch.testing.assert_close(output, expected)
    output.square().mean().backward()
    assert _nonzero_gradient(model.decoder.parameters())
    assert all(parameter.grad is None for parameter in model.encoder.parameters())
    assert all(parameter.grad is None for parameter in model.processor.parameters())
    # The decoder-only checkpoint can still seed the next forecasting stage.
    pretrained.load_state_dict(model.state_dict(), strict=True)


def test_current_analysis_only_matches_explicit_reconstruction_then_transition_decode() -> None:
    model = _make_model(use_previous_state=False)
    model.eval()
    inputs = _input(1, use_previous_state=False)

    with torch.no_grad():
        current, shard_sizes = model.encode_snapshot(inputs, 0, batch_size=2)
        reconstruction = model.decode_snapshot(
            current,
            inputs,
            0,
            batch_size=2,
            ensemble_size=1,
            shard_sizes_hidden=shard_sizes,
            in_out_sharded={"data": False},
        )["data"]
        context, _ = model.encode_forcing_context(inputs, 1, batch_size=2)
        predicted = model.transition_latent(
            None,
            current,
            context,
            batch_size=2,
            shard_sizes_hidden=shard_sizes,
        )
        forecast = model.decode_snapshot(
            predicted,
            inputs,
            1,
            batch_size=2,
            ensemble_size=1,
            shard_sizes_hidden=shard_sizes,
            in_out_sharded={"data": False},
        )["data"]
        output = model(inputs)["data"]

    torch.testing.assert_close(output[:, :1], reconstruction)
    torch.testing.assert_close(output[:, 1:], forecast)


def test_forward_matches_explicit_reconstruction_then_transition_decode() -> None:
    model = _make_model()
    model.eval()
    inputs = _input(1)

    with torch.no_grad():
        previous, shard_sizes = model.encode_snapshot(inputs, 0, batch_size=2)
        current, _ = model.encode_snapshot(inputs, 1, batch_size=2)
        reconstruction = model.decode_snapshot(
            current,
            inputs,
            1,
            batch_size=2,
            ensemble_size=1,
            shard_sizes_hidden=shard_sizes,
            in_out_sharded={"data": False},
        )["data"]
        context, _ = model.encode_forcing_context(inputs, 2, batch_size=2)
        predicted = model.transition_latent(
            previous,
            current,
            context,
            batch_size=2,
            shard_sizes_hidden=shard_sizes,
        )
        forecast = model.decode_snapshot(
            predicted,
            inputs,
            2,
            batch_size=2,
            ensemble_size=1,
            shard_sizes_hidden=shard_sizes,
            in_out_sharded={"data": False},
        )["data"]
        output = model(inputs)["data"]

    torch.testing.assert_close(output[:, :1], reconstruction)
    torch.testing.assert_close(output[:, 1:], forecast)


def test_future_prognostics_are_never_consumed() -> None:
    model = _make_model(forecast_steps=2)
    model.eval()
    inputs = _input(2)
    perturbed = {"data": inputs["data"].clone()}
    perturbed["data"][:, 2:, ..., 1:] += 1000

    with torch.no_grad():
        baseline = model(inputs)["data"]
        changed = model(perturbed)["data"]

    torch.testing.assert_close(changed, baseline)


def test_target_time_forcing_changes_forecast_but_not_reconstruction() -> None:
    model = _make_model()
    model.eval()
    inputs = _input(1)
    perturbed = {"data": inputs["data"].clone()}
    perturbed["data"][:, 2:, ..., 0] += 1000

    with torch.no_grad():
        baseline = model(inputs)["data"]
        changed = model(perturbed)["data"]

    torch.testing.assert_close(changed[:, :1], baseline[:, :1])
    assert not torch.allclose(changed[:, 1:], baseline[:, 1:])


def test_shared_encoder_runs_only_for_the_two_history_snapshots() -> None:
    model = _make_model(forecast_steps=2)
    encoder_calls = 0
    forcing_encoder_calls = 0

    def count_encoder(*_args) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    def count_forcing_encoder(*_args) -> None:
        nonlocal forcing_encoder_calls
        forcing_encoder_calls += 1

    encoder_hook = model.encoder["data"].register_forward_hook(count_encoder)
    forcing_hook = model.forcing_encoder["data"].register_forward_hook(count_forcing_encoder)
    try:
        model(_input(2))
    finally:
        encoder_hook.remove()
        forcing_hook.remove()

    assert encoder_calls == 2
    assert forcing_encoder_calls == 2


def test_current_analysis_only_encodes_one_state() -> None:
    model = _make_model(forecast_steps=2, use_previous_state=False)
    encoder_calls = 0

    def count_encoder(*_args) -> None:
        nonlocal encoder_calls
        encoder_calls += 1

    hook = model.encoder["data"].register_forward_hook(count_encoder)
    try:
        model(_input(2, use_previous_state=False))
    finally:
        hook.remove()

    assert encoder_calls == 1


def test_static_forcing_encoder_runs_once_per_forward() -> None:
    config = _model_config()
    config.model.static_forcing_variables = ["forcing"]
    config.model.temporal_forcing_variables = []
    config.model.static_forcing_context_channels = 1
    model = AnemoiModelPredictiveAutoEncoder(
        model_config=config,
        data_indices=_data_indices(),
        statistics={"data": {}},
        n_step_input=4,
        n_step_output=3,
        graph_data=_graph(),
    )
    static_encoder_calls = 0

    def count_static_encoder(*_args) -> None:
        nonlocal static_encoder_calls
        static_encoder_calls += 1

    hook = model.static_forcing_encoder["data"].register_forward_hook(count_static_encoder)
    try:
        model(_input(2))
    finally:
        hook.remove()

    assert static_encoder_calls == 1


def test_reconstruction_and_forecast_losses_reach_expected_modules() -> None:
    model = _make_model(forecast_steps=2)
    output = model(_input(2))["data"]

    output[:, 0].square().mean().backward()
    assert _nonzero_gradient(model.encoder.parameters())
    assert _nonzero_gradient(model.decoder.parameters())

    model.zero_grad(set_to_none=True)
    output = model(_input(2))["data"]
    output[:, 1:].square().mean().backward()
    assert _nonzero_gradient(model.encoder.parameters())
    assert _nonzero_gradient(model.forcing_encoder.parameters())
    assert _nonzero_gradient(model.state_context_mixer.parameters())
    assert _nonzero_gradient(model.processor.parameters())
    assert _nonzero_gradient(model.decoder.parameters())


def test_latent_skip_adds_the_current_state_to_transition_delta() -> None:
    model = _make_model(latent_skip=True)
    inputs = _input(1)
    previous, shard_sizes = model.encode_snapshot(inputs, 0, batch_size=2)
    current, _ = model.encode_snapshot(inputs, 1, batch_size=2)
    context, _ = model.encode_forcing_context(inputs, 2, batch_size=2)

    mixed = model.state_context_mixer(previous, current, context)
    expected_delta = model.processor(
        x=mixed,
        batch_size=2,
        shard_info=GraphShardInfo(nodes=shard_sizes, edges=None),
        edge_attr=None,
        edge_index=None,
    )
    transitioned = model.transition_latent(
        previous,
        current,
        context,
        batch_size=2,
        shard_sizes_hidden=shard_sizes,
    )

    torch.testing.assert_close(transitioned, current + expected_delta)


def test_scalar_bottleneck_statistics_are_exposed() -> None:
    statistics = _make_model().latent_scalar_statistics["data"]

    assert statistics["physical_prognostic_scalars_per_snapshot"] == 12
    assert statistics["latent_scalars_per_snapshot"] == 8
    assert statistics["ratio"] == pytest.approx(2 / 3)
    assert statistics["transition_state_count"] == 2
    assert statistics["transition_state_scalars"] == 16
    assert statistics["two_state_transition_scalars"] == 16


def test_single_state_scalar_statistics_are_exposed() -> None:
    statistics = _make_model(use_previous_state=False).latent_scalar_statistics["data"]

    assert statistics["transition_state_count"] == 1
    assert statistics["transition_state_scalars"] == 8


def test_model_schema_accepts_predictive_components() -> None:
    model_section = OmegaConf.to_container(_model_config().model, resolve=True)

    validated = BaseModelSchema.model_validate(model_section)

    assert validated.model.target_ == "anemoi.models.models.AnemoiModelPredictiveAutoEncoder"
    assert validated.forcing_encoder is not None
    assert validated.state_context_mixer is not None


def test_noop_transition_processor_is_rejected() -> None:
    config = _model_config()
    config.model.processor = _component("anemoi.models.layers.processor.NoOpProcessor")

    with pytest.raises(TypeError, match="real latent transition processor"):
        AnemoiModelPredictiveAutoEncoder(
            model_config=config,
            data_indices=_data_indices(),
            statistics={"data": {}},
            n_step_input=3,
            n_step_output=2,
            graph_data=_graph(),
        )


@pytest.mark.parametrize(
    ("setting", "expected", "message"),
    [
        ("expected_num_forcing_fields", 2, "exactly 2 forcing fields"),
        ("expected_num_prognostic_fields", 4, "exactly 4 prognostic fields"),
    ],
)
def test_expected_channel_count_mismatch_fails_with_actionable_message(
    setting: str,
    expected: int,
    message: str,
) -> None:
    config = _model_config()
    config.model.model[setting] = expected

    with pytest.raises(ValueError, match=message):
        AnemoiModelPredictiveAutoEncoder(
            model_config=config,
            data_indices=_data_indices(),
            statistics={"data": {}},
            n_step_input=3,
            n_step_output=2,
            graph_data=_graph(),
        )
