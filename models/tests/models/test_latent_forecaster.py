# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Latent state, prescribed forcing and optional coupling contracts."""

from unittest.mock import patch

import pytest
import torch
from hydra import compose
from hydra import initialize_config_module
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.models.latent_forecaster import AnemoiModelLatentForecaster


def make_model(decoder_features=None):
    torch.manual_seed(3)
    with initialize_config_module(config_module="anemoi.training.config", version_base=None):
        c = compose(config_name="latent_forecaster")
    OmegaConf.set_struct(c, False)
    if decoder_features is not None:
        c.model.decoders["0"].target_node_features = decoder_features
    c.model.num_channels = 8
    c.model.processor.num_layers = 2
    c.model.processor.num_chunks = 1
    c.model.processor.num_heads = 2
    c.model.processor.graph_attention_backend = "pyg"
    for routes in (c.model.encoders, c.model.decoders):
        for route in routes.values():
            route.mapper.num_layers = 2
            route.mapper.num_chunks = 1
            route.mapper.num_heads = 2
            route.mapper.graph_attention_backend = "pyg"
    graph = HeteroData()
    coordinates = torch.tensor([[-0.8, 0.0], [-0.3, 1.0], [0.3, 2.0], [0.8, 3.0]])
    for name in ["data", "hidden"]:
        graph[name].x = coordinates.clone()
        graph[name].num_nodes = 4
    graph["data"].area_weight = torch.ones(4, 1)
    edges = torch.cartesian_prod(torch.arange(4), torch.arange(4)).T
    for source, target in [("data", "hidden"), ("hidden", "hidden"), ("hidden", "data")]:
        relation = graph[source, "to", target]
        relation.edge_index = edges
        relation.edge_length = torch.ones(edges.shape[1], 1)
        relation.edge_dirs = torch.ones(edges.shape[1], 2)
    indices = {
        "data": IndexCollection(c.data.datasets.data, {name: i for i, name in enumerate(c.dataloader.dataset.select)})
    }
    model = AnemoiModelLatentForecaster(
        model_config=c.model,
        data_indices=indices,
        statistics={"data": {}},
        n_step_input=2,
        n_step_output=2,
        graph_data=graph,
    )
    return model.eval(), indices


def inputs(model):
    return {"data": torch.randn(1, 2, 1, 4, len(model.data_indices["data"].model.input.full))}


def test_encode_advance_decode_and_no_message_contract():
    model, _ = make_model()
    x = inputs(model)
    z, sizes = model.encode_snapshot(x, 0, batch_size=1)
    static = model.static_condition(x, batch_size=1)
    condition = model.condition(x, 1, batch_size=1, static=static)
    result = model.advance(z, condition, batch_size=1, shard_sizes_hidden=sizes, step="6h")
    zero = model.advance(z, condition, batch_size=1, shard_sizes_hidden=sizes, message=torch.zeros_like(z))
    torch.testing.assert_close(result, zero, rtol=0, atol=0)
    assert z.shape == (4, 8) and condition.shape == (4, 64)
    assert model(x)["data"].shape == (1, 2, 1, 4, 87)
    with pytest.raises(ValueError, match="interval"):
        model.advance(z, condition, batch_size=1, shard_sizes_hidden=sizes, step="3h")
    with pytest.raises(ValueError, match="receiving"):
        model.advance(z, condition, batch_size=1, shard_sizes_hidden=sizes, message=z[:, :2])


def test_latent_target_has_no_gradient_and_forcings_do_not_depend_on_future_state():
    model, indices = make_model()
    x = inputs(model)
    x["data"].requires_grad_()
    z, sizes = model.encode_snapshot(x, 0, batch_size=1)
    with torch.no_grad():
        target, _ = model.encode_snapshot(x, 1, batch_size=1)
    static = model.static_condition(x, batch_size=1)
    condition = model.condition(x, 1, batch_size=1, static=static)
    alternate = {"data": x["data"].detach().clone()}
    alternate["data"][:, 1, ..., indices["data"].model.input.prognostic.long()] += 100
    torch.testing.assert_close(model.condition(alternate, 1, batch_size=1, static=static), condition)
    predicted = model.advance(z, condition, batch_size=1, shard_sizes_hidden=sizes)
    (predicted - target).square().mean().backward()
    assert not target.requires_grad
    assert x["data"].grad[:, 0].abs().sum() > 0
    assert x["data"].grad[:, 1, ..., indices["data"].model.input.prognostic.long()].abs().sum() == 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.processor.parameters())


def test_static_encoding_once_per_rollout():
    model, _ = make_model()
    x = inputs(model)
    with patch.object(
        model.static_embeddings["data"], "forward", wraps=model.static_embeddings["data"].forward
    ) as static:
        model(x)
        assert static.call_count == 1
        model(x)
        assert static.call_count == 2


@pytest.mark.parametrize(
    "features,width",
    [
        (["coordinates"], 4),
        (["trainable_parameters"], 8),
        (["forcings"], 11),
        (["forcings", "coordinates", "trainable_parameters"], 23),
    ],
)
def test_decoder_respects_selected_target_features(features, width):
    model, _ = make_model(features)
    x = inputs(model)
    assert model.target_dim["data"] == width
    assert model.decoders_target_input["0"].dim == width
    target, _ = model._assemble_targets(x["data"][:, :1], None, 1, dataset_name="data")
    assert target.shape == (4, width)
    latent, sizes = model.encode_snapshot(x, 0, batch_size=1)
    args = {"batch_size": 1, "ensemble_size": 1, "shard_sizes_hidden": sizes, "in_out_sharded": {"data": False}}
    output = model.decode_snapshot(latent, x, 0, **args)["data"]
    assert output.shape == (1, 1, 1, 4, 87)
    if "forcings" not in features:
        altered = {"data": x["data"].clone()}
        altered["data"][..., model._forcing_input_idx["data"]] += 100
        torch.testing.assert_close(model.decode_snapshot(latent, altered, 0, **args)["data"], output)


@pytest.mark.parametrize("feature", ["prognostics", "encoded_data"])
def test_decoder_rejects_atmospheric_input_shortcuts(feature):
    with pytest.raises(ValueError, match="must not bypass"):
        make_model([feature])
