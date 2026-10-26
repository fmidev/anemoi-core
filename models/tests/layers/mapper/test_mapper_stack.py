# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Opt-in mapper stacks preserve the standard mapper execution interface."""

import json
from pathlib import Path

import pytest
import torch

from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper as StandardBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper as StandardForwardMapper
from anemoi.models.layers.mapper_stack import GraphTransformerMapperStack
from anemoi.models.layers.mapper_stack import StackedGraphTransformerBackwardMapper as GraphTransformerBackwardMapper
from anemoi.models.layers.mapper_stack import StackedGraphTransformerForwardMapper as GraphTransformerForwardMapper
from anemoi.models.layers.utils import load_layer_kernels


@pytest.mark.parametrize("mapper_class", [GraphTransformerForwardMapper, GraphTransformerBackwardMapper])
@pytest.mark.parametrize("strategy", ["edges", "heads"])
@pytest.mark.parametrize("checkpointing", [False, True])
def test_multilayer_mapper_gradients_and_chunking(mapper_class, strategy, checkpointing):
    """All blocks refine the destination, independently of memory chunking."""
    torch.manual_seed(42)
    is_encoder = mapper_class is GraphTransformerForwardMapper
    mapper = mapper_class(
        in_channels_src=5,
        in_channels_dst=3,
        num_channels=16,
        out_channels_dst=None if is_encoder else 7,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        edge_dim=3,
        num_layers=3,
        shard_strategy=strategy,
        graph_attention_backend="pyg",
        gradient_checkpointing=checkpointing,
    )
    x = (torch.randn(4, 5), torch.randn(6, 3))
    edge_index = torch.tensor([[src, dst] for dst in range(6) for src in range(4)]).T
    edge_attr = torch.randn(24, 3)
    shard_info = BipartiteGraphShardInfo(src_nodes=[4], dst_nodes=[6], edges=[24])

    def run():
        output = mapper(x, 1, shard_info, edge_attr, edge_index)
        if is_encoder:
            assert torch.equal(output[0], x[0])
            output = output[1]
        assert output.shape == (6, 16 if is_encoder else 7)
        return output

    output = run()
    output.square().mean().backward()
    assert isinstance(mapper.proc, GraphTransformerMapperStack)
    blocks = list(mapper.proc.blocks)
    assert len(blocks) == 3
    assert len({block.lin_query.weight.data_ptr() for block in blocks}) == 3
    for block in blocks:
        assert block.lin_query.weight.grad is not None
        assert torch.isfinite(block.lin_query.weight.grad).all()
        assert block.lin_query.weight.grad.abs().sum() > 0
    mapper.num_chunks = 3
    torch.testing.assert_close(run(), output)


def test_standard_mapper_serialization(tmp_path):
    """State dictionaries and full-model serialization preserve predictions."""
    mapper = StandardForwardMapper(
        in_channels_src=5,
        in_channels_dst=3,
        num_channels=16,
        num_chunks=1,
        num_heads=4,
        mlp_hidden_ratio=2,
        edge_dim=3,
        graph_attention_backend="pyg",
        gradient_checkpointing=False,
    )
    assert isinstance(mapper.proc, GraphTransformerMapperBlock)
    assert "proc.lin_query.weight" in mapper.state_dict()
    old_state = mapper.state_dict()
    path = tmp_path / "old.ckpt"
    torch.save(mapper, path)
    restored = torch.load(path, weights_only=False)
    restored.load_state_dict(old_state, strict=True)
    edge_index = torch.tensor([[src, dst] for dst in range(6) for src in range(4)]).T
    edge_attr = torch.randn(24, 3)
    shard_info = BipartiteGraphShardInfo(src_nodes=[4], dst_nodes=[6], edges=[24])
    physical = (torch.randn(4, 5), torch.randn(6, 3))
    torch.testing.assert_close(
        restored(physical, 1, shard_info, edge_attr, edge_index),
        mapper(physical, 1, shard_info, edge_attr, edge_index),
    )


@pytest.mark.parametrize("depth", [0, -1, 1.5, True])
def test_invalid_mapper_depth(depth):
    with pytest.raises(ValueError, match="num_layers must be a positive integer"):
        GraphTransformerForwardMapper(
            in_channels_src=5,
            in_channels_dst=3,
            num_channels=16,
            num_chunks=1,
            num_heads=4,
            mlp_hidden_ratio=2,
            edge_dim=3,
            graph_attention_backend="pyg",
            num_layers=depth,
        )


def test_mapper_stack_reuses_source_and_forwards_block_arguments():
    """Stack composition must preserve source features and graph arguments."""
    from unittest.mock import Mock

    stack = GraphTransformerMapperStack(
        num_layers=3,
        in_channels=8,
        hidden_dim=16,
        out_channels=8,
        num_heads=2,
        edge_dim=3,
        layer_kernels=load_layer_kernels(),
        graph_attention_backend="pyg",
    )
    source = torch.randn(4, 8)
    destination = torch.randn(6, 8)
    edges = torch.randn(24, 3)
    expected_source = source
    for block in stack.blocks:
        block.forward = Mock(side_effect=lambda x, edge_attr, *args, **kwargs: ((x[0] + 100, x[1] + 1), edge_attr + 1))
    (out_source, out_destination), out_edges = stack(
        (source, destination), edges, "graph-argument", cond="forcing-argument"
    )
    torch.testing.assert_close(out_source, source + 100)
    torch.testing.assert_close(out_destination, destination + 3)
    torch.testing.assert_close(out_edges, edges + 3)
    for index, block in enumerate(stack.blocks):
        args, kwargs = block.forward.call_args
        assert args[0][0] is expected_source
        torch.testing.assert_close(args[0][1], destination + index)
        torch.testing.assert_close(args[1], edges + index)
        assert args[2] == "graph-argument" and kwargs == {"cond": "forcing-argument"}


@pytest.mark.parametrize(
    "direction,mapper_class", [("Forward", StandardForwardMapper), ("Backward", StandardBackwardMapper)]
)
def test_standard_mapper_matches_upstream_checkpoint_reference(direction, mapper_class):
    reference = json.loads((Path(__file__).parent / "fixtures/upstream_graph_transformer.json").read_text())
    case = reference["cases"][direction]
    mapper = mapper_class(**case["kwargs"]).eval()
    mapper.load_state_dict({key: torch.tensor(value) for key, value in case["state"].items()}, strict=True)
    sizes = BipartiteGraphShardInfo(src_nodes=[2], dst_nodes=[3], edges=[6])
    output = mapper(
        (torch.tensor(case["source"]), torch.tensor(case["target"])),
        1,
        sizes,
        torch.tensor(case["edge_attr"]),
        torch.tensor(case["edge_index"]),
    )
    if direction == "Forward":
        output = output[1]
    torch.testing.assert_close(output, torch.tensor(case["output"]), rtol=1e-5, atol=1e-6)
