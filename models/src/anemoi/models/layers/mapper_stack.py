# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Multi-layer graph-transformer encoders and decoders."""

from torch import Tensor
from torch import nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch_geometric.typing import PairTensor

from anemoi.models.layers.block import GraphTransformerMapperBlock
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.mapper import GraphTransformerForwardMapper


class GraphTransformerMapperStack(nn.Module):
    """Stack of graph-transformer blocks for bipartite node mapping.

    Parameters
    ----------
    num_layers : int
        Number of independently parameterised mapper blocks.
    **block_kwargs
        Arguments passed to each ``GraphTransformerMapperBlock``.
    """

    def __init__(self, *, num_layers: int, **block_kwargs) -> None:
        super().__init__()
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            message = "num_layers must be a positive integer."
            raise ValueError(message)
        self.blocks = nn.ModuleList(GraphTransformerMapperBlock(**block_kwargs) for _ in range(num_layers))

    def forward(self, x: PairTensor, edge_attr: Tensor, *args, **kwargs):
        """Apply the mapper blocks to source and destination node features.

        Parameters
        ----------
        x : PairTensor
            Source and destination node-feature tensors.
        edge_attr : Tensor
            Bipartite edge features.
        *args
            Positional graph and sharding arguments passed to each block.
        **kwargs
            Keyword arguments passed to each block, including conditioning.

        Returns
        -------
        PairTensor
            Source and updated destination node features.
        Tensor
            Edge features returned by the final block.
        """
        source = x[0]
        for block in self.blocks:
            x, edge_attr = block((source, x[1]), edge_attr, *args, **kwargs)
        return x, edge_attr


class _StackedGraphTransformerMapper:
    """Shared construction of multi-layer graph-transformer mappers."""

    def __init__(self, *, num_layers: int = 1, **kwargs) -> None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int) or num_layers < 1:
            message = "num_layers must be a positive integer."
            raise ValueError(message)
        self.num_layers = num_layers
        super().__init__(**kwargs)

    def _build_mapper_block(self, **kwargs) -> nn.Module:
        return GraphTransformerMapperStack(num_layers=self.num_layers, **kwargs)

    def offload_layers(self, cpu_offload):
        if cpu_offload:
            self.proc = offload_wrapper(self.proc)


class StackedGraphTransformerForwardMapper(_StackedGraphTransformerMapper, GraphTransformerForwardMapper):
    """Graph-transformer encoder with multiple bipartite mapping blocks.

    Parameters
    ----------
    num_layers : int, default 1
        Number of mapper blocks.
    **kwargs
        Arguments passed to ``GraphTransformerForwardMapper``.
    """


class StackedGraphTransformerBackwardMapper(_StackedGraphTransformerMapper, GraphTransformerBackwardMapper):
    """Graph-transformer decoder with multiple bipartite mapping blocks.

    Parameters
    ----------
    num_layers : int, default 1
        Number of mapper blocks.
    **kwargs
        Arguments passed to ``GraphTransformerBackwardMapper``.
    """
