# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from typing import Optional
from typing import Union

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.typing import Adj
from torch_geometric.utils import bipartite_subgraph
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.utils import mask_to_index


def get_k_hop_edges(
    nodes: Tensor,
    edge_attr: Tensor,
    edge_index: Adj,
    num_hops: int = 1,
    num_nodes: Optional[int] = None,
    relabel_nodes: bool = False,
) -> tuple[Adj, Tensor]:
    """Return 1 hop subgraph.

    Parameters
    ----------
    nodes : Tensor
        destination nodes
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_hops: int, Optional, by default 1
        number of required hops

    Returns
    -------
    tuple[Adj, Tensor]
        K-hop subgraph of edge index and edge attributes
    """
    _, edge_index_k, _, edge_mask_k = k_hop_subgraph(
        node_idx=nodes,
        num_hops=num_hops,
        edge_index=edge_index,
        directed=True,
        num_nodes=num_nodes,
        relabel_nodes=relabel_nodes,
    )

    return edge_attr[mask_to_index(edge_mask_k)], edge_index_k


def sort_edges_1hop_sharding(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    mgroup: Optional[ProcessGroup] = None,
    relabel_nodes: bool = False,
) -> tuple[Adj, Tensor, list, list]:
    """Rearanges edges into 1 hop neighbourhoods for sharding across GPUs.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    mgroup : ProcessGroup
        model communication group

    Returns
    -------
    tuple[Adj, Tensor, list, list]
        edges sorted according to k hop neigh., edge attributes of sorted edges,
        shapes of edge indices for partitioning between GPUs, shapes of edge attr for partitioning between GPUs
    """
    if mgroup:
        num_chunks = dist.get_world_size(group=mgroup)

        edge_attr_list, edge_index_list = sort_edges_1hop_chunks(
            num_nodes, edge_attr, edge_index, num_chunks, relabel_nodes
        )

        edge_index_shapes = [x.shape for x in edge_index_list]
        edge_attr_shapes = [x.shape for x in edge_attr_list]

        return torch.cat(edge_attr_list, dim=0), torch.cat(edge_index_list, dim=1), edge_attr_shapes, edge_index_shapes

    return edge_attr, edge_index, [], []


def sort_edges_1hop_chunks(
    num_nodes: Union[int, tuple[int, int]],
    edge_attr: Tensor,
    edge_index: Adj,
    num_chunks: int,
    relabel_nodes: bool = False,
) -> tuple[list[Tensor], list[Adj]]:
    """Rearanges edges into 1 hop neighbourhood chunks.

    Parameters
    ----------
    num_nodes : Union[int, tuple[int, int]]
        Number of (target) nodes in Graph, tuple for bipartite graph
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_chunks : int
        number of chunks used if mgroup is None

    Returns
    -------
    tuple[list[Tensor], list[Adj]]
        list of sorted edge attribute chunks, list of sorted edge_index chunks
    """
    if isinstance(num_nodes, int):
        node_chunks = torch.arange(num_nodes, device=edge_index.device).tensor_split(num_chunks)
    else:
        nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
        node_chunks = torch.arange(num_nodes[1], device=edge_index.device).tensor_split(num_chunks)

    edge_index_list = []
    edge_attr_list = []
    for node_chunk in node_chunks:
        if isinstance(num_nodes, int):
            edge_attr_chunk, edge_index_chunk = get_k_hop_edges(
                node_chunk, edge_attr, edge_index, num_nodes=num_nodes, relabel_nodes=relabel_nodes
            )
        else:
            edge_index_chunk, edge_attr_chunk = bipartite_subgraph(
                (nodes_src, node_chunk),
                edge_index,
                edge_attr,
                size=(num_nodes[0], num_nodes[1]),
                relabel_nodes=relabel_nodes,
            )
        edge_index_list.append(edge_index_chunk)
        edge_attr_list.append(edge_attr_chunk)

    return edge_attr_list, edge_index_list


def drop_unconnected_src_nodes(x_src: Tensor, edge_index: Adj, num_nodes: tuple[int, int]) -> tuple[Tensor, Adj]:
    """Drop unconnected nodes from x_src and relabel edges.

    Parameters
    ----------
    x_src : Tensor
        source node features
    edge_attr : Tensor
        edge attributes
    edge_index : Adj
        edge index
    num_nodes : tuple[int, int]
        number of nodes in graph (src, dst)

    Returns
    -------
    tuple[Tensor, Adj]
        reduced node features, relabeled edge index (contiguous, starting from 0)
    """
    connected_src_nodes = torch.unique(edge_index[0])
    dst_nodes = torch.arange(num_nodes[1], device=x_src.device)

    edge_index_new, _ = bipartite_subgraph(
        (connected_src_nodes, dst_nodes),
        edge_index,
        size=num_nodes,
        relabel_nodes=True,
    )

    return x_src[connected_src_nodes], edge_index_new


def get_edges_sharding(
    num_nodes: tuple[int, int],
    edge_attr: Tensor,
    edge_index: Adj,
    mgroup: Optional[ProcessGroup] = None,
    relabel_nodes: bool = False,
):
    if mgroup:
        num_chunks = dist.get_world_size(group=mgroup)
        dst_node_chunks, edge_counts = partition_dst_nodes(
            edge_index,
            num_chunks,
            balanced=False,  # use unbalanced partitioning to match sharding of dst nodes
            return_edge_counts=True,
        )

        nodes_src = torch.arange(num_nodes[0], device=edge_index.device)
        my_dst_node_chunk = dst_node_chunks[dist.get_rank(group=mgroup)]

        edge_index, edge_attr = bipartite_subgraph(
            (nodes_src, my_dst_node_chunk),
            edge_index,
            edge_attr,
            size=num_nodes,
            relabel_nodes=relabel_nodes,  # relabel dst nodes to be contiguous, starting from 0
        )

        edge_attr_shapes = [(edge_count, *edge_attr.shape[1:]) for edge_count in edge_counts]
        edge_index_shapes = [(2, edge_count) for edge_count in edge_counts]

        return edge_attr, edge_index, edge_attr_shapes, edge_index_shapes

    return edge_attr, edge_index, [], []


def partition_dst_nodes(
    edge_index, num_chunks, balanced=False, return_edge_counts=False
) -> Union[list, tuple[list, list]]:
    """Partition destination nodes into chunks for distributed processing.
    Args:
        edge_index (torch.Tensor): The edge index tensor
            !! expected to hold contiguous src/dst node indices starting from 0 !!
        num_chunks (int): The number of chunks to split the destination nodes into.
        balanced (bool, optional): If True, partition the destination nodes into chunks
            balanced by the number of edges (greedy, not necessarily optimal).
    Returns:
        tuple[list, Optional[list]]: A tuple containing the partitioned destination nodes and
            the corresponding edge counts for each chunk.
    """

    # unique dst nodes (sorted), #adjacent edges per node
    dst_nodes, edge_counts = torch.unique(edge_index[1], return_counts=True, sorted=True)

    assert num_chunks < len(
        dst_nodes
    ), f"num_chunks ({num_chunks}) must be smaller than the number of unique dst nodes ({len(dst_nodes)})"

    if balanced:
        # split dst nodes into chunks s.t. each chunk has roughly the same number of edges
        cumsum_edge_counts = torch.cumsum(edge_counts, dim=0)
        # calculate target cumsum edge counts for each chunk with optimal balancing
        targets = float(cumsum_edge_counts[-1] / num_chunks) * torch.arange(1, num_chunks, device=edge_index.device)
        # approximate optimal target chunk boundaries
        boundaries = torch.searchsorted(cumsum_edge_counts, targets)

        # append first and last boundary [inclusive, exclusive)
        boundaries = torch.cat(
            [
                torch.tensor([0], device=edge_index.device),
                boundaries,
                torch.tensor([len(dst_nodes)], device=edge_index.device),
            ]
        )

        # maybe fall back to unbalanced partitioning instead?
        assert torch.all(boundaries[1:] > boundaries[:-1]), "Empty chunk in partition_dst_nodes."

        dst_node_chunks = [dst_nodes[boundaries[i] : boundaries[i + 1]] for i in range(num_chunks)]
    else:
        dst_node_chunks = torch.tensor_split(dst_nodes, num_chunks)

    if return_edge_counts:
        edge_counts = [edge_counts[dst_node_chunk].sum().item() for dst_node_chunk in dst_node_chunks]
        return dst_node_chunks, edge_counts

    return dst_node_chunks
