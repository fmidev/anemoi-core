# anemoi/models/models/temporal_prognostic_decoder.py

import logging
from typing import Optional

import einops
import torch
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

# Import the specific Graph Transformer Backward Mapper
from anemoi.models.layers.mapper import GraphTransformerBackwardMapper
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.utils.config import DotDict  # For DotDict in layer_kernels if used

LOGGER = logging.getLogger(__name__)


class AnemoiTemporalPrognosticDecoder(nn.Module):
    """
    A new trainable decoder to produce high temporal resolution (hourly) prognostic
    variable predictions (e.g., 2t) based on the latent states from a frozen
    AnemoiModelEncProcDec (main model).
    This decoder interpolates between latent states at t0 and t+6h, conditioned
    on a time fraction.
    """

    def __init__(
        self,
        *,
        latent_dim: int,  # e.g., 1024
        output_channels: int,  # e.g., 1 (for 2t)
        hidden_graph_name: str,  # e.g., "hidden"
        data_graph_name: str,  # e.g., "grid_720x1440" or whatever your output grid is called
        graph_data: HeteroData,  # The overall graph definition
        # Additional Anemoi-specific parameters to pass down to GraphTransformerBackwardMapper
        # These would normally come from model_config.model.attributes or similar
        sub_graph_edge_attributes: list[str],  # e.g., ["edge_dist", "edge_direction"]
        cpu_offload: bool = False,
        layer_kernels: Optional[
            DotDict
        ] = None,  # Use DotDict for type hint consistency
    ) -> None:
        super().__init__()

        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self._graph_name_hidden = hidden_graph_name
        self._graph_name_data = data_graph_name
        self.graph_data = graph_data
        self.cpu_offload = cpu_offload
        self.layer_kernels = layer_kernels

        # --- Time Embedding ---
        self.time_conditioning_dim = 64  # Size of the time embedding vector
        self.time_embedding = nn.Sequential(
            nn.Linear(
                1, self.time_conditioning_dim
            ),  # Maps scalar time_fraction [0,1] to a vector
            nn.SiLU(),
            nn.Linear(self.time_conditioning_dim, self.time_conditioning_dim),
        )

        # --- Node Attributes for GraphTransformerBackwardMapper ---
        # The GraphTransformer* mappers use `NamedNodesAttributes` internally or expect them
        # to be implicitly handled by the passed sub_graph.
        # However, `in_channels_dst` refers to the number of *attribute channels* for the destination nodes.
        # We need to correctly pass the node attributes for the hidden and data graphs.
        # The `NamedNodesAttributes` is primarily for fetching these for mapping.
        # The internal 'attr_ndims' is the crucial part for in_channels_dst.

        # For the GraphTransformerBackwardMapper's internal needs, it will look up node attributes
        # using 'sub_graph' for the src and dst nodes. We provide that via the graph_data.
        # The `in_channels_dst` parameter to GraphTransformerBackwardMapper is the number of features
        # associated with each node in the *destination* (data) graph, *in addition* to the features
        # being mapped from the source (hidden) graph. These are typically geographic coordinates, LSM etc.

        # We will assume that node attributes for _graph_name_data are accessible through graph_data's nodes.
        # The number of such channels is accessed via self.graph_data[self._graph_name_data].x.shape[-1]
        # or via NamedNodesAttributes if that's how they are consistently handled.
        # Let's instantiate NamedNodesAttributes correctly, assuming a dummy config dict for it for now.
        # You'll need to confirm the actual config key for these when integrating.
        self.node_attributes_hidden = NamedNodesAttributes(
            {
                "_graph_name_hidden": {}
            },  # dummy config, `NamedNodesAttributes` needs dict-like input
            self.graph_data,
        )
        self.node_attributes_data = NamedNodesAttributes(
            {"_graph_name_data": {}}, self.graph_data  # dummy config
        )

        # --- Core Decoder (GraphTransformerBackwardMapper) ---
        # This module will map from the hidden graph latent space to the data grid.
        self.decoder_core = GraphTransformerBackwardMapper(
            in_channels_src=self.latent_dim * 2
            + self.time_conditioning_dim,  # Concatenated latent states + time embedding
            in_channels_dst=self.node_attributes_data.attr_ndims[
                self._graph_name_data
            ],  # Features of data grid nodes (e.g., lat/lon)
            hidden_dim=self.latent_dim,  # Internal dimension for transformer layers
            out_channels_dst=self.output_channels,  # 1 for '2t'
            sub_graph=self.graph_data[
                (self._graph_name_hidden, "to", self._graph_name_data)
            ],
            sub_graph_edge_attributes=sub_graph_edge_attributes,  # Explicitly passed, e.g., ["edge_dist", "edge_direction"]
            src_grid_size=self.node_attributes_hidden.num_nodes[
                self._graph_name_hidden
            ],
            dst_grid_size=self.node_attributes_data.num_nodes[self._graph_name_data],
            # Parameters from your config:
            trainable_size=self.latent_dim,  # Common choice: matching latent_dim or hidden_dim
            num_chunks=1,
            num_heads=8,
            mlp_hidden_ratio=4,
            initialise_data_extractor_zero=False,
            qk_norm=False,
            cpu_offload=self.cpu_offload,
            layer_kernels=self.layer_kernels,
        )

    def forward(
        self,
        x_latent_t0: Tensor,
        x_latent_t6: Tensor,
        time_fraction: Tensor,  # Shape: [batch_size, 1] (scalar for each batch item)
        model_comm_group: Optional[ProcessGroup] = None,
        # grid_shard_shapes is now handled internally by GraphTransformerBackwardMapper
        **kwargs,  # Catch any extra args not used by this decoder
    ) -> Tensor:
        num_hidden_nodes = self.node_attributes_hidden.num_nodes[
            self._graph_name_hidden
        ]
        # Calculate actual_batch_ensemble_size if ensemble is folded into batch dimension of latent states
        # The assumption is that x_latent_t0.shape[0] = batch_size * ensemble_size * num_hidden_nodes
        # So (batch * ensemble) = x_latent_t0.shape[0] // num_hidden_nodes
        actual_batch_ensemble_size = x_latent_t0.shape[0] // num_hidden_nodes

        # Expand time_fraction to match the number of combined batch*ensemble*nodes for concatenation
        # time_fraction.shape is (batch_size, 1)
        # We need it to be (batch_size * ensemble_size * num_hidden_nodes, 1) to match concatenated latent features
        # If actual_batch_ensemble_size is already (batch_size * ensemble_size), then:
        time_fraction_expanded = einops.repeat(
            time_fraction,
            "b 1 -> (b e g) 1",  # Repeat for each ensemble and each hidden node
            e=actual_batch_ensemble_size
            // time_fraction.shape[0],  # Deduce ensemble size from shapes
            g=num_hidden_nodes,
        )

        time_embedding = self.time_embedding(
            time_fraction_expanded
        )  # Output: (batch*ensemble*num_hidden_nodes, time_conditioning_dim)

        # Concatenate t0, t+6h latent states and time embedding
        x_combined_latent = torch.cat(
            (x_latent_t0, x_latent_t6, time_embedding),
            dim=-1,  # Concatenate along feature dimension
        )

        # The GraphTransformerBackwardMapper will internally fetch destination node attributes
        # and handle sharding if model_comm_group is provided.
        # Its 'x_src' argument is our x_combined_latent.
        # Its 'x_dst' argument is the destination node attributes (passed as None, it fetches them)
        # We pass batch_size as actual_batch_ensemble_size as that's what the mapper expects for its first dim.
        x_out = self.decoder_core(
            x_combined_latent,
            None,  # x_dst is None, GraphTransformerBackwardMapper fetches it internally via sub_graph
            batch_size=actual_batch_ensemble_size,
            shard_shapes=(
                get_shard_shapes(
                    x_combined_latent, 0, model_comm_group
                ),  # Source (hidden) shards
                None,  # Destination shards are handled internally when x_dst is None
            ),
            model_comm_group=model_comm_group,
        )

        # Reshape output back to (batch, ensemble, grid, vars)
        num_data_nodes = self.node_attributes_data.num_nodes[self._graph_name_data]

        # We need to properly separate batch and ensemble here.
        # Assuming `time_fraction.shape[0]` is the true `batch_size`
        original_batch_size = time_fraction.shape[0]
        ensemble_size_deduced = actual_batch_ensemble_size // original_batch_size

        x_out = (
            einops.rearrange(
                x_out,
                "(b_e num_data_nodes) vars -> b_e num_data_nodes vars",  # Temporary reshape to 3D
                num_data_nodes=num_data_nodes,
            )
            .to(dtype=x_latent_t0.dtype)
            .clone()
        )

        # Finally, split batch_ensemble back into batch and ensemble
        x_out = einops.rearrange(
            x_out,
            "(b e) g v -> b e g v",
            b=original_batch_size,
            e=ensemble_size_deduced,
        )

        return x_out
