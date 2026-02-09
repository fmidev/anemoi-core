# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import Optional
import einops
import torch
from omegaconf import ListConfig
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes

LOGGER = logging.getLogger(__name__)


class AnemoiObsFuser(nn.Module):
    """Multi-dataset observation fuser model for Anemoi."""
    
    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: ListConfig,
        graph_data: HeteroData,
        statistics: dict,
        truncation_data: dict | None = None,
    ) -> None:
        super().__init__()
        model_config = DotDict(model_config)

        self.use_obs_fuser = model_config.model.use_obs_fuser
        self.use_skip_connection_for_decoder1 = getattr(model_config.model, "use_skip_connection_for_decoder1", False)
        self._truncation_data = truncation_data

        self._graph_data = graph_data
        self._graph_name_hidden = model_config.graph.hidden
        self._graph_names_data = tuple(name for name in model_config.graph.input_nodes)
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        assert isinstance(data_indices, (list, ListConfig)), f"data_indices must be a list or ListConfig, is a {type(data_indices)}"

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)
        self.data_indices = data_indices
        self.statistics = statistics


        # we can't register these as buffers because DDP does not support sparse tensors
        # these will be moved to the GPU when first used via sefl.interpolate_down/interpolate_up
        self.A_down, self.A_up = None, None
        if "down" in self._truncation_data:
            self.A_down = self._make_truncation_matrix(self._truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if "up" in self._truncation_data:
            self.A_up = self._make_truncation_matrix(self._truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

        self.supports_sharded_input = False # for now

        input_dim = tuple(
            self.multi_step * self.num_input_channels[dset_idx] + self.node_attributes.attr_ndims[dset]
            for dset_idx, dset in enumerate(self._graph_names_data)
        )

        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=input_dim[0],
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_names_data[0], "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        if self.use_obs_fuser:
            self.encoders_obs = nn.ModuleList(
                [
                    instantiate(
                        model_config.model.encoder_obs,
                        _recursive_=False,  # Avoids instantiation of layer_kernels here
                        in_channels_src=input_dim[dset_idx],
                        in_channels_dst=self.num_channels,
                        hidden_dim=self.num_channels,
                        sub_graph=self._graph_data[(self._graph_names_data[dset_idx], "to", self._graph_name_hidden)],
                        src_grid_size=self.node_attributes.num_nodes[self._graph_names_data[dset_idx]],
                        dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                    )
                    for dset_idx, dset in enumerate(self._graph_names_data)
                    if dset != self._graph_names_data[0]
                ]
            )

        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,  # Avoids instantiation of layer_kernels here
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim[0],
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels[0],
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_names_data[0])],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]],
        )

        self.decoders_extra = nn.ModuleList(
            [
                instantiate(
                    model_config.model.decoder_extra,
                    _recursive_=False,  # Avoids instantiation of layer_kernels here
                    in_channels_src=self.num_channels,
                    in_channels_dst=input_dim[dset_idx] + (input_dim[0] if self.use_skip_connection_for_decoder1 else 0),
                    hidden_dim=self.num_channels,
                    out_channels_dst=self.num_output_channels[dset_idx],
                    sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_names_data[dset_idx])],
                    src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                    dst_grid_size=self.node_attributes.num_nodes[self._graph_names_data[dset_idx]],
                )
                for dset_idx, dset in enumerate(self._graph_names_data)
                if dset != self._graph_names_data[0]
            ]
        )

        self.processors_extra = None
        if hasattr(model_config.model, "processor_extra") and model_config.model.processor_extra is not None:
            LOGGER.info("Initializing implementations of processor_extra")
            self.processors_extra = nn.ModuleList(
                [
                    instantiate(
                        model_config.model.processor_extra,
                        _recursive_=False,  # Avoids instantiation of layer_kernels here
                        num_channels=self.num_channels,
                        sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
                        src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                        dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                    )
                    for dset_idx, dset in enumerate(self._graph_names_data)
                    if dset != self._graph_names_data[0]
                ]
            )

        self.boundings = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        instantiate(
                            cfg, 
                            name_to_index=self.data_indices[dset_index].model.output.name_to_index,
                            statistics=self.statistics,
                            name_to_index_stats=self.data_indices[dset_index].data.input.name_to_index)
                        for cfg in dset_boundings
                    ]
                )
                for dset_index, dset_boundings in enumerate(getattr(model_config.model, "bounding", []))
            ]
        )

    def _make_truncation_matrix(self, A, data_type=torch.float32):
        A_ = torch.sparse_coo_tensor(
            torch.tensor(np.vstack(A.nonzero()), dtype=torch.long),
            torch.tensor(A.data, dtype=data_type),
            size=A.shape,
        ).coalesce()
        return A_

    def _multiply_sparse(self, x, A):
        return torch.sparse.mm(A, x)

    def _truncate_fields(self, x, A, batch_size=None, auto_cast=False):
        if not batch_size:
            batch_size = x.shape[0]
        out = []
        with torch.amp.autocast(device_type="cuda", enabled=auto_cast):
            for i in range(batch_size):
                out.append(self._multiply_sparse(x[i, ...], A))
        return torch.stack(out)


    def _get_shard_shapes(self, x, dim=0, shard_shapes_dim=None, model_comm_group=None):
        if shard_shapes_dim is None:
            return get_shard_shapes(x, dim, model_comm_group)
        else:
            return apply_shard_shapes(x, dim, shard_shapes_dim)

    def _apply_truncation(self, x, grid_shard_shapes=None, model_comm_group=None):
        if self.A_down is not None or self.A_up is not None:
            if grid_shard_shapes is not None:
                shard_shapes = self._get_shard_shapes(x, 0, grid_shard_shapes, model_comm_group)
                # grid-sharded input: reshard to channel-shards to apply truncation
                x = shard_channels(x, shard_shapes, model_comm_group)  # we get the full sequence here

            # these can't be registered as buffers because ddp does not like to broadcast sparse tensors
            # hence we check that they are on the correct device ; copy should only happen in the first forward run
            if self.A_down is not None:
                self.A_down = self.A_down.to(x.device)
                x = self._truncate_fields(x, self.A_down)  # to coarse resolution
            if self.A_up is not None:
                self.A_up = self.A_up.to(x.device)
                x = self._truncate_fields(x, self.A_up)  # back to high resolution

            if grid_shard_shapes is not None:
                # back to grid-sharding as before
                x = gather_channels(x, shard_shapes, model_comm_group)

        return x

    def _calculate_shapes_and_indices(self, data_indices: tuple) -> None:
        """Calculate input/output shapes and indices for each dataset."""
        self.num_input_channels = tuple(len(indices.model.input) for indices in data_indices)
        self.num_output_channels = tuple(len(indices.model.output) for indices in data_indices)
        self.num_input_channels_prognostic = len(data_indices[0].model.input.prognostic)
        self._internal_input_idx = tuple(indices.model.input.prognostic for indices in data_indices)
        self._internal_output_idx = tuple(indices.model.output.prognostic for indices in data_indices)
        self.input_dim = (
            self.multi_step * self.num_input_channels[0] + self.node_attributes.attr_ndims[self._graph_names_data[0]]
        )

    def _assert_matching_indices(self, data_indices: tuple) -> None:
        """Assert that indices are consistent across datasets."""
        # Match cloudy-skies approach exactly
        for dset, indices in enumerate(data_indices):
            assert len(self._internal_output_idx[dset]) == len(indices.model.output.full) - len(
                indices.model.output.diagnostic
            )
            assert len(self._internal_input_idx[dset]) == len(self._internal_input_idx[dset])


    def _run_mapper(
        self,
        mapper: nn.Module,
        data: tuple[Tensor],
        batch_size: int,
        shard_shapes: tuple[tuple[int, int], tuple[int, int]],
        model_comm_group: Optional[ProcessGroup] = None,
        x_src_is_sharded: bool = False,
        x_dst_is_sharded: bool = False,
        keep_x_dst_sharded: bool = False,
        use_reentrant: bool = False,
    ) -> Tensor:
        """Run a mapper with optional checkpointing for memory efficiency."""
        # Direct call without activation checkpointing for deterministic eval
        return checkpoint(
            mapper,
            data,
            batch_size=batch_size,
            shard_shapes=shard_shapes,
            model_comm_group=model_comm_group,
            x_src_is_sharded=x_src_is_sharded,
            x_dst_is_sharded=x_dst_is_sharded,
            keep_x_dst_sharded=keep_x_dst_sharded,
            use_reentrant=use_reentrant,
        )

    def _assemble_input(self, x, batch_size, grid_shard_shapes=None, model_comm_group=None):
        x_skip = [None] * len(x)
        for dset, x_elem in enumerate(x):
            x_skip[dset] = x_elem[:, -1, ...]
            x_skip[dset] = einops.rearrange(x_skip[dset], "batch ensemble grid vars -> (batch ensemble) grid vars")
            x_skip[dset] = self._apply_truncation(x_skip[dset], grid_shard_shapes, model_comm_group)
            x_skip[dset] = einops.rearrange(x_skip[dset], "(batch ensemble) grid vars -> batch ensemble grid vars", batch=batch_size)

        node_attributes_data = self.node_attributes(self._graph_names_data[0], batch_size=batch_size)
        if grid_shard_shapes is not None:
            shard_shapes_nodes = self._get_shard_shapes(node_attributes_data, 0, grid_shard_shapes, model_comm_group)
            node_attributes_data = shard_tensor(node_attributes_data, 0, shard_shapes_nodes, model_comm_group)

        # normalize and add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x[0], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,  # feature dimension
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        x_obs_latent = [
            torch.cat(
                (
                    einops.rearrange(x_elem, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    self.node_attributes(self._graph_names_data[dset], batch_size=batch_size),
                ),
                dim=-1,
            )
            for dset, x_elem in enumerate(x[1:], start=1)
        ]

        shard_shapes_obs = [get_shard_shapes(x_data, 0, model_comm_group) for x_data in x_obs_latent]

        return x_data_latent, x_obs_latent, x_skip, shard_shapes_data, shard_shapes_obs

    def _assemble_output(self, x_out: list[Tensor], x_skip: list[Tensor], batch_size: int, ensemble_size: int, dtype: torch.dtype):
        for dset, x_out_elem in enumerate(x_out):
            x_out[dset] = (
                einops.rearrange(
                    x_out_elem,
                    "(batch ensemble grid) vars -> batch ensemble grid vars",
                    batch=batch_size,
                    ensemble=ensemble_size,
                )
                .to(dtype=dtype)
                .clone()
            )

            x_out[dset][..., self._internal_output_idx[dset]] += x_skip[dset][
                    :, :, :, self._internal_input_idx[dset]
                ]

            for bounding in self.boundings[dset]:
                x_out[dset] = bounding(x_out[dset])


    def forward(self, 
        x: list[Tensor], 
        *,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> list[Tensor]:
        """Forward pass through the observation fuser model."""
        batch_size = x[0].shape[0]
        ensemble_size = x[0].shape[2]

        in_out_sharded = grid_shard_shapes is not None

        assert not (
            in_out_sharded and (grid_shard_shapes is None or model_comm_group is None)
        ), "If input is sharded, grid_shard_shapes and model_comm_group must be provided."

        x_data_latent, x_obs_latent, x_skip, shard_shapes_data, shard_shapes_obs = self._assemble_input(
            x, batch_size, grid_shard_shapes, model_comm_group
        )

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)

        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)
        
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            x_dst_is_sharded=False,  # x_latent does not come sharded
            keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
        )

        if self.use_obs_fuser:
            x_latent = gather_tensor(
                x_latent, 0, change_channels_in_shape(shard_shapes_hidden, self.num_channels), model_comm_group
            )
            shard_shapes_latent = get_shard_shapes(x_latent, 0, model_comm_group)
            for dset, obs_encoder in enumerate(self.encoders_obs):
                x_obs_latent[dset], x_latent = self._run_mapper(
                    obs_encoder,
                    (x_obs_latent[dset], x_latent),
                    batch_size=batch_size,
                    shard_shapes=(shard_shapes_obs[dset], shard_shapes_latent),
                    model_comm_group=model_comm_group,
                )

        x_latent_proc = self.processor(
            x_latent,
            batch_size=batch_size,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        x_latent_proc = x_latent_proc + x_latent

        x_out = [None for _ in range(len(x))]

        x_out[0] = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=batch_size,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True,  # x_latent always comes sharded
            x_dst_is_sharded=in_out_sharded,  # x_data_latent comes sharded iff in_out_sharded
            keep_x_dst_sharded=in_out_sharded,  # keep x_out sharded iff in_out_sharded
        )

        for dset, decoder_extra in enumerate(self.decoders_extra):
            x_dst = x_obs_latent[dset]
            if self.use_skip_connection_for_decoder1:
                x_dst = torch.cat([x_dst, x_data_latent], dim=-1)

            x_latent_dset = x_latent_proc
            if self.processors_extra is not None:
                x_latent_dset = self.processors_extra[dset](
                    x_latent_dset,
                    batch_size=batch_size,
                    shard_shapes=shard_shapes_hidden,
                    model_comm_group=model_comm_group,
                )
                x_latent_dset = x_latent_dset + x_latent_proc

            x_out[dset + 1] = self._run_mapper(
                decoder_extra,
                (x_latent_dset, x_dst),
                batch_size=batch_size,
                shard_shapes=(shard_shapes_hidden, shard_shapes_obs[dset]),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
            )

        self._assemble_output(x_out, x_skip, batch_size, ensemble_size, x[0].dtype)
        return list(x_out)

