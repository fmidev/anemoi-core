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
import numpy as np
from hydra.utils import instantiate
from omegaconf import ListConfig
from torch import Tensor
from torch import nn
from torch.distributed.distributed_c10d import ProcessGroup
from torch.utils.checkpoint import checkpoint
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import gather_channels
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_channels
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.distributed.shapes import change_channels_in_shape
from anemoi.models.distributed.shapes import get_shard_shapes
from anemoi.models.layers.graph import NamedNodesAttributes
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class AnemoiEnsObsFuser(nn.Module):
    """Ensemble model with multiple decoders (ObsFuser style) and noise injection."""

    def __init__(
        self,
        *,
        model_config: DotDict,
        data_indices: ListConfig | list,
        graph_data: HeteroData,
        statistics: dict,
        truncation_data: dict | None = None,
    ) -> None:
        super().__init__()
        model_config = DotDict(model_config)

        if not isinstance(data_indices, (list, tuple, ListConfig)):
             if isinstance(data_indices, (dict, DotDict)):
                 data_indices = [data_indices]
             else:
                 raise TypeError(f"data_indices must be list or tuple, got {type(data_indices)}")

        self.data_indices = data_indices
        # Ensure we have the full indices available
        self.data_indices_full = data_indices
        
        self.statistics = statistics
        self._truncation_data = truncation_data
        self._graph_data = graph_data
        
        self.use_obs_fuser = False # model_config.model.use_obs_fuser
        self.use_skip_connection_for_decoder1 = getattr(model_config.model, "use_skip_connection_for_decoder1", False)

        self._graph_name_hidden = model_config.graph.hidden
        self._graph_names_data = tuple(name for name in model_config.graph.input_nodes)
        self.multi_step = model_config.training.multistep_input
        self.num_channels = model_config.model.num_channels

        self.node_attributes = NamedNodesAttributes(model_config.model.trainable_parameters.hidden, self._graph_data)

        self._calculate_shapes_and_indices(data_indices)
        self._assert_matching_indices(data_indices)

        # Truncation matrices
        self.A_down, self.A_up = None, None
        if self._truncation_data and "down" in self._truncation_data:
            self.A_down = self._make_truncation_matrix(self._truncation_data["down"])
            LOGGER.info("Truncation: A_down %s", self.A_down.shape)
        if self._truncation_data and "up" in self._truncation_data:
            self.A_up = self._make_truncation_matrix(self._truncation_data["up"])
            LOGGER.info("Truncation: A_up %s", self.A_up.shape)

        self.supports_sharded_input = False # for now

        # Input dimensions
        input_dim = tuple(
            self.multi_step * self.num_input_channels[dset_idx] + self.node_attributes.attr_ndims[dset]
            for dset_idx, dset in enumerate(self._graph_names_data)
        )
        
        # Adjust input dim for dataset 0 (prognostic) + noise injection (if needed in input? No, noise is latent)
        # But wait, AnemoiEnsModelEncProcDec adds to input_dim:
        # self.input_dim += self.num_input_channels_prognostic
        # self.input_dim += 1
        # This is for the "skip" connection features IN THE INPUT (x_skip appended to x) + fcstep
        # Let's match that logic for dataset 0
        input_dim_0_adjusted = input_dim[0] + self.num_input_channels_prognostic + 1

        # 1. Enzyme: Noise Injector
        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
        )

        # 2. Encoder (Dataset 0)
        self.encoder = instantiate(
            model_config.model.encoder,
            _recursive_=False,
            in_channels_src=input_dim_0_adjusted,
            in_channels_dst=self.node_attributes.attr_ndims[self._graph_name_hidden],
            hidden_dim=self.num_channels,
            sub_graph=self._graph_data[(self._graph_names_data[0], "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # 3. Processor
        self.processor = instantiate(
            model_config.model.processor,
            _recursive_=False,
            num_channels=self.num_channels,
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
        )

        # 4. Decoder (Dataset 0)
        self.decoder = instantiate(
            model_config.model.decoder,
            _recursive_=False,
            in_channels_src=self.num_channels,
            in_channels_dst=input_dim_0_adjusted, # Standard Decoder uses same dim for SRC and DST usually?
            # Wait, standard decoder uses (x_latent_proc, x_data_latent)
            # x_data_latent has dim input_dim_0_adjusted
            # So in_channels_dst should be input_dim_0_adjusted
            hidden_dim=self.num_channels,
            out_channels_dst=self.num_output_channels[0],
            sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_names_data[0])],
            src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            dst_grid_size=self.node_attributes.num_nodes[self._graph_names_data[0]],
        )

        # 5. Extra Decoders
        self.decoders_extra = nn.ModuleList(
            [
                instantiate(
                    model_config.model.decoder_extra,
                    _recursive_=False,
                    in_channels_src=self.num_channels,
                    in_channels_dst=input_dim[dset_idx] + (input_dim_0_adjusted if self.use_skip_connection_for_decoder1 else 0),
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
             self.processors_extra = nn.ModuleList(
                 [
                     instantiate(
                         model_config.model.processor_extra,
                         _recursive_=False,
                         num_channels=self.num_channels,
                         sub_graph=self._graph_data[(self._graph_name_hidden, "to", self._graph_name_hidden)],
                         src_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                         dst_grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                     )
                     for dset_idx, dset in enumerate(self._graph_names_data)
                     if dset != self._graph_names_data[0]
                 ]
             )

        # 6. Boundings
        # Handle list of lists properly
        boundings_cfg = getattr(model_config.model, "bounding", [])
        if boundings_cfg and isinstance(boundings_cfg[0], (list, tuple, ListConfig)):
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
                    for dset_index, dset_boundings in enumerate(boundings_cfg)
                ]
            )
        else:
             # Fallback if config is flat (should not happen in multi-decoder usually)
             self.boundings = nn.ModuleList([
                 nn.ModuleList([
                    instantiate(
                        cfg,
                         name_to_index=self.data_indices[0].model.output.name_to_index,
                         statistics=self.statistics,
                         name_to_index_stats=self.data_indices[0].data.input.name_to_index
                    ) for cfg in boundings_cfg
                 ])
             ])

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
                x = shard_channels(x, shard_shapes, model_comm_group)

            if self.A_down is not None:
                self.A_down = self.A_down.to(x.device)
                x = self._truncate_fields(x, self.A_down)
            if self.A_up is not None:
                self.A_up = self.A_up.to(x.device)
                x = self._truncate_fields(x, self.A_up)

            if grid_shard_shapes is not None:
                x = gather_channels(x, shard_shapes, model_comm_group)
        return x

    def _calculate_shapes_and_indices(self, data_indices: tuple) -> None:
        self.num_input_channels = tuple(len(indices.model.input) for indices in data_indices)
        self.num_output_channels = tuple(len(indices.model.output) for indices in data_indices)
        self.num_input_channels_prognostic = len(data_indices[0].model.input.prognostic)
        self._internal_input_idx = tuple(indices.model.input.prognostic for indices in data_indices)
        self._internal_output_idx = tuple(indices.model.output.prognostic for indices in data_indices)

    def _assert_matching_indices(self, data_indices: tuple) -> None:
        for dset, indices in enumerate(data_indices):
            assert len(self._internal_output_idx[dset]) == len(indices.model.output.full) - len(
                indices.model.output.diagnostic
            )

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

    def _assemble_input(self, x, fcstep, bse, grid_shard_shapes=None, model_comm_group=None):
        x_skip = [None] * len(x)
        
        # --- Dataset 0 (Main/Prognostic) ---
        x_skip[0] = x[0][:, -1, :, :, self._internal_input_idx[0]]
        x_skip[0] = einops.rearrange(x_skip[0], "batch ensemble grid vars -> (batch ensemble) grid vars")
        x_skip[0] = self._apply_truncation(x_skip[0], grid_shard_shapes, model_comm_group)
        
        node_attributes_data_0 = self.node_attributes(self._graph_names_data[0], batch_size=bse)
        if grid_shard_shapes is not None:
             shard_shapes_nodes_0 = self._get_shard_shapes(node_attributes_data_0, 0, grid_shard_shapes, model_comm_group)
             node_attributes_data_0 = shard_tensor(node_attributes_data_0, 0, shard_shapes_nodes_0, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x[0], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                einops.rearrange(x_skip[0], "bse grid vars -> (bse grid) vars"),
                node_attributes_data_0,
            ),
            dim=-1,
        )
        x_data_latent = torch.cat(
            (x_data_latent, torch.ones(x_data_latent.shape[:-1], device=x_data_latent.device).unsqueeze(-1) * fcstep),
            dim=-1,
        )
        shard_shapes_data = self._get_shard_shapes(x_data_latent, 0, grid_shard_shapes, model_comm_group)

        # --- Dataset > 0 (Extra/Diagnostic) ---
        x_obs_latent = []
        for dset_idx in range(1, len(x)):
            # Skip connection processing if needed for residual add later
            skip_tmp = x[dset_idx][:, -1, ...]
            # Assuming we want to use prognostic indices if they exist, or just all?
            # Encoders usually specific input indices. Output usually adds to prognostic.
            # Let's use internal_input_idx logic for consistency
            indices = self._internal_input_idx[dset_idx]
            if len(indices) > 0:
                skip_tmp = skip_tmp[..., indices]
            
            skip_tmp = einops.rearrange(skip_tmp, "batch ensemble grid vars -> (batch ensemble) grid vars")
            skip_tmp = self._apply_truncation(skip_tmp, grid_shard_shapes, model_comm_group)
            x_skip[dset_idx] = skip_tmp
            
            node_attr = self.node_attributes(self._graph_names_data[dset_idx], batch_size=bse)
            if grid_shard_shapes is not None:
                  shard_shapes_nodes = self._get_shard_shapes(node_attr, 0, grid_shard_shapes, model_comm_group)
                  node_attr = shard_tensor(node_attr, 0, shard_shapes_nodes, model_comm_group)

            tmp = torch.cat(
                (
                    einops.rearrange(x[dset_idx], "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                    node_attr,
                ),
                dim=-1
            )
            x_obs_latent.append(tmp)

        return x_data_latent, x_obs_latent, x_skip, shard_shapes_data

    def _assemble_output(self, x_out, x_skip, batch_size, bse, dtype):
        for dset_idx, out in enumerate(x_out):
             out = einops.rearrange(out, "(bse n) f -> bse n f", bse=bse)
             out = einops.rearrange(out, "(bs e) n f -> bs e n f", bs=batch_size).to(dtype=dtype).clone()

             assert not torch.isnan(out).any(), f"NaN in _assemble_output dset={dset_idx} after reshape (before skip): {torch.isnan(out).sum()} NaNs"

             internal_output_idx = self._internal_output_idx[dset_idx]

             if x_skip[dset_idx] is not None and len(internal_output_idx) > 0:
                  skip_reshaped = einops.rearrange(
                     x_skip[dset_idx],
                     "(batch ensemble) grid var -> batch ensemble grid var",
                     batch=batch_size,
                  ).to(dtype=dtype)

                  assert not torch.isnan(skip_reshaped).any(), f"NaN in _assemble_output dset={dset_idx} skip_reshaped: {torch.isnan(skip_reshaped).sum()} NaNs"

                  out[..., internal_output_idx] += skip_reshaped

                  assert not torch.isnan(out).any(), f"NaN in _assemble_output dset={dset_idx} after skip connection: {torch.isnan(out).sum()} NaNs"

             if len(self.boundings) > dset_idx:
                 for bounding in self.boundings[dset_idx]:
                     out = bounding(out)
                     assert not torch.isnan(out).any(), f"NaN in _assemble_output dset={dset_idx} after bounding {bounding.__class__.__name__}: {torch.isnan(out).sum()} NaNs"

             x_out[dset_idx] = out
        return x_out


    def forward(
        self,
        x: list[torch.Tensor],
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_shapes: Optional[list] = None,
        **kwargs,
    ) -> list[torch.Tensor]:
        if not isinstance(x, (list, tuple)):
            x = [x]

        batch_size, ensemble_size = x[0].shape[0], x[0].shape[2]
        bse = batch_size * ensemble_size
        in_out_sharded = grid_shard_shapes is not None

        assert not (
            in_out_sharded and model_comm_group is None
        ), "If input is sharded, model_comm_group must be provided."

        fcstep = min(1, fcstep)

        # Assemble Inputs
        # Note: assemble_input now returns x_obs_latent as list
        x_data_latent, x_obs_latent, x_skip, shard_shapes_data = self._assemble_input(
            x, fcstep, bse, grid_shard_shapes, model_comm_group
        )
        assert not torch.isnan(x_data_latent).any(), f"NaN after _assemble_input: x_data_latent has {torch.isnan(x_data_latent).sum()} NaNs"
        for _i, _obs in enumerate(x_obs_latent):
            assert not torch.isnan(_obs).any(), f"NaN after _assemble_input: x_obs_latent[{_i}] has {torch.isnan(_obs).sum()} NaNs"
        
        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=bse)
        shard_shapes_hidden = get_shard_shapes(x_hidden_latent, 0, model_comm_group)

        # 1. Encoder (Dataset 0)
        x_data_latent, x_latent = self._run_mapper(
            self.encoder,
            (x_data_latent, x_hidden_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_data, shard_shapes_hidden),
            model_comm_group=model_comm_group,
            x_src_is_sharded=in_out_sharded,
            x_dst_is_sharded=False,
            keep_x_dst_sharded=True,
        )

        assert not torch.isnan(x_data_latent).any(), f"NaN after encoder: x_data_latent has {torch.isnan(x_data_latent).sum()} NaNs"
        assert not torch.isnan(x_latent).any(), f"NaN after encoder: x_latent has {torch.isnan(x_latent).sum()} NaNs"

        # 2. Noise Injection (Ensemble specific)
        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            noise_ref=x_hidden_latent,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
        )

        assert not torch.isnan(x_latent_proc).any(), f"NaN after noise_injector: x_latent_proc has {torch.isnan(x_latent_proc).sum()} NaNs"
        if latent_noise is not None:
            assert not torch.isnan(latent_noise).any(), f"NaN after noise_injector: latent_noise has {torch.isnan(latent_noise).sum()} NaNs"

        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        # 3. Processor
        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=bse,
            shard_shapes=shard_shapes_hidden,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        assert not torch.isnan(x_latent_proc).any(), f"NaN after processor (before residual): x_latent_proc has {torch.isnan(x_latent_proc).sum()} NaNs"

        x_latent_proc = x_latent_proc + x_latent

        assert not torch.isnan(x_latent_proc).any(), f"NaN after processor+residual: x_latent_proc has {torch.isnan(x_latent_proc).sum()} NaNs"

        # 4. Decoders
        x_out = [None] * len(x)
        
        # Decoder 0
        x_out[0] = self._run_mapper(
            self.decoder,
            (x_latent_proc, x_data_latent),
            batch_size=bse,
            shard_shapes=(shard_shapes_hidden, shard_shapes_data),
            model_comm_group=model_comm_group,
            x_src_is_sharded=True, 
            x_dst_is_sharded=in_out_sharded,
            keep_x_dst_sharded=in_out_sharded,
        )

        assert not torch.isnan(x_out[0]).any(), f"NaN after decoder_0: x_out[0] has {torch.isnan(x_out[0]).sum()} NaNs"

        # Extra Decoders
        for i, decoder_extra in enumerate(self.decoders_extra):
            dset_idx = i + 1
            x_dst = x_obs_latent[i]
            
            if self.use_skip_connection_for_decoder1:
                 x_dst = torch.cat([x_dst, x_data_latent], dim=-1)
            
            x_latent_dset = x_latent_proc
            if self.processors_extra is not None:
                x_latent_dset = self.processors_extra[i](
                    x_latent_dset,
                    batch_size=bse,
                    shard_shapes=shard_shapes_hidden,
                    model_comm_group=model_comm_group,
                )
                x_latent_dset = x_latent_dset + x_latent_proc
            
            shard_shapes_dst = get_shard_shapes(x_dst, 0, model_comm_group)

            x_out[dset_idx] = self._run_mapper(
                decoder_extra,
                (x_latent_dset, x_dst),
                batch_size=bse,
                shard_shapes=(shard_shapes_hidden, shard_shapes_dst),
                model_comm_group=model_comm_group,
                x_src_is_sharded=True,
                x_dst_is_sharded=in_out_sharded,
                keep_x_dst_sharded=in_out_sharded,
            )

        for _i, _out in enumerate(x_out):
            assert not torch.isnan(_out).any(), f"NaN after all decoders: x_out[{_i}] has {torch.isnan(_out).sum()} NaNs"

        # 5. Assemble and Bound Outputs
        x_out = self._assemble_output(x_out, x_skip, batch_size, bse, x[0].dtype)

        for _i, _out in enumerate(x_out):
            assert not torch.isnan(_out).any(), f"NaN after _assemble_output: x_out[{_i}] has {torch.isnan(_out).sum()} NaNs"

        return x_out
