# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Graph-based forecasting with autoregressive latent-state dynamics."""

from functools import cached_property

import einops
import torch
from hydra.utils import instantiate
from torch import Tensor
from torch import nn
from torch.distributed import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.target_features import InputForcingsFeature
from anemoi.models.layers.utils import load_layer_kernels
from anemoi.models.models.encoder_processor_decoder import AnemoiModelEncProcDec
from anemoi.utils.dates import frequency_to_timedelta


class _SnapshotForcingsFeature(InputForcingsFeature):
    """Prescribed forcing features for a single decoder valid time."""

    @cached_property
    def dim(self) -> int:
        return self.model.num_input_channels_forcings[self.datasets_names[0]]


class AnemoiModelLatentForecaster(AnemoiModelEncProcDec):
    """Encoder-processor-decoder model with autoregressive latent dynamics.

    Parameters
    ----------
    model_config
        Model configuration, including latent-state and forcing settings.
    **kwargs
        Arguments passed to ``AnemoiModelEncProcDec``.
    """

    def __init__(self, *, model_config, **kwargs):
        self.options = model_config.latent
        self.num_channels = model_config.processor.num_channels
        self.step = frequency_to_timedelta(self.options.timestep)
        self.decode_steps = tuple(self.options.decode_steps)
        super().__init__(model_config=model_config, **kwargs)
        if len(self.dataset_names) != 1:
            message = "Use one latent forecaster per domain; compose domains through latent messages."
            raise ValueError(message)
        self.state_norm = (
            nn.LayerNorm(self.num_channels, elementwise_affine=False) if self.options.normalize_state else nn.Identity()
        )

    def _build_decoder_routing(self, decoders_config):
        super()._build_decoder_routing(decoders_config)
        for target in self.decoders_target_input.values():
            target.features = [
                (
                    _SnapshotForcingsFeature(self, feature.datasets_names)
                    if isinstance(feature, InputForcingsFeature)
                    else feature
                )
                for feature in target.features
            ]

    def _build_networks(self, model_config):
        for decoder in model_config.decoders.values():
            if {"encoded_data", "prognostics"}.intersection(decoder.target_node_features):
                message = "Latent decoding must not bypass the atmospheric latent state."
                raise ValueError(message)
        width = sum(
            int(template.out_features)
            for fields, template in [
                (self.options.static_forcings, self.options.static_embedding),
                (self.options.temporal_forcings, self.options.temporal_embedding),
            ]
            if fields
        )
        norm = model_config.processor.layer_kernels.LayerNorm
        if width and (not norm._target_.endswith("ConditionalLayerNorm") or int(norm.condition_shape) != width):
            message = "Processor conditional-normalisation width must equal the configured forcing embeddings."
            raise ValueError(message)
        super()._build_networks(model_config)
        self.forcing_projector = instantiate(self.options.forcing_projector)
        self.static_embeddings = nn.ModuleDict()
        self.temporal_embeddings = nn.ModuleDict()
        for name in self.input_datasets:
            static, temporal = list(self.options.static_forcings), list(self.options.temporal_forcings)
            if set(static) & set(temporal) or not set(static + temporal) <= set(self.data_indices[name].forcing):
                message = "Static/temporal forcing selections must be disjoint subsets of configured forcings."
                raise ValueError(message)
            for fields, template, destination in [
                (static, self.options.static_embedding, self.static_embeddings),
                (temporal, self.options.temporal_embedding, self.temporal_embeddings),
            ]:
                if fields:
                    destination[name] = instantiate(
                        template,
                        in_features=len(fields),
                        layer_kernels=load_layer_kernels(template.layer_kernels),
                        _recursive_=False,
                    )
            # Inverse chord-distance weights on the unit sphere.
            edges = self._graph_data[(name, "to", self._graph_name_hidden)].edge_index
            src, dst = edges
            a = self._graph_data[name].x[src, :2].float()
            b = self._graph_data[self._graph_name_hidden].x[dst, :2].float()
            chord = (
                torch.sin((a[:, 0] - b[:, 0]) / 2) ** 2
                + torch.cos(a[:, 0]) * torch.cos(b[:, 0]) * torch.sin((a[:, 1] - b[:, 1]) / 2) ** 2
            )
            weights = 1 / chord.clamp_min(1e-10).sqrt()
            count = self.node_attributes.num_nodes[self._graph_name_hidden]
            total = weights.new_zeros(count).index_add_(0, dst, weights)
            if (total == 0).any():
                message = "Forcing projection has isolated hidden nodes."
                raise ValueError(message)
            matrix = torch.sparse_coo_tensor(
                torch.stack((dst, src)), weights / total[dst], (count, self.node_attributes.num_nodes[name])
            ).coalesce()
            self.register_buffer("_forcing_map", matrix, persistent=False)

    def project_forcings(self, x, fields, *, batch_size, model_comm_group=None, grid_shard_sizes=None):
        """Project prescribed fields to hidden nodes using inverse-distance weights."""
        name = self.input_datasets[0]
        columns = [self.data_indices[name].model.input.name_to_index[f] for f in fields]
        value = x[name][:, 0, ..., columns]
        if grid_shard_sizes is not None and grid_shard_sizes.get(name) is not None:
            value = gather_tensor(value, 2, grid_shard_sizes[name], model_comm_group)
        result = self.forcing_projector(value.float(), self._forcing_map)
        coords = self._graph_data[self._graph_name_hidden].x
        coordinate = {
            "cos_latitude": coords[:, 0].cos(),
            "sin_latitude": coords[:, 0].sin(),
            "cos_longitude": coords[:, 1].cos(),
            "sin_longitude": coords[:, 1].sin(),
        }
        for j, field in enumerate(fields):
            if field in coordinate:
                result[..., j] = coordinate[field]
        result = result.reshape(-1, len(fields))
        return shard_tensor(result, 0, get_shard_sizes(result, 0, model_comm_group), model_comm_group)

    def static_condition(self, x, *, batch_size, model_comm_group=None, grid_shard_sizes=None):
        if not self.options.static_forcings:
            return None
        value = self.project_forcings(
            x,
            list(self.options.static_forcings),
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        return self.static_embeddings[self.input_datasets[0]](value)

    def condition(self, x, time_index, *, batch_size, static=None, model_comm_group=None, grid_shard_sizes=None):
        parts = [] if static is None else [static]
        if self.options.temporal_forcings:
            snapshot = {name: value[:, time_index : time_index + 1] for name, value in x.items()}
            value = self.project_forcings(
                snapshot,
                list(self.options.temporal_forcings),
                batch_size=batch_size,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            )
            parts.append(self.temporal_embeddings[self.input_datasets[0]](value))
        return torch.cat(parts, dim=-1) if parts else None

    def advance(
        self, current, condition, *, batch_size, shard_sizes_hidden, model_comm_group=None, message=None, step=None
    ):
        """Advance the latent state by one trained timestep.

        Parameters
        ----------
        current : Tensor
            Latent node features, shaped ``(local_hidden_nodes, channels)``.
        condition : Tensor or None
            Processor conditioning features at the target valid time.
        batch_size : int
            Number of samples in the model-parallel group.
        shard_sizes_hidden
            Hidden-node shard sizes for the model-parallel group.
        model_comm_group : ProcessGroup, optional
            Model communication group.
        message : Tensor, optional
            Additive processor input with the same shape as ``current``.
        step : str or timedelta, optional
            Requested interval, which must equal the configured timestep.

        Returns
        -------
        Tensor
            Latent node features at the next timestep.
        """
        if step is not None and frequency_to_timedelta(step) != self.step:
            message = "Requested interval differs from the trained processor interval."
            raise ValueError(message)
        if message is not None and message.shape != current.shape:
            message = "A message must match the receiving latent node/channel layout."
            raise ValueError(message)
        source = current if message is None else current + message
        edge_attr, edge_index, sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_size, model_comm_group=model_comm_group
        )
        kwargs = {} if condition is None else {"cond": condition}
        delta = self.processor(
            source,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            **kwargs,
        )
        return self.state_norm(current + delta if self.latent_skip else delta)

    def forward(self, x, *, model_comm_group=None, grid_shard_sizes=None, **kwargs):
        """Return decoded physical states at the configured lead times."""
        del kwargs
        batch = self._get_consistent_dim(x, 0)
        ensemble = self._get_consistent_dim(x, 2)
        steps = self._get_consistent_dim(x, 1) - 1
        if not self.decode_steps or max(self.decode_steps) > steps:
            message = "Input forcing times do not cover requested decoded steps."
            raise ValueError(message)
        io = self._resolve_in_out_sharded(self.dataset_names, grid_shard_sizes)
        args = {"batch_size": batch, "model_comm_group": model_comm_group, "grid_shard_sizes": grid_shard_sizes}
        current, sizes = self.encode_snapshot(x, 0, **args)
        static = self.static_condition(x, **args) if steps else None
        outputs = {name: [] for name in self.target_datasets}
        for k in range(steps + 1):
            if k:
                cond = self.condition(x, k, static=static, **args)
                current = self.advance(
                    current, cond, batch_size=batch, shard_sizes_hidden=sizes, model_comm_group=model_comm_group
                )
            if k in self.decode_steps:
                decoded = self.decode_snapshot(
                    current, x, k, ensemble_size=ensemble, shard_sizes_hidden=sizes, in_out_sharded=io, **args
                )
                for name, value in decoded.items():
                    outputs[name].append(value)
        return {name: torch.cat(values, dim=1) for name, values in outputs.items()}

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Size one encoder for one physical snapshot, irrespective of rollout length."""
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _calculate_output_dim(self, dataset_name: str) -> int:
        """Each decoder invocation emits exactly one physical snapshot."""
        return self.num_output_channels[dataset_name]

    def _assemble_input(
        self,
        x: Tensor,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Assemble exactly one physical snapshot for the shared encoder."""
        if x.shape[1] != 1:
            message = f"encode_snapshot expects one time step, got tensor shape {tuple(x.shape)}."
            raise ValueError(message)
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_size)
        dataset_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
        if dataset_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, dataset_shard_sizes, model_comm_group)

        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
            ),
            dim=-1,
        )
        return x_data_latent, dataset_shard_sizes

    def _assemble_output(
        self,
        x_out: Tensor,
        batch_size: int,
        ensemble_size: int,
        dtype: torch.dtype,
        dataset_name: str | None = None,
    ) -> Tensor:
        """Assemble and bound one decoder output snapshot."""
        x_out = (
            einops.rearrange(
                x_out,
                "(batch ensemble grid) vars -> batch ensemble grid vars",
                batch=batch_size,
                ensemble=ensemble_size,
            )
            .unsqueeze(1)
            .to(dtype=dtype)
            .clone()
        )
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        for bounding in self.boundings[dataset_name]:
            x_out = bounding(x_out)
        return x_out

    @staticmethod
    def _time_slice(x: Tensor, time_index: int) -> Tensor:
        """Select one non-negative time index without dropping the time dimension."""
        if time_index < 0 or time_index >= x.shape[1]:
            message = f"Time index {time_index} is outside tensor with {x.shape[1]} time steps."
            raise IndexError(message)
        return x[:, time_index : time_index + 1]

    def _initial_hidden_state(
        self,
        batch_size: int,
        model_comm_group: ProcessGroup | None,
    ) -> tuple[Tensor, ShardSizes]:
        hidden = self.node_attributes(self._graph_name_hidden, batch_size=batch_size)
        shard_sizes_hidden = get_shard_sizes(hidden, 0, model_comm_group)
        return shard_tensor(hidden, 0, shard_sizes_hidden, model_comm_group), shard_sizes_hidden

    def encode_snapshot(
        self,
        x: dict[str, Tensor],
        time_index: int,
        *,
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Encode one physical snapshot with the shared data-to-hidden mapper."""
        hidden, shard_sizes_hidden = self._initial_hidden_state(batch_size, model_comm_group)
        dataset_latents = {}
        for dataset_name in self.input_datasets:
            snapshot = self._time_slice(x[dataset_name], time_index)
            x_data_latent, shard_sizes_data = self._assemble_input(
                snapshot,
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            edge_attr, edge_index, edge_shard_sizes = self.encoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            _, latent = self.encoder[self.dataset2encoder[dataset_name]](
                (x_data_latent, hidden),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_data,
                    dst_nodes=shard_sizes_hidden,
                    edges=edge_shard_sizes,
                ),
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )
            dataset_latents[dataset_name] = latent

        return self.state_norm(self.latent_aggregator(hidden, dataset_latents)), shard_sizes_hidden

    def decode_snapshot(
        self,
        latent: Tensor,
        x: dict[str, Tensor],
        time_index: int,
        *,
        batch_size: int,
        ensemble_size: int,
        shard_sizes_hidden: ShardSizes,
        in_out_sharded: dict[str, bool],
        model_comm_group: ProcessGroup | None = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
    ) -> dict[str, Tensor]:
        """Decode one latent using forcing-only context at its valid time."""
        outputs = {}
        for dataset_name in self.target_datasets:
            valid_time = self._time_slice(x[dataset_name], time_index)
            target_data, shard_sizes_target = self._assemble_targets(
                valid_time,
                None,
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            edge_attr, edge_index, edge_shard_sizes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            x_out = self.decoder[self.dataset2decoder[dataset_name]](
                (latent, target_data),
                batch_size=batch_size,
                shard_info=BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_hidden,
                    dst_nodes=shard_sizes_target,
                    edges=edge_shard_sizes,
                ),
                edge_attr=edge_attr,
                edge_index=edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=in_out_sharded[dataset_name],
            )
            outputs[dataset_name] = self._assemble_output(
                x_out,
                batch_size,
                ensemble_size,
                x[dataset_name].dtype,
                dataset_name,
            )
        return outputs
