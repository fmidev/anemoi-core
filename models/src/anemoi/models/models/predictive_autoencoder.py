# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""A single-snapshot graph codec with learned latent-time evolution."""

import logging

import einops
import torch
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import (
    BipartiteGraphShardInfo,
    DatasetShardSizes,
    GraphShardInfo,
    ShardSizes,
    get_shard_sizes,
)
from anemoi.models.layers.graph_provider import create_graph_provider
from anemoi.models.layers.processor import NoOpProcessor
from anemoi.models.models.autoencoder import AnemoiModelAutoEncoder

LOGGER = logging.getLogger(__name__)


class AnemoiModelPredictiveAutoEncoder(AnemoiModelAutoEncoder):
    """Encode snapshots independently and evolve a two-state persistent latent.

    ``latent_skip`` controls the transition residual: when enabled, the shared
    processor predicts a delta that is added to the latest persistent latent.
    It never creates a physical-space skip connection.
    """

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        n_step_input: int,
        n_step_output: int,
        graph_data: HeteroData,
    ) -> None:
        if n_step_output < 2:
            raise ValueError("Predictive autoencoding requires reconstruction plus at least one forecast output.")
        if n_step_input != n_step_output + 1:
            raise ValueError(
                "Predictive autoencoder input must contain two history snapshots and one forcing snapshot per "
                f"forecast step; got n_step_input={n_step_input}, n_step_output={n_step_output}."
            )

        model_settings = model_config.model.model
        self.expected_num_forcing_fields = model_settings.get("expected_num_forcing_fields")
        self.expected_num_prognostic_fields = model_settings.get("expected_num_prognostic_fields")
        self.require_bottleneck = model_settings.get("require_bottleneck", False)
        self.forecast_steps = n_step_output - 1

        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
            graph_data=graph_data,
        )

        self._validate_expected_channel_counts()
        self.latent_scalar_statistics = self._log_latent_scalar_statistics()

    def _calculate_input_dim(self, dataset_name: str) -> int:
        """Size one encoder for one physical snapshot, irrespective of rollout length."""
        return self.num_input_channels[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _calculate_target_dim(self, dataset_name: str) -> int:
        """Size one decoder target from one valid-time forcing snapshot."""
        return self.num_input_channels_decoding_forcings[dataset_name] + self.node_attributes.attr_ndims[dataset_name]

    def _calculate_output_dim(self, dataset_name: str) -> int:
        """Each decoder invocation emits exactly one physical snapshot."""
        return self.num_output_channels[dataset_name]

    def _build_networks(self, model_config: DotDict) -> None:
        """Build the shared codec, forcing mapper, temporal mixer, and transition processor."""
        super()._build_networks(model_config)
        if isinstance(self.processor, NoOpProcessor):
            raise TypeError("Predictive autoencoding requires a real latent transition processor, not NoOpProcessor.")

        forcing_encoder_config = model_config.model.get("forcing_encoder")
        if forcing_encoder_config is None:
            raise ValueError("Predictive autoencoder configuration must define model.forcing_encoder.")

        self.forcing_encoder_graph_provider = torch.nn.ModuleDict()
        self.forcing_encoder = torch.nn.ModuleDict()
        for dataset_name in self.dataset_names:
            self.forcing_encoder_graph_provider[dataset_name] = create_graph_provider(
                graph=self._graph_data[(dataset_name, "to", self._graph_name_hidden)],
                edge_attributes=forcing_encoder_config.get("sub_graph_edge_attributes"),
                src_size=self.node_attributes.num_nodes[dataset_name],
                dst_size=self.node_attributes.num_nodes[self._graph_name_hidden],
                trainable_size=forcing_encoder_config.get("trainable_size", 0),
            )
            forcing_input_dim = (
                self.num_input_channels_decoding_forcings[dataset_name]
                + self.node_attributes.attr_ndims[dataset_name]
            )
            self.forcing_encoder[dataset_name] = instantiate(
                forcing_encoder_config,
                _recursive_=False,
                in_channels_src=forcing_input_dim,
                in_channels_dst=self.input_dim_latent,
                hidden_dim=self.num_channels,
                edge_dim=self.forcing_encoder_graph_provider[dataset_name].edge_dim,
            )

        temporal_mixer_config = model_config.model.get("temporal_mixer")
        if temporal_mixer_config is None:
            raise ValueError("Predictive autoencoder configuration must define model.temporal_mixer.")
        self.temporal_mixer = instantiate(
            temporal_mixer_config,
            _recursive_=False,
            num_channels=self.num_channels,
        )

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
            raise ValueError(f"encode_snapshot expects one time step, got tensor shape {tuple(x.shape)}.")
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

    def _assemble_forcings(
        self,
        x: Tensor,
        batch_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Assemble forcing-only data-node context for one valid time."""
        if x.shape[1] != 1:
            raise ValueError(f"Valid-time forcing context expects one time step, got tensor shape {tuple(x.shape)}.")
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_target = self.node_attributes(dataset_name, batch_size=batch_size)
        dataset_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None
        if dataset_shard_sizes is not None:
            node_attributes_target = shard_tensor(node_attributes_target, 0, dataset_shard_sizes, model_comm_group)

        x_target_latent = torch.cat(
            (
                einops.rearrange(
                    x[..., self._decoding_forcing_input_idx[dataset_name]],
                    "batch time ensemble grid vars -> (batch ensemble grid) (time vars)",
                ),
                node_attributes_target,
            ),
            dim=-1,
        )
        return x_target_latent, dataset_shard_sizes

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
            raise IndexError(f"Time index {time_index} is outside tensor with {x.shape[1]} time steps.")
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
        dataset_latents = []
        for dataset_name in self.dataset_names:
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
            _, latent = self.encoder[dataset_name](
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
            dataset_latents.append(latent)

        latent = dataset_latents[0]
        for dataset_latent in dataset_latents[1:]:
            latent = latent + dataset_latent
        return latent, shard_sizes_hidden

    def encode_forcing_context(
        self,
        x: dict[str, Tensor],
        time_index: int,
        *,
        batch_size: int,
        model_comm_group: ProcessGroup | None = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
    ) -> tuple[Tensor, ShardSizes]:
        """Map forcing-only target-time context onto the hidden grid."""
        hidden, shard_sizes_hidden = self._initial_hidden_state(batch_size, model_comm_group)
        dataset_contexts = []
        for dataset_name in self.dataset_names:
            valid_time = self._time_slice(x[dataset_name], time_index)
            forcing_data, shard_sizes_data = self._assemble_forcings(
                valid_time,
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            edge_attr, edge_index, edge_shard_sizes = self.forcing_encoder_graph_provider[
                dataset_name
            ].get_edges(batch_size=batch_size, model_comm_group=model_comm_group)
            _, context = self.forcing_encoder[dataset_name](
                (forcing_data, hidden),
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
            dataset_contexts.append(context)

        context = dataset_contexts[0]
        for dataset_context in dataset_contexts[1:]:
            context = context + dataset_context
        return context, shard_sizes_hidden

    def transition_latent(
        self,
        previous: Tensor,
        current: Tensor,
        target_context: Tensor,
        *,
        batch_size: int,
        shard_sizes_hidden: ShardSizes,
        model_comm_group: ProcessGroup | None = None,
    ) -> Tensor:
        """Apply one shared, forcing-conditioned hidden-grid transition."""
        mixed = self.temporal_mixer(previous, current, target_context)
        edge_attr, edge_index, edge_shard_sizes = self.processor_graph_provider.get_edges(
            batch_size=batch_size,
            model_comm_group=model_comm_group,
        )
        transition_delta = self.processor(
            x=mixed,
            batch_size=batch_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=edge_shard_sizes),
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
        )
        return current + transition_delta if self.latent_skip else transition_delta

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
        for dataset_name in self.dataset_names:
            valid_time = self._time_slice(x[dataset_name], time_index)
            target_data, shard_sizes_target = self._assemble_forcings(
                valid_time,
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )
            edge_attr, edge_index, edge_shard_sizes = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_size,
                model_comm_group=model_comm_group,
            )
            x_out = self.decoder[dataset_name](
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

    def forward(
        self,
        x: dict[str, Tensor],
        *,
        model_comm_group: ProcessGroup | None = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Return reconstruction first, followed by increasing free forecast times."""
        del kwargs
        dataset_names = list(x.keys())
        if set(dataset_names) != set(self.dataset_names):
            raise ValueError(f"Expected datasets {self.dataset_names}, got {dataset_names}.")

        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)
        expected_input_steps = self.forecast_steps + 2
        for dataset_name in dataset_names:
            if x[dataset_name].shape[1] != expected_input_steps:
                raise ValueError(
                    f"Dataset '{dataset_name}' must provide {expected_input_steps} time steps "
                    f"[-timestep, 0, +timestep, ...], got {x[dataset_name].shape[1]}."
                )

        in_out_sharded = self._resolve_in_out_sharded(dataset_names, grid_shard_sizes)
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        previous, shard_sizes_hidden = self.encode_snapshot(
            x,
            0,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        current, _ = self.encode_snapshot(
            x,
            1,
            batch_size=batch_size,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )

        reconstruction = self.decode_snapshot(
            current,
            x,
            1,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            shard_sizes_hidden=shard_sizes_hidden,
            in_out_sharded=in_out_sharded,
            model_comm_group=model_comm_group,
            grid_shard_sizes=grid_shard_sizes,
        )
        outputs = {dataset_name: [reconstruction[dataset_name]] for dataset_name in dataset_names}

        for forecast_step in range(self.forecast_steps):
            target_time_index = forecast_step + 2
            target_context, _ = self.encode_forcing_context(
                x,
                target_time_index,
                batch_size=batch_size,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            )
            predicted = self.transition_latent(
                previous,
                current,
                target_context,
                batch_size=batch_size,
                shard_sizes_hidden=shard_sizes_hidden,
                model_comm_group=model_comm_group,
            )
            forecast = self.decode_snapshot(
                predicted,
                x,
                target_time_index,
                batch_size=batch_size,
                ensemble_size=ensemble_size,
                shard_sizes_hidden=shard_sizes_hidden,
                in_out_sharded=in_out_sharded,
                model_comm_group=model_comm_group,
                grid_shard_sizes=grid_shard_sizes,
            )
            for dataset_name in dataset_names:
                outputs[dataset_name].append(forecast[dataset_name])
            previous, current = current, predicted

        return {dataset_name: torch.cat(dataset_outputs, dim=1) for dataset_name, dataset_outputs in outputs.items()}

    def _validate_expected_channel_counts(self) -> None:
        for dataset_name in self.dataset_names:
            forcing_fields = self.num_input_channels_decoding_forcings[dataset_name]
            prognostic_fields = self.num_input_channels_prognostic[dataset_name]
            if self.expected_num_forcing_fields is not None and forcing_fields != self.expected_num_forcing_fields:
                raise ValueError(
                    f"Dataset '{dataset_name}' must contain exactly {self.expected_num_forcing_fields} forcing fields; "
                    f"found {forcing_fields}. Check data.datasets.{dataset_name}.forcing and dataloader.dataset.select."
                )
            if self.expected_num_prognostic_fields is not None and prognostic_fields != self.expected_num_prognostic_fields:
                raise ValueError(
                    f"Dataset '{dataset_name}' must contain exactly {self.expected_num_prognostic_fields} prognostic "
                    f"fields; found {prognostic_fields}. Check dataloader.dataset.select for missing or extra fields."
                )

    def _log_latent_scalar_statistics(self) -> dict[str, dict[str, int | float]]:
        hidden_nodes = self.node_attributes.num_nodes[self._graph_name_hidden]
        latent_scalars = hidden_nodes * self.num_channels
        statistics = {}
        for dataset_name in self.dataset_names:
            physical_nodes = self.node_attributes.num_nodes[dataset_name]
            prognostic_fields = self.num_input_channels_prognostic[dataset_name]
            physical_scalars = physical_nodes * prognostic_fields
            if physical_scalars == 0:
                raise ValueError(
                    f"Dataset '{dataset_name}' has no physical prognostic scalars; predictive autoencoding "
                    "requires at least one prognostic field and one data node."
                )
            ratio = latent_scalars / physical_scalars
            two_state_scalars = 2 * latent_scalars
            statistics[dataset_name] = {
                "physical_nodes": physical_nodes,
                "prognostic_fields": prognostic_fields,
                "hidden_nodes": hidden_nodes,
                "latent_channels": self.num_channels,
                "latent_scalars_per_snapshot": latent_scalars,
                "physical_prognostic_scalars_per_snapshot": physical_scalars,
                "ratio": ratio,
                "two_state_transition_scalars": two_state_scalars,
            }
            LOGGER.info(
                "Predictive autoencoder bottleneck [%s]: physical_nodes=%d, prognostic_fields=%d, "
                "hidden_nodes=%d, latent_channels=%d, latent_scalars_per_snapshot=%d, "
                "physical_prognostic_scalars_per_snapshot=%d, ratio=%.6f, two_state_transition_scalars=%d",
                dataset_name,
                physical_nodes,
                prognostic_fields,
                hidden_nodes,
                self.num_channels,
                latent_scalars,
                physical_scalars,
                ratio,
                two_state_scalars,
            )
            if self.require_bottleneck and ratio >= 1:
                raise ValueError(
                    f"Persistent latent is not a scalar bottleneck for dataset '{dataset_name}': ratio={ratio:.6f} "
                    f"({latent_scalars} latent scalars >= {physical_scalars} physical prognostic scalars). "
                    "Reduce model.num_channels or graph.nodes.hidden.node_builder.resolution."
                )
        return statistics
