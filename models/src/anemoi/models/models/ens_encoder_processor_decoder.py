# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from typing import Callable
from typing import Optional

import einops
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch import Tensor
from torch.distributed.distributed_c10d import ProcessGroup
from torch_geometric.data import HeteroData

from anemoi.models.distributed.graph import shard_tensor
from anemoi.models.distributed.shapes import BipartiteGraphShardInfo
from anemoi.models.distributed.shapes import DatasetShardSizes
from anemoi.models.distributed.shapes import GraphShardInfo
from anemoi.models.distributed.shapes import ShardSizes
from anemoi.models.distributed.shapes import get_shard_sizes
from anemoi.models.layers.utils import maybe_checkpoint
from anemoi.models.models import AnemoiModelEncProcDec
from anemoi.utils.config import DotDict

LOGGER = logging.getLogger(__name__)


class InterpolationHeadProcessor(torch.nn.Module):
    """Optional per-dataset processor for a fraction-conditioned (interpolation) head.

    See ``anemoi.models.schemas.models.HeadProcessorSchema`` for the config surface.
    MUST be called once per target fraction, from inside the fraction loop
    (``AnemoiEnsModelEncProcDec._decode_target_times``), with that fraction as
    conditioning - never once before the loop, which would make it fraction-
    independent (parameters with no capability: it could not place a feature in
    time).

    Runs on the trunk's EXISTING hidden-to-hidden graph and edges (the same
    ``processor_graph_provider``, so no new graph or trainable edge parameters).
    ``latent_input`` selects which latent(s) the head reads (the frozen trunk's
    processed latent, the anchor encoder's output, or both) and which of them its
    output residual is taken on; ``warm_start_layers`` marks the head processor for
    copying the trunk processor's last/first layers into it at transfer-learning
    load time (identity ``in_proj`` so those layers see the latent they were
    trained on). The up-projection is always zero-initialised, so a freshly
    constructed head processor is a no-op regardless of fraction - training then
    learns a fraction-conditioned departure from its residual base.
    """

    def __init__(self, *, trunk_channels: int, edge_dim: int, config: dict) -> None:
        super().__init__()
        config = dict(config)

        # Consumed here, not constructor kwargs of the wrapped processor class.
        self.checkpoint_fraction_loop = bool(config.pop("checkpoint_fraction_loop", True))
        anchor_skip = bool(config.pop("anchor_skip", False))
        latent_input = str(config.pop("latent_input", "trunk"))
        if latent_input not in ("trunk", "anchor", "both"):
            raise ValueError(f"latent_input must be one of trunk/anchor/both, got {latent_input!r}")
        if anchor_skip:
            if latent_input == "anchor":
                raise ValueError("anchor_skip=True means latent_input='both'; it conflicts with latent_input='anchor'")
            latent_input = "both"
        self.latent_input = latent_input
        # Kept for old call sites / pickles: True iff the head reads BOTH latents.
        self.anchor_skip = latent_input == "both"
        self.warm_start_layers = config.pop("warm_start_layers", None)
        if self.warm_start_layers is not None and self.warm_start_layers not in ("last", "first"):
            raise ValueError(f"warm_start_layers must be 'last', 'first' or None, got {self.warm_start_layers!r}")
        num_channels = int(config["num_channels"])  # kept in config: instantiate() reads it too
        if self.warm_start_layers is not None and num_channels != trunk_channels:
            raise ValueError(
                f"warm_start_layers={self.warm_start_layers!r} copies trunk processor layers, so num_channels "
                f"({num_channels}) must equal the trunk's ({trunk_channels})"
            )

        # The fraction is passed as a single raw scalar in [0, 1] (the same
        # convention _decode_target_times already uses for the decoder's own
        # time_fraction feature), broadcast to every hidden node of a given
        # (batch, ensemble) row. condition_shape below must match this width.
        self.cond_dim = 1

        layer_kernels = dict(config.get("layer_kernels") or {})
        if "LayerNorm" not in layer_kernels:
            # Default: condition the processor's own layer norms on the fraction via
            # AdaLN-style modulation (the same `cond` mechanism the trunk's noise
            # injector uses on the shared processor - see ConditionalLayerNorm).
            # zero_init=True: the modulation starts at (scale=0, bias=0), so this
            # kernel choice does not by itself perturb a freshly built model (see
            # InterpolationHeadProcessor's own docstring on the up-projection for the
            # other half of that no-op-at-init guarantee).
            layer_kernels["LayerNorm"] = {
                "_target_": "anemoi.models.layers.normalization.ConditionalLayerNorm",
                "normalized_shape": num_channels,
                "condition_shape": self.cond_dim,
                "zero_init": True,
                "autocast": False,
            }
        config["layer_kernels"] = layer_kernels

        if self.latent_input in ("trunk", "both"):
            self.in_proj = torch.nn.Linear(trunk_channels, num_channels)
        if self.latent_input in ("anchor", "both"):
            self.skip_proj = torch.nn.Linear(trunk_channels, num_channels)
        self.out_proj = torch.nn.Linear(num_channels, trunk_channels)
        torch.nn.init.zeros_(self.out_proj.weight)
        torch.nn.init.zeros_(self.out_proj.bias)

        if self.warm_start_layers is not None:
            # The copied trunk layers were trained on the trunk latent: hand it to them
            # unchanged (identity in_proj). With both inputs the anchor branch starts silent
            # (zero skip_proj) and is learned; with the anchor alone it is the identity too.
            with torch.no_grad():
                if hasattr(self, "in_proj"):
                    self.in_proj.weight.copy_(torch.eye(num_channels))
                    self.in_proj.bias.zero_()
                if hasattr(self, "skip_proj"):
                    if self.latent_input == "both":
                        self.skip_proj.weight.zero_()
                    else:
                        self.skip_proj.weight.copy_(torch.eye(num_channels))
                    self.skip_proj.bias.zero_()

        self.processor = instantiate(config, _recursive_=False, edge_dim=edge_dim)

    @property
    def wants_anchor(self) -> bool:
        """True iff forward() needs x_anchor. Works on modules unpickled from checkpoints written
        before latent_input existed (they carry anchor_skip only, or neither attribute)."""
        latent_input = getattr(self, "latent_input", None)
        if latent_input is None:
            return bool(getattr(self, "anchor_skip", False))
        return latent_input != "trunk"

    def warm_start_key_map(self, trunk_num_layers: int) -> dict[str, str]:
        """Destination -> source module paths, relative to THIS module, for copying trunk
        processor layers into the head (empty unless warm_start_layers is set)."""
        if getattr(self, "warm_start_layers", None) is None:
            return {}
        num_layers = len(self.processor.proc)
        if num_layers > trunk_num_layers:
            raise ValueError(
                f"head processor has {num_layers} layers but the trunk only {trunk_num_layers}; cannot warm start"
            )
        offset = trunk_num_layers - num_layers if self.warm_start_layers == "last" else 0
        return {f"processor.proc.{i}": f"processor.proc.{offset + i}" for i in range(num_layers)}

    def _forward_impl(
        self,
        x_latent_proc: Tensor,
        fraction_col: Tensor,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Tensor,
        edge_index: Tensor,
        model_comm_group: Optional[ProcessGroup],
        x_anchor: Optional[Tensor] = None,
    ) -> Tensor:
        latent_input = getattr(self, "latent_input", "both" if getattr(self, "anchor_skip", False) else "trunk")
        if latent_input == "anchor":
            h = self.skip_proj(x_anchor)
            residual_base = x_anchor
        else:
            h = self.in_proj(x_latent_proc)
            if latent_input == "both":
                h = h + self.skip_proj(x_anchor)
            residual_base = x_latent_proc
        h = self.processor(
            x=h,
            batch_size=batch_size,
            shard_info=shard_info,
            edge_attr=edge_attr,
            edge_index=edge_index,
            model_comm_group=model_comm_group,
            cond=fraction_col,
        )
        return residual_base + self.out_proj(h)

    def forward(
        self,
        x_latent_proc: Tensor,
        fraction: Tensor | float,
        *,
        batch_size: int,
        shard_info: GraphShardInfo,
        edge_attr: Tensor,
        edge_index: Tensor,
        model_comm_group: Optional[ProcessGroup] = None,
        x_anchor: Optional[Tensor] = None,
    ) -> Tensor:
        if self.wants_anchor:
            if x_anchor is None:
                raise ValueError(
                    f"InterpolationHeadProcessor(latent_input={getattr(self, 'latent_input', 'both')!r}) requires x_anchor"
                )
        else:
            x_anchor = None  # never read: nothing to project it with
        fraction_col = torch.full(
            (x_latent_proc.shape[0], self.cond_dim),
            float(fraction),
            device=x_latent_proc.device,
            dtype=x_latent_proc.dtype,
        )
        return maybe_checkpoint(
            self._forward_impl,
            self.checkpoint_fraction_loop,
            x_latent_proc,
            fraction_col,
            batch_size,
            shard_info,
            edge_attr,
            edge_index,
            model_comm_group,
            x_anchor,
        )


class AnemoiEnsModelEncProcDec(AnemoiModelEncProcDec):
    """Message passing graph neural network with ensemble functionality."""

    def __init__(
        self,
        *,
        model_config: DictConfig,
        data_indices: dict,
        statistics: dict,
        graph_data: HeteroData,
        n_step_input: int,
        n_step_output: int,
    ) -> None:
        self.condition_on_residual = DotDict(model_config).model.condition_on_residual
        super().__init__(
            model_config=model_config,
            data_indices=data_indices,
            statistics=statistics,
            graph_data=graph_data,
            n_step_input=n_step_input,
            n_step_output=n_step_output,
        )

    def _build_networks(self, model_config: DotDict) -> None:
        super()._build_networks(model_config)

        self.noise_injector = instantiate(
            model_config.model.noise_injector,
            _recursive_=False,
            num_channels=self.num_channels,
            graph_data=self._graph_data,
            sparse_projector_num_chunks=model_config.model.get("sparse_projector", {}).get("num_chunks", 1),
        )

        # Optional per-dataset processor for fraction-conditioned (interpolation)
        # heads (see TargetForcingSchema.processor / InterpolationHeadProcessor).
        # Named distinctly from `processor` (the shared trunk processor above) so
        # that `training.submodules_to_freeze: [..., processor, ...]` cannot match
        # it by accident. Empty ModuleDict (nothing configured) is the default:
        # forward() then takes the exact code path it did before this option
        # existed.
        self.head_processor = torch.nn.ModuleDict()
        for dataset_name, target_forcing_config in self.target_forcing.items():
            processor_config = target_forcing_config.get("processor")
            if not processor_config:
                continue
            self.head_processor[dataset_name] = InterpolationHeadProcessor(
                trunk_channels=self.num_channels,
                edge_dim=self.processor_graph_provider.edge_dim,
                config=processor_config,
            )

        # Optional TRAINABLE anchor encoder per interpolation dataset (see
        # TargetForcingSchema.anchor_encoder): the trunk encoder of `source`, rebuilt with
        # the same constructor arguments over the same data->hidden graph provider, but
        # with its own (trainable) weights. Named `anchor_encoder`, outside every
        # submodules_to_freeze entry. Empty ModuleDict == the pre-existing forward() path.
        self.anchor_encoder = torch.nn.ModuleDict()
        self.anchor_encoder_source: dict[str, str] = {}
        self.anchor_encoder_warm_start: dict[str, bool] = {}
        for dataset_name, target_forcing_config in self.target_forcing.items():
            anchor_config = target_forcing_config.get("anchor_encoder")
            if not anchor_config:
                continue
            source = str(anchor_config.get("source", "data"))
            if source in self.encoders_to_skip or source not in self.encoder_graph_provider:
                raise ValueError(
                    f"anchor_encoder for '{dataset_name}': source '{source}' has no trunk encoder "
                    f"(encoders: {sorted(self.encoder_graph_provider.keys())}, skipped: {list(self.encoders_to_skip)})"
                )
            self.anchor_encoder_source[dataset_name] = source
            self.anchor_encoder_warm_start[dataset_name] = bool(anchor_config.get("warm_start", True))
            self.anchor_encoder[dataset_name] = instantiate(
                model_config.model.encoder,
                _recursive_=False,
                in_channels_src=self.input_dim[source],
                in_channels_dst=self.input_dim_latent,
                hidden_dim=self.num_channels,
                edge_dim=self.encoder_graph_provider[source].edge_dim,
            )

    def warm_start_key_map(self) -> dict[str, str]:
        """Destination -> source module paths (relative to this graph model) to copy at
        transfer-learning load time: warm-started head processor layers and anchor encoders.
        Consumed by anemoi.training.utils.checkpoint.apply_warm_start_map."""
        key_map: dict[str, str] = {}
        trunk_num_layers = len(self.processor.proc)
        for dataset_name, head_processor in (getattr(self, "head_processor", None) or {}).items():
            for dst, src in head_processor.warm_start_key_map(trunk_num_layers).items():
                key_map[f"head_processor.{dataset_name}.{dst}"] = src
        for dataset_name, source in (getattr(self, "anchor_encoder_source", None) or {}).items():
            if self.anchor_encoder_warm_start.get(dataset_name, False):
                key_map[f"anchor_encoder.{dataset_name}"] = f"encoder.{source}"
        return key_map

    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual and dataset_name not in self.residuals_to_skip:
            base_input_dim += self.num_input_channels_prognostic[dataset_name]
        return base_input_dim

    def _assemble_input(
        self,
        x: torch.Tensor,
        fcstep: int,
        batch_ens_size: int,
        grid_shard_sizes: DatasetShardSizes | None = None,
        model_comm_group: ProcessGroup | None = None,
        dataset_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, ShardSizes]:
        assert dataset_name is not None, "dataset_name must be provided when using multiple datasets."
        node_attributes_data = self.node_attributes(dataset_name, batch_size=batch_ens_size)
        grid_shard_sizes = grid_shard_sizes[dataset_name] if grid_shard_sizes is not None else None

        # TODO: return type hints incorrect for x_skip none
        if dataset_name in self.residuals_to_skip:
            x_skip = None
        else:
            x_skip = self.residual[dataset_name](
                x,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                n_step_output=self.n_step_output,
            )

        if grid_shard_sizes is not None:
            node_attributes_data = shard_tensor(node_attributes_data, 0, grid_shard_sizes, model_comm_group)

        # add data positional info (lat/lon)
        x_data_latent = torch.cat(
            (
                einops.rearrange(x, "batch time ensemble grid vars -> (batch ensemble grid) (time vars)"),
                node_attributes_data,
                torch.ones(batch_ens_size * x.shape[3], device=x.device).unsqueeze(-1) * fcstep,
            ),
            dim=-1,  # feature dimension
        )

        if self.condition_on_residual and x_skip is not None:
            x_skip_cond = x_skip[:, 0] if x_skip.ndim == 5 else x_skip
            x_data_latent = torch.cat(
                (
                    x_data_latent,
                    einops.rearrange(x_skip_cond, "bse grid vars -> (bse grid) vars"),
                ),
                dim=-1,
            )

        return x_data_latent, x_skip, grid_shard_sizes

    def _assemble_output(
        self,
        x_out: torch.Tensor,
        x_skip: torch.Tensor,
        batch_size: int,
        batch_ens_size: int,
        dtype: torch.dtype,
        dataset_name: str | None = None,
    ):
        ensemble_size = batch_ens_size // batch_size
        x_out = (
            einops.rearrange(
                x_out,
                "(bs e n) (time vars) -> bs time e n vars",
                bs=batch_size,
                e=ensemble_size,
                time=self.n_step_output,
            )
            .to(dtype=dtype)
            .clone()
        )

        # residual connection (just for the prognostic variables)
        assert dataset_name is not None, "dataset_name must be provided for multi-dataset case"
        if dataset_name not in self.residuals_to_skip:
            assert x_skip.ndim == 5, "Residual must be (batch, time, ensemble, grid, vars)."
            assert (
                x_skip.shape[1] == x_out.shape[1]
            ), f"Residual time dimension ({x_skip.shape[1]}) must match output time dimension ({x_out.shape[1]})."
            x_out[..., self._internal_output_idx[dataset_name]] += x_skip[..., self._internal_input_idx[dataset_name]]

        for bounding in self.boundings[dataset_name]:
            # bounding performed in the order specified in the config file
            x_out = bounding(x_out)
        return x_out

    def forward(
        self,
        x: dict[str, torch.Tensor],
        *,
        fcstep: int,
        model_comm_group: Optional[ProcessGroup] = None,
        grid_shard_sizes: DatasetShardSizes | None = None,
        target_times: dict[str, dict[str, Tensor]] | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Forward operator.

        Parameters
        ----------
        x : dict[str, torch.Tensor]
            Input tensor, shape (bs, m, e, n, f)
        fcstep : int
            Forecast step
        model_comm_group : ProcessGroup, optional
            Model communication group
        grid_shard_sizes : DatasetShardSizes, optional
            Per-dataset shard sizes for the grid dimension. ``None`` means the
            corresponding dataset is replicated, not sharded.
        target_times : dict[str, dict[str, Tensor]], optional
            Per-dataset conditioning for fraction-conditioned decoding, required
            for every dataset configured in ``model.target_forcing``. Entries are
            ``{"fractions": (T,), "forcings": (bs, T, ens, grid, F) | None}``;
            the dataset's decoder is invoked once per target time and the outputs
            are stacked along the time dimension.
        **kwargs
            Additional keyword arguments

        Returns
        -------
        dict[str, Tensor]
            Output tensor per dataset
        """
        dataset_names = list(x.keys())

        # Extract and validate batch & ensemble sizes across datasets
        batch_size = self._get_consistent_dim(x, 0)
        ensemble_size = self._get_consistent_dim(x, 2)

        batch_ens_size = batch_size * ensemble_size  # batch and ensemble dimensions are merged
        in_out_sharded = self._resolve_in_out_sharded(
            dataset_names=dataset_names,
            grid_shard_sizes=grid_shard_sizes,
        )
        for dataset_name in dataset_names:
            self._assert_valid_sharding(batch_size, ensemble_size, in_out_sharded[dataset_name], model_comm_group)

        fcstep = min(1, fcstep)
        # Process each dataset through its corresponding encoder
        dataset_latents = {}
        x_skip_dict = {}
        x_data_latent_dict = {}
        shard_sizes_data_dict = {}
        raw_x_data_latent_dict = {}  # features as they go INTO the trunk encoder
        encoder_edges_dict = {}  # (edge_attr, edge_index, shard_info) per encoded dataset

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_ens_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)
        for dataset_name in dataset_names:
            x_data_latent, x_skip, shard_sizes_data = self._assemble_input(
                x[dataset_name],
                fcstep=fcstep,
                batch_ens_size=batch_ens_size,
                grid_shard_sizes=grid_shard_sizes,
                model_comm_group=model_comm_group,
                dataset_name=dataset_name,
            )
            x_skip_dict[dataset_name] = x_skip
            shard_sizes_data_dict[dataset_name] = shard_sizes_data
            raw_x_data_latent_dict[dataset_name] = x_data_latent

            if dataset_name not in self.encoders_to_skip:
                (
                    encoder_edge_attr,
                    encoder_edge_index,
                    enc_edge_shard_sizes,
                ) = self.encoder_graph_provider[dataset_name].get_edges(
                    batch_size=batch_ens_size,
                    model_comm_group=model_comm_group,
                )

                enc_shard_info = BipartiteGraphShardInfo(
                    src_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                    dst_nodes=shard_sizes_hidden,
                    edges=enc_edge_shard_sizes,
                )
                encoder_edges_dict[dataset_name] = (encoder_edge_attr, encoder_edge_index, enc_shard_info)

                # Encoder for this dataset
                x_data_latent, x_latent = self.encoder[dataset_name](
                    (x_data_latent, x_hidden_latent),
                    batch_size=batch_ens_size,
                    shard_info=enc_shard_info,
                    edge_attr=encoder_edge_attr,
                    edge_index=encoder_edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
                )
                dataset_latents[dataset_name] = x_latent
            x_data_latent_dict[dataset_name] = x_data_latent

        # Combine all dataset latents
        x_latent = sum(dataset_latents.values())

        # Trainable anchor encoders (TargetForcingSchema.anchor_encoder). Same input tensor
        # and edges as the trunk encoder of `source`, own weights. getattr: models unpickled
        # from older checkpoints have no such attribute.
        x_anchor_dict = {}
        for dataset_name, source in (getattr(self, "anchor_encoder_source", None) or {}).items():
            encoder_edge_attr, encoder_edge_index, enc_shard_info = encoder_edges_dict[source]
            _, x_anchor_dict[dataset_name] = self.anchor_encoder[dataset_name](
                (raw_x_data_latent_dict[source], x_hidden_latent),
                batch_size=batch_ens_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,
            )

        x_latent_proc, latent_noise = self.noise_injector(
            x=x_latent,
            batch_size=batch_size,
            ensemble_size=ensemble_size,
            grid_size=self.node_attributes.num_nodes[self._graph_name_hidden],
            grid_shard_sizes=shard_sizes_hidden,
            model_comm_group=model_comm_group,
        )

        (
            processor_edge_attr,
            processor_edge_index,
            proc_edge_shard_sizes,
        ) = self.processor_graph_provider.get_edges(
            batch_size=batch_ens_size,
            model_comm_group=model_comm_group,
        )
        processor_kwargs = {"cond": latent_noise} if latent_noise is not None else {}

        # Processor
        x_latent_proc = self.processor(
            x=x_latent_proc,
            batch_size=batch_ens_size,
            shard_info=GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes),
            edge_attr=processor_edge_attr,
            edge_index=processor_edge_index,
            model_comm_group=model_comm_group,
            **processor_kwargs,
        )

        if self.latent_skip:
            x_latent_proc = x_latent_proc + x_latent

        proc_shard_info = GraphShardInfo(nodes=shard_sizes_hidden, edges=proc_edge_shard_sizes)

        # getattr, not self.head_processor: inference unpickles a saved model object
        # from a checkpoint rather than constructing one via __init__/_build_networks,
        # so a checkpoint written before this attribute existed has no
        # `head_processor` in its restored __dict__ at all. Falling back to `{}`
        # (never containing any dataset name) makes that case take the exact code
        # path forward() took before the head-processor option existed - the same
        # guarantee as a dataset simply having no `processor` configured.
        head_processor_dict = getattr(self, "head_processor", None) or {}

        x_out_dict = {}
        for dataset_name in dataset_names:
            # Compute decoder edges using updated latent representation
            (
                decoder_edge_attr,
                decoder_edge_index,
                dec_edge_shard_sizes,
            ) = self.decoder_graph_provider[dataset_name].get_edges(
                batch_size=batch_ens_size,
                model_comm_group=model_comm_group,
            )

            dec_shard_info = BipartiteGraphShardInfo(
                src_nodes=shard_sizes_hidden,
                dst_nodes=shard_sizes_data_dict[dataset_name],  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            def _decode(
                x_dst: torch.Tensor,
                x_src: torch.Tensor | None = None,
                dataset_name: str = dataset_name,
                edge_attr: Tensor = decoder_edge_attr,
                edge_index: Tensor = decoder_edge_index,
                shard_info: BipartiteGraphShardInfo = dec_shard_info,
            ) -> torch.Tensor:
                return self.decoder[dataset_name](
                    # noqa: B023 called within the same loop iteration
                    (x_latent_proc if x_src is None else x_src, x_dst),
                    batch_size=batch_ens_size,
                    shard_info=shard_info,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=in_out_sharded[dataset_name],  # keep x_out sharded iff in_out_sharded
                )

            if dataset_name in self.target_forcing:
                assert (
                    target_times is not None and dataset_name in target_times
                ), f"forward() requires target_times['{dataset_name}'] for fraction-conditioned dataset"

                head_process = None
                if dataset_name in head_processor_dict:

                    def head_process(
                        fraction: Tensor | float,
                        head_processor: torch.nn.Module = head_processor_dict[dataset_name],
                        x_latent_proc: torch.Tensor = x_latent_proc,  # noqa: B023
                        # the anchor encoder's output if one is configured for this dataset
                        # (TargetForcingSchema.anchor_encoder), else the frozen trunk encoder's
                        # own output - same layout as x_latent_proc either way. Only read when
                        # the head processor was built with wants_anchor (getattr: old pickles).
                        x_anchor: torch.Tensor = x_anchor_dict.get(dataset_name, x_latent),  # noqa: B023
                        proc_edge_attr: Tensor = processor_edge_attr,
                        proc_edge_index: Tensor = processor_edge_index,
                        proc_shard_info: GraphShardInfo = proc_shard_info,
                    ) -> torch.Tensor:
                        # See InterpolationHeadProcessor: this MUST be called once
                        # per fraction, from inside _decode_target_times's loop -
                        # that is the entire point of conditioning on `fraction`.
                        return head_processor(
                            x_latent_proc,
                            fraction,
                            batch_size=batch_ens_size,
                            shard_info=proc_shard_info,
                            edge_attr=proc_edge_attr,
                            edge_index=proc_edge_index,
                            model_comm_group=model_comm_group,
                            x_anchor=x_anchor if getattr(head_processor, "wants_anchor", False) else None,
                        )

                x_out_dict[dataset_name] = self._decode_target_times(
                    _decode,
                    x=x[dataset_name],
                    x_data_latent=x_data_latent_dict[dataset_name],
                    x_skip=x_skip_dict[dataset_name],
                    dataset_target_times=target_times[dataset_name],
                    batch_size=batch_size,
                    batch_ens_size=batch_ens_size,
                    ensemble_size=ensemble_size,
                    dataset_name=dataset_name,
                    head_process=head_process,
                )
            else:
                x_out = _decode(x_data_latent_dict[dataset_name])

                x_out_dict[dataset_name] = self._assemble_output(
                    x_out,
                    x_skip_dict[dataset_name],
                    batch_size,
                    batch_ens_size,
                    dtype=x[dataset_name].dtype,
                    dataset_name=dataset_name,
                )

        return x_out_dict

    def _decode_target_times(
        self,
        decode: callable,
        *,
        x: torch.Tensor,
        x_data_latent: torch.Tensor,
        x_skip: torch.Tensor | None,
        dataset_target_times: dict[str, Tensor],
        batch_size: int,
        batch_ens_size: int,
        ensemble_size: int,
        dataset_name: str,
        head_process: Callable[[Tensor | float], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Decode a fraction-conditioned dataset once per target time.

        Each decoder invocation receives the shared dst features extended with
        the target-time forcings and (optionally) the time-fraction scalar; the
        per-time outputs are stacked along the time dimension.

        Parameters
        ----------
        decode : callable
            Single-invocation decoder closure mapping dst features (and, when
            `head_process` is given, an alternate src latent `x_src`) to raw output.
        x : torch.Tensor
            The dataset's (normalized) input states, shape (bs, 2, ens, grid, vars).
        x_data_latent : torch.Tensor
            Assembled dst features shared by all target times, shape ((bs ens grid), D).
        x_skip : torch.Tensor | None
            Standard residual, or None if the dataset's residual is skipped.
        dataset_target_times : dict[str, Tensor]
            ``{"fractions": (T,), "forcings": (bs, T, ens, grid, F) | None}``.
        batch_size : int
            Batch size.
        batch_ens_size : int
            Merged batch and ensemble size.
        ensemble_size : int
            Ensemble size per device.
        dataset_name : str
            Dataset being decoded.
        head_process : callable, optional
            The dataset's optional head processor (see InterpolationHeadProcessor),
            bound to everything except the fraction. When given, called ONCE PER
            TARGET TIME with that time's fraction, and its result is used as the
            decoder's src latent for that time only - this is the entire point of
            the head processor (a fraction-independent call would add parameters
            with no capability). ``None`` (the default, and the only path taken
            when the dataset has no `processor` configured) preserves the exact
            behaviour this method had before the head-processor option existed.

        Returns
        -------
        torch.Tensor
            Stacked output, shape (bs, T, ens, grid, vars_out).
        """
        config = self.target_forcing[dataset_name]
        fractions = dataset_target_times["fractions"]
        forcings = dataset_target_times.get("forcings")

        expected_forcings = len(config.get("data", []))
        if expected_forcings:
            assert forcings is not None and forcings.shape[-1] == expected_forcings, (
                f"Dataset '{dataset_name}': expected {expected_forcings} target-time forcings "
                f"{config.get('data')}, got {None if forcings is None else forcings.shape[-1]}"
            )
            if forcings.shape[2] != ensemble_size:
                forcings = forcings.expand(-1, -1, ensemble_size, -1, -1)

        outputs = []
        for step in range(fractions.shape[0]):
            features = [x_data_latent]
            if expected_forcings:
                features.append(
                    einops.rearrange(forcings[:, step], "batch ensemble grid vars -> (batch ensemble grid) vars"),
                )
            n_time_noise = int(config.get("time_noise_channels", 0) or 0)
            if n_time_noise:
                # Drawn INSIDE the fraction loop: the trunk's noise injector fires once
                # per forward, so without this every target time of a given member
                # shares one perturbation and the member's trajectory is a smooth
                # deterministic function of the fraction alone. Shared across grid
                # points (repeat_interleave over the (batch ensemble grid) row order)
                # so it modulates the spatially structured latent instead of adding
                # spatially white noise.
                grid_points = x_data_latent.shape[0] // batch_ens_size
                noise = torch.randn(
                    batch_ens_size,
                    n_time_noise,
                    device=x_data_latent.device,
                    dtype=x_data_latent.dtype,
                )
                features.append(noise.repeat_interleave(grid_points, dim=0))
            if config.get("time_fraction", True):
                features.append(
                    torch.full(
                        (x_data_latent.shape[0], 1),
                        float(fractions[step]),
                        device=x_data_latent.device,
                        dtype=x_data_latent.dtype,
                    ),
                )

            if head_process is not None:
                # Applied HERE, inside the loop, once per fraction, with the
                # fraction as conditioning - see InterpolationHeadProcessor.
                x_out = decode(torch.cat(features, dim=-1), x_src=head_process(fractions[step]))
            else:
                x_out = decode(torch.cat(features, dim=-1))

            out_step = self._assemble_output(
                x_out,
                x_skip,
                batch_size,
                batch_ens_size,
                dtype=x.dtype,
                dataset_name=dataset_name,
            )

            # Anchor state added to the prognostic outputs (normalized space): the head
            # then learns the departure from it. Must not rebind x_skip: that is the
            # standard residual reused by every subsequent _assemble_output call.
            anchor = self.anchor_for_dataset(dataset_name, x, fractions[step], config)
            if anchor is not None:
                out_step[..., self._internal_output_idx[dataset_name]] += anchor

            outputs.append(out_step)

        return torch.cat(outputs, dim=1)

    ANCHOR_MODES = ("linear", "backward", "forward", "none")

    @classmethod
    def anchor_mode(cls, config: dict) -> str | dict[str, str]:
        """Resolve the anchor setting of a fraction-conditioned dataset.

        Returns one mode for all prognostic variables, or a ``{variable: mode}`` mapping
        (optionally with a ``default`` key) when the config gives one per variable.
        ``anchor`` takes precedence; the older boolean ``linear_residual`` maps to
        ``linear`` / ``none`` so existing configs and checkpoints keep their behaviour.
        """
        mode = config.get("anchor")
        if mode is None:
            mode = "linear" if config.get("linear_residual", False) else "none"
        if isinstance(mode, dict):
            bad = {k: v for k, v in mode.items() if v not in cls.ANCHOR_MODES}
            if bad:
                msg = f"Unknown anchor mode(s) {bad}; expected one of {cls.ANCHOR_MODES}"
                raise ValueError(msg)
            return dict(mode)
        if mode not in cls.ANCHOR_MODES:
            msg = f"Unknown anchor mode {mode!r}; expected one of {cls.ANCHOR_MODES}"
            raise ValueError(msg)
        return mode

    def anchor_for_dataset(
        self, dataset_name: str, x: torch.Tensor, fraction: torch.Tensor | float, config: dict
    ) -> torch.Tensor | None:
        """Anchor for the dataset's prognostic outputs, shape (bs, 1, ens, grid, n_prognostic).

        Uniform mode: the same anchor state for every prognostic variable. Per-variable
        mapping: each prognostic output variable gets its own mode (``default`` covers the
        rest; without ``default`` the fallback is the ``linear_residual`` behaviour), so
        e.g. 2t/msl can keep the linear-interpolation prior while cloud or wind anchor on
        x(t+6) or on nothing. Returns None when no variable is anchored.
        """
        mode = self.anchor_mode(config)
        in_idx = self._internal_input_idx[dataset_name]
        if isinstance(mode, str):
            if mode == "none":
                return None
            return self.anchor_state(x, fraction, mode)[..., in_idx]

        output_index = self.data_indices[dataset_name].model.output
        names = [output_index.full_index_to_name[int(i)] for i in self._internal_output_idx[dataset_name]]
        unknown = set(mode) - set(names) - {"default"}
        if unknown:
            msg = (
                f"anchor mapping for dataset {dataset_name!r} names variables that are not "
                f"prognostic outputs: {sorted(unknown)} (prognostic: {names})"
            )
            raise ValueError(msg)
        default = mode.get("default", "linear" if config.get("linear_residual", False) else "none")
        modes = [mode.get(name, default) for name in names]
        if all(m == "none" for m in modes):
            return None
        in_positions = [int(i) for i in in_idx]
        anchor = torch.zeros_like(x[:, 0:1][..., in_positions])
        for m in set(modes):
            if m == "none":
                continue
            sel = [k for k, mm in enumerate(modes) if mm == m]
            anchor[..., sel] = self.anchor_state(x, fraction, m)[..., [in_positions[k] for k in sel]]
        return anchor

    @staticmethod
    def anchor_state(x: torch.Tensor, fraction: torch.Tensor | float, mode: str) -> torch.Tensor:
        """State added to the decoder output at time fraction ``fraction`` of the window.

        x holds the two bracketing input states, shape (bs, 2, ens, grid, vars).

        linear   : (1-f) x(t0) + f x(t1)  - the residual is the departure from linear
                   interpolation (small and smooth for 2t / msl, a poor prior for cloud
                   or wind).
        backward : x(t1)                  - HourGlass's backward skip: residual from the
                   LATER analysis, exact at f=1.
        forward  : x(t0)                  - persistence from the earlier analysis, exact
                   at f=0.
        Returns shape (bs, 1, ens, grid, vars).
        """
        if mode == "linear":
            return (1 - fraction) * x[:, 0:1] + fraction * x[:, 1:2]
        if mode == "backward":
            return x[:, 1:2]
        if mode == "forward":
            return x[:, 0:1]
        msg = f"anchor_state called with mode {mode!r}"
        raise ValueError(msg)
