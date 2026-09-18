# (C) Copyright 2024-2026 Anemoi contributors.
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
            model_config.noise_injector,
            _recursive_=False,
            graph_data=self._graph_data,
            sparse_projector_num_chunks=model_config.get("sparse_projector", {}).get("num_chunks", 1),
        )


    def _calculate_input_dim(self, dataset_name: str) -> int:
        base_input_dim = super()._calculate_input_dim(dataset_name)
        base_input_dim += 1  # for forecast step (fcstep)
        if self.condition_on_residual:
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

        if self.condition_on_residual:
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
        x_skip: torch.Tensor | None,
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
        if x_skip is not None:
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

        x_hidden_latent = self.node_attributes(self._graph_name_hidden, batch_size=batch_ens_size)
        shard_sizes_hidden = get_shard_sizes(x_hidden_latent, 0, model_comm_group)
        x_hidden_latent = shard_tensor(x_hidden_latent, 0, shard_sizes_hidden, model_comm_group)
        for dataset_name in x.keys():
            if dataset_name not in self.input_datasets:
                continue

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

            # Encoder for this dataset
            encoder_name = self.dataset2encoder[dataset_name]
            x_data_latent, x_latent = self.encoder[encoder_name](
                (x_data_latent, x_hidden_latent),
                batch_size=batch_ens_size,
                shard_info=enc_shard_info,
                edge_attr=encoder_edge_attr,
                edge_index=encoder_edge_index,
                model_comm_group=model_comm_group,
                keep_x_dst_sharded=True,  # always keep x_latent sharded for the processor
            )
            x_data_latent_dict[dataset_name] = x_data_latent
            dataset_latents[dataset_name] = x_latent

        # Combine all dataset latents
        x_latent = self.latent_aggregator(x_hidden_latent, dataset_latents)

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

        x_out_dict = {}
        for dataset_name in self.target_datasets:
            x_target_latent, shard_sizes_target = self._assemble_targets(
                x[dataset_name],
                x_data_latent_dict.get(dataset_name, None),
                batch_size,
                grid_shard_sizes,
                model_comm_group,
                dataset_name,
            )

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
                dst_nodes=shard_sizes_target,  # None if not sharded
                edges=dec_edge_shard_sizes,
            )

            decoder_name = self.dataset2decoder[dataset_name]

            def _decode(
                x_dst: torch.Tensor,
                x_src: torch.Tensor | None = None,
                decoder_name: str = decoder_name,
                edge_attr: Tensor = decoder_edge_attr,
                edge_index: Tensor = decoder_edge_index,
                shard_info: BipartiteGraphShardInfo = dec_shard_info,
            ) -> torch.Tensor:
                return self.decoder[decoder_name](
                    (x_latent_proc if x_src is None else x_src, x_dst),
                    batch_size=batch_ens_size,
                    shard_info=shard_info,
                    edge_attr=edge_attr,
                    edge_index=edge_index,
                    model_comm_group=model_comm_group,
                    keep_x_dst_sharded=in_out_sharded[dataset_name],
                )

            if dataset_name in self.target_forcing:
                assert (
                    target_times is not None and dataset_name in target_times
                ), f"forward() requires target_times['{dataset_name}'] for fraction-conditioned dataset"

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
    ) -> torch.Tensor:
        """Decode a fraction-conditioned dataset once per target time.

        Each decoder invocation receives the shared dst features extended with
        the target-time forcings and (optionally) the time-fraction scalar; the
        per-time outputs are stacked along the time dimension.

        Parameters
        ----------
        decode : callable
            Single-invocation decoder closure mapping dst features to raw output.
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
