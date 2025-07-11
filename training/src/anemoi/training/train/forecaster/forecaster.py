# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Tuple, Union, Mapping, Dict

import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from timm.scheduler import CosineLRScheduler
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.utils.checkpoint import checkpoint

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.distributed.graph import gather_tensor
from anemoi.models.distributed.shapes import apply_shard_shapes
from anemoi.models.interface import AnemoiModelInterface
from anemoi.training.losses import get_loss_function
from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.loss import get_metric_ranges
from anemoi.training.losses.scaler_tensor import grad_scaler
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses.utils import print_variable_scaling
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.utils.enums import TensorDim

from anemoi.models.models.temporal_prognostic_decoder import (
    AnemoiTemporalPrognosticDecoder,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection


LOGGER = logging.getLogger(__name__)


class GraphForecaster(pl.LightningModule):
    """Graph neural network forecaster for PyTorch Lightning."""

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        truncation_data: dict,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: IndexCollection,
        metadata: dict,
        supporting_arrays: dict,
    ) -> None:
        """Initialize graph neural network forecaster.

        Parameters
        ----------
        config : DictConfig
            Job configuration
        graph_data : HeteroData
            Graph object
        statistics : dict
            Statistics of the training data
        data_indices : IndexCollection
            Indices of the training data,
        metadata : dict
            Provenance information
        supporting_arrays : dict
            Supporting NumPy arrays to store in the checkpoint

        """
        super().__init__()

        graph_data = graph_data.to(self.device)

        self.output_mask = instantiate(
            config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data
        )

        self.model = AnemoiModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            truncation_data=truncation_data,
            config=convert_to_omegaconf(config),
        )
        self.config = config
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = graph_data[config.graph.data].x
        self.statistics_tendencies = statistics_tendencies

        self.logger_enabled = (
            config.diagnostics.log.wandb.enabled
            or config.diagnostics.log.mlflow.enabled
        )

        # Instantiate all scalers with the training configuration
        self.scalers, self.delayed_scaler_builders = create_scalers(
            config.model_dump(by_alias=True).training.scalers,
            group_config=config.model_dump(by_alias=True).training.variable_groups,
            data_indices=data_indices,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            metadata_variables=metadata["dataset"].get("variables_metadata"),
            output_mask=self.output_mask,
        )

        self.val_metric_ranges = get_metric_ranges(
            config,
            data_indices,
            metadata["dataset"].get("variables_metadata"),
        )

        self.loss = get_loss_function(
            config.model_dump(by_alias=True).training.training_loss,
            scalers=self.scalers,
            data_indices=self.data_indices,
        )
        print_variable_scaling(self.loss, data_indices)

        self.metrics = torch.nn.ModuleDict(
            {
                metric_name: get_loss_function(
                    val_metric_config,
                    scalers=self.scalers,
                    data_indices=self.data_indices,
                )
                for metric_name, val_metric_config in config.model_dump(
                    by_alias=True,
                ).training.validation_metrics.items()
            },
        )

        if config.training.loss_gradient_scaling:
            self.loss.register_full_backward_hook(grad_scaler, prepend=False)

        self.is_first_step = True
        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_warmup = config.training.lr.warmup
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.optimizer_settings = config.training.optimizer

        self.model_comm_group = None
        self.reader_groups = None

        reader_group_size = self.config.dataloader.read_group_size
        self.grid_indices = instantiate(
            self.config.model_dump(by_alias=True).dataloader.grid_indices,
            reader_group_size=reader_group_size,
        )
        self.grid_indices.setup(graph_data)
        self.grid_dim = -2

        # check sharding support
        self.keep_batch_sharded = self.config.model.keep_batch_sharded
        read_group_supports_sharding = (
            reader_group_size == self.config.hardware.num_gpus_per_model
        )
        assert read_group_supports_sharding or not self.keep_batch_sharded, (
            f"Reader group size {reader_group_size} does not match the number of GPUs per model "
            f"{self.config.hardware.num_gpus_per_model}, but `model.keep_batch_sharded=True` was set. ",
            "Please set `model.keep_batch_sharded=False` or set `dataloader.read_group_size` ="
            "`hardware.num_gpus_per_model`.",
        )
        model_supports_sharding = getattr(
            self.model.model, "supports_sharded_input", False
        )
        assert model_supports_sharding or not self.keep_batch_sharded, (
            f"Model {self.model.model} does not support sharded inputs, but `model.keep_batch_sharded=True` was set. ",
            "Please set `model.keep_batch_sharded=False` or use a model that supports sharded inputs.",
        )
        # set flag if loss and metrics support sharding
        self.loss_supports_sharding = getattr(self.loss, "supports_sharding", False)
        self.metrics_support_sharding = all(
            getattr(metric, "supports_sharding", False)
            for metric in self.metrics.values()
        )

        if not self.loss_supports_sharding and self.keep_batch_sharded:
            LOGGER.warning(
                "Loss function %s does not support sharding. "
                "This may lead to increased memory usage and slower training.",
                self.loss.name,
            )
        if not self.metrics_support_sharding and self.keep_batch_sharded:
            LOGGER.warning(
                "Validation metrics %s do not support sharding. "
                "This may lead to increased memory usage and slower training.",
                ", ".join(self.metrics.keys()),
            )

        LOGGER.debug("Rollout window length: %d", self.rollout)
        LOGGER.debug("Rollout increase every : %d epochs", self.rollout_epoch_increment)
        LOGGER.debug("Rollout max : %d", self.rollout_max)
        LOGGER.debug("Multistep: %d", self.multi_step)

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_id = 0
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_size = 1

        self.reader_group_id = 0
        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.grid_shard_shapes = None
        self.grid_shard_slice = None

        # --- NEW CODE START: Instantiate additional decoders ---

        # Access the *actual* main model (AnemoiModelEncProcDec)
        # Assuming AnemoiModelInterface.model is the AnemoiModelEncProcDec instance
        self.main_core_model = self.model.model

        # Instantiate additional decoders from the 'additional_decoders' section in config
        self.additional_decoders = nn.ModuleDict()
        self.additional_losses = nn.ModuleDict()

        if self.config.model.get("additional_decoders"):
            # Get common parameters for all additional decoders
            latent_dim = self.config.model.main_anemoi_model.num_channels
            hidden_graph_name = self.config.graph.hidden
            data_graph_name = self.config.graph.input_nodes[0]
            # Edge attributes for graph mappers, assuming a common list in config or determined here
            # This needs to be passed to AnemoiTemporalPrognosticDecoder's __init__
            # Let's assume it's directly accessible via config.model.attributes.edges for simplicity.
            # You might need to refine this path based on your exact config structure.
            sub_graph_edge_attributes = self.config.model.attributes.edges

            for (
                decoder_name,
                decoder_config,
            ) in self.config.model.additional_decoders.items():
                LOGGER.info(f"Instantiating additional decoder: {decoder_name}")

                if decoder_name == "temporal_prognostic":
                    self.additional_decoders[
                        decoder_name
                    ] = AnemoiTemporalPrognosticDecoder(
                        latent_dim=latent_dim,
                        output_channels=decoder_config.output_channels,
                        hidden_graph_name=hidden_graph_name,
                        data_graph_name=data_graph_name,
                        graph_data=graph_data,
                        sub_graph_edge_attributes=sub_graph_edge_attributes,
                        cpu_offload=self.config.model.cpu_offload,  # Inherit from global model config
                        layer_kernels=self.config.model.layer_kernels,  # Inherit from global model config
                    )
                    # We also need a specific loss for this decoder. Add it to self.losses.
                    # We'll need a way to define its loss in config, let's assume 'temporal_prognostic_loss'
                    self.additional_losses[f"loss_{decoder_name}"] = get_loss_function(
                        decoder_config.loss,  # Assume config.model.additional_decoders.temporal_prognostic.loss
                        scalers=self.scalers,  # Reuse existing scalers if relevant
                        data_indices=self.data_indices,
                    )
                    LOGGER.info(
                        f"Temporal prognostic loss function: {self.loss_temporal_prognostic.name}"
                    )

                elif decoder_name.startswith("obsfuser_"):
                    # This will be for the modified ObsFuserDecoder
                    # For now, we'll just log a warning and skip instantiation.
                    LOGGER.warning(
                        f"ObsFuser decoder '{decoder_name}' instantiation is a placeholder for now. Skipping."
                    )
                    # self.additional_decoders[decoder_name] = instantiate(
                    #     decoder_config,
                    #     latent_dim=latent_dim,
                    #     data_indices=self.data_indices.get_collection(decoder_name),
                    #     graph_data=graph_data,
                    #     # ... other relevant ObsFuser parameters ...
                    # )
                    pass
                else:
                    LOGGER.error(f"Unknown additional decoder type: {decoder_name}")
                    raise ValueError(f"Unknown additional decoder type: {decoder_name}")

    def forward(
        self,
        batch_dict: Dict[str, torch.Tensor],
        return_latent_states: bool = False,
        **kwargs,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        main_x_input = batch_dict["main_x"]

        return self.model(
            main_x_input,
            model_comm_group=self.model_comm_group,
            grid_shard_shapes=self.grid_shard_shapes,
            return_latent_states=return_latent_states,
            **kwargs,
        )

    def on_load_checkpoint(self, checkpoint: torch.nn.module) -> None:
        self._ckpt_model_name_to_index = checkpoint["hyper_parameters"][
            "data_indices"
        ].name_to_index

    def define_delayed_scalers(self) -> None:
        """Update delayed scalers such as the loss weights mask for imputed variables."""
        for name, scaler_builder in self.delayed_scaler_builders.items():
            self.scalers[name] = scaler_builder.get_delayed_scaling(model=self.model)
            self.loss.update_scaler(scaler=self.scalers[name][1], name=name)

    def set_model_comm_group(
        self,
        model_comm_group: ProcessGroup,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        model_comm_group_size: int,
    ) -> None:
        self.model_comm_group = model_comm_group
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_size = model_comm_group_size

    def set_reader_groups(
        self,
        reader_groups: list[ProcessGroup],
        reader_group_id: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.reader_groups = reader_groups
        self.reader_group_id = reader_group_id
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

    def advance_input(
        self,
        x: torch.Tensor,
        y_pred: torch.Tensor,
        batch: torch.Tensor,
        rollout_step: int,
    ) -> torch.Tensor:
        x = x.roll(-1, dims=1)

        # Get prognostic variables
        x[:, -1, :, :, self.data_indices.model.input.prognostic] = y_pred[
            ...,
            self.data_indices.model.output.prognostic,
        ]

        x[:, -1] = self.output_mask.rollout_boundary(
            x[:, -1],
            batch[:, self.multi_step + rollout_step],
            self.data_indices,
        )

        # get new "constants" needed for time-varying fields
        x[:, -1, :, :, self.data_indices.model.input.forcing] = batch[
            :,
            self.multi_step + rollout_step,
            :,
            :,
            self.data_indices.data.input.forcing,
        ]
        return x

    def compute_loss_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> torch.Tensor:
        is_sharded = self.grid_shard_slice is not None

        sharding_supported = (self.loss_supports_sharding or not training_mode) and (
            self.metrics_support_sharding or not validation_mode
        )
        if (
            is_sharded and not sharding_supported
        ):  # gather tensors if loss or metrics do not support sharding
            shard_shapes = apply_shard_shapes(
                y_pred, self.grid_dim, self.grid_shard_shapes
            )
            y_pred_full = gather_tensor(
                torch.clone(y_pred), self.grid_dim, shard_shapes, self.model_comm_group
            )
            y_full = gather_tensor(
                torch.clone(y), self.grid_dim, shard_shapes, self.model_comm_group
            )
            grid_shard_slice = None
        else:
            y_pred_full, y_full = y_pred, y
            grid_shard_slice = self.grid_shard_slice

        loss = (
            self.loss(
                y_pred_full,
                y_full,
                grid_shard_slice=grid_shard_slice,
                group=self.model_comm_group,
            )
            if training_mode
            else None
        )

        metrics_next = {}
        if validation_mode:
            metrics_next = self.calculate_val_metrics(
                y_pred_full,
                y_full,
                rollout_step,
                grid_shard_slice=grid_shard_slice,
            )

        return loss, metrics_next

    def rollout_step(
        self,
        batch: torch.Tensor,
        rollout: int | None = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ) -> Generator[tuple[torch.Tensor | None, dict, list], None, None]:
        """Rollout step for the forecaster.

        Will run pre_processors on batch, but not post_processors on predictions.

        Parameters
        ----------
        batch : torch.Tensor
            Batch to use for rollout
        rollout : Optional[int], optional
            Number of times to rollout for, by default None
            If None, will use self.rollout
        training_mode : bool, optional
            Whether in training mode and to calculate the loss, by default True
            If False, loss will be None
        validation_mode : bool, optional
            Whether in validation mode, and to calculate validation metrics, by default False
            If False, metrics will be empty

        Yields
        ------
        Generator[tuple[Union[torch.Tensor, None], dict, list], None, None]
            Loss value, metrics, and predictions (per step)

        """
        batch = self.model.pre_processors(batch)  # normalized in-place

        # Delayed scalers need to be initialized after the pre-processors once
        if self.is_first_step:
            self.define_delayed_scalers()
            self.is_first_step = False

        # start rollout of preprocessed batch
        x = batch[
            :,
            0 : self.multi_step,
            ...,
            self.data_indices.data.input.full,
        ]  # (bs, multi_step, latlon, nvar)
        msg = (
            "Batch length not sufficient for requested multi_step length!"
            f", {batch.shape[1]} !>= {rollout + self.multi_step}"
        )
        assert batch.shape[1] >= rollout + self.multi_step, msg

        # -- NEW CODE --
        all_additional_predictions_per_rollout_step = {}

        for rollout_step in range(rollout or self.rollout):
            # prediction at rollout step rollout_step, shape = (bs, latlon, nvar)
            y_pred_main_model, x_latent_t0, x_latent_t6 = self(
                x, return_latent_states=True
            )

            (
                total_additional_loss,
                current_additional_predictions,
            ) = self._run_additional_decoders_and_compute_losses(
                x_latent_t0=x_latent_t0,
                x_latent_t6=x_latent_t6,
                batch=batch,
                rollout_step=rollout_step,
                training_mode=training_mode,
                validation_mode=validation_mode,
                model_comm_group=self.model_comm_group,
            )

            y_main_gt = batch[
                :,
                self.multi_step + rollout_step,
                ...,
                self.data_indices.data.output.full,
            ]
            # y includes the auxiliary variables, so we must leave those out when computing the loss
            main_loss, metrics_next = checkpoint(
                self.compute_loss_metrics,
                y_pred_main_model,
                y,
                rollout_step,
                training_mode,
                validation_mode,
                use_reentrant=False,
            )

            # Combine main loss and additional losses (only if in training mode)
            combined_loss = main_loss
            if training_mode:  # Only add additional loss if training
                combined_loss += total_additional_loss

            x = self.advance_input(x, y_pred, batch, rollout_step)

            yield combined_loss, metrics_next, y_pred_main_model, current_additional_predictions

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Assemble batch after transfer to GPU by gathering the batch shards if needed.

        Parameters
        ----------
        batch : Any (Dict[str, torch.Tensor] or torch.Tensor)
            Batch to transfer. Assumed to be a dictionary when using NativeGridMultiDataset.
        dataloader_idx : int
            Dataloader index (unused).

        Returns
        -------
        Any
            Batch after transfer to device (Dict[str, torch.Tensor] or torch.Tensor).
        """

        if isinstance(batch, dict):
            processed_batch = {}
            # Apply allgather to each tensor in the dictionary if not keeping batch sharded
            if not self.keep_batch_sharded and self.model_comm_group_size > 1:
                LOGGER.debug("Gathering batch tensors from dictionary batch.")
                for key, value in batch.items():
                    if isinstance(value, torch.Tensor):
                        # Assuming allgather_batch can handle individual tensors
                        processed_batch[key] = self.allgather_batch(value)
                    else:
                        processed_batch[key] = value  # Non-tensor items

                # If not keeping sharded, clear shard info
                self.grid_shard_shapes, self.grid_shard_slice = None, None
            else:
                # If keeping batch sharded (or not distributed), just pass the batch through.
                # All tensors in the dictionary should already be in their correct sharded state.
                processed_batch = batch

                # Set shard info based on the main data for other parts of the model that query it.
                # This needs to be done even if keeping sharded, for correct properties.
                if self.keep_batch_sharded and self.model_comm_group_size > 1:
                    # These properties are set by the dataloader for the *main grid points*
                    # and are retrieved from `self.grid_indices` which is setup based on the main grid.
                    self.grid_shard_shapes = self.grid_indices.shard_shapes
                    self.grid_shard_slice = self.grid_indices.get_shard_indices(
                        self.reader_group_rank
                    )
                else:
                    # If not sharded, ensure these properties are None
                    self.grid_shard_shapes, self.grid_shard_slice = None, None

            return processed_batch

        # Original logic still kept
        else:
            if self.keep_batch_sharded and self.model_comm_group_size > 1:
                self.grid_shard_shapes = self.grid_indices.shard_shapes
                self.grid_shard_slice = self.grid_indices.get_shard_indices(
                    self.reader_group_rank
                )
            else:
                batch = self.allgather_batch(batch)
            self.grid_shard_shapes, self.grid_shard_slice = None, None

        return batch

    def _step(
        self,
        batch: torch.Tensor,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> tuple[
        torch.Tensor,
        Mapping[str, torch.Tensor],
        list[torch.Tensor],
        Dict[str, torch.Tensor],
    ]:
        del batch_idx

        loss = torch.zeros(
            1, dtype=batch.dtype, device=self.device, requires_grad=False
        )
        main_metrics_per_rollout = {}
        main_y_preds_per_rollout = []

        # To store additional decoder predictions, structured by rollout step
        # { 'decoder_name_pred_type': [pred_step0, pred_step1, ...] }
        all_additional_preds_by_type = {}

        for (
            loss_next,
            metrics_next,
            y_preds_next,
            additional_preds_next,
        ) in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            loss += loss_next

            for mkey, mvalue in metrics_next.item():
                if mkey not in main_metrics_per_rollout:
                    main_metrics_per_rollout[mkey] = []
                main_metrics_per_rollout[mkey].append(mvalue)

            # Aggregate main model predictions
            main_y_preds_per_rollout.append(y_preds_next)

            # Aggregate additional decoder predictions
            for pred_key, pred_tensor in additional_preds_next.items():
                if pred_key not in all_additional_preds_by_type:
                    all_additional_preds_by_type[pred_key] = []
                all_additional_preds_by_type[pred_key].append(pred_tensor)

        loss *= 1.0 / self.rollout

        averaged_main_metrics = {
            mkey: torch.stack(mvalues).mean()
            for mkey, mvalues in main_metrics_per_rollout.items()
        }

        return (
            loss,
            averaged_main_metrics,
            main_y_preds_per_rollout,
            all_additional_preds_by_type,
        )

    def allgather_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """Allgather the batch-shards across the reader group.

        Parameters
        ----------
        batch : torch.Tensor
            Batch-shard of current reader rank

        Returns
        -------
        torch.Tensor
            Allgathered (full) batch
        """
        grid_shard_shapes = self.grid_indices.shard_shapes
        grid_size = self.grid_indices.grid_size

        if grid_size == batch.shape[self.grid_dim] or self.reader_group_size == 1:
            return batch  # already have the full grid

        shard_shapes = apply_shard_shapes(batch, self.grid_dim, grid_shard_shapes)
        tensor_list = [
            torch.empty(shard_shape, device=batch.device, dtype=batch.dtype)
            for shard_shape in shard_shapes
        ]

        torch.distributed.all_gather(
            tensor_list,
            batch,
            group=self.reader_groups[self.reader_group_id],
        )

        return torch.cat(tensor_list, dim=self.grid_dim)

    def calculate_val_metrics(
        self,
        y_pred: torch.Tensor,
        y: torch.Tensor,
        rollout_step: int,
        grid_shard_slice: slice | None = None,
    ) -> dict[str, torch.Tensor]:
        """Calculate metrics on the validation output.

        Parameters
        ----------
        y_pred: torch.Tensor
            Predicted ensemble
        y: torch.Tensor
            Ground truth (target).
        rollout_step: int
            Rollout step

        Returns
        -------
        val_metrics : dict[str, torch.Tensor]
            validation metrics and predictions
        """
        metrics = {}
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for metric_name, metric in self.metrics.items():
            if not isinstance(metric, BaseLoss):
                # If not a loss, we cannot feature scale, so call normally
                metrics[f"{metric_name}_metric/{rollout_step + 1}"] = metric(
                    y_pred_postprocessed, y_postprocessed
                )
                continue

            for mkey, indices in self.val_metric_ranges.items():
                metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                    exception_msg = (
                        "Validation metrics cannot be scaled over the variable dimension"
                        " in the post processed space."
                    )
                    raise ValueError(exception_msg)

                metrics[metric_step_name] = metric(
                    y_pred_postprocessed,
                    y_postprocessed,
                    scaler_indices=[..., indices],
                    grid_shard_slice=grid_shard_slice,
                    group=self.model_comm_group,
                )

        return metrics

    def training_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        (
            train_loss,
            main_metrics,
            _,
            additional_preds,
            individual_additional_losses,
        ) = self._step(batch, batch_idx, validation_mode=False)
        self.log(
            "train_" + self.loss.name + "_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch["main_x"].shape[0],
            sync_dist=True,
        )
        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

        # Log specific main model metrics (if any were returned by _step)
        for mkey, mvalue in main_metrics.items():
            self.log(
                "train_" + mkey,  # Example: train_MSELoss_metric/2t_1/1
                mvalue,
                on_epoch=True,
                on_step=True,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch["main_x"].shape[0],
                sync_dist=True,
            )

        if individual_additional_losses:
            for loss_key, loss_value in individual_additional_losses.items():
                self.log(
                    f"train_{loss_key}",
                    loss_value,
                    on_epoch=True,
                    on_step=True,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch["main_x"].shape[0],
                    sync_dist=True,
                )

        self.log(
            "rollout",
            float(self.rollout),
            on_step=True,
            logger=self.logger_enabled,
            rank_zero_only=True,
            sync_dist=False,
        )

        return train_loss

    def lr_scheduler_step(
        self, scheduler: CosineLRScheduler, metric: None = None
    ) -> None:
        """Step the learning rate scheduler by Pytorch Lightning.

        Parameters
        ----------
        scheduler : CosineLRScheduler
            Learning rate scheduler object.
        metric : Optional[Any]
            Metric object for e.g. ReduceLRonPlateau. Default is None.

        """
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if (
            self.rollout_epoch_increment > 0
            and self.current_epoch % self.rollout_epoch_increment == 0
        ):
            self.rollout += 1
            LOGGER.debug("Rollout window length: %d", self.rollout)
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        """Calculate the loss over a validation batch using the training loss function.

        Parameters
        ----------
        batch : torch.Tensor
            Validation batch
        batch_idx : int
            Batch inces

        """
        with torch.no_grad():
            (
                val_loss,
                metrics,
                y_preds,
                additional_preds,
                individual_additional_losses,
            ) = self._step(batch, batch_idx, validation_mode=True)

        self.log(
            "val_" + self.loss.name + "_loss",
            val_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch["main_x"].shape[0],
            sync_dist=True,
        )

        for mname, mvalue in metrics.items():
            self.log(
                "val_" + mname,
                mvalue,
                on_epoch=True,
                on_step=False,
                prog_bar=False,
                logger=self.logger_enabled,
                batch_size=batch["main_x"].shape[0],
                sync_dist=True,
            )

        if individual_additional_losses:
            for loss_key, loss_value in individual_additional_losses.items():
                self.log(
                    f"val_{loss_key}",
                    loss_value,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch["main_x"].shape[0],
                    sync_dist=True,
                )

        return val_loss, y_preds

    def configure_optimizers(self) -> Tuple[List[torch.optim.Optimizer], List[Dict]]:
        """Configure the optimizers and learning rate scheduler.

        Returns
        -------
        tuple[list[torch.optim.Optimizer], list[dict]]
            List of optimizers and list of dictionaries containing the
            learning rate scheduler
        """

        trainable_params: List[torch.nn.Parameter] = []

        if hasattr(self, "additional_decoders") and self.additional_decoders:
            for decoder_name, decoder_module in self.additional_decoders.items():
                params_count_for_decoder = 0
                for param in decoder_module.parameters():
                    if (
                        param.requires_grad
                    ):  # Only add parameters that are set to be trainable
                        trainable_params.append(param)
                        params_count_for_decoder += 1
                if params_count_for_decoder > 0:
                    LOGGER.info(
                        f"Collected {params_count_for_decoder} trainable parameters from '{decoder_name}'."
                    )
                else:
                    LOGGER.warning(
                        f"No trainable parameters found for '{decoder_name}'. Check module definition."
                    )
        else:
            LOGGER.warning(
                "No 'additional_decoders' module found or it is empty. This training run might have no trainable parameters."
            )

        if not trainable_params:
            LOGGER.warning(
                "No trainable parameters found for additional decoders. Skipping optimizer setup for them."
            )
            # If the main model is completely frozen, and no additional decoders are trainable,
            # then there are no parameters to optimize. This could lead to an error.
            # It's crucial that either main_core_model has some trainable parts OR additional_decoders exist.
            # Given your requirement to keep main model frozen, this is important.
            # If there are NO trainable parameters, PyTorch Lightning's Trainer will raise an error.
            # We must ensure `configure_optimizers` returns *something* that can be optimized
            # if we expect a training run.
            return [], []  # No optimizers or schedulers if nothing to train.

        if self.optimizer_settings.zero:
            optimizer = ZeroRedundancyOptimizer(
                trainable_params,
                lr=self.lr,
                optimizer_class=torch.optim.AdamW,
                **self.optimizer_settings.kwargs,
            )
        else:
            optimizer = torch.optim.AdamW(
                trainable_params,
                lr=self.lr,
                **self.optimizer_settings.kwargs,
            )

        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=self.lr_warmup,
        )

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _run_additional_decoders_and_compute_losses(
        self,
        x_latent_t0: torch.Tensor,
        x_latent_t6: torch.Tensor,
        batch: torch.Tensor,  # Full batch for ground truth
        rollout_step: int,
        training_mode: bool,
        validation_mode: bool,
        model_comm_group: Optional[ProcessGroup] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Runs additional decoders (temporal, diagnostic) and computes their losses.

        Parameters
        ----------
        x_latent_t0 : torch.Tensor
            Latent state from the main model's encoder (at t0).
        x_latent_t6 : torch.Tensor
            Latent state from the main model's processor (at t+6h).
        batch : torch.Tensor
            The full input batch, containing ground truth for all time steps.
        rollout_step : int
            The current step in the autoregressive rollout.
        training_mode : bool
            Whether currently in training mode (affects loss computation).
        validation_mode : bool
            Whether currently in validation mode (for metrics, though not used here directly yet).
        model_comm_group : Optional[ProcessGroup], optional
            Distributed communication group.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, torch.Tensor]]
            A tuple containing:
            - total_additional_loss: Sum of losses from all active additional decoders.
            - additional_predictions: Dictionary of predictions from additional decoders.
        """
        total_additional_loss = torch.zeros(
            1,
            dtype=batch.dtype,
            device=self.device,
            requires_grad=True if training_mode else False,
        )
        additional_predictions = {}

        # --- Temporal Prognostic Decoder (for 2t) ---
        if "temporal_prognostic" in self.additional_decoders:
            temporal_decoder = self.additional_decoders["temporal_prognostic"]
            loss_temporal_prognostic_fn = self.additional_losses[
                "loss_temporal_prognostic"
            ]

            temporal_decoder_config = (
                self.config.model.additional_decoders.temporal_prognostic
            )
            output_variables = temporal_decoder_config.output_variables
            num_intermediate_hours = temporal_decoder_config.num_intermediate_hours

            output_global_indices = [
                self.data_indices.name_to_index[var_name]
                for var_name in output_variables
            ]

            hourly_preds_list = []
            hourly_gts_list = []

            for i in range(1, num_intermediate_hours + 1):
                current_time_fraction = torch.full(
                    (
                        x_latent_t0.shape[0]
                        // self.main_core_model.node_attributes(
                            self.config.graph.hidden, batch_size=1
                        ).shape[0],
                        1,
                    ),  # Use x_latent_t0 for batch_size, infer ensemble from it
                    fill_value=float(i) / 6.0,
                    dtype=batch.dtype,  # Use batch dtype for consistency
                    device=self.device,
                )

                hourly_pred = temporal_decoder(
                    x_latent_t0=x_latent_t0,
                    x_latent_t6=x_latent_t6,
                    time_fraction=current_time_fraction,
                    model_comm_group=model_comm_group,
                )
                hourly_preds_list.append(hourly_pred)

                gt_time_idx = self.multi_step + (rollout_step * 6) + i

                if gt_time_idx >= batch.shape[1]:
                    LOGGER.warning(
                        f"Ground truth for time index {gt_time_idx} (rollout_step {rollout_step}, intermediate {i})"
                        f" is out of batch bounds ({batch.shape[1]}). Skipping loss for this intermediate step."
                    )
                    continue

                hourly_gt = batch[:, gt_time_idx, :, :, output_global_indices]
                hourly_gts_list.append(hourly_gt)

            if (
                hourly_preds_list and training_mode
            ):  # Only compute loss if predictions and in training mode
                hourly_preds_stacked = torch.stack(hourly_preds_list, dim=1)
                hourly_gts_stacked = torch.stack(hourly_gts_list, dim=1)

                additional_predictions[
                    "temporal_prognostic_hourly_pred"
                ] = hourly_preds_stacked

                loss_temporal = loss_temporal_prognostic_fn(
                    hourly_preds_stacked,
                    hourly_gts_stacked,
                    grid_shard_slice=self.grid_shard_slice
                    if self.loss_temporal_prognostic.supports_sharding
                    else None,
                    group=model_comm_group,
                )
                total_additional_loss += loss_temporal
                LOGGER.debug(
                    f"Temporal prognostic loss for step {rollout_step}: {loss_temporal.item()}"
                )
            elif (
                hourly_preds_list
            ):  # If not training mode, but predictions were made, still store them
                hourly_preds_stacked = torch.stack(hourly_preds_list, dim=1)
                additional_predictions[
                    "temporal_prognostic_hourly_pred"
                ] = hourly_preds_stacked

        # --- ObsFuser Decoder (placeholder for now) ---
        # if "obsfuser_diagnostic" in self.additional_decoders:
        #    ... (add logic for ObsFuser, compute its loss, add to total_additional_loss) ...

        return total_additional_loss, additional_predictions

    # --- NEW HELPER FUNCTION END ---
