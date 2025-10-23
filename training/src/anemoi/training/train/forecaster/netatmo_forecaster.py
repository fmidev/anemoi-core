import logging
import math
import os
from typing import Optional
from typing import TYPE_CHECKING

import pytorch_lightning as pl
import torch
from anemoi.models.interface import FuserModelInterface
#    from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.config import DotDict
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch_geometric.data import HeteroData
from timm.scheduler import CosineLRScheduler
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.zip import ZipLoss
#from anemoi.training.train.forecaster import GraphForecaster
from anemoi.training.losses.base import BaseLoss
from anemoi.training.utils.jsonify import map_config_to_primitives
from anemoi.training.losses.scalers import create_scalers
from anemoi.training.losses import get_loss_function
from anemoi.training.utils.enums import TensorDim
from anemoi.training.schemas.base_schema import convert_to_omegaconf
from anemoi.training.schemas.base_schema import BaseSchema

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Generator
    from collections.abc import Mapping

    from torch.distributed.distributed_c10d import ProcessGroup
    from torch_geometric.data import HeteroData

    from anemoi.models.data_indices.collection import IndexCollection

class NetatmoGraphForecaster(pl.LightningModule):

    def __init__(
        self,
        *,
        config: BaseSchema,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: tuple,
        metadata: dict,
        supporting_arrays: dict | None = None,
        truncation_data: dict | None = None,
    ) -> None:
        super().__init__()

        graph_data = graph_data.to(self.device)

        assert isinstance(data_indices, tuple), f"data_indices must be a tuple, is a {type(data_indices)}"

        self.statistics_tendencies = statistics_tendencies
        self.output_mask = instantiate(config.model_dump(by_alias=True).model.output_mask, graph_data=graph_data)

        self.model = FuserModelInterface(
            statistics=statistics,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays | self.output_mask.supporting_arrays,
            graph_data=graph_data,
            truncation_data=truncation_data,
            config=convert_to_omegaconf(config),
        )
        self.data_indices = data_indices

        self.save_hyperparameters()

        self.latlons_data = [graph_data[mesh].x for mesh in config.graph.input_nodes.values()]
        self.node_weights = self.get_node_weights(config, graph_data)

        self.node_weights = self.output_mask.apply(self.node_weights, dim=0, fill_value=0.0)

        self.dset_weights = config.training.dataset_loss_scaling

        self.logger_enabled = config.diagnostics.log.wandb.enabled or config.diagnostics.log.mlflow.enabled

        self.val_metric_ranges = self.get_val_metric_ranges(config, data_indices)

        zip_loss = []
        
        for dset, loss_config in enumerate(config.training.training_loss):
            loss_config = config.model_dump(by_alias=True).training.training_loss[dset]

            # Create scalers for this dataset
            data_index = data_indices[dset]
            scalers_result = create_scalers(
                config.model_dump(by_alias=True).training.scalers,
                group_config=config.training.variable_groups,
                data_indices=data_index,
                graph_data=graph_data,
                statistics=statistics[dset] if isinstance(statistics, (list, tuple)) else statistics,
                statistics_tendencies=statistics_tendencies[dset] if isinstance(statistics_tendencies, (list, tuple)) else statistics_tendencies,
                metadata_variables=metadata["dataset"].get("variables_metadata"),
                output_mask=self.output_mask,
            )
            
            # Extract scalers dictionary
            scalers_dict = scalers_result[0] if isinstance(scalers_result, (tuple, list)) else scalers_result

            # Create loss for this dataset 
            loss = get_loss_function(
                loss_config,
                scalers=scalers_dict,
                data_indices=data_index,
            )
            zip_loss.append(loss)
        
        self.loss = ZipLoss(zip_loss)

        # Create validation metrics for each dataset
        zip_metrics = []
        for dset, metrics_dict in enumerate(config.model_dump(by_alias=True).training.validation_metrics):
            # Create scalers for this dataset's metrics
            scalers_result = create_scalers(
                config.model_dump(by_alias=True).training.scalers,
                group_config=config.training.variable_groups,
                data_indices=data_indices[dset],
                graph_data=graph_data,
                statistics=statistics[dset] if isinstance(statistics, (list, tuple)) else statistics,
                statistics_tendencies=statistics_tendencies[dset] if isinstance(statistics_tendencies, (list, tuple)) else statistics_tendencies,
                metadata_variables=metadata["dataset"].get("variables_metadata"),
                output_mask=self.output_mask,
            )
            scalers_dict = scalers_result[0] if isinstance(scalers_result, (tuple, list)) else scalers_result

            dataset_metrics = torch.nn.ModuleDict({
                metric_name: get_loss_function(
                    val_metric_config,
                    scalers=scalers_dict,
                    data_indices=data_indices[dset],
                )
                for metric_name, val_metric_config in metrics_dict.items()
            })
            zip_metrics.append(dataset_metrics)

        self.metrics = zip_metrics

        self.multi_step = config.training.multistep_input
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_model
        )
        self.lr_iterations = config.training.lr.iterations
        self.lr_min = config.training.lr.min
        self.rollout = config.training.rollout.start
        self.rollout_epoch_increment = config.training.rollout.epoch_increment
        self.rollout_max = config.training.rollout.max

        self.model_comm_group = None

        self.model_comm_group_id = int(os.environ.get("SLURM_PROCID", "0")) // config.hardware.num_gpus_per_model
        self.model_comm_group_rank = int(os.environ.get("SLURM_PROCID", "0")) % config.hardware.num_gpus_per_model
        self.model_comm_num_groups = math.ceil(
            config.hardware.num_gpus_per_node * config.hardware.num_nodes / config.hardware.num_gpus_per_model,
        )

    def forward(self, x: list[torch.Tensor]) -> list[torch.Tensor]:
        return self.model(x, model_comm_group=self.model_comm_group)

    @staticmethod
    def get_val_metric_ranges(
        config: DictConfig,
        data_indices: list,
    ) -> list[dict]:
        from anemoi.training.losses.loss import get_metric_ranges
        return [get_metric_ranges(config, data_index, None) for data_index in data_indices]


    @staticmethod
    def get_node_weights(
        config: DictConfig,
        graph_data: HeteroData,
    ) -> list:
        # Use scalers from the scalers section instead of node_loss_weights
        scalers = config.training.scalers
        node_weights = []
        for scaler_name, scaler_config in scalers.items():
            if scaler_name.endswith('_node_weights'):
                # Create scaler and get the actual values using the scaler system
                scaler = instantiate(scaler_config.model_dump(by_alias=True), graph_data=graph_data)
                _, weights = scaler.get_scaling()  # Returns (dimensions, values)
                node_weights.append(torch.from_numpy(weights))
        return node_weights

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
        x: list,
        y_pred: list,
        batch: list,
        rollout_step: int,
    ) -> list:
        for dset_idx, (x_elem, y_pred_elem, batch_elem) in enumerate(zip(x, y_pred, batch)):
            x_elem = x_elem.roll(-1, dims=1)
            x_elem[:, -1, :, :, self.data_indices[dset_idx].model.input.prognostic] = y_pred_elem[
                ...,
                self.data_indices[dset_idx].model.output.prognostic,
            ]
            x_elem[:, -1] = self.output_mask.rollout_boundary(x_elem[:, -1], batch_elem[:, -1], self.data_indices[dset_idx])
            x_elem[:, -1, :, :, self.data_indices[dset_idx].model.input.forcing] = batch_elem[
                :,
                self.multi_step + rollout_step,
                :,
                :,
                self.data_indices[dset_idx].data.input.forcing,
            ]
        return x

    def rollout_step(
        self,
        batch: list,
        rollout: Optional[int] = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ):
        num_dsets = len(batch)
        batch = self.model.pre_processors(batch, in_place=not validation_mode)
        x = [None] * num_dsets
        for batch_idx, batch_elem in enumerate(batch):
            x[batch_idx] = batch_elem[
                :,
                0 : self.multi_step,
                ...,
                self.data_indices[batch_idx].data.input.full,
            ]
            assert batch[batch_idx].shape[1] >= (rollout or self.rollout) + self.multi_step

        for rollout_step in range(rollout or self.rollout):
            y_pred = self(x)
            assert isinstance(y_pred, list), f"y_pred must be a list, is a {type(y_pred)}"

            y = [None] * num_dsets
            for batch_idx, batch_elem in enumerate(batch):
                y[batch_idx] = batch_elem[
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices[batch_idx].data.output.full,
                ]

            loss = self.loss(y_pred, y) if training_mode else None

            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = [{} for _ in range(len(batch))]
            if validation_mode:
                metrics_next = self.calculate_val_metrics(
                    y_pred,
                    y,
                    rollout_step,
                )
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: list,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> list:
        del batch_idx
        num_dsets = len(batch)
        loss = [torch.zeros(1, dtype=batch[i].dtype, device=self.device, requires_grad=False) for i in range(num_dsets)]
        metrics = [{} for _ in range(num_dsets)]
        y_preds = [[] for _ in range(num_dsets)]

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
            for dset in range(num_dsets):
                if loss_next is not None:
                    loss[dset] += loss_next[dset]
                metrics[dset].update(metrics_next[dset])
                y_preds[dset].extend(y_preds_next[dset])

        for dset in range(num_dsets):
            loss[dset] *= 1.0 / self.rollout

        return loss, metrics, y_preds

    def calculate_val_metrics(
        self,
        y_pred: list[torch.Tensor],
        y: list[torch.Tensor],
        rollout_step: int,
        grid_shard_slice: slice | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        num_dsets = len(y_pred)
        metrics = [{} for _ in range(num_dsets)]
        y_postprocessed = self.model.post_processors(y, in_place=False)
        y_pred_postprocessed = self.model.post_processors(y_pred, in_place=False)

        for dset, dataset_metrics in enumerate(self.metrics):
            for metric_name, metric in dataset_metrics.items():
                if not isinstance(metric, BaseLoss):
                    # If not a loss, we cannot feature scale, so call normally
                    metrics[dset][f"{metric_name}_metric/{rollout_step + 1}"] = metric(
                        y_pred_postprocessed[dset],
                        y_postprocessed[dset],
                    )
                    continue

                for mkey, indices in self.val_metric_ranges[dset].items():
                    metric_step_name = f"{metric_name}_metric/{mkey}/{rollout_step + 1}"
                    if len(metric.scaler.subset_by_dim(TensorDim.VARIABLE.value)):
                        exception_msg = (
                            "Validation metrics cannot be scaled over the variable dimension"
                            " in the post processed space."
                        )
                        raise ValueError(exception_msg)

                    metrics[dset][metric_step_name] = metric(
                        y_pred_postprocessed[dset],
                        y_postprocessed[dset],
                        scaler_indices=[..., indices],
                        grid_shard_slice=grid_shard_slice,
                        group=self.model_comm_group,
                    )

        return metrics

    def training_step(self, batch: list, batch_idx: int) -> torch.Tensor:
        train_loss, _, _ = self._step(batch, batch_idx)
        for i in range(len(train_loss)):
            train_loss[i] = train_loss[i] * self.dset_weights[i]
        combined_loss = sum(train_loss)
        self.log(
            "train_wmse",
            combined_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )
        for dset, loss in enumerate(train_loss):
            self.log(
                f"train_{getattr(self.loss.losses[dset], 'name', self.loss.losses[dset].__class__.__name__.lower())}_dset{dset}",
                loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )
        return combined_loss

    def lr_scheduler_step(self, scheduler: CosineLRScheduler, metric: None = None) -> None:
        del metric
        scheduler.step(epoch=self.trainer.global_step)

    def on_train_epoch_end(self) -> None:
        if self.rollout_epoch_increment > 0 and self.current_epoch % self.rollout_epoch_increment == 0:
            self.rollout += 1
        self.rollout = min(self.rollout, self.rollout_max)

    def validation_step(self, batch: list, batch_idx: int) -> None:
        with torch.no_grad():
            val_loss, metrics, y_preds = self._step(batch, batch_idx, validation_mode=True)
        for i in range(len(val_loss)):
            val_loss[i] = val_loss[i] * self.dset_weights[i]
        combined_loss = sum(val_loss)
        self.log(
            "val_wmse",
            combined_loss,
            on_epoch=True,
            on_step=True,
            prog_bar=True,
            logger=self.logger_enabled,
            batch_size=batch[0].shape[0],
            sync_dist=True,
        )
        for dset, loss in enumerate(val_loss):
            self.log(
                f"val_{getattr(self.loss.losses[dset], 'name', self.loss.losses[dset].__class__.__name__.lower())}_dset{dset}",
                loss,
                on_epoch=True,
                on_step=True,
                prog_bar=True,
                logger=self.logger_enabled,
                batch_size=batch[0].shape[0],
                sync_dist=True,
            )
            for mname, mvalue in metrics[dset].items():
                self.log(
                    f"val_{mname}_dset{dset}",
                    mvalue,
                    on_epoch=True,
                    on_step=False,
                    prog_bar=False,
                    logger=self.logger_enabled,
                    batch_size=batch[0].shape[0],
                    sync_dist=True,
                )
        return combined_loss, y_preds

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.trainer.model.parameters(),
            betas=(0.9, 0.95),
            lr=self.lr,
        )
        scheduler = CosineLRScheduler(
            optimizer,
            lr_min=self.lr_min,
            t_initial=self.lr_iterations,
            warmup_t=1000,
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

