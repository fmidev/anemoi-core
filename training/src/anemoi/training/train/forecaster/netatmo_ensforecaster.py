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
from typing import TYPE_CHECKING
from typing import Optional

import torch
from torch.utils.checkpoint import checkpoint
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.models.distributed.graph import gather_tensor
from anemoi.training.utils.inicond import EnsembleInitialConditions

from .netatmo_forecaster import NetatmoGraphForecaster

if TYPE_CHECKING:
    from collections.abc import Generator
    from omegaconf import DictConfig
    from torch_geometric.data import HeteroData

LOGGER = logging.getLogger(__name__)


class NetatmoGraphEnsForecaster(NetatmoGraphForecaster):
    """Netatmo Graph neural network forecaster for ensembles."""

    def __init__(
        self,
        *,
        config: DictConfig,
        graph_data: HeteroData,
        statistics: dict,
        statistics_tendencies: dict,
        data_indices: tuple,
        metadata: dict,
        supporting_arrays: dict | None = None,
        truncation_data: dict | None = None,
    ) -> None:
        super().__init__(
            config=config,
            graph_data=graph_data,
            statistics=statistics,
            statistics_tendencies=statistics_tendencies,
            data_indices=data_indices,
            metadata=metadata,
            supporting_arrays=supporting_arrays,
            truncation_data=truncation_data,
        )



        # Ensemble attributes from GraphEnsForecaster
        self.model_comm_group_size = config.hardware.num_gpus_per_model
        assert config.hardware.num_gpus_per_ensemble % config.hardware.num_gpus_per_model == 0

        self.num_gpus_per_model = config.hardware.num_gpus_per_model
        self.num_gpus_per_ensemble = config.hardware.num_gpus_per_ensemble

        # Recalculate LR for ensemble (divide by num_gpus_per_ensemble instead of model)
        self.lr = (
            config.hardware.num_nodes
            * config.hardware.num_gpus_per_node
            * config.training.lr.rate
            / config.hardware.num_gpus_per_ensemble
        )

        self.nens_per_device = config.training.ensemble_size_per_device
        self.nens_per_group = (
            config.training.ensemble_size_per_device 
            * self.num_gpus_per_ensemble 
            // config.hardware.num_gpus_per_model
        )
        
        # Lazy init ensemble group info
        self.ens_comm_group = None
        self.ens_comm_group_id = None
        self.ens_comm_group_rank = None
        self.ens_comm_num_groups = None
        self.ens_comm_group_size = None
        self.ens_comm_subgroup = None

        # Ensemble IC generator
        self.ensemble_ic_generator = EnsembleInitialConditions(
            config=config, 
            data_indices=data_indices[0] if isinstance(data_indices, (tuple, list)) else data_indices
        ) # Initial conditions usually driven by the main dataset (dataset 0) logic

        self.is_first_step = True

    def set_ens_comm_group(
        self,
        ens_comm_group: ProcessGroup,
        ens_comm_group_id: int,
        ens_comm_group_rank: int,
        ens_comm_num_groups: int,
        ens_comm_group_size: int,
    ) -> None:
        self.ens_comm_group = ens_comm_group
        self.ens_comm_group_id = ens_comm_group_id
        self.ens_comm_group_rank = ens_comm_group_rank
        self.ens_comm_num_groups = ens_comm_num_groups
        self.ens_comm_group_size = ens_comm_group_size

    def set_ens_comm_subgroup(
        self,
        ens_comm_subgroup: ProcessGroup,
        ens_comm_subgroup_id: int,
        ens_comm_subgroup_rank: int,
        ens_comm_subgroup_num_groups: int,
        ens_comm_subgroup_size: int,
    ) -> None:
        self.ens_comm_subgroup = ens_comm_subgroup
        self.ens_comm_subgroup_id = ens_comm_subgroup_id
        self.ens_comm_subgroup_rank = ens_comm_subgroup_rank
        self.ens_comm_subgroup_num_groups = ens_comm_subgroup_num_groups
        self.ens_comm_subgroup_size = ens_comm_subgroup_size

    def forward(self, x: list[torch.Tensor], fcstep: int = 0) -> list[torch.Tensor]:
        # Pass fcstep if model supports it, typically for noise scheduling
        return self.model(
            x, 
            model_comm_group=self.model_comm_group,
            fcstep=fcstep,
            grid_shard_shapes=getattr(self, 'grid_shard_shapes', None) # GraphForecaster has this, check usage
        )

    def define_delayed_scalers(self):
        # GraphEnsForecaster uses this, assume inherited logic or similar needed
        pass # Placeholder if specific scaler logic is needed, base Netatmo does it in init

    def advance_input(
        self,
        x: list,
        y_pred: list,
        batch: list,
        rollout_step: int,
    ) -> list:
        for dset_idx, (x_elem, y_pred_elem, batch_elem) in enumerate(zip(x, y_pred, batch)):
            # x_elem: (bs, multi_step, nens, nvar, latlon)
            # y_pred_elem: (bs, nens, latlon, nvar)

            # Shift time
            x_elem = x_elem.roll(-1, dims=1)
            
            # Update prognostic variables at last time step
            prog_indices = self.data_indices[dset_idx].model.output.prognostic
            y_prog = y_pred_elem[..., prog_indices] # (bs, ens, latlon, nvar_prog)

            x_elem[:, -1, :, :, self.data_indices[dset_idx].model.input.prognostic] = y_prog
            
            # Boundary handling (if any)
            # x_elem[:, -1] = self.output_mask.rollout_boundary(x_elem[:, -1], batch_elem[:, -1], self.data_indices[dset_idx])
            
            # Forcing
            forcing_data = batch_elem[
                :,
                self.multi_step + rollout_step,
                :,
                :,
                self.data_indices[dset_idx].data.input.forcing,
            ]
            
            # forcing_data is (bs, ens, grid, var)
            # Expand ensemble
            forcing_data = forcing_data.expand(-1, self.nens_per_device, -1, -1) # (bs, nens, grid, var)
            
            # Update forcing variables (last dimension)
            x_elem[:, -1, :, :, self.data_indices[dset_idx].model.input.forcing] = forcing_data
            
            x[dset_idx] = x_elem

        return x

    def gather_and_compute_loss(
        self,
        y_pred: list[torch.Tensor],
        y: list[torch.Tensor],
        loss_fn: ZipLoss,
        ens_comm_group_size: int,
        ens_comm_subgroup: ProcessGroup,
        model_comm_group: ProcessGroup,
        return_pred_ens: bool = False,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor] | None]:
        
        num_dsets = len(y_pred)
        loss_values = []
        y_pred_ens_list = []

        for i in range(num_dsets):
            # Gather ensemble: (bs, nens, latlon, nvar) -> (bs, total_ens, latlon, nvar)
            y_pred_gathered = gather_tensor(
                y_pred[i].clone(),
                dim=1,
                shapes=[y_pred[i].shape] * ens_comm_group_size,
                mgroup=ens_comm_subgroup
            )

            if return_pred_ens:
                y_pred_ens_list.append(y_pred_gathered)

            single_loss_fn = loss_fn.losses[i]

            # Ground truth from ZipDataset has a singleton ensemble dimension (bs, 1, grid, var)
            curr_y = y[i].squeeze(1) # (bs, grid, var)
            curr_pred = y_pred_gathered # (bs, total_ens, latlon, nvar)

            assert not torch.isnan(curr_pred).any(), f"NaN in gather_and_compute_loss: curr_pred[{i}] has {torch.isnan(curr_pred).sum()} NaNs"
            assert not torch.isnan(curr_y).any(), f"NaN in gather_and_compute_loss: curr_y[{i}] has {torch.isnan(curr_y).sum()} NaNs"

            l = single_loss_fn(
                curr_pred, 
                curr_y, 
                squash=True, 
                grid_shard_slice=getattr(self, 'grid_shard_slice', None),
                group=model_comm_group
            )
            loss_values.append(l)

        return loss_values, y_pred_ens_list if return_pred_ens else None


    def rollout_step(
        self,
        batch: list,
        rollout: Optional[int] = None,
        training_mode: bool = True,
        validation_mode: bool = False,
    ):
        num_dsets = len(batch)
        for _i, _b in enumerate(batch):
            assert not torch.isnan(_b).any(), f"NaN in raw batch[{_i}] before pre_processors: {torch.isnan(_b).sum()} NaNs"

        # Pre-process batch
        batch = self.model.pre_processors(batch, in_place=not validation_mode)
        for _i, _b in enumerate(batch):
            assert not torch.isnan(_b).any(), f"NaN after pre_processors: batch[{_i}] has {torch.isnan(_b).sum()} NaNs"

        x = [None] * num_dsets
        
        # Initialize x for all datasets and apply IC generation (perturbations)
        # For dataset 0 (main):
        base_x_0 = batch[0][:, 0 : self.multi_step, ..., self.data_indices[0].data.input.full]
        x[0] = self.ensemble_ic_generator(base_x_0, None) # (bs, step, ens, var, latlon)
        
        # For other datasets, expand to ensemble dim without perturbation
        for i in range(1, num_dsets):
            base_x_i = batch[i][:, 0 : self.multi_step, ..., self.data_indices[i].data.input.full]
            if base_x_i.ndim == 4:
                x[i] = base_x_i.unsqueeze(2).expand(-1, -1, self.nens_per_device, -1, -1)
            elif base_x_i.ndim == 5:
                x[i] = base_x_i.expand(-1, -1, self.nens_per_device, -1, -1)
            else:
                raise ValueError(f"Unexpected dimensions for dataset {i}: {base_x_i.shape}")

        # Lazy init scalers
        if self.is_first_step:
             self.is_first_step = False

        for _i, _x in enumerate(x):
            assert not torch.isnan(_x).any(), f"NaN in initial x[{_i}] before rollout: {torch.isnan(_x).sum()} NaNs"

        for rollout_step in range(rollout or self.rollout):
            # Predict
            y_pred = self(x, fcstep=rollout_step) # list of (bs, nens, latlon, nvar)
            for _i, _yp in enumerate(y_pred):
                assert not torch.isnan(_yp).any(), f"NaN in y_pred[{_i}] at rollout_step={rollout_step}: {torch.isnan(_yp).sum()} NaNs"

            # Extract Target y
            y = [None] * num_dsets
            for i in range(num_dsets):
                y[i] = batch[i][
                    :,
                    self.multi_step + rollout_step,
                    ...,
                    self.data_indices[i].data.output.full,
                ]

            # Compute Loss
            if training_mode:
                loss, y_pred_ens = checkpoint(
                    self.gather_and_compute_loss,
                    y_pred,
                    y,
                    self.loss,
                    self.ens_comm_subgroup_size,
                    self.ens_comm_subgroup,
                    self.model_comm_group,
                    validation_mode,
                    use_reentrant=False
                )
            else:
                 loss = None
                 y_pred_ens = None
                 if validation_mode:
                     # Gather for metrics if needed
                     _, y_pred_ens = self.gather_and_compute_loss(
                        y_pred, y, self.loss,
                        self.ens_comm_subgroup_size, self.ens_comm_subgroup, self.model_comm_group,
                        return_pred_ens=True
                    )

            # Advance
            x = self.advance_input(x, y_pred, batch, rollout_step)

            metrics_next = [{} for _ in range(num_dsets)]
            
            yield loss, metrics_next, y_pred

    def _step(
        self,
        batch: list,
        batch_idx: int,
        validation_mode: bool = False,
    ) -> list:
        # Override to handle ensemble yield structure
        num_dsets = len(batch)
        loss = [torch.zeros(1, dtype=batch[i].dtype, device=self.device, requires_grad=False) for i in range(num_dsets)]
        metrics = [{} for _ in range(num_dsets)]

        for loss_next, metrics_next, y_preds_next in self.rollout_step(
            batch,
            rollout=self.rollout,
            training_mode=True,
            validation_mode=validation_mode,
        ):
             if loss_next:
                 for i in range(num_dsets):
                     loss[i] += loss_next[i]
        
        for i in range(num_dsets):
            loss[i] *= 1.0 / self.rollout
            
        return loss, metrics, [] # Return empty preds to save memory
