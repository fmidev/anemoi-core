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

import torch

from anemoi.training.losses.base import BaseLoss

if TYPE_CHECKING:
    from torch.distributed.distributed_c10d import ProcessGroup

LOGGER = logging.getLogger(__name__)


class CategoricalCrossEntropyLoss(BaseLoss):
    """Categorical Cross Entropy loss for multi-class classification.

    Expects one-hot encoded targets and either raw logits or probabilities.

    Works with both deterministic and ensemble models:
    - Deterministic: pred (bs, 1, grid, n_classes), target (bs, 1, grid, n_classes)
    - Ensemble: pred (bs, ens, grid, n_classes), target (bs, grid, n_classes)

    Parameters
    ----------
    from_logits : bool
        If True (default), pred contains raw logits and log_softmax is applied
        internally. Set to False when SoftmaxBounding has already been applied
        (pred contains probabilities); log is applied directly in that case.
    """

    def __init__(self, ignore_nans: bool = False, from_logits: bool = True, **kwargs) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.from_logits = from_logits

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        squash: bool = True,
        *,
        scaler_indices: tuple[int, ...] | None = None,
        without_scalers: list[str] | list[int] | None = None,
        grid_shard_slice: slice | None = None,
        group: ProcessGroup | None = None,
    ) -> torch.Tensor:
        """Compute area-weighted categorical cross entropy loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction logits or probabilities.
            Shape (bs, ensemble, grid, n_classes) for both det and ensemble.
        target : torch.Tensor
            One-hot encoded targets.
            Shape (bs, ensemble, grid, n_classes) for deterministic (ensemble=1),
            or (bs, grid, n_classes) for ensemble models.
        squash : bool, optional
            Average last dimension, by default True
        scaler_indices : tuple[int,...], optional
            Indices to subset the calculated scaler with, by default None
        without_scalers : list[str] | list[int] | None, optional
            Scalers to exclude from scaling, by default None
        grid_shard_slice : slice, optional
            Slice of the grid if x comes sharded, by default None
        group : ProcessGroup, optional
            Distributed group to reduce over, by default None

        Returns
        -------
        torch.Tensor
            Weighted loss
        """
        is_sharded = grid_shard_slice is not None

        assert pred.ndim == 4, f"CCE: pred must be 4D, got shape {pred.shape}"
        bs, ens, grid, n_classes = pred.shape

        # Bring target to (bs, ens, grid, n_classes) matching pred
        if target.ndim == 3:
            # Ensemble case: target (bs, grid, n_classes) — expand over ensemble
            target_4d = target.unsqueeze(1).expand(bs, ens, grid, n_classes).float()
        else:
            assert target.ndim == 4, f"CCE: target must be 3D or 4D, got shape {target.shape}"
            target_4d = target.float()

        if self.from_logits:
            # Numerically stable: log_softmax applied to raw logits
            log_probs = torch.nn.functional.log_softmax(pred, dim=-1)
        else:
            # SoftmaxBounding already applied — pred is probabilities
            log_probs = torch.log(pred.clamp(min=1e-7))

        ce = -(log_probs * target_4d).sum(dim=-1, keepdim=True)    # (bs, ens, grid, 1)

        # Expand to (bs, ens, grid, n_classes) so the training framework can index
        # per-variable with scaler_indices. All class slots hold the same CE value;
        # reduce() averages over the variable dim so the result is unchanged.
        ce = ce.expand(bs, ens, grid, n_classes).contiguous()

        ce = self.scale(ce, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)
        return self.reduce(ce, squash=squash, group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return "cce"
