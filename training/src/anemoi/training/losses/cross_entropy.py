# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
from contextlib import nullcontext

import einops
import torch
from torch.distributed.distributed_c10d import ProcessGroup
from torch.nn import functional

from anemoi.training.losses.base import BaseLoss
from anemoi.training.losses.base import Squash_mode
from anemoi.training.utils.enums import TensorDim

LOG = logging.getLogger(__name__)


class CrossEntropyLoss(BaseLoss):
    """Cross-entropy loss for categorical variables predicted by an ensemble.

    The variables passed to the loss are the classes of one categorical variable: the predictions
    are unnormalised logits, one per class, and the target is a one-hot or soft class distribution.

    The loss is returned per class, ``-q(c) log p(c)``, so that scalers and the reduction work as for
    any other loss. With the default ``squash_mode="avg"`` the variable dimension is averaged, so the
    reported loss is the cross-entropy divided by the number of classes.
    """

    def __init__(
        self,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Cross-entropy loss for categorical variables predicted by an ensemble.

        Parameters
        ----------
        no_autocast : bool, optional
            Compute the softmax and the cross-entropy in float32 with autocast disabled, by default True.
        ignore_nans : bool, optional
            Ignore grid points where the target (any class) or the prediction (any class or member) is NaN,
            by default False.
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.no_autocast = no_autocast

    def _ce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Per-class cross-entropy term -q(c) log p(c).

        Parameters
        ----------
        pred : torch.Tensor
            Logits, shape (bs, t, ens, grid, classes).
        target : torch.Tensor
            Target class distribution, shape (bs, t, grid, classes).

        Returns
        -------
        torch.Tensor
            Shape (bs, t, grid, classes); summing over the last dimension gives the cross-entropy.
        """
        # Log-probabilities per ensemble member: (bs, t, ens, grid, classes)
        log_probs = functional.log_softmax(pred.float(), dim=TensorDim.VARIABLE)

        # Mean of the members' own cross-entropies: (bs, t, grid, classes)
        return -(target.float() * log_probs.mean(dim=TensorDim.ENSEMBLE_DIM))

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
        squash_mode: Squash_mode = "avg",
        **_kwargs,
    ) -> torch.Tensor:
        is_sharded = grid_shard_slice is not None

        # pred:   (bs, t, ens, grid, classes) - logits per ensemble member
        # target: (bs, t, 1, grid, classes)   - one-hot or soft class distribution
        if pred.ndim != 5:
            msg = f"pred must have shape (bs, t, ens, grid, classes), got {tuple(pred.shape)}."
            raise ValueError(msg)
        if pred.shape[TensorDim.VARIABLE] != target.shape[-1]:
            msg = (
                f"pred has {pred.shape[TensorDim.VARIABLE]} classes but target has {target.shape[-1]}; "
                "the loss must see exactly the class variables, in the same order."
            )
            raise ValueError(msg)

        if self.ignore_nans:
            # A grid point with a NaN in any class of the target, or of any member's logits, does not
            # contribute: its target is set to zero for all classes.
            nan_point = torch.isnan(target).any(dim=TensorDim.VARIABLE, keepdim=True) | torch.isnan(pred).any(
                dim=(TensorDim.ENSEMBLE_DIM, TensorDim.VARIABLE),
                keepdim=True,
            )
            target = target.masked_fill(nan_point, 0.0)
            pred = pred.masked_fill(nan_point, 0.0)

        target = einops.rearrange(target, "bs t 1 grid v -> bs t grid v")

        context = torch.amp.autocast(device_type=pred.device.type, enabled=False) if self.no_autocast else nullcontext()
        with context:
            ce = self._ce_loss(pred, target)

        ce = einops.rearrange(ce, "bs t grid v -> bs t 1 grid v")
        ce = self.scale(ce, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(ce, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return "cross_entropy"
