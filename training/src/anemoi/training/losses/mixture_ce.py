# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import einops
import logging
import math

import torch
from torch.nn import functional
from torch.distributed.distributed_c10d import ProcessGroup

from anemoi.training.losses.base import BaseLoss, Squash_mode
from anemoi.training.utils.enums import TensorDim

LOG = logging.getLogger(__name__)


class MixtureCrossEntropyLoss(BaseLoss):
    """Ensemble mixture cross-entropy loss for categorical variables.

    Treats the ensemble as a single mixture distribution and applies cross-entropy
    to the mixture, analogous to CRPS for continuous ensemble predictions.

    For an ensemble of M members each producing log-probabilities log p_m(c),
    the mixture log-probability is:

        log p_ens(c) = logsumexp_m(log p_m(c)) - log(M)

    The loss is then:

        L = -sum_c q(c) * log p_ens(c)

    where q(c) is the target distribution (one-hot or soft labels).
    """

    def __init__(
        self,
        label_smoothing: float = 0.0,
        no_autocast: bool = True,
        ignore_nans: bool = False,
        **kwargs,
    ) -> None:
        """Ensemble mixture cross-entropy loss.

        Parameters
        ----------
        label_smoothing : float, optional
            Label smoothing epsilon applied to target distribution before CE.
            Smoothed target = (1 - eps) * target + eps / num_classes.
            By default 0.0 (no smoothing).
        ignore_nans : bool, optional
            Allow NaNs in the loss and use nan-safe reduction functions, by default False
        """
        super().__init__(ignore_nans=ignore_nans, **kwargs)

        self.label_smoothing = label_smoothing
        self.no_autocast = no_autocast

    def _ce_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        ens_size = pred.shape[TensorDim.ENSEMBLE_DIM]
        # From logits to log-probabilities per ensemble member: (bs, t, ens, grid, var
        log_probs = functional.log_softmax(pred.float(), dim=TensorDim.VARIABLE)

        # Mixture log-probability via logsumexp over ensemble dim: (bs, t, grid, var)
        log_p_ens = torch.logsumexp(log_probs, dim=TensorDim.ENSEMBLE_DIM) - math.log(ens_size)

        # Cross-entropy on mixture distribution: (bs, t, grid, var)
        return -(target.float() * log_p_ens)

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

        # y_pred:   (bs, t, ens, grid, var) - one-hot-encoded predictions
        # y_target: (bs, t, grid, var)      - single ground truth, no ensemble dim
        target = einops.rearrange(target, "bs t 1 latlon v -> bs t latlon v")
        num_classes = pred.shape[TensorDim.VARIABLE]

        # Apply label smoothing to target distribution
        if self.label_smoothing > 0.0:
            target = (1.0 - self.label_smoothing) * target + self.label_smoothing / num_classes
        else:
            target = target

        if self.no_autocast:
            with torch.amp.autocast(device_type=pred.device.type, enabled=False):
                ce = self._ce_loss(pred, target)
        else:
            ce = self._ce_loss(pred, target)

        ce = einops.rearrange(ce, "bs t grid v -> bs t 1 grid v")

        ce = self.scale(ce, scaler_indices, without_scalers=without_scalers, grid_shard_slice=grid_shard_slice)

        return self.reduce(ce, squash=squash, squash_mode=squash_mode, group=group if is_sharded else None)

    @property
    def name(self) -> str:
        return "mixture_ce"
