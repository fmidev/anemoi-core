# (C) Copyright 2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging

import torch

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class MAELogitLoss(FunctionalLoss):
    """MAE logit loss."""

    name: str = "mae_logit"

    def __init__(self, ignore_nans: bool = False, eps: float = 1e-4, **kwargs) -> None:
        super().__init__(ignore_nans=ignore_nans, **kwargs)
        self.eps = eps

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the MAE logit loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            MAE logit loss
        """
        # Clamp precision to avoid log(0) and log(1)
        pred = torch.clamp(pred, min=self.eps, max=1.0 - self.eps)
        target = torch.clamp(target, min=self.eps, max=1.0 - self.eps)

        return torch.abs(torch.logit(pred) - torch.logit(target))
