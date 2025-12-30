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
import torch.nn.functional as F

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class BCELoss(FunctionalLoss):
    """BCE loss."""

    name: str = "bce"

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the BCE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            BCE loss
        """
        # Clamp precision to avoid log(0)
        with torch.amp.autocast(device_type="cuda", enabled=False):
            return F.binary_cross_entropy(torch.clamp(pred.float(), min=1e-6, max=1.0-1e-6), target.float(), reduction='none')
