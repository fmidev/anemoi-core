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

import torch

from anemoi.training.losses.base import FunctionalLoss

LOGGER = logging.getLogger(__name__)


class CategoricalCrossEntropyLoss(FunctionalLoss):
    """Categorical Cross Entropy loss."""

    name: str = "cce"

    def calculate_difference(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate the MSE loss.

        Parameters
        ----------
        pred : torch.Tensor
            Prediction tensor, shape (bs, ensemble, lat*lon, n_outputs)
        target : torch.Tensor
            Target tensor, shape (bs, ensemble, lat*lon, n_outputs)

        Returns
        -------
        torch.Tensor
            Categorical Cross Entropy loss
        """

        return torch.nn.functional.cross_entropy(pred, target)
