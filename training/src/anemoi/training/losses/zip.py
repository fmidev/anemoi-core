# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import torch.nn as nn
from anemoi.training.losses.base import BaseLoss

class ZipLoss(BaseLoss):
    """A loss function that applies multiple loss functions to corresponding predictions and targets."""

    def __init__(
        self,
        loss_functions: list,
    ) -> None:
        """Initialize the ZipLoss.

        Parameters
        ----------
        loss_functions : list
            List of loss functions to apply
        """
        super().__init__()
        self.losses = nn.ModuleList(loss_functions)

    def forward(
        self,
        pred: list,
        target: list,
        squash: bool = True,
    ) -> tuple:
        """Forward pass of the ZipLoss.

        Parameters
        ----------
        pred : list
            List of predictions
        target : list
            List of targets
        squash : bool
            Whether to squash the output

        Returns
        -------
        tuple
            Tuple of loss values
        """
        out = ()

        assert isinstance(pred, list), f"pred must be a list, is a {type(pred)}"
        for i, loss in enumerate(self.losses):
            out += (loss(pred[i], target[i], squash),)

        return out
