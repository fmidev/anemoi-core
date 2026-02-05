# (C) Copyright 2024- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from .combined import CombinedLoss
from .huber import HuberLoss
from .kcrps import AlmostFairKernelCRPS
from .kcrps import KernelCRPS
from .logcosh import LogCoshLoss
from .loss import get_loss_function
from .mae import MAELoss
from .mae_logit import MAELogitLoss
from .mse import MSELoss
from .rmse import RMSELoss
from .zip import ZipLoss
from .cce import CategoricalCrossEntropyLoss
from .bce import BCELoss

__all__ = [
    "AlmostFairKernelCRPS",
    "BCELoss",
    "CombinedLoss",
    "HuberLoss",
    "KernelCRPS",
    "LogCoshLoss",
    "MAELoss",
    "MAELogitLoss",
    "MSELoss",
    "RMSELoss",
    "ZipLoss",
    "CategoricalCrossEntropyLoss",
    "get_loss_function",
]
