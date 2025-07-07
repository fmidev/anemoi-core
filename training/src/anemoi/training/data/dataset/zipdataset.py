# training/src/anemoi/training/data/dataset/zipdataset.py
from __future__ import annotations
import logging
import os
import random
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
import torch
from einops import rearrange
from torch.utils.data import IterableDataset

from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from anemoi.training.data.grid_indices import BaseGridIndices

from typing import Sequence, Callable
from anemoi.training.data.dataset.singledataset import NativeGridDataset
from anemoi.training.data.grid_indices import BaseGridIndices


class ZipDataset(IterableDataset):
    """Zip multiple NativeGridDatasets into one aligned dataset."""

    def __init__(
        self,
        data_readers: Sequence[Callable],
        grid_indices: BaseGridIndices,
        rollout: int,
        multistep: int,
        timeincrement: int,
        shuffle: bool = True,
        label: str = "zip",
    ) -> None:
        # compute the list of offsets for each sample
        rel_dates = self._get_relative_date_indices(
            rollout=rollout,
            multistep=multistep,
            timeincrement=timeincrement,
        )

        # create one NativeGridDataset per reader
        self.datasets = [
            NativeGridDataset(
                data_reader=reader,
                grid_indices=grid_indices,
                relative_date_indices=rel_dates,
                timestep=f"{timeincrement}h",
                shuffle=shuffle,
                label=label,
            )
            for reader in data_readers
        ]

    def __len__(self) -> int:
        # assume all sub-datasets have same length
        return len(self.datasets[0])

    def __iter__(self):
        # get an iterator for each dataset
        iters = [iter(ds) for ds in self.datasets]
        for batch_tuple in zip(*iters):
            # each batch_tuple is a list of tensors
            # combine them into one dict or tuple as needed
            yield batch_tuple

    def _get_relative_date_indices(self, rollout, multistep, timeincrement):
        rel_dates = [i * timeincrement for i in range(rollout + 1)]
        return rel_dates
