# (C) Copyright 2024-2025 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


from __future__ import annotations

import logging
import os
import random
from typing import Callable

import numpy as np
import torch
from einops import rearrange
from omegaconf import DictConfig
from torch.utils.data import IterableDataset

from anemoi.training.utils.seeding import get_base_seed
from anemoi.training.utils.usable_indices import get_usable_indices

LOGGER = logging.getLogger(__name__)


class ZipDataset(IterableDataset):
    """Iterable dataset for multiple AnemoI datasets zipped together."""

    def __init__(
        self,
        data_reader: Callable,
        rollout: int = 1,
        multistep: int = 1,
        timeincrement: int = 1,
        model_comm_group_rank: int = 0,
        model_comm_group_id: int = 0,
        model_comm_num_groups: int = 1,
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        self.label = label
        self.data = data_reader
        self.rollout = rollout
        self.timeincrement = timeincrement

        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.model_comm_group_id = model_comm_group_id
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        self.multi_step = multistep
        assert self.multi_step > 0, "Multistep value must be greater than zero."
        self.ensemble_dim: int = 2
        assert all(
            dset_shape[self.ensemble_dim] == self.data.shape[0][self.ensemble_dim] for dset_shape in self.data.shape
        ), "Ensemble size must match for all datasets"
        self.ensemble_size = self.data.shape[0][self.ensemble_dim]

        LOGGER.debug(f"Name-to-index {self.data.name_to_index}")

    @property
    def statistics(self) -> dict:
        """Return statistics as a dictionary for each dataset in the zip."""
        if hasattr(self.data, 'statistics'):
            stats = self.data.statistics
            if isinstance(stats, dict):
                return stats
            elif isinstance(stats, (list, tuple)):
                # If statistics is a list/tuple, convert to dict with integer keys
                return {i: stat for i, stat in enumerate(stats)}
            else:
                # If it's a single value, wrap it in a dict
                return {0: stats}
        return {}

    @property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timeincrement)
        except (KeyError, AttributeError):
            return None

    @property
    def metadata(self) -> dict:
        return self.data.metadata()

    @property
    def name_to_index(self) -> dict:
        return self.data.name_to_index
 

    @property
    def resolution(self) -> dict:
        return self.data.resolution

    @property
    def supporting_arrays(self) -> dict:
        """Return supporting arrays as a flat dictionary. Also adds decoder output coordinates."""
        raw_arrays = dict(self.data.supporting_arrays())
        
        # If it's already a dict, clean up any tuple values
        result = {}
        for key, value in raw_arrays.items():
            # Remove numeric prefixes to match FROZEN structure
            clean_key = key.split('/', 1)[1] if key.startswith(('0/', '1/', '2/', '3/', '4/', '5/', '6/', '7/', '8/', '9/')) else key

            # Handle cases where value might be a tuple of arrays
            if isinstance(value, tuple) and len(value) > 0:
                result[clean_key] = value[0]
            elif hasattr(value, 'shape'):
                # It's already a proper array
                result[clean_key] = value

        # Add decoder output coordinates (decoderN -> datasetN, starting from decoder1)
        # Zip.latitudes and Zip.longitudes return tuples, so we index into them
        for decoder_idx in range(1, len(self.data.datasets)):
            result[f'output{decoder_idx}/latitudes'] = self.data.latitudes[decoder_idx]
            result[f'output{decoder_idx}/longitudes'] = self.data.longitudes[decoder_idx]

        for key, value in result.items():
            assert not isinstance(value, tuple), f"Value for key {key} is a tuple"
        return result

    @property
    def valid_date_indices(self) -> np.ndarray:
        # Use the cloudy-skies compatible approach
        prev_invalid_dates = (self.multi_step - 1) * self.timeincrement
        next_invalid_dates = self.rollout * self.timeincrement

        usable_indices = np.arange(len(self.data))  # set of all indices

        if self.data.missing is None:
            missing_indices = set()
        else:
            missing_indices = set(self.data.missing)

        missing_indices |= {-1, len(self.data)}  # to filter initial and final indices

        # Missing indices
        for i in missing_indices:
            usable_indices = usable_indices[
                (usable_indices < i - next_invalid_dates) + (usable_indices > i + prev_invalid_dates)
            ]

        return usable_indices

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "ZipDataset.set_group_info(): global_rank %d, model_comm_group_id %d, model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        self.worker_id = worker_id

        shard_size = len(self.valid_date_indices) // self.model_comm_num_groups
        shard_start = self.model_comm_group_id * shard_size
        shard_end = (self.model_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        base_seed = get_base_seed()
        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)

    def __iter__(self) -> torch.Tensor:
        if self.shuffle:
            shuffled_chunk_indices = self.rng.choice(
                self.valid_date_indices,
                size=len(self.valid_date_indices),
                replace=False,
            )[self.chunk_index_range]
        else:
            shuffled_chunk_indices = self.valid_date_indices[self.chunk_index_range]

        LOGGER.debug(
            (
                "Worker pid %d, label %s, worker id %d, global_rank %d, "
                "model comm group %d, group_rank %d using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            getattr(self, "worker_id", -1),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i - (self.multi_step - 1) * self.timeincrement
            end = i + (self.rollout + 1) * self.timeincrement

            x = self.data[start : end : self.timeincrement]
            x = tuple(
                torch.from_numpy(
                    rearrange(data, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")
                )
                for data in x
            )
            self.ensemble_dim = 1

            yield x
