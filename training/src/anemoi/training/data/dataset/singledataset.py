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


class NativeGridDataset(IterableDataset):
    """Iterable dataset for AnemoI data on the arbitrary grids."""

    def __init__(
        self,
        data_reader: Callable,
        grid_indices: type[BaseGridIndices],
        relative_date_indices: list,
        timestep: str = "6h",
        shuffle: bool = True,
        label: str = "generic",
    ) -> None:
        """Initialize (part of) the dataset state.

        Parameters
        ----------
        data_reader : Callable
            user function that opens and returns the anemoi-datasets array data
        grid_indices : Type[BaseGridIndices]
            indices of the grid to keep. Defaults to None, which keeps all spatial indices.
        relative_date_indices: list
            list of time indices to load from the data relative to the current sample i in __iter__
        timestep : int, optional
            the time frequency of the samples, by default '6h'
        shuffle : bool, optional
            Shuffle batches, by default True
        label : str, optional
            label for the dataset, by default "generic"
        """
        self.label = label

        self.data = data_reader

        self.timestep = timestep
        self.grid_indices = grid_indices

        # lazy init
        self.n_samples_per_epoch_total: int = 0
        self.n_samples_per_epoch_per_worker: int = 0

        # lazy init model and reader group info, will be set by the DDPGroupStrategy:
        self.model_comm_group_rank = 0
        self.model_comm_num_groups = 1
        self.model_comm_group_id = 0
        self.global_rank = 0

        self.reader_group_rank = 0
        self.reader_group_size = 1

        self.sample_comm_num_groups = 1  # groups that work on the same sample / batch
        self.sample_comm_group_id = 0

        # additional state vars (lazy init)
        self.n_samples_per_worker = 0
        self.chunk_index_range: np.ndarray | None = None
        self.shuffle = shuffle

        # Data dimensions
        self.ensemble_dim: int = 2
        self.ensemble_size = self.data.shape[self.ensemble_dim]

        # relative index of dates to extract
        self.relative_date_indices = relative_date_indices

        # Determine the time increment for slicing based on NativeGridDataset's timestep
        # and the raw data_reader's fundamental resolution (`self.data.resolution`).
        # This is `timeincrement` from the original __iter__ that was based on relative_date_indices.
        # Now, it's explicitly derived from frequencies.
        try:
            native_data_frequency_seconds = frequency_to_seconds(self.data.resolution)
        except AttributeError:
            # Fallback if data_reader doesn't have .resolution
            LOGGER.warning(
                f"Data reader for {self.label} does not have a 'resolution' attribute. Assuming default to 1h for time_slice_increment calculation."
            )
            native_data_frequency_seconds = 3600  # Assume 1h if not specified

        self.time_slice_increment = (
            frequency_to_seconds(self.timestep) // native_data_frequency_seconds
        )

        if self.time_slice_increment == 0:
            LOGGER.warning(
                f"Calculated time_slice_increment is 0 for timestep {self.timestep} and data resolution {self.data.resolution}. Setting to 1."
            )
            self.time_slice_increment = 1  # Prevent infinite loop or errors

    @cached_property
    def statistics(self) -> dict:
        """Return dataset statistics."""
        return self.data.statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        """Return dataset tendency statistics."""
        try:
            return self.data.statistics_tendencies(self.timestep)
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        """Return dataset metadata."""
        return self.data.metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        """Return dataset supporting_arrays."""
        return self.data.supporting_arrays()

    @cached_property
    def name_to_index(self) -> dict:
        """Return dataset statistics."""
        return self.data.name_to_index

    @cached_property
    def resolution(self) -> dict:
        """Return dataset resolution."""
        return self.data.resolution

    @cached_property
    def valid_date_indices(self) -> np.ndarray:
        """Return valid date indices.

        A date t is valid if we can sample the elements t + i
        for every relative_date_index i.
        """
        return get_usable_indices(
            self.data.missing,
            len(self.data),
            np.array(self.relative_date_indices, dtype=np.int64),
            self.data.trajectory_ids,
        )

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Set model and reader communication group information (called by DDPGroupStrategy).

        Parameters
        ----------
        global_rank : int
            Global rank
        model_comm_group_id : int
            Model communication group ID
        model_comm_group_rank : int
            Model communication group rank
        model_comm_num_groups : int
            Number of model communication groups
        reader_group_rank : int
            Reader group rank
        reader_group_size : int
            Reader group size
        """
        self.global_rank = global_rank
        self.model_comm_group_id = model_comm_group_id
        self.model_comm_group_rank = model_comm_group_rank
        self.model_comm_num_groups = model_comm_num_groups
        self.reader_group_rank = reader_group_rank
        self.reader_group_size = reader_group_size

        self.sample_comm_group_id = model_comm_group_id
        self.sample_comm_num_groups = model_comm_num_groups

        assert self.reader_group_size >= 1, "reader_group_size must be positive"

        LOGGER.debug(
            "NativeGridDataset.set_group_info(): global_rank %d, model_comm_group_id %d, "
            "model_comm_group_rank %d, model_comm_num_groups %d, reader_group_rank %d",
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
        )

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Called by worker_init_func on each copy of dataset.

        This initialises after the worker process has been spawned.

        Parameters
        ----------
        n_workers : int
            Number of workers
        worker_id : int
            Worker ID

        """
        self.worker_id = worker_id

        # Divide this equally across shards (one shard per group!)
        shard_size = len(self.valid_date_indices) // self.sample_comm_num_groups
        shard_start = self.sample_comm_group_id * shard_size
        shard_end = (self.sample_comm_group_id + 1) * shard_size

        shard_len = shard_end - shard_start
        self.n_samples_per_worker = shard_len // n_workers

        low = shard_start + worker_id * self.n_samples_per_worker
        high = min(shard_start + (worker_id + 1) * self.n_samples_per_worker, shard_end)
        self.chunk_index_range = np.arange(low, high, dtype=np.uint32)

        LOGGER.info(
            "Worker %d (pid %d, global_rank %d, model comm group %d)  has low/high range %d / %d",
            worker_id,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            low,
            high,
        )

        base_seed = get_base_seed()

        torch.manual_seed(base_seed)
        random.seed(base_seed)
        self.rng = np.random.default_rng(seed=base_seed)
        sanity_rnd = self.rng.random(1)

        LOGGER.info(
            (
                "Worker %d (%s, pid %d, glob. rank %d, model comm group %d, "
                "group_rank %d, seed group id %d, base_seed %d, sanity rnd %f)"
            ),
            worker_id,
            self.label,
            os.getpid(),
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            base_seed,
            sanity_rnd,
        )

    def _get_datetime_from_index(self, index_in_raw_data: int) -> datetime.datetime:
        """Converts a numerical index from `self.data.dates` to a datetime object."""
        if index_in_raw_data < 0 or index_in_raw_data >= len(self.data.dates):
            raise IndexError(
                f"Index {index_in_raw_data} out of bounds for data.dates (length {len(self.data.dates)})"
            )
        return self.data.dates[index_in_raw_data].astype(datetime.datetime)

    def get_time_slice_by_datetime(
        self,
        start_datetime: datetime.datetime,
        end_datetime: datetime.datetime,
        variables: List[str],  # List of variable names to retrieve
    ) -> Tuple[torch.Tensor, List[datetime.datetime]]:
        """
        Retrieves a slice of data for specified variables within a datetime range.
        This method is designed to be called by NativeCombinedGridDataset.

        Parameters
        ----------
        start_datetime : datetime.datetime
            The start of the datetime range (inclusive).
        end_datetime : datetime.datetime
            The end of the datetime range (inclusive).
        variables : List[str]
            List of variable names to retrieve.

        Returns
        -------
        Tuple[torch.Tensor, List[datetime.datetime]]
            A tuple containing:
            - data_tensor: torch.Tensor of shape (num_time_steps, ensemble_size, grid_points, num_variables)
            - actual_datetimes: List of datetime objects for the retrieved time steps.
        """
        # Convert datetimes to internal numerical indices for self.data (data_reader)
        # Use np.searchsorted to find the indices within `self.data.dates`
        start_idx = np.searchsorted(self.data.dates, np.datetime64(start_datetime))
        end_idx = np.searchsorted(
            self.data.dates, np.datetime64(end_datetime), side="right"
        )

        # Ensure indices are within bounds
        start_idx = max(0, start_idx)
        end_idx = min(len(self.data.dates), end_idx)

        # Get variable indices
        try:
            variable_indices = [self.name_to_index[var_name] for var_name in variables]
        except KeyError as e:
            LOGGER.error(
                f"One or more variables {variables} not found in {self.label}'s name_to_index: {e}"
            )
            raise

        # Load the data slice using the data_reader
        if start_idx >= end_idx:
            LOGGER.warning(
                f"No data found for time slice from {start_datetime} to {end_datetime} for {variables} in {self.label}."
            )
            # Return a tensor with correct dimensions but 0 for time steps
            # Assuming shape (dates, ensemble, gridpoints, variables)
            dummy_shape = (
                0,
                self.ensemble_size,
                self.data.shape[-1],
                len(variable_indices),
            )  # Assuming last dim of self.data.shape is gridpoints, and we need ensemble_size.
            # This needs to be robust if data.shape varies.
            # Correct dummy shape: num_time_steps=0, ensemble_size=self.data.shape[self.ensemble_dim], grid_points=self.data.shape[3], num_variables=len(variable_indices)
            return (
                torch.empty(
                    0,
                    self.data.shape[self.ensemble_dim],
                    self.data.shape[-1],
                    len(variable_indices),
                ),
                [],
            )

        # Load data (similar to __iter__ logic, but for a specific slice)
        # Assuming `self.data` can be sliced by time indices and variable indices
        try:
            # `self.data` expects [time, variable, ensemble, gridpoints] -> rearrange to [dates variables ensemble gridpoints]
            # `self.data` is accessed as `self.data[time_slice, :, :, variable_slice]` typically.
            # So `variable_indices` should be the last dim in the slice.

            # The original `__iter__` does `x = self.data[start:end:timeincrement, :, :, grid_shard_indices]`
            # and then `rearrange(x, "dates variables ensemble gridpoints -> dates ensemble gridpoints variables")`.
            # So variable dimension is second-to-last in raw data.

            # Correct slicing: `self.data[time_indices_slice, var_indices_slice, ensemble_slice, grid_slice]`
            # Assuming `self.data` provides [time, variables, ensemble, gridpoints] when fully loaded.

            # Here `variable_indices` would be applied to the second dimension after `time_indices_slice`.
            # `self.data[time_slice, variable_indices_slice, :, :]`

            # Let's adjust based on typical anemoi-datasets slicing:
            # `data_reader[time_slice, var_slice, ensemble_slice, grid_slice]`

            # The `__iter__` rearranges to `dates ensemble gridpoints variables`.
            # So `variable_indices` should be applied to the variable dimension.

            # First, load the data over the required time range (full variables, then select)
            data_slice_full_vars = self.data[
                start_idx : end_idx : self.time_slice_increment, :, :, :
            ]  # Load all variables, full ensemble/grid

            # Now select only the required variables
            # data_slice_full_vars is (time_steps, vars_total, ensemble, gridpoints)
            # We need to slice its `vars_total` dimension with `variable_indices`.
            data_slice_selected_vars = data_slice_full_vars[:, variable_indices, :, :]

            # Rearrange to (dates, ensemble, gridpoints, variables)
            data_slice_rearranged = rearrange(
                data_slice_selected_vars,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )

            # Get actual datetimes for the retrieved slice
            actual_datetimes_in_slice = [
                self._get_datetime_from_index(j)
                for j in range(start_idx, end_idx, self.time_slice_increment)
            ]

            return torch.from_numpy(data_slice_rearranged), actual_datetimes_in_slice
        except Exception as e:
            LOGGER.error(
                f"Error loading data slice for {self.label} from {start_datetime} to {end_datetime} for {variables}: {e}"
            )
            # If an error occurs, return empty tensors of correct final shape
            dummy_shape = (
                0,
                self.data.shape[self.ensemble_dim],
                self.data.shape[-1],
                len(variable_indices),
            )
            return torch.empty(dummy_shape), []

    def __iter__(self) -> torch.Tensor:
        """Return an iterator over the dataset.

        The datasets are retrieved by anemoi.datasets from anemoi datasets. This iterator yields
        chunked batches for DDP and sharded training.

        Currently it receives data with an ensemble dimension, which is discarded for
        now. (Until the code is "ensemble native".)
        """
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
                "model comm group %d, group_rank %d, seed comm group id %d, using indices[0:10]: %s"
            ),
            os.getpid(),
            self.label,
            self.worker_id,
            self.global_rank,
            self.model_comm_group_id,
            self.model_comm_group_rank,
            self.sample_comm_group_id,
            shuffled_chunk_indices[:10],
        )

        for i in shuffled_chunk_indices:
            start = i + self.relative_date_indices[0]
            end = i + self.relative_date_indices[-1] + 1
            timeincrement = (
                self.relative_date_indices[1] - self.relative_date_indices[0]
            )
            # NOTE: this is temporary until anemoi datasets allows indexing with arrays or lists
            # data[start...] will be replaced with data[self.relative_date_indices + i]

            grid_shard_indices = self.grid_indices.get_shard_indices(
                self.reader_group_rank
            )
            if isinstance(grid_shard_indices, slice):
                # Load only shards into CPU memory
                x = self.data[start:end:timeincrement, :, :, grid_shard_indices]

            else:
                # Load full grid in CPU memory, select grid_shard after
                # Note that anemoi-datasets currently doesn't support slicing + indexing
                # in the same operation.
                x = self.data[start:end:timeincrement, :, :, :]
                x = x[..., grid_shard_indices]  # select the grid shard
            x = rearrange(
                x,
                "dates variables ensemble gridpoints -> dates ensemble gridpoints variables",
            )
            self.ensemble_dim = 1

            yield torch.from_numpy(x)

    def __repr__(self) -> str:
        return f"""
            {super().__repr__()}
            Dataset: {self.data}
            Relative dates: {self.relative_date_indices}
        """
