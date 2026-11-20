# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import datetime as dt
from numbers import Integral

import numpy as np
import torch
from rich.tree import Tree

from anemoi.training.data.readers.gridded import GriddedDataReader
from anemoi.training.data.usable_indices import ReaderAnchors
from anemoi.training.data.usable_indices import get_usable_indices
from anemoi.training.utils.time_indices import TimeIndices


class SegmentedGriddedDataReader(GriddedDataReader):
    """Read a regular valid-time series stitched from fixed-length forecast segments.

    Boundaries are anchored at ``reference_date + k * segment_length * frequency``.
    Subsetting keeps any partial segments without changing where the boundaries fall.
    The segments have unique valid times, so alignment with other readers is by valid time.
    """

    def __init__(
        self,
        dataset_config: str | dict,
        start: dt.datetime | int | None = None,
        end: dt.datetime | int | None = None,
        *,
        segment_length: int,
        reference_date: str | dt.datetime,
        sampling: dict | None = None,
    ) -> None:
        if isinstance(segment_length, bool) or not isinstance(segment_length, Integral) or segment_length < 1:
            msg = f"segmented.segment_length must be a positive number of steps, got {segment_length}."
            raise ValueError(msg)
        super().__init__(dataset_config, start=start, end=end)
        self.segment_length = segment_length
        self.reference_date = np.datetime64(reference_date, "s")
        self.sampling = sampling

        dates = np.asarray(self.dates).astype("datetime64[s]")
        step = np.timedelta64(int(self.frequency.total_seconds()), "s")
        if not dates.size or step <= np.timedelta64(0, "s") or np.any(np.diff(dates) != step):
            msg = "Segmented dataset requires a non-empty, regular date axis."
            raise ValueError(msg)
        distance = dates[0] - self.reference_date
        if np.isnat(distance) or distance % step != np.timedelta64(0, "s"):
            msg = f"reference_date {self.reference_date} is not on the dataset date grid ({dates[0]}, {step})."
            raise ValueError(msg)
        offset = int(-(distance // step) % segment_length)
        starts = np.arange(offset, dates.size, segment_length, dtype=np.int64)
        self._segment_starts = np.r_[0, starts] if offset else starts
        self._segment_ends = np.r_[self._segment_starts[1:], dates.size]

    def valid_anchors(self, relative_indices: list[int] | np.ndarray) -> ReaderAnchors:
        """Return only anchors whose requested offsets stay within a single segment."""
        dates = np.asarray(self.dates)
        positions = get_usable_indices(self.missing, len(dates), relative_indices)
        sequences = np.searchsorted(self._segment_starts, positions, side="right") - 1
        starts, ends = self._segment_starts[sequences], self._segment_ends[sequences]
        keep = (positions + np.min(relative_indices) >= starts) & (positions + np.max(relative_indices) < ends)
        return ReaderAnchors(
            times=dates[positions[keep]],
            sequences=sequences[keep],
            positions=(positions - starts)[keep],
            stride_by_sequence=True,
        )

    def get_data(self, sequence: int, positions: TimeIndices) -> torch.Tensor:
        """Translate segment-local positions and use the usual gridded read and sharding."""
        start = int(self._segment_starts[sequence])
        length = int(self._segment_ends[sequence]) - start
        if isinstance(positions, slice):
            if any(bound is not None and not 0 <= bound <= length for bound in (positions.start, positions.stop)):
                raise IndexError(positions)
            if positions.step is not None and positions.step < 0:
                indices = range(*positions.indices(length))
                positions = [start + index for index in indices]
            else:
                first, stop, step = positions.indices(length)
                positions = slice(start + first, start + stop, step)
        else:
            indices = np.asarray(positions, dtype=np.int64)
            if np.any((indices < 0) | (indices >= length)):
                raise IndexError(positions)
            positions = (start + np.atleast_1d(indices)).tolist()
        return super().get_data(sequence, positions)

    def tree(self, prefix: str = "") -> Tree:
        tree = super().tree(prefix)
        tree.add(
            f"Segments: {len(self._segment_starts)} (length {self.segment_length}, reference {self.reference_date})",
        )
        tree.add(f"Sampling: {self.sampling}")
        return tree
