# (C) Copyright 2024-2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anemoi.training.data.data_reader import BaseAnemoiReader

LOGGER = logging.getLogger(__name__)


def _intersect_anchor_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the ``(sequence, position)`` rows present in both anchor arrays.

    Parameters
    ----------
    a, b : np.ndarray
        Arrays of shape ``(n, 2)`` of ``(sequence, position)`` anchors.

    Returns
    -------
    np.ndarray
        Array of shape ``(m, 2)`` with the common anchors, sorted.
    """
    if a.size == 0 or b.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    dtype = np.dtype((np.void, a.dtype.itemsize * a.shape[1]))
    common = np.intersect1d(
        np.ascontiguousarray(a).view(dtype),
        np.ascontiguousarray(b).view(dtype),
    )
    rows = common.view(a.dtype).reshape(-1, 2)
    # intersect1d sorts the void views byte-wise (little-endian), which is not
    # numeric order for multi-byte values; restore (sequence, position) order.
    return rows[np.lexsort((rows[:, 1], rows[:, 0]))]


def compute_valid_anchors(
    data_readers: dict[str, "BaseAnemoiReader"],
    relative_date_indices: dict[str, np.ndarray | list[int]],
) -> np.ndarray:
    """Return the valid ``(sequence, anchor_key)`` anchors shared by all readers.

    An anchor is valid if every reader can sample all of its relative offsets
    around the anchor date within the anchor's sequence. Each reader's local
    positions are canonicalized to reader-independent anchor keys
    (``BaseAnemoiReader.positions_to_anchor_keys``) before intersecting, so
    readers with different frequencies or start dates intersect by date rather
    than by raw index.

    Parameters
    ----------
    data_readers : dict[str, BaseAnemoiReader]
        Mapping of dataset name to data reader.
    relative_date_indices : dict[str, np.ndarray | list[int]]
        Relative offsets (in positions) requested for each reader.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_anchors, 2)`` with the shared
        ``(sequence, anchor_key)`` anchors.
    """
    intersection: np.ndarray | None = None
    for dataset_name, ds in data_readers.items():
        anchors = ds.compute_anchors(relative_date_indices[dataset_name])

        if len(anchors) == 0:
            msg = f"No valid anchors found for data reader '{dataset_name}': {ds}"
            raise ValueError(msg)

        # Canonicalize local positions to reader-independent anchor keys.
        anchors = np.stack(
            [anchors[:, 0], ds.positions_to_anchor_keys(anchors[:, 1])],
            axis=1,
        )

        intersection = anchors if intersection is None else _intersect_anchor_rows(intersection, anchors)

        LOGGER.info("Data reader '%s' has %d valid anchors", dataset_name, len(anchors))

    if intersection is None or len(intersection) == 0:
        msg = "No valid anchors found after intersection across all datasets."
        raise ValueError(msg)

    LOGGER.info("MultiDataset has %d valid anchors after intersection.", len(intersection))

    return intersection


def get_usable_indices(
    missing_indices: set[int],
    series_length: int,
    relative_indices: np.ndarray | list[int],
) -> np.ndarray:
    """Get the usable indices of a series with missing indices.

    Parameters
    ----------
    missing_indices : set[int]
        Set of missing indices in the series.
    series_length : int
        Length of the series.
    relative_indices: np.ndarray | list[int]
        Array of relative indices requested at each index i.

    Returns
    -------
    usable_indices : np.array
        Array of usable indices.
    """
    if isinstance(relative_indices, list):
        relative_indices = np.array(relative_indices)

    usable_indices = np.arange(series_length)

    # Restrict to indices where all relative positions are within bounds
    max_offset = int(max(relative_indices))
    min_offset = int(min(relative_indices))
    usable_indices = usable_indices[(usable_indices + min_offset >= 0) & (usable_indices + max_offset < series_length)]

    # Missing indices
    for i in missing_indices:
        rel_missing = i - relative_indices  # indices which have their relative indices match the missing.
        usable_indices = usable_indices[np.all(usable_indices != rel_missing[:, np.newaxis], axis=0)]

    return usable_indices
