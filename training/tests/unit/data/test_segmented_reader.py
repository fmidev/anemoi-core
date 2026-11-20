# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from datetime import timedelta
from typing import ClassVar

import numpy as np
import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from anemoi.training.data.multidataset import MultiDataset
from anemoi.training.data.readers import SegmentedGriddedDataReader
from anemoi.training.data.readers import create_reader
from anemoi.training.schemas.dataloader import DatasetReaderSchema


class _Dataset:
    dates = np.datetime64("2026-01-01T01") + np.arange(12) * np.timedelta64(1, "h")
    frequency = timedelta(hours=1)
    missing: ClassVar[set[int]] = {6}
    data = np.arange(48).reshape(12, 1, 1, 4)
    shape = data.shape

    def __getitem__(self, index: tuple) -> np.ndarray:
        return self.data[index]


@pytest.fixture
def dataset(monkeypatch: pytest.MonkeyPatch) -> _Dataset:
    dataset = _Dataset()
    monkeypatch.setattr("anemoi.training.data.readers.base.open_dataset", lambda _: dataset)
    return dataset


def test_segmented_reader_keeps_absolute_phase_and_partial_segments(dataset: _Dataset) -> None:
    reader = create_reader(
        OmegaConf.create(
            {
                "dataset_config": "example",
                "segmented": {"segment_length": 5, "reference_date": "2026-01-01T00"},
            },
        ),
    )
    assert isinstance(reader, SegmentedGriddedDataReader)

    anchors = reader.valid_anchors([-1, 0, 1])
    # Segments cover global indices [0,4), [4,9), [9,12). The missing date at 6
    # invalidates all full three-step windows in the middle segment.
    np.testing.assert_array_equal(anchors.times, dataset.dates[[1, 2, 10]])
    np.testing.assert_array_equal(anchors.sequences, [0, 0, 2])
    np.testing.assert_array_equal(anchors.positions, [1, 2, 1])

    reader.set_reader_group_info(reader_group_rank=1, reader_group_size=2)
    np.testing.assert_array_equal(
        reader.get_data(2, slice(0, 3)).numpy(),
        dataset.data[9:12, :, :, 2:].transpose(0, 2, 3, 1),
    )
    np.testing.assert_array_equal(
        reader.get_data(0, [0, 2]).numpy(),
        dataset.data[[0, 2], :, :, 2:].transpose(0, 2, 3, 1),
    )
    with pytest.raises(IndexError):
        reader.get_data(0, slice(2, 5))


def test_segmented_reader_stride_restarts_at_segment_boundaries(dataset: _Dataset) -> None:
    reader = SegmentedGriddedDataReader(
        "example",
        segment_length=5,
        reference_date="2026-01-01T00",
        sampling={"stride": 3},
    )
    other = create_reader({"dataset_config": "example"})
    multidataset = MultiDataset({"stitched": reader, "native": other}, {"stitched": [0], "native": [0]})
    np.testing.assert_array_equal(multidataset.anchors.times, dataset.dates[[0, 3, 4, 7, 9]])
    np.testing.assert_array_equal(multidataset.anchors.rows["stitched"], [[0, 0], [0, 3], [1, 0], [1, 3], [2, 0]])
    assert multidataset.anchors.base_dates is None


def test_segmented_and_trajectory_are_mutually_exclusive() -> None:
    config = {
        "dataset_config": "example",
        "segmented": {"segment_length": 5, "reference_date": "2026-01-01T00"},
        "trajectory": {},
    }
    with pytest.raises(ValidationError, match="not both"):
        DatasetReaderSchema.model_validate(config)
    with pytest.raises(ValueError, match="not both"):
        create_reader(config)
