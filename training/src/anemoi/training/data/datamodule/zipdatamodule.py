# src/anemoi/training/data/datamodule/zipdatamodule.py
from __future__ import annotations
import logging
from functools import cached_property
from typing import TYPE_CHECKING, Tuple, Callable

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra.utils import instantiate

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset, ZipDataset
from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.data.datamodule.singledatamodule import (
    AnemoiDatasetsDataModule as UpstreamDataModule,
)
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData
    from anemoi.training.data.grid_indices import BaseGridIndices

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsZipDataModule(UpstreamDataModule):
    """Zip loader combining multiple input streams into one Lightning DataModule."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        # Initialize base with config and graph for grid indices, splits, etc.
        super().__init__(config, graph_data)
        # Capture zip-specific sections
        self.forcing_configs = config.dataloader.zip.forcing
        self.diagnostic_config = config.dataloader.zip.diagnostic
        self.adjust = config.dataloader.zip.adjust

    @cached_property
    def ds_train(self) -> ZipDataset:
        # Build two readers: forcing & diagnostic
        readers = []
        for fc in self.forcing_configs:
            readers.append(open_dataset(fc))
        readers.append(open_dataset(self.diagnostic_config))
        return self._get_dataset(readers, label="train")

    @cached_property
    def ds_valid(self) -> ZipDataset:
        readers = []
        for fc in self.forcing_configs:
            readers.append(open_dataset(fc))
        readers.append(open_dataset(self.diagnostic_config))
        return self._get_dataset(
            readers,
            shuffle=False,
            rollout=self.config.dataloader.validation_rollout,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> ZipDataset:
        readers = []
        for fc in self.forcing_configs:
            readers.append(open_dataset(fc))
        readers.append(open_dataset(self.diagnostic_config))
        return self._get_dataset(readers, shuffle=False, label="test")

    def _get_dataset(
        self,
        data_readers: list[Callable],
        shuffle: bool = True,
        rollout: int = 1,
        label: str = "generic",
    ) -> ZipDataset:
        # Use max rollout logic from base
        r = max(rollout, self.config.training.rollout.max)
        # Instantiate ZipDataset with same args as NativeGridDataset
        ds = ZipDataset(
            data_readers=data_readers,
            rollout=r,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            model_comm_group_rank=self.model_comm_group_rank,
            model_comm_group_id=self.model_comm_group_id,
            model_comm_num_groups=self.model_comm_num_groups,
            shuffle=shuffle,
            label=label,
        )
        # Optionally align start/end times
        ds = ds.adjust(self.adjust)
        self._check_resolution(ds.resolution)
        return ds

    @cached_property
    def data_indices(self) -> Tuple[IndexCollection, ...]:
        # Build one IndexCollection per stream
        return tuple(
            IndexCollection(self.config, ds.name_to_index) for ds in (self.ds_train,)
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
