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
import warnings
from functools import cached_property
from typing import Callable

# Suppress various warnings
warnings.filterwarnings("ignore", message="The behaviour of Zip.collect_supporting_arrays\\(\\) is not well defined")

from omegaconf import DictConfig
from omegaconf import OmegaConf
from torch_geometric.data import HeteroData

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import ZipDataset
from anemoi.training.data.dataset import NativeGridDataset
from anemoi.training.schemas.base_schema import BaseSchema

from .singledatamodule import AnemoiDatasetsDataModule

LOGGER = logging.getLogger(__name__)


class AnemoiDatasetsZipModule(AnemoiDatasetsDataModule):
    """Data module for multiple datasets zipped together."""

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        super().__init__(config, graph_data)
        
        # Set the maximum rollout to be expected
        self.rollout = (
            self.config.training.rollout.max
            if self.config.training.rollout.epoch_increment > 0
            else self.config.training.rollout.start
        )

    @cached_property
    def supporting_arrays(self) -> dict:
        return self.ds_train.supporting_arrays | self.grid_indices.supporting_arrays

    @cached_property
    def statistics(self) -> dict:
        """Return statistics as a dictionary with integer keys for each dataset in the zip."""
        # For zip datasets, we need to return statistics for each dataset
        if hasattr(self.ds_train, 'statistics'):
            # If ds_train.statistics is already a dict with integer keys, return it
            if isinstance(self.ds_train.statistics, dict) and all(isinstance(k, int) for k in self.ds_train.statistics.keys()):
                return self.ds_train.statistics
            # Otherwise, wrap single statistics in dict
            return {0: self.ds_train.statistics}
        return {}

    @cached_property
    def data_indices(self) -> tuple[IndexCollection, ...]:
        """Return data indices as a tuple of IndexCollection objects for each dataset in the zip."""
        return self._zip_index_collection(self.config, self.ds_train.name_to_index)

    def _get_dataset(
        self,
        data_reader: Callable,
        shuffle: bool = True,
        val_rollout: int = 1,
        label: str = "generic",
    ) -> ZipDataset:
        r = max(val_rollout, self.rollout)
        data = ZipDataset(
            data_reader=data_reader,
            rollout=r,
            multistep=self.config.training.multistep_input,
            timeincrement=self.timeincrement,
            model_comm_group_rank=0,
            model_comm_group_id=0,
            model_comm_num_groups=1,
            shuffle=shuffle,
            label=label,
        )

        return data

    @staticmethod
    def _zip_index_collection(config: DictConfig, name_to_index: tuple) -> tuple[IndexCollection, ...]:
        zip_return = ()
        for dset_index, dset_config in enumerate(config.data.zip):
            # Create a deep copy of the config and resolve it
            temp_config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
            temp_config.data = dset_config
            temp_config.data.frequency = config.data.frequency
            temp_config.data.timestep = config.data.timestep
            
            # Ensure name_to_index[dset_index] is a dictionary
            name_to_index_dict = name_to_index[dset_index]
            if not isinstance(name_to_index_dict, dict):
                raise TypeError(f"name_to_index[{dset_index}] must be a dict, got {type(name_to_index_dict)}")
            
            zip_return += (IndexCollection(temp_config, name_to_index_dict),)
        return zip_return
