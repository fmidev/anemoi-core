# anemoi/training/data/multidatamodule.py (NEW CLASS)

from __future__ import annotations

import logging
import os
from functools import cached_property
import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import numpy as np
import pytorch_lightning as pl
from hydra.utils import instantiate
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from anemoi.datasets.data import open_dataset
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.training.data.dataset import NativeGridDataset , NativeGridMultiDataset

from anemoi.training.schemas.base_schema import BaseSchema
from anemoi.training.utils.worker_init import worker_init_func
from anemoi.utils.dates import frequency_to_seconds

LOGGER = logging.getLogger(__name__)

if TYPE_CHECKING:
    from collections.abc import Callable
    from torch_geometric.data import HeteroData
    from anemoi.training.data.grid_indices import BaseGridIndices


class AnemoiDatasetsMultiDataModule(pl.LightningDataModule):
    """
    An enhanced DataModule designed to handle multiple, independent NativeGridDataset
    instances, and combine them into a single dictionary batch for training.
    This class replaces AnemoiDatasetsDataModule for multi-source data loading.
    """

    def __init__(self, config: BaseSchema, graph_data: HeteroData) -> None:
        """Initialize Anemoi Multi DataModule.

        Parameters
        ----------
        config : BaseSchema
            Job configuration (full root config).
        graph_data : HeteroData
            Graph object.
        """
        super().__init__()
        self.config = config  # This is the full root config from AnemoiTrainer
        self.graph_data = graph_data

        if self.config.dataloader.training.end is None:
            self.config.dataloader.training.end = (
                self.config.dataloader.validation.start - 1
            )
            LOGGER.info(
                "No end date specified for training data, setting default before validation start date %s.",
                self.config.dataloader.training.end,
            )

        if not self.config.dataloader.pin_memory:
            LOGGER.info("Data loader memory pinning disabled.")

        # --- Instantiation of individual NativeGridDataset objects ---

        # Dictionary to store all instantiated NativeGridDataset objects, keyed by their instance name
        self.instantiated_native_datasets: Dict[str, NativeGridDataset] = {}

        # Loop through each dataset instance definition under `dataloader.datasets` in config
        for (
            dataset_instance_name,
            dataset_instance_config,
        ) in self.config.dataloader.datasets.items():
            LOGGER.info(
                f"Setting up NativeGridDataset instance: '{dataset_instance_name}'"
            )

            # Retrieve the full data source config from `config.data` using its index
            # (e.g., config.data[0] for 6h prognostic, config.data[1] for 1h 2t)
            data_source_config = self.config.data[
                dataset_instance_config.data_multi_config_index
            ]

            # Construct the temp config for `open_dataset` using path from `dataset_instance_config`
            # and frequency/timestep/variables/processors from `data_source_config`.
            temp_open_dataset_config = OmegaConf.create(
                {
                    "path": dataset_instance_config.path,  # Path from dataloader.datasets config
                    "frequency": data_source_config.frequency,  # Frequency from data/multi.yaml
                    "timestep": data_source_config.timestep,  # Timestep from data/multi.yaml
                    "variables": data_source_config.variables,  # Variables from data/multi.yaml
                    "normalizer": data_source_config.get(
                        "normalizer"
                    ),  # Preprocessors from data/multi.yaml
                    "imputer": data_source_config.get("imputer"),
                    "remapper": data_source_config.get("remapper"),
                    "processors": data_source_config.get("processors"),
                    # Add date ranges from the main dataloader config (for filtering by stage)
                    "training": self.config.dataloader.training,
                    "validation": self.config.dataloader.validation,
                    "test": self.config.dataloader.test,
                }
            )

            data_reader_for_source = open_dataset(temp_open_dataset_config)

            # Inherited from AnemoiDatasetsDataModule, ensure this method exists (it should in the original)
            if hasattr(self, "add_trajectory_ids"):
                data_reader_for_source = self.add_trajectory_ids(data_reader_for_source)

            # Determine relative_date_indices for this specific NativeGridDataset
            rdi = self._get_relative_date_indices_for_source_type(
                source_frequency=data_source_config.frequency,
                model_timestep_frequency=self.config.data[
                    0
                ].timestep,  # Use the main (6h) model's timestep
                global_multistep_input_h=self.config.training.multistep_input
                * (frequency_to_seconds(self.config.data[0].timestep) // 3600),
                global_rollout_max_h=self.config.training.rollout.max
                * (frequency_to_seconds(self.config.data[0].timestep) // 3600),
                is_main_prognostic_source=(
                    dataset_instance_name
                    == self.config.dataloader.batch_output_spec.main_data_source_name
                ),
            )

            # Instantiate the NativeGridDataset wrapper
            self.instantiated_native_datasets[
                dataset_instance_name
            ] = NativeGridDataset(
                data_reader=data_reader_for_source,
                relative_date_indices=rdi,
                timestep=data_source_config.timestep,  # Use its own defined timestep (e.g., "1h" or "6h")
                shuffle=True,  # Will be overridden by _get_native_grid_multi_dataset for specific stages
                grid_indices=self.grid_indices,  # Re-use global grid_indices
                label=f"{dataset_instance_name}",
            )

    # --- Cached properties (statistics, data_indices, etc.) ---
    # These correctly reference the main data source by its name in `datasets`.

    @cached_property
    def statistics(self) -> dict:
        main_dataset_instance_name = (
            self.config.dataloader.batch_output_spec.main_data_source_name
        )
        return self.instantiated_native_datasets[main_dataset_instance_name].statistics

    @cached_property
    def statistics_tendencies(self) -> dict:
        main_dataset_instance_name = (
            self.config.dataloader.batch_output_spec.main_data_source_name
        )
        main_dataset_instance = self.instantiated_native_datasets[
            main_dataset_instance_name
        ]
        try:
            return main_dataset_instance.statistics_tendencies(
                main_dataset_instance.timestep
            )
        except (KeyError, AttributeError):
            return None

    @cached_property
    def metadata(self) -> dict:
        main_dataset_instance_name = (
            self.config.dataloader.batch_output_spec.main_data_source_name
        )
        return self.instantiated_native_datasets[main_dataset_instance_name].metadata()

    @cached_property
    def supporting_arrays(self) -> dict:
        main_dataset_instance_name = (
            self.config.dataloader.batch_output_spec.main_data_source_name
        )
        return (
            self.instantiated_native_datasets[
                main_dataset_instance_name
            ].supporting_arrays
            | self.grid_indices.supporting_arrays
        )

    @cached_property
    def data_indices(self) -> IndexCollection:
        # The IndexCollection should be built from the name_to_index of the main data source.
        main_dataset_instance_name = (
            self.config.dataloader.batch_output_spec.main_data_source_name
        )
        return IndexCollection(
            self.config,
            self.instantiated_native_datasets[main_dataset_instance_name].name_to_index,
        )

    # Helper for generating relative_date_indices based on source type (main vs hourly target)
    def _get_relative_date_indices_for_source_type(
        self,
        source_frequency: str,
        model_timestep_frequency: str,  # Global model timestep (e.g., 6h), from config.data[0].timestep
        global_multistep_input_h: int,  # Total history duration in hours from global config
        global_rollout_max_h: int,  # Total rollout duration in hours from global config
        is_main_prognostic_source: bool,
    ) -> List[int]:
        """
        Calculates relative date indices for a NativeGridDataset based on its role (main or hourly target).
        """
        source_freq_seconds = frequency_to_seconds(source_frequency)
        model_timestep_seconds = frequency_to_seconds(model_timestep_frequency)

        if source_freq_seconds == 0:
            LOGGER.error(
                f"Source frequency '{source_frequency}' evaluates to 0 seconds. Cannot calculate relative_date_indices."
            )
            raise ValueError("Source frequency cannot be zero.")

        if is_main_prognostic_source:
            # For the main 6h dataset, its `relative_date_indices` needs to cover:
            # - `multistep_input` history
            # - `rollout_max` future steps

            # Use `config.training.explicit_times` if it exists.
            if hasattr(self.config.training, "explicit_times"):
                return sorted(
                    set(
                        self.config.training.explicit_times.input
                        + self.config.training.explicit_times.target
                    )
                )

            # Otherwise, calculate based on multistep and rollout:
            num_input_steps = global_multistep_input_h // (source_freq_seconds // 3600)
            num_rollout_steps = global_rollout_max_h // (source_freq_seconds // 3600)

            relative_indices = []
            for step in range(num_input_steps):
                relative_indices.append(-(num_input_steps - 1 - step))
            for step in range(num_rollout_steps):
                relative_indices.append(step + 1)

            return sorted(list(set(relative_indices)))

        else:  # This is for hourly target datasets (e.g., 1h source for a 6h model's targets)
            # `relative_date_indices` for the 1h NativeGridDataset should cover the 6-hour window at 1h steps.
            steps_in_model_timestep_at_source_freq = (
                model_timestep_seconds // source_freq_seconds
            )
            return list(
                range(steps_in_model_timestep_at_source_freq + 1)
            )  # [0, 1, ..., 6] for 1h source in 6h window

    # This method creates a single NativeGridDataset instance based on a source config.
    # It's called by _get_native_grid_multi_dataset.
    def _create_native_grid_dataset_instance(
        self,
        data_config_for_stage: DictConfig,  # e.g. self.config.dataloader.training
        source_config: DictConfig,  # The specific source config from self.config.data (e.g. data[0] or data[1])
        label: str = "generic",
        shuffle: bool = True,  # Explicitly pass shuffle for each instance
        val_rollout_max_h: int = 1,  # Max rollout in hours, for RDI calculation (needed for main source RDI)
    ) -> NativeGridDataset:

        # Construct a temp config for open_dataset using the stage's date range
        temp_open_dataset_config = OmegaConf.create(
            {
                "path": data_config_for_stage.dataset,  # Path comes from dataloader.stage.dataset
                "frequency": source_config.frequency,
                "timestep": source_config.timestep,
                "variables": source_config.variables,
                "normalizer": source_config.get("normalizer"),
                "imputer": source_config.get("imputer"),
                "remapper": source_config.get("remapper"),
                "processors": source_config.get("processors"),
                "start": data_config_for_stage.start,
                "end": data_config_for_stage.end,
            }
        )

        data_reader = open_dataset(temp_open_dataset_config)
        data_reader = self.add_trajectory_ids(
            data_reader
        )  # Assuming this method is defined in this class

        # Determine relative_date_indices for this specific source/dataset
        is_main_source = (
            source_config.name
            == self.config.dataloader.batch_output_spec.main_data_source_name
        )
        rdi = self._get_relative_date_indices_for_source_type(
            source_frequency=source_config.frequency,
            model_timestep_frequency=self.config.data[
                0
            ].timestep,  # Always use the first source (main model's) for global timestep ref
            global_multistep_input_h=self.config.training.multistep_input
            * (frequency_to_seconds(self.config.data[0].timestep) // 3600),
            global_rollout_max_h=self.config.training.rollout.max
            * (frequency_to_seconds(self.config.data[0].timestep) // 3600),
            is_main_prognostic_source=is_main_source,
        )

        return NativeGridDataset(
            data_reader=data_reader,
            relative_date_indices=rdi,
            timestep=source_config.timestep,  # Pass source's own timestep to NativeGridDataset
            shuffle=shuffle,
            grid_indices=self.grid_indices,
            label=label,
        )

    # These properties now return `NativeGridMultiDataset` instances.
    @cached_property
    def ds_train(self) -> "NativeGridMultiDataset":
        return self._get_native_grid_multi_dataset(
            stage="training",
            shuffle_for_stage=True,
            label="train",
        )

    @cached_property
    def ds_valid(self) -> "NativeGridMultiDataset":
        if (
            not self.config.dataloader.training.end
            < self.config.dataloader.validation.start
        ):
            LOGGER.warning(
                "Training end date %s is not before validation start date %s.",
                self.config.dataloader.training.end,
                self.config.dataloader.validation.start,
            )
        return self._get_native_grid_multi_dataset(
            stage="validation",
            shuffle_for_stage=False,
            label="validation",
        )

    @cached_property
    def ds_test(self) -> "NativeGridMultiDataset":
        assert (
            self.config.dataloader.training.end < self.config.dataloader.test.start
        ), (
            f"Training end date {self.config.dataloader.training.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        assert (
            self.config.dataloader.validation.end < self.config.dataloader.test.start
        ), (
            f"Validation end date {self.config.dataloader.validation.end} is not before"
            f"test start date {self.config.dataloader.test.start}"
        )
        return self._get_native_grid_multi_dataset(
            stage="test",
            shuffle_for_stage=False,
            label="test",
        )

    # New helper method to create NativeGridMultiDataset for a stage
    def _get_native_grid_multi_dataset(
        self,
        stage: str,
        shuffle_for_stage: bool = True,  # Controls shuffling for the returned dataset
        label: str = "generic",
    ) -> "NativeGridMultiDataset":

        # Determine the date range config for this stage
        if stage == "training":
            date_range_config = self.config.dataloader.training
        elif stage == "validation":
            date_range_config = self.config.dataloader.validation
        else:  # test
            date_range_config = self.config.dataloader.test

        # Instantiate the main prognostic NativeGridDataset
        main_data_source_config_list_idx = (
            self.config.dataloader.batch_output_spec.main_data.data_multi_config_index
        )
        main_data_source_config = self.config.data[
            main_data_source_config_list_idx
        ]  # Full config for this source

        main_dataset_instance = self._create_native_grid_dataset_instance(
            data_config_for_stage=date_range_config,
            source_config=main_data_source_config,
            label=label,
            shuffle=shuffle_for_stage,  # Pass shuffle to individual dataset
            val_rollout_max_h=self.config.training.rollout.max
            * (
                frequency_to_seconds(self.config.data[0].timestep) // 3600
            ),  # Max rollout for RDI
        )

        # Instantiate NativeGridDataset for each hourly target source
        hourly_target_datasets_instances: Dict[str, Dict[str, Any]] = {}
        for target_output_spec in self.config.dataloader.batch_output_spec.hourly_data:
            # Get the full source config from config.data list using its index
            hourly_source_config = self.config.data[
                target_output_spec.data_source_index
            ]

            hourly_dataset_instance = self._create_native_grid_dataset_instance(
                data_config_for_stage=date_range_config,
                source_config=hourly_source_config,
                label=f"{label}_{target_output_spec.name}",
                shuffle=shuffle_for_stage,  # Pass shuffle to individual dataset
                val_rollout_max_h=self.config.training.rollout.max
                * (
                    frequency_to_seconds(self.config.data[0].timestep) // 3600
                ),  # Max rollout for RDI
            )

            hourly_target_datasets_instances[target_output_spec.name] = {
                "dataset": hourly_dataset_instance,
                "variables": target_output_spec.variables,  # Variables specified in batch_output_spec.hourly_data
                "num_intermediate_hours": target_output_spec.num_intermediate_hours,
            }

        # Instantiate our new combined dataset
        return NativeGridMultiDataset(
            main_dataset=main_dataset_instance,
            hourly_target_datasets_spec=hourly_target_datasets_instances,
            main_data_indices=self.data_indices,  # The global IndexCollection (from main source)
            main_data_source_config=main_data_source_config,  # Pass the 6h config (data[0])
            global_model_timestep_h=frequency_to_seconds(self.config.data[0].timestep)
            // 3600,
        )

    # --- Keep existing _get_dataloader, train_dataloader, val_dataloader, test_dataloader ---
    def _get_dataloader(self, ds: "NativeGridMultiDataset", stage: str) -> DataLoader:
        assert stage in {"training", "validation", "test"}
        return DataLoader(
            ds,
            batch_size=self.config.dataloader.batch_size[stage],
            num_workers=self.config.dataloader.num_workers[stage],
            shuffle=False,  # Shuffle handled by IterableDataset internally
            pin_memory=self.config.dataloader.pin_memory,
            worker_init_fn=worker_init_func,
            prefetch_factor=self.config.dataloader.prefetch_factor,
            persistent_workers=True,
        )

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_train, "training")

    def val_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_valid, "validation")

    def test_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.ds_test, "test")
