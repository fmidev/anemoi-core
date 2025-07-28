# In anemoi/training/data/multidataset.py

import logging
import torch
import datetime
from torch.utils.data import IterableDataset
from typing import Dict, Any, List, Tuple, Generator, Union
import numpy as np
import math

# Re-import necessary Anemoi components (ensure all these are available in this file or imported)
from anemoi.models.data_indices.collection import IndexCollection
from anemoi.utils.dates import frequency_to_seconds
from anemoi.training.data.dataset.singledataset import NativeGridDataset


LOGGER = logging.getLogger(__name__)


class NativeGridMultiDataset(IterableDataset):
    """
    Combines data from multiple NativeGridDataset instances (e.g., 6h main data, 1h 2t data)
    into a single dictionary batch for heterogeneous resolution training.
    """

    def __init__(
        self,
        main_dataset: NativeGridDataset,  # The instantiated 6h dataset (primary source)
        hourly_target_datasets_spec: Dict[
            str, Dict[str, Any]
        ],  # {'name': {'dataset': NativeGridDataset, 'variables': [...], 'num_intermediate_hours': int}}
        main_data_indices: IndexCollection,  # The global IndexCollection from main model
        model_multistep_input_h: int,  # Main model's input history in hours
        model_rollout_max_h: int,  # Main model's max rollout in hours
        global_model_timestep_h: int,  # Main model's timestep in hours (e.g., 6h)
        global_raw_data_frequency_h: int,  # Base raw data frequency in hours (e.g., 1h, from data.frequency)
        # ... other args if needed from datamodule config
    ) -> None:
        super().__init__()
        self.main_dataset = main_dataset
        self.hourly_target_datasets_spec = hourly_target_datasets_spec
        self.main_data_indices = main_data_indices
        self.model_multistep_input_h = model_multistep_input_h
        self.model_rollout_max_h = model_rollout_max_h
        self.global_model_timestep_h = global_model_timestep_h
        self.global_raw_data_frequency_h = global_raw_data_frequency_h

        # Validate that hourly target variables are in main_data_indices
        for target_spec in self.hourly_target_datasets_spec.values():
            for var_name in target_spec["variables"]:
                if var_name not in self.main_data_indices.name_to_index:
                    LOGGER.error(
                        f"Hourly target variable '{var_name}' not found in main_data_indices. Check data config."
                    )
                    raise KeyError(
                        f"Missing hourly target variable in main_data_indices: {var_name}"
                    )

        self._len = len(
            self.main_dataset
        )

    def __len__(self) -> int:
        return self._len

    def per_worker_init(self, n_workers: int, worker_id: int) -> None:
        """Propagates worker initialization to all wrapped datasets."""
        self.main_dataset.per_worker_init(n_workers, worker_id)
        for spec in self.hourly_target_datasets_spec.values():
            spec["dataset"].per_worker_init(n_workers, worker_id)

    def set_comm_group_info(
        self,
        global_rank: int,
        model_comm_group_id: int,
        model_comm_group_rank: int,
        model_comm_num_groups: int,
        reader_group_rank: int,
        reader_group_size: int,
    ) -> None:
        """Propagates communication group info to all wrapped datasets."""
        self.main_dataset.set_comm_group_info(
            global_rank,
            model_comm_group_id,
            model_comm_group_rank,
            model_comm_num_groups,
            reader_group_rank,
            reader_group_size,
        )
        for spec in self.hourly_target_datasets_spec.values():
            spec["dataset"].set_comm_group_info(
                global_rank,
                model_comm_group_id,
                model_comm_group_rank,
                model_comm_num_groups,
                reader_group_rank,
                reader_group_size,
            )

    def __iter__(self) -> Generator[Dict[str, torch.Tensor], None, None]:
        """
        Yields a dictionary batch containing main model data and hourly ground truths.
        """
        # Iterate over the main 6h dataset's iterator
        for main_x_chunk, current_6h_start_datetime in self.main_dataset:
            # main_x_chunk is (num_time_steps_in_chunk, ensemble, gridpoints, variables)
            # current_6h_start_datetime is the datetime for the first valid date index (t0) of this chunk.

            # Determine main model's input steps from its config in hours, then convert to steps of its own timestep
            # e.g., if model_multistep_input_h=12 (hours) and global_model_timestep_h=6 (hours), then 2 steps.
            main_model_input_steps = (
                self.model_multistep_input_h // self.global_model_timestep_h
            )

            # Extract `main_x` (input for model proper) from the chunk:
            # It's the history portion: [0 : main_model_input_steps]
            main_x = main_x_chunk[
                0:main_model_input_steps, :, :, self.main_data_indices.data.input.full
            ]

            # Extract `main_y` (6h ground truth for model proper) from the chunk:
            # It's the next step's prediction: [main_model_input_steps]
            main_y = main_x_chunk[
                main_model_input_steps, :, :, self.main_data_indices.data.output.full
            ]

            current_sample_output: Dict[str, torch.Tensor] = {
                "main_x": main_x,
                "main_y": main_y,
                "main_start_datetime": torch.tensor(
                    current_6h_start_datetime.timestamp(),
                    dtype=torch.float64,
                    device=main_x.device,
                ),  # Pass datetime as timestamp
            }

            # Fetch hourly target data for each specified hourly source
            for (
                target_spec_name,
                target_spec,
            ) in self.hourly_target_datasets_spec.items():
                hourly_source_dataset_instance = target_spec["dataset"]
                target_variables = target_spec["variables"]
                num_intermediate_hours = target_spec["num_intermediate_hours"]

                # We need hourly data from `current_6h_start_datetime + 1h` up to 
                # `current_6h_start_datetime + num_intermediate_hours hours`.

                # This corresponds to the range (+1h to +5h) within the 6h forecast window.

                # The `get_time_slice_by_datetime` method in NativeGridDataset needs inclusive end.
                # So for 5 intermediate hours (e.g. +1h, ..., +5h), the range is from `+1h` to `+5h`.
                # Example: num_intermediate_hours=5, so from +1h to +5h.
                # start_datetime = current_6h_start_datetime + 1 hour
                # end_datetime = current_6h_start_datetime + num_intermediate_hours hours

                hourly_fetch_start_dt = current_6h_start_datetime + datetime.timedelta(
                    hours=1
                )
                # `num_intermediate_hours` is how many *intermediate* steps.
                # So if `num_intermediate_hours=5`, we want t+1h, t+2h, t+3h, t+4h, t+5h.
                # This means the last datetime is `current_6h_start_datetime + 5h`.
                hourly_fetch_end_dt = current_6h_start_datetime + datetime.timedelta(
                    hours=num_intermediate_hours
                )

                try:
                    # `get_time_slice_by_datetime` returns: (data_tensor, actual_datetimes_list)
                    (
                        hourly_targets_data_chunk,
                        actual_retrieved_datetimes,
                    ) = hourly_source_dataset_instance.get_time_slice_by_datetime(
                        start_datetime=hourly_fetch_start_dt,
                        end_datetime=hourly_fetch_end_dt,
                        variables=target_variables,
                    )

                    # Verify we got the expected number of time steps (num_intermediate_hours)
                    if hourly_targets_data_chunk.shape[0] != num_intermediate_hours:
                        LOGGER.warning(
                            f"Retrieved {hourly_targets_data_chunk.shape[0]} hourly steps for {target_spec_name} "
                            f"from {hourly_fetch_start_dt} to {hourly_fetch_end_dt}, "
                            f"but expected {num_intermediate_hours}. Skipping this hourly target for this sample."
                        )
                        continue  # Skip this hourly target if data is not complete for the window

                    # Generate time_fractions for these intermediate steps: [1/6, 2/6, ..., 5/6]
                    hourly_fractions = torch.tensor(
                        [
                            float(j + 1) / self.global_model_timestep_h
                            for j in range(num_intermediate_hours)
                        ],
                        dtype=hourly_targets_data_chunk.dtype,
                        device=hourly_targets_data_chunk.device,
                    ).unsqueeze(
                        -1
                    )  # Shape (num_intermediate_hours, 1)

                    current_sample_output[
                        f"{target_spec_name}_gt"
                    ] = hourly_targets_data_chunk
                    current_sample_output[
                        f"{target_spec_name}_time_fractions"
                    ] = hourly_fractions

                except Exception as e:
                    LOGGER.warning(
                        f"Failed to retrieve hourly data for {target_spec_name} for window starting {current_6h_start_datetime}: {e}. Skipping this hourly target."
                    )
                    # In production, might log, but not re-raise or skip entire batch
                    continue

            yield current_sample_output
