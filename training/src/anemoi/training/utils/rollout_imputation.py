# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Keep imputed inputs imputed through a training rollout.

An imputer fills missing input values (NaN) before the model sees them - for instance a
variable that exists only over a limited area, filled with a constant outside it. At
inference the post-processor puts the NaN back and the next step's pre-processor fills it
again, so those positions receive the FILL VALUE at every step. In a training rollout the
prognostic inputs are overwritten with the model's predictions everywhere instead, so the
same positions receive the model's own, unsupervised, output - a train/inference mismatch.

`imputed_input_mask` returns where the imputers filled, in model-input layout, so the rollout
can restore the fill value there (`training.reimpute_rollout_inputs`).
"""

from __future__ import annotations

import torch

from anemoi.models.preprocessing.imputer import BaseImputer


def imputed_input_mask(pre_processors, why: list | None = None) -> torch.Tensor | None:
    """(See module docstring.) ``why``, if given, collects the reason whenever None is returned."""
    """Boolean (batch, grid, model-input variable) mask of imputed positions, or None.

    Reads the NaN locations each imputer recorded during its last TRAINING-shaped transform
    (the batch just pre-processed), restricted to the variables that imputer fills. Returns
    None if there is no imputer or nothing was imputed.
    """
    why = [] if why is None else why
    masks = []
    for key, processor in getattr(pre_processors, "processors", {}).items():
        if not isinstance(processor, BaseImputer):
            why.append(f"{key}: not an imputer ({type(processor).__name__})")
            continue
        nan_locations = processor.nan_locations
        if nan_locations.ndim != 3 or not bool(nan_locations.any()):
            why.append(f"{key}: nan_locations {tuple(nan_locations.shape)} sum {int(nan_locations.sum())}")
            continue
        input_full = [int(i) for i in processor.data_indices.data.input.full.tolist()]
        position = {data_index: i for i, data_index in enumerate(input_full)}
        columns = torch.zeros(nan_locations.shape[-1], dtype=torch.bool, device=nan_locations.device)
        for data_index in processor.index_training_input:
            if int(data_index) in position:
                columns[position[int(data_index)]] = True
        if not bool(columns.any()):
            why.append(f"{key}: imputed indices {list(processor.index_training_input)} not among inputs")
        masks.append(nan_locations & columns)
    if not masks:
        why.append("no imputer with recorded NaN")
        return None
    mask = torch.stack(masks).any(dim=0)
    if not bool(mask.any()):
        why.append("recorded NaN not at imputed variables")
        return None
    return mask
