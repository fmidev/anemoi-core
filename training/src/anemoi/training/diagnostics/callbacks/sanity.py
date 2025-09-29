# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import pytorch_lightning as pl

LOGGER = logging.getLogger(__name__)


class CheckVariableOrder(pl.callbacks.Callback):
    """Check the order of the variables in a pre-trained / fine-tuning model."""

    def __init__(self) -> None:
        super().__init__()

    def on_train_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the training data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        # Get data_name_to_index from data_indices (handles both single and multi-dataset cases)
        di = trainer.datamodule.data_indices
        data_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        # Handle both single IndexCollection and tuple of IndexCollections
        if isinstance(di, (list, tuple)):
            # For multi-dataset case, use the first dataset only for comparison
            di[0].compare_variables(self._model_name_to_index, data_name_to_index)
        else:
            # For single dataset case
            di.compare_variables(self._model_name_to_index, data_name_to_index)

    def on_validation_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the validation data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        di = trainer.datamodule.data_indices
        data_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        if isinstance(di, (list, tuple)):
            di[0].compare_variables(self._model_name_to_index, data_name_to_index)
        else:
            di.compare_variables(self._model_name_to_index, data_name_to_index)

    def on_test_start(self, trainer: pl.Trainer, _: pl.LightningModule) -> None:
        """Check the order of the variables in the model from checkpoint and the test data.

        Parameters
        ----------
        trainer : pl.Trainer
            Pytorch Lightning trainer
        _ : pl.LightningModule
            Not used
        """
        di = trainer.datamodule.data_indices
        data_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        if hasattr(trainer.model.module, "_ckpt_model_name_to_index"):
            self._model_name_to_index = trainer.model.module._ckpt_model_name_to_index
        else:
            self._model_name_to_index = di[0].name_to_index if isinstance(di, (list, tuple)) else di.name_to_index

        if isinstance(di, (list, tuple)):
            di[0].compare_variables(self._model_name_to_index, data_name_to_index)
        else:
            di.compare_variables(self._model_name_to_index, data_name_to_index)
