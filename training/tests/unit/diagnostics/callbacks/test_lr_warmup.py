# (C) Copyright 2026- Anemoi contributors.
# Licensed under the Apache Licence, Version 2.0.

import pytest
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from anemoi.training.diagnostics.callbacks.lr_warmup import LinearLearningRateWarmup


class WarmupModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.observed_lrs = []

    def training_step(self, batch, batch_idx):
        return self.weight.square()

    def validation_step(self, batch, batch_idx):
        self.log("val_loss", torch.ones(()), on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.8, patience=0)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}}

    def on_before_optimizer_step(self, optimizer):
        # Module hook follows callback hooks, recording the LR used by this update.
        self.observed_lrs.append(optimizer.param_groups[0]["lr"])


def _trainer(tmp_path, warmup, **kwargs):
    return Trainer(
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        accumulate_grad_batches=2,
        callbacks=[warmup],
        default_root_dir=tmp_path,
        **kwargs,
    )


def _loader():
    return DataLoader(TensorDataset(torch.ones(6, 1)), batch_size=1)


def test_optimizer_step_warmup_then_plateau(tmp_path):
    model = WarmupModel()
    trainer = _trainer(tmp_path, LinearLearningRateWarmup(3), max_epochs=3)
    trainer.fit(model, _loader(), _loader())
    assert model.observed_lrs == pytest.approx([1e-3 / 3, 2e-3 / 3, 1e-3, 1e-3, 1e-3, 1e-3, 8e-4, 8e-4, 8e-4])


def test_resume_mid_warmup_restores_target_and_step(tmp_path):
    model = WarmupModel()
    trainer = _trainer(tmp_path, LinearLearningRateWarmup(3), max_steps=2)
    trainer.fit(model, _loader(), _loader())
    path = tmp_path / "warmup.ckpt"
    trainer.save_checkpoint(path)
    resumed = WarmupModel()
    callback = LinearLearningRateWarmup(3)
    trainer = _trainer(tmp_path, callback, max_steps=4)
    trainer.fit(resumed, _loader(), _loader(), ckpt_path=path)
    assert resumed.observed_lrs == pytest.approx([1e-3, 1e-3])
    assert callback.target_lrs == [1e-3]


@pytest.mark.parametrize("steps", [0, -1, True, 1.5])
def test_invalid_warmup_steps(steps):
    with pytest.raises(ValueError):
        LinearLearningRateWarmup(steps)
