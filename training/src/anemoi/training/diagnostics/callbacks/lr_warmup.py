# (C) Copyright 2026- Anemoi contributors.
# Licensed under the Apache Licence, Version 2.0.

"""Optimizer-step warmup before validation-based plateau scheduling."""

from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from torch.optim import Optimizer


class LinearLearningRateWarmup(Callback):
    """Ramp each optimizer group to its configured LR, then leave scheduling untouched."""

    def __init__(self, warmup_steps: int = 240) -> None:
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 1:
            raise ValueError("warmup_steps must be a positive integer.")
        self.warmup_steps = warmup_steps
        self.target_lrs: list[float] | None = None

    def on_fit_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        if len(trainer.optimizers) != 1:
            raise ValueError("LinearLearningRateWarmup requires one optimizer.")
        groups = trainer.optimizers[0].param_groups
        if self.target_lrs is None:
            self.target_lrs = [group["lr"] for group in groups]
        if len(self.target_lrs) != len(groups):
            raise ValueError("Warmup checkpoint and optimizer parameter groups do not match.")

    def on_before_optimizer_step(self, trainer: Trainer, pl_module: LightningModule, optimizer: Optimizer) -> None:
        if trainer.global_step >= self.warmup_steps:
            return
        if self.target_lrs is None:
            raise RuntimeError("Warmup must be initialized before optimizer steps.")
        factor = (trainer.global_step + 1) / self.warmup_steps
        for group, target in zip(optimizer.param_groups, self.target_lrs, strict=True):
            group["lr"] = target * factor

    def state_dict(self) -> dict:
        return {"target_lrs": self.target_lrs, "warmup_steps": self.warmup_steps}

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["warmup_steps"] != self.warmup_steps:
            raise ValueError("Cannot change warmup_steps when resuming optimizer state.")
        self.target_lrs = state_dict["target_lrs"]
