# (C) Copyright 2026 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Joint codec/latent learning and independent frozen-codec refinement stages."""

from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING
from typing import Any
from typing import Self

import torch
from torch import Tensor

from anemoi.models.distributed.graph import reduce_tensor
from anemoi.training.train.methods.single import SingleTraining
from anemoi.training.train.step_output import TrainingStepOutput
from anemoi.training.utils.index_space import IndexSpace

if TYPE_CHECKING:
    from anemoi.models.distributed.shapes import ShardSizes


class LatentForecastTraining(SingleTraining):
    """Training objectives for reconstruction and latent-state prediction."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        options = self.config.training.latent
        if options.mode != "joint" and self.config.training.checkpoint is None:
            message = "Frozen-codec refinement requires an explicit source checkpoint."
            raise ValueError(message)
        if options.mode == "joint" and (options.reconstruction_weight <= 0 or options.latent_weight <= 0):
            message = "Joint training requires reconstruction and prediction supervision."
            raise ValueError(message)
        if options.mode != "decoder" and not any(k > 0 for k in self.task.loss_steps):
            message = "Latent prediction training requires positive supervised leads."
            raise ValueError(message)
        if options.mode == "decoder" and self.task.forecast_steps:
            message = "Decoder refinement must not execute forecast steps."
            raise ValueError(message)

    def on_fit_start(self) -> None:
        options = self.config.training.latent
        if options.mode == "joint":
            return
        core = self.model.model
        frozen = [
            core.encoder,
            core.encoder_graph_provider,
            core.node_attributes,
            core.latent_aggregator,
            self.model.pre_processors,
            self.model.post_processors,
        ]
        frozen += (
            [core.processor, core.processor_graph_provider, core.static_embeddings, core.temporal_embeddings]
            if options.mode == "decoder"
            else [core.decoder, core.decoder_graph_provider]
        )
        if any(parameter.requires_grad for module in frozen for parameter in module.parameters()):
            message = "Refinement requires YAML checkpoint freezing modifiers for all frozen-codec dependencies."
            raise ValueError(message)

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        from anemoi.training.checkpoint.modifiers.latent import representation_fingerprint

        super().on_save_checkpoint(checkpoint)
        checkpoint["latent_representation_id"] = representation_fingerprint(self)

    def train(self, mode: bool = True) -> Self:
        super().train(mode)
        if self.config.training.latent.mode != "joint":
            self.model.model.encoder.eval()
            self.model.model.latent_aggregator.eval()
            self.model.pre_processors.eval()
        return self

    def _latent_loss(self, predicted: Tensor, target: Tensor, shard_sizes: ShardSizes) -> Tensor:
        # Reduce the squared-error sum across the model-parallel group.
        squared = (predicted.float() - target.float()).square().sum()
        return reduce_tensor(squared, self.model_comm_group) / (sum(shard_sizes) * predicted.shape[-1])

    def latent_rollout_loss(
        self,
        initial: Tensor,
        targets: dict[int, Tensor],
        conditions: dict[int, Tensor | None],
        *,
        batch_size: int,
        shard_sizes_hidden: ShardSizes,
        capture_steps: tuple[int, ...] | list[int] = (),
    ) -> tuple[Tensor, dict[int, Tensor]]:
        """Compute the weighted prediction loss for a latent-state rollout.

        Parameters
        ----------
        initial : Tensor
            Encoded initial state.
        targets : dict[int, Tensor]
            Target latent states indexed by rollout step.
        conditions : dict[int, Tensor or None]
            Processor conditioning indexed by rollout step.
        batch_size : int
            Number of samples in the model-parallel group.
        shard_sizes_hidden : ShardSizes
            Hidden-node shard sizes for the model-parallel group.
        capture_steps : sequence of int, optional
            Steps whose predicted latent states are returned.

        Returns
        -------
        Tensor
            Weighted mean prediction loss over supervised steps.
        dict[int, Tensor]
            Predicted latent states at the requested capture steps.
        """
        current = initial
        captured = {}
        losses = []
        weights = self.config.training.latent.lead_weights
        steps = [k for k in self.task.loss_steps if k > 0]
        if len(weights) != len(steps) or any(w < 0 for w in weights) or sum(weights) <= 0:
            message = "lead_weights must match the positive supervised leads and have positive sum."
            raise ValueError(message)
        for k in range(1, self.task.forecast_steps + 1):
            current = self.model.model.advance(
                current,
                conditions[k],
                batch_size=batch_size,
                shard_sizes_hidden=shard_sizes_hidden,
                model_comm_group=self.model_comm_group,
                step=self.task._get_timestep_for_metadata(),
            )
            if k in capture_steps:
                captured[k] = current
            if k in steps:
                losses.append(self._latent_loss(current, targets[k].detach(), shard_sizes_hidden))
        return sum(w * loss for w, loss in zip(weights, losses, strict=True)) / sum(weights), captured

    def _encode_targets(self, batch: dict[str, Tensor], **args) -> dict[int, Tensor]:
        """Encode supervised truth without gradients or training-mode stochasticity."""
        core = self.model.model
        target_inputs = self.task.get_encoder_targets(batch, self.data_indices)
        target_latents = {}
        modules = [core.encoder, core.encoder_graph_provider, core.node_attributes, core.latent_aggregator]
        training_modes = [module.training for module in modules]
        for module in modules:
            module.eval()
        try:
            with torch.no_grad():
                for k in self.task.loss_steps:
                    if k > 0:
                        target_latents[k] = core.encode_snapshot(target_inputs, k, **args)[0]
        finally:
            for module, was_training in zip(modules, training_modes, strict=True):
                module.train(was_training)
        return target_latents

    def _step(self, batch: dict[str, Tensor], validation_mode: bool = False) -> TrainingStepOutput:
        core = self.model.model
        options = self.config.training.latent
        mode = options.mode
        x = self.task.get_inputs(batch, self.data_indices)
        batch_size, ensemble = next(iter(x.values())).shape[0], next(iter(x.values())).shape[2]
        args = {
            "batch_size": batch_size,
            "model_comm_group": self.model_comm_group,
            "grid_shard_sizes": self.grid_shard_sizes,
        }
        with torch.no_grad() if mode != "joint" else nullcontext():
            current, sizes = core.encode_snapshot(x, 0, **args)
        rec_loss = current.new_zeros(())
        metrics, predictions = {}, {}
        if mode != "rollout":
            decoded = core.decode_snapshot(
                current,
                x,
                0,
                ensemble_size=ensemble,
                shard_sizes_hidden=sizes,
                in_out_sharded=core._resolve_in_out_sharded(core.dataset_names, self.grid_shard_sizes),
                **args,
            )
            targets = {name: value[:, 0:1] for name, value in batch.items()}
            rec_loss, rec_metrics, predictions = self.compute_loss_metrics(
                decoded,
                targets,
                validation_mode=validation_mode,
                pred_layout=IndexSpace.MODEL_OUTPUT,
                target_layout=IndexSpace.DATA_FULL,
            )
            metrics.update(rec_metrics)
        latent_loss = current.new_zeros(())
        if mode != "decoder":
            target_latents = self._encode_targets(batch, **args)
            static = core.static_condition(x, **args)
            conditions = {
                k: core.condition(x, k, static=static, **args) for k in range(1, self.task.forecast_steps + 1)
            }
            requested = list(options.validation_decode_steps) if validation_mode else []
            if any(k < 1 or k > self.task.forecast_steps for k in requested):
                message = "Validation decode steps must lie within the forecast horizon."
                raise ValueError(message)
            latent_loss, captured = self.latent_rollout_loss(
                current,
                target_latents,
                conditions,
                batch_size=batch_size,
                shard_sizes_hidden=sizes,
                capture_steps=requested,
            )
            for k, latent in captured.items():
                decoded = core.decode_snapshot(
                    latent,
                    x,
                    k,
                    ensemble_size=ensemble,
                    shard_sizes_hidden=sizes,
                    in_out_sharded=core._resolve_in_out_sharded(core.dataset_names, self.grid_shard_sizes),
                    **args,
                )
                truth = {name: value[:, k : k + 1] for name, value in batch.items()}
                _, values, _ = self.compute_loss_metrics(
                    decoded,
                    truth,
                    validation_mode=True,
                    pred_layout=IndexSpace.MODEL_OUTPUT,
                    target_layout=IndexSpace.DATA_FULL,
                )
                metrics.update({f"forecast_step_{k}_{name}": value for name, value in values.items()})
        total = options.reconstruction_weight * rec_loss + options.latent_weight * latent_loss
        if validation_mode:
            with torch.no_grad():
                count = sum(sizes)
                mean = reduce_tensor(current.float().sum(0), self.model_comm_group) / count
                second = reduce_tensor(current.float().square().sum(0), self.model_comm_group) / count
                spread = (second - mean.square()).clamp_min(0).sqrt().mean()
            metrics.update(
                reconstruction=rec_loss.detach(),
                latent_prediction=latent_loss.detach(),
                latent_channel_std=spread,
            )
            for name in self.target_dataset_names:
                metrics[self._dataset_loss_metric_name(name)] = total.detach()
        return TrainingStepOutput(loss=total, metrics=metrics, predictions=[predictions] if predictions else [])
