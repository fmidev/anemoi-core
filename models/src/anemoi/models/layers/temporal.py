# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Point-wise layers for latent-state conditioning."""

import torch
from anemoi.utils.config import DotDict
from torch import nn

from anemoi.models.layers.mlp import MLP, MLPImplementation
from anemoi.models.layers.utils import compute_mlp_hidden_dim, load_layer_kernels


class LatentStateContextMixer(nn.Module):
    """Fuse one or two persistent latent states with target-time context point-wise."""

    def __init__(
        self,
        *,
        num_channels: int,
        num_state_inputs: int = 2,
        context_channels: int | None = None,
        mlp_hidden_ratio: float,
        layer_kernels: DotDict,
        n_extra_layers: int = 0,
        final_activation: bool = False,
        layer_norm: bool = True,
        mlp_implementation: MLPImplementation = "mlp",
    ) -> None:
        super().__init__()
        if num_state_inputs not in (1, 2):
            raise ValueError(f"num_state_inputs must be 1 or 2, got {num_state_inputs}.")
        self.num_state_inputs = num_state_inputs
        layer_factory = load_layer_kernels(layer_kernels)
        if context_channels is None:
            context_channels = num_channels
        self.mlp = MLP(
            in_features=num_state_inputs * num_channels + context_channels,
            hidden_dim=compute_mlp_hidden_dim(num_channels, mlp_hidden_ratio),
            out_features=num_channels,
            n_extra_layers=n_extra_layers,
            final_activation=final_activation,
            layer_norm=layer_norm,
            mlp_implementation=mlp_implementation,
            layer_kernels=layer_factory,
        )

    def forward(
        self,
        previous: torch.Tensor | None,
        current: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return a channel-preserving fusion on each local hidden node."""
        if current.shape[:-1] != target_context.shape[:-1]:
            raise ValueError(
                "LatentStateContextMixer inputs must have matching node dimensions; "
                f"got current={tuple(current.shape)}, target_context={tuple(target_context.shape)}."
            )
        if self.num_state_inputs == 1:
            if previous is not None:
                raise ValueError("A one-state LatentStateContextMixer expects previous=None.")
            inputs = (current, target_context)
        else:
            if previous is None or previous.shape != current.shape:
                previous_shape = None if previous is None else tuple(previous.shape)
                raise ValueError(
                    "A two-state LatentStateContextMixer requires equal previous and current shapes; "
                    f"got previous={previous_shape}, current={tuple(current.shape)}."
                )
            inputs = (previous, current, target_context)
        return self.mlp(torch.cat(inputs, dim=-1))
