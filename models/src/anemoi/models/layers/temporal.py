# (C) Copyright 2026- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

"""Point-wise layers for latent temporal evolution."""

import torch
from anemoi.utils.config import DotDict
from torch import nn

from anemoi.models.layers.mlp import MLP, MLPImplementation
from anemoi.models.layers.utils import compute_mlp_hidden_dim, load_layer_kernels


class LatentTemporalMixer(nn.Module):
    """Fuse two persistent latent states and target-time context point-wise."""

    def __init__(
        self,
        *,
        num_channels: int,
        context_channels: int | None = None,
        mlp_hidden_ratio: float,
        layer_kernels: DotDict,
        n_extra_layers: int = 0,
        final_activation: bool = False,
        layer_norm: bool = True,
        mlp_implementation: MLPImplementation = "mlp",
    ) -> None:
        super().__init__()
        layer_factory = load_layer_kernels(layer_kernels)
        if context_channels is None:
            context_channels = num_channels
        self.mlp = MLP(
            in_features=2 * num_channels + context_channels,
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
        previous: torch.Tensor,
        current: torch.Tensor,
        target_context: torch.Tensor,
    ) -> torch.Tensor:
        """Return a channel-preserving fusion on each local hidden node."""
        if previous.shape != current.shape or current.shape[:-1] != target_context.shape[:-1]:
            raise ValueError(
                "LatentTemporalMixer inputs must have matching node dimensions and equal state shapes; "
                f"got previous={tuple(previous.shape)}, current={tuple(current.shape)}, "
                f"target_context={tuple(target_context.shape)}."
            )
        return self.mlp(torch.cat((previous, current, target_context), dim=-1))
