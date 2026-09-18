import torch

from anemoi.training.losses import MixtureCrossEntropyLoss
from anemoi.training.utils.enums import TensorDim


def test_mixture_ce_does_not_crash_with_scalers() -> None:
    bs, t, ens, grid, c = 2, 3, 4, 5, 6
    loss = MixtureCrossEntropyLoss()
    loss.add_scaler(TensorDim.GRID, torch.ones(grid), name="grid")
    loss.add_scaler(TensorDim.VARIABLE, torch.ones(c), name="var")

    pred = torch.ones((bs, t, ens, grid, c))
    target = torch.ones((bs, t, 1, grid, c), dtype=torch.float32)

    loss(pred, target)
