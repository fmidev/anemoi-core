import pytest
import torch

from anemoi.training.losses import MixtureCrossEntropyLoss
from anemoi.training.utils.enums import TensorDim


def test_mixture_ce_matches_expected_mixture_cross_entropy() -> None:
    bs, t, ens, grid, c = 2, 3, 4, 5, 6
    loss = MixtureCrossEntropyLoss(num_classes=c)
    loss.add_scaler(TensorDim.GRID, torch.ones(grid), name="grid")

    pred = torch.ones((bs, t, ens, grid, c))
    target = torch.ones((bs, t, 1, grid, c), dtype=torch.float32)

    loss(pred, target)

    # probs_member_1 = torch.softmax(pred[0, 0, 0], dim=-1)
    # probs_member_2 = torch.softmax(pred[0, 0, 1], dim=-1)
    # mixture = 0.5 * (probs_member_1 + probs_member_2)
    # expected = -torch.log(torch.tensor([mixture[0, 0], mixture[1, 1]])).sum()

    # torch.testing.assert_close(result, expected)


# def test_mixture_ce_supports_label_smoothing_and_class_weights() -> None:
#    loss = MixtureCrossEntropyLoss(num_classes=3, label_smoothing=0.3, class_weights=[1.0, 2.0, 4.0])
#    loss.add_scaler(TensorDim.GRID, torch.ones(1), name="grid")
#
#    pred = torch.tensor([[[[[3.0, 1.0, -1.0]], [[1.0, 0.0, 2.0]]]]], dtype=torch.float32)
#    target = torch.tensor([[[[0.0, 1.0, 0.0]]]], dtype=torch.float32)
#
#    result = loss(pred, target)
#
#    smoothed = (1.0 - 0.3) * target[0, 0, 0] + 0.3 / 3.0
#    mixture = torch.logsumexp(torch.log_softmax(pred[0, 0, :, 0], dim=-1), dim=0) - torch.log(torch.tensor(2.0))
#    expected = -(smoothed * mixture).sum() * (smoothed * torch.tensor([1.0, 2.0, 4.0])).sum()
#
#    torch.testing.assert_close(result, expected)
#
#
# @pytest.mark.parametrize("bad_shape", [torch.randn(1, 1, 3, 2), torch.randn(1, 2, 1, 1, 4)])
# def test_mixture_ce_rejects_incompatible_shapes(bad_shape: torch.Tensor) -> None:
#    loss = MixtureCrossEntropyLoss(num_classes=2)
#    loss.add_scaler(TensorDim.GRID, torch.ones(1), name="grid")
#    target = torch.randn(1, 1, 1, 2)
#
#    with pytest.raises(ValueError, match=r"Expected pred with shape|Expected last dimension|Prediction and target"):
#        loss(bad_shape, target)
