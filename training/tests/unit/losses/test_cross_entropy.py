# (C) Copyright 2025- Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import math

import pytest
import torch
from omegaconf import DictConfig

from anemoi.training.losses import CombinedLoss
from anemoi.training.losses import CrossEntropyLoss
from anemoi.training.losses import get_loss_function
from anemoi.training.schemas.training import CombinedLossSchema
from anemoi.training.utils.enums import TensorDim


def _random_inputs(ens: int, num_classes: int = 4, soft: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    """Random logits (bs=2, t=1, ens, grid=7, C) and targets (bs=2, t=1, 1, grid=7, C)."""
    gen = torch.Generator().manual_seed(0)
    pred = torch.randn(2, 1, ens, 7, num_classes, generator=gen)
    if soft:
        target = torch.softmax(torch.randn(2, 1, 1, 7, num_classes, generator=gen), dim=-1)
    else:
        target = torch.nn.functional.one_hot(torch.randint(0, num_classes, (2, 1, 1, 7), generator=gen), num_classes)
    return pred, target.float()


def test_worked_example() -> None:
    # class 1 observed; member 1 gave it 10 %, member 2 gave it 0.1 %
    pred = torch.log(torch.tensor([[0.90, 0.10], [0.999, 0.001]]))[None, None, :, None, :]
    target = torch.tensor([0.0, 1.0])[None, None, None, None, :]
    loss = CrossEntropyLoss()(pred, target, squash_mode="sum")
    assert loss.item() == pytest.approx((-math.log(0.10) - math.log(0.001)) / 2, rel=1e-5)


@pytest.mark.parametrize("soft", [False, True], ids=["one_hot", "soft"])
def test_single_member_equals_torch_cross_entropy(soft: bool) -> None:
    pred, target = _random_inputs(ens=1, soft=soft)
    ce = CrossEntropyLoss()._ce_loss(pred, target[:, :, 0]).sum(dim=-1)
    expected = torch.nn.functional.cross_entropy(pred.movedim(-1, 1), target.movedim(-1, 1), reduction="none")
    torch.testing.assert_close(ce, expected[:, :, 0])


def test_avg_squash_reports_ce_over_num_classes() -> None:
    pred, target = _random_inputs(ens=3, num_classes=5)
    loss = CrossEntropyLoss()
    loss.add_scaler(TensorDim.GRID, torch.ones(7), name="grid")
    loss.add_scaler(TensorDim.VARIABLE, torch.ones(5), name="variable")
    ce = loss._ce_loss(pred, target[:, :, 0]).sum(dim=-1)  # (bs, t, grid)
    expected = ce.sum(dim=(1, 2)).mean()  # summed over time and grid, averaged over the batch
    torch.testing.assert_close(loss(pred, target, squash_mode="sum"), expected)
    torch.testing.assert_close(loss(pred, target), expected / 5)
    assert loss(pred, target, squash=False).shape == (5,)


@pytest.mark.parametrize(
    ("pred_slice", "target_slice"),
    [((slice(None), 0), ()), ((), (Ellipsis, slice(0, 3)))],
    ids=["pred_not_5d", "class_count_mismatch"],
)
def test_rejects_invalid_shapes(pred_slice: tuple, target_slice: tuple) -> None:
    pred, target = _random_inputs(ens=2)
    with pytest.raises(ValueError, match="pred"):
        CrossEntropyLoss()(pred[pred_slice], target[target_slice])


@pytest.mark.parametrize("nan_in", ["target", "pred"])
def test_ignore_nans_drops_the_grid_point(nan_in: str) -> None:
    pred, target = _random_inputs(ens=3)
    loss = CrossEntropyLoss(ignore_nans=True)
    dropped = target.clone()
    dropped[0, 0, 0, 2] = 0.0
    expected = loss(pred, dropped)

    if nan_in == "target":
        target[0, 0, 0, 2] = float("nan")
    else:
        pred[0, 0, 1, 2, 0] = float("nan")

    assert torch.isnan(CrossEntropyLoss()(pred, target))
    torch.testing.assert_close(loss(pred, target), expected)


def test_built_from_config_inside_combined_loss() -> None:
    config = {
        "_target_": "anemoi.training.losses.combined.CombinedLoss",
        "scalers": ["grid"],
        "losses": [{"_target_": "anemoi.training.losses.CrossEntropyLoss", "no_autocast": False}],
    }
    schema = CombinedLossSchema.model_validate(config)
    loss = get_loss_function(
        DictConfig(schema.model_dump(by_alias=True)),
        scalers={"grid": (TensorDim.GRID, torch.ones(7))},
    )
    assert isinstance(loss, CombinedLoss)
    assert isinstance(loss.losses[0], CrossEntropyLoss)
    assert loss.losses[0].no_autocast is False
    assert "grid" in loss.losses[0].scaler
