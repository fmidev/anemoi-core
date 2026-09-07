# (C) Copyright 2026- Anemoi contributors.
#
# Licensed under the Apache Licence, Version 2.0.

import pytest
import torch

from anemoi.models.utils.predictive_autoencoder_migration import migrate_state_dict


@pytest.mark.parametrize("prefix", ["", "model.model."])
def test_migration_preserves_tensors_and_unrelated_keys(prefix):
    names = {
        "encoder.data.weight": "encoder.0.weight",
        "decoder.data.weight": "decoder.0.weight",
        "state_context_mixer.mlp.weight": "state_context_aggregator.mlp.weight",
        "processor.weight": "processor.weight",
        "static_forcing_encoder.data.weight": "static_forcing_encoder.data.weight",
    }
    state = {prefix + name: torch.randn(2, 2) for name in names}
    migrated = migrate_state_dict(state, ["data"], prefix=prefix)
    assert list(migrated) == [prefix + name for name in names.values()]
    for old, new in names.items():
        assert migrated[prefix + new] is state[prefix + old]
    assert list(state) == [prefix + name for name in names]


def test_migration_rejects_colliding_keys():
    state = {"encoder.data.weight": torch.ones(1), "encoder.0.weight": torch.zeros(1)}
    with pytest.raises(ValueError, match="duplicate parameter key"):
        migrate_state_dict(state, ["data"])
