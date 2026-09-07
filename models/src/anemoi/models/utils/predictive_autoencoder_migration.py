# (C) Copyright 2026- Anemoi contributors.
#
# Licensed under the Apache Licence, Version 2.0.

"""Explicit conversion of pre-multi-encoder predictive-autoencoder configurations.

This is a migration utility, not a runtime compatibility layer. Call it on a
fully composed legacy config before constructing the upstream-compatible model.
"""

from copy import deepcopy
from pathlib import Path

from omegaconf import DictConfig
from omegaconf import OmegaConf


def migrate_config(config: DictConfig, dataset_names: list[str]) -> DictConfig:
    """Return a new config preserving mapper widths and conditioning order."""
    result = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
    model = OmegaConf.to_container(config.model, resolve=True)
    if "encoders" in model:
        raise ValueError("Expected a legacy config with model.encoder, not model.encoders.")
    width = model["num_channels"]
    trainable = model.pop("trainable_parameters")
    model["node_trainable_parameters"] = {
        **{name: trainable.get("data", 0) for name in dataset_names},
        model["model"]["hidden_nodes_name"]: trainable.get("hidden", 0),
    }
    model["edge_trainable_parameters"] = {
        key: value for key, value in trainable.items() if key not in ("data", "hidden")
    }
    encoder = model.pop("encoder")
    decoder = model.pop("decoder")
    encoder["num_channels"] = decoder["num_channels"] = width
    model["processor"]["num_channels"] = width
    model["encoders"] = {}
    model["decoders"] = {}
    for index, name in enumerate(dataset_names):
        key = str(index)
        model["encoders"][key] = {
            "source_datasets": [name],
            "dataset_fusing_strategy": "not_supported",
            "mapper": deepcopy(encoder),
        }
        features = ["forcings", "coordinates"]
        if trainable.get("data", 0):
            features.append("trainable_parameters")
        model["decoders"][key] = {
            "target_datasets": [name],
            "target_node_features": features,
            "mapper": deepcopy(decoder),
        }
    model["latent_aggregator"] = {"_target_": "anemoi.models.layers.aggregator.SumAggregator"}
    mixer = model.pop("state_context_mixer")
    mixer["_target_"] = "anemoi.models.layers.aggregator.ConcatMLPAggregator"
    mixer.pop("context_channels", None)
    model["state_context_aggregator"] = mixer
    model["forcing_encoder"]["num_channels"] = width
    for field in ("residual", "bounding", "output_mask"):
        value = model.get(field, [] if field == "bounding" else None)
        if value is not None and not (isinstance(value, dict) and "datasets" in value):
            model[field] = {"datasets": {name: deepcopy(value) for name in dataset_names}}
    result.model = model
    if "training" in result and "submodules_to_freeze" in result.training:
        result.training.submodules_to_freeze = [
            name.replace("state_context_mixer", "state_context_aggregator")
            for name in result.training.submodules_to_freeze
        ]
    return result


def migrate_checkpoint(source: Path, destination: Path) -> None:
    """Write a strictly checked, weights-only warm-start checkpoint.

    Only use trusted Lightning checkpoints: loading their task objects requires
    pickle. Optimizer and training-loop state are intentionally not transferred;
    this output starts a new training stage, not an exact continuation of a run.
    """
    import torch

    from anemoi.models.interface import AnemoiModelInterface

    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}")
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    hparams = checkpoint["hyper_parameters"]
    datasets = list(hparams["data_indices"])
    config = migrate_config(hparams["config"], datasets)
    hparams["config"] = config
    hparams["metadata"]["config"] = OmegaConf.to_container(config, resolve=True)
    task = hparams["task"]
    model = AnemoiModelInterface(
        config=config,
        n_step_input=task.num_input_timesteps,
        n_step_output=task.num_output_timesteps,
        **{
            key: hparams[key]
            for key in (
                "graph_data",
                "statistics",
                "statistics_tendencies",
                "data_indices",
                "metadata",
                "supporting_arrays",
            )
        },
    )
    state = migrate_state_dict(checkpoint["state_dict"], datasets, prefix="model.model.")
    if any(not key.startswith("model.") for key in state):
        raise ValueError("Expected a checkpoint containing only model state.")
    model.load_state_dict({key.removeprefix("model."): value for key, value in state.items()}, strict=True)
    checkpoint["state_dict"] = state
    checkpoint["predictive_autoencoder_migration"] = {
        "source": str(source),
        "source_global_step": checkpoint["global_step"],
        "source_epoch": checkpoint["epoch"],
        "weights_only": True,
    }
    for key in ("optimizer_states", "lr_schedulers", "loops", "callbacks"):
        checkpoint.pop(key, None)
    checkpoint["epoch"] = checkpoint["global_step"] = 0
    with destination.open("xb") as output:
        torch.save(checkpoint, output)
    print(f"Strict model load passed. Saved weights-only warm start: {destination}")


def migrate_state_dict(state_dict: dict, dataset_names: list[str], prefix: str = "") -> dict:
    """Rename model parameter keys; tensor values and ordering remain unchanged.

    Pass the model's exact prefix for a Lightning state dictionary. Strict-load
    the result into a model built with migrate_config before saving a checkpoint.
    """
    replacements = [(prefix + "state_context_mixer.", prefix + "state_context_aggregator.")]
    for index, name in enumerate(dataset_names):
        for component in ("encoder", "decoder"):
            replacements.append((f"{prefix}{component}.{name}.", f"{prefix}{component}.{index}."))
    result = {}
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacements:
            if key.startswith(old):
                new_key = new + key[len(old):]
                break
        if new_key in result:
            raise ValueError(f"Migration produces a duplicate parameter key: {new_key}")
        result[new_key] = value
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Migrate a trusted legacy predictive-codec checkpoint for a new stage.")
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    migrate_checkpoint(args.source, args.destination)
