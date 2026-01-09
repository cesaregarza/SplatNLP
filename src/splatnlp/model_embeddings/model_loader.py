from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.model_embeddings.io import load_json

ModelParams = dict[str, int | float | bool]

def _is_url(path: str) -> bool:
    return path.startswith(("http://", "https://"))

def resolve_model_params(
    checkpoint_path: str | Path,
    model_params_path: str | None,
    overrides: dict[str, int | float | bool | None],
) -> ModelParams:
    if model_params_path:
        return load_json(model_params_path)

    checkpoint_str = str(checkpoint_path)
    if not _is_url(checkpoint_str):
        candidate = Path(checkpoint_path).with_name("model_params.json")
        if candidate.is_file():
            return load_json(str(candidate))

    required = [
        "embedding_dim",
        "hidden_dim",
        "num_layers",
        "num_heads",
        "num_inducing_points",
        "use_layer_norm",
        "dropout",
    ]
    missing = [name for name in required if overrides.get(name) is None]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(
            "Missing model params. Provide --model-params or explicit "
            f"args: {missing_str}"
        )

    return {name: overrides[name] for name in required}  # type: ignore[misc]


def build_model(
    params: ModelParams,
    vocab_size: int,
    weapon_vocab_size: int,
    pad_token_id: int,
) -> SetCompletionModel:
    return SetCompletionModel(
        vocab_size=vocab_size,
        weapon_vocab_size=weapon_vocab_size,
        embedding_dim=int(params["embedding_dim"]),
        hidden_dim=int(params["hidden_dim"]),
        output_dim=vocab_size,
        num_layers=int(params["num_layers"]),
        num_heads=int(params["num_heads"]),
        num_inducing_points=int(params["num_inducing_points"]),
        use_layer_norm=bool(params["use_layer_norm"]),
        dropout=float(params["dropout"]),
        pad_token_id=pad_token_id,
    )


def load_checkpoint(
    checkpoint_path: str,
    *,
    map_location: str = "cpu",
) -> dict[str, torch.Tensor]:
    if _is_url(checkpoint_path):
        return torch.hub.load_state_dict_from_url(
            checkpoint_path, map_location=map_location
        )
    return torch.load(checkpoint_path, map_location=map_location)
