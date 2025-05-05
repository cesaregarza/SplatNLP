"""
Helper functions for loading data and setting up hooks.
"""

import io
import json
import logging
import sys
from pathlib import Path

import orjson
import pandas as pd
import requests
import torch
import torch.nn as nn

from ..model.models import SetCompletionModel
from .data_objects import ActivationHook, register_activation_hook_generic

logger = logging.getLogger(__name__)


def load_json_from_path(path_str: str) -> dict:
    """Loads JSON/ORJSON from local path, URL, or S3 path."""
    path = Path(path_str)
    if path.is_file():
        logger.info(f"Loading JSON from local file: {path}")
        with open(path, "rb") as f:
            # Try orjson first, fallback to standard json
            try:
                return orjson.loads(f.read())
            except orjson.JSONDecodeError:
                f.seek(0)  # Reset file pointer
                return json.load(f)
    elif path_str.startswith(("http://", "https://")):
        logger.info(f"Loading JSON from URL: {path_str}")
        response = requests.get(path_str)
        response.raise_for_status()  # Raise an exception for bad status codes
        try:
            return orjson.loads(response.content)
        except orjson.JSONDecodeError:
            return json.loads(response.text)
    # Add S3 support if needed, requires boto3
    # elif path_str.startswith("s3://"):
    #     # Add S3 loading logic here using boto3
    #     raise NotImplementedError("S3 loading not implemented yet.")
    else:
        raise FileNotFoundError(f"File or URL not found: {path_str}")


def load_tokenized_data(path_str: str) -> pd.DataFrame:
    """Loads tokenized data CSV, handling orjson parsing."""
    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"Data CSV file not found: {path_str}")

    logger.info(f"Loading tokenized CSV from {path_str}")
    df = pd.read_csv(path_str, sep="\t", header=0)
    logger.info("Parsing ability_tags column (assuming orjson format)...")
    try:
        df["ability_tags"] = df["ability_tags"].apply(orjson.loads)
    except Exception as e:
        logger.warning(
            f"Failed to parse 'ability_tags' with orjson ({e}). Check data format."
        )
        # Optionally, add fallback to standard json if needed
        # try:
        #     logger.info("Attempting fallback parsing with standard json...")
        #     df["ability_tags"] = df["ability_tags"].apply(json.loads)
        # except Exception as e2:
        #     logger.error(f"Fallback json parsing also failed: {e2}")
        #     raise ValueError("Could not parse 'ability_tags' column.") from e2
        raise ValueError("Could not parse 'ability_tags' column.") from e

    return df


def setup_hook(
    model: SetCompletionModel,
    layer_index: int | None = None,
    feedforward_module_index: int | None = None,
    *,
    target: str = "masked_mean",
) -> tuple[ActivationHook, torch.utils.hooks.RemovableHandle]:
    """
    Register an activation hook.

    `target` options:
      • **'masked_mean'**  - captures the **input** to `output_layer`
      • **'token_ff'**     - captures a feed-forward activation
    """

    if target == "masked_mean":
        # input[0] of output_layer **is** the masked-mean vector we want
        hook = ActivationHook(capture="input")
        handle = model.output_layer.register_forward_hook(hook)
        logger.info(
            "Registered hook on input to model.output_layer (masked-mean)"
        )
        return hook, handle

    # ---------- legacy / token-level FF hook ----------------------------
    if layer_index is None or feedforward_module_index is None:
        raise ValueError(
            "layer_index and feedforward_module_index required for token_ff target"
        )

    try:
        target_module = model.transformer_layers[layer_index].feedforward[
            feedforward_module_index
        ]
    except (IndexError, AttributeError) as e:
        raise RuntimeError(
            "Could not locate the requested feed-forward module"
        ) from e

    hook = ActivationHook(capture="output")
    handle = register_activation_hook_generic(target_module, hook)
    logger.info(
        "Registered hook on transformer_layers[%s].feedforward[%s] (token_ff)",
        layer_index,
        feedforward_module_index,
    )
    return hook, handle
