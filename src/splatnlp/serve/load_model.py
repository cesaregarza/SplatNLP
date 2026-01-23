"""Utility functions for loading model artifacts used by the API server.

The helpers in this module download vocabularies, model parameters and the
pre-trained ``SetCompletionModel`` from URLs.  ``load_from_env`` derives those
URLs from a set of environment variables and provides a single entry point used
by ``splatnlp.serve.app`` during application startup.
"""

import io
import logging
import os

import orjson
import requests
import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.utils.constants import PAD

logger = logging.getLogger(__name__)


def load_vocabulary(vocab_url: str) -> dict[str, int]:
    """Download and deserialize a vocabulary JSON file."""

    logger.info(f"Loading vocabulary from {vocab_url}")
    response = requests.get(vocab_url)
    return orjson.loads(response.content)


def load_model(model_url: str, model_params: dict) -> SetCompletionModel:
    """Instantiate and load a ``SetCompletionModel`` from a URL."""

    logger.info(f"Loading model from {model_url}")
    response = requests.get(model_url)
    response_data = io.BytesIO(response.content)
    logger.info(f"Model size: {len(response.content)} bytes")
    logger.info("Creating model")
    model = SetCompletionModel(**model_params)
    model.load_state_dict(
        torch.load(response_data, map_location=torch.device("cpu"))
    )
    model.eval()
    logger.info("Model loaded and ready for inference")
    return model


def load_model_params(params_url: str) -> dict:
    """Load the model parameter dictionary from ``params_url``."""

    logger.info(f"Loading model parameters from {params_url}")
    response = requests.get(params_url)
    return orjson.loads(response.content)


def load(
    vocab_url: str,
    weapon_vocab_url: str,
    model_url: str,
    params_url: str,
    info_url: str,
) -> tuple[dict, dict, int, dict, SetCompletionModel]:
    """Load the model, vocabularies and metadata from the provided URLs."""

    vocab = load_vocabulary(vocab_url)
    weapon_vocab = load_vocabulary(weapon_vocab_url)
    model_params = load_model_params(params_url)
    model_info = load_vocabulary(info_url)
    model = load_model(model_url, model_params)
    pad_token_id = vocab[PAD]
    return vocab, weapon_vocab, pad_token_id, model_info, model


def load_from_env() -> tuple[dict, dict, int, dict, SetCompletionModel]:
    """Load all required artifacts using environment variables.

    The function expects either individual URLs or a base DigitalOcean
    Spaces configuration:

    ``VOCAB_URL`` / ``WEAPON_VOCAB_URL`` / ``MODEL_URL`` / ``PARAMS_URL`` /
    ``INFO_URL``
        Direct links to the respective JSON or model files.
    ``DO_SPACES_ML_ENDPOINT`` and ``DO_SPACES_ML_DIR``
        If any of the above URLs are missing, these variables are combined to
        build a base path ``{ENDPOINT}/{DIR}`` from which default file names are
        inferred.
    """

    vocab_url = os.getenv("VOCAB_URL")
    weapon_vocab_url = os.getenv("WEAPON_VOCAB_URL")
    model_url = os.getenv("MODEL_URL")
    params_url = os.getenv("PARAMS_URL")
    info_url = os.getenv("INFO_URL")
    if not all([vocab_url, weapon_vocab_url, model_url, params_url]):
        base_url = os.getenv("DO_SPACES_ML_ENDPOINT")
        base_dir = os.getenv("DO_SPACES_ML_DIR")
        base_path = f"{base_url}/{base_dir}"
        vocab_url = vocab_url or f"{base_path}/vocab.json"
        weapon_vocab_url = weapon_vocab_url or f"{base_path}/weapon_vocab.json"
        model_url = model_url or f"{base_path}/model.pth"
        params_url = params_url or f"{base_path}/model_params.json"
        info_url = info_url or f"{base_path}/model_info.json"
    return load(vocab_url, weapon_vocab_url, model_url, params_url, info_url)
