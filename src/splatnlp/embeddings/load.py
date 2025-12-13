from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import orjson  # Use orjson if available for faster parsing
import pandas as pd

from splatnlp.embeddings._optional import require_doc2vec

if TYPE_CHECKING:  # pragma: no cover
    from gensim.models import Doc2Vec

logger = logging.getLogger(__name__)


def load_tokenized_data(path: str | Path) -> pd.DataFrame:
    """Load tokenized data from a CSV file into a DataFrame.

    Expects columns 'ability_tags' (parsable as list of ints) and
    'weapon_id' (int).

    Args:
        path (str | Path): The path to the tokenized CSV file (e.g., tokenized_data.csv).

    Returns:
        pd.DataFrame: The loaded DataFrame with 'ability_tags' as lists of ints.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If 'ability_tags' cannot be parsed.
    """
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Tokenized data file not found at: {data_path}"
        )

    logger.info("Loading tokenized data from %s", data_path)
    # Assume tab-separated as used in other parts of the project
    df = pd.read_csv(data_path, sep="\t", header=0, low_memory=False)

    # Ensure 'ability_tags' is parsed correctly (might be string representation of list)
    if not df.empty and isinstance(df["ability_tags"].iloc[0], str):
        logger.info("Parsing 'ability_tags' column (assuming orjson format)...")
        try:
            # Vectorized parsing might be faster if orjson supports it directly,
            # otherwise apply row-wise.
            df["ability_tags"] = df["ability_tags"].apply(orjson.loads)
        except Exception as parse_error:
            logger.error(
                f"Failed to parse 'ability_tags' with orjson: {parse_error}. Check data format."
            )
            raise ValueError(
                "Could not parse 'ability_tags' column."
            ) from parse_error
    elif not df.empty and not isinstance(df["ability_tags"].iloc[0], list):
        logger.warning(
            "Loaded 'ability_tags' column is not a string or list. Type: %s",
            type(df["ability_tags"].iloc[0]),
        )
        # Attempt conversion if possible, or raise error if type is unexpected
        try:
            df["ability_tags"] = df["ability_tags"].apply(
                lambda x: (
                    list(x)
                    if hasattr(x, "__iter__") and not isinstance(x, str)
                    else []
                )
            )
        except Exception:
            raise ValueError(
                "Unexpected type in 'ability_tags' column during loading."
            )

    # Ensure weapon_id is integer
    if "weapon_id" in df.columns:
        df["weapon_id"] = pd.to_numeric(
            df["weapon_id"], errors="coerce"
        ).astype(
            "Int64"
        )  # Use Int64 for nullable int support
        if df["weapon_id"].isnull().any():
            logger.warning(
                "Found null values in 'weapon_id' column after conversion."
            )

    logger.info("Tokenized data loaded successfully. Shape: %s", df.shape)
    return df


def load_vocab_json(path: str | Path) -> dict:
    """Load a vocabulary JSON file (standard or orjson)."""
    vocab_path = Path(path)
    if not vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary file not found at: {vocab_path}")
    logger.info("Loading vocabulary from %s", vocab_path)
    with open(vocab_path, "rb") as f:
        try:
            return orjson.loads(f.read())
        except orjson.JSONDecodeError:
            logger.warning(
                "Failed to parse vocab with orjson, trying standard json."
            )
            f.seek(0)
            return json.load(f)


def load_doc2vec_model(path: str | Path) -> "Doc2Vec":
    """Load a Doc2Vec model from a file.

    Args:
        path (str | Path): The path to the model file.

    Returns:
        Doc2Vec: The loaded Doc2Vec model.
    """
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Doc2Vec model file not found at: {model_path}"
        )

    logger.info("Loading Doc2Vec model from %s", model_path)
    doc2vec_cls = require_doc2vec()
    return doc2vec_cls.load(str(model_path))  # Ensure path is string
