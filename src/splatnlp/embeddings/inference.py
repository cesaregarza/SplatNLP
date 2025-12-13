from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from gensim.models import Doc2Vec


def infer_doc2vec_vectors(
    model: "Doc2Vec",
    token_lists: pd.Series | list[list[int]],
) -> np.ndarray:
    """Infer Doc2Vec vectors for lists of integer ability tokens.

    Args:
        model (Doc2Vec): The trained Doc2Vec model.
        token_lists (pd.Series | List[List[int]]): A pandas Series or a list
            where each element is a list of integer ability tokens.

    Returns:
        np.ndarray: A numpy array where each row is the inferred vector for a
                    corresponding token list.
    """
    logger.info("Starting infer_doc2vec_vectors function")
    if isinstance(token_lists, pd.Series):
        num_sentences = len(token_lists)
        iterator = token_lists
    elif isinstance(token_lists, list):
        num_sentences = len(token_lists)
        iterator = token_lists
    else:
        raise ValueError(f"Invalid input type: {type(token_lists)}")

    logger.debug(f"Input Doc2Vec model: {model}")
    logger.debug(f"Number of token lists (sentences) to infer: {num_sentences}")

    inferred_vectors = []
    logger.info("Inferring vectors...")
    for token_list_int in tqdm(iterator, desc="Inferring vectors"):
        if not isinstance(token_list_int, list):
            logger.warning(
                f"Skipping item, expected list of int tokens, got: "
                f"{type(token_list_int)}"
            )

            inferred_vectors.append(np.zeros(model.vector_size))
            continue

        token_list_str = [str(token) for token in token_list_int]

        inferred_vector = model.infer_vector(token_list_str)
        inferred_vectors.append(inferred_vector)

    if not inferred_vectors:
        logger.warning("No vectors were inferred.")
        return np.array([])

    inferred_vectors_array = np.array(inferred_vectors, dtype=np.float32)
    logger.debug(f"Inferred vectors shape: {inferred_vectors_array.shape}")
    logger.info("Finished inferring vectors.")

    return inferred_vectors_array
