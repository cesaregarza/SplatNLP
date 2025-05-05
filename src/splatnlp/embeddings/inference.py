import logging

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

logger = logging.getLogger(__name__)


def infer_doc2vec_vectors(
    model: Doc2Vec,
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
                f"Skipping item, expected list of int tokens, got: {type(token_list_int)}"
            )
            # Append a placeholder or handle error as needed. Appending zeros for now.
            # Ensure the shape matches model.vector_size
            inferred_vectors.append(np.zeros(model.vector_size))
            continue

        # Convert integer tokens to strings for Doc2Vec inference
        token_list_str = [str(token) for token in token_list_int]

        # Infer vector using the list of string tokens
        inferred_vector = model.infer_vector(token_list_str)
        inferred_vectors.append(inferred_vector)

    if not inferred_vectors:
        logger.warning("No vectors were inferred.")
        return np.array([])  # Return empty array

    inferred_vectors_array = np.array(
        inferred_vectors, dtype=np.float32
    )  # Ensure consistent dtype
    logger.debug(f"Inferred vectors shape: {inferred_vectors_array.shape}")
    logger.info("Finished inferring vectors.")

    return inferred_vectors_array
