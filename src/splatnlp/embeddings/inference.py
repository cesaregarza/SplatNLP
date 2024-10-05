import logging

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from tqdm import tqdm

logger = logging.getLogger(__name__)


def infer_vectors(
    model: Doc2Vec, sentences: pd.Series, remove_weapon_id: bool = True
) -> np.ndarray:
    """Infer vectors for a list of sentences using a trained Doc2Vec model.

    Args:
        model (Doc2Vec): The trained Doc2Vec model.
        sentences (pd.Series): The list of sentences.
        remove_weapon_id (bool, optional): Whether to remove the weapon_id from
            the sentences. Defaults to True.

    Returns:
        np.ndarray: The numpy array with the inferred vectors.
    """
    logger.info("Starting infer_vectors function")
    logger.debug(f"Input model: {model}")
    logger.debug(f"Number of sentences: {len(sentences)}")

    logger.info("Infering vectors")
    sentence_list = sentences.str.split(" ").rename("sentence")
    if remove_weapon_id:
        logger.info("Removing weapon_id from ability_tags")
        sentence_list = sentence_list.str[:-1]

    inferred_vectors = []
    for sentence in tqdm(sentence_list, desc="Infering vectors"):
        inferred_vectors.append(model.infer_vector(sentence))

    inferred_vectors_array = np.array(inferred_vectors)
    logger.debug(f"Inferred vectors shape: {inferred_vectors_array.shape}")

    return inferred_vectors_array
