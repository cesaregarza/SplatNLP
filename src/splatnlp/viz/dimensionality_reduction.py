import logging

import numpy as np
import pandas as pd
from gensim.models import Doc2Vec
from sklearn.manifold import TSNE

from splatnlp.preprocessing.transform.parse import generate_maps

logger = logging.getLogger(__name__)


def reduce_dimensions(
    model: Doc2Vec,
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
) -> np.ndarray:
    """Reduce the dimensionality of the vectors in a Doc2Vec model.

    Args:
        model (Doc2Vec): The trained Doc2Vec model.
        n_components (int, optional): The dimension of the embedded space.
            Defaults to 2.
        perplexity (float, optional): The perplexity of the t-SNE algorithm.
            Defaults to 30.0.
        n_iter (int, optional): Maximum number of iterations for the optimization.
            Defaults to 1000.

    Returns:
        np.ndarray: The reduced vectors.
    """
    _, id_to_name, _ = generate_maps()

    logger.info("Starting reduce_dimensions function")
    logger.debug(f"Input model: {model}")

    logger.info("Extracting vectors from model")
    vectors = model.dv.vectors

    logger.info("Reducing dimensions with t-SNE")
    reduced_vectors = TSNE(
        n_components=n_components, perplexity=perplexity, n_iter=n_iter
    ).fit_transform(vectors)
    logger.debug(f"Reduced vectors shape: {reduced_vectors.shape}")

    return reduced_vectors
