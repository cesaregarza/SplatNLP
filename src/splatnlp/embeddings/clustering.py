import logging

import numpy as np
import umap
from sklearn.cluster import DBSCAN

from splatnlp.embeddings.inference import infer_vectors

logger = logging.getLogger(__name__)


def cluster(
    vector_array: np.ndarray,
    n_dimensions: int = 2,
    eps: float = 0.5,
    min_samples: int = 5,
    percent_sample: float | None = None,
) -> np.ndarray:
    """Cluster vectors using UMAP and DBSCAN.

    Args:
        vector_array (np.ndarray): The array with the vectors.
        n_dimensions (int): The number of dimensions for UMAP. Defaults to 2.
        eps (float): The maximum distance between two samples for one to be
            considered as in the neighborhood of the other. Defaults to 0.5.
        min_samples (int): The number of samples in a neighborhood for a point
            to be considered as a core point. Defaults to 5.
        percent_sample (float | None): If not None, this will override the
            number of samples to use for clustering. It defines the number of
            samples to use as a percentage of the total number of samples.
            Defaults to None.

    Returns:
        np.ndarray: The clustered vectors.
    """
    logger.info("Starting cluster function")
    logger.debug(f"Input vector_array shape: {vector_array.shape}")

    logger.info("Reducing dimensions with UMAP")
    reducer = umap.UMAP(n_components=n_dimensions)
    reduced_vectors = reducer.fit_transform(vector_array)
    logger.debug(f"Reduced vectors shape: {reduced_vectors.shape}")

    logger.info("Clustering with DBSCAN")
    if percent_sample is not None:
        min_samples = int(percent_sample * len(reduced_vectors))
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    clustered_vectors = dbscan.fit_predict(reduced_vectors)
    logger.debug(f"Clustered vectors shape: {clustered_vectors.shape}")

    return clustered_vectors
