import logging

import numpy as np
import umap
from sklearn.cluster import DBSCAN

logger = logging.getLogger(__name__)


def cluster_vectors(
    vector_array: np.ndarray,
    n_dimensions: int = 2,
    eps: float = 0.5,
    min_samples: int = 5,
    umap_neighbors: int = 15,
    umap_min_dist: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster vectors using UMAP for dimensionality reduction and DBSCAN for
    clustering.

    Args:
        vector_array (np.ndarray): Array of vectors to cluster (shape:
            [n_samples, n_features]).
        n_dimensions (int): Target dimension for UMAP reduction. Defaults to 2.
        eps (float): DBSCAN max distance between samples for neighborhood.
            Defaults to 0.5.
        min_samples (int): DBSCAN min samples in neighborhood for core point.
            Defaults to 5.
        umap_neighbors (int): UMAP number of neighbors. Defaults to 15.
        umap_min_dist (float): UMAP minimum distance. Defaults to 0.1.
        random_state (int | None): Random seed for UMAP. Defaults to None.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - np.ndarray: The vectors reduced to n_dimensions by UMAP.
            - np.ndarray: Cluster labels assigned by DBSCAN (-1 for noise
                points).

    Raises:
        ValueError: If input array is empty or not 2D.
    """
    logger.info("Starting cluster_vectors function")
    if vector_array.ndim != 2 or vector_array.shape[0] == 0:
        raise ValueError(
            "Input vector_array must be 2D and non-empty, got shape "
            f"{vector_array.shape}"
        )
    logger.debug(f"Input vector_array shape: {vector_array.shape}")

    logger.info(
        f"Reducing dimensions ({vector_array.shape[1]}D -> {n_dimensions}D) "
        "with UMAP..."
    )
    reducer = umap.UMAP(
        n_components=n_dimensions,
        n_neighbors=umap_neighbors,
        min_dist=umap_min_dist,
        random_state=random_state,
        metric="cosine",
    )
    reduced_vectors = reducer.fit_transform(vector_array)
    logger.debug(f"Reduced vectors shape: {reduced_vectors.shape}")

    logger.info(
        f"Clustering {reduced_vectors.shape[0]} reduced vectors with DBSCAN..."
    )
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean")
    cluster_labels = dbscan.fit_predict(reduced_vectors)
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    logger.debug(
        f"Clustering complete. Found {n_clusters} clusters and {n_noise} "
        "noise points."
    )
    logger.debug(f"Cluster labels shape: {cluster_labels.shape}")

    return reduced_vectors, cluster_labels
