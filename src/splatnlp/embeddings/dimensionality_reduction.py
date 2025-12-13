from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List

import numpy as np
from sklearn.manifold import TSNE

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from gensim.models import Doc2Vec


def reduce_doc2vec_dimensions_by_tag(
    model: "Doc2Vec",
    tags_to_reduce: List[int] | None = None,  # Expect integer tags (weapon IDs)
    *,
    n_components: int = 2,
    perplexity: float = 30.0,
    n_iter: int = 1000,
    random_state: int | None = None,  # Add random state for reproducibility
) -> tuple[np.ndarray, List[int]]:
    """Reduce the dimensionality of vectors associated with specific integer tags
       in a Doc2Vec model using t-SNE.

    Args:
        model (Doc2Vec): The trained Doc2Vec model.
        tags_to_reduce (List[int] | None, optional): A list of integer tags
            (e.g., weapon ID tokens) whose vectors should be reduced. If None,
            attempts to reduce all tags present in model.dv. Defaults to None.
        n_components (int, optional): Dimension of the embedded space. Defaults to 2.
        perplexity (float, optional): Perplexity for t-SNE. Defaults to 30.0.
        n_iter (int, optional): Max iterations for t-SNE optimization. Defaults to 1000.
        random_state (int | None, optional): Random seed for t-SNE. Defaults to None.

    Returns:
        tuple[np.ndarray, List[int]]:
            - np.ndarray: The reduced vectors (shape: [num_tags, n_components]).
            - List[int]: The list of integer tags corresponding to the rows in the
                         reduced vectors array, in the same order.

    Raises:
        ValueError: If no valid tags are found or provided.
        KeyError: If a provided tag is not found in model.dv.
    """
    logger.info("Starting reduce_doc2vec_dimensions_by_tag function")
    logger.debug(f"Input model: {model}")
    logger.debug(
        f"Tags to reduce: {tags_to_reduce if tags_to_reduce else 'All in model.dv'}"
    )

    if tags_to_reduce is None:
        logger.info(
            "No specific tags provided, using all integer tags found in model.dv"
        )
        # Filter keys in model.dv to only include integers (or things castable to int)
        all_tags = []
        for tag in model.dv.index_to_key:  # Access tags safely
            try:
                # Attempt to convert tag to int. Skip if it fails (e.g., if words were also indexed)
                int_tag = int(tag)
                all_tags.append(int_tag)
            except (ValueError, TypeError):
                logger.debug(
                    f"Skipping non-integer tag found in model.dv: {tag}"
                )
                continue
        tags_to_process = sorted(list(set(all_tags)))  # Get unique sorted list
    else:
        # Ensure provided tags are integers
        tags_to_process = sorted(list(set(int(tag) for tag in tags_to_reduce)))

    if not tags_to_process:
        raise ValueError("No valid integer tags found or provided to reduce.")

    logger.info(
        f"Extracting vectors for {len(tags_to_process)} tags from model.dv"
    )
    vectors_list = []
    valid_tags_list = []
    missing_tags = []
    for tag_int in tags_to_process:
        try:
            # Access vector using the integer tag directly
            vectors_list.append(model.dv[tag_int])
            valid_tags_list.append(tag_int)
        except KeyError:
            missing_tags.append(tag_int)
            logger.warning(f"Tag {tag_int} not found in model.dv. Skipping.")

    if not vectors_list:
        raise ValueError(
            "No vectors could be extracted for the provided/found tags."
        )
    if missing_tags:
        logger.warning(f"Could not find vectors for tags: {missing_tags}")

    vector_array = np.array(vectors_list, dtype=np.float32)
    logger.debug(f"Extracted vector array shape: {vector_array.shape}")

    logger.info(
        f"Reducing dimensions ({vector_array.shape[1]}D -> {n_components}D) with t-SNE..."
    )
    tsne = TSNE(
        n_components=n_components,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=random_state,
        init="pca",  # PCA initialization is often more stable
        learning_rate="auto",  # Use automatic learning rate selection
    )
    reduced_vectors = tsne.fit_transform(vector_array)
    logger.debug(f"Reduced vectors shape: {reduced_vectors.shape}")
    logger.info("Dimensionality reduction complete.")

    return reduced_vectors, valid_tags_list
