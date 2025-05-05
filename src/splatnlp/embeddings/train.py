import logging

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_doc2vec_embeddings(
    df: pd.DataFrame,
    *,
    vector_size: int = 100,
    window: int = 5,  # Smaller window might make sense for token IDs
    min_count: int = 1,
    epochs: int = 10,
    dm: int = 1,  # PV-DM often works well
    workers: int = 4,
) -> Doc2Vec:
    """Train a Doc2Vec model on tokenized ability and weapon data.

    Uses integer ability tokens (converted to strings) as words and
    integer weapon ID tokens as tags.

    Args:
        df (pd.DataFrame): DataFrame with 'ability_tags' (list of int) and
                           'weapon_id' (int) columns.
        vector_size (int, optional): Dimensionality of the feature vectors. Defaults to 100.
        window (int, optional): Max distance between current/predicted word. Defaults to 5.
        min_count (int, optional): Ignore words with frequency lower than this. Defaults to 1.
        epochs (int, optional): Number of iterations over the corpus. Defaults to 10.
        dm (int, optional): Defines training algorithm (1=PV-DM, 0=PV-DBOW). Defaults to 1.
        workers (int, optional): Number of worker threads. Defaults to 4.

    Returns:
        Doc2Vec: The trained Doc2Vec model.
    """
    logger.info("Starting Doc2Vec training with tokenized data")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    # Check required columns
    if "ability_tags" not in df.columns or "weapon_id" not in df.columns:
        raise ValueError(
            "DataFrame must contain 'ability_tags' and 'weapon_id' columns."
        )

    # Ensure columns have expected types
    if not isinstance(df["ability_tags"].iloc[0], list):
        logger.warning("Expected 'ability_tags' to be lists of integers.")
        # Attempt conversion or raise error
    if not pd.api.types.is_integer_dtype(
        df["weapon_id"]
    ) and not pd.api.types.is_float_dtype(df["weapon_id"]):
        logger.warning("Expected 'weapon_id' to be integer type.")
        # Attempt conversion or raise error

    logger.info("Creating tagged documents from token lists...")
    tagged_data = []
    for index, row in tqdm(
        df.iterrows(), total=len(df), desc="Processing rows"
    ):
        ability_tokens_int = row["ability_tags"]
        weapon_id_int = row["weapon_id"]

        # Ensure ability_tokens is a list
        if not isinstance(ability_tokens_int, list):
            logger.debug(
                f"Skipping row {index}: 'ability_tags' is not a list ({type(ability_tokens_int)})."
            )
            continue
        # Ensure weapon_id is not null/NaN
        if pd.isna(weapon_id_int):
            logger.debug(f"Skipping row {index}: 'weapon_id' is null.")
            continue

        # Convert integer ability tokens to strings for Doc2Vec 'words'
        # Using str() ensures compatibility with Doc2Vec which expects string tokens
        ability_tokens_str = [str(token) for token in ability_tokens_int]

        # Use the integer weapon ID as the tag (Doc2Vec tags can be integers)
        # Cast to int just in case it was loaded as float/object
        tag = [int(weapon_id_int)]

        tagged_data.append(TaggedDocument(words=ability_tokens_str, tags=tag))

    logger.info(f"Created {len(tagged_data)} tagged documents.")
    if not tagged_data:
        raise ValueError(
            "No valid tagged documents could be created from the input data."
        )

    logger.info("Training Doc2Vec model...")
    # Note: Consider adjusting hyperparameters like window size, vector_size based on the new token format
    model = Doc2Vec(
        documents=tagged_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=workers,
        dm_mean=1,  # Use mean of context vectors for PV-DM
        dbow_words=1 if dm == 0 else 0,  # Train word vectors if using PV-DBOW
    )

    logger.info("Doc2Vec model training completed.")
    return model
