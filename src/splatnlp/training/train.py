import logging

import pandas as pd
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument

logger = logging.getLogger(__name__)


def train_embeddings(
    df: pd.DataFrame,
    *,
    vector_size: int = 100,
    window: int = 20,
    min_count: int = 1,
    epochs: int = 10,
    dm: int = 0,
    workers: int = 4,
    remove_weapon_id: bool = True,
) -> Doc2Vec:
    """Train a Doc2Vec model on a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame with text data.
        vector_size (int, optional): The dimensionality of the feature vectors.
            Defaults to 100.
        window (int, optional): The maximum distance between the current and
            predicted word within a sentence. Defaults to 20.
        min_count (int, optional): Ignores all words with total frequency lower
            than this. Defaults to 1.
        epochs (int, optional): Number of iterations (epochs) over the corpus.
            Defaults to 10.
        dm (int, optional): Defines the training algorithm. If dm=1, 'distributed
            memory' (PV-DM) is used. Otherwise, distributed bag of words (PV-DBOW)
            is employed. Defaults to 0.
        workers (int, optional): Use these many worker threads to train the model
            (=faster training with multicore machines). Defaults to 4.

    Returns:
        Doc2Vec: The trained Doc2Vec model.
    """
    logger.info("Starting train_embeddings function")
    logger.debug(f"Input DataFrame shape: {df.shape}")

    sentence_list = df["ability_tags"].str.split(" ")
    weapon_ids = df["weapon_id"].tolist()
    if remove_weapon_id:
        logger.info("Removing weapon_id from ability_tags")
        sentence_list = sentence_list.str[:-1]

    logger.info("Creating tagged documents")
    tagged_data = [
        TaggedDocument(words=sentence, tags=[weapon_id])
        for sentence, weapon_id in zip(sentence_list, weapon_ids)
    ]
    logger.debug(f"Number of tagged documents: {len(tagged_data)}")

    logger.info("Training Doc2Vec model")
    model = Doc2Vec(
        documents=tagged_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        dm=dm,
        workers=workers,
    )

    logger.info("Doc2Vec model training completed")
    return model
