import logging

import pandas as pd
from gensim.models import Doc2Vec

logger = logging.getLogger(__name__)


def load_data(path: str | None = None) -> pd.DataFrame:
    """Load a CSV file into a DataFrame.

    Args:
        path (str): The path to the CSV file.

    Returns:
        pd.DataFrame: The loaded DataFrame.
    """
    path = path or "data/weapon_partitioned.csv"
    logger.info("Loading data from %s", path)
    return pd.read_csv(
        path, dtype={"weapon_id": "Int64", "ability_tags": "str"}
    )


def load_model(path: str | None = None) -> Doc2Vec:
    """Load a Doc2Vec model from a file.

    Args:
        path (str): The path to the model file.

    Returns:
        Doc2Vec: The loaded Doc2Vec model.
    """
    path = path or "models/doc2vec.model"
    logger.info("Loading model from %s", path)
    return Doc2Vec.load(path)
