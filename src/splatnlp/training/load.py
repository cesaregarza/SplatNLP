import logging

import pandas as pd

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
