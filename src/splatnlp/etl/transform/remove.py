import logging

import pandas as pd

from splatnlp.etl.constants import REMOVE_COLUMNS

logger = logging.getLogger(__name__)


def remove(df: pd.DataFrame) -> pd.DataFrame:
    """Remove columns from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with columns removed.
    """
    logger.info("Starting to remove columns")
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    result = df.drop(columns=REMOVE_COLUMNS)
    logger.info(f"Final DataFrame shape: {result.shape}")
    return result
