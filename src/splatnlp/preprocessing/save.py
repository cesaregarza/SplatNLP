import logging
import os

import pandas as pd

logger = logging.getLogger(__name__)


def save(df: pd.DataFrame, path: str, *, index: bool = False) -> None:
    """Save a DataFrame to a file, appending if the file already exists.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (str): The file path.
        index (bool): Whether to write row names (index). Defaults to False.
    """
    logger.info(f"Saving DataFrame to {path}")

    if os.path.exists(path):
        logger.info(f"File {path} already exists. Appending data.")
        df.to_csv(path, mode="a", header=False, index=index)
    else:
        logger.info(f"Creating new file {path}")
        df.to_csv(path, index=index)
