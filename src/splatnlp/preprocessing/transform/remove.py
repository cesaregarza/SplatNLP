import logging

import pandas as pd

from splatnlp.preprocessing.constants import (
    BUFFER_DAYS_FOR_MAJOR_PATCH,
    BUFFER_DAYS_FOR_MINOR_PATCH,
    REMOVE_COLUMNS,
)

logger = logging.getLogger(__name__)


def remove_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Remove specified columns from a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with specified columns removed.
    """
    logger.info("Starting to remove specified columns")
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    result = df.drop(columns=REMOVE_COLUMNS)
    logger.info(f"Final DataFrame shape: {result.shape}")
    return result


def remove_buffer_days(df: pd.DataFrame, date_df: pd.DataFrame) -> pd.DataFrame:
    """Remove entries from a DataFrame that are within a buffer period of a
    patch.

    Args:
        df (pd.DataFrame): The input DataFrame.
        date_df (pd.DataFrame): The DataFrame containing the dates of patches.

    Returns:
        pd.DataFrame: The DataFrame with entries removed that are within a
            buffer period of a patch.
    """
    logger.info("Starting to remove buffer days")
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    buffer_days = (
        date_df[date_df["major"]]["date"]
        .sub(BUFFER_DAYS_FOR_MAJOR_PATCH, fill_value=pd.Timestamp.min)
        .append(
            date_df[date_df["minor"]]["date"].sub(
                BUFFER_DAYS_FOR_MINOR_PATCH, fill_value=pd.Timestamp.min
            )
        )
    )
    result_df = df.loc[~df["period"].isin(buffer_days)]

    logger.info(
        "Finished removing buffer days. "
        f"Final DataFrame shape: {result_df.shape}"
    )
    return result_df
