"""
TF-IDF utilities for the SAE dashboard.
"""

import numpy as np
import polars as pl


def compute_idf(df: pl.DataFrame) -> pl.DataFrame:
    """Compute the IDF for a given dataframe.

    Args:
        df: The dataframe to compute the IDF for.

    Returns:
        A dataframe with the IDF for each ability_input_token.
    """
    LOG_N = np.log(len(df) + 1)
    return (
        df.explode("ability_input_tokens")
        .group_by("ability_input_tokens")
        .agg(pl.col("index").count().alias("count"))
        .with_columns(
            (pl.lit(LOG_N).sub(pl.col("count").add(1).log())).alias("idf")
        )
        .select(["ability_input_tokens", "idf"])
    )


def compute_tf_idf(idf: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    """Compute the TF-IDF for a given dataframe.

    Args:
        idf: The IDF dataframe.
        df: The dataframe to compute the TF-IDF for.

    Returns:
        A dataframe with the TF-IDF for each ability_input_token.
    """
    return (
        df.explode("ability_input_tokens")
        .sort(["index", "ability_input_tokens"])
        .group_by("ability_input_tokens")
        .agg(pl.col("index").count().alias("tf"))
        .join(idf, on="ability_input_tokens", how="left")
        .with_columns(pl.col("tf").mul(pl.col("idf")).alias("tf_idf"))
        .sort("tf_idf", descending=True)
    )
