import logging

import orjson
import pandas as pd

from splatnlp.preprocessing.constants import (
    BUCKET_THRESHOLDS,
    MAIN_ONLY_ABILITIES,
    STANDARD_ABILITIES,
)

logger = logging.getLogger(__name__)


def process_abilities(df: pd.DataFrame) -> pd.DataFrame:
    """Process the abilities column of a DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with the abilities column processed.
    """
    logger.info("Starting to process abilities")
    abilities_df: pd.DataFrame = (
        df["abilities"]
        .apply(orjson.loads)
        .pipe(pd.json_normalize)
        .fillna(0)
        .astype(float)
        .mul(10)
        .astype(int)
    )
    abilities_df.index = df.index
    logger.debug(f"Processed abilities DataFrame shape: {abilities_df.shape}")

    cols = []
    for main_ability in MAIN_ONLY_ABILITIES:
        logger.debug(f"Processing main ability: {main_ability}")
        cols.append(
            abilities_df[main_ability]
            .gt(0)
            .map({True: f"{main_ability} ", False: ""})
            .rename(main_ability)
        )

    for ability in STANDARD_ABILITIES:
        logger.debug(f"Processing standard ability: {ability}")
        for threshold in BUCKET_THRESHOLDS:
            tag = f"{ability}_{threshold}"
            cols.append(
                abilities_df[ability]
                .ge(threshold)
                .map({True: f"{tag} ", False: ""})
                .rename(tag)
            )

    ability_col = (
        pd.concat(cols, axis=1).sum(axis=1).str.strip().rename("ability_tags")
    )
    result_df = pd.concat([df, ability_col], axis=1)

    result_df["ability_tags"] = (
        result_df["ability_tags"]
        .add(" weapon_id_")
        .add(result_df["weapon_id"].astype(str))
    )

    logger.info(
        f"Finished processing abilities. Final DataFrame shape: %s",
        str(result_df.shape),
    )
    return result_df
