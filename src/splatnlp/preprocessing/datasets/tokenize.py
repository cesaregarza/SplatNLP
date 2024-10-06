import logging

import pandas as pd

from splatnlp.preprocessing.constants import MASK, PAD

logger = logging.getLogger(__name__)


def tokenize(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Tokenize the ability tags and weapon IDs in a DataFrame.

    This function processes the 'ability_tags' and 'weapon_id' columns of the
    input DataFrame, converting them into integer IDs for use in machine
    learning models.

    Args:
        df (pd.DataFrame): The DataFrame containing 'ability_tags' and
            'weapon_id' columns to tokenize.

    Returns:
        tuple[pd.DataFrame, dict[str, int], dict[int, int]]: A tuple containing:
            - pd.DataFrame: A DataFrame with tokenized ability tags and weapon
                IDs.
            - dict[str, int]: A mapping of ability tags to their corresponding
                integer IDs.
            - dict[str, int]: A mapping of weapon IDs to their corresponding
                integer IDs.
    """
    logger.info("Starting tokenize function")

    exploded_abilities = df["ability_tags"].str.split(" ").str[:-1].explode()
    unique_abilities = exploded_abilities.unique()

    ability_to_id = {ability: i for i, ability in enumerate(unique_abilities)}
    ability_to_id[PAD] = len(ability_to_id)

    weapon_to_id = {
        weapon: i for i, weapon in enumerate(df["weapon_id"].unique())
    }

    logger.debug("Mapping ability tags to integer IDs and grouping to lists")
    ability_tags_ids = (
        exploded_abilities.map(ability_to_id).groupby(level=0).agg(list)
    )

    logger.debug("Tokenizing weapon IDs")
    weapon_ids = df["weapon_id"].map(weapon_to_id)

    logger.debug("Updating weapon ID mapping")
    weapon_to_id = {f"weapon_id_{k}": v for k, v in weapon_to_id.items()}

    logger.debug("Combining weapon IDs with ability tags")
    out_df = pd.concat([ability_tags_ids, weapon_ids], axis=1)

    logger.info("Tokenization complete")
    return out_df, ability_to_id, weapon_to_id
