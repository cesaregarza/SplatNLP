import logging
from typing import Optional

import pandas as pd

from splatnlp.preprocessing.constants import MASK, PAD

logger = logging.getLogger(__name__)


def tokenize(
    df: pd.DataFrame,
    ability_to_id: Optional[dict[str, int]] = None,
    weapon_to_id: Optional[dict[str, int]] = None,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int]]:
    """Tokenize the ability tags and weapon IDs in a DataFrame.

    This function processes the 'ability_tags' and 'weapon_id' columns of the
    input DataFrame, converting them into integer IDs for use in machine
    learning models.

    Args:
        df (pd.DataFrame): The DataFrame containing 'ability_tags' and
            'weapon_id' columns to tokenize.
        ability_to_id (Optional[dict[str, int]]): Pre-initialized mapping of
            ability tags to their corresponding integer IDs. If None, a new
            mapping will be created.
        weapon_to_id (Optional[dict[str, int]]): Pre-initialized mapping of
            weapon IDs to their corresponding integer IDs. If None, a new
            mapping will be created.

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

    if ability_to_id is None:
        ability_to_id = {}
    else:
        ability_to_id.pop(PAD, None)

    for ability in unique_abilities:
        if ability not in ability_to_id:
            ability_to_id[ability] = len(ability_to_id)

    ability_to_id[PAD] = len(ability_to_id)

    if weapon_to_id is None:
        weapon_to_id = {}
    else:
        new_weapon_to_id = {}
        for key, _ in weapon_to_id.items():
            new_key = key.split("_")[-1]
            new_weapon_to_id[new_key] = weapon_to_id[key]

        weapon_to_id = new_weapon_to_id

    for weapon in df["weapon_id"].unique():
        if weapon not in weapon_to_id:
            weapon_to_id[weapon] = len(weapon_to_id)

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
