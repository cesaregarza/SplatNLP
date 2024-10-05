import logging

import pandas as pd

from splatnlp.preprocessing.constants import MASK, PAD

logger = logging.getLogger(__name__)


def tokenize(
    df: pd.DataFrame,
) -> tuple[pd.Series, dict[str, int]]:
    """Tokenize the ability tags in a DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame containing ability tags to tokenize.

    Returns:
        tuple[pd.Series, dict[str, int]]: A tuple containing:
            - pd.Series: The tokenized ability tags as lists of integer IDs.
            - dict[str, int]: A mapping of ability tags to their corresponding
                integer IDs.
    """
    logger.info("Starting tokenize function")
    exploded_abilities = df["ability_tags"].str.split(" ").explode()
    unique_abilities = exploded_abilities.unique()
    ability_to_id = {ability: i for i, ability in enumerate(unique_abilities)}
    ability_to_id[MASK] = len(ability_to_id)
    ability_to_id[PAD] = len(ability_to_id)

    logger.debug("Mapping ability tags to integer IDs and grouping to lists")
    ability_tags_ids = (
        exploded_abilities.map(ability_to_id).groupby(level=0).agg(list)
    )
    logger.info("Tokenization complete")
    return ability_tags_ids, ability_to_id
