import logging

import pandas as pd

from splatnlp.etl.transform.create import add_columns, create_weapon_df
from splatnlp.etl.transform.process import process_abilities
from splatnlp.etl.transform.remove import remove
from splatnlp.etl.transform.sample import sample

logger = logging.getLogger(__name__)


def transform(
    df: pd.DataFrame,
    *,
    max_player_entries: int = 100,
    sample_frac: float = 0.1,
    loss_frac: float = 0.9,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Transform a DataFrame by applying several operations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        max_player_entries (int): Max entries per ability_hash. Defaults to 100.
        sample_frac (float): Fraction of final DataFrame to return.
            Defaults to 0.1.
        loss_frac (float): Fraction of losing entries to retain.
            Defaults to 0.9.
        random_state (int | None): Controls sampling randomness. Defaults to
            None (different output each time).

    Returns:
        pd.DataFrame: The transformed DataFrame.
    """
    logger.info("Starting to transform DataFrame")

    return (
        df.sample(frac=1, random_state=random_state)  # Shuffle the DataFrame
        .pipe(add_columns)
        .pipe(
            sample,
            max_player_entries=max_player_entries,
            frac=sample_frac,
            loss_frac=loss_frac,
            random_state=random_state,
        )
        .pipe(process_abilities)
        .pipe(remove)
    )
