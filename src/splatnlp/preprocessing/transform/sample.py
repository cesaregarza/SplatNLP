import logging

import pandas as pd

logger = logging.getLogger(__name__)


def sample(
    df: pd.DataFrame,
    max_player_entries: int = 100,
    frac: float = 0.1,
    loss_frac: float = 0.9,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sample a DataFrame to address imbalances, introduce intentional bias, or
    reduce size for testing.

    This function performs several sampling operations on the input DataFrame:
    1. Limits the number of entries per player proxy field, `ability_hash`.
    2. Undersamples the losing entries.
    3. Randomly samples a fraction of the resulting data.

    Args:
        df (pd.DataFrame): The input DataFrame to sample.
        max_player_entries (int, optional): The maximum number of entries to
            keep per ability_hash. Defaults to 100.
        frac (float, optional): The fraction of the final DataFrame to return.
            Defaults to 0.1.
        loss_frac (float, optional): The fraction of losing entries to retain.
            Defaults to 0.9.
        random_state (int | None, optional): Controls the randomness of the
            sampling. Pass an int for reproducible output. Defaults to None.

    Returns:
        pd.DataFrame: The sampled DataFrame.

    Note:
        - The function assumes the DataFrame has 'ability_hash' and 'win'
            columns.
        - The 'win' column is expected to be boolean, where True represents a
            win.
    """
    logger.info("Starting to sample DataFrame")
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    df = df.groupby("ability_hash").head(max_player_entries)
    logger.debug(f"DataFrame shape after grouping by ability_hash: {df.shape}")

    win_df = df[df["win"].astype(bool)]
    loss_df = df[~df["win"].astype(bool)]
    logger.debug(f"Win DataFrame shape: {win_df.shape}")
    logger.debug(f"Loss DataFrame shape: {loss_df.shape}")

    sampled_loss_df = loss_df.sample(frac=loss_frac, random_state=random_state)
    logger.debug(f"Sampled loss DataFrame shape: {sampled_loss_df.shape}")

    result_df = pd.concat([win_df, sampled_loss_df]).sample(
        frac=frac, random_state=random_state
    )
    logger.info(f"Finished sampling. Final DataFrame shape: {result_df.shape}")
    return result_df
