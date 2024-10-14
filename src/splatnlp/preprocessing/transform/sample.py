import logging

import pandas as pd

from splatnlp.preprocessing.constants import TARGET_WEAPON_WINRATE

logger = logging.getLogger(__name__)


def sample(
    df: pd.DataFrame,
    max_player_entries: int = 100,
    frac: float = 0.1,
    target_winrate: float | None = None,
    *,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Sample a DataFrame to address imbalances, introduce intentional bias, or
    reduce size for testing.

    This function performs several sampling operations on the input DataFrame:
    1. Limits the number of entries per player proxy field, `ability_hash`.
    2. Adjusts the win/loss ratio to match the target win rate (if specified).
    3. Randomly samples a fraction of the resulting data.

    Args:
        df (pd.DataFrame): The input DataFrame to sample. This should contain
            data for a single weapon only.
        max_player_entries (int, optional): The maximum number of entries to
            keep per ability_hash. Defaults to 100.
        frac (float, optional): The fraction of the final DataFrame to return.
            Defaults to 0.1.
        target_winrate (float | None, optional): The desired win rate for the
            weapon. If None, defaults to TARGET_WEAPON_WINRATE. Defaults to
            None.
        random_state (int | None, optional): Controls the randomness of the
            sampling. Pass an int for reproducible output. Defaults to None.

    Returns:
        pd.DataFrame: The sampled DataFrame.

    Note:
        - The function assumes the DataFrame has 'ability_hash' and 'win'
            columns.
        - The 'win' column is expected to be boolean, where True represents a
            win.
        - This function is designed to work on data for a single weapon. Using
            it on a DataFrame with multiple weapons may lead to unexpected
            results.
    """
    logger.info("Starting to sample DataFrame")
    logger.debug(f"Initial DataFrame shape: {df.shape}")

    if target_winrate is None:
        target_winrate = TARGET_WEAPON_WINRATE

    df = df.groupby("ability_hash").head(max_player_entries)
    logger.debug(f"DataFrame shape after grouping by ability_hash: {df.shape}")

    win_df = df[df["win"].astype(bool)]
    loss_df = df[~df["win"].astype(bool)]
    current_winrate = len(win_df) / len(df)
    logger.debug(f"Win DataFrame shape: {win_df.shape}")
    logger.debug(f"Loss DataFrame shape: {loss_df.shape}")
    logger.debug(f"Current win rate: {current_winrate:.2%}")

    if current_winrate < target_winrate:
        logger.info(
            f"Win rate below target ({target_winrate:.2%}). "
            "Sampling loss entries."
        )
        target_rows = int(len(win_df) / target_winrate)
        target_loss_rows = target_rows - len(win_df)
        sampled_loss_df = loss_df.sample(
            n=min(target_loss_rows, len(loss_df)), random_state=random_state
        )
    else:
        logger.info(
            f"Win rate at or above target ({target_winrate:.2%}). "
            "Sampling win entries."
        )
        target_rows = int(len(loss_df) / (1 - target_winrate))
        target_win_rows = target_rows - len(loss_df)
        sampled_win_df = win_df.sample(
            n=min(target_win_rows, len(win_df)), random_state=random_state
        )
        win_df = sampled_win_df
        sampled_loss_df = loss_df

    logger.debug(f"Sampled loss DataFrame shape: {sampled_loss_df.shape}")
    logger.debug(f"Final win DataFrame shape: {win_df.shape}")

    result_df = pd.concat([win_df, sampled_loss_df]).sample(
        frac=frac, random_state=random_state
    )
    logger.info(f"Finished sampling. Final DataFrame shape: {result_df.shape}")
    return result_df
