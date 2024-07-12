import pandas as pd


def sample(
    df: pd.DataFrame,
    max_player_entries: int = 100,
    frac: float = 0.1,
    loss_frac: float = 0.9,
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

    Returns:
        pd.DataFrame: The sampled DataFrame.

    Note:
        - The function assumes the DataFrame has 'ability_hash' and 'win'
            columns.
        - The 'win' column is expected to be boolean, where True represents a
            win.
    """
    df = df.groupby("ability_hash").head(max_player_entries)
    win_df = df[df["win"]]
    loss_df = df[~df["win"]]
    sampled_loss_df = loss_df.sample(frac=loss_frac)
    return pd.concat([win_df, sampled_loss_df]).sample(frac=frac)
