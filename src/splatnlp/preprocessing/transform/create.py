import logging

import pandas as pd
import xxhash

from splatnlp.preprocessing.transform.mappings import generate_maps

logger = logging.getLogger(__name__)


def create_weapon_df(df: pd.DataFrame) -> pd.DataFrame:
    """Create a weapon DataFrame by processing and combining player data from
    both teams.

    Args:
        df (pd.DataFrame): The input DataFrame containing game data.

    Returns:
        pd.DataFrame: A new DataFrame with individual player data and game
            information.

    This function processes the input DataFrame to extract individual player
    data from both teams, combines it into a single DataFrame, and merges it
    with relevant game information.
    """
    logger.info("Starting create_weapon_df function")
    player_dfs = []
    for team in "AB":
        for player_no in range(1, 5):
            logger.debug(f"Processing team {team}, player {player_no}")
            player_cols = [
                col for col in df.columns if f"{team}{player_no}" in col
            ]
            player_df = (
                df.loc[:, player_cols]
                .copy()
                .rename(
                    columns={
                        col: col.replace(f"{team}{player_no}-", "")
                        for col in df.columns
                    }
                )
                .assign(player_no=player_no, team=team)
            )
            player_dfs.append(player_df)

    columns_to_keep = ["period", "game-ver", "lobby", "mode", "win"]
    logger.debug(f"Columns to keep: {columns_to_keep}")

    concatenated_df = pd.concat(player_dfs)
    logger.info(f"Concatenated DataFrame shape: {concatenated_df.shape}")

    result = concatenated_df.merge(
        df[columns_to_keep],
        left_index=True,
        right_index=True,
    )
    logger.info(f"Final DataFrame shape: {result.shape}")
    return result


def add_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add new columns to the DataFrame including weapon_id, ability_hash, and
    win.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The DataFrame with new columns added.

    This function adds weapon_id based on a mapping, computes an ability_hash,
    and determines the win status for each row. It also drops rows with missing
    abilities.
    """
    logger.info("Starting add_columns function")
    key_to_id, _, _ = generate_maps()
    df["weapon_id"] = df["weapon"].map(key_to_id)

    # Drop rows with missing abilities
    df.dropna(subset=["abilities"], inplace=True)

    df["ability_hash"] = (
        df["abilities"]
        .apply(compute_ability_to_id)
        .astype(str)
        .add(df["weapon_id"])
    )
    df["win"] = df["win"].str[0].str.upper().eq(df["team"])
    logger.info(f"Final DataFrame shape: {df.shape}")
    return df


def compute_ability_to_id(abilities: str) -> int:
    """Compute a hash value for the given abilities string.

    Args:
        abilities (str): A string representation of abilities.

    Returns:
        int: An integer hash value computed from the abilities string.

    This function uses the xxhash algorithm to compute a 128-bit hash of the
    input abilities string, and returns it as an integer.
    """
    return xxhash.xxh128(abilities).intdigest()
