import logging

import pandas as pd
import xxhash

from splatnlp.preprocessing.transform.parse import generate_maps

logger = logging.getLogger(__name__)


def create_weapon_df(df: pd.DataFrame) -> pd.DataFrame:
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

    columns_to_keep = ["lobby", "mode", "win"]
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
    logger.info("Starting add_columns function")
    key_to_id, _ = generate_maps()
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
    return xxhash.xxh128(abilities).intdigest()
