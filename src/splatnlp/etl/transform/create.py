from typing import TypeVar

import dask.dataframe as dd
import pandas as pd

from splatnlp.etl.extract.parse import generate_maps

S = TypeVar("S", pd.DataFrame, dd.DataFrame)
T = TypeVar("T")


def create_weapon_df(df: S) -> S:
    player_dfs = []
    for team in "AB":
        for player_no in range(1, 5):
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

    if isinstance(df, pd.DataFrame):
        concatenated_df = pd.concat(player_dfs)
    elif isinstance(df, dd.DataFrame):
        concatenated_df = dd.concat(player_dfs)
    else:
        raise TypeError(
            "Input dataframe must be either pandas or dask DataFrame"
        )

    return concatenated_df.merge(
        df[columns_to_keep],
        left_index=True,
        right_index=True,
    )


def add_columns(df: S) -> S:
    key_to_id, _ = generate_maps()
    df["weapon_id"] = df["weapon"].map(key_to_id)
    df["abilities_str"] = df["ability"].astype(str)

    return df
