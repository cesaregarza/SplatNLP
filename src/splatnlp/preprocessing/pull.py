import io
import logging
import os
import sqlite3
import zipfile
from datetime import datetime
from typing import Literal, overload

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

BASE_URL = "https://dl-stats.stats.ink/splatoon-3/battle-results-csv/"

logger = logging.getLogger(__name__)


def pull_all() -> pd.DataFrame:
    logger.info("Starting to pull all data")
    URL = BASE_URL + "battle-results-csv.zip"
    response = requests.get(URL)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    dfs = []
    for filename in zip_file.namelist()[1:]:
        logger.debug(f"Processing file: {filename}")
        with zip_file.open(filename) as file:
            df = pd.read_csv(file, low_memory=False)
            dfs.append(df)

    result_df = pd.concat(dfs, ignore_index=True)
    logger.info(
        f"Finished pulling all data. DataFrame shape: {result_df.shape}"
    )
    return result_df


def update_data(base_path: str) -> pd.DataFrame:
    logger.info("Starting to update data")
    metadata_path = os.path.join(base_path, "metadata.txt")
    try:
        with open(metadata_path, "r") as f:
            last_period = f.read().strip()
        logger.debug(f"Last period found: {last_period}")
    except FileNotFoundError:
        logger.info("No metadata found. Pulling all data.")
        df = pull_all()
        return df

    parsed_period: datetime = pd.to_datetime(last_period)
    new_dfs = []

    while True:
        parsed_period += pd.DateOffset(days=1)
        URL = (
            BASE_URL
            + f"/{parsed_period.year:04d}"
            + f"/{parsed_period.month:02d}"
            + f"/{parsed_period.strftime('%Y-%m-%d')}.csv"
        )
        logger.debug(
            "Attempting to fetch data for: %s",
            parsed_period.strftime("%Y-%m-%d"),
        )
        response = requests.get(URL)
        if response.status_code == 404:
            logger.debug("No more data available. Breaking loop.")
            break
        response.raise_for_status()

        new_df = pd.read_csv(io.StringIO(response.text))
        new_dfs.append(new_df)

    if new_dfs:
        result_df = pd.concat(new_dfs, ignore_index=True)
        logger.info(
            f"Finished updating data. New DataFrame shape: {result_df.shape}"
        )
        return result_df
    else:
        logger.info("No new data to update.")
        return pd.DataFrame()


def save_data(df: pd.DataFrame, base_path: str) -> None:
    logger.info("Starting to save data")
    db_path = os.path.join(base_path, "statink", "data.sqlite")
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

    conn = sqlite3.connect(db_path)

    if not df.empty:
        df.to_sql("data", conn, if_exists="append", index=False)
        logger.debug(f"Data appended to {db_path}")

        metadata_path = os.path.join(base_path, "metadata.txt")
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        max_date = df["period"].max()
        with open(metadata_path, "w") as f:
            f.write(str(max_date))
        logger.debug(f"Metadata updated with max date: {max_date}")
    else:
        logger.info("No data to save.")

    conn.close()


@overload
def main(base_path: str, return_df: Literal[False]) -> None: ...


@overload
def main(base_path: str, return_df: Literal[True]) -> pd.DataFrame: ...


def main(base_path: str, return_df: bool = False) -> pd.DataFrame | None:
    logger.info("Starting main function")
    df = update_data(base_path)

    if return_df:
        return df

    save_data(df, base_path)
    logger.info("Main function completed")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_path = "data/"
    main(base_path)
