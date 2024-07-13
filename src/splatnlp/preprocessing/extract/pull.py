import io
import os
import zipfile
from datetime import datetime

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

BASE_URL = "https://dl-stats.stats.ink/splatoon-3/battle-results-csv/"


def pull_all() -> pd.DataFrame:
    URL = BASE_URL + "battle-results-csv.zip"
    response = requests.get(URL)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    dfs = []
    for filename in zip_file.namelist()[1:]:
        with zip_file.open(filename) as file:
            df = pd.read_csv(file)
            dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


def update_data(base_path: str) -> pd.DataFrame:
    metadata_path = os.path.join(base_path, "metadata.txt")
    try:
        with open(metadata_path, "r") as f:
            last_period = f.read().strip()
    except FileNotFoundError:
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
        response = requests.get(URL)
        if response.status_code == 404:
            break
        response.raise_for_status()

        new_df = pd.read_csv(io.StringIO(response.text))
        new_dfs.append(new_df)

    if new_dfs:
        return pd.concat(new_dfs, ignore_index=True)
    else:
        return pd.DataFrame()


def save_data(df: pd.DataFrame, base_path: str) -> None:
    if not df.empty:
        table = pa.Table.from_pandas(df)
        pq.write_to_dataset(
            table,
            root_path=os.path.join(base_path, "statink"),
            partition_cols=["weapon"],
        )

        max_date = df["period"].max()
        with open(os.path.join(base_path, "metadata.txt"), "w") as f:
            f.write(str(max_date))


def main(base_path: str):
    df = update_data(base_path)
    save_data(df, base_path)


if __name__ == "__main__":
    base_path = "data/"
    main(base_path)
