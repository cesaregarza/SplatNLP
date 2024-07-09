import io
import sqlite3
import zipfile
from datetime import datetime

import pandas as pd
import requests

BASE_URL = "https://dl-stats.stats.ink/splatoon-3/battle-results-csv/"


def pull_all(db: sqlite3.Connection) -> None:
    URL = BASE_URL + "battle-results-csv.zip"
    response = requests.get(URL)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))

    for filename in zip_file.namelist():
        with zip_file.open(filename) as file:
            df = pd.read_csv(file)
            df.to_sql(
                "StatInkBattleResults", db, if_exists="append", index=False
            )

    db.commit()


def update_db(db: sqlite3.Connection) -> None:
    cursor = db.cursor()
    cursor.execute("SELECT MAX(period) FROM StatInkBattleResults")
    last_period = cursor.fetchone()[0]
    parsed_period: datetime = pd.to_datetime(last_period)

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

        df = pd.read_csv(io.StringIO(response.text))
        df.to_sql("StatInkBattleResults", db, if_exists="append", index=False)

    db.commit()
