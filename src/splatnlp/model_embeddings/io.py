from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import orjson
import pandas as pd


def load_json(path_str: str) -> dict:
    """Load JSON from a local path, URL, or S3 URI."""
    if path_str.startswith(("http://", "https://", "s3://")):
        if path_str.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            bucket, key = path_str[5:].split("/", 1)
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        else:
            import requests

            response = requests.get(path_str, timeout=30)
            response.raise_for_status()
            content = response.content
        return orjson.loads(content)

    path = Path(path_str)
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("rb") as handle:
        try:
            return orjson.loads(handle.read())
        except orjson.JSONDecodeError:
            handle.seek(0)
            return json.load(handle)


def load_data(
    data_path: str, table_name: str | None = None, max_rows: int | None = None
) -> pd.DataFrame:
    """Load tabular data from local file, URL, S3, or SQLite DB."""
    if data_path.startswith(("http://", "https://", "s3://")):
        if data_path.startswith("s3://"):
            import boto3

            s3 = boto3.client("s3")
            bucket, key = data_path[5:].split("/", 1)
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
            return pd.read_csv(
                io.BytesIO(content), sep="\t", header=0, nrows=max_rows
            )

        return pd.read_csv(
            data_path,
            sep="\t",
            header=0,
            nrows=max_rows,
        )

    if data_path.endswith((".db", ".sqlite")):
        limit_clause = f" LIMIT {max_rows}" if max_rows else ""
        query = f"SELECT * FROM {table_name or 'data'}{limit_clause}"
        with sqlite3.connect(data_path) as conn:
            return pd.read_sql_query(query, conn)

    return pd.read_csv(data_path, sep="\t", header=0, nrows=max_rows)


def load_tokenized_data(
    data_path: str,
    table_name: str | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
    """Load tokenized data with ability_tags parsed into lists."""
    df = load_data(data_path, table_name=table_name, max_rows=max_rows)
    if df.empty:
        raise ValueError(f"No rows found in {data_path}")
    if "ability_tags" not in df.columns or "weapon_id" not in df.columns:
        raise ValueError(
            "Expected columns 'ability_tags' and 'weapon_id' in tokenized data"
        )
    if df["ability_tags"].map(lambda v: isinstance(v, str)).any():
        df["ability_tags"] = df["ability_tags"].apply(
            lambda v: orjson.loads(v) if isinstance(v, str) else v
        )
    return df
