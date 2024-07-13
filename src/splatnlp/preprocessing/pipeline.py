import logging
import os

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from tqdm import tqdm

from splatnlp.preprocessing.pull import main as pull_main
from splatnlp.preprocessing.save import save
from splatnlp.preprocessing.transform import transform
from splatnlp.preprocessing.transform.create import create_weapon_df

logger = logging.getLogger(__name__)


def run(base_path: str | None = None, persist: bool = False) -> None:
    logger.info("Starting pipeline run")
    base_path = base_path or "data/"
    logger.info("Base path: %s", base_path)

    # Pull data
    if persist:
        logger.info("Persisting data to disk")
        pull_main(base_path, False)
        logger.info("Reading persisted data from CSV")
        base_df = pd.read_csv(os.path.join(base_path, "statink", "data.csv"))
    else:
        logger.info("Pulling data without persisting")
        base_df = pull_main(base_path, return_df=True)

    logger.info("Creating weapon DataFrame")
    table = create_weapon_df(base_df).pipe(pa.Table.from_pandas)
    logger.info("Deleting base DataFrame to free memory")
    del base_df

    output_path = os.path.join(base_path, "weapon_partitioned.parquet")
    logger.info("Writing partitioned dataset to: %s", output_path)
    pq.write_to_dataset(
        table,
        root_path=output_path,
        partition_cols=["weapon"],
        compression="snappy",
    )
    # MUST be written to disk before calling transform otherwise anything but
    # the largest of machines will run out of memory
    logger.info("Deleting table to free memory")
    del table

    logger.info("Creating dataset from partitioned Parquet file")
    dataset = ds.dataset(output_path, format="parquet")
    partitions = list(dataset.get_fragments())
    logger.info("Number of partitions: %d", len(partitions))

    for partition in tqdm(partitions, desc="Processing partitions"):
        logger.debug("Processing partition: %s", partition.path)
        df = partition.to_table().to_pandas()
        weapon = os.path.dirname(partition.path).split("/")[-1][
            len("weapon=") :
        ]
        df["weapon"] = weapon
        logger.debug("Transforming and saving data for weapon: %s", weapon)
        (transform(df).pipe(save, f"{base_path}/weapon_partitioned.csv"))
        logger.debug("Removing partition file: %s", partition.path)
        os.remove(partition.path)

    logger.info("Finished preprocessing pipeline run")
