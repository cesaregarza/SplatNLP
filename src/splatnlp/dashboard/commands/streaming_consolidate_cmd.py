"""
Command for streaming consolidation of activation data into a database
and computing analytics in a streaming fashion.
"""

import argparse
import gc

# import json # Not directly used in the core functions moved, but db_manager might use it.
import logging
import pickle
from pathlib import Path
from typing import (  # Ensure all necessary types are imported
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
)

import h5py  # Used by stream_chunk_data if it were to handle .h5 chunks, but current handles .npz, .pkl
import numpy as np

# import pandas as pd # Not directly used in the core functions moved here
from tqdm import tqdm

# Assuming database_manager is in the same directory level or accessible via python path
# For commands, it's typical to use relative imports if part of the same package,
# or absolute if the package structure is well-defined and installed.
# from splatnlp.dashboard.database_manager import DashboardDatabase
# For now, to make it self-contained for refactoring, let's assume DashboardDatabase would be imported.
# If DashboardDatabase is complex, it should remain separate. For this exercise, we'll assume it's available.
# If it's not found during execution, that's a separate dependency issue.
try:
    from splatnlp.dashboard.database_manager import DashboardDatabase
except ImportError:
    # This is a fallback or placeholder. In a real system, this dependency needs to be resolvable.
    logger = logging.getLogger(__name__)
    logger.warning(
        "splatnlp.dashboard.database_manager.DashboardDatabase not found. Command may fail if database operations are called."
    )

    # Define a dummy class if not found, so the rest of the script can be parsed.
    class DashboardDatabase:
        def __init__(self, db_path):
            logger.warning(
                f"Using dummy DashboardDatabase for {db_path}. Operations will likely fail."
            )
            self.db_path = db_path

        def get_connection(self):
            raise NotImplementedError("Dummy DB")

        def insert_examples_batch(self, examples_batch):
            raise NotImplementedError("Dummy DB")

        def insert_activations_batch(
            self, activations_array, example_ids_batch, activation_threshold
        ):
            raise NotImplementedError("Dummy DB")

        def get_database_info(self):
            raise NotImplementedError("Dummy DB")

        def compute_and_store_feature_stats(self, feature_id):
            raise NotImplementedError("Dummy DB")

        def get_feature_activations(self, feature_id):
            raise NotImplementedError("Dummy DB")

        def store_top_examples(
            self, feature_id, example_ids, activations, top_k
        ):
            raise NotImplementedError("Dummy DB")


logger = logging.getLogger(__name__)


def stream_chunk_data(
    chunk_file: Path,
) -> Iterator[Tuple[int, Dict, np.ndarray]]:
    """Stream data from a chunk file one example at a time.

    Args:
        chunk_file: Path to chunk file (supports .npz and .pkl)

    Yields:
        Tuple of (example_index_in_chunk, metadata_dict, activation_vector)
    """
    logger.debug(f"Streaming data from chunk: {chunk_file}")
    if chunk_file.suffix == ".npz":
        # Handle NPZ files (typically from generate_activations_cmd chunked output)
        try:
            with np.load(chunk_file) as data:
                # 'hidden_activations' is the key used in generate_activations_cmd
                if "hidden_activations" not in data:
                    logger.warning(
                        f"'hidden_activations' key not found in NPZ file: {chunk_file}. Skipping."
                    )
                    return

                activations = data["hidden_activations"]
                n_examples_in_chunk = activations.shape[0]

                # Minimal metadata if NPZ doesn't store more detailed per-example info.
                # The original script created dummy metadata. We can enhance this if NPZ
                # chunks from generate_activations_cmd start including more metadata.
                for i in range(n_examples_in_chunk):
                    # Create a basic metadata dict. Actual content might vary.
                    # This part needs to align with what `process_single_chunk_streaming` expects.
                    metadata = {
                        "weapon_id": (
                            data.get(
                                "weapon_ids",
                                np.array([-1] * n_examples_in_chunk),
                            )[i]
                            if "weapon_ids" in data
                            else -1
                        ),
                        "weapon_name": (
                            data.get(
                                "weapon_names",
                                ["Unknown"] * n_examples_in_chunk,
                            )[i]
                            if "weapon_names" in data
                            else "Unknown"
                        ),
                        "ability_input_tokens": (
                            data.get(
                                "ability_input_tokens_list",
                                [[]] * n_examples_in_chunk,
                            )[i]
                            if "ability_input_tokens_list" in data
                            else []
                        ),
                        # Add other fields if they exist in your NPZ structure
                        "is_null_token": (
                            bool(
                                data.get(
                                    "is_null_token_flags",
                                    [False] * n_examples_in_chunk,
                                )[i]
                            )
                            if "is_null_token_flags" in data
                            else False
                        ),
                        "chunk_source_file": chunk_file.name,  # For tracking origin
                    }
                    yield i, metadata, activations[i]
        except Exception as e:
            logger.error(f"Error reading NPZ chunk {chunk_file}: {e}")
            return  # Stop iteration for this chunk

    elif chunk_file.suffix == ".pkl":
        # Handle PKL files (legacy or alternative chunk format)
        try:
            with open(chunk_file, "rb") as f:
                chunk_data = pickle.load(
                    f
                )  # Expects dict with 'activations' and 'metadata' (DataFrame)

            if (
                not isinstance(chunk_data, dict)
                or "activations" not in chunk_data
                or "metadata" not in chunk_data
            ):
                logger.warning(
                    f"Invalid PKL chunk format in {chunk_file}. Expected dict with 'activations' and 'metadata'. Skipping."
                )
                return

            activations = chunk_data["activations"]  # Should be np.ndarray
            metadata_df = chunk_data["metadata"]  # Should be pd.DataFrame

            if not isinstance(activations, np.ndarray) or not isinstance(
                metadata_df, pd.DataFrame
            ):
                logger.warning(
                    f"Invalid data types for activations or metadata in PKL chunk {chunk_file}. Skipping."
                )
                return
            if activations.shape[0] != len(metadata_df):
                logger.warning(
                    f"Mismatch in record count between activations ({activations.shape[0]}) and metadata ({len(metadata_df)}) in {chunk_file}. Skipping."
                )
                return

            for i in range(len(activations)):
                metadata = metadata_df.iloc[i].to_dict()
                metadata["chunk_source_file"] = (
                    chunk_file.name
                )  # For tracking origin
                yield i, metadata, activations[i]
        except Exception as e:
            logger.error(f"Error reading PKL chunk {chunk_file}: {e}")
            return  # Stop iteration for this chunk
    else:
        logger.warning(
            f"Unsupported chunk file type: {chunk_file.suffix} for file {chunk_file}. Skipping."
        )
        return


def process_single_chunk_streaming(
    chunk_file: Path,
    db: DashboardDatabase,
    starting_example_id: int,
    activation_threshold: float = 1e-6,  # Default from original
    batch_size: int = 1000,  # Default from original
) -> int:
    """Process a single chunk file in streaming fashion and insert into DB."""
    examples_batch_to_insert = []
    activations_batch_to_insert = []  # List of np.ndarray
    example_ids_for_activations_batch = []

    current_global_example_id = starting_example_id
    examples_processed_in_this_chunk = 0

    logger.info(
        f"Streaming data from chunk: {chunk_file.name}, starting global example ID: {current_global_example_id}"
    )

    for _, metadata_dict, activation_vector_data in stream_chunk_data(
        chunk_file
    ):
        # Prepare example data for DB insertion
        example_data_for_db = {
            "id": current_global_example_id,
            "weapon_id": metadata_dict.get("weapon_id", -1),
            "weapon_name": metadata_dict.get("weapon_name", "Unknown"),
            "ability_input_tokens": metadata_dict.get(
                "ability_input_tokens", []
            ),  # Ensure this is a list
            "input_abilities_str": metadata_dict.get(
                "input_abilities_str", ""
            ),  # Ensure this is a string
            "top_predicted_abilities_str": metadata_dict.get(
                "top_predicted_abilities_str", ""
            ),  # Ensure this is a string
            "is_null_token": bool(metadata_dict.get("is_null_token", False)),
            # Store other metadata fields in a JSON-able 'metadata' column or similar
            "metadata": {
                k: v
                for k, v in metadata_dict.items()
                if k
                not in [
                    "weapon_id",
                    "weapon_name",
                    "ability_input_tokens",
                    "input_abilities_str",
                    "top_predicted_abilities_str",
                    "is_null_token",
                ]
            },
        }

        examples_batch_to_insert.append(example_data_for_db)
        activations_batch_to_insert.append(
            activation_vector_data
        )  # Append the numpy array directly
        example_ids_for_activations_batch.append(current_global_example_id)

        current_global_example_id += 1
        examples_processed_in_this_chunk += 1

        if len(examples_batch_to_insert) >= batch_size:
            db.insert_examples_batch(examples_batch_to_insert)
            # Stack activations just before DB insertion
            activations_array_for_db = np.stack(
                activations_batch_to_insert, axis=0
            )
            db.insert_activations_batch(
                activations_array_for_db,
                example_ids_for_activations_batch,
                activation_threshold=activation_threshold,
            )
            examples_batch_to_insert.clear()
            activations_batch_to_insert.clear()
            example_ids_for_activations_batch.clear()
            gc.collect()  # Optional: force garbage collection

    # Process any remaining examples in the last batch
    if examples_batch_to_insert:
        db.insert_examples_batch(examples_batch_to_insert)
        if activations_batch_to_insert:  # Ensure there's something to stack
            activations_array_for_db = np.stack(
                activations_batch_to_insert, axis=0
            )
            db.insert_activations_batch(
                activations_array_for_db,
                example_ids_for_activations_batch,
                activation_threshold=activation_threshold,
            )

    logger.info(
        f"Finished processing chunk {chunk_file.name}. Processed {examples_processed_in_this_chunk} examples."
    )
    return examples_processed_in_this_chunk


def _streaming_consolidate_main_logic(
    cache_dir: Path,
    db_path: Path,
    activation_threshold: float,
    batch_size: int,
    resume: bool,
):
    """Core logic for streaming consolidation."""
    db = DashboardDatabase(db_path)  # Initializes DB schema if not exists

    # Discover chunk files (supports .pkl and .npz)
    chunk_files = sorted(
        list(cache_dir.glob("*.pkl")) + list(cache_dir.glob("*.npz"))
    )

    if not chunk_files:
        logger.error(
            f"No chunk files (.pkl or .npz) found in cache directory: {cache_dir}"
        )
        raise ValueError(f"No chunk files found in {cache_dir}")

    logger.info(
        f"Found {len(chunk_files)} chunk files for potential processing."
    )

    processed_chunk_filenames = set()
    current_global_example_id_start = 0  # Start from ID 0 unless resuming

    if resume and db_path.exists():  # Check if DB exists for resuming
        try:
            with db.get_connection() as conn:
                cursor = conn.execute("SELECT MAX(id) FROM examples")
                max_id_result = cursor.fetchone()
                if max_id_result is not None and max_id_result[0] is not None:
                    current_global_example_id_start = max_id_result[0] + 1

                # Determine already processed chunks by checking metadata stored in DB
                # Assumes 'chunk_source_file' is stored in example's metadata JSON field.
                cursor = conn.execute(
                    """
                    SELECT DISTINCT json_extract(metadata, '$.chunk_source_file') as chunk_filename 
                    FROM examples 
                    WHERE json_extract(metadata, '$.chunk_source_file') IS NOT NULL
                """
                )
                for row in cursor.fetchall():
                    if row[0]:  # Ensure filename is not null
                        processed_chunk_filenames.add(row[0])
            logger.info(
                f"Resuming. Starting next global example ID at {current_global_example_id_start}. Found {len(processed_chunk_filenames)} already processed chunk filenames in DB."
            )
        except (
            Exception
        ) as e:  # Catch broad exceptions for DB issues during resume setup
            logger.error(
                f"Error during resume setup (accessing DB {db_path}): {e}. Will attempt to run without resuming."
            )
            current_global_example_id_start = 0  # Reset if resume fails
            processed_chunk_filenames.clear()

    chunks_to_process_paths = [
        cf_path
        for cf_path in chunk_files
        if cf_path.name not in processed_chunk_filenames
    ]

    if not chunks_to_process_paths:
        logger.info("No new chunk files to process based on resume data.")
    else:
        logger.info(
            f"Processing {len(chunks_to_process_paths)} new chunk files."
        )

    total_examples_processed_this_run = 0
    for chunk_file_path_iter in tqdm(
        chunks_to_process_paths, desc="Streaming chunks to database"
    ):
        try:
            num_examples_from_chunk = process_single_chunk_streaming(
                chunk_file_path_iter,
                db,
                current_global_example_id_start,  # Pass the starting ID for this chunk
                activation_threshold,
                batch_size,
            )
            current_global_example_id_start += (
                num_examples_from_chunk  # Update for next chunk
            )
            total_examples_processed_this_run += num_examples_from_chunk
        except Exception as e:
            logger.error(
                f"Failed to process chunk {chunk_file_path_iter}: {e}",
                exc_info=True,
            )
            # Decide if to continue with next chunk or stop. For now, continue.

    logger.info(
        f"Streaming consolidation run finished. Processed {total_examples_processed_this_run} examples in this run."
    )

    try:
        db_info = db.get_database_info()
        logger.info(f"Current database info: {db_info}")
    except Exception as e:
        logger.error(f"Could not retrieve database info: {e}")


def _compute_analytics_streaming_main_logic(
    db_path: Path, features_per_batch: int, max_features: Optional[int]
):
    """Core logic for computing analytics in a streaming fashion from DB."""
    db = DashboardDatabase(db_path)

    all_feature_ids_from_db = []
    try:
        with db.get_connection() as conn:
            # Get distinct feature IDs that have non-zero activations stored
            cursor = conn.execute(
                "SELECT DISTINCT feature_id FROM activations ORDER BY feature_id"
            )
            all_feature_ids_from_db = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        logger.error(
            f"Failed to retrieve feature IDs from database {db_path}: {e}"
        )
        return

    if not all_feature_ids_from_db:
        logger.info(
            "No feature IDs found in the database. Analytics computation cannot proceed."
        )
        return

    if (
        max_features is not None and max_features > 0
    ):  # Apply max_features limit if specified
        all_feature_ids_to_process = all_feature_ids_from_db[:max_features]
    else:
        all_feature_ids_to_process = all_feature_ids_from_db

    num_features_for_analytics = len(all_feature_ids_to_process)
    if num_features_for_analytics == 0:
        logger.info("No features selected for analytics processing.")
        return

    logger.info(
        f"Computing streaming analytics for {num_features_for_analytics} features, in batches of {features_per_batch}."
    )

    for i in tqdm(
        range(0, num_features_for_analytics, features_per_batch),
        desc="Processing feature batches for analytics",
    ):
        current_batch_feature_ids = all_feature_ids_to_process[
            i : i + features_per_batch
        ]

        for feature_id_val in current_batch_feature_ids:
            try:
                # This computes stats (mean, std, sparsity, histogram) using SQL queries
                db.compute_and_store_feature_stats(feature_id_val)

                # This gets all activations for a feature to find top examples
                # Potentially memory intensive if a feature is very dense across many examples.
                # The DB method should handle this efficiently if possible.
                activations_for_feature, example_ids_for_feature = (
                    db.get_feature_activations(feature_id_val)
                )

                if (
                    activations_for_feature is not None
                    and len(activations_for_feature) > 0
                ):
                    # This stores top N examples based on activation values
                    # The DB method needs to fetch metadata for these top examples.
                    db.store_top_examples(
                        feature_id_val,
                        example_ids_for_feature,  # Original example IDs
                        activations_for_feature.tolist(),  # Pass activations as a list
                        top_k=50,  # Default top_k from original script
                    )
                else:
                    logger.debug(
                        f"No activations retrieved for feature {feature_id_val}, skipping top examples."
                    )
            except Exception as e:
                logger.error(
                    f"Error computing analytics for feature {feature_id_val}: {e}",
                    exc_info=True,
                )

        gc.collect()  # Optional: collect garbage after each batch of features

    logger.info("Streaming analytics computation finished.")


# Main command function
def streaming_consolidate_command(args: argparse.Namespace):
    """Main command function for streaming consolidation and analytics."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(
        f"Executing streaming_consolidate_command with command: {args.command}"
    )

    if args.command == "stream-to-db":
        if not all(
            hasattr(args, attr) for attr in ["cache_dir", "output_db_path"]
        ):
            logger.error(
                "For 'stream-to-db', --cache-dir and --output-db-path are required."
            )
            return
        # Use defaults for optional args if not provided by CLI
        activation_threshold = (
            args.activation_threshold
            if hasattr(args, "activation_threshold")
            else 1e-6
        )
        batch_size = args.batch_size if hasattr(args, "batch_size") else 1000
        resume = (
            args.resume if hasattr(args, "resume") else True
        )  # Default to resume

        _streaming_consolidate_main_logic(
            Path(args.cache_dir),
            Path(args.output_db_path),
            activation_threshold,
            batch_size,
            resume,
        )
    elif args.command == "compute-db-analytics":
        if not hasattr(args, "db_path"):
            logger.error("For 'compute-db-analytics', --db-path is required.")
            return
        features_per_batch = (
            args.features_per_batch
            if hasattr(args, "features_per_batch")
            else 100
        )
        max_features = (
            args.max_features if hasattr(args, "max_features") else None
        )

        _compute_analytics_streaming_main_logic(
            Path(args.db_path), features_per_batch, max_features
        )
    else:
        logger.error(
            f"Unknown command for streaming_consolidate_command: {args.command}"
        )
        # Consider printing help or raising an error here if integrated with a CLI parser.


# Example CLI setup (to be integrated into a main CLI script later)
# def main_cli_example():
#     parser = argparse.ArgumentParser(description="Streaming consolidation and analytics for activations.")
#     subparsers = parser.add_subparsers(dest='command', help='Sub-command to run', required=True)

#     # Stream to DB sub-command
#     stream_parser = subparsers.add_parser('stream-to-db', help='Stream activation chunks from files into a database.')
#     stream_parser.add_argument('--cache-dir', type=str, required=True, help='Directory containing cached .npz or .pkl chunk files.')
#     stream_parser.add_argument('--output-db-path', type=str, required=True, help='Path to the output SQLite database file.')
#     stream_parser.add_argument('--batch-size', type=int, default=1000, help='Number of examples to batch before database insertion.')
#     stream_parser.add_argument('--activation-threshold', type=float, default=1e-6, help='Minimum activation value to store in the database.')
#     stream_parser.add_argument('--no-resume', action='store_false', dest='resume', help='Do not attempt to resume from existing DB data.')

#     # Compute DB Analytics sub-command
#     analytics_parser = subparsers.add_parser('compute-db-analytics', help='Compute analytics (stats, top examples) from data in the database.')
#     analytics_parser.add_argument('--db-path', type=str, required=True, help='Path to the SQLite database file containing activations.')
#     analytics_parser.add_argument('--features-per-batch', type=int, default=100, help='Number of features to process per analytics batch.')
#     analytics_parser.add_argument('--max-features', type=int, default=None, help='Maximum number of features to process for analytics (optional, for testing).')

#     cli_args = parser.parse_args()
#     streaming_consolidate_command(cli_args)

# if __name__ == '__main__':
#     # main_cli_example() # For direct testing
#     pass
