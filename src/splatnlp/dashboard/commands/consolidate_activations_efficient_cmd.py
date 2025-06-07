"""Command for consolidating activation data efficiently and creating dashboard caches."""

import argparse
import gc
import json
import logging
import pickle

# import shutil # Not used in the functions being moved
from pathlib import Path
from typing import (  # Ensure all necessary types are imported
    Dict,
    List,
    Optional,
)

import h5py
import numpy as np
import pandas as pd  # Keep for metadata handling if consolidate_to_hdf5_streaming uses it
from tqdm import tqdm

logger = logging.getLogger(__name__)


def consolidate_to_hdf5_streaming(
    cache_dir: Path,
    output_file: Path,
    # batch_size: int = 50000 # This param was in original but not used by the function
):
    """Consolidate cached chunks into HDF5 file with streaming.

    Args:
        cache_dir: Directory containing cached chunks (from generate_activations_cmd chunked output - .npz files)
        output_file: Output HDF5 file for consolidated activations.
                     A corresponding .metadata.pkl file will also be created.
    """
    logger.info(
        f"Starting consolidation from cache directory: {cache_dir} to output: {output_file}"
    )

    # generate_activations_cmd (chunked mode) saves chunks as chunk_XXXX.npz
    # It also saves a main .metadata.joblib file and a pointer .pkl file at the "output_path"
    # This consolidation function is intended to be run if individual .npz chunks were somehow
    # generated separately OR if we want to re-consolidate from an existing set of .npz chunks.
    # The more common path is that generate_activations_cmd already creates the final HDF5.
    # However, providing this as a standalone command could be useful for recovery/custom pipelines.

    # We expect NPZ files from `generate_activations_cmd`'s chunking.
    # Each NPZ chunk should contain 'hidden_activations' and 'num_records'.
    # Metadata is typically handled by the main `generate_activations_cmd` output,
    # but if we are consolidating raw .npz, we need to construct some minimal metadata.

    chunk_files = sorted(cache_dir.glob("chunk_*.npz"))

    if not chunk_files:
        logger.error(f"No NPZ chunk files (chunk_*.npz) found in {cache_dir}")
        raise ValueError(f"No NPZ chunk files found in {cache_dir}")

    logger.info(f"Found {len(chunk_files)} NPZ chunk files to consolidate.")

    # Get dimensions from first chunk
    try:
        with np.load(chunk_files[0]) as first_chunk_data:
            if "hidden_activations" not in first_chunk_data:
                logger.error(
                    f"Key 'hidden_activations' not found in first chunk: {chunk_files[0]}"
                )
                raise KeyError(
                    f"Key 'hidden_activations' not found in first chunk: {chunk_files[0]}"
                )
            # Ensure it's a 2D array before accessing shape[1]
            if first_chunk_data["hidden_activations"].ndim != 2:
                logger.error(
                    f"'hidden_activations' in {chunk_files[0]} is not 2D (shape: {first_chunk_data['hidden_activations'].shape})"
                )
                raise ValueError(
                    f"'hidden_activations' in {chunk_files[0]} is not 2D"
                )
            n_features = first_chunk_data["hidden_activations"].shape[1]
    except Exception as e:
        logger.error(
            f"Could not read first chunk {chunk_files[0]} to determine dimensions: {e}"
        )
        raise

    # Count total examples and collect basic metadata from chunks
    total_examples = 0
    per_chunk_metadata_list = []  # Store metadata from each chunk if available

    logger.info(
        "Scanning chunks to count total examples and gather metadata..."
    )
    for i, chunk_file_path in enumerate(
        tqdm(chunk_files, desc="Scanning chunks")
    ):
        try:
            with np.load(chunk_file_path) as chunk_data_scan:
                num_records_in_chunk = int(
                    chunk_data_scan.get("num_records", 0)
                )  # Default to 0 if not found
                hidden_acts_shape = chunk_data_scan.get(
                    "hidden_activations", np.array([])
                ).shape

                if num_records_in_chunk == 0 and hidden_acts_shape[0] != 0:
                    logger.warning(
                        f"Chunk {chunk_file_path} has num_records=0 but hidden_activations shape {hidden_acts_shape}. Using shape[0]."
                    )
                    num_records_in_chunk = hidden_acts_shape[0]
                elif (
                    num_records_in_chunk != 0
                    and hidden_acts_shape[0] == 0
                    and num_records_in_chunk > 0
                ):
                    logger.warning(
                        f"Chunk {chunk_file_path} has num_records={num_records_in_chunk} but hidden_activations is empty. Assuming num_records is correct."
                    )
                elif (
                    num_records_in_chunk != hidden_acts_shape[0]
                    and hidden_acts_shape[0] != 0
                ):  # if hidden_acts_shape[0] is 0, num_records might be right
                    logger.warning(
                        f"Mismatch in chunk {chunk_file_path}: num_records={num_records_in_chunk}, activations shape={hidden_acts_shape}. Using num_records."
                    )

                total_examples += num_records_in_chunk
                # Minimal metadata for each example in this chunk
                for record_idx_in_chunk in range(num_records_in_chunk):
                    # This metadata is very basic, assuming raw NPZ consolidation
                    # A more complete solution would require metadata saved alongside chunks by the generator
                    per_chunk_metadata_list.append(
                        {
                            "original_chunk_filename": chunk_file_path.name,
                            "original_chunk_index_within_file": i,  # Index of the chunk file itself
                            "index_within_chunk": record_idx_in_chunk,
                            "global_example_id_in_consolidation": total_examples
                            - num_records_in_chunk
                            + record_idx_in_chunk,  # Tentative global ID
                        }
                    )
        except Exception as e:
            logger.error(
                f"Error scanning chunk {chunk_file_path}: {e}. Skipping this chunk for totals."
            )
            continue  # Skip corrupted chunk

    if total_examples == 0:
        logger.warning(
            "Total examples from chunks is 0. Output HDF5 will be empty."
        )
    if n_features == 0:  # Should ideally not happen if first chunk was valid
        logger.warning(
            "Number of features is 0. Output HDF5 will have 0 columns for activations."
        )

    logger.info(
        f"Consolidating {total_examples} total examples, {n_features} features per example."
    )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as hf:
        acts_dataset = hf.create_dataset(
            "activations",
            shape=(
                total_examples,
                n_features,
            ),  # Ensure shape is correctly determined
            dtype="float32",
            # Chunks for HDF5 - good defaults
            chunks=(
                min(1024, total_examples if total_examples > 0 else 1),
                n_features if n_features > 0 else 1,
            ),
            compression="gzip",
            compression_opts=4,  # Standard gzip compression level
        )

        current_example_write_idx = 0
        consolidated_metadata_records = []  # This will store the final metadata

        for i, chunk_file_path in enumerate(
            tqdm(chunk_files, desc="Writing chunks to HDF5")
        ):
            try:
                with np.load(chunk_file_path) as chunk_data_load:
                    chunk_acts_data = chunk_data_load.get("hidden_activations")
                    num_records_meta = int(
                        chunk_data_load.get("num_records", 0)
                    )

                    if chunk_acts_data is None:
                        logger.warning(
                            f"Chunk {chunk_file_path} missing 'hidden_activations'. Skipping."
                        )
                        continue
                    if (
                        chunk_acts_data.ndim != 2
                        or chunk_acts_data.shape[1] != n_features
                    ):
                        logger.warning(
                            f"Chunk {chunk_file_path} has unexpected activation shape {chunk_acts_data.shape} (expected N x {n_features}). Skipping."
                        )
                        continue

                    n_examples_in_current_chunk = chunk_acts_data.shape[0]

                    if (
                        n_examples_in_current_chunk == 0
                        and num_records_meta > 0
                    ):
                        # If activations are empty but num_records says there should be some,
                        # we might have an issue. For now, we write empty data if activations are empty.
                        logger.warning(
                            f"Chunk {chunk_file_path}: 'hidden_activations' is empty but num_records={num_records_meta}. Writing no activation data for this chunk's records."
                        )
                        # We still need to account for these "metadata-only" records if that's intended.
                        # For now, assuming if activations are empty, we don't write them.

                    if n_examples_in_current_chunk > 0:
                        if (
                            current_example_write_idx
                            + n_examples_in_current_chunk
                            > total_examples
                        ):
                            logger.error(
                                f"Exceeding total_examples ({total_examples}) while writing chunk {chunk_file_path}. Current write index {current_example_write_idx}, chunk size {n_examples_in_current_chunk}. Stopping."
                            )
                            break  # Avoid writing out of bounds

                        acts_dataset[
                            current_example_write_idx : current_example_write_idx
                            + n_examples_in_current_chunk
                        ] = chunk_acts_data

                    # Construct metadata for records in this chunk
                    for record_idx_in_chunk_iter in range(
                        n_examples_in_current_chunk
                    ):  # Iterate based on actual data written
                        consolidated_metadata_records.append(
                            {
                                "source_chunk_file": chunk_file_path.name,
                                "source_chunk_file_index": i,  # Original file order index
                                "index_within_source_chunk": record_idx_in_chunk_iter,
                                "consolidated_global_example_id": current_example_write_idx
                                + record_idx_in_chunk_iter,
                            }
                        )
                    current_example_write_idx += n_examples_in_current_chunk

            except Exception as e:
                logger.error(
                    f"Error processing or writing chunk {chunk_file_path}: {e}. Skipping."
                )
                continue  # Skip corrupted/problematic chunk
            finally:
                # Explicitly clear memory for large arrays
                # chunk_data_load should be out of scope, but manual gc can help in loops
                del chunk_acts_data  # if defined
                gc.collect()

        if current_example_write_idx != total_examples:
            logger.warning(
                f"Final write index {current_example_write_idx} does not match total expected examples {total_examples}. HDF5 file might be smaller or there were errors."
            )
            # If it's smaller, we might need to resize the dataset, but HDF5 should handle this if pre-allocated.
            # If current_example_write_idx is larger, that's a serious bug.

    # Save metadata (now using pandas DataFrame for robustness and common format)
    metadata_df = pd.DataFrame(consolidated_metadata_records)
    metadata_output_file = output_file.with_suffix(
        ".metadata.pkl"
    )  # Or .csv, .parquet
    try:
        metadata_df.to_pickle(
            metadata_output_file
        )  # Pandas pickle is generally fine
        logger.info(
            f"Consolidated metadata saved to {metadata_output_file} ({len(metadata_df)} records)"
        )
    except Exception as e:
        logger.error(f"Failed to save consolidated metadata: {e}")

    logger.info(
        f"Consolidation complete. Activations: {output_file}, Metadata: {metadata_output_file}"
    )
    return output_file, metadata_output_file


def create_dashboard_cache(
    activations_file: Path,  # HDF5 file from consolidation
    metadata_file: Path,  # Pickle file (DataFrame) from consolidation
    neuron_summary_dir: Path,  # Dir containing neuron_XXXXX.json files (from extract_top_examples)
    dashboard_cache_output_dir: Path,
    n_neurons_sample: int = 100,
    n_examples_sample: int = 50000,  # Added for flexibility
):
    """Create optimized cache files for dashboard quick loading."""
    logger.info(f"Creating dashboard cache in {dashboard_cache_output_dir}...")
    dashboard_cache_output_dir.mkdir(parents=True, exist_ok=True)

    # Load full metadata to know total number of examples
    try:
        metadata_df = pd.read_pickle(metadata_file)
        n_total_examples_from_meta = len(metadata_df)
        logger.info(
            f"Loaded metadata for {n_total_examples_from_meta} examples from {metadata_file}"
        )
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        return

    # Create sample for quick loading from the main HDF5 activations file
    try:
        with h5py.File(activations_file, "r") as hf:
            if "activations" not in hf:
                logger.error(
                    f"'activations' dataset not found in {activations_file}"
                )
                return
            acts_dataset = hf["activations"]
            n_total_examples_in_h5, n_total_features_in_h5 = acts_dataset.shape

            if n_total_examples_from_meta != n_total_examples_in_h5:
                logger.warning(
                    f"Mismatch in example count: metadata has {n_total_examples_from_meta}, HDF5 has {n_total_examples_in_h5}. Using HDF5 count for sampling."
                )

            actual_n_examples_to_use = n_total_examples_in_h5

            # Sample neuron indices (evenly spaced)
            # Ensure n_neurons_sample is not more than available features
            n_neurons_to_sample_actual = min(
                n_neurons_sample, n_total_features_in_h5
            )
            if n_neurons_to_sample_actual == 0:
                logger.warning(
                    f"No features in HDF5 file {activations_file}, cannot sample neurons."
                )
                sampled_neuron_indices = np.array([], dtype=int)
            elif n_neurons_to_sample_actual < 0:  # Should not happen with min
                logger.error(
                    f"n_neurons_sample is negative: {n_neurons_to_sample_actual}"
                )
                return
            else:
                sampled_neuron_indices = np.linspace(
                    0,
                    n_total_features_in_h5 - 1,
                    n_neurons_to_sample_actual,
                    dtype=int,
                )

            # Sample example indices (randomly)
            # Ensure n_examples_sample is not more than available examples
            n_examples_to_sample_actual = min(
                n_examples_sample, actual_n_examples_to_use
            )
            if n_examples_to_sample_actual == 0:
                logger.warning(
                    f"No examples in HDF5 file {activations_file} or n_examples_sample is 0, cannot sample examples."
                )
                sampled_example_indices = np.array([], dtype=int)
                sample_activations_matrix = np.zeros(
                    (0, n_neurons_to_sample_actual), dtype=np.float32
                )
            elif n_examples_to_sample_actual < 0:
                logger.error(
                    f"n_examples_sample is negative: {n_examples_to_sample_actual}"
                )
                return
            else:
                sampled_example_indices = np.random.choice(
                    actual_n_examples_to_use,
                    n_examples_to_sample_actual,
                    replace=False,
                )
                sampled_example_indices.sort()  # Good practice for HDF5 access patterns

                # Create sample activation matrix
                # Pre-allocate if there are neurons and examples to sample
                if (
                    n_neurons_to_sample_actual > 0
                    and n_examples_to_sample_actual > 0
                ):
                    sample_activations_matrix = np.zeros(
                        (
                            n_examples_to_sample_actual,
                            n_neurons_to_sample_actual,
                        ),
                        dtype=np.float32,
                    )
                    for i, neuron_idx_val in enumerate(
                        tqdm(
                            sampled_neuron_indices,
                            desc="Sampling neuron activations",
                        )
                    ):
                        sample_activations_matrix[:, i] = acts_dataset[
                            sampled_example_indices, neuron_idx_val
                        ]
                else:  # Handle cases where no neurons or no examples are sampled
                    sample_activations_matrix = np.zeros(
                        (
                            n_examples_to_sample_actual,
                            n_neurons_to_sample_actual,
                        ),
                        dtype=np.float32,
                    )

    except Exception as e:
        logger.error(
            f"Error during HDF5 processing or sampling for dashboard cache: {e}"
        )
        return

    # Save sample activation data
    dashboard_sample_file = (
        dashboard_cache_output_dir / "dashboard_sample_activations.npz"
    )
    try:
        np.savez_compressed(
            dashboard_sample_file,
            activations=sample_activations_matrix,
            sampled_neuron_indices=sampled_neuron_indices,
            sampled_example_indices=sampled_example_indices,  # Global indices from HDF5
        )
        logger.info(
            f"Dashboard sample activations saved to {dashboard_sample_file}"
        )
    except Exception as e:
        logger.error(f"Failed to save dashboard sample activations: {e}")
        return  # Don't proceed if this fails

    # Save corresponding sample metadata (subset of the full metadata)
    # Ensure sampled_example_indices are valid for metadata_df if counts mismatched earlier
    # Assuming metadata_df indices align with HDF5 indices up to min(len(metadata_df), n_total_examples_in_h5)
    valid_sampled_example_indices_for_meta = [
        idx for idx in sampled_example_indices if idx < len(metadata_df)
    ]
    if len(valid_sampled_example_indices_for_meta) != len(
        sampled_example_indices
    ):
        logger.warning(
            "Some sampled example indices were out of bounds for the loaded metadata. Sampled metadata will be smaller."
        )

    sample_metadata_df = metadata_df.iloc[
        valid_sampled_example_indices_for_meta
    ].copy()
    # Add a column for the original HDF5 index if it's not already the DataFrame index
    sample_metadata_df["original_hdf5_index"] = (
        valid_sampled_example_indices_for_meta
    )

    sample_metadata_output_file = (
        dashboard_cache_output_dir / "dashboard_sample_metadata.pkl"
    )
    try:
        sample_metadata_df.to_pickle(sample_metadata_output_file)
        logger.info(
            f"Dashboard sample metadata saved to {sample_metadata_output_file}"
        )
    except Exception as e:
        logger.error(f"Failed to save dashboard sample metadata: {e}")
        # Allow to continue to save index file

    # Create main index file for the dashboard cache
    dashboard_index_data = {
        "n_total_neurons": (
            n_total_features_in_h5
            if "n_total_features_in_h5" in locals()
            else -1
        ),
        "n_total_examples": (
            actual_n_examples_to_use
            if "actual_n_examples_to_use" in locals()
            else -1
        ),
        "n_neurons_in_sample": (
            len(sampled_neuron_indices)
            if "sampled_neuron_indices" in locals()
            else -1
        ),
        "n_examples_in_sample": (
            len(sampled_example_indices)
            if "sampled_example_indices" in locals()
            else -1
        ),
        "path_to_full_activations_h5": str(activations_file.resolve()),
        "path_to_full_metadata_pkl": str(metadata_file.resolve()),
        "path_to_neuron_summaries_dir": str(
            neuron_summary_dir.resolve()
        ),  # From extract_top_examples
        "path_to_sample_activations_npz": str(dashboard_sample_file.resolve()),
        "path_to_sample_metadata_pkl": str(
            sample_metadata_output_file.resolve()
        ),
    }

    dashboard_index_file = (
        dashboard_cache_output_dir / "dashboard_cache_index.json"
    )
    try:
        with open(dashboard_index_file, "w") as f:
            json.dump(dashboard_index_data, f, indent=2)
        logger.info(f"Dashboard cache index saved to {dashboard_index_file}")
    except Exception as e:
        logger.error(f"Failed to save dashboard cache index: {e}")

    logger.info("Dashboard cache creation process finished.")


def consolidate_activations_efficient_command(args: argparse.Namespace):
    """Main command function for consolidating activations and creating dashboard caches."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info(
        f"Executing consolidate_activations_efficient_command with command: {args.command}"
    )

    if args.command == "consolidate":
        if not hasattr(args, "cache_dir") or not hasattr(args, "output_hdf5"):
            logger.error(
                "For 'consolidate' command, --cache-dir and --output-hdf5 are required."
            )
            return
        consolidate_to_hdf5_streaming(
            Path(args.cache_dir),
            Path(args.output_hdf5),
            # args.batch_size # This arg was not used in the original consolidate_to_hdf5_streaming
        )
    elif args.command == "create-dashboard-cache":
        if not all(
            hasattr(args, attr)
            for attr in [
                "consolidated_activations_h5",
                "consolidated_metadata_pkl",
                "neuron_summaries_dir",
                "dashboard_cache_out_dir",
            ]
        ):
            logger.error(
                "Missing one or more required arguments for 'create-dashboard-cache' command."
            )
            return

        # Use defaults for optional args if not provided
        n_neurons_sample = (
            args.n_neurons_sample
            if hasattr(args, "n_neurons_sample")
            and args.n_neurons_sample is not None
            else 100
        )
        n_examples_sample = (
            args.n_examples_sample
            if hasattr(args, "n_examples_sample")
            and args.n_examples_sample is not None
            else 50000
        )

        create_dashboard_cache(
            Path(args.consolidated_activations_h5),
            Path(args.consolidated_metadata_pkl),
            Path(args.neuron_summaries_dir),
            Path(args.dashboard_cache_out_dir),
            n_neurons_sample=n_neurons_sample,
            n_examples_sample=n_examples_sample,
        )
    # The 'extract' subcommand from the original script is handled by extract_top_examples_cmd.py
    # and should not be called from here to avoid circular dependencies or misplaced logic.
    else:
        logger.error(
            f"Unknown command for consolidate_activations_efficient_command: {args.command}"
        )
        # It might be good to print help here if parser was available.


# Example CLI setup (to be integrated into a main CLI script later)
# def main_cli_example():
#     parser = argparse.ArgumentParser(description="Consolidate activations efficiently and manage dashboard caches.")
#     subparsers = parser.add_subparsers(dest='command', help='Sub-command to run', required=True)

#     # Consolidate command
#     consolidate_parser = subparsers.add_parser('consolidate', help='Consolidate chunked .npz activation files into a single HDF5 file.')
#     consolidate_parser.add_argument('--cache-dir', type=str, required=True, help='Directory containing .npz chunk files (e.g., from generate_activations_cmd chunked output).')
#     consolidate_parser.add_argument('--output-hdf5', type=str, required=True, help='Output HDF5 file path for consolidated activations. A .metadata.pkl file will be created alongside.')
#     # consolidate_parser.add_argument('--batch-size', type=int, default=50000, help='Batch size for processing (currently unused).') # Original arg, but unused

#     # Create dashboard cache command
#     cache_parser = subparsers.add_parser('create-dashboard-cache', help='Create cache files optimized for quick dashboard loading.')
#     cache_parser.add_argument('--consolidated-activations-h5', type=str, required=True, help='Path to the consolidated HDF5 activations file.')
#     cache_parser.add_argument('--consolidated-metadata-pkl', type=str, required=True, help='Path to the consolidated .metadata.pkl file (often created by the consolidate step).')
#     cache_parser.add_argument('--neuron-summaries-dir', type=str, required=True, help='Directory containing per-neuron summary JSON files (output from extract_top_examples_cmd).')
#     cache_parser.add_argument('--dashboard-cache-out-dir', type=str, required=True, help='Output directory where dashboard cache files will be stored.')
#     cache_parser.add_argument('--n-neurons-sample', type=int, default=100, help='Number of neurons to include in the dashboard sample.')
#     cache_parser.add_argument('--n-examples-sample', type=int, default=50000, help='Number of examples to include in the dashboard sample.')

#     cli_args = parser.parse_args()
#     consolidate_activations_efficient_command(cli_args)

# if __name__ == '__main__':
#     # main_cli_example() # For direct testing
#     pass
