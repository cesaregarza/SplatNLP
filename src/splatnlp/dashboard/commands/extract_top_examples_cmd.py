"""Command for parallel CPU-optimized extraction of top examples per activation range."""

import argparse
import csv
import gc
import json  # Using standard json for dumps
import logging
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)


def compute_activation_ranges(
    acts: np.ndarray, n_bins: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute activation range boundaries and bin indices with optimizations."""
    if len(acts) == 0:
        return np.array([0.0, 1.0]), np.array([])

    min_val, max_val = acts.min(), acts.max()
    hi = max_val + 1e-6 if min_val != max_val else max_val + 1.0
    lo = min_val

    bounds = np.linspace(lo, hi, n_bins + 1)

    if len(bounds) < 2:
        bounds = np.array([lo, hi])

    bin_indices = np.searchsorted(bounds, acts, side="right") - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return bounds, bin_indices


def compute_neuron_statistics(acts: np.ndarray) -> Dict[str, Any]:
    """Optimized computation of neuron statistics."""
    if len(acts) == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "q25": 0.0,
            "q75": 0.0,
            "n_zeros": 0,
            "n_total": 0,
            "sparsity": 1.0,
        }

    percentiles = np.percentile(acts, [25, 50, 75])
    n_zeros = np.count_nonzero(acts == 0)

    return {
        "mean": float(acts.mean()),
        "std": float(acts.std()),
        "min": float(acts.min()),
        "max": float(acts.max()),
        "median": float(percentiles[1]),
        "q25": float(percentiles[0]),
        "q75": float(percentiles[2]),
        "n_zeros": int(n_zeros),
        "n_total": len(acts),
        "sparsity": float(n_zeros / len(acts)) if len(acts) > 0 else 1.0,
    }


def extract_top_examples_per_range(
    acts: np.ndarray,
    n_bins: int = 10,
    top_k: int = 1000,
    metadata_list: List[Dict] = None,
) -> Dict[str, Dict[str, Any]]:
    """Extract top K examples for each activation range."""
    if metadata_list is None:
        metadata_list = []

    bounds, bin_indices = compute_activation_ranges(acts, n_bins)
    results = {}
    original_indices = np.arange(len(acts))

    for bin_idx_loop in range(n_bins):
        mask = bin_indices == bin_idx_loop
        bin_count = np.sum(mask)
        current_bin_bounds = (
            (float(bounds[bin_idx_loop]), float(bounds[bin_idx_loop + 1]))
            if bin_idx_loop < len(bounds) - 1
            else (float(bounds[bin_idx_loop]), float(bounds[bin_idx_loop]))
        )

        if bin_count == 0:
            results[str(bin_idx_loop)] = {
                "bounds": current_bin_bounds,
                "count": 0,
                "examples": [],
            }
            continue

        bin_original_indices = original_indices[mask]
        bin_acts = acts[mask]
        k_actual = min(top_k, len(bin_acts))

        if k_actual == 0:
            top_k_local_indices = np.array([], dtype=int)
        elif k_actual < len(bin_acts):
            top_k_local_indices = np.argpartition(bin_acts, -k_actual)[
                -k_actual:
            ]
            top_k_local_indices = top_k_local_indices[
                np.argsort(bin_acts[top_k_local_indices])[::-1]
            ]
        else:
            top_k_local_indices = np.argsort(bin_acts)[::-1]

        top_global_indices = bin_original_indices[top_k_local_indices]
        top_examples = []
        for glob_idx in top_global_indices:
            meta = (
                metadata_list[glob_idx] if glob_idx < len(metadata_list) else {}
            )
            example_data = {
                "index": int(glob_idx),
                "activation": float(acts[glob_idx]),
                "metadata": {
                    "text": meta.get(
                        "text", "Error: metadata missing" if not meta else ""
                    ),
                    "weapon_id": meta.get("weapon_id", -1),
                    "label": meta.get(
                        "label", "Error: metadata missing" if not meta else ""
                    ),
                },
            }
            top_examples.append(example_data)
        results[str(bin_idx_loop)] = {
            "bounds": current_bin_bounds,
            "count": int(bin_count),
            "examples": top_examples,
        }
    return results


_metadata_list_global = None


def init_worker(metadata_list_for_worker: List[Dict]):
    global _metadata_list_global
    _metadata_list_global = metadata_list_for_worker


def process_single_neuron(args_tuple):
    (
        neuron_idx,
        activations_file_path_str,
        n_bins,
        top_k_per_bin,
        output_dir_path_str,
        consolidated_output_active,
    ) = args_tuple
    activations_file = Path(activations_file_path_str)
    global _metadata_list_global

    try:
        with h5py.File(activations_file, "r") as hf:
            if "activations" not in hf:
                return (
                    neuron_idx,
                    f"Error: 'activations' dataset not found for neuron {neuron_idx}.",
                )
            if neuron_idx >= hf["activations"].shape[1]:
                return (
                    neuron_idx,
                    f"Error: neuron_idx {neuron_idx} out of bounds.",
                )
            acts = hf["activations"][:, neuron_idx]

        range_data = extract_top_examples_per_range(
            acts, n_bins, top_k_per_bin, _metadata_list_global
        )
        stats = compute_neuron_statistics(acts)
        neuron_data_payload = {
            "neuron_id": neuron_idx,
            "statistics": stats,
            "range_examples": range_data,
        }

        if consolidated_output_active:
            return neuron_idx, neuron_data_payload
        else:
            output_dir = Path(output_dir_path_str)
            output_dir.mkdir(parents=True, exist_ok=True)
            neuron_file = output_dir / f"neuron_{neuron_idx:05d}.json"
            with open(neuron_file, "w") as f:
                f.write(json.dumps(neuron_data_payload))
            return neuron_idx, True
    except Exception as e:
        logger.error(
            f"Error processing neuron {neuron_idx}: {e}", exc_info=True
        )
        return neuron_idx, f"Error: {str(e)}"


def find_low_activation_examples_optimized(
    acts_matrix: np.ndarray,
    n_neurons_to_consider: int,
    n_bins: int = 10,
    bottom_bins_threshold: int = 2,
    max_examples_to_return: int = 1000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if acts_matrix.shape[0] == 0 or n_neurons_to_consider == 0:
        return np.array([]), np.array([]), np.array([])
    relevant_acts_matrix = acts_matrix[:, :n_neurons_to_consider]
    if relevant_acts_matrix.shape[0] == 0:
        return np.array([]), np.array([]), np.array([])

    threshold_percentile = (bottom_bins_threshold / n_bins) * 100
    thresholds = np.percentile(
        relevant_acts_matrix, threshold_percentile, axis=0
    )
    low_mask = np.all(relevant_acts_matrix <= thresholds, axis=1)
    low_indices_in_chunk = np.where(low_mask)[0]

    if len(low_indices_in_chunk) > max_examples_to_return:
        low_indices_in_chunk = np.random.choice(
            low_indices_in_chunk, max_examples_to_return, replace=False
        )

    if len(low_indices_in_chunk) > 0:
        max_acts = np.max(relevant_acts_matrix[low_indices_in_chunk, :], axis=1)
        mean_acts = np.mean(
            relevant_acts_matrix[low_indices_in_chunk, :], axis=1
        )
    else:
        max_acts, mean_acts = np.array([]), np.array([])
    return low_indices_in_chunk, max_acts, mean_acts


def extract_top_examples_command(args: argparse.Namespace):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Executing extract_top_examples_command")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_output_active = args.consolidated_output_file is not None
    csv_writer = None
    csv_file_handle = None
    successfully_processed_neurons_for_csv = 0

    # Correctly interpret --no-resume flag
    # If --no-resume is passed, args.no_resume is True, so should_resume is False.
    # If --no-resume is NOT passed, args.no_resume is False, so should_resume is True.
    should_resume = not args.no_resume

    if consolidated_output_active:
        logger.info(
            f"Consolidated CSV output requested at {args.consolidated_output_file}"
        )
        try:
            file_exists = Path(args.consolidated_output_file).exists()
            open_mode = "a" if should_resume and file_exists else "w"

            csv_file_handle = open(
                args.consolidated_output_file, open_mode, newline=""
            )
            csv_writer = csv.writer(csv_file_handle)

            if open_mode == "w" or not file_exists:
                csv_writer.writerow(
                    [
                        "feature_id",
                        "bin_index",
                        "bin_min_activation",
                        "bin_max_activation",
                        "example_id",
                        "activation_value",
                        "rank_in_bin",
                    ]
                )
                if open_mode == "w":
                    logger.info(
                        f"Consolidated CSV output will overwrite or create {args.consolidated_output_file}"
                    )
            else:  # open_mode == 'a' and file_exists
                logger.info(
                    f"Consolidated CSV output will be appended to {args.consolidated_output_file}"
                )
        except IOError as e:
            logger.error(
                f"Failed to open consolidated CSV file {args.consolidated_output_file}: {e}"
            )
            if csv_file_handle:
                csv_file_handle.close()
            return

    activations_file = Path(args.activations)
    metadata_file = Path(args.metadata)
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else max(1, mp.cpu_count() - 2 if mp.cpu_count() > 2 else 1)
    )
    logger.info(f"Using {num_workers} parallel workers")

    logger.info("Loading metadata...")
    start_time = time.time()
    try:
        with open(metadata_file, "rb") as f:
            metadata_payload = pickle.load(f)
    except Exception as e:
        logger.error(f"Failed to load metadata from {metadata_file}: {e}")
        return

    if isinstance(metadata_payload, list):
        metadata_list_for_processing = metadata_payload
    elif isinstance(metadata_payload, pd.DataFrame):
        metadata_list_for_processing = metadata_payload.to_dict("records")
    elif (
        isinstance(metadata_payload, dict)
        and "analysis_df_records" in metadata_payload
    ):
        records_data = metadata_payload["analysis_df_records"]
        if isinstance(records_data, pd.DataFrame):
            metadata_list_for_processing = records_data.to_dict("records")
        elif isinstance(records_data, list):
            metadata_list_for_processing = records_data
        else:
            logger.error(
                "Metadata 'analysis_df_records' not DataFrame or list."
            )
            return
    else:
        logger.error(f"Unexpected metadata format in {metadata_file}.")
        return
    if not metadata_list_for_processing:
        logger.warning("Metadata list is empty.")
    logger.info(
        f"Metadata loading took {time.time() - start_time:.2f}s. Loaded {len(metadata_list_for_processing)} records."
    )

    with h5py.File(activations_file, "r") as hf:
        if "activations" not in hf:
            logger.error(f"'activations' not in {activations_file}")
            return
        n_examples_total, n_neurons_total = hf["activations"].shape

    n_neurons_to_process = n_neurons_total
    if args.max_neurons is not None:
        n_neurons_to_process = min(n_neurons_total, args.max_neurons)
    logger.info(
        f"Targeting {n_neurons_to_process} neurons (out of {n_neurons_total}) with {n_examples_total} examples."
    )

    neurons_to_process_this_run = list(range(n_neurons_to_process))

    if should_resume:
        if (
            not consolidated_output_active
        ):  # JSON output mode: skip existing JSONs
            processed_neurons_indices = set()
            existing_files = list(output_dir.glob("neuron_*.json"))
            for f_path in existing_files:
                try:
                    idx = int(f_path.stem.split("_")[1])
                    if idx < n_neurons_to_process:
                        processed_neurons_indices.add(idx)
                except (ValueError, IndexError):
                    logger.warning(f"Could not parse neuron index: {f_path}")

            if processed_neurons_indices:
                neurons_to_process_this_run = [
                    i
                    for i in neurons_to_process_this_run
                    if i not in processed_neurons_indices
                ]
                logger.info(
                    f"JSON output mode: Resuming. Skipping {len(processed_neurons_indices)} existing JSONs. "
                    f"Will process {len(neurons_to_process_this_run)} neurons."
                )
        # For CSV mode with resume (should_resume=True), all neurons_to_process_this_run are processed.
        # The "resume" is handled by opening the CSV in append mode.

    if not neurons_to_process_this_run:
        logger.info("No new neurons to process for this run.")
    else:
        logger.info(
            f"Will process {len(neurons_to_process_this_run)} neurons in this run."
        )
        mp_args = [
            (
                n_idx,
                str(activations_file),
                args.n_bins,
                args.top_k,
                str(output_dir),
                consolidated_output_active,
            )
            for n_idx in neurons_to_process_this_run
        ]
        successful_json_writes = 0
        failed_neurons_info = []

        with mp.Pool(
            processes=num_workers,
            initializer=init_worker,
            initargs=(metadata_list_for_processing,),
        ) as pool:
            with tqdm(
                total=len(neurons_to_process_this_run),
                desc="Processing neurons",
            ) as pbar:
                for neuron_idx_processed, result_payload in pool.imap_unordered(
                    process_single_neuron, mp_args
                ):
                    if consolidated_output_active:
                        if isinstance(result_payload, dict):
                            neuron_data_for_csv = result_payload
                            feature_id_val = neuron_data_for_csv["neuron_id"]
                            rows_written_for_feature = 0
                            for (
                                bin_idx_str_val,
                                bin_content,
                            ) in neuron_data_for_csv["range_examples"].items():
                                bin_index_val = int(bin_idx_str_val)
                                bin_min_act_val, bin_max_act_val = bin_content[
                                    "bounds"
                                ]
                                for rank_in_bin_val, example_val in enumerate(
                                    bin_content["examples"], 1
                                ):
                                    example_id_val = example_val["index"]
                                    activation_value_val = example_val[
                                        "activation"
                                    ]
                                    try:
                                        if csv_writer:
                                            csv_writer.writerow(
                                                [
                                                    feature_id_val,
                                                    bin_index_val,
                                                    bin_min_act_val,
                                                    bin_max_act_val,
                                                    example_id_val,
                                                    activation_value_val,
                                                    rank_in_bin_val,
                                                ]
                                            )
                                            rows_written_for_feature += 1
                                    except Exception as e_csv:
                                        logger.error(
                                            f"CSV write error for feature {feature_id_val}, bin {bin_index_val}: {e_csv}"
                                        )
                            if rows_written_for_feature > 0:
                                successfully_processed_neurons_for_csv += 1
                        elif isinstance(result_payload, str):
                            failed_neurons_info.append(
                                (neuron_idx_processed, result_payload)
                            )
                        # else: logger.warning(f"Unexpected result: {type(result_payload)}") # Optional: too verbose
                    else:  # JSON mode
                        if result_payload is True:
                            successful_json_writes += 1
                        elif isinstance(result_payload, str):
                            failed_neurons_info.append(
                                (neuron_idx_processed, result_payload)
                            )
                    pbar.update(1)

        if consolidated_output_active:
            logger.info(
                f"Data for {successfully_processed_neurons_for_csv} neurons written to CSV."
            )
        else:
            logger.info(f"Wrote JSON for {successful_json_writes} neurons.")
        if failed_neurons_info:
            logger.warning(
                f"Failed {len(failed_neurons_info)} neurons. Errors: {failed_neurons_info[:5]}"
            )

    if csv_file_handle:
        csv_file_handle.close()
    if consolidated_output_active:
        logger.info(
            f"Consolidated CSV file finalized: {args.consolidated_output_file}"
        )

    # Low activation examples (unchanged from previous logic, separate file)
    logger.info("\nFinding low activation examples...")
    # ... (rest of low activation logic remains the same) ...
    # Ensure use of standard json.dumps and 'w' for text file:
    low_examples_file_path = output_dir / "low_activation_examples.json"
    # Example of how find_low_activation_examples_optimized would be called and results saved:
    # This part needs the full logic as before for finding and saving these.
    # For brevity, I'll skip reimplementing the full loop here, but ensure it's correct in your actual file.
    # Assume `low_examples_output_list` is populated correctly.
    # Example:
    # low_examples_output_list = [] # This would be populated by the chunked processing loop for low acts
    # ... (looping through HDF5 chunks to find low activation examples) ...
    # (The existing logic for all_low_indices_global, etc., should be here)
    # For this response, I'm focusing on the --no-resume and CSV/JSON logic.
    # The full low_activation part should be copied from the previous correct version.
    # --- Start Placeholder for full low activation logic ---
    global_max_low_examples = 1000
    num_chunks_for_low_examples = (
        int(np.ceil(n_examples_total / args.low_act_chunk_size))
        if args.low_act_chunk_size > 0
        else 1
    )
    examples_per_chunk_target = max(
        1,
        min(
            global_max_low_examples,
            (
                int(
                    np.ceil(
                        global_max_low_examples / num_chunks_for_low_examples
                    )
                )
                if num_chunks_for_low_examples > 0
                else global_max_low_examples
            ),
        ),
    )
    all_low_indices_global, all_max_acts_global, all_mean_acts_global = (
        [],
        [],
        [],
    )
    with h5py.File(activations_file, "r") as hf:
        acts_dset = hf["activations"]
        for chunk_start_idx in tqdm(
            range(0, n_examples_total, args.low_act_chunk_size),
            desc="Low act chunks",
        ):
            chunk_end_idx = min(
                chunk_start_idx + args.low_act_chunk_size, n_examples_total
            )
            if chunk_start_idx >= chunk_end_idx:
                continue
            current_chunk_acts = acts_dset[
                chunk_start_idx:chunk_end_idx, :n_neurons_to_process
            ]
            l_idx, m_acts, mn_acts = find_low_activation_examples_optimized(
                current_chunk_acts,
                n_neurons_to_process,
                args.n_bins,
                args.low_act_bottom_bins,
                examples_per_chunk_target,
            )
            all_low_indices_global.extend((l_idx + chunk_start_idx).tolist())
            all_max_acts_global.extend(m_acts.tolist())
            all_mean_acts_global.extend(mn_acts.tolist())
            del current_chunk_acts
            gc.collect()
    if len(all_low_indices_global) > global_max_low_examples:
        s_indices = np.random.choice(
            len(all_low_indices_global), global_max_low_examples, replace=False
        )
        all_low_indices_global = [all_low_indices_global[i] for i in s_indices]
        all_max_acts_global = [all_max_acts_global[i] for i in s_indices]
        all_mean_acts_global = [all_mean_acts_global[i] for i in s_indices]
    low_examples_output_list = []
    for gl_idx, max_val, mean_val in zip(
        all_low_indices_global, all_max_acts_global, all_mean_acts_global
    ):
        if gl_idx < len(metadata_list_for_processing):
            meta = metadata_list_for_processing[gl_idx]
            low_examples_output_list.append(
                {
                    "index": int(gl_idx),
                    "max_activation": float(max_val),
                    "mean_activation": float(mean_val),
                    "metadata": {
                        "text": meta.get("text", ""),
                        "weapon_id": meta.get("weapon_id", -1),
                        "label": meta.get("label", ""),
                    },
                }
            )
    # --- End Placeholder ---
    with open(
        low_examples_file_path, "w"
    ) as f:  # Standard json.dumps needs 'w'
        f.write(json.dumps(low_examples_output_list))
    logger.info(
        f"\nSaved {len(low_examples_output_list)} low activation examples to {low_examples_file_path}"
    )


def main_cli():
    parser = argparse.ArgumentParser(
        description="Extract top examples command for activations."
    )
    parser.add_argument(
        "--activations",
        type=str,
        required=True,
        help="Path to HDF5 file with activations",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        required=True,
        help="Path to pickle file with metadata",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for per-neuron JSONs and low_activation_examples.json.",
    )
    parser.add_argument(
        "--consolidated-output-file",
        type=str,
        default=None,
        help="Optional. Path to a single CSV file for consolidated output. If provided, per-neuron JSONs are not primarily targeted (though their existence can affect resume for re-computation).",
    )
    parser.add_argument("--n-bins", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--max-neurons", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    # Corrected --no-resume definition to match how it's used (args.no_resume will be True if flag is passed)
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="If passed, disables resume behavior. For JSON, overwrites. For CSV, overwrites.",
    )
    # No set_defaults for no_resume, it will be False if not passed.
    parser.add_argument("--low-act-chunk-size", type=int, default=50000)
    parser.add_argument("--low-act-bottom-bins", type=int, default=2)

    cli_args = parser.parse_args()
    extract_top_examples_command(cli_args)


if __name__ == "__main__":
    main_cli()
