"""
Command for ingesting binned sample data (from extract-top-examples CSV)
and feature correlations into an SQLite database.
"""
import argparse
import sqlite3
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

DDL_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS feature_binned_samples (
        feature_id INTEGER NOT NULL,
        bin_index INTEGER NOT NULL,
        bin_min_activation REAL,
        bin_max_activation REAL,
        example_id INTEGER,
        activation_value REAL,
        rank_in_bin INTEGER NOT NULL,
        PRIMARY KEY (feature_id, bin_index, rank_in_bin),
        FOREIGN KEY (example_id) REFERENCES examples(id)
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_fbs_feature_bin ON feature_binned_samples(feature_id, bin_index);",
    "CREATE INDEX IF NOT EXISTS idx_fbs_example ON feature_binned_samples(example_id);",
    """
    CREATE TABLE IF NOT EXISTS feature_stats_new (
        feature_id INTEGER PRIMARY KEY NOT NULL,
        overall_min_activation REAL,
        overall_max_activation REAL,
        estimated_mean REAL,
        estimated_median REAL,
        num_sampled_examples INTEGER,
        num_bins INTEGER,
        sampled_histogram_data TEXT
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS top_examples (
        feature_id INTEGER NOT NULL,
        example_id INTEGER NOT NULL,
        rank INTEGER NOT NULL,
        activation_value REAL,
        PRIMARY KEY (feature_id, rank),
        FOREIGN KEY (example_id) REFERENCES examples(id)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS feature_correlations (
        feature_a INTEGER NOT NULL,
        feature_b INTEGER NOT NULL,
        correlation REAL,
        PRIMARY KEY (feature_a, feature_b)
    );
    """,
    """
    CREATE TABLE IF NOT EXISTS logit_influences (
        feature_id INTEGER,
        influence_type TEXT, -- 'positive' or 'negative'
        rank INTEGER,        -- 1 to K for that type
        token_id INTEGER,
        token_name TEXT,
        influence_value REAL,
        PRIMARY KEY (feature_id, influence_type, rank)
    );
    """
]

def init_db(db_path: Path, ddl_statements: List[str]):
    """Initializes the database and creates tables if they don't exist."""
    logger.info(f"Initializing database at {db_path}...")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        for statement in ddl_statements:
            try:
                cursor.execute(statement)
            except sqlite3.Error as e:
                logger.error(f"Failed to execute DDL: {statement}\nError: {e}")
                raise
        conn.commit()
    logger.info("Database initialized and tables ensured.")

def clear_data(conn: sqlite3.Connection, handle_correlations: bool):
    """Clears data from relevant tables before new insertion."""
    logger.info("Clearing existing data from target tables...")
    cursor = conn.cursor()
    tables_to_clear = [
        "feature_binned_samples",
        "feature_stats_new", 
        "top_examples",
        "logit_influences" # Added logit_influences
    ]
    if handle_correlations: # This implies correlations_json_path was provided
        tables_to_clear.append("feature_correlations")
    
    # Always attempt to clear logit_influences if its DDL is present.
    # The actual population will only happen if logit_influences_path is provided.
    # This ensures that if the table exists, it's cleared.
    # No, clear_data should only clear if we intend to populate.
    # So, let's pass another flag or check args inside.
    # For simplicity in this refactor, clear_data will now take specific flags for optional data.
    # However, the prompt says "if --logit-influences-path is provided ... add a step to DELETE".
    # This means the decision to clear is linked to the intention to populate.
    # The `tables_to_clear` list is fine, the calling logic in `ingest_binned_samples_command`
    # will manage whether to call the population part.
    # The DDL ensures the table exists. Clear always attempts it.

    for table in tables_to_clear:
        try:
            # Check if table exists before attempting to delete, to avoid "no such table" for optional tables
            # if table exists (check from sqlite_master)
            res = cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}';").fetchone()
            if res:
                cursor.execute(f"DELETE FROM {table};")
                logger.info(f"Cleared data from {table}.")
            else:
                logger.info(f"Table {table} does not exist, skipping clearing.")
        except sqlite3.Error as e:
            logger.error(f"Error during clearing of {table}: {e}")
            raise # Re-raise if it's a real error beyond "no such table" if not caught by check
    conn.commit()
    logger.info("Data clearing complete.")

def populate_feature_binned_samples(conn: sqlite3.Connection, csv_path: Path):
    """Populates the feature_binned_samples table from the provided CSV."""
    logger.info(f"Populating feature_binned_samples from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
        # Ensure column names match: feature_id,bin_index,bin_min_activation,bin_max_activation,example_id,activation_value,rank_in_bin
        # The CSV from extract-top-examples should have these columns.
        expected_columns = [
            "feature_id", "bin_index", "bin_min_activation",
            "bin_max_activation", "example_id", "activation_value", "rank_in_bin"
        ]
        if not all(col in df.columns for col in expected_columns):
            logger.error(f"CSV file {csv_path} is missing one or more expected columns: {expected_columns}. Found: {list(df.columns)}")
            raise ValueError("CSV column mismatch")
        
        df.to_sql("feature_binned_samples", conn, if_exists="append", index=False, chunksize=10000)
        logger.info(f"Successfully populated feature_binned_samples with {len(df)} rows.")
    except FileNotFoundError:
        logger.error(f"Binned samples CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error populating feature_binned_samples: {e}", exc_info=True)
        raise

def populate_feature_stats_new(conn: sqlite3.Connection, csv_path: Path):
    """Populates the feature_stats_new table by processing data from the binned samples CSV."""
    logger.info(f"Populating feature_stats_new using data from {csv_path}...")
    try:
        df_samples = pd.read_csv(csv_path)
        if df_samples.empty:
            logger.warning("Binned samples CSV is empty. No data to populate feature_stats_new.")
            return

        stats_data = []
        grouped_by_feature = df_samples.groupby("feature_id")

        for feature_id, feature_df in grouped_by_feature:
            overall_min_activation = feature_df["bin_min_activation"].min()
            overall_max_activation = feature_df["bin_max_activation"].max()
            num_sampled_examples = len(feature_df)
            num_bins = feature_df["bin_index"].nunique()

            # Estimated Median
            estimated_median = np.median(feature_df["activation_value"])

            # Estimated Mean
            feature_df["bin_midpoint"] = (feature_df["bin_min_activation"] + feature_df["bin_max_activation"]) / 2
            # Count samples per original bin_index for weighting
            samples_per_bin = feature_df.groupby("bin_index")["example_id"].count()
            
            # A feature_df might not have all bin_indices if some bins were empty.
            # We need to map these counts back to the unique bins present for this feature.
            weighted_sum = 0
            total_weight = 0
            
            # Iterate over unique bins present for this feature to calculate weighted mean
            unique_bins_for_feature = feature_df[["bin_index", "bin_midpoint"]].drop_duplicates().set_index("bin_index")
            
            for bin_idx, row in unique_bins_for_feature.iterrows():
                bin_mid = row["bin_midpoint"]
                count_in_bin = samples_per_bin.get(bin_idx, 0)
                weighted_sum += bin_mid * count_in_bin
                total_weight += count_in_bin
            
            estimated_mean = weighted_sum / total_weight if total_weight > 0 else 0

            # Sampled Histogram Data
            # We need counts of *sampled examples* per bin_index for this feature
            hist_counts = feature_df.groupby("bin_index").size().reindex(
                np.arange(feature_df["bin_index"].min(), feature_df["bin_index"].max() + 1), 
                fill_value=0
            )
            
            # Get unique bin ranges for this feature, sorted by bin_index
            bin_ranges_df = feature_df[["bin_index", "bin_min_activation", "bin_max_activation"]].drop_duplicates().sort_values("bin_index")
            
            # Align hist_counts index with bin_ranges_df if necessary, though groupby().size() should align if reindexed properly.
            # For JSON, ensure bin_ranges match the order and extent of hist_counts.
            
            # Create a full range of bin_indices from min to max observed for this feature
            min_bin_idx = bin_ranges_df["bin_index"].min()
            max_bin_idx = bin_ranges_df["bin_index"].max()
            full_bin_indices = pd.Series(range(min_bin_idx, max_bin_idx + 1), name="bin_index")

            # Merge to get all bin ranges, then fill missing counts with 0
            # This assumes bin_ranges_df has unique bin_index entries
            temp_hist_df = pd.merge(full_bin_indices, bin_ranges_df, on="bin_index", how="left")
            
            # Forward fill and backward fill for bins that might be missing if they had no samples.
            # This is a heuristic if a bin truly has no samples and its bounds are unknown.
            # A better approach is to ensure extract-top-examples outputs all bin bounds even if empty.
            # For now, let's assume bin_ranges_df is reasonably complete for bins that *do* have samples.
            # And hist_counts from groupby.size().reindex will provide counts for all bins in range.
            
            # Use the bin ranges from the actual data, and ensure counts match this.
            # If a bin_index has samples, its range will be in bin_ranges_df.
            # hist_counts should be based on the actual bin_indices present in feature_df.
            
            actual_bin_indices_with_samples = sorted(feature_df["bin_index"].unique())
            final_bin_ranges = []
            final_counts = []

            # Use a consistent set of bins for ranges and counts
            # We'll use the bins defined in bin_ranges_df, which are bins with actual samples.
            # If a bin had 0 samples but was between other bins with samples, extract-top-examples
            # should still output its bounds in the CSV. If not, this histogram will only show sampled bins.
            
            # Rebuilding based on what `extract_top_examples_cmd.py` output means for `range_examples`:
            # It iterates `range(n_bins)`, so all bins are present in the JSON, even if count is 0.
            # The CSV, however, is generated from the 'examples' list within each bin.
            # So, the CSV will only contain rows for (feature_id, bin_index) pairs that had examples.
            
            # Let's use the min/max bin_index found in the CSV for this feature to define the histogram range.
            # This might mean intermediate empty bins are not explicitly represented if their bounds aren't in the CSV.
            
            current_feature_bins = feature_df[["bin_index", "bin_min_activation", "bin_max_activation"]].drop_duplicates().sort_values("bin_index")
            
            # Create a map from bin_index to its [min, max] activation
            bin_map = {row.bin_index: [row.bin_min_activation, row.bin_max_activation] for _, row in current_feature_bins.iterrows()}
            
            # Counts per bin for the current feature
            counts_per_bin = feature_df.groupby("bin_index").size()
            
            # Construct histogram for all bins from min to max for this feature
            # This ensures continuity if `extract-top-examples` CSV omits empty bins.
            # We need a reliable way to get bin bounds for potentially empty bins.
            # The current CSV only has bins with examples. This means the histogram will only reflect those.
            
            hist_bin_indices = sorted(counts_per_bin.index.tolist())
            hist_json_ranges = [bin_map[b_idx] for b_idx in hist_bin_indices]
            hist_json_counts = [counts_per_bin[b_idx] for b_idx in hist_bin_indices]

            sampled_histogram_data = json.dumps({
                "bin_ranges": hist_json_ranges, # List of [min_act, max_act]
                "counts": hist_json_counts     # List of counts for sampled examples
            })

            stats_data.append((
                feature_id, overall_min_activation, overall_max_activation,
                estimated_mean, estimated_median, num_sampled_examples,
                num_bins, sampled_histogram_data
            ))
        
        if stats_data:
            conn.executemany(
                """INSERT INTO feature_stats_new 
                   (feature_id, overall_min_activation, overall_max_activation, 
                    estimated_mean, estimated_median, num_sampled_examples, 
                    num_bins, sampled_histogram_data) 
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                stats_data
            )
            conn.commit()
            logger.info(f"Successfully populated feature_stats_new for {len(stats_data)} features.")
        else:
            logger.info("No data to insert into feature_stats_new.")
            
    except FileNotFoundError: # Should be caught by populate_feature_binned_samples if CSV is central
        logger.error(f"Binned samples CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error populating feature_stats_new: {e}", exc_info=True)
        raise

def populate_top_examples(conn: sqlite3.Connection, csv_path: Path, top_n: int = 20):
    """Populates the top_examples table."""
    logger.info(f"Populating top_examples (top {top_n}) using data from {csv_path}...")
    try:
        df_samples = pd.read_csv(csv_path)
        if df_samples.empty:
            logger.warning("Binned samples CSV is empty. No data to populate top_examples.")
            return

        # Sort by feature_id, then by activation_value (desc), then by bin_max_activation (desc) as tie-breaker
        df_sorted = df_samples.sort_values(
            ["feature_id", "activation_value", "bin_max_activation"],
            ascending=[True, False, False]
        )

        # Get top N for each feature
        top_n_df = df_sorted.groupby("feature_id").head(top_n).copy() # Use .copy() to avoid SettingWithCopyWarning

        # Assign rank
        top_n_df["rank"] = top_n_df.groupby("feature_id").cumcount() + 1
        
        # Select columns for insertion
        top_examples_to_insert = top_n_df[["feature_id", "example_id", "rank", "activation_value"]]

        conn.executemany(
            "INSERT INTO top_examples (feature_id, example_id, rank, activation_value) VALUES (?, ?, ?, ?)",
            top_examples_to_insert.to_records(index=False).tolist()
        )
        conn.commit()
        logger.info(f"Successfully populated top_examples with {len(top_examples_to_insert)} rows.")
    except FileNotFoundError:
        logger.error(f"Binned samples CSV file not found: {csv_path}")
        raise
    except Exception as e:
        logger.error(f"Error populating top_examples: {e}", exc_info=True)
        raise

def populate_feature_correlations(conn: sqlite3.Connection, json_path: Optional[Path]):
    """Populates the feature_correlations table from a JSON file."""
    if not json_path:
        logger.info("Correlations JSON path not provided, skipping population of feature_correlations.")
        return
    if not json_path.exists():
        logger.warning(f"Correlations JSON file not found at {json_path}, skipping.")
        return

    logger.info(f"Populating feature_correlations from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            correlations_data_wrapper = json.load(f)
        
        if not correlations_data_wrapper or 'correlations' not in correlations_data_wrapper:
            logger.warning(f"Correlations data at {json_path} is missing 'correlations' key or is empty.")
            return
            
        correlations_list = correlations_data_wrapper['correlations']
        if not isinstance(correlations_list, list):
            logger.warning(f"Correlations data in {json_path} is not a list as expected.")
            return

        db_corr_data = []
        for corr_item in correlations_list:
            if all(k in corr_item for k in ['neuron_i', 'neuron_j', 'correlation']):
                feat_a = min(corr_item['neuron_i'], corr_item['neuron_j'])
                feat_b = max(corr_item['neuron_i'], corr_item['neuron_j'])
                # Ensure feature_a != feature_b to avoid self-correlations if they exist
                if feat_a == feat_b: continue
                db_corr_data.append((feat_a, feat_b, corr_item['correlation']))
            else:
                logger.warning(f"Skipping malformed correlation item: {corr_item}")
        
        if db_corr_data:
            conn.executemany(
                "INSERT OR REPLACE INTO feature_correlations (feature_a, feature_b, correlation) VALUES (?, ?, ?)",
                db_corr_data
            )
            conn.commit()
            logger.info(f"Successfully populated feature_correlations with {len(db_corr_data)} items.")
        else:
            logger.info("No valid correlation data to insert.")

    except Exception as e:
        logger.error(f"Error populating feature_correlations: {e}", exc_info=True)
        # Do not re-raise if correlations are optional and fail

def populate_logit_influences(conn: sqlite3.Connection, jsonl_path: Path):
    """Populates the logit_influences table from a JSONL file."""
    logger.info(f"Populating logit_influences from {jsonl_path}...")
    if not jsonl_path.exists():
        logger.warning(f"Logit influences file not found at {jsonl_path}, skipping population.")
        return

    logit_influence_records = []
    try:
        with open(jsonl_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    feature_id = data["feature_id"]
                    
                    for influence_item in data.get("positive_influences", []):
                        logit_influence_records.append((
                            feature_id,
                            "positive",
                            influence_item["rank"],
                            influence_item["token_id"],
                            influence_item["token_name"],
                            influence_item["influence"]
                        ))
                    
                    for influence_item in data.get("negative_influences", []):
                        logit_influence_records.append((
                            feature_id,
                            "negative",
                            influence_item["rank"],
                            influence_item["token_id"],
                            influence_item["token_name"],
                            influence_item["influence"]
                        ))
                except json.JSONDecodeError:
                    logger.error(f"Skipping malformed JSON line {line_num} in {jsonl_path}: {line.strip()}")
                except KeyError as e:
                    logger.error(f"Skipping record due to missing key {e} in line {line_num} in {jsonl_path}")

        if logit_influence_records:
            conn.executemany(
                """INSERT OR REPLACE INTO logit_influences 
                   (feature_id, influence_type, rank, token_id, token_name, influence_value) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                logit_influence_records
            )
            conn.commit()
            logger.info(f"Successfully populated logit_influences with {len(logit_influence_records)} records.")
        else:
            logger.info("No valid logit influence data to insert.")

    except Exception as e:
        logger.error(f"Error populating logit_influences: {e}", exc_info=True)
        # Decide whether to re-raise or not. If this is critical, re-raise.
        # For now, log and continue, similar to correlations.
        
def finalize_feature_stats_table(conn: sqlite3.Connection):
    """Renames feature_stats_new to feature_stats."""
    logger.info("Finalizing feature_stats table...")
    cursor = conn.cursor()
    try:
        cursor.execute("DROP TABLE IF EXISTS feature_stats;")
        cursor.execute("ALTER TABLE feature_stats_new RENAME TO feature_stats;")
        conn.commit()
        logger.info("Successfully renamed feature_stats_new to feature_stats.")
    except Exception as e:
        logger.error(f"Error finalizing feature_stats table: {e}", exc_info=True)
        raise

def ingest_binned_samples_command(args: argparse.Namespace):
    """Main command function for ingesting binned samples and other analytics."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger.info(f"Starting data ingestion process for database: {args.db_path}")

    db_path = Path(args.db_path)
    binned_samples_csv = Path(args.binned_samples_csv)
    correlations_json_path = Path(args.correlations_json) if args.correlations_json else None
    logit_influences_path = Path(args.logit_influences_path) if args.logit_influences_path else None

    # 1. Initialize DB and tables
    init_db(db_path, DDL_STATEMENTS)

    # 2. Connect and clear data
    # Keep connection open for all operations for transactional consistency (or manage per function)
    try:
        with sqlite3.connect(db_path) as conn:
            handle_correlations = correlations_json_path is not None
            clear_data(conn, handle_correlations)

            # 3. Populate feature_binned_samples
            populate_feature_binned_samples(conn, binned_samples_csv)

            # 4. Populate feature_stats_new
            # This function reads the CSV again. If CSV is very large, could pass df from previous step.
            # For simplicity and modularity, it reads again.
            populate_feature_stats_new(conn, binned_samples_csv)

            # 5. Populate top_examples
            populate_top_examples(conn, binned_samples_csv, top_n=args.top_n_examples) # Add top_n_examples to args

            # 6. Populate feature_correlations (optional)
            if correlations_json_path and correlations_json_path.exists():
                populate_feature_correlations(conn, correlations_json_path)
            elif args.correlations_json: # Path provided but file doesn't exist
                 logger.warning(f"Correlations JSON path {args.correlations_json} provided but file not found. Skipping.")
            
            # 7. Populate logit_influences (optional)
            if logit_influences_path and logit_influences_path.exists():
                populate_logit_influences(conn, logit_influences_path)
            elif args.logit_influences_path: # Path provided but file doesn't exist
                logger.warning(f"Logit influences path {args.logit_influences_path} provided but file not found. Skipping.")

            # 8. Rename feature_stats_new to feature_stats
            finalize_feature_stats_table(conn)
            
            logger.info("Data ingestion process completed successfully.")

    except Exception as e:
        logger.critical(f"Data ingestion failed: {e}", exc_info=True)
        # Consider cleanup or state restoration if needed

    try:
        logger.info("Optimizing database (VACUUM)...")
        with sqlite3.connect(db_path) as conn:
            conn.execute("VACUUM;")
        logger.info("Database optimization complete.")
    except Exception as e:
        logger.error(f"Failed to optimize database: {e}", exc_info=True)

# This function will be called from cli.py to set up the subparser
def register_ingest_binned_samples_parser(subparsers: argparse._SubParsersAction):
    p = subparsers.add_parser('ingest-binned-samples', help='Ingest binned samples from CSV and compute feature statistics for the database.')
    p.add_argument('--db-path', type=str, required=True, help='Path to the SQLite database file.')
    p.add_argument('--binned-samples-csv', type=str, required=True, help='Path to the CSV file containing binned samples data (output of extract-top-examples).')
    p.add_argument('--correlations-json', type=str, default=None, help='Optional. Path to JSON file with feature correlations.')
    p.add_argument('--logit-influences-path', type=str, default=None, help='Optional. Path to JSONL file with logit influences.')
    p.add_argument('--top-n-examples', type=int, default=20, help='Number of top examples to store per feature.')
    p.set_defaults(func=ingest_binned_samples_command)

if __name__ == '__main__':
    # Example direct execution (for testing)
    # Create a dummy parser
    parser = argparse.ArgumentParser()
    # In a real scenario, this would be part of a larger CLI setup like in splatnlp.dashboard.cli
    # For testing, we can simulate the args
    # test_args = parser.parse_args([
    #     '--db-path', 'test_dashboard.db',
    #     '--binned-samples-csv', 'path/to/your/binned_samples.csv', # Replace with actual path
    #     # '--correlations-json', 'path/to/correlations.json' # Optional
    # ])
    # ingest_binned_samples_command(test_args)
    
    # This script is intended to be called via the main CLI (cli.py),
    # so direct execution here would require manual setup of args.
    # The register_ingest_binned_samples_parser function is for integration with cli.py.
    logger.warning("This script is designed to be run as a command through cli.py, not directly.")
    pass
