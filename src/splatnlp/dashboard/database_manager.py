#!/usr/bin/env python3
"""
Database manager for scalable dashboard data storage.

This module provides both SQLite and DuckDB-based solutions to replace memory-intensive 
data loading patterns in the dashboard. It stores activations, metadata,
and precomputed analytics in an efficient queryable format.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DashboardDatabase:
    """SQLite-based database manager for dashboard data."""

    def __init__(self, db_path: Union[str, Path]):
        """Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Enable WAL mode for better concurrent access
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database schema if not exists."""
        with self.get_connection() as conn:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=memory")

            # Create schema
            self._create_schema(conn)

    @contextmanager
    def get_connection(self) -> Iterator[sqlite3.Connection]:
        """Get database connection with proper cleanup."""
        conn = sqlite3.connect(
            self.db_path, timeout=30.0, check_same_thread=False
        )
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()

    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema."""

        # Examples metadata table
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS examples (
                id INTEGER PRIMARY KEY,
                weapon_id INTEGER,
                weapon_name TEXT,
                ability_input_tokens TEXT,  -- JSON array
                input_abilities_str TEXT,
                top_predicted_abilities_str TEXT,
                is_null_token BOOLEAN DEFAULT FALSE,
                metadata TEXT  -- JSON object for additional data
            )
        """
        )

        # Feature activations table (sparse storage) - DEPRECATED / NO LONGER POPULATED by new commands
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS activations (
                example_id INTEGER,
                feature_id INTEGER,
                activation_value REAL,
                PRIMARY KEY (example_id, feature_id),
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """
        )

        # New table for binned samples
        conn.execute(
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
            )
        """
        )

        # Feature statistics table (schema matches feature_stats_new from ingest command)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_stats (
                feature_id INTEGER PRIMARY KEY NOT NULL,
                overall_min_activation REAL,
                overall_max_activation REAL,
                estimated_mean REAL,
                estimated_median REAL,
                num_sampled_examples INTEGER,
                num_bins INTEGER,
                sampled_histogram_data TEXT -- JSON for sampled histogram
            )
        """
        )

        # Top examples per feature (precomputed for speed)
        # Schema should match what ingest-binned-samples populates
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS top_examples (
                feature_id INTEGER NOT NULL,
                example_id INTEGER NOT NULL,
                rank INTEGER NOT NULL,
                activation_value REAL,
                PRIMARY KEY (feature_id, rank), 
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """
        )

        # Feature correlations (precomputed)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feature_correlations (
                feature_a INTEGER NOT NULL,
                feature_b INTEGER NOT NULL,
                correlation REAL,
                PRIMARY KEY (feature_a, feature_b)
            )
        """
        )

        # Logit influences (precomputed) - schema matches ingest-binned-samples
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS logit_influences (
                feature_id INTEGER,
                influence_type TEXT, 
                rank INTEGER,
                token_id INTEGER,
                token_name TEXT,
                influence_value REAL,
                PRIMARY KEY (feature_id, influence_type, rank)
            )
        """
        )

        # Create indexes for performance
        indexes = [
            # Indexes for old 'activations' table (can be removed if table is fully deprecated and dropped)
            "CREATE INDEX IF NOT EXISTS idx_activations_feature ON activations(feature_id)",
            "CREATE INDEX IF NOT EXISTS idx_activations_example ON activations(example_id)",
            "CREATE INDEX IF NOT EXISTS idx_activations_value ON activations(activation_value)",
            # Indexes for new 'feature_binned_samples' table
            "CREATE INDEX IF NOT EXISTS idx_fbs_feature_bin ON feature_binned_samples(feature_id, bin_index)",
            "CREATE INDEX IF NOT EXISTS idx_fbs_example ON feature_binned_samples(example_id)",
            # Indexes for 'top_examples' (rank is part of PK, feature_id is good for lookups)
            "CREATE INDEX IF NOT EXISTS idx_top_examples_feature ON top_examples(feature_id)",
            # "CREATE INDEX IF NOT EXISTS idx_top_examples_rank ON top_examples(feature_id, rank)", # Already covered by PK
            # Indexes for 'feature_correlations' (feature_a is part of PK)
            # "CREATE INDEX IF NOT EXISTS idx_correlations_feature_a ON feature_correlations(feature_a)", # Already covered by PK
            "CREATE INDEX IF NOT EXISTS idx_correlations_feature_b ON feature_correlations(feature_b)",
            # Indexes for 'logit_influences' (feature_id and type are part of PK)
            # "CREATE INDEX IF NOT EXISTS idx_logit_influences_feature ON logit_influences(feature_id)", # Already covered by PK
            # "CREATE INDEX IF NOT EXISTS idx_logit_influences_rank ON logit_influences(feature_id, influence_type, rank)", # Already covered by PK
            "CREATE INDEX IF NOT EXISTS idx_examples_weapon ON examples(weapon_id)",
        ]

        for idx_sql in indexes:
            conn.execute(idx_sql)

        conn.commit()
        logger.info("Database schema initialized/updated")

    def insert_examples_batch(
        self, examples_data: List[Dict[str, Any]]
    ) -> None:
        """Insert examples in batch for efficiency.

        Args:
            examples_data: List of example dictionaries
        """
        with self.get_connection() as conn:
            # Ensure all required keys are present with defaults if necessary
            processed_examples_data = []
            for ex in examples_data:
                processed_examples_data.append(
                    (
                        ex.get("id"),  # Assuming ID is always present
                        ex.get("weapon_id"),
                        ex.get("weapon_name"),
                        json.dumps(ex.get("ability_input_tokens")),
                        ex.get("input_abilities_str"),
                        ex.get("top_predicted_abilities_str"),
                        ex.get("is_null_token", False),  # Default for boolean
                        json.dumps(ex.get("metadata", {})),  # Default for JSON
                    )
                )

            conn.executemany(
                """
                INSERT OR REPLACE INTO examples 
                (id, weapon_id, weapon_name, ability_input_tokens, 
                 input_abilities_str, top_predicted_abilities_str, 
                 is_null_token, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                processed_examples_data,
            )
            conn.commit()
            logger.info(f"Inserted {len(examples_data)} examples")

    def insert_activations_batch(
        self,
        activations: np.ndarray,
        example_ids: List[int],
        activation_threshold: float = 1e-6,
    ) -> None:
        """
        DEPRECATED. This method used to insert into the old 'activations' table.
        The new data flow uses 'feature_binned_samples' populated by other commands.
        """
        logger.warning(
            "DEPRECATED: insert_activations_batch is no longer supported for the new schema. Activations are now stored in feature_binned_samples."
        )
        # Optionally, raise an error or simply do nothing.
        # raise NotImplementedError("This method is deprecated. Use commands that populate feature_binned_samples.")
        return

    def get_feature_activations(
        self, feature_id: int, limit: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """
        DEPRECATED. Feature activations are now stored in a binned format.
        Use get_binned_feature_samples instead.

        Args:
            feature_id: Feature to query
            limit: Optional limit on number of results (ignored)

        Returns:
            Tuple of (empty_numpy_array, empty_list)
        """
        logger.warning(
            "DEPRECATED: get_feature_activations is deprecated. "
            "Use get_binned_feature_samples for binned sample data. "
            f"Call for feature_id {feature_id} will return empty data."
        )
        return np.array([]), []

    def get_binned_feature_samples(
        self, feature_id: int
    ) -> List[Dict[str, Any]]:
        """
        Retrieves binned samples for a given feature_id.

        Args:
            feature_id: The ID of the feature to retrieve samples for.

        Returns:
            A list of dictionaries, where each dictionary represents a bin and its samples.
            Example:
            [
              {
                "bin_index": 0,
                "bin_min_activation": 0.0,
                "bin_max_activation": 0.1,
                "samples": [
                  {"example_id": 101, "activation_value": 0.05, "rank_in_bin": 1},
                  ...
                ]
              }, ...
            ]
            Returns an empty list if the feature_id is not found or has no samples.
        """
        logger.debug(f"Fetching binned samples for feature_id: {feature_id}")
        query = """
            SELECT 
                bin_index, 
                bin_min_activation, 
                bin_max_activation, 
                example_id, 
                activation_value, 
                rank_in_bin
            FROM feature_binned_samples
            WHERE feature_id = ?
            ORDER BY bin_index ASC, rank_in_bin ASC;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (feature_id,))
            rows = cursor.fetchall()

        if not rows:
            logger.debug(
                f"No binned samples found for feature_id: {feature_id}"
            )
            return []

        binned_data: Dict[Tuple[int, float, float], List[Dict[str, Any]]] = {}
        for row in rows:
            bin_key = (
                row["bin_index"],
                row["bin_min_activation"],
                row["bin_max_activation"],
            )
            if bin_key not in binned_data:
                binned_data[bin_key] = []

            binned_data[bin_key].append(
                {
                    "example_id": row["example_id"],
                    "activation_value": row["activation_value"],
                    "rank_in_bin": row["rank_in_bin"],
                }
            )

        result_list = []
        # Sort by bin_index after grouping (though SQL ORDER BY should handle overall order)
        # The dict keys won't maintain order, so we re-sort based on bin_index from the key.
        sorted_bin_keys = sorted(binned_data.keys(), key=lambda k: k[0])

        for bin_key in sorted_bin_keys:
            samples = binned_data[bin_key]
            # Samples are already ordered by rank_in_bin due to the SQL query.
            result_list.append(
                {
                    "bin_index": bin_key[0],
                    "bin_min_activation": bin_key[1],
                    "bin_max_activation": bin_key[2],
                    "samples": samples,
                }
            )

        logger.debug(
            f"Returning {len(result_list)} bins for feature_id: {feature_id}"
        )
        return result_list

    def get_top_examples(
        self, feature_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top activating examples for a feature with metadata.
        This queries the `top_examples` table.

        Args:
            feature_id: Feature to query
            limit: Number of top examples to return

        Returns:
            List of example dictionaries with metadata. Assumes 'examples' table has relevant fields like
            'id', 'weapon_name', 'input_abilities_str', 'top_predicted_abilities_str',
            'ability_input_tokens', 'metadata'.
        """
        query = """
            SELECT 
                t.rank,
                t.activation_value,
                e.id AS example_id,  -- Alias to avoid conflict if e also has a 'rank' or 'activation_value'
                e.weapon_name,
                e.input_abilities_str,
                e.top_predicted_abilities_str,
                e.ability_input_tokens,
                e.metadata
            FROM top_examples t
            JOIN examples e ON t.example_id = e.id
            WHERE t.feature_id = ?
            ORDER BY t.rank ASC
            LIMIT ?;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (feature_id, limit))
            results = []
            for row in cursor.fetchall():
                record = dict(row)
                # Parse JSON fields from 'examples' table
                if record.get("ability_input_tokens"):
                    try:
                        record["ability_input_tokens"] = json.loads(
                            record["ability_input_tokens"]
                        )
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse ability_input_tokens for example {record['example_id']}"
                        )
                        record["ability_input_tokens"] = (
                            []
                        )  # Default to empty list
                if record.get("metadata"):
                    try:
                        record["metadata"] = json.loads(record["metadata"])
                    except json.JSONDecodeError:
                        logger.warning(
                            f"Failed to parse metadata for example {record['example_id']}"
                        )
                        record["metadata"] = {}  # Default to empty dict
                else:  # Ensure metadata key exists even if NULL in DB
                    record["metadata"] = {}

                results.append(record)
            return results

    def get_feature_statistics(
        self, feature_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Get precomputed statistics for a feature from the `feature_stats` table.

        Args:
            feature_id: Feature to query.

        Returns:
            Dictionary with statistics or None if not found.
            The 'sampled_histogram_data' JSON string is parsed into a dictionary
            and returned under the key 'histogram'.
        """
        query = """
            SELECT 
                feature_id,
                overall_min_activation,
                overall_max_activation,
                estimated_mean,
                estimated_median,
                num_sampled_examples,
                num_bins,
                sampled_histogram_data 
            FROM feature_stats 
            WHERE feature_id = ?;
        """
        with self.get_connection() as conn:
            cursor = conn.execute(query, (feature_id,))
            row = cursor.fetchone()

            if not row:
                return None

            stats = dict(row)
            # Parse histogram data and rename key for consistency if needed
            histogram_json_str = stats.pop("sampled_histogram_data", None)
            if histogram_json_str:
                try:
                    stats["histogram"] = json.loads(histogram_json_str)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse sampled_histogram_data for feature {feature_id}"
                    )
                    stats["histogram"] = {}  # Default to empty dict on error
            else:
                stats["histogram"] = {}  # Default if no histogram data

            return stats

    def get_feature_correlations(
        self, feature_id: int, limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top correlated features.

        Args:
            feature_id: Feature to query
            limit: Number of correlations to return

        Returns:
            List of correlation dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT feature_b as feature_id, correlation
                FROM feature_correlations 
                WHERE feature_a = ?
                ORDER BY ABS(correlation) DESC
                LIMIT ?
            """,
                (feature_id, limit),
            )

            return [dict(row) for row in cursor.fetchall()]

    def get_logit_influences(
        self,
        feature_id: int,
        limit: int = 10,  # Limit for positive and negative influences each
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Get top logit influences for a feature, separated by positive and negative.

        Args:
            feature_id: Feature to query.
            limit: Number of influences to return for each type (positive/negative).

        Returns:
            A dictionary with 'positive' and 'negative' influence lists.
            Example:
            {
              "positive": [{"rank": 1, "token_id": 123, "token_name": "foo", "influence_value": 0.5}, ...],
              "negative": [{"rank": 1, "token_id": 456, "token_name": "bar", "influence_value": -0.4}, ...]
            }
        """
        results: Dict[str, List[Dict[str, Any]]] = {
            "positive": [],
            "negative": [],
        }

        query_positive = """
            SELECT rank, token_id, token_name, influence_value
            FROM logit_influences
            WHERE feature_id = ? AND influence_type = 'positive'
            ORDER BY rank ASC
            LIMIT ?;
        """
        query_negative = """
            SELECT rank, token_id, token_name, influence_value
            FROM logit_influences
            WHERE feature_id = ? AND influence_type = 'negative'
            ORDER BY rank ASC
            LIMIT ?;
        """

        with self.get_connection() as conn:
            cursor_pos = conn.execute(query_positive, (feature_id, limit))
            for row in cursor_pos.fetchall():
                results["positive"].append(dict(row))

            cursor_neg = conn.execute(query_negative, (feature_id, limit))
            for row in cursor_neg.fetchall():
                results["negative"].append(dict(row))

        return results

    def compute_and_store_feature_stats(self, feature_id: int) -> None:
        """
        DEPRECATED. Feature statistics are now precomputed and stored by ingest commands.
        This method might compute stats based on the old 'activations' table if it still exists.
        """
        logger.warning(
            "DEPRECATED: compute_and_store_feature_stats is deprecated. Stats are precomputed by ingestion commands."
        )
        # Optionally, raise an error or simply do nothing.
        # raise NotImplementedError("This method is deprecated.")
        return

        # --- Original logic (kept for reference, but should not be used for new schema) ---
        # activations, _ = self.get_feature_activations(feature_id) # This itself is deprecated
        # if len(activations) == 0:
        #     logger.warning(f"No activations found for feature {feature_id} (using deprecated method)")
        #     return
        # # ... rest of the original computation ...

    def store_top_examples(
        self,
        feature_id: int,
        example_ids: List[int],
        activations: List[float],
        top_k: int = 50,
    ) -> None:
        """Store top examples for a feature.

        Args:
            feature_id: Feature to store top examples for
            example_ids: List of example IDs
            activations: Corresponding activation values
            top_k: Number of top examples to store
        """
        activations, _ = self.get_feature_activations(feature_id)

        if len(activations) == 0:
            logger.warning(f"No activations found for feature {feature_id}")
            return

        # Compute statistics
        percentiles = np.percentile(activations, [25, 50, 75])
        n_zeros = np.sum(activations == 0)

        # Compute histogram
        hist, bin_edges = np.histogram(activations, bins=50)
        histogram_data = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        }

        stats = {
            "feature_id": feature_id,
            "mean": float(activations.mean()),
            "std": float(activations.std()),
            "min_val": float(activations.min()),
            "max_val": float(activations.max()),
            "median": float(percentiles[1]),
            "q25": float(percentiles[0]),
            "q75": float(percentiles[2]),
            "n_zeros": int(n_zeros),
            "n_total": len(activations),
            "sparsity": float(n_zeros / len(activations)),
            "histogram_data": json.dumps(histogram_data),
        }

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO feature_stats
                (feature_id, mean, std, min_val, max_val, median, q25, q75,
                 n_zeros, n_total, sparsity, histogram_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    stats["feature_id"],
                    stats["mean"],
                    stats["std"],
                    stats["min_val"],
                    stats["max_val"],
                    stats["median"],
                    stats["q25"],
                    stats["q75"],
                    stats["n_zeros"],
                    stats["n_total"],
                    stats["sparsity"],
                    stats["histogram_data"],
                ),
            )
            conn.commit()

    def store_top_examples(
        self,
        feature_id: int,
        example_ids: List[int],
        activations: List[float],
        top_k: int = 50,
    ) -> None:
        """Store top examples for a feature.

        Args:
            feature_id: Feature ID
            example_ids: List of example IDs
            activations: Corresponding activation values
            top_k: Number of top examples to store
        """
        if len(example_ids) != len(activations):
            raise ValueError(
                "Example IDs and activations must have same length"
            )

        # Sort by activation value (descending) and take top k
        sorted_pairs = sorted(
            zip(example_ids, activations), key=lambda x: x[1], reverse=True
        )[:top_k]

        top_data = [
            (feature_id, ex_id, rank + 1, activation)
            for rank, (ex_id, activation) in enumerate(sorted_pairs)
        ]

        with self.get_connection() as conn:
            # Delete existing entries for this feature
            conn.execute(
                "DELETE FROM top_examples WHERE feature_id = ?", (feature_id,)
            )

            # Insert new entries
            conn.executemany(
                """
                INSERT INTO top_examples (feature_id, example_id, rank, activation_value)
                VALUES (?, ?, ?, ?)
            """,
                top_data,
            )
            conn.commit()

    def get_database_info(self) -> Dict[str, Any]:
        """Get general database information."""
        with self.get_connection() as conn:
            info = {}

            # Count rows in each table
            tables = [
                "examples",
                "activations",
                "feature_stats",
                "top_examples",
                "feature_correlations",
                "logit_influences",
            ]

            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                info[f"{table}_count"] = cursor.fetchone()[0]

            # Get number of unique features
            cursor = conn.execute(
                "SELECT COUNT(DISTINCT feature_id) FROM activations"
            )
            info["unique_features"] = cursor.fetchone()[0]

            # Database file size
            info["db_size_mb"] = self.db_path.stat().st_size / (1024 * 1024)

            return info

    def vacuum(self) -> None:
        """Optimize database by running VACUUM."""
        with self.get_connection() as conn:
            conn.execute("VACUUM")
            logger.info("Database optimized with VACUUM")


class DatabaseBackedContext:
    """Context manager that provides database-backed data access."""

    def __init__(self, db_manager: DashboardDatabase):
        """Initialize with database manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self._cache = {}  # Simple in-memory cache

    def get_all_feature_ids(self) -> List[int]:
        """Get a sorted list of all unique feature IDs from feature_stats."""
        # This method could be cached if the list of features is static during a session.
        # For simplicity, direct DB call first.
        return self.db.get_all_feature_ids()

    def get_feature_activations(self, feature_id: int) -> np.ndarray:
        """
        DEPRECATED. Calls the deprecated DashboardDatabase.get_feature_activations.
        Returns empty data and logs a warning via the underlying method.
        """
        # No caching needed as it returns empty and logs.
        activations_array, _ = self.db.get_feature_activations(feature_id)
        return activations_array

    def get_binned_feature_samples(
        self, feature_id: int
    ) -> List[Dict[str, Any]]:
        """
        Get binned feature samples. This method is not typically cached
        unless specific performance analysis suggests it's a bottleneck for repeated calls
        with the same feature_id within a short timeframe/single request.
        """
        # Caching for this method might be complex if results are large.
        # For now, direct call without caching.
        return self.db.get_binned_feature_samples(feature_id)

    def get_top_examples(
        self, feature_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top examples for a feature."""
        # Typically, top examples might not change rapidly for a given feature,
        # but caching depends on usage patterns. For simplicity, direct call.
        return self.db.get_top_examples(feature_id, limit)

    def get_feature_statistics(
        self, feature_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get feature statistics (cached)."""
        cache_key = f"stats_{feature_id}"
        if cache_key not in self._cache:
            stats = self.db.get_feature_statistics(feature_id)
            # Cache even if stats is None to avoid re-querying for non-existent features
            self._cache[cache_key] = stats
        # Return a copy if stats is a mutable dict to prevent modification of cached object
        cached_val = self._cache.get(cache_key)
        return cached_val.copy() if isinstance(cached_val, dict) else cached_val

    def get_logit_influences(
        self, feature_id: int, limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get logit influences (cached)."""
        cache_key = f"logit_influences_{feature_id}_{limit}"  # Include limit in cache key
        if cache_key not in self._cache:
            influences = self.db.get_logit_influences(feature_id, limit)
            self._cache[cache_key] = influences
        # Return a copy to prevent modification of cached object
        cached_val = self._cache.get(
            cache_key, {"positive": [], "negative": []}
        )
        return {
            "positive": [
                item.copy() for item in cached_val.get("positive", [])
            ],
            "negative": [
                item.copy() for item in cached_val.get("negative", [])
            ],
        }

    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("Cache cleared")


class DuckDBDatabase:
    """DuckDB-based database manager for dashboard data."""

    def __init__(self, db_path: Union[str, Path]):
        """Initialize database connection.

        Args:
            db_path: Path to DuckDB database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))

        # Initialize schema
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema."""
        # Create tables for neuron data
        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS neuron_pairs (
            id INTEGER PRIMARY KEY,
            token1 TEXT,
            token2 TEXT,
            activation FLOAT,
            weapon_id INTEGER
        )
        """
        )

        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS neuron_single_tokens (
            id INTEGER PRIMARY KEY,
            token TEXT,
            activation FLOAT,
            weapon_id INTEGER
        )
        """
        )

        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS neuron_triples (
            id INTEGER PRIMARY KEY,
            token1 TEXT,
            token2 TEXT,
            token3 TEXT,
            activation FLOAT,
            weapon_id INTEGER
        )
        """
        )

        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS neuron_activations (
            id INTEGER PRIMARY KEY,
            activation FLOAT,
            example_idx INTEGER
        )
        """
        )

        self.conn.execute(
            """
        CREATE TABLE IF NOT EXISTS analysis_data (
            id INTEGER PRIMARY KEY,
            feature_id INTEGER,
            mean_activation FLOAT,
            std_activation FLOAT,
            max_activation FLOAT,
            min_activation FLOAT,
            sparsity FLOAT,
            n_examples INTEGER
        )
        """
        )

    def insert_dataframe(self, table_name: str, df: pd.DataFrame) -> None:
        """Insert a DataFrame into a table.

        Args:
            table_name: Name of the table to insert into
            df: DataFrame to insert
        """
        self.conn.execute(f"INSERT INTO {table_name} SELECT * FROM df")

    def get_feature_histogram(self, feature_id: int) -> pd.DataFrame:
        """Get activation histogram for a feature.

        Args:
            feature_id: ID of the feature

        Returns:
            DataFrame with histogram data
        """
        return self.conn.execute(
            """
        SELECT 
            activation,
            COUNT(*) as count
        FROM neuron_activations
        WHERE feature_id = ?
        GROUP BY activation
        ORDER BY activation
        """,
            [feature_id],
        ).df()

    def get_top_examples(
        self, feature_id: int, limit: int = 100
    ) -> pd.DataFrame:
        """Get top examples for a feature.

        Args:
            feature_id: ID of the feature
            limit: Maximum number of examples to return

        Returns:
            DataFrame with top examples
        """
        return self.conn.execute(
            """
        SELECT 
            token1,
            token2,
            activation,
            weapon_id
        FROM neuron_pairs
        WHERE feature_id = ?
        ORDER BY activation DESC
        LIMIT ?
        """,
            [feature_id, limit],
        ).df()

    def get_single_token_examples(
        self, feature_id: int, limit: int = 100
    ) -> pd.DataFrame:
        """Get single token examples for a feature.

        Args:
            feature_id: ID of the feature
            limit: Maximum number of examples to return

        Returns:
            DataFrame with single token examples
        """
        return self.conn.execute(
            """
        SELECT 
            token,
            activation,
            weapon_id
        FROM neuron_single_tokens
        WHERE feature_id = ?
        ORDER BY activation DESC
        LIMIT ?
        """,
            [feature_id, limit],
        ).df()

    def get_triple_examples(
        self, feature_id: int, limit: int = 100
    ) -> pd.DataFrame:
        """Get triple token examples for a feature.

        Args:
            feature_id: ID of the feature
            limit: Maximum number of examples to return

        Returns:
            DataFrame with triple token examples
        """
        return self.conn.execute(
            """
        SELECT 
            token1,
            token2,
            token3,
            activation,
            weapon_id
        FROM neuron_triples
        WHERE feature_id = ?
        ORDER BY activation DESC
        LIMIT ?
        """,
            [feature_id, limit],
        ).df()

    def get_feature_stats(self, feature_id: int) -> Dict[str, float]:
        """Get statistics for a feature.

        Args:
            feature_id: ID of the feature

        Returns:
            Dictionary with feature statistics
        """
        stats = self.conn.execute(
            """
        SELECT 
            mean_activation,
            std_activation,
            max_activation,
            min_activation,
            sparsity,
            n_examples
        FROM analysis_data
        WHERE feature_id = ?
        """,
            [feature_id],
        ).fetchone()

        if stats is None:
            return {}

        return {
            "mean": stats[0],
            "std": stats[1],
            "max": stats[2],
            "min": stats[3],
            "sparsity": stats[4],
            "n_examples": stats[5],
        }

    def get_all_feature_ids(self) -> List[int]:
        """Get all feature IDs in the database.

        Returns:
            List of feature IDs
        """
        return self.conn.execute(
            """
        SELECT DISTINCT feature_id
        FROM analysis_data
        ORDER BY feature_id
        """
        ).fetchall()

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the database.

        Returns:
            Dictionary with database information
        """
        info = {}

        # Get table sizes
        for table in [
            "neuron_pairs",
            "neuron_single_tokens",
            "neuron_triples",
            "neuron_activations",
            "analysis_data",
        ]:
            count = self.conn.execute(
                f"SELECT COUNT(*) FROM {table}"
            ).fetchone()[0]
            info[f"{table}_count"] = count

        # Get feature count
        feature_count = self.conn.execute(
            "SELECT COUNT(DISTINCT feature_id) FROM analysis_data"
        ).fetchone()[0]
        info["feature_count"] = feature_count

        return info

    def get_logit_influences(
        self, feature_id: int, limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get top logit influences for a feature, separated by positive and negative.

        Args:
            feature_id: Feature to query.
            limit: Number of influences to return for each type (positive/negative).

        Returns:
            A dictionary with 'positive' and 'negative' influence lists.
            Since logit_influences table is not available, returns empty lists.
        """
        logger.warning("Logit influences table not available in database")
        return {"positive": [], "negative": []}
