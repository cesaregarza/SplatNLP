#!/usr/bin/env python3
"""
Database manager for scalable dashboard data storage.

This module provides a SQLite-based solution to replace memory-intensive 
data loading patterns in the dashboard. It stores activations, metadata,
and precomputed analytics in an efficient queryable format.
"""

import json
import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

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
            self.db_path,
            timeout=30.0,
            check_same_thread=False
        )
        conn.row_factory = sqlite3.Row  # Enable dict-like access
        try:
            yield conn
        finally:
            conn.close()
    
    def _create_schema(self, conn: sqlite3.Connection) -> None:
        """Create database schema."""
        
        # Examples metadata table
        conn.execute("""
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
        """)
        
        # Feature activations table (sparse storage)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS activations (
                example_id INTEGER,
                feature_id INTEGER,
                activation_value REAL,
                PRIMARY KEY (example_id, feature_id),
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """)
        
        # Feature statistics table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_stats (
                feature_id INTEGER PRIMARY KEY,
                mean REAL,
                std REAL,
                min_val REAL,
                max_val REAL,
                median REAL,
                q25 REAL,
                q75 REAL,
                n_zeros INTEGER,
                n_total INTEGER,
                sparsity REAL,
                histogram_data TEXT  -- JSON with counts and bin_edges
            )
        """)
        
        # Top examples per feature (precomputed for speed)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS top_examples (
                feature_id INTEGER,
                example_id INTEGER,
                rank INTEGER,
                activation_value REAL,
                PRIMARY KEY (feature_id, example_id),
                FOREIGN KEY (example_id) REFERENCES examples(id)
            )
        """)
        
        # Feature correlations (precomputed)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS feature_correlations (
                feature_a INTEGER,
                feature_b INTEGER,
                correlation REAL,
                PRIMARY KEY (feature_a, feature_b)
            )
        """)
        
        # Logit influences (precomputed)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS logit_influences (
                feature_id INTEGER,
                token_id INTEGER,
                token_name TEXT,
                influence REAL,
                rank INTEGER,
                PRIMARY KEY (feature_id, token_id)
            )
        """)
        
        # Create indexes for performance
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_activations_feature ON activations(feature_id)",
            "CREATE INDEX IF NOT EXISTS idx_activations_example ON activations(example_id)", 
            "CREATE INDEX IF NOT EXISTS idx_activations_value ON activations(activation_value)",
            "CREATE INDEX IF NOT EXISTS idx_top_examples_feature ON top_examples(feature_id)",
            "CREATE INDEX IF NOT EXISTS idx_top_examples_rank ON top_examples(feature_id, rank)",
            "CREATE INDEX IF NOT EXISTS idx_correlations_feature_a ON feature_correlations(feature_a)",
            "CREATE INDEX IF NOT EXISTS idx_logit_influences_feature ON logit_influences(feature_id)",
            "CREATE INDEX IF NOT EXISTS idx_logit_influences_rank ON logit_influences(feature_id, rank)",
            "CREATE INDEX IF NOT EXISTS idx_examples_weapon ON examples(weapon_id)",
        ]
        
        for idx_sql in indexes:
            conn.execute(idx_sql)
        
        conn.commit()
        logger.info("Database schema initialized")
    
    def insert_examples_batch(self, examples_data: List[Dict[str, Any]]) -> None:
        """Insert examples in batch for efficiency.
        
        Args:
            examples_data: List of example dictionaries
        """
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO examples 
                (id, weapon_id, weapon_name, ability_input_tokens, 
                 input_abilities_str, top_predicted_abilities_str, 
                 is_null_token, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    ex['id'],
                    ex.get('weapon_id', -1),
                    ex.get('weapon_name', ''),
                    json.dumps(ex.get('ability_input_tokens', [])),
                    ex.get('input_abilities_str', ''),
                    ex.get('top_predicted_abilities_str', ''),
                    ex.get('is_null_token', False),
                    json.dumps(ex.get('metadata', {}))
                )
                for ex in examples_data
            ])
            conn.commit()
            logger.info(f"Inserted {len(examples_data)} examples")
    
    def insert_activations_batch(
        self, 
        activations: np.ndarray, 
        example_ids: List[int],
        activation_threshold: float = 1e-6
    ) -> None:
        """Insert activations efficiently using sparse storage.
        
        Args:
            activations: Array of shape (n_examples, n_features)
            example_ids: List of example IDs corresponding to rows
            activation_threshold: Only store activations above this threshold
        """
        if len(example_ids) != activations.shape[0]:
            raise ValueError("Number of example IDs must match activation rows")
        
        # Convert to sparse format (only non-zero activations)
        rows, cols = np.where(np.abs(activations) > activation_threshold)
        values = activations[rows, cols]
        
        activation_data = [
            (int(example_ids[row]), int(col), float(val))
            for row, col, val in zip(rows, cols, values)
        ]
        
        with self.get_connection() as conn:
            conn.executemany("""
                INSERT OR REPLACE INTO activations (example_id, feature_id, activation_value)
                VALUES (?, ?, ?)
            """, activation_data)
            conn.commit()
            
        logger.info(f"Inserted {len(activation_data)} sparse activations")
    
    def get_feature_activations(
        self, 
        feature_id: int, 
        limit: Optional[int] = None
    ) -> Tuple[np.ndarray, List[int]]:
        """Get all activations for a specific feature.
        
        Args:
            feature_id: Feature to query
            limit: Optional limit on number of results
            
        Returns:
            Tuple of (activation_values, example_ids)
        """
        with self.get_connection() as conn:
            query = """
                SELECT example_id, activation_value 
                FROM activations 
                WHERE feature_id = ?
                ORDER BY activation_value DESC
            """
            if limit:
                query += f" LIMIT {limit}"
            
            cursor = conn.execute(query, (feature_id,))
            results = cursor.fetchall()
        
        if not results:
            return np.array([]), []
        
        example_ids = [row['example_id'] for row in results]
        activations = np.array([row['activation_value'] for row in results])
        
        return activations, example_ids
    
    def get_top_examples(
        self, 
        feature_id: int, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top activating examples for a feature with metadata.
        
        Args:
            feature_id: Feature to query
            limit: Number of top examples to return
            
        Returns:
            List of example dictionaries with metadata
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT 
                    t.rank,
                    t.activation_value,
                    e.id,
                    e.weapon_name,
                    e.input_abilities_str,
                    e.top_predicted_abilities_str,
                    e.ability_input_tokens,
                    e.metadata
                FROM top_examples t
                JOIN examples e ON t.example_id = e.id
                WHERE t.feature_id = ?
                ORDER BY t.rank
                LIMIT ?
            """, (feature_id, limit))
            
            results = []
            for row in cursor.fetchall():
                result = {
                    'rank': row['rank'],
                    'activation_value': row['activation_value'],
                    'example_id': row['id'],
                    'weapon_name': row['weapon_name'],
                    'input_abilities_str': row['input_abilities_str'],
                    'top_predicted_abilities_str': row['top_predicted_abilities_str'],
                    'ability_input_tokens': json.loads(row['ability_input_tokens']),
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
                results.append(result)
            
            return results
    
    def get_feature_statistics(self, feature_id: int) -> Optional[Dict[str, Any]]:
        """Get precomputed statistics for a feature.
        
        Args:
            feature_id: Feature to query
            
        Returns:
            Dictionary with statistics or None if not found
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM feature_stats WHERE feature_id = ?
            """, (feature_id,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            stats = dict(row)
            # Parse histogram data
            if stats['histogram_data']:
                stats['histogram'] = json.loads(stats['histogram_data'])
            del stats['histogram_data']
            
            return stats
    
    def get_feature_correlations(
        self, 
        feature_id: int, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top correlated features.
        
        Args:
            feature_id: Feature to query
            limit: Number of correlations to return
            
        Returns:
            List of correlation dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT feature_b as feature_id, correlation
                FROM feature_correlations 
                WHERE feature_a = ?
                ORDER BY ABS(correlation) DESC
                LIMIT ?
            """, (feature_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def get_logit_influences(
        self, 
        feature_id: int, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top logit influences for a feature.
        
        Args:
            feature_id: Feature to query
            limit: Number of influences to return
            
        Returns:
            List of influence dictionaries
        """
        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT token_id, token_name, influence, rank
                FROM logit_influences 
                WHERE feature_id = ?
                ORDER BY rank
                LIMIT ?
            """, (feature_id, limit))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def compute_and_store_feature_stats(self, feature_id: int) -> None:
        """Compute and store statistics for a feature.
        
        Args:
            feature_id: Feature to compute stats for
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
            'counts': hist.tolist(),
            'bin_edges': bin_edges.tolist()
        }
        
        stats = {
            'feature_id': feature_id,
            'mean': float(activations.mean()),
            'std': float(activations.std()),
            'min_val': float(activations.min()),
            'max_val': float(activations.max()),
            'median': float(percentiles[1]),
            'q25': float(percentiles[0]),
            'q75': float(percentiles[2]),
            'n_zeros': int(n_zeros),
            'n_total': len(activations),
            'sparsity': float(n_zeros / len(activations)),
            'histogram_data': json.dumps(histogram_data)
        }
        
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO feature_stats
                (feature_id, mean, std, min_val, max_val, median, q25, q75,
                 n_zeros, n_total, sparsity, histogram_data)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                stats['feature_id'], stats['mean'], stats['std'],
                stats['min_val'], stats['max_val'], stats['median'],
                stats['q25'], stats['q75'], stats['n_zeros'],
                stats['n_total'], stats['sparsity'], stats['histogram_data']
            ))
            conn.commit()
    
    def store_top_examples(
        self, 
        feature_id: int, 
        example_ids: List[int], 
        activations: List[float],
        top_k: int = 50
    ) -> None:
        """Store top examples for a feature.
        
        Args:
            feature_id: Feature ID
            example_ids: List of example IDs
            activations: Corresponding activation values
            top_k: Number of top examples to store
        """
        if len(example_ids) != len(activations):
            raise ValueError("Example IDs and activations must have same length")
        
        # Sort by activation value (descending) and take top k
        sorted_pairs = sorted(
            zip(example_ids, activations), 
            key=lambda x: x[1], 
            reverse=True
        )[:top_k]
        
        top_data = [
            (feature_id, ex_id, rank + 1, activation)
            for rank, (ex_id, activation) in enumerate(sorted_pairs)
        ]
        
        with self.get_connection() as conn:
            # Delete existing entries for this feature
            conn.execute("DELETE FROM top_examples WHERE feature_id = ?", (feature_id,))
            
            # Insert new entries
            conn.executemany("""
                INSERT INTO top_examples (feature_id, example_id, rank, activation_value)
                VALUES (?, ?, ?, ?)
            """, top_data)
            conn.commit()
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get general database information."""
        with self.get_connection() as conn:
            info = {}
            
            # Count rows in each table
            tables = ['examples', 'activations', 'feature_stats', 
                     'top_examples', 'feature_correlations', 'logit_influences']
            
            for table in tables:
                cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
                info[f'{table}_count'] = cursor.fetchone()[0]
            
            # Get number of unique features
            cursor = conn.execute("SELECT COUNT(DISTINCT feature_id) FROM activations")
            info['unique_features'] = cursor.fetchone()[0]
            
            # Database file size
            info['db_size_mb'] = self.db_path.stat().st_size / (1024 * 1024)
            
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
        self._cache = {}  # Simple in-memory cache for frequently accessed data
        
    def get_feature_activations(self, feature_id: int) -> np.ndarray:
        """Get activations for a feature (cached)."""
        cache_key = f"activations_{feature_id}"
        if cache_key not in self._cache:
            activations, _ = self.db.get_feature_activations(feature_id)
            self._cache[cache_key] = activations
        return self._cache[cache_key]
    
    def get_top_examples(self, feature_id: int, limit: int = 20) -> List[Dict]:
        """Get top examples for a feature."""
        return self.db.get_top_examples(feature_id, limit)
    
    def get_feature_statistics(self, feature_id: int) -> Dict[str, Any]:
        """Get feature statistics (cached)."""
        cache_key = f"stats_{feature_id}"
        if cache_key not in self._cache:
            stats = self.db.get_feature_statistics(feature_id)
            if stats:
                self._cache[cache_key] = stats
        return self._cache.get(cache_key, {})
    
    def clear_cache(self) -> None:
        """Clear in-memory cache."""
        self._cache.clear()
        logger.info("Cache cleared")