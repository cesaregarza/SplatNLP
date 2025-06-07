#!/usr/bin/env python3
"""
Database manager for scalable dashboard data storage.

This module provides filesystem-based solutions to replace memory-intensive 
data loading patterns in the dashboard. It stores activations, metadata,
and precomputed analytics in an efficient queryable format.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DatabaseBackedContext:
    """Context manager for database-backed operations."""

    def __init__(self, db_manager: Any):
        """Initialize context manager.

        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self._cache = {}

    def get_all_feature_ids(self) -> List[int]:
        """Get all feature IDs."""
        return self.db.get_all_feature_ids()

    def get_feature_activations(self, feature_id: int) -> np.ndarray:
        """Get activations for a feature."""
        return self.db.get_feature_activations(feature_id)

    def get_binned_feature_samples(
        self, feature_id: int
    ) -> List[Dict[str, Any]]:
        """Get binned samples for a feature."""
        return self.db.get_binned_feature_samples(feature_id)

    def get_top_examples(
        self, feature_id: int, limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get top examples for a feature."""
        return self.db.get_top_examples(feature_id, limit)

    def get_feature_statistics(
        self, feature_id: int
    ) -> Optional[Dict[str, Any]]:
        """Get statistics for a feature."""
        return self.db.get_feature_statistics(feature_id)

    def get_logit_influences(
        self, feature_id: int, limit: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get logit influences for a feature."""
        return self.db.get_logit_influences(feature_id, limit)

    def clear_cache(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        logger.info("Cache cleared")
