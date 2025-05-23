"""Feature naming management component for the dashboard."""

import json
from pathlib import Path
from typing import Dict, Optional

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, State, dcc, html


class FeatureNamesManager:
    """Manages feature names with persistent storage."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize the feature names manager.

        Args:
            storage_path: Path to store feature names JSON.
                         Defaults to dashboard directory.
        """
        if storage_path is None:
            storage_path = Path(__file__).parent.parent / "feature_names.json"
        self.storage_path = storage_path
        self.feature_names = self._load_names()

    def _load_names(self) -> Dict[int, str]:
        """Load feature names from storage."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    data = json.load(f)
                    # Convert string keys to int
                    return {int(k): v for k, v in data.items()}
            except Exception:
                return {}
        return {}

    def save_names(self) -> None:
        """Save feature names to storage."""
        with open(self.storage_path, "w") as f:
            json.dump(self.feature_names, f, indent=2)

    def get_name(self, feature_id: int) -> Optional[str]:
        """Get the name for a feature ID."""
        return self.feature_names.get(feature_id)

    def set_name(self, feature_id: int, name: str) -> None:
        """Set the name for a feature ID."""
        if name.strip():
            self.feature_names[feature_id] = name.strip()
        elif feature_id in self.feature_names:
            del self.feature_names[feature_id]
        self.save_names()

    def get_display_name(self, feature_id: int) -> str:
        """Get display name (custom name or default)."""
        name = self.get_name(feature_id)
        if name:
            return f"Feature {feature_id}: {name}"
        return f"Feature {feature_id}"

    def search_names(self, query: str) -> Dict[int, str]:
        """Search feature names by query (searches both ID and name)."""
        query = query.lower().strip()
        results = {}

        # Search by ID
        try:
            feature_id = int(query)
            if 0 <= feature_id < 1000:  # Assuming reasonable feature count
                results[feature_id] = self.get_display_name(feature_id)
        except ValueError:
            pass

        # Search by name
        for fid, name in self.feature_names.items():
            if query in name.lower():
                results[fid] = self.get_display_name(fid)

        return results
