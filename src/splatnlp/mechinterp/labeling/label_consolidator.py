"""Label consolidation between ResearchState and Dashboard systems.

This module provides two-way sync between:
- Dashboard labels (FeatureLabelsManager): name, category, notes
- ResearchState labels: feature_label, hypothesis-derived confidence

The consolidated view provides a single source of truth for labeling workflows.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# Default paths
LABELS_DIR = Path("/mnt/e/mechinterp_runs/labels")
STATE_DIR = Path("/mnt/e/mechinterp_runs/state")


@dataclass
class ConsolidatedLabel:
    """Merged label from all sources."""

    feature_id: int
    model_type: str

    # Dashboard fields
    dashboard_name: str | None = None
    dashboard_category: str | None = (
        None  # none, mechanical, tactical, strategic
    )
    dashboard_notes: str | None = None

    # ResearchState fields
    research_label: str | None = None
    research_state_path: str | None = None
    hypothesis_confidence: float | None = None  # From top hypothesis

    # Merged view
    display_name: str = ""  # Best available name
    source: str = "none"  # dashboard, research, merged, none
    last_updated: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["last_updated"] = self.last_updated.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "ConsolidatedLabel":
        """Create from dict."""
        if isinstance(data.get("last_updated"), str):
            data["last_updated"] = datetime.fromisoformat(data["last_updated"])
        return cls(**data)


class LabelConsolidator:
    """Sync labels between ResearchState and Dashboard.

    Provides a unified view of all labels and enables two-way sync.
    """

    def __init__(
        self,
        model_type: Literal["full", "ultra"],
        labels_dir: Path | None = None,
        state_dir: Path | None = None,
    ):
        """Initialize the consolidator.

        Args:
            model_type: Model type (full or ultra)
            labels_dir: Directory for consolidated labels
            state_dir: Directory for ResearchState files
        """
        self.model_type = model_type
        self.labels_dir = labels_dir or LABELS_DIR
        self.state_dir = state_dir or STATE_DIR

        # Ensure directories exist
        self.labels_dir.mkdir(parents=True, exist_ok=True)

        # Consolidated file path
        self.consolidated_path = (
            self.labels_dir / f"consolidated_{model_type}.json"
        )

        # Cache
        self._labels: dict[int, ConsolidatedLabel] = {}
        self._dashboard_manager = None

    @property
    def dashboard_manager(self):
        """Lazy load dashboard manager."""
        if self._dashboard_manager is None:
            from splatnlp.dashboard.components.feature_labels import (
                FeatureLabelsManager,
            )

            self._dashboard_manager = FeatureLabelsManager(
                model_type=self.model_type
            )
        return self._dashboard_manager

    def load_consolidated(self) -> dict[int, ConsolidatedLabel]:
        """Load consolidated labels from file."""
        if self.consolidated_path.exists():
            with open(self.consolidated_path) as f:
                data = json.load(f)
            self._labels = {
                int(k): ConsolidatedLabel.from_dict(v) for k, v in data.items()
            }
        else:
            self._labels = {}
        return self._labels

    def save_consolidated(self) -> None:
        """Save consolidated labels to file."""
        data = {str(k): v.to_dict() for k, v in self._labels.items()}
        with open(self.consolidated_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(
            f"Saved {len(self._labels)} labels to {self.consolidated_path}"
        )

    def get_label(self, feature_id: int) -> ConsolidatedLabel | None:
        """Get consolidated label for a feature."""
        if not self._labels:
            self.load_consolidated()
        return self._labels.get(feature_id)

    def set_label(
        self,
        feature_id: int,
        name: str,
        category: str = "none",
        notes: str = "",
        source: str = "dashboard",
        sync_to_dashboard: bool = True,
        sync_to_research: bool = True,
    ) -> ConsolidatedLabel:
        """Set a label for a feature.

        Updates both the consolidated store and optionally syncs to
        dashboard and research state.

        Args:
            feature_id: Feature ID
            name: Label name
            category: Category (none, mechanical, tactical, strategic)
            notes: Additional notes
            source: Label source (e.g., "dashboard", "claude code", "research")
            sync_to_dashboard: Update dashboard labels
            sync_to_research: Update research state

        Returns:
            Updated ConsolidatedLabel
        """
        # Load existing if present
        existing = self.get_label(feature_id) or ConsolidatedLabel(
            feature_id=feature_id,
            model_type=self.model_type,
        )

        # Update fields
        existing.display_name = name
        existing.dashboard_name = name
        existing.dashboard_category = category
        existing.dashboard_notes = notes
        existing.source = source
        existing.last_updated = datetime.now()

        self._labels[feature_id] = existing

        # Sync to dashboard
        if sync_to_dashboard:
            self._sync_to_dashboard(feature_id, name, category, notes)

        # Sync to research state
        if sync_to_research:
            self._sync_to_research_state(feature_id, name)

        # Save consolidated
        self.save_consolidated()

        return existing

    def _sync_to_dashboard(
        self, feature_id: int, name: str, category: str, notes: str
    ) -> None:
        """Sync a label to the dashboard FeatureLabelsManager."""
        try:
            from splatnlp.dashboard.components.feature_labels import (
                FeatureLabel,
            )

            label = FeatureLabel(
                name=name,
                category=category,
                notes=notes,
            )
            self.dashboard_manager.set_label(feature_id, label)
            logger.info(f"Synced label to dashboard for feature {feature_id}")
        except Exception as e:
            logger.warning(f"Failed to sync to dashboard: {e}")

    def _sync_to_research_state(self, feature_id: int, name: str) -> None:
        """Sync a label to the ResearchState file."""
        try:
            from splatnlp.mechinterp.state import ResearchStateManager

            manager = ResearchStateManager(
                feature_id=feature_id,
                model_type=self.model_type,
            )
            state = manager.state
            state.feature_label = name
            state.add_history(
                action="label_synced",
                details=f"Label set to '{name}' via consolidator",
            )
            manager.save()
            logger.info(
                f"Synced label to research state for feature {feature_id}"
            )
        except Exception as e:
            logger.debug(
                f"No research state to sync for feature {feature_id}: {e}"
            )

    def sync_from_all_sources(self) -> int:
        """Pull labels from dashboard and research states.

        Scans all available sources and consolidates them.

        Returns:
            Number of labels consolidated
        """
        # Start from existing consolidated data so we don't drop prior labels
        self.load_consolidated()
        count = 0

        # 1. Load from dashboard
        logger.info("Loading labels from dashboard...")
        for fid, label in self.dashboard_manager.feature_labels.items():
            existing = self._labels.get(fid)

            if existing is None:
                existing = ConsolidatedLabel(
                    feature_id=fid,
                    model_type=self.model_type,
                )

            # Update from dashboard
            if label.name:
                existing.dashboard_name = label.name
                existing.dashboard_category = label.category
                existing.dashboard_notes = label.notes

                # Set display name if not already set
                if not existing.display_name:
                    existing.display_name = label.name
                    existing.source = "dashboard"

                self._labels[fid] = existing
                count += 1

        # 2. Load from research states
        logger.info("Loading labels from research states...")
        state_dirs = []
        if self.state_dir.exists():
            state_dirs.append(self.state_dir)
        model_subdir = self.state_dir / self.model_type
        if model_subdir.exists():
            # Prefer model-specific subdirectory if present
            state_dirs.insert(0, model_subdir)

        seen_files = set()
        for state_dir in state_dirs:
            for state_file in state_dir.glob("feature_*_*.json"):
                if state_file in seen_files:
                    continue
                seen_files.add(state_file)
                try:
                    with open(state_file) as f:
                        data = json.load(f)

                    fid = data.get("feature_id")
                    if fid is None:
                        continue
                    if data.get("model_type") != self.model_type:
                        continue

                    research_label = data.get("feature_label")
                    hypotheses = data.get("hypotheses", [])

                    # Get confidence from top hypothesis
                    confidence = None
                    if hypotheses:
                        top_h = max(
                            hypotheses,
                            key=lambda h: h.get("confidence", 0),
                        )
                        confidence = top_h.get("confidence")

                    existing = self._labels.get(fid)
                    if existing is None:
                        existing = ConsolidatedLabel(
                            feature_id=fid,
                            model_type=self.model_type,
                        )

                    # Update from research state
                    existing.research_label = research_label
                    existing.research_state_path = str(state_file)
                    existing.hypothesis_confidence = confidence

                    # Set display name if not already set
                    if not existing.display_name and research_label:
                        existing.display_name = research_label
                        existing.source = "research"

                    self._labels[fid] = existing
                    count += 1

                except Exception as e:
                    logger.warning(f"Error loading {state_file}: {e}")

        # 3. Merge sources
        for label in self._labels.values():
            if label.dashboard_name and label.research_label:
                label.source = "merged"

        self.save_consolidated()
        logger.info(f"Consolidated {count} labels from all sources")
        return count

    def get_unlabeled_features(self, all_features: list[int]) -> list[int]:
        """Get features that don't have labels.

        Args:
            all_features: List of all feature IDs to check

        Returns:
            List of unlabeled feature IDs
        """
        if not self._labels:
            self.load_consolidated()

        labeled = set(self._labels.keys())
        return [fid for fid in all_features if fid not in labeled]

    def get_labeled_features(self) -> list[int]:
        """Get all labeled feature IDs."""
        if not self._labels:
            self.load_consolidated()
        return list(self._labels.keys())

    def get_by_category(self, category: str) -> list[ConsolidatedLabel]:
        """Get all labels with a specific category."""
        if not self._labels:
            self.load_consolidated()
        return [
            label
            for label in self._labels.values()
            if label.dashboard_category == category
        ]

    def search(self, query: str) -> list[ConsolidatedLabel]:
        """Search labels by name, category, or notes."""
        if not self._labels:
            self.load_consolidated()

        query = query.lower()
        results = []
        for label in self._labels.values():
            if (
                (label.display_name and query in label.display_name.lower())
                or (
                    label.dashboard_category
                    and query in label.dashboard_category.lower()
                )
                or (
                    label.dashboard_notes
                    and query in label.dashboard_notes.lower()
                )
                or (
                    label.research_label
                    and query in label.research_label.lower()
                )
            ):
                results.append(label)
        return results

    def export_csv(self, output_path: Path | str) -> None:
        """Export labels to CSV format."""
        import csv

        if not self._labels:
            self.load_consolidated()

        output_path = Path(output_path)
        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "feature_id",
                    "display_name",
                    "category",
                    "confidence",
                    "source",
                    "notes",
                ]
            )
            for label in sorted(
                self._labels.values(), key=lambda x: x.feature_id
            ):
                writer.writerow(
                    [
                        label.feature_id,
                        label.display_name,
                        label.dashboard_category or "",
                        label.hypothesis_confidence or "",
                        label.source,
                        label.dashboard_notes or "",
                    ]
                )
        logger.info(f"Exported {len(self._labels)} labels to {output_path}")

    def get_statistics(self) -> dict:
        """Get labeling statistics."""
        if not self._labels:
            self.load_consolidated()

        stats = {
            "total_labeled": len(self._labels),
            "from_dashboard": sum(
                1 for l in self._labels.values() if l.source == "dashboard"
            ),
            "from_research": sum(
                1 for l in self._labels.values() if l.source == "research"
            ),
            "merged": sum(
                1 for l in self._labels.values() if l.source == "merged"
            ),
            "by_category": {},
        }

        # Count by category
        for label in self._labels.values():
            cat = label.dashboard_category or "uncategorized"
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1

        return stats
