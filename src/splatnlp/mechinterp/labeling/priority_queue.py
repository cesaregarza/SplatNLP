"""Priority queue for feature labeling workflow.

Manages which features to label next based on priority scoring.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

logger = logging.getLogger(__name__)

QUEUE_DIR = Path("/mnt/e/mechinterp_runs/labels")


@dataclass
class QueueEntry:
    """A single entry in the labeling queue."""

    feature_id: int
    model_type: str
    priority: float  # Higher = label sooner
    reason: str  # Why this priority
    added_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        d = asdict(self)
        d["added_at"] = self.added_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "QueueEntry":
        """Create from dict."""
        if isinstance(data.get("added_at"), str):
            data["added_at"] = datetime.fromisoformat(data["added_at"])
        return cls(**data)


@dataclass
class LabelingQueue:
    """Priority queue for labeling workflow.

    Tracks which features need labeling and their priority.
    """

    model_type: str
    entries: list[QueueEntry] = field(default_factory=list)
    completed: list[int] = field(default_factory=list)
    skipped: list[int] = field(default_factory=list)
    queue_dir: Path = field(default_factory=lambda: QUEUE_DIR)

    def __post_init__(self):
        """Ensure queue directory exists."""
        self.queue_dir.mkdir(parents=True, exist_ok=True)

    @property
    def queue_path(self) -> Path:
        """Path to queue JSON file."""
        return self.queue_dir / f"queue_{self.model_type}.json"

    def save(self) -> None:
        """Save queue to disk."""
        data = {
            "model_type": self.model_type,
            "entries": [e.to_dict() for e in self.entries],
            "completed": self.completed,
            "skipped": self.skipped,
        }
        with open(self.queue_path, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved queue with {len(self.entries)} entries")

    @classmethod
    def load(
        cls, model_type: Literal["full", "ultra"], queue_dir: Path | None = None
    ) -> "LabelingQueue":
        """Load queue from disk."""
        queue_dir = queue_dir or QUEUE_DIR
        queue_path = queue_dir / f"queue_{model_type}.json"

        if queue_path.exists():
            with open(queue_path) as f:
                data = json.load(f)
            return cls(
                model_type=data["model_type"],
                entries=[
                    QueueEntry.from_dict(e) for e in data.get("entries", [])
                ],
                completed=data.get("completed", []),
                skipped=data.get("skipped", []),
                queue_dir=queue_dir,
            )
        return cls(model_type=model_type, queue_dir=queue_dir)

    def add(
        self,
        feature_id: int,
        priority: float = 0.5,
        reason: str = "manual add",
    ) -> QueueEntry:
        """Add a feature to the queue.

        Args:
            feature_id: Feature ID to add
            priority: Priority score (higher = sooner)
            reason: Why this feature was added

        Returns:
            The created QueueEntry
        """
        # Check if already in queue
        for entry in self.entries:
            if entry.feature_id == feature_id:
                logger.info(f"Feature {feature_id} already in queue")
                return entry

        # Check if already completed or skipped
        if feature_id in self.completed:
            logger.info(f"Feature {feature_id} already completed")
            return None
        if feature_id in self.skipped:
            logger.info(f"Feature {feature_id} was skipped, re-adding")
            self.skipped.remove(feature_id)

        entry = QueueEntry(
            feature_id=feature_id,
            model_type=self.model_type,
            priority=priority,
            reason=reason,
        )
        self.entries.append(entry)
        self._sort_by_priority()
        self.save()
        return entry

    def add_batch(
        self,
        feature_ids: list[int],
        priority: float = 0.5,
        reason: str = "batch add",
    ) -> int:
        """Add multiple features to the queue.

        Returns:
            Number of features added
        """
        added = 0
        for fid in feature_ids:
            if self.add(fid, priority, reason):
                added += 1
        return added

    def get_next(self) -> QueueEntry | None:
        """Get the next feature to label (highest priority).

        Returns:
            Next QueueEntry or None if queue is empty
        """
        self._sort_by_priority()
        if not self.entries:
            return None
        return self.entries[0]

    def peek(self, n: int = 5) -> list[QueueEntry]:
        """Peek at the top N entries without removing.

        Args:
            n: Number of entries to return

        Returns:
            List of top N QueueEntries
        """
        self._sort_by_priority()
        return self.entries[:n]

    def mark_complete(self, feature_id: int, label: str | None = None) -> bool:
        """Mark a feature as labeled.

        Args:
            feature_id: Feature ID that was labeled
            label: The label that was applied (for logging)

        Returns:
            True if feature was in queue
        """
        for i, entry in enumerate(self.entries):
            if entry.feature_id == feature_id:
                self.entries.pop(i)
                self.completed.append(feature_id)
                self.save()
                logger.info(
                    f"Marked feature {feature_id} complete"
                    + (f" with label '{label}'" if label else "")
                )
                return True
        return False

    def mark_skipped(self, feature_id: int, reason: str = "") -> bool:
        """Mark a feature as skipped.

        Args:
            feature_id: Feature ID to skip
            reason: Why it was skipped

        Returns:
            True if feature was in queue
        """
        for i, entry in enumerate(self.entries):
            if entry.feature_id == feature_id:
                self.entries.pop(i)
                self.skipped.append(feature_id)
                self.save()
                logger.info(f"Skipped feature {feature_id}: {reason}")
                return True
        return False

    def update_priority(self, feature_id: int, new_priority: float) -> bool:
        """Update priority for a feature.

        Args:
            feature_id: Feature ID
            new_priority: New priority score

        Returns:
            True if feature was found and updated
        """
        for entry in self.entries:
            if entry.feature_id == feature_id:
                entry.priority = new_priority
                self._sort_by_priority()
                self.save()
                return True
        return False

    def remove(self, feature_id: int) -> bool:
        """Remove a feature from the queue without marking complete/skipped.

        Returns:
            True if feature was found and removed
        """
        for i, entry in enumerate(self.entries):
            if entry.feature_id == feature_id:
                self.entries.pop(i)
                self.save()
                return True
        return False

    def clear(self) -> None:
        """Clear the queue (keeps completed/skipped history)."""
        self.entries = []
        self.save()

    def reset(self) -> None:
        """Reset everything including history."""
        self.entries = []
        self.completed = []
        self.skipped = []
        self.save()

    def _sort_by_priority(self) -> None:
        """Sort entries by priority (highest first)."""
        self.entries.sort(key=lambda e: e.priority, reverse=True)

    def get_statistics(self) -> dict:
        """Get queue statistics."""
        return {
            "pending": len(self.entries),
            "completed": len(self.completed),
            "skipped": len(self.skipped),
            "total_processed": len(self.completed) + len(self.skipped),
        }

    def __len__(self) -> int:
        """Number of pending entries."""
        return len(self.entries)


class QueueBuilder:
    """Helper to build labeling queues with smart prioritization."""

    def __init__(self, model_type: Literal["full", "ultra"]):
        """Initialize queue builder.

        Args:
            model_type: Model type for database access
        """
        self.model_type = model_type
        self._db = None
        self._consolidator = None
        self._ctx = None

    @property
    def db(self):
        """Lazy load database."""
        if self._db is None:
            self._db = self.ctx.db
        return self._db

    @property
    def ctx(self):
        """Lazy load mechinterp context for model-aware paths."""
        if self._ctx is None:
            from splatnlp.mechinterp.skill_helpers.context_loader import (
                load_context,
            )

            self._ctx = load_context(self.model_type)
        return self._ctx

    @property
    def consolidator(self):
        """Lazy load label consolidator."""
        if self._consolidator is None:
            from splatnlp.mechinterp.labeling.label_consolidator import (
                LabelConsolidator,
            )

            self._consolidator = LabelConsolidator(self.model_type)
        return self._consolidator

    def build_by_activation_count(
        self,
        top_k: int = 100,
        exclude_labeled: bool = True,
    ) -> LabelingQueue:
        """Build queue prioritizing features by activation count.

        Features with more activations are easier to interpret.

        Args:
            top_k: Number of features to add
            exclude_labeled: Skip already-labeled features

        Returns:
            LabelingQueue with prioritized entries
        """
        queue = LabelingQueue.load(self.model_type)

        # Get feature counts from database
        logger.info("Getting feature activation counts...")
        feature_counts = self._get_feature_counts()

        if not feature_counts:
            logger.warning("Could not load feature counts from database")
            return queue

        # Sort by count (descending)
        feature_counts.sort(key=lambda x: x[1], reverse=True)

        # Get labeled features
        labeled = set()
        if exclude_labeled:
            self.consolidator.load_consolidated()
            labeled = set(self.consolidator.get_labeled_features())
            labeled.update(queue.completed)
            labeled.update(queue.skipped)

        # Add to queue
        added = 0
        for fid, count in feature_counts:
            if added >= top_k:
                break
            if fid in labeled:
                continue
            if fid in [e.feature_id for e in queue.entries]:
                continue

            # Priority based on count (normalized)
            max_count = feature_counts[0][1] if feature_counts else 1
            priority = count / max_count

            queue.add(
                feature_id=fid,
                priority=priority,
                reason=f"activation_count={count}",
            )
            added += 1

        logger.info(f"Added {added} features to queue by activation count")
        return queue

    def _get_feature_counts(self) -> list[tuple[int, int]]:
        """Extract feature counts from available database metadata."""
        counts: list[tuple[int, int]] = []
        db = self.db

        # Prefer explicit feature_index if available
        if hasattr(db, "feature_index") and getattr(db, "feature_index"):
            index: Any = db.feature_index

            # Polars DataFrame (ultra efficient store)
            try:
                import polars as pl

                if isinstance(index, pl.DataFrame):
                    count_col = None
                    for candidate in (
                        "total_activations",
                        "n_examples",
                        "count",
                    ):
                        if candidate in index.columns:
                            count_col = candidate
                            break

                    if count_col:
                        counts = [
                            (int(row["feature_id"]), int(row[count_col]))
                            for row in index.select(
                                ["feature_id", count_col]
                            ).to_dicts()
                        ]
            except Exception as e:  # pragma: no cover - defensive
                logger.debug(f"Could not parse feature_index DataFrame: {e}")

            # Dict-style indexes (for other DB types)
            if not counts and isinstance(index, dict):
                for fid, info in index.items():
                    if isinstance(info, dict):
                        count = info.get("n_examples") or info.get(
                            "total_activations"
                        )
                        count = info.get("count", count)
                    else:
                        count = None
                    if count is not None:
                        counts.append((int(fid), int(count)))

        # Fall back to feature_lookup sizes if present
        if not counts and hasattr(db, "feature_lookup"):
            lookup = getattr(db, "feature_lookup", {})
            counts = [
                (int(fid), int(end - start))
                for fid, (start, end) in lookup.items()
            ]

        # Last resort: uniform priority across all features
        if not counts and hasattr(db, "get_all_feature_ids"):
            counts = [(int(fid), 1) for fid in db.get_all_feature_ids()]

        return counts

    def build_from_cluster(
        self,
        seed_feature: int,
        top_k: int = 10,
    ) -> LabelingQueue:
        """Build queue from features similar to a seed feature.

        Args:
            seed_feature: Feature ID to find similar features for
            top_k: Number of similar features to add

        Returns:
            LabelingQueue with similar features
        """
        queue = LabelingQueue.load(self.model_type)

        # Use similar finder to get related features
        from splatnlp.mechinterp.labeling.similar_finder import SimilarFinder

        finder = SimilarFinder(self.model_type)
        similar = finder.find_by_top_tokens(seed_feature, top_k=top_k)

        for fid, similarity in similar:
            queue.add(
                feature_id=fid,
                priority=similarity,
                reason=f"similar_to_{seed_feature} (sim={similarity:.3f})",
            )

        return queue
