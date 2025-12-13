"""Research state manager for mechanistic interpretability workflows.

This module provides the ResearchStateManager class which handles
all operations on research state: creating, loading, updating hypotheses,
adding evidence, and maintaining history.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from splatnlp.mechinterp.schemas.experiment_results import ExperimentResult
from splatnlp.mechinterp.schemas.research_state import (
    EvidenceItem,
    EvidenceStrength,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
)
from splatnlp.mechinterp.state.io import (
    create_initial_state,
    get_notes_path,
    get_state_path,
    load_state,
    save_state,
    state_exists,
)

logger = logging.getLogger(__name__)


class ResearchStateManager:
    """Manager for research state operations.

    This class provides a high-level interface for working with research
    state, including hypothesis management, evidence linking, and
    generating summaries.

    Usage:
        manager = ResearchStateManager(feature_id=42, model_type="ultra")
        manager.add_hypothesis("Feature 42 detects SCU presence")
        manager.save()
    """

    def __init__(
        self,
        feature_id: int,
        model_type: Literal["full", "ultra"] = "ultra",
        auto_load: bool = True,
        auto_save: bool = True,
    ):
        """Initialize the state manager.

        Args:
            feature_id: SAE feature ID
            model_type: Model type (full or ultra)
            auto_load: Load existing state if available
            auto_save: Automatically save after modifications
        """
        self.feature_id = feature_id
        self.model_type = model_type
        self.auto_save = auto_save
        self._state: ResearchState | None = None

        if auto_load and state_exists(feature_id, model_type):
            self._state = load_state(feature_id, model_type)
            logger.info(f"Loaded existing state for feature {feature_id}")

    @property
    def state(self) -> ResearchState:
        """Get the current state, creating if needed."""
        if self._state is None:
            self._state = create_initial_state(self.feature_id, self.model_type)
            logger.info(f"Created new state for feature {self.feature_id}")
        return self._state

    def save(self) -> Path:
        """Save the current state to disk."""
        return save_state(self.state)

    def _maybe_save(self) -> None:
        """Save if auto_save is enabled."""
        if self.auto_save:
            self.save()

    # -------------------------------------------------------------------------
    # Hypothesis Management
    # -------------------------------------------------------------------------

    def add_hypothesis(
        self,
        statement: str,
        confidence: float = 0.5,
        tags: list[str] | None = None,
        parent_hypothesis: str | None = None,
    ) -> Hypothesis:
        """Add a new hypothesis.

        Args:
            statement: Clear statement of the hypothesis
            confidence: Initial confidence level (0-1)
            tags: Optional tags for categorization
            parent_hypothesis: ID of hypothesis this refines

        Returns:
            The created Hypothesis
        """
        h_id = self.state.next_hypothesis_id()
        hypothesis = Hypothesis(
            id=h_id,
            statement=statement,
            confidence=confidence,
            tags=tags or [],
            parent_hypothesis=parent_hypothesis,
        )
        self.state.hypotheses.append(hypothesis)
        self.state.add_history(
            action="hypothesis_added",
            details=f"Added hypothesis {h_id}: {statement[:50]}...",
        )
        logger.info(f"Added hypothesis {h_id}")
        self._maybe_save()
        return hypothesis

    def update_hypothesis(
        self,
        h_id: str,
        status: HypothesisStatus | None = None,
        confidence_delta: float | None = None,
        confidence_absolute: float | None = None,
    ) -> Hypothesis | None:
        """Update an existing hypothesis.

        Args:
            h_id: Hypothesis ID to update
            status: New status (if changing)
            confidence_delta: Change to confidence (+/-)
            confidence_absolute: Set confidence to this value

        Returns:
            Updated Hypothesis, or None if not found
        """
        hypothesis = self.state.get_hypothesis(h_id)
        if hypothesis is None:
            logger.warning(f"Hypothesis {h_id} not found")
            return None

        changes = []
        if status is not None:
            hypothesis.status = status
            changes.append(f"status={status.value}")

        if confidence_absolute is not None:
            hypothesis.confidence = max(0.0, min(1.0, confidence_absolute))
            changes.append(f"confidence={hypothesis.confidence:.2f}")
        elif confidence_delta is not None:
            hypothesis.update_confidence(confidence_delta)
            changes.append(f"confidence={hypothesis.confidence:.2f}")

        hypothesis.updated_at = datetime.now()

        self.state.add_history(
            action="hypothesis_updated",
            details=f"Updated {h_id}: {', '.join(changes)}",
        )
        self._maybe_save()
        return hypothesis

    def supersede_hypothesis(
        self,
        old_h_id: str,
        new_statement: str,
        new_confidence: float = 0.5,
    ) -> Hypothesis | None:
        """Mark a hypothesis as superseded and create a refined version.

        Args:
            old_h_id: ID of hypothesis to supersede
            new_statement: Statement for the new hypothesis
            new_confidence: Confidence for new hypothesis

        Returns:
            The new Hypothesis, or None if old not found
        """
        old = self.state.get_hypothesis(old_h_id)
        if old is None:
            logger.warning(f"Hypothesis {old_h_id} not found")
            return None

        # Mark old as superseded
        old.status = HypothesisStatus.SUPERSEDED
        old.updated_at = datetime.now()

        # Create new with reference to old
        new_h = self.add_hypothesis(
            statement=new_statement,
            confidence=new_confidence,
            tags=old.tags.copy(),
            parent_hypothesis=old_h_id,
        )

        self.state.add_history(
            action="hypothesis_superseded",
            details=f"{old_h_id} superseded by {new_h.id}",
        )
        self._maybe_save()
        return new_h

    # -------------------------------------------------------------------------
    # Evidence Management
    # -------------------------------------------------------------------------

    def add_evidence(
        self,
        experiment_id: str,
        result_path: str,
        summary: str,
        strength: EvidenceStrength = EvidenceStrength.MODERATE,
        supports: list[str] | None = None,
        refutes: list[str] | None = None,
        key_metrics: dict[str, float] | None = None,
    ) -> EvidenceItem:
        """Add evidence from an experiment.

        Args:
            experiment_id: ID of the experiment
            result_path: Path to result JSON
            summary: Human-readable summary
            strength: Strength of evidence
            supports: Hypothesis IDs this supports
            refutes: Hypothesis IDs this refutes
            key_metrics: Key numeric metrics

        Returns:
            The created EvidenceItem
        """
        e_id = self.state.next_evidence_id()
        evidence = EvidenceItem(
            id=e_id,
            experiment_id=experiment_id,
            result_path=result_path,
            summary=summary,
            strength=strength,
            supports_hypotheses=supports or [],
            refutes_hypotheses=refutes or [],
            key_metrics=key_metrics or {},
        )
        self.state.evidence_index.append(evidence)

        # Update hypothesis evidence links
        for h_id in supports or []:
            h = self.state.get_hypothesis(h_id)
            if h:
                h.supporting_evidence.append(e_id)

        for h_id in refutes or []:
            h = self.state.get_hypothesis(h_id)
            if h:
                h.refuting_evidence.append(e_id)

        self.state.add_history(
            action="evidence_added",
            details=f"Added evidence {e_id} from {experiment_id}",
            result_path=result_path,
        )
        logger.info(f"Added evidence {e_id}")
        self._maybe_save()
        return evidence

    def add_evidence_from_result(
        self,
        result: ExperimentResult,
        summary: str | None = None,
        strength: EvidenceStrength = EvidenceStrength.MODERATE,
        supports: list[str] | None = None,
        refutes: list[str] | None = None,
    ) -> EvidenceItem:
        """Add evidence directly from an ExperimentResult.

        Args:
            result: The experiment result
            summary: Override auto-generated summary
            strength: Strength of evidence
            supports: Hypothesis IDs this supports
            refutes: Hypothesis IDs this refutes

        Returns:
            The created EvidenceItem
        """
        if summary is None:
            summary = result.get_summary()[:200]

        key_metrics = {}
        if result.aggregates.mean_delta is not None:
            key_metrics["mean_delta"] = result.aggregates.mean_delta
        if result.aggregates.effect_size is not None:
            key_metrics["effect_size"] = result.aggregates.effect_size
        key_metrics.update(result.aggregates.custom)

        return self.add_evidence(
            experiment_id=result.spec_id,
            result_path=result.spec_path,
            summary=summary,
            strength=strength,
            supports=supports,
            refutes=refutes,
            key_metrics=key_metrics,
        )

    # -------------------------------------------------------------------------
    # Notes and Pitfalls
    # -------------------------------------------------------------------------

    def add_pitfall(self, pitfall: str) -> None:
        """Add a known pitfall to avoid.

        Args:
            pitfall: Description of the pitfall
        """
        if pitfall not in self.state.known_pitfalls:
            self.state.known_pitfalls.append(pitfall)
            self.state.add_history(
                action="pitfall_added",
                details=f"Added pitfall: {pitfall[:50]}",
            )
            self._maybe_save()

    def update_notes(self, notes: str, append: bool = False) -> None:
        """Update free-form research notes.

        Args:
            notes: Note content
            append: If True, append to existing notes
        """
        if append:
            self.state.notes = f"{self.state.notes}\n\n{notes}".strip()
        else:
            self.state.notes = notes
        self.state.add_history(
            action="notes_updated",
            details="Updated research notes",
        )
        self._maybe_save()

    # -------------------------------------------------------------------------
    # Summaries and Reports
    # -------------------------------------------------------------------------

    def get_summary(self) -> str:
        """Generate a summary of the current research state."""
        lines = [
            f"# Research State: Feature {self.feature_id}",
            f"Model: {self.model_type}",
            f"Label: {self.state.feature_label or 'unlabeled'}",
            "",
        ]

        # Hypotheses summary
        active = self.state.get_active_hypotheses()
        lines.append(f"## Hypotheses ({len(active)} active)")
        for h in sorted(active, key=lambda x: -x.confidence):
            lines.append(
                f"- [{h.id}] ({h.status.value}, {h.confidence:.0%}) {h.statement}"
            )

        supported = [
            h
            for h in self.state.hypotheses
            if h.status == HypothesisStatus.SUPPORTED
        ]
        if supported:
            lines.append(f"\n### Supported ({len(supported)})")
            for h in supported:
                lines.append(f"- [{h.id}] {h.statement}")

        # Evidence summary
        lines.append(f"\n## Evidence ({len(self.state.evidence_index)} items)")
        for e in self.state.evidence_index[-5:]:  # Last 5
            lines.append(f"- [{e.id}] {e.summary[:80]}...")

        # Pitfalls
        if self.state.known_pitfalls:
            lines.append(f"\n## Known Pitfalls")
            for p in self.state.known_pitfalls:
                lines.append(f"- {p}")

        # Recent history
        lines.append(f"\n## Recent History")
        for entry in self.state.history[-5:]:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M")
            lines.append(f"- [{ts}] {entry.action}: {entry.details}")

        return "\n".join(lines)

    def export_notes(self) -> Path:
        """Export research notes to Markdown file.

        Returns:
            Path to the notes file
        """
        notes_path = get_notes_path(self.feature_id, self.model_type)
        notes_path.parent.mkdir(parents=True, exist_ok=True)

        content = [
            f"# Feature {self.feature_id} Research Notes",
            f"*Model: {self.model_type}*",
            f"*Label: {self.state.feature_label or 'unlabeled'}*",
            f"*Last updated: {self.state.updated_at.strftime('%Y-%m-%d %H:%M')}*",
            "",
        ]

        # Hypotheses
        content.append("## Hypotheses")
        for h in self.state.hypotheses:
            status_emoji = {
                HypothesisStatus.PROPOSED: "?",
                HypothesisStatus.TESTING: "...",
                HypothesisStatus.SUPPORTED: "Y",
                HypothesisStatus.REFUTED: "X",
                HypothesisStatus.SUPERSEDED: "~",
            }.get(h.status, "?")
            content.append(
                f"- [{status_emoji}] **{h.id}** ({h.confidence:.0%}): {h.statement}"
            )

        # Evidence
        content.append("\n## Evidence")
        for e in self.state.evidence_index:
            strength_stars = {"strong": "***", "moderate": "**", "weak": "*"}
            stars = strength_stars.get(e.strength.value, "")
            content.append(f"- {stars}[{e.id}]{stars}: {e.summary}")
            if e.key_metrics:
                metrics_str = ", ".join(
                    f"{k}={v:.3f}" for k, v in e.key_metrics.items()
                )
                content.append(f"  - Metrics: {metrics_str}")

        # Free-form notes
        if self.state.notes:
            content.append("\n## Notes")
            content.append(self.state.notes)

        notes_path.write_text("\n".join(content))
        logger.info(f"Exported notes to {notes_path}")
        return notes_path

    def get_next_experiment_suggestions(self) -> list[str]:
        """Get suggestions for next experiments based on state.

        Returns:
            List of suggested experiment descriptions
        """
        suggestions = []

        # If no hypotheses, suggest exploration
        if not self.state.hypotheses:
            suggestions.append(
                "No hypotheses yet - run PageRank and itemset analysis to explore"
            )
            return suggestions

        # Check for untested hypotheses
        untested = [
            h
            for h in self.state.hypotheses
            if h.status == HypothesisStatus.PROPOSED
            and not h.supporting_evidence
        ]
        for h in untested[:2]:
            suggestions.append(f"Test hypothesis {h.id}: {h.statement[:50]}")

        # Check for hypotheses needing validation
        high_conf = [
            h
            for h in self.state.hypotheses
            if h.status
            in (HypothesisStatus.SUPPORTED, HypothesisStatus.TESTING)
            and h.confidence > 0.7
        ]
        for h in high_conf[:1]:
            suggestions.append(
                f"Validate {h.id} with split-half or shuffle null"
            )

        # Check for evidence gaps
        if len(self.state.evidence_index) < 3:
            suggestions.append(
                "Limited evidence - run family sweeps or interaction analysis"
            )

        return suggestions
