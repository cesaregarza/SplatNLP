"""Research state schemas for mechanistic interpretability workflows.

This module defines the core data structures for tracking research progress
on SAE features, including hypotheses, evidence, and session history.
"""

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class HypothesisStatus(str, Enum):
    """Status of a research hypothesis."""

    PROPOSED = "proposed"  # Initial hypothesis, not yet tested
    TESTING = "testing"  # Currently being tested
    SUPPORTED = "supported"  # Evidence supports the hypothesis
    REFUTED = "refuted"  # Evidence refutes the hypothesis
    SUPERSEDED = "superseded"  # Replaced by a more refined hypothesis


class EvidenceStrength(str, Enum):
    """Strength of evidence from an experiment."""

    STRONG = "strong"  # Clear, reproducible effect
    MODERATE = "moderate"  # Consistent but noisy effect
    WEAK = "weak"  # Marginal or inconsistent effect


class Hypothesis(BaseModel):
    """A research hypothesis about a feature's behavior.

    Hypotheses are the core unit of research progress. They start as
    proposals and are updated based on experimental evidence.
    """

    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'h001')",
        pattern=r"^h\d{3,}$",
    )
    statement: str = Field(
        ...,
        description="Clear statement of the hypothesis",
        min_length=10,
    )
    status: HypothesisStatus = Field(
        default=HypothesisStatus.PROPOSED,
        description="Current status of the hypothesis",
    )
    confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0-1.0)",
    )
    supporting_evidence: list[str] = Field(
        default_factory=list,
        description="List of evidence IDs that support this hypothesis",
    )
    refuting_evidence: list[str] = Field(
        default_factory=list,
        description="List of evidence IDs that refute this hypothesis",
    )
    parent_hypothesis: str | None = Field(
        default=None,
        description="ID of hypothesis this refines/supersedes",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="Tags for categorization (e.g., 'family-specific')",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def update_confidence(self, delta: float) -> None:
        """Adjust confidence by delta, clamping to [0, 1]."""
        self.confidence = max(0.0, min(1.0, self.confidence + delta))
        self.updated_at = datetime.now()


class EvidenceItem(BaseModel):
    """Evidence from an experiment that relates to hypotheses.

    Evidence links experimental results to hypotheses, providing the
    basis for confidence updates and status changes.
    """

    id: str = Field(
        ...,
        description="Unique identifier (e.g., 'e001')",
        pattern=r"^e\d{3,}$",
    )
    experiment_id: str = Field(
        ...,
        description="ID of the experiment that produced this evidence",
    )
    result_path: str = Field(
        ...,
        description="Path to the experiment result JSON",
    )
    summary: str = Field(
        ...,
        description="Human-readable summary of the evidence",
        min_length=10,
    )
    strength: EvidenceStrength = Field(
        default=EvidenceStrength.MODERATE,
        description="Strength of the evidence",
    )
    supports_hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypothesis IDs this evidence supports",
    )
    refutes_hypotheses: list[str] = Field(
        default_factory=list,
        description="Hypothesis IDs this evidence refutes",
    )
    key_metrics: dict[str, float] = Field(
        default_factory=dict,
        description="Key numeric metrics from the experiment",
    )
    timestamp: datetime = Field(default_factory=datetime.now)


class HistoryEntry(BaseModel):
    """A single entry in the research history log."""

    timestamp: datetime = Field(default_factory=datetime.now)
    action: str = Field(
        ...,
        description="Action taken (e.g., 'hypothesis_added', 'experiment_run')",
    )
    details: str = Field(
        ...,
        description="Human-readable description of the action",
    )
    spec_path: str | None = Field(
        default=None,
        description="Path to experiment spec if applicable",
    )
    result_path: str | None = Field(
        default=None,
        description="Path to experiment result if applicable",
    )


class ResearchState(BaseModel):
    """Complete research state for a feature.

    This is the root object for tracking all research progress on a
    single SAE feature. It contains hypotheses, evidence, constraints,
    and a full history of research actions.
    """

    feature_id: int = Field(..., description="SAE feature ID")
    model_type: Literal["full", "ultra"] = Field(
        ...,
        description="Model type (full=2K features, ultra=24K features)",
    )
    feature_label: str | None = Field(
        default=None,
        description="Human-assigned label for the feature",
    )
    hypotheses: list[Hypothesis] = Field(
        default_factory=list,
        description="All hypotheses about this feature",
    )
    evidence_index: list[EvidenceItem] = Field(
        default_factory=list,
        description="All evidence collected",
    )
    active_constraints: list[str] = Field(
        default_factory=lambda: ["one_rung_per_family"],
        description="IDs of active domain constraints",
    )
    known_pitfalls: list[str] = Field(
        default_factory=list,
        description="Known issues to avoid (e.g., 'relu_floor_at_low_activation')",
    )
    history: list[HistoryEntry] = Field(
        default_factory=list,
        description="Chronological log of research actions",
    )
    notes: str = Field(
        default="",
        description="Free-form research notes",
    )
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def get_hypothesis(self, h_id: str) -> Hypothesis | None:
        """Get a hypothesis by ID."""
        for h in self.hypotheses:
            if h.id == h_id:
                return h
        return None

    def get_evidence(self, e_id: str) -> EvidenceItem | None:
        """Get an evidence item by ID."""
        for e in self.evidence_index:
            if e.id == e_id:
                return e
        return None

    def next_hypothesis_id(self) -> str:
        """Generate the next hypothesis ID."""
        if not self.hypotheses:
            return "h001"
        max_id = max(int(h.id[1:]) for h in self.hypotheses)
        return f"h{max_id + 1:03d}"

    def next_evidence_id(self) -> str:
        """Generate the next evidence ID."""
        if not self.evidence_index:
            return "e001"
        max_id = max(int(e.id[1:]) for e in self.evidence_index)
        return f"e{max_id + 1:03d}"

    def add_history(
        self,
        action: str,
        details: str,
        spec_path: str | None = None,
        result_path: str | None = None,
    ) -> None:
        """Add an entry to the history log."""
        self.history.append(
            HistoryEntry(
                action=action,
                details=details,
                spec_path=spec_path,
                result_path=result_path,
            )
        )
        self.updated_at = datetime.now()

    def get_active_hypotheses(self) -> list[Hypothesis]:
        """Get hypotheses that are proposed or testing."""
        return [
            h
            for h in self.hypotheses
            if h.status in (HypothesisStatus.PROPOSED, HypothesisStatus.TESTING)
        ]

    def get_top_hypothesis(self) -> Hypothesis | None:
        """Get the highest-confidence active hypothesis."""
        active = self.get_active_hypotheses()
        if not active:
            return None
        return max(active, key=lambda h: h.confidence)
