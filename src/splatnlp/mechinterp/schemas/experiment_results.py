"""Experiment result schemas for mechanistic interpretability.

This module defines the data structures for storing experiment results,
including aggregates, tables, diagnostics, and figure references.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class Aggregates(BaseModel):
    """Summary statistics from an experiment.

    These are the high-level numeric results that can be compared across
    experiments or used to update hypothesis confidence.
    """

    # Common statistics
    mean_delta: float | None = Field(
        default=None,
        description="Mean activation change across conditions",
    )
    std_delta: float | None = Field(
        default=None,
        description="Standard deviation of activation change",
    )
    max_delta: float | None = Field(
        default=None,
        description="Maximum activation change observed",
    )
    min_delta: float | None = Field(
        default=None,
        description="Minimum activation change observed",
    )
    median_delta: float | None = Field(
        default=None,
        description="Median activation change",
    )

    # Base activation stats
    base_activation_mean: float | None = Field(
        default=None,
        description="Mean base activation across contexts",
    )
    base_activation_std: float | None = Field(
        default=None,
        description="Std of base activation across contexts",
    )

    # Sample counts
    n_samples: int = Field(
        default=0,
        description="Number of samples/contexts analyzed",
    )
    n_conditions: int = Field(
        default=0,
        description="Number of experimental conditions",
    )

    # Effect sizes
    effect_size: float | None = Field(
        default=None,
        description="Standardized effect size (Cohen's d or similar)",
    )
    explained_variance: float | None = Field(
        default=None,
        description="Variance explained by the manipulation",
    )

    # Experiment-specific metrics
    custom: dict[str, float] = Field(
        default_factory=dict,
        description="Experiment-type-specific metrics",
    )


class DiagnosticInfo(BaseModel):
    """Experiment health and validity diagnostics.

    These flags and metrics help identify potential issues with
    experiment execution or interpretation.
    """

    # ReLU floor detection
    relu_floor_detected: bool = Field(
        default=False,
        description="Whether base activations were near ReLU floor",
    )
    relu_floor_rate: float = Field(
        default=0.0,
        description="Fraction of contexts at/near ReLU floor",
    )

    # Activation ranges
    base_activation_range: tuple[float, float] = Field(
        default=(0.0, 0.0),
        description="(min, max) base activation across contexts",
    )

    # Context statistics
    n_contexts_tested: int = Field(
        default=0,
        description="Number of contexts successfully tested",
    )
    n_contexts_skipped: int = Field(
        default=0,
        description="Number of contexts skipped (invalid, missing data)",
    )
    n_contexts_errored: int = Field(
        default=0,
        description="Number of contexts that errored during computation",
    )

    # Constraint violations
    constraint_violations: list[str] = Field(
        default_factory=list,
        description="Constraint violations detected (e.g., multi-rung conflicts)",
    )

    # Performance
    cache_hit_rate: float = Field(
        default=0.0,
        description="Fraction of computations served from cache",
    )
    compute_time_seconds: float = Field(
        default=0.0,
        description="Time spent in computation (excluding I/O)",
    )

    # Warnings
    warnings: list[str] = Field(
        default_factory=list,
        description="Non-fatal warnings during execution",
    )


class TableRow(BaseModel):
    """A single row in a result table."""

    values: dict[str, Any] = Field(
        default_factory=dict,
        description="Column name -> value mapping",
    )


class ResultTable(BaseModel):
    """A named table of results.

    Tables are used for detailed results like token sweeps, itemsets,
    or interaction matrices.
    """

    name: str = Field(..., description="Table name (e.g., 'token_deltas')")
    columns: list[str] = Field(
        default_factory=list,
        description="Column names in display order",
    )
    rows: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Row data as list of dicts",
    )
    description: str | None = Field(
        default=None,
        description="Description of what this table contains",
    )

    def to_markdown(self, max_rows: int = 20) -> str:
        """Convert table to Markdown format."""
        if not self.rows:
            return f"*{self.name}: No data*"

        lines = [f"### {self.name}"]
        if self.description:
            lines.append(f"*{self.description}*\n")

        # Header
        cols = self.columns or list(self.rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

        # Rows
        for row in self.rows[:max_rows]:
            values = [str(row.get(c, "")) for c in cols]
            lines.append("| " + " | ".join(values) + " |")

        if len(self.rows) > max_rows:
            lines.append(f"*... and {len(self.rows) - max_rows} more rows*")

        return "\n".join(lines)


class ExperimentResult(BaseModel):
    """Complete result of an experiment execution.

    This is the primary output of mechinterp-runner, containing all
    data needed to interpret the experiment and update research state.
    """

    # Identification
    spec_id: str = Field(
        ...,
        description="ID of the experiment spec that produced this",
    )
    spec_path: str = Field(
        ...,
        description="Path to the experiment spec JSON",
    )
    feature_id: int = Field(..., description="Feature that was analyzed")
    experiment_type: str = Field(..., description="Type of experiment run")

    # Core results
    aggregates: Aggregates = Field(
        default_factory=Aggregates,
        description="Summary statistics",
    )
    tables: dict[str, ResultTable] = Field(
        default_factory=dict,
        description="Named result tables",
    )
    diagnostics: DiagnosticInfo = Field(
        default_factory=DiagnosticInfo,
        description="Execution diagnostics",
    )

    # Figure references
    figures: list[str] = Field(
        default_factory=list,
        description="Paths to generated figure files",
    )

    # Execution metadata
    started_at: datetime = Field(
        default_factory=datetime.now,
        description="When execution started",
    )
    completed_at: datetime = Field(
        default_factory=datetime.now,
        description="When execution completed",
    )
    duration_seconds: float = Field(
        default=0.0,
        description="Total execution time",
    )

    # Status
    success: bool = Field(
        default=True,
        description="Whether execution completed successfully",
    )
    error_message: str | None = Field(
        default=None,
        description="Error message if execution failed",
    )

    def to_filename(self) -> str:
        """Generate a descriptive filename for this result."""
        return f"{self.spec_id}__result.json"

    def get_summary(self) -> str:
        """Generate a human-readable summary of results."""
        lines = [
            f"## Experiment Result: {self.experiment_type}",
            f"Feature: {self.feature_id}",
            f"Status: {'Success' if self.success else 'FAILED'}",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
        ]

        if not self.success:
            lines.append(f"**Error:** {self.error_message}")
            return "\n".join(lines)

        # Aggregates
        lines.append("### Key Metrics")
        if self.aggregates.mean_delta is not None:
            lines.append(f"- Mean delta: {self.aggregates.mean_delta:.4f}")
        if self.aggregates.std_delta is not None:
            lines.append(f"- Std delta: {self.aggregates.std_delta:.4f}")
        if self.aggregates.n_samples > 0:
            lines.append(f"- Samples: {self.aggregates.n_samples}")

        for key, val in self.aggregates.custom.items():
            if isinstance(val, (int, float)):
                lines.append(f"- {key}: {val:.4f}")
            elif isinstance(val, dict):
                lines.append(f"- {key}: {val}")
            else:
                lines.append(f"- {key}: {val}")

        # Diagnostics
        if self.diagnostics.warnings:
            lines.append("\n### Warnings")
            for w in self.diagnostics.warnings:
                lines.append(f"- {w}")

        if self.diagnostics.relu_floor_detected:
            lines.append(
                f"\n**ReLU floor detected** "
                f"({self.diagnostics.relu_floor_rate:.1%} of contexts)"
            )

        return "\n".join(lines)

    def add_table(
        self,
        name: str,
        rows: list[dict[str, Any]],
        columns: list[str] | None = None,
        description: str | None = None,
    ) -> None:
        """Add a result table."""
        if columns is None and rows:
            columns = list(rows[0].keys())
        self.tables[name] = ResultTable(
            name=name,
            columns=columns or [],
            rows=rows,
            description=description,
        )
