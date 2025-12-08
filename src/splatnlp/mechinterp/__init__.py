"""
MechInterp: Mechanistic Interpretability Workflow Module

This module provides tools for iterative mechanistic interpretability
research on SAE features, including:
- Research state management (hypotheses, evidence, history)
- Experiment specification and execution
- Analysis utilities for feature interpretation
"""

from splatnlp.mechinterp.schemas.experiment_results import (
    Aggregates,
    DiagnosticInfo,
    ExperimentResult,
)
from splatnlp.mechinterp.schemas.experiment_specs import (
    DatasetSlice,
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.schemas.glossary import (
    DOMAIN_CONSTRAINTS,
    STANDARD_RUNGS,
    TOKEN_FAMILIES,
    DomainConstraint,
    TokenFamily,
)
from splatnlp.mechinterp.schemas.research_state import (
    EvidenceItem,
    EvidenceStrength,
    Hypothesis,
    HypothesisStatus,
    ResearchState,
)

__all__ = [
    # Research State
    "ResearchState",
    "Hypothesis",
    "HypothesisStatus",
    "EvidenceItem",
    "EvidenceStrength",
    # Experiment Specs
    "ExperimentSpec",
    "ExperimentType",
    "DatasetSlice",
    # Experiment Results
    "ExperimentResult",
    "Aggregates",
    "DiagnosticInfo",
    # Glossary
    "TokenFamily",
    "DomainConstraint",
    "TOKEN_FAMILIES",
    "DOMAIN_CONSTRAINTS",
    "STANDARD_RUNGS",
]
