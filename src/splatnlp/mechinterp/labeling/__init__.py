"""Labeling workflow components for mechanistic interpretability.

This submodule provides tools for interactive feature labeling:
- FeatureOverview: Quick "first look" at a feature
- LabelingQueue: Priority queue for features to label
- LabelConsolidator: Sync labels between ResearchState and dashboard
- SimilarFinder: Find features similar to a given one
"""

from splatnlp.mechinterp.labeling.label_consolidator import (
    ConsolidatedLabel,
    LabelConsolidator,
)
from splatnlp.mechinterp.labeling.overview import (
    FeatureOverview,
    SampleContext,
    compute_overview,
)
from splatnlp.mechinterp.labeling.priority_queue import (
    LabelingQueue,
    QueueBuilder,
    QueueEntry,
)
from splatnlp.mechinterp.labeling.similar_finder import SimilarFinder

__all__ = [
    # Overview
    "FeatureOverview",
    "SampleContext",
    "compute_overview",
    # Label consolidation
    "ConsolidatedLabel",
    "LabelConsolidator",
    # Queue management
    "LabelingQueue",
    "QueueBuilder",
    "QueueEntry",
    # Similarity
    "SimilarFinder",
]
