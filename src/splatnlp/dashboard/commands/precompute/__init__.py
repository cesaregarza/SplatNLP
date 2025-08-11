"""
Precomputation utilities for SplatNLP dashboard.

This module contains various precomputation scripts for optimizing dashboard performance:
- unified: Main precomputation orchestrator for all tasks
- influences: Compute feature influences on output tokens
- histograms_parallel: Parallel histogram computation for large feature sets
- histograms_fast: Fast single-pass histogram computation
- idf: Inverse Document Frequency computation
- transpose: Transpose activation data for feature-wise access
"""

from splatnlp.dashboard.commands.precompute.histograms_fast import (
    FastHistogramComputer,
)
from splatnlp.dashboard.commands.precompute.histograms_fast import (
    main as histograms_fast_main,
)
from splatnlp.dashboard.commands.precompute.histograms_parallel import (
    compute_histograms_parallel,
    compute_histograms_vectorized,
)
from splatnlp.dashboard.commands.precompute.histograms_parallel import (
    main as histograms_parallel_main,
)
from splatnlp.dashboard.commands.precompute.idf import compute_idf_from_batches
from splatnlp.dashboard.commands.precompute.idf import main as idf_main
from splatnlp.dashboard.commands.precompute.influences import (
    main as influences_main,
)
from splatnlp.dashboard.commands.precompute.transpose import (
    main as transpose_main,
)
from splatnlp.dashboard.commands.precompute.transpose import (
    transpose_activations,
)
from splatnlp.dashboard.commands.precompute.unified import PrecomputeManager
from splatnlp.dashboard.commands.precompute.unified import main as unified_main

__all__ = [
    # Main orchestrator
    "PrecomputeManager",
    "unified_main",
    # Individual components
    "influences_main",
    "compute_histograms_parallel",
    "compute_histograms_vectorized",
    "histograms_parallel_main",
    "FastHistogramComputer",
    "histograms_fast_main",
    "compute_idf_from_batches",
    "idf_main",
    "transpose_activations",
    "transpose_main",
]
