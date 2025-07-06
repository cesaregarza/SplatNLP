"""
Feature analysis module for SAE features.

This module provides tools for analyzing Sparse Autoencoder (SAE) features,
including comprehensive feature analysis, activation pattern exploration,
and LLM prompt generation.
"""

from .analyzer import FeatureAnalyzer
from .config import FEATURE_CATEGORIES, FeatureAnalysisConfig, default_config
from .factory import (
    create_feature_analyzer,
    create_feature_analyzer_from_notebook_config,
    create_quick_analyzer,
)
from .utils import (
    batch_analyze_features,
    generate_comprehensive_llm_prompt,
    save_feature_analysis,
    show_highest_activating_examples,
    test_feature_analyzer,
    test_multiple_features,
)

__all__ = [
    # Main analyzer class
    "FeatureAnalyzer",
    # Utility functions
    "generate_comprehensive_llm_prompt",
    "show_highest_activating_examples",
    "test_feature_analyzer",
    "test_multiple_features",
    "save_feature_analysis",
    "batch_analyze_features",
    # Configuration
    "FeatureAnalysisConfig",
    "default_config",
    "FEATURE_CATEGORIES",
    # Factory functions
    "create_feature_analyzer",
    "create_feature_analyzer_from_notebook_config",
    "create_quick_analyzer",
]
