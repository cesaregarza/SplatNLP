"""
Feature analysis module for SAE features.

This module provides tools for analyzing Sparse Autoencoder (SAE) features,
including comprehensive feature analysis, activation pattern exploration,
and LLM prompt generation.
"""

from splatnlp.feature_analysis.analyzer import FeatureAnalyzer
from splatnlp.feature_analysis.config import (
    FeatureAnalysisConfig,
    default_config,
)
from splatnlp.feature_analysis.defaults import (
    FEATURE_CATEGORIES,
    HIGH_AP_PATTERN,
    SPECIAL_TOKENS,
)
from splatnlp.feature_analysis.factory import (
    create_feature_analyzer,
    create_feature_analyzer_from_notebook_config,
    create_quick_analyzer,
)
from splatnlp.feature_analysis.utils import (
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
    "HIGH_AP_PATTERN",
    "SPECIAL_TOKENS",
    # Factory functions
    "create_feature_analyzer",
    "create_feature_analyzer_from_notebook_config",
    "create_quick_analyzer",
]
