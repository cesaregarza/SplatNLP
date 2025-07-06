# Feature Analysis Module

This module provides comprehensive tools for analyzing Sparse Autoencoder (SAE) features from your SplatNLP model. It includes functionality for feature interpretation, activation analysis, and LLM prompt generation.

## Features

- **Feature Analyzer**: Comprehensive analysis of SAE features including output influences, activation patterns, and TF-IDF analysis
- **Factory Functions**: Easy creation of feature analyzers with sensible defaults
- **Utility Functions**: Tools for displaying examples, generating prompts, and batch processing
- **Configuration Management**: Flexible configuration system for different setups
- **LLM Integration**: Generate structured prompts for automated feature interpretation

## Quick Start

### Basic Usage

```python
from splatnlp.feature_analysis import create_quick_analyzer, show_highest_activating_examples

# Create analyzer with default paths
analyzer = create_quick_analyzer()

# Show highest activating examples for a feature
show_highest_activating_examples(analyzer, feature_id=1257, limit=10)

# Get comprehensive analysis
analysis = analyzer.analyze_feature_comprehensively(1257)
print(analysis['interpretation'])
```

### Notebook-Style Setup

If you're migrating from the notebook, use the notebook-style setup:

```python
from splatnlp.feature_analysis import create_feature_analyzer_from_notebook_config

analyzer = create_feature_analyzer_from_notebook_config(
    primary_model_checkpoint="saved_models/dataset_v0_2_full/model.pth",
    sae_model_checkpoint="saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_model_final.pth",
    sae_config_path="saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_run_config.json",
    vocab_path="saved_models/dataset_v0_2_full/vocab.json",
    weapon_vocab_path="saved_models/dataset_v0_2_full/weapon_vocab.json",
    meta_path="/mnt/e/activations2/outputs/",
    neurons_root="/mnt/e/activations2/outputs/neuron_acts",
    device="cuda"
)
```

## Main Components

### FeatureAnalyzer

The main class that provides comprehensive feature analysis:

```python
# Get basic feature information
info = analyzer.get_feature_info(feature_id)

# Compute output influences (what tokens this feature promotes/suppresses)
influences = analyzer.compute_output_influences(feature_id)

# Get activation buckets (examples at different activation levels)
buckets = analyzer.get_activation_buckets(feature_id)

# Compute TF-IDF scores for characteristic tokens
tfidf = analyzer.compute_feature_tfidf(feature_id)

# Full comprehensive analysis
analysis = analyzer.analyze_feature_comprehensively(feature_id)
```

### Utility Functions

#### Display Functions

```python
from splatnlp.feature_analysis import show_highest_activating_examples

# Show highest activating examples
show_highest_activating_examples(analyzer, feature_id, limit=15)
```

#### LLM Prompt Generation

```python
from splatnlp.feature_analysis import generate_comprehensive_llm_prompt, save_feature_analysis

# Generate structured prompt for LLM analysis
prompt = generate_comprehensive_llm_prompt(analyzer, feature_id)

# Save analysis to file
save_feature_analysis(analyzer, feature_id, "feature_analysis.txt")
```

#### Batch Processing

```python
from splatnlp.feature_analysis import batch_analyze_features

# Analyze multiple features at once
feature_ids = [1257, 959, 2021, 845, 1820]
output_files = batch_analyze_features(analyzer, feature_ids, output_dir="./outputs")
```

### Configuration

Use the configuration system for custom setups:

```python
from splatnlp.feature_analysis import FeatureAnalysisConfig, create_feature_analyzer

# Create custom configuration
config = FeatureAnalysisConfig(
    model_paths={
        "primary_model": "path/to/model.pth",
        "sae_model": "path/to/sae_model.pth",
        # ... other paths
    },
    data_paths={
        "meta_path": "/path/to/meta",
        "neurons_root": "/path/to/neurons",
    },
    device="cuda"
)

# Create analyzer with custom config
analyzer = create_feature_analyzer(config)
```

## Analysis Output

The comprehensive analysis provides:

1. **Feature Information**: Name, category, human labels, statistics
2. **Output Influences**: Which tokens the feature promotes/suppresses
3. **Activation Buckets**: Examples at different activation levels
4. **TF-IDF Analysis**: Most characteristic tokens for the feature
5. **Interpretation**: Human-readable summary of the feature's behavior

## Example Analysis Output

```
Feature 1257: Stamper Balanced | Category: tactical | Sparsity: 68.37%, Max: 3.605 | 
Promotes: sub_power_up_6, stealth_jump, comeback | 
Suppresses: special_saver_21, special_charge_up_12, special_charge_up_15 | 
Key tokens (TF-IDF): ink_saver_main_3, ink_resistance_up_3, sub_power_up_6 | 
Top activation buckets: 5 buckets with 4438273 total examples
```

## Testing

Run the included tests to verify functionality:

```python
from splatnlp.feature_analysis import test_feature_analyzer, test_multiple_features

# Test basic functionality
test_feature_analyzer(analyzer, feature_id=1257)

# Test multiple features
test_multiple_features(analyzer, [1257, 959, 2021])
```

## Requirements

- PyTorch
- NumPy
- Pandas
- Access to trained SplatNLP models and SAE
- Optional: Dashboard components for enhanced functionality

## Migration from Notebook

If you're migrating from the original notebook code:

1. Replace the notebook's `FeatureAnalyzer` creation with `create_quick_analyzer()` or `create_feature_analyzer_from_notebook_config()`
2. Replace function calls like `show_highest_activating_examples(feature_id)` with `show_highest_activating_examples(analyzer, feature_id)`
3. Use the new batch processing functions for analyzing multiple features

## Advanced Usage

### Custom Analysis

```python
# Get raw activation data
if analyzer.db:
    activations_df = analyzer.db.get_feature_activations(feature_id, limit=1000)
    
# Compute custom metrics
influences = analyzer.compute_output_influences(feature_id, limit=20)
positive_influences = [item for item in influences['positive'] if item['influence_value'] > 0.1]
```

### Extending the Analyzer

```python
class CustomFeatureAnalyzer(FeatureAnalyzer):
    def custom_analysis(self, feature_id):
        # Add your custom analysis logic here
        pass
```

## Files Structure

```
src/splatnlp/feature_analysis/
├── __init__.py          # Main module exports
├── analyzer.py          # FeatureAnalyzer class
├── utils.py             # Utility functions
├── config.py            # Configuration management
├── factory.py           # Factory functions for creating analyzers
├── example.py           # Example usage script
└── README.md            # This file
```

## Support

For issues or questions, please check the example script (`example.py`) or refer to the original notebook for usage patterns. 