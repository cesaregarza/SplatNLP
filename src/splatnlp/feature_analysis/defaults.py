"""
Default configurations and constants for feature analysis.

This module contains all default values, paths, and constants
used throughout the feature analysis package.
"""

# Default paths
DEFAULT_FEATURE_LABELS_PATH = "src/splatnlp/dashboard/feature_labels.json"
DEFAULT_VOCAB_PATH = "saved_models/dataset_v0_2_full/vocab.json"
DEFAULT_WEAPON_VOCAB_PATH = "saved_models/dataset_v0_2_full/weapon_vocab.json"

# Default model paths
DEFAULT_MODEL_PATHS = {
    "primary_model": "saved_models/dataset_v0_2_full/model.pth",
    "sae_model": "saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_model_final.pth",
    "sae_config": "saved_models/dataset_v0_2_full/sae_runs/run_20250429_023422/sae_run_config.json",
    "vocab": "saved_models/dataset_v0_2_full/vocab.json",
    "weapon_vocab": "saved_models/dataset_v0_2_full/weapon_vocab.json",
}

# Default data paths
DEFAULT_DATA_PATHS = {
    "tokenized_data": "/root/dev/SplatNLP/test_data/tokenized/tokenized_data.csv",
    "meta_path": "/mnt/e/activations2/outputs/",
    "neurons_root": "/mnt/e/activations2/outputs/neuron_acts",
}

# Default model parameters
DEFAULT_PRIMARY_MODEL_PARAMS = {
    "embedding_dim": 32,
    "hidden_dim": 512,
    "num_layers": 3,
    "num_heads": 8,
    "num_inducing": 32,
    "use_layer_norm": True,
    "dropout": 0.0,  # Set to 0 for eval
}

# Default SAE parameters
DEFAULT_SAE_PARAMS = {
    "input_dim": 512,  # Should match primary model's hidden_dim
    "expansion_factor": 4.0,
}

# Analysis parameters
DEFAULT_ANALYSIS_PARAMS = {
    "num_examples_to_inspect": 50000,
    "n_top_examples_per_feature": 10,
    "num_activation_buckets": 5,
    "tfidf_top_k": 20,
    "output_influences_limit": 10,
}

# Device configuration
DEFAULT_DEVICE = "cuda"

# Special tokens
SPECIAL_TOKENS = {"<PAD>", "<NULL>"}

# High AP pattern (for regex matching)
HIGH_AP_PATTERN = r"_(21|29|38|51|57)$"

# Feature categories
FEATURE_CATEGORIES = {
    "tactical": "Gameplay strategies and build patterns",
    "mechanical": "Specific ability patterns and relationships",
    "strategic": "High-level build types and substitutions",
    "unknown": "Uncategorized features",
}
