#!/usr/bin/env python3
"""
Example script demonstrating how to use the feature analysis module.

This script shows how to:
1. Create a feature analyzer using the factory functions
2. Analyze individual features
3. Generate LLM prompts for features
4. Batch analyze multiple features
"""

import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Import the feature analysis module
from splatnlp.feature_analysis import (
    FeatureAnalysisConfig,
    batch_analyze_features,
    create_feature_analyzer_from_notebook_config,
    create_quick_analyzer,
    generate_comprehensive_llm_prompt,
    save_feature_analysis,
    show_highest_activating_examples,
    test_feature_analyzer,
)


def example_quick_setup():
    """Example 1: Quick setup with default paths."""

    print("=" * 60)
    print("Example 1: Quick Setup")
    print("=" * 60)

    try:
        # Create analyzer with default paths
        analyzer = create_quick_analyzer(
            sae_run_path="run_20250429_023422",
            model_base_path="saved_models/dataset_v0_2_full",
            meta_path="/mnt/e/activations2/outputs/",
            neurons_root="/mnt/e/activations2/outputs/neuron_acts",
            device="cuda",
        )

        print("✅ Feature analyzer created successfully!")
        print(
            f"SAE dimensions: {analyzer.sae_input_dim} → {analyzer.sae_hidden_dim}"
        )
        print(f"Vocabulary size: {len(analyzer.vocab)}")
        print(f"Weapon vocabulary size: {len(analyzer.weapon_vocab)}")

        return analyzer

    except Exception as e:
        logger.error(f"Failed to create analyzer: {e}")
        return None


def example_notebook_style_setup():
    """Example 2: Notebook-style setup (like the original notebook)."""

    print("\n" + "=" * 60)
    print("Example 2: Notebook-Style Setup")
    print("=" * 60)

    try:
        # Paths from the notebook
        PRIMARY_MODEL_CHECKPOINT = "saved_models/dataset_v0_2_full/model.pth"
        SAE_MODEL_PATH = "run_20250429_023422"
        SAE_MODEL_CHECKPOINT = f"saved_models/dataset_v0_2_full/sae_runs/{SAE_MODEL_PATH}/sae_model_final.pth"
        SAE_MODEL_CONFIGS = f"saved_models/dataset_v0_2_full/sae_runs/{SAE_MODEL_PATH}/sae_run_config.json"
        VOCAB_PATH = "saved_models/dataset_v0_2_full/vocab.json"
        WEAPON_VOCAB_PATH = "saved_models/dataset_v0_2_full/weapon_vocab.json"
        META_PATH = "/mnt/e/activations2/outputs/"
        NEURONS_ROOT = "/mnt/e/activations2/outputs/neuron_acts"

        # Create analyzer
        analyzer = create_feature_analyzer_from_notebook_config(
            primary_model_checkpoint=PRIMARY_MODEL_CHECKPOINT,
            sae_model_checkpoint=SAE_MODEL_CHECKPOINT,
            sae_config_path=SAE_MODEL_CONFIGS,
            vocab_path=VOCAB_PATH,
            weapon_vocab_path=WEAPON_VOCAB_PATH,
            meta_path=META_PATH,
            neurons_root=NEURONS_ROOT,
            device="cuda",
        )

        print("✅ Feature analyzer created successfully!")

        return analyzer

    except Exception as e:
        logger.error(f"Failed to create analyzer: {e}")
        return None


def example_feature_analysis(analyzer):
    """Example 3: Analyzing individual features."""

    if analyzer is None:
        print("❌ No analyzer available for feature analysis")
        return

    print("\n" + "=" * 60)
    print("Example 3: Feature Analysis")
    print("=" * 60)

    # Test features from the notebook
    test_features = [1257, 959, 2021]

    for feature_id in test_features:
        try:
            print(f"\n--- Analyzing Feature {feature_id} ---")

            # Get basic info
            info = analyzer.get_feature_info(feature_id)
            print(f"Name: {info['name']}")
            print(f"Category: {info['category']}")
            print(f"Has Human Label: {info['has_human_label']}")

            # Get output influences
            influences = analyzer.compute_output_influences(feature_id, limit=3)
            if influences["positive"]:
                print(
                    f"Top promoted: {influences['positive'][0]['token_name']}"
                )
            if influences["negative"]:
                print(
                    f"Top suppressed: {influences['negative'][0]['token_name']}"
                )

            # Show highest activating examples
            print(f"\nHighest activating examples for Feature {feature_id}:")
            show_highest_activating_examples(analyzer, feature_id, limit=5)

        except Exception as e:
            logger.error(f"Error analyzing feature {feature_id}: {e}")


def example_llm_prompt_generation(analyzer):
    """Example 4: Generate LLM prompts for features."""

    if analyzer is None:
        print("❌ No analyzer available for LLM prompt generation")
        return

    print("\n" + "=" * 60)
    print("Example 4: LLM Prompt Generation")
    print("=" * 60)

    feature_id = 1257  # "Stamper Balanced"

    try:
        # Generate comprehensive LLM prompt
        prompt = generate_comprehensive_llm_prompt(analyzer, feature_id)

        print(f"Generated prompt for Feature {feature_id}:")
        print(f"Length: {len(prompt)} characters")
        print(f"Preview (first 300 chars):")
        print("-" * 40)
        print(prompt[:300] + "...")

        # Save to file
        output_file = save_feature_analysis(
            analyzer, feature_id, f"example_feature_{feature_id}.txt"
        )
        print(f"\n✅ Full analysis saved to: {output_file}")

    except Exception as e:
        logger.error(f"Error generating LLM prompt: {e}")


def example_batch_analysis(analyzer):
    """Example 5: Batch analysis of multiple features."""

    if analyzer is None:
        print("❌ No analyzer available for batch analysis")
        return

    print("\n" + "=" * 60)
    print("Example 5: Batch Analysis")
    print("=" * 60)

    # Features to analyze
    feature_ids = [1257, 959, 2021]

    try:
        # Batch analyze and save to files
        output_files = batch_analyze_features(
            analyzer, feature_ids, output_dir="./example_outputs"
        )

        print(f"\n✅ Batch analysis completed!")
        print(f"Generated {len(output_files)} files")

    except Exception as e:
        logger.error(f"Error in batch analysis: {e}")


def main():
    """Main function to run all examples."""

    print("🚀 Feature Analysis Module Examples")
    print("=" * 60)

    # Try to create analyzer
    analyzer = example_quick_setup()

    if analyzer is None:
        print("❌ Could not create analyzer, trying notebook-style setup...")
        analyzer = example_notebook_style_setup()

    if analyzer is None:
        print(
            "❌ Could not create analyzer with any method. Please check your paths."
        )
        return

    # Run examples
    example_feature_analysis(analyzer)
    example_llm_prompt_generation(analyzer)
    example_batch_analysis(analyzer)

    print("\n" + "=" * 60)
    print("🎉 All examples completed!")
    print("=" * 60)

    # Show available functions
    print("\n📚 Available Functions:")
    print("- create_quick_analyzer(): Quick setup with default paths")
    print(
        "- create_feature_analyzer_from_notebook_config(): Notebook-style setup"
    )
    print("- analyzer.get_feature_info(feature_id): Get basic feature info")
    print(
        "- analyzer.compute_output_influences(feature_id): Get token influences"
    )
    print(
        "- analyzer.analyze_feature_comprehensively(feature_id): Full analysis"
    )
    print(
        "- show_highest_activating_examples(analyzer, feature_id): Show examples"
    )
    print(
        "- generate_comprehensive_llm_prompt(analyzer, feature_id): Generate prompt"
    )
    print(
        "- save_feature_analysis(analyzer, feature_id): Save analysis to file"
    )
    print("- batch_analyze_features(analyzer, feature_ids): Batch processing")

    print("\n📝 Example Usage:")
    print("```python")
    print(
        "from splatnlp.feature_analysis import create_quick_analyzer, show_highest_activating_examples"
    )
    print("analyzer = create_quick_analyzer()")
    print("show_highest_activating_examples(analyzer, 1257)")
    print("```")


if __name__ == "__main__":
    main()
