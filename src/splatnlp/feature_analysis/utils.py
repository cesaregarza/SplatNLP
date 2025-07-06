"""
Utility functions for feature analysis.

This module contains helper functions for generating LLM prompts,
displaying examples, and testing the feature analyzer.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Union

import pandas as pd

logger = logging.getLogger(__name__)


def generate_comprehensive_llm_prompt(feature_analyzer, feature_id: int) -> str:
    """Generate a comprehensive LLM prompt for feature analysis."""

    # Get comprehensive analysis
    analysis = feature_analyzer.analyze_feature_comprehensively(
        feature_id, include_buckets=True, include_tfidf=True
    )

    # Start building the prompt
    prompt = f"""# SAE Feature Analysis: {analysis['info']['name']}

## Feature Overview
- **Feature ID**: {feature_id}
- **Human Name**: {analysis['info']['name']}
- **Category**: {analysis['info']['category']}
- **Has Human Label**: {analysis['info']['has_human_label']}
- **Last Updated**: {analysis['info']['last_updated']}

"""

    # Add notes if available
    if analysis["info"]["notes"]:
        prompt += f"**Human Notes**: {analysis['info']['notes']}\n\n"

    # Add statistics if available
    if "statistics" in analysis["info"]:
        stats = analysis["info"]["statistics"]
        prompt += f"""## Feature Statistics
- **Sparsity**: {stats.get('sparsity', 0):.2%} (proportion of zero activations)
- **Max Activation**: {stats.get('max', 0):.4f}
- **Mean Activation**: {stats.get('mean', 0):.4f}
- **Standard Deviation**: {stats.get('std', 0):.4f}
- **Total Examples**: {stats.get('n_total', 0):,}
- **Non-zero Examples**: {stats.get('n_total', 0) - stats.get('n_zeros', 0):,}

"""

    # Add output influences
    if (
        analysis["output_influences"]["positive"]
        or analysis["output_influences"]["negative"]
    ):
        prompt += "## Output Token Influences\n\n"

        if analysis["output_influences"]["positive"]:
            prompt += "**Tokens this feature PROMOTES** (positive influence on logits):\n"
            for item in analysis["output_influences"]["positive"][:8]:
                prompt += (
                    f"- {item['token_name']}: {item['influence_value']:.4f}\n"
                )
            prompt += "\n"

        if analysis["output_influences"]["negative"]:
            prompt += "**Tokens this feature SUPPRESSES** (negative influence on logits):\n"
            for item in analysis["output_influences"]["negative"][:8]:
                prompt += (
                    f"- {item['token_name']}: {item['influence_value']:.4f}\n"
                )
            prompt += "\n"

    # Add TF-IDF analysis
    if (
        analysis["tfidf_analysis"]
        and "top_tfidf_tokens" in analysis["tfidf_analysis"]
    ):
        tfidf = analysis["tfidf_analysis"]
        prompt += f"""## Token Frequency Analysis (TF-IDF)
Based on {tfidf['total_examples']} examples with {tfidf['total_tokens']} total tokens:

**Most characteristic tokens** (high TF-IDF scores):
"""
        for item in tfidf["top_tfidf_tokens"][:10]:
            prompt += f"- {item['token']}: TF-IDF={item['tfidf_score']:.3f}, Frequency={item['frequency']}\n"
        prompt += "\n"

    # Add activation buckets (now showing TOP buckets first)
    if analysis["activation_buckets"]:
        prompt += "## Activation Examples by Strength (Top Buckets First)\n\n"

        for bucket in analysis["activation_buckets"]:
            bucket_label = bucket.get(
                "bucket_label", f"Bucket {bucket.get('bucket_rank', '?')}"
            )
            prompt += f"**{bucket_label} Activation Range {bucket['bucket_range']}** ({bucket['num_examples']} examples):\n"

            for example in bucket["examples"]:
                abilities_str = (
                    example["abilities_str"]
                    if example["abilities_str"] != "None"
                    else "No abilities"
                )
                prompt += f"- Activation {example['activation']:.3f}: {example['weapon']} with {abilities_str}\n"
            prompt += "\n"

    # Add analysis questions
    prompt += """## Analysis Questions

Please analyze this SAE feature and provide insights on:

1. **Feature Interpretation**: Based on the output influences, TF-IDF tokens, and examples, what gameplay concept or build pattern does this feature represent?

2. **Semantic Coherence**: Do the promoted/suppressed tokens make sense together? Is there a coherent theme?

3. **Build Strategy**: What specific build strategy or playstyle does this feature seem to capture?

4. **Activation Patterns**: Looking at the examples across different activation strengths, what triggers this feature to activate more strongly?

5. **Model Reasoning**: Why would the model learn this particular feature? What aspect of the build recommendation task does it serve?

6. **Label Quality**: How well does the human-provided label (if any) capture what this feature actually represents?

## Context
- This is from a Splatoon 3 build recommendation model
- Features are learned by a Sparse Autoencoder (SAE) to capture meaningful build patterns
- The model predicts what abilities to add to complete a build
- Categories: 'tactical' (gameplay strategies), 'mechanical' (specific ability patterns), 'strategic' (high-level build types)
"""

    return prompt


def show_highest_activating_examples(
    feature_analyzer, feature_id: int, limit: int = 10
) -> Optional[Any]:
    """Show the highest activating examples for a feature - the core functionality requested."""

    print(f"=== HIGHEST ACTIVATING EXAMPLES FOR FEATURE {feature_id} ===\n")

    # Get feature info
    info = feature_analyzer.get_feature_info(feature_id)
    print(f"Feature: {info['name']} (Category: {info['category']})")
    if info["notes"]:
        print(f"Notes: {info['notes']}")
    print()

    # Get the highest activating examples directly from the database
    if feature_analyzer.db:
        try:
            # Get top activating examples
            examples_df = feature_analyzer.db.get_feature_activations(
                feature_id, limit=limit
            )

            if len(examples_df) > 0:
                print(f"TOP {len(examples_df)} ACTIVATING EXAMPLES:")
                print("-" * 60)

                for i, example in enumerate(examples_df.to_dicts(), 1):
                    activation = example.get("activation", 0)
                    weapon_id = example.get("weapon_id", 0)
                    weapon_name = feature_analyzer.weapon_name_mapping.get(
                        int(weapon_id), f"Weapon_{weapon_id}"
                    )

                    # Get ability names
                    abilities = []
                    if (
                        "ability_input_tokens" in example
                        and example["ability_input_tokens"]
                    ):
                        try:
                            abilities = [
                                feature_analyzer.inv_vocab.get(
                                    int(token), f"Token_{token}"
                                )
                                for token in example["ability_input_tokens"]
                                if token
                                not in [
                                    feature_analyzer.vocab.get("<PAD>", 0),
                                    feature_analyzer.vocab.get("<NULL>", 1),
                                ]
                            ]
                        except:
                            abilities = ["Error processing abilities"]

                    abilities_str = (
                        ", ".join(abilities) if abilities else "None"
                    )

                    print(
                        f"{i:2d}. Activation: {activation:.4f} | {weapon_name} | {abilities_str}"
                    )

                return examples_df
            else:
                print("No activating examples found.")
                return None

        except Exception as e:
            print(f"Error getting examples: {e}")
            return None
    else:
        print("Dashboard database not available.")
        return None


def test_feature_analyzer(feature_analyzer, feature_id: int = 1257):
    """Test the feature analyzer with a specific feature."""

    print("=== TESTING FEATURE ANALYZER ===\n")

    print(f"Testing Feature {feature_id}:")
    print("-" * 50)

    # Test basic info
    info = feature_analyzer.get_feature_info(feature_id)
    print(f"Name: {info['name']}")
    print(f"Category: {info['category']}")
    print(f"Has Human Label: {info['has_human_label']}")

    # Test output influences (should work now with fixed dimensions)
    influences = feature_analyzer.compute_output_influences(feature_id, limit=5)
    print(f"\nOutput Influences:")
    print(f"  Promotes ({len(influences['positive'])} tokens):")
    for item in influences["positive"]:
        print(f"    {item['token_name']}: {item['influence_value']:.4f}")
    print(f"  Suppresses ({len(influences['negative'])} tokens):")
    for item in influences["negative"]:
        print(f"    {item['token_name']}: {item['influence_value']:.4f}")

    # Test comprehensive analysis (WITH buckets/tfidf since DB is available)
    print(f"\nComprehensive Analysis:")
    analysis = feature_analyzer.analyze_feature_comprehensively(
        feature_id,
        include_buckets=True,  # DB is available!
        include_tfidf=True,  # DB is available!
    )
    print(f"  Interpretation: {analysis['interpretation']}")

    # Show activation buckets (now showing TOP buckets first)
    if analysis["activation_buckets"]:
        print(
            f"\nActivation Buckets ({len(analysis['activation_buckets'])} buckets, showing TOP buckets first):"
        )
        for bucket in analysis["activation_buckets"]:
            bucket_label = bucket.get(
                "bucket_label", f"Bucket {bucket.get('bucket_rank', '?')}"
            )
            print(
                f"  {bucket_label} {bucket['bucket_range']}: {bucket['num_examples']} examples"
            )
            for example in bucket["examples"]:
                print(
                    f"    - {example['activation']:.3f}: {example['weapon']} with {example['abilities_str']}"
                )

    # Show TF-IDF results
    if (
        analysis["tfidf_analysis"]
        and "top_tfidf_tokens" in analysis["tfidf_analysis"]
    ):
        tfidf = analysis["tfidf_analysis"]
        print(
            f"\nTF-IDF Analysis ({tfidf['total_examples']} examples, {tfidf['unique_tokens']} unique tokens):"
        )
        for item in tfidf["top_tfidf_tokens"][:5]:
            print(
                f"  {item['token']}: TF-IDF={item['tfidf_score']:.3f}, Freq={item['frequency']}"
            )

    return analysis


def test_multiple_features(feature_analyzer, test_features: List[int] = None):
    """Test analysis of multiple features from our previous SAE results."""

    print("\n=== ANALYZING MULTIPLE FEATURES ===\n")

    # Use features we found in previous analysis that were active
    if test_features is None:
        test_features = [
            959,
            1257,
            2021,
            845,
            1820,
        ]  # Features from our enhanced analysis

    summaries = []
    for feature_id in test_features:
        try:
            info = feature_analyzer.get_feature_info(feature_id)
            influences = feature_analyzer.compute_output_influences(
                feature_id, limit=3
            )

            summary = {
                "feature_id": feature_id,
                "name": info["name"],
                "category": info["category"],
                "has_label": info["has_human_label"],
                "top_promoted": (
                    influences["positive"][0]["token_name"]
                    if influences["positive"]
                    else "None"
                ),
                "top_suppressed": (
                    influences["negative"][0]["token_name"]
                    if influences["negative"]
                    else "None"
                ),
                "notes": (
                    info["notes"][:50] + "..."
                    if len(info["notes"]) > 50
                    else info["notes"]
                ),
            }
            summaries.append(summary)

        except Exception as e:
            print(f"Error analyzing feature {feature_id}: {e}")

    # Create summary table
    df = pd.DataFrame(summaries)
    print("FEATURE SUMMARY:")
    print(df.to_string(index=False))

    return df


def save_feature_analysis(
    feature_analyzer, feature_id: int, output_file: str = None
) -> str:
    """Generate and save comprehensive LLM prompt for a feature."""

    if output_file is None:
        output_file = f"feature_{feature_id}_analysis.txt"

    print(f"Generating comprehensive LLM prompt for feature {feature_id}...")

    llm_prompt = generate_comprehensive_llm_prompt(feature_analyzer, feature_id)

    # Save the prompt to a file
    with open(output_file, "w") as f:
        f.write(llm_prompt)

    print(f"✅ LLM prompt saved to {output_file}")
    print(f"📊 Prompt length: {len(llm_prompt)} characters")

    return output_file


def batch_analyze_features(
    feature_analyzer, feature_ids: List[int], output_dir: str = "."
) -> List[str]:
    """Analyze multiple features and save their analysis to files."""

    output_files = []

    for feature_id in feature_ids:
        try:
            output_file = f"{output_dir}/feature_{feature_id}_analysis.txt"
            saved_file = save_feature_analysis(
                feature_analyzer, feature_id, output_file
            )
            output_files.append(saved_file)

        except Exception as e:
            logger.error(f"Error analyzing feature {feature_id}: {e}")

    print(f"\n🎉 Batch analysis completed!")
    print(f"📁 Files created: {len(output_files)}")
    for file in output_files:
        print(f"   - {file}")

    return output_files
