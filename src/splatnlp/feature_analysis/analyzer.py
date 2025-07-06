"""
Feature analyzer for SAE features.

This module provides the main FeatureAnalyzer class that offers
comprehensive analysis of SAE features including output influences,
activation buckets, TF-IDF analysis, and more.
"""

import json
import logging
import re
from collections import Counter
from typing import Any

import numpy as np

from splatnlp.dashboard.fs_database import FSDatabase
from splatnlp.dashboard.utils.converters import generate_weapon_name_mapping

logger = logging.getLogger(__name__)


class FeatureAnalyzer:
    """Feature analysis with TFIDF, examples, and activation buckets."""

    def __init__(
        self,
        primary_model,
        sae_model,
        vocab: dict,
        weapon_vocab: dict,
        feature_labels_path: str = "src/splatnlp/dashboard/feature_labels.json",
        meta_path: str = None,
        neurons_root: str = None,
        device: str = "cuda",
    ):
        """Initialize the enhanced feature analyzer."""
        self.primary_model = primary_model
        self.sae_model = sae_model
        self.vocab = vocab
        self.weapon_vocab = weapon_vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        self.inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}
        self.device = device

        # Load feature labels
        self.feature_labels = self._load_feature_labels(feature_labels_path)

        # Initialize dashboard database if paths provided
        self.db = None
        if meta_path and neurons_root:
            try:
                self.db = FSDatabase(meta_path, neurons_root)
                logger.info("Dashboard database initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize dashboard database: {e}")

        # Generate weapon name mapping
        self.weapon_name_mapping = generate_weapon_name_mapping(
            self.inv_weapon_vocab
        )

        # Pre-compute patterns for analysis
        self.HIGH_AP_PATTERN = re.compile(r"_(21|29|38|51|57)$")
        self.SPECIAL_TOKENS = {"<PAD>", "<NULL>"}

        # SAE dimensions
        self.sae_input_dim = self.sae_model.decoder.weight.shape[0]  # 512
        self.sae_hidden_dim = self.sae_model.decoder.weight.shape[1]  # 2048

        logger.info(
            f"SAE dimensions: input={self.sae_input_dim}, hidden={self.sae_hidden_dim}"
        )

    def _load_feature_labels(self, labels_path: str) -> dict:
        """Load feature labels from JSON file."""
        try:
            with open(labels_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Feature labels file not found: {labels_path}")
            return {}
        except Exception as e:
            logger.warning(f"Error loading feature labels: {e}")
            return {}

    def get_feature_info(self, feature_id: int) -> dict:
        """Get comprehensive info about a feature."""
        feature_str = str(feature_id)
        label_info = self.feature_labels.get(feature_str, {})

        info = {
            "feature_id": feature_id,
            "name": label_info.get("name", f"Feature {feature_id}"),
            "category": label_info.get("category", "unknown"),
            "notes": label_info.get("notes", ""),
            "last_updated": label_info.get("timestamp", ""),
            "has_human_label": bool(label_info.get("name", "").strip()),
        }

        # Add dashboard stats if available
        if self.db:
            try:
                stats = self.db.get_feature_stats(feature_id)
                info.update(
                    {
                        "statistics": stats,
                        "sparsity": stats.get("sparsity", 0),
                        "max_activation": stats.get("max", 0),
                        "mean_activation": stats.get("mean", 0),
                    }
                )
            except Exception as e:
                logger.warning(
                    f"Could not get stats for feature {feature_id}: {e}"
                )

        return info

    def compute_output_influences(
        self, feature_id: int, limit: int = 10
    ) -> dict[str, list[dict[str, Any]]]:
        """Compute how this feature influences output logits (FIXED DIMENSIONS)."""
        try:
            if feature_id >= self.sae_hidden_dim:
                logger.warning(
                    f"Feature ID {feature_id} is out of bounds for SAE with {self.sae_hidden_dim} features"
                )
                return {"positive": [], "negative": []}

            # Get SAE decoder weights: [input_dim, hidden_dim] = [512, 2048]
            sae_decoder_weights = (
                self.sae_model.decoder.weight.data.cpu().numpy()
            )

            # Get the decoder vector for this feature: [512]
            feature_decoder_vector = sae_decoder_weights[:, feature_id]

            # Get output layer weights: [output_vocab_size, input_dim] = [140, 512]
            output_weights = (
                self.primary_model.output_layer.weight.data.cpu().numpy()
            )

            # Compute influence: how much this SAE feature affects each output token [140]
            influences = np.dot(output_weights, feature_decoder_vector)

            # Get top positive and negative influences
            top_pos_indices = np.argsort(influences)[-limit * 3 :][::-1]
            top_neg_indices = np.argsort(influences)[: limit * 10]

            positive = [
                {
                    "token_id": int(idx),
                    "token_name": self.inv_vocab.get(int(idx), f"Token_{idx}"),
                    "influence_value": float(influences[idx]),
                }
                for idx in top_pos_indices
                if self.inv_vocab.get(int(idx), f"Token_{idx}")
                not in self.SPECIAL_TOKENS
            ][:limit]

            negative = [
                {
                    "token_id": int(idx),
                    "token_name": self.inv_vocab.get(int(idx), f"Token_{idx}"),
                    "influence_value": float(influences[idx]),
                }
                for idx in top_neg_indices
                if self.inv_vocab.get(int(idx), f"Token_{idx}")
                not in self.SPECIAL_TOKENS
            ][:limit]

            return {"positive": positive, "negative": negative}

        except Exception as e:
            logger.warning(
                f"Error computing output influences for feature {feature_id}: {e}"
            )
            return {"positive": [], "negative": []}

    def get_activation_buckets(
        self, feature_id: int, num_buckets: int = 5
    ) -> list[dict[str, Any]]:
        """Get examples from different activation buckets for a feature, prioritizing TOP buckets."""
        if not self.db:
            logger.warning(
                "Dashboard database not available for activation buckets"
            )
            return []

        try:
            # Get all activations for this feature (already sorted by activation desc)
            activations_df = self.db.get_feature_activations(
                feature_id, limit=None
            )

            if len(activations_df) == 0:
                return []

            # Create activation buckets focusing on TOP activation ranges
            activations = activations_df["activation"].to_numpy()
            min_act, max_act = activations.min(), activations.max()

            # Define bucket edges - but we'll prioritize the TOP buckets
            bucket_edges = np.linspace(min_act, max_act, num_buckets + 1)

            buckets = []
            # Process buckets from HIGHEST to LOWEST activation (reverse order)
            for i in range(num_buckets - 1, -1, -1):
                lower = bucket_edges[i]
                upper = bucket_edges[i + 1]

                # Get examples in this bucket
                mask = (activations >= lower) & (activations < upper)
                if i == num_buckets - 1:  # Include max value in last bucket
                    mask = (activations >= lower) & (activations <= upper)

                bucket_examples = activations_df.filter(mask)

                # Sample a few examples from this bucket
                sample_size = min(3, len(bucket_examples))
                if sample_size > 0:
                    # Sort by activation descending within bucket to get the highest ones
                    bucket_examples = bucket_examples.sort(
                        "activation", descending=True
                    )
                    sampled = bucket_examples.head(sample_size)

                    examples = []
                    for example in sampled.to_dicts():
                        # Get weapon name
                        weapon_name = self.weapon_name_mapping.get(
                            int(example.get("weapon_id", 0)),
                            f"Weapon_{example.get('weapon_id', 'unknown')}",
                        )

                        # Get ability names
                        ability_tags = []
                        if (
                            "ability_input_tokens" in example
                            and example["ability_input_tokens"] is not None
                        ):
                            try:
                                ability_tags = [
                                    self.inv_vocab.get(int(tag), f"Token_{tag}")
                                    for tag in example["ability_input_tokens"]
                                    if tag != self.vocab.get("<PAD>", 0)
                                ]
                            except Exception as e:
                                logger.warning(
                                    f"Error processing ability tags: {e}"
                                )
                                ability_tags = ["Error processing tags"]

                        examples.append(
                            {
                                "activation": float(
                                    example.get("activation", 0)
                                ),
                                "weapon": weapon_name,
                                "abilities": ability_tags,
                                "abilities_str": (
                                    ", ".join(ability_tags)
                                    if ability_tags
                                    else "None"
                                ),
                            }
                        )

                    # Calculate bucket rank (1 = highest activation bucket)
                    bucket_rank = num_buckets - i

                    buckets.append(
                        {
                            "bucket_range": f"[{lower:.3f}, {upper:.3f}]",
                            "bucket_index": i,
                            "bucket_rank": bucket_rank,  # 1 = highest, 2 = second highest, etc.
                            "bucket_label": (
                                f"Top {bucket_rank}"
                                if bucket_rank <= 3
                                else f"Bucket {bucket_rank}"
                            ),
                            "num_examples": len(bucket_examples),
                            "examples": examples,
                        }
                    )

            return buckets

        except Exception as e:
            logger.warning(
                f"Error getting activation buckets for feature {feature_id}: {e}"
            )
            return []

    def compute_feature_tfidf(
        self, feature_id: int, top_k: int = 20
    ) -> dict[str, Any]:
        """Compute TFIDF scores for tokens associated with this feature."""
        if not self.db:
            logger.warning(
                "Dashboard database not available for TFIDF computation"
            )
            return {}

        try:
            # Get top activating examples
            examples_df = self.db.get_feature_activations(feature_id, limit=100)

            if len(examples_df) == 0:
                return {}

            # Extract all tokens from examples
            all_tokens = []
            for example in examples_df.to_dicts():
                if (
                    "ability_input_tokens" in example
                    and example["ability_input_tokens"] is not None
                ):
                    try:
                        tokens = [
                            self.inv_vocab.get(int(tag), f"Token_{tag}")
                            for tag in example["ability_input_tokens"]
                            if tag != self.vocab.get("<PAD>", 0)
                            and tag != self.vocab.get("<NULL>", 1)
                        ]
                        all_tokens.extend(tokens)
                    except Exception as e:
                        logger.warning(
                            f"Error processing tokens for TFIDF: {e}"
                        )

            # Compute token frequencies
            token_counts = Counter(all_tokens)

            # Simple TF-IDF approximation (would need corpus for proper IDF)
            # For now, we'll use inverse frequency as a proxy
            total_tokens = len(all_tokens)
            tfidf_scores = {}

            for token, count in token_counts.items():
                if token not in self.SPECIAL_TOKENS:
                    tf = count / total_tokens
                    # Simple inverse frequency (not true IDF)
                    idf = np.log(total_tokens / count) if count > 0 else 0
                    tfidf_scores[token] = tf * idf

            # Get top tokens by TF-IDF score
            top_tokens = sorted(
                tfidf_scores.items(), key=lambda x: x[1], reverse=True
            )[:top_k]

            return {
                "total_examples": len(examples_df),
                "total_tokens": total_tokens,
                "unique_tokens": len(token_counts),
                "top_tfidf_tokens": [
                    {
                        "token": token,
                        "tfidf_score": score,
                        "frequency": token_counts[token],
                    }
                    for token, score in top_tokens
                ],
            }

        except Exception as e:
            logger.warning(
                f"Error computing TFIDF for feature {feature_id}: {e}"
            )
            return {}

    def analyze_feature_comprehensively(
        self,
        feature_id: int,
        include_buckets: bool = True,
        include_tfidf: bool = True,
    ) -> dict[str, Any]:
        """Get comprehensive analysis including buckets and TFIDF."""

        # Get basic info
        info = self.get_feature_info(feature_id)

        # Get output influences
        influences = self.compute_output_influences(feature_id)

        # Get activation buckets if requested
        buckets = []
        if include_buckets:
            buckets = self.get_activation_buckets(feature_id)

        # Get TFIDF if requested
        tfidf_data = {}
        if include_tfidf:
            tfidf_data = self.compute_feature_tfidf(feature_id)

        return {
            "feature_id": feature_id,
            "info": info,
            "output_influences": influences,
            "activation_buckets": buckets,
            "tfidf_analysis": tfidf_data,
            "interpretation": self._generate_comprehensive_interpretation(
                info, influences, buckets, tfidf_data
            ),
        }

    def _generate_comprehensive_interpretation(
        self, info: dict, influences: dict, buckets: list, tfidf: dict[str, Any]
    ) -> str:
        """Generate comprehensive interpretation including all analysis."""
        interpretation = []

        # Basic info
        name = info.get("name", f"Feature {info['feature_id']}")
        category = info.get("category", "unknown")
        notes = info.get("notes", "")

        interpretation.append(f"Feature {info['feature_id']}: {name}")
        interpretation.append(f"Category: {category}")

        if notes:
            interpretation.append(f"Notes: {notes}")

        # Statistics
        if "statistics" in info:
            stats = info["statistics"]
            interpretation.append(
                f"Sparsity: {stats.get('sparsity', 0):.2%}, Max: {stats.get('max', 0):.3f}"
            )

        # Output influences
        if influences["positive"]:
            top_positive = influences["positive"][:3]
            pos_names = [item["token_name"] for item in top_positive]
            interpretation.append(f"Promotes: {', '.join(pos_names)}")

        if influences["negative"]:
            top_negative = influences["negative"][:3]
            neg_names = [item["token_name"] for item in top_negative]
            interpretation.append(f"Suppresses: {', '.join(neg_names)}")

        # TFIDF insights
        if tfidf and "top_tfidf_tokens" in tfidf:
            top_tfidf = tfidf["top_tfidf_tokens"][:3]
            if top_tfidf:
                tfidf_tokens = [item["token"] for item in top_tfidf]
                interpretation.append(
                    f"Key tokens (TF-IDF): {', '.join(tfidf_tokens)}"
                )

        # Bucket insights (now showing top buckets)
        if buckets:
            total_examples = sum(b["num_examples"] for b in buckets)
            interpretation.append(
                f"Top activation buckets: {len(buckets)} buckets with {total_examples} total examples"
            )

        return " | ".join(interpretation)
