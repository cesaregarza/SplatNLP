"""Decoder output analysis experiment runner.

This module implements the decoder_output_analysis experiment which analyzes
what tokens a feature PROMOTES or SUPPRESSES via the path:

    feature_decoder_vector → output_layer_weights → token logits

This is complementary to activation analysis: instead of asking "what activates
this feature?", we ask "what does this feature recommend?".
"""

import logging
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import torch

from splatnlp.mechinterp.experiments.base import (
    ExperimentRunner,
    register_runner,
)
from splatnlp.mechinterp.schemas.experiment_results import ExperimentResult
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)

# Model and SAE paths
ULTRA_MODEL_PATH = Path(
    "/mnt/e/dev_spillover/SplatNLP/saved_models/dataset_v0_2_super/clean_slate.pth"
)
ULTRA_SAE_PATH = Path(
    "/mnt/e/dev_spillover/SplatNLP/sae_runs/run_20250704_191557/sae_model_final.pth"
)

FULL_MODEL_PATH = Path(
    "/root/dev/SplatNLP/saved_models/dataset_v0_2_full/model.pth"
)
FULL_SAE_PATH = Path(
    "/mnt/d/activations/sae_checkpoint.pth"  # Adjust as needed
)


def _extract_ap_level(token: str) -> tuple[str, int | None]:
    """Extract ability family and AP level from token.

    Examples:
        'swim_speed_up_57' -> ('swim_speed_up', 57)
        'respawn_punisher' -> ('respawn_punisher', None)
        'special_charge_up_6' -> ('special_charge_up', 6)
    """
    if not token:
        return "", None

    match = re.match(r"^(.+?)_(\d+)$", token)
    if match:
        return match.group(1), int(match.group(2))
    return token, None


@lru_cache(maxsize=2)
def _load_model_weights(
    model_type: Literal["full", "ultra"],
) -> torch.Tensor:
    """Load model output layer weights.

    Returns:
        Output layer weight tensor [vocab_size, hidden_dim]
    """
    if model_type == "ultra":
        path = ULTRA_MODEL_PATH
    else:
        path = FULL_MODEL_PATH

    if not path.exists():
        raise FileNotFoundError(f"Model not found at {path}")

    logger.info(f"Loading model weights from {path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    state_dict = torch.load(path, map_location=device, weights_only=True)

    # Handle different state dict formats
    if "output_layer.weight" in state_dict:
        output_weight = state_dict["output_layer.weight"]
    elif "model.output_layer.weight" in state_dict:
        output_weight = state_dict["model.output_layer.weight"]
    else:
        # Try to find it
        for key in state_dict:
            if "output_layer.weight" in key:
                output_weight = state_dict[key]
                break
        else:
            raise KeyError("Could not find output_layer.weight in state dict")

    logger.info(f"Loaded output layer: {output_weight.shape}")
    return output_weight.cpu()


@lru_cache(maxsize=2)
def _load_sae_decoder(
    model_type: Literal["full", "ultra"],
) -> torch.Tensor:
    """Load SAE decoder weights.

    Returns:
        Decoder weight tensor [hidden_dim, n_features]
    """
    if model_type == "ultra":
        path = ULTRA_SAE_PATH
    else:
        path = FULL_SAE_PATH

    if not path.exists():
        raise FileNotFoundError(f"SAE not found at {path}")

    logger.info(f"Loading SAE decoder from {path}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location=device, weights_only=True)

    # SAE decoder is typically stored as 'decoder.weight'
    if "decoder.weight" in checkpoint:
        decoder_weight = checkpoint["decoder.weight"]
    elif "decoder" in checkpoint:
        decoder_weight = checkpoint["decoder"]
    else:
        raise KeyError("Could not find decoder weights in SAE checkpoint")

    logger.info(f"Loaded SAE decoder: {decoder_weight.shape}")
    return decoder_weight.cpu()


def get_feature_decoder_vector(
    feature_id: int,
    model_type: Literal["full", "ultra"],
) -> torch.Tensor:
    """Get the decoder vector for a specific feature.

    Args:
        feature_id: SAE feature ID
        model_type: Model type

    Returns:
        Feature decoder vector [hidden_dim]
    """
    decoder_weight = _load_sae_decoder(model_type)

    # decoder_weight shape: [hidden_dim, n_features]
    if feature_id >= decoder_weight.shape[1]:
        raise ValueError(
            f"Feature ID {feature_id} out of range "
            f"(max: {decoder_weight.shape[1] - 1})"
        )

    return decoder_weight[:, feature_id]


def compute_output_contribution(
    feature_id: int,
    model_type: Literal["full", "ultra"],
) -> torch.Tensor:
    """Compute output contribution for a feature.

    Formula: contribution = output_weight @ feature_decoder

    Args:
        feature_id: SAE feature ID
        model_type: Model type

    Returns:
        Contribution vector [vocab_size] showing how this feature
        affects each token's logit.
    """
    output_weight = _load_model_weights(model_type)  # [vocab_size, hidden_dim]
    feature_decoder = get_feature_decoder_vector(feature_id, model_type)  # [hidden_dim]

    # Compute contribution: [vocab_size, hidden_dim] @ [hidden_dim] = [vocab_size]
    contribution = torch.matmul(output_weight, feature_decoder)

    return contribution


@register_runner
class DecoderOutputRunner(ExperimentRunner):
    """Runner for decoder output analysis experiments.

    Analyzes what tokens a feature PROMOTES (positive contribution) or
    SUPPRESSES (negative contribution) via the decoder → output layer path.

    This answers: "What does this feature recommend?" rather than
    "What activates this feature?"
    """

    name = "decoder_output_analysis"
    handles_types = [ExperimentType.DECODER_OUTPUT_ANALYSIS]

    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute decoder output analysis."""
        # Parse variables
        vars_dict = spec.variables
        top_k_promoted = vars_dict.get("top_k_promoted", 15)
        top_k_suppressed = vars_dict.get("top_k_suppressed", 15)
        group_by_family = vars_dict.get("group_by_family", True)
        include_ap_level = vars_dict.get("include_ap_level", True)
        exclude_special_tokens = vars_dict.get("exclude_special_tokens", True)

        logger.info(
            f"Running decoder output analysis for feature {spec.feature_id}"
        )

        # Compute output contribution
        contribution = compute_output_contribution(
            spec.feature_id, spec.model_type
        )

        # Get token names
        vocab_size = contribution.shape[0]
        token_contributions: list[dict[str, Any]] = []

        special_tokens = {"<PAD>", "<MASK>", "<NULL>", "<UNK>"}

        for token_id in range(vocab_size):
            token_name = ctx.inv_vocab.get(token_id, f"token_{token_id}")

            if exclude_special_tokens and token_name in special_tokens:
                continue

            contrib_val = float(contribution[token_id])
            family, ap_level = _extract_ap_level(token_name)

            token_contributions.append(
                {
                    "token": token_name,
                    "token_id": token_id,
                    "contribution": contrib_val,
                    "family": family,
                    "ap_level": ap_level,
                }
            )

        # Split into promoted and suppressed
        promoted = sorted(
            [t for t in token_contributions if t["contribution"] > 0],
            key=lambda x: -x["contribution"],
        )[:top_k_promoted]

        suppressed = sorted(
            [t for t in token_contributions if t["contribution"] < 0],
            key=lambda x: x["contribution"],
        )[:top_k_suppressed]

        # Add tables
        result.add_table(
            "promoted_tokens",
            promoted,
            columns=["token", "contribution", "family", "ap_level"],
            description="Tokens this feature PROMOTES (positive output contribution)",
        )

        result.add_table(
            "suppressed_tokens",
            suppressed,
            columns=["token", "contribution", "family", "ap_level"],
            description="Tokens this feature SUPPRESSES (negative output contribution)",
        )

        # Group by family if requested
        if group_by_family:
            family_stats = self._compute_family_summary(token_contributions)
            result.add_table(
                "family_summary",
                family_stats,
                columns=[
                    "family",
                    "mean_contribution",
                    "max_contribution",
                    "min_contribution",
                    "n_tokens",
                ],
                description="Summary of output contribution by ability family",
            )

        # AP level patterns if requested
        if include_ap_level:
            ap_patterns = self._compute_ap_patterns(token_contributions)
            result.add_table(
                "ap_level_patterns",
                ap_patterns,
                columns=["ap_level", "mean_contribution", "n_tokens"],
                description="Output contribution patterns by AP level (e.g., _3 vs _57)",
            )

        # Compute aggregates
        result.aggregates.n_samples = vocab_size
        result.aggregates.n_conditions = len(token_contributions)

        if promoted:
            result.aggregates.custom["top_promoted"] = promoted[0]["token"]
            result.aggregates.custom["top_promoted_value"] = round(
                promoted[0]["contribution"], 4
            )
        if suppressed:
            result.aggregates.custom["top_suppressed"] = suppressed[0]["token"]
            result.aggregates.custom["top_suppressed_value"] = round(
                suppressed[0]["contribution"], 4
            )

        result.aggregates.custom["n_promoted"] = len(promoted)
        result.aggregates.custom["n_suppressed"] = len(suppressed)
        result.aggregates.custom["total_positive"] = sum(
            1 for t in token_contributions if t["contribution"] > 0
        )
        result.aggregates.custom["total_negative"] = sum(
            1 for t in token_contributions if t["contribution"] < 0
        )

        # Compute decoder weight magnitude (importance)
        feature_decoder = get_feature_decoder_vector(
            spec.feature_id, spec.model_type
        )
        decoder_magnitude = float(torch.norm(feature_decoder, p=2))
        result.aggregates.custom["decoder_magnitude"] = round(decoder_magnitude, 4)

        # Diagnostics
        result.diagnostics.n_contexts_tested = vocab_size

        logger.info(
            f"Decoder output analysis complete: "
            f"{len(promoted)} promoted, {len(suppressed)} suppressed"
        )

    def _compute_family_summary(
        self, token_contributions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compute summary statistics by family."""
        from collections import defaultdict

        family_contribs: dict[str, list[float]] = defaultdict(list)

        for tc in token_contributions:
            family = tc["family"]
            if family:
                family_contribs[family].append(tc["contribution"])

        summary = []
        for family, contribs in family_contribs.items():
            summary.append(
                {
                    "family": family,
                    "mean_contribution": round(sum(contribs) / len(contribs), 4),
                    "max_contribution": round(max(contribs), 4),
                    "min_contribution": round(min(contribs), 4),
                    "n_tokens": len(contribs),
                }
            )

        # Sort by mean contribution (net effect)
        summary.sort(key=lambda x: -x["mean_contribution"])
        return summary

    def _compute_ap_patterns(
        self, token_contributions: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Compute patterns by AP level."""
        from collections import defaultdict

        ap_contribs: dict[int | str, list[float]] = defaultdict(list)

        for tc in token_contributions:
            ap = tc["ap_level"]
            if ap is not None:
                ap_contribs[ap].append(tc["contribution"])
            else:
                ap_contribs["binary"].append(tc["contribution"])

        patterns = []
        for ap, contribs in ap_contribs.items():
            patterns.append(
                {
                    "ap_level": str(ap) if ap != "binary" else "binary",
                    "mean_contribution": round(sum(contribs) / len(contribs), 4),
                    "n_tokens": len(contribs),
                }
            )

        # Sort by AP level
        def sort_key(x):
            if x["ap_level"] == "binary":
                return -1
            return int(x["ap_level"])

        patterns.sort(key=sort_key)
        return patterns
