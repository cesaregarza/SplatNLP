"""Context loader for mechanistic interpretability experiments.

This module provides model-agnostic loading of databases, vocabularies,
and other context needed for running experiments.
"""

import json
import logging
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from splatnlp.preprocessing.transform.mappings import generate_maps

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_local_weapon_names() -> dict[str, str]:
    """Load weapon ID -> name mapping from the local splatoon3-meta reference.

    Falls back to the network-based `generate_maps` if the local reference
    is unavailable.
    """
    # Search upward from this file to find the repo root that contains .claude
    candidates: list[Path] = []
    for parent in Path(__file__).resolve().parents:
        candidates.append(
            parent
            / ".claude"
            / "skills"
            / "splatoon3-meta"
            / "references"
            / "weapons.md"
        )

    for path in candidates:
        if path.exists():
            try:
                mapping: dict[str, str] = {}
                with path.open() as f:
                    for line in f:
                        line = line.strip()
                        if not line.startswith("|"):
                            continue
                        # Expected row format: | ID | Name | Sub | Special |
                        parts = [p.strip() for p in line.strip("|").split("|")]
                        if len(parts) < 2:
                            continue
                        weapon_id, weapon_name = parts[0], parts[1]
                        if weapon_id.isdigit() and weapon_name:
                            mapping[weapon_id] = weapon_name
                if mapping:
                    logger.info(
                        "Loaded local weapon names from %s (%d entries)",
                        path,
                        len(mapping),
                    )
                    return mapping
            except Exception as exc:  # pragma: no cover - best effort
                logger.debug(
                    f"Failed to load local weapon names from {path}: {exc}"
                )

    # Fallback: defer to generate_maps (network)
    try:
        _, id_to_name, _ = generate_maps()
        return id_to_name
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug(f"Fallback weapon name mapping failed: {exc}")
        return {}


@lru_cache(maxsize=1)
def _get_weapon_id_to_name() -> dict[str, str]:
    """Get mapping from weapon ID string to display name.

    Returns:
        Dict mapping "1111" -> "Octobrush Nouveau", etc.
    """
    return _load_local_weapon_names()


# Default paths for models and data
FULL_MODEL_PATHS = {
    "vocab": "saved_models/dataset_v0_2_full/vocab.json",
    "weapon_vocab": "saved_models/dataset_v0_2_full/weapon_vocab.json",
    "data_dir": "/mnt/e/activations_full_efficient",
    "examples_dir": "/mnt/e/activations_full_efficient/examples",
}

ULTRA_MODEL_PATHS = {
    "vocab": "saved_models/dataset_v0_2_full/vocab.json",  # Same vocab
    "weapon_vocab": "saved_models/dataset_v0_2_full/weapon_vocab.json",
    "data_dir": "/mnt/e/activations_ultra_efficient",
    "examples_dir": "/mnt/e/dashboard_examples_optimized",
}


@dataclass
class MechInterpContext:
    """Context object for running mechinterp experiments.

    This provides all the data access and vocabulary information needed
    by experiment runners.
    """

    model_type: Literal["full", "ultra"]
    db: Any  # FSDatabase or EfficientFSDatabase
    vocab: dict[str, int]
    inv_vocab: dict[int, str]
    weapon_vocab: dict[str, int]
    inv_weapon_vocab: dict[int, str]
    pad_token_id: int
    mask_token_id: int
    null_token_id: int
    project_root: Path = field(
        default_factory=lambda: Path("/root/dev/SplatNLP")
    )

    def token_to_id(self, token: str) -> int | None:
        """Convert token string to ID."""
        return self.vocab.get(token)

    def id_to_token(self, token_id: int) -> str | None:
        """Convert token ID to string."""
        return self.inv_vocab.get(token_id)

    def weapon_to_id(self, weapon: str) -> int | None:
        """Convert weapon name to ID."""
        return self.weapon_vocab.get(weapon)

    def id_to_weapon(self, weapon_id: int) -> str | None:
        """Convert weapon ID to internal name (weapon_id_XXXX format)."""
        return self.inv_weapon_vocab.get(weapon_id)

    def id_to_weapon_display_name(self, weapon_id: int) -> str:
        """Convert weapon ID to human-readable display name.

        Args:
            weapon_id: Internal weapon token ID

        Returns:
            Display name like "Octobrush Nouveau" or fallback "Weapon 1111"
        """
        # Get the internal name first (e.g., "weapon_id_1111")
        internal_name = self.inv_weapon_vocab.get(weapon_id)
        if not internal_name:
            return f"Unknown Weapon {weapon_id}"

        # Extract the numeric ID from "weapon_id_1111" format
        if internal_name.startswith("weapon_id_"):
            numeric_id = internal_name[len("weapon_id_") :]
        else:
            numeric_id = internal_name.split("_")[-1]

        # Look up display name
        id_to_name = _get_weapon_id_to_name()
        return id_to_name.get(numeric_id, f"Weapon {numeric_id}")

    def get_ability_tokens(self) -> list[str]:
        """Get all ability tokens (excluding special tokens and weapons)."""
        special = {"<PAD>", "<MASK>", "<NULL>"}
        return [
            t
            for t in self.vocab.keys()
            if t not in special and t not in self.weapon_vocab
        ]

    def get_feature_ids(self) -> list[int]:
        """Get all available feature IDs from the database."""
        return self.db.get_all_feature_ids()


def load_context(
    model_type: Literal["full", "ultra"] = "ultra",
    project_root: Path | str | None = None,
) -> MechInterpContext:
    """Load context for running mechinterp experiments.

    Args:
        model_type: Which model to load ("full" for 2K, "ultra" for 24K)
        project_root: Override project root path

    Returns:
        MechInterpContext with database and vocabularies loaded
    """
    if project_root is None:
        project_root = Path("/root/dev/SplatNLP")
    else:
        project_root = Path(project_root)

    # Load vocabularies (same for both models)
    vocab_path = project_root / FULL_MODEL_PATHS["vocab"]
    weapon_vocab_path = project_root / FULL_MODEL_PATHS["weapon_vocab"]

    with open(vocab_path) as f:
        vocab = json.load(f)
    with open(weapon_vocab_path) as f:
        weapon_vocab = json.load(f)

    inv_vocab = {v: k for k, v in vocab.items()}
    inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}

    # Load appropriate database
    if model_type == "ultra":
        db = _load_ultra_db()
    else:
        db = _load_full_db()

    context = MechInterpContext(
        model_type=model_type,
        db=db,
        vocab=vocab,
        inv_vocab=inv_vocab,
        weapon_vocab=weapon_vocab,
        inv_weapon_vocab=inv_weapon_vocab,
        pad_token_id=vocab.get("<PAD>", 0),
        mask_token_id=vocab.get("<MASK>", 1),
        null_token_id=vocab.get("<NULL>", 2),
        project_root=project_root,
    )

    logger.info(
        f"Loaded {model_type} context: "
        f"{len(vocab)} tokens, {len(weapon_vocab)} weapons"
    )

    return context


def _load_full_db():
    """Load database for Full model (2K features).

    Uses ServerBackedDatabase which automatically connects to the activation
    server if it's running, otherwise falls back to direct database access.
    """
    try:
        from splatnlp.mechinterp.server.client import ServerBackedDatabase

        db = ServerBackedDatabase(
            data_dir=FULL_MODEL_PATHS["data_dir"],
            examples_dir=FULL_MODEL_PATHS["examples_dir"],
        )
        logger.info("Loaded ServerBackedDatabase for Full model")
        return db
    except Exception as e:
        logger.error(f"Failed to load Full model database: {e}")
        raise


def _load_ultra_db():
    """Load database for Ultra model (24K features).

    Uses ServerBackedDatabase which automatically connects to the activation
    server if it's running, otherwise falls back to direct database access.
    """
    try:
        from splatnlp.mechinterp.server.client import ServerBackedDatabase

        db = ServerBackedDatabase(
            data_dir=ULTRA_MODEL_PATHS["data_dir"],
            examples_dir=ULTRA_MODEL_PATHS["examples_dir"],
        )
        logger.info("Loaded ServerBackedDatabase for Ultra model")
        return db
    except Exception as e:
        logger.error(f"Failed to load Ultra model database: {e}")
        raise


def get_model_info(model_type: Literal["full", "ultra"]) -> dict[str, Any]:
    """Get information about a model configuration.

    Args:
        model_type: Model type to query

    Returns:
        Dict with model configuration info
    """
    if model_type == "ultra":
        return {
            "name": "Ultra",
            "n_features": 24576,
            "embedding_dim": 32,
            "hidden_dim": 512,
            "expansion_factor": 48.0,
            "storage_format": "zarr/parquet",
            "paths": ULTRA_MODEL_PATHS,
        }
    else:
        return {
            "name": "Full",
            "n_features": 2048,
            "embedding_dim": 512,
            "hidden_dim": 512,
            "expansion_factor": 4.0,
            "storage_format": "numpy",
            "paths": FULL_MODEL_PATHS,
        }
