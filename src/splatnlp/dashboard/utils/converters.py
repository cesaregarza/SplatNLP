import logging
from functools import lru_cache
from pathlib import Path
from typing import Sequence, TypedDict

import numpy as np
import polars as pl
import requests

from splatnlp.preprocessing.transform.mappings import (
    FINAL_REF_URL,
    TRANSLATION_URL,
    generate_maps,
)

logger = logging.getLogger(__name__)


class AbilityTagParser:
    """Handles parsing of ability tags from various formats."""

    @staticmethod
    def parse(ability_tags: np.ndarray | list[int] | str) -> list[int]:
        """Parse ability tags from various formats into a list of integers."""
        if isinstance(ability_tags, np.ndarray):
            return ability_tags.tolist()

        if isinstance(ability_tags, list):
            return ability_tags

        ability_tags_str = str(ability_tags)
        if ability_tags_str == "[]":
            return []

        try:
            if ability_tags_str.startswith("[") and ability_tags_str.endswith(
                "]"
            ):
                content = ability_tags_str.strip("[]")
                if not content:
                    return []
                return [int(x.strip()) for x in content.split(",") if x.strip()]
        except ValueError:
            logger.warning(f"Failed to parse ability tags: {ability_tags_str}")
            return []

        return []


def ability_ids_to_names(
    ids: Sequence[int],
    inv_vocab: dict[str, str],
) -> list[str]:
    names: list[str] = []
    for tid in ids:
        name = inv_vocab.get(str(tid), inv_vocab.get(tid, f"ID_{tid}"))
        if name not in ("<PAD>", "<NULL>"):
            names.append(name)
    return names


def weapon_to_name(
    weapon_id: int | str,
    inv_weapon_vocab: dict[int, str],
    id_to_name: dict[str, str],
) -> str:
    try:
        weapon_id = int(weapon_id)
    except ValueError:
        return "Unknown Weapon"

    raw_wpn = inv_weapon_vocab.get(weapon_id, f"WPN_{weapon_id}")
    return id_to_name.get(raw_wpn.split("_")[-1], raw_wpn)


def generate_weapon_name_mapping(
    inv_weapon_vocab: dict[int, str],
) -> dict[int, str]:
    """Generate a mapping of weapon IDs to weapon names.

    Prefers the local splatoon3-meta reference to avoid network calls.
    Falls back to `generate_maps` if the local file is unavailable.
    """
    id_to_name = _load_local_weapon_names()
    result: dict[int, str] = {}
    for k, v in inv_weapon_vocab.items():
        weapon_id_str = v.split("_")[-1]
        result[k] = id_to_name.get(weapon_id_str, f"Weapon {weapon_id_str}")
    return result


@lru_cache(maxsize=1)
def _load_local_weapon_names() -> dict[str, str]:
    """Load weapon ID -> name mapping from the local splatoon3-meta reference."""
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

    # Fallback to network-based mapping as a last resort
    try:
        _, id_to_name, _ = generate_maps()
        return id_to_name
    except Exception as exc:  # pragma: no cover - best effort
        logger.debug(f"Fallback weapon name mapping failed: {exc}")
        return {}


class WeaponProperties(TypedDict):
    season: int
    sub: str
    special: str
    class_: str
    kit: str
    reference_kit: str
    reference_id: str


@lru_cache(maxsize=None)
def get_weapon_properties() -> dict[str, WeaponProperties]:
    """Get properties of a weapon."""
    weapon_ref: dict[str, dict[str, int | str]] = requests.get(
        FINAL_REF_URL
    ).json()
    out = {}
    for k, v in weapon_ref.items():
        out[k] = {kk if kk != "class" else "class_": vv for kk, vv in v.items()}
    return out


@lru_cache(maxsize=None)
def get_translation() -> dict[str, dict[str, str]]:
    """Get translation of a weapon."""
    translation = requests.get(TRANSLATION_URL).json()
    return translation


@lru_cache(maxsize=None)
def get_weapon_properties_df() -> pl.DataFrame:
    """Get properties of a weapon."""
    props = get_weapon_properties()
    translation = get_translation()
    subs = translation["USen"]["WeaponName_Sub"]
    specials = translation["USen"]["WeaponName_Special"]
    classes = translation["USen"]["WeaponTypeName"]
    rows = [{"weapon_id": k, **v} for k, v in props.items()]
    return (
        pl.DataFrame(rows)
        .with_columns(
            pl.lit("weapon_id_").add(pl.col("weapon_id")).alias("weapon_id"),
            pl.col("sub").replace(subs),
            pl.col("special").replace(specials),
            pl.col("class_").replace(classes).alias("class"),
        )
        .drop(["class_", "reference_kit", "reference_id", "season", "kit"])
    )


def get_weapon_properties_for_id(
    weapon_id: int | str,
) -> WeaponProperties:
    """Get properties of a weapon."""
    weapon_ref = get_weapon_properties()
    if str(weapon_id).startswith("weapon_id_"):
        weapon_id = int(weapon_id[len("weapon_id_") :])
    return weapon_ref[str(weapon_id)]
