"""Domain glossary and constraints for mechanistic interpretability.

This module defines the domain-specific knowledge needed for valid
experiment design, including ability families, AP rungs, and constraints.
"""

from dataclasses import dataclass
from typing import Literal

from splatnlp.utils.constants import (
    BUCKET_THRESHOLDS,
    CANONICAL_MAIN_ONLY_ABILITIES,
    MAIN_ONLY_ABILITIES,
    STANDARD_ABILITIES,
)

# Standard AP rungs used in tokenization
# These represent the cumulative ability points from gear slots
STANDARD_RUNGS = [3, 6, 12, 21, 29, 35, 41, 45, 48, 51, 54, 57]

# Extended rungs including some intermediate values from BUCKET_THRESHOLDS
EXTENDED_RUNGS = sorted(set(BUCKET_THRESHOLDS + STANDARD_RUNGS))

# Main-only abilities have fixed 10 AP (one main slot only)
MAIN_ONLY_AP = 10


@dataclass(frozen=True)
class TokenFamily:
    """Definition of an ability token family.

    A token family groups related tokens that represent different
    AP levels of the same ability (e.g., swim_speed_up_3, swim_speed_up_6, ...).
    """

    name: str  # Full name (e.g., "swim_speed_up")
    short_code: str  # Abbreviation (e.g., "SSU")
    is_main_only: bool = False  # True for abilities like ninja_squid
    gear_slot: Literal["head", "clothes", "shoes"] | None = (
        None  # For main-only
    )
    rungs: tuple[int, ...] = tuple(STANDARD_RUNGS)  # Available AP values

    def token_at_rung(self, rung: int) -> str:
        """Get the token name for a specific AP rung."""
        if self.is_main_only:
            return self.name  # Main-only have no AP suffix
        return f"{self.name}_{rung}"

    def all_tokens(self) -> list[str]:
        """Get all possible tokens for this family."""
        if self.is_main_only:
            return [self.name]
        return [f"{self.name}_{r}" for r in self.rungs]


# Standard stackable abilities with their short codes
_STANDARD_FAMILY_CODES = {
    "ink_recovery_up": "IRU",
    "ink_resistance_up": "INKR",
    "ink_saver_main": "ISM",
    "ink_saver_sub": "ISS",
    "intensify_action": "IA",
    "quick_respawn": "QR",
    "quick_super_jump": "QSJ",
    "run_speed_up": "RSU",
    "special_charge_up": "SCU",
    "special_power_up": "SPU",
    "special_saver": "SS",
    "sub_power_up": "SubPU",
    "sub_resistance_up": "SRU",
    "swim_speed_up": "SSU",
}

# Build TOKEN_FAMILIES dict
TOKEN_FAMILIES: dict[str, TokenFamily] = {}

# Add standard stackable abilities
for name in STANDARD_ABILITIES:
    code = _STANDARD_FAMILY_CODES.get(name, name.upper()[:3])
    TOKEN_FAMILIES[name] = TokenFamily(
        name=name,
        short_code=code,
        is_main_only=False,
        rungs=tuple(STANDARD_RUNGS),
    )

# Add main-only abilities
for name in MAIN_ONLY_ABILITIES:
    slot = CANONICAL_MAIN_ONLY_ABILITIES.get(name)
    code = "".join(word[0].upper() for word in name.split("_"))
    TOKEN_FAMILIES[name] = TokenFamily(
        name=name,
        short_code=code,
        is_main_only=True,
        gear_slot=slot,
        rungs=(MAIN_ONLY_AP,),
    )

# Convenience aliases using short codes
TOKEN_FAMILIES_BY_CODE: dict[str, TokenFamily] = {
    f.short_code: f for f in TOKEN_FAMILIES.values()
}


@dataclass(frozen=True)
class DomainConstraint:
    """A constraint that experiments must respect.

    Constraints encode domain knowledge about valid experimental
    manipulations to prevent confounded or invalid results.
    """

    id: str  # Unique identifier
    description: str  # Human-readable explanation
    enforcement: Literal["hard", "soft"]  # hard=error, soft=warning
    rationale: str = ""  # Why this constraint exists


# Domain constraints that experiments should enforce
DOMAIN_CONSTRAINTS: dict[str, DomainConstraint] = {
    "one_rung_per_family": DomainConstraint(
        id="one_rung_per_family",
        description="Only one AP rung per ability family can appear in a build",
        enforcement="hard",
        rationale=(
            "In Splatoon, gear slots contribute AP additively. A build cannot "
            "have both swim_speed_up_21 and swim_speed_up_12 simultaneously - "
            "they represent different total AP allocations for the same ability."
        ),
    ),
    "no_weapon_gating_if_relu_floor": DomainConstraint(
        id="no_weapon_gating_if_relu_floor",
        description=(
            "Skip weapon-specific experiments when base activation is near zero"
        ),
        enforcement="soft",
        rationale=(
            "When base activation is at or near the ReLU floor, activation "
            "deltas become quantized and unreliable. Weapon gating analysis "
            "requires sufficient base activation to detect modulation effects."
        ),
    ),
    "valid_gear_combinations": DomainConstraint(
        id="valid_gear_combinations",
        description="Main-only abilities can only appear on their designated gear slot",
        enforcement="hard",
        rationale=(
            "Main-only abilities like Ninja Squid (clothes), Drop Roller (shoes), "
            "or Comeback (head) can only be equipped in specific slots. Creating "
            "builds with invalid slot assignments produces OOD inputs."
        ),
    ),
    "max_ap_budget": DomainConstraint(
        id="max_ap_budget",
        description="Total AP across all abilities should not exceed 57 (realistic max)",
        enforcement="soft",
        rationale=(
            "A full gear set provides 3 main slots (10 AP each) + 9 sub slots "
            "(3 AP each) = 57 AP maximum. Builds exceeding this are unrealistic."
        ),
    ),
    "avoid_ood_token_combinations": DomainConstraint(
        id="avoid_ood_token_combinations",
        description="Avoid token combinations that never appear in training data",
        enforcement="soft",
        rationale=(
            "Some ability combinations are extremely rare or impossible in "
            "practice (e.g., maximum investment in 5+ different abilities). "
            "Testing such combinations may produce unreliable activations."
        ),
    ),
}


def get_family_for_token(token: str) -> TokenFamily | None:
    """Get the TokenFamily for a given token string.

    Args:
        token: Token string like "swim_speed_up_21" or "ninja_squid"

    Returns:
        TokenFamily if found, None otherwise
    """
    # Check if it's a main-only ability (exact match)
    if token in TOKEN_FAMILIES:
        family = TOKEN_FAMILIES[token]
        if family.is_main_only:
            return family

    # Try to parse as standard ability with AP suffix
    import re

    match = re.match(r"^([a-z_]+?)_(\d+)$", token)
    if match:
        family_name = match.group(1)
        if family_name in TOKEN_FAMILIES:
            return TOKEN_FAMILIES[family_name]

    return None


def parse_token(token: str) -> tuple[str, int | None]:
    """Parse a token into (family_name, ap_value).

    Args:
        token: Token string like "swim_speed_up_21" or "ninja_squid"

    Returns:
        Tuple of (family_name, ap_value or None for main-only)

    Examples:
        >>> parse_token("swim_speed_up_21")
        ("swim_speed_up", 21)
        >>> parse_token("ninja_squid")
        ("ninja_squid", None)
    """
    import re

    # Check main-only first
    if token in MAIN_ONLY_ABILITIES:
        return token, None

    # Try standard pattern
    match = re.match(r"^([a-z_]+?)_(\d+)$", token)
    if match:
        return match.group(1), int(match.group(2))

    # Unknown format
    return token, None


def validate_build_tokens(tokens: list[str]) -> list[str]:
    """Validate a list of build tokens against domain constraints.

    Args:
        tokens: List of token strings

    Returns:
        List of constraint violation messages (empty if valid)
    """
    violations = []
    family_rungs: dict[str, list[int]] = {}

    for token in tokens:
        family_name, ap = parse_token(token)
        if family_name not in TOKEN_FAMILIES:
            continue

        if family_name not in family_rungs:
            family_rungs[family_name] = []

        if ap is not None:
            family_rungs[family_name].append(ap)

    # Check one-rung-per-family
    for family, rungs in family_rungs.items():
        if len(rungs) > 1:
            violations.append(
                f"Constraint 'one_rung_per_family' violated: "
                f"{family} has multiple rungs: {rungs}"
            )

    return violations


def get_glossary_text() -> str:
    """Generate a human-readable glossary for the Skills."""
    lines = [
        "# MechInterp Domain Glossary",
        "",
        "## Ability Families",
        "",
        "### Standard Stackable Abilities",
        "These abilities can have varying AP levels based on gear investment:",
        "",
        "| Short | Full Name | AP Rungs |",
        "|-------|-----------|----------|",
    ]

    for name in STANDARD_ABILITIES:
        family = TOKEN_FAMILIES[name]
        rungs_str = ", ".join(str(r) for r in family.rungs[:5]) + "..."
        lines.append(f"| {family.short_code} | {name} | {rungs_str} |")

    lines.extend(
        [
            "",
            "### Main-Only Abilities",
            "These abilities occupy a main slot and have fixed 10 AP:",
            "",
            "| Short | Full Name | Gear Slot |",
            "|-------|-----------|-----------|",
        ]
    )

    for name in MAIN_ONLY_ABILITIES:
        family = TOKEN_FAMILIES[name]
        lines.append(f"| {family.short_code} | {name} | {family.gear_slot} |")

    lines.extend(
        [
            "",
            "## AP Rung Meanings",
            "",
            "Standard rungs represent cumulative AP from gear:",
            "- 3 AP: 1 sub slot",
            "- 6 AP: 2 sub slots",
            "- 12 AP: 4 sub slots OR 1 main + 0-1 subs",
            "- 21 AP: 1 main + 3-4 subs",
            "- 29 AP: 1 main + 6 subs OR 2 mains + 3 subs",
            "- 57 AP: Maximum (3 mains + 9 subs all same ability)",
            "",
            "## Constraints",
            "",
        ]
    )

    for c in DOMAIN_CONSTRAINTS.values():
        lines.append(f"### {c.id}")
        lines.append(f"**{c.description}**")
        lines.append(f"- Enforcement: {c.enforcement}")
        lines.append(f"- Rationale: {c.rationale}")
        lines.append("")

    return "\n".join(lines)
