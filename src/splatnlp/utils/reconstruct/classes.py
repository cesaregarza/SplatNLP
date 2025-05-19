import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import ClassVar

from splatnlp.utils.constants import (
    CANONICAL_MAIN_ONLY_ABILITIES,
    CLOTHING_ABILITIES,
    HEADGEAR_ABILITIES,
    MAIN_ONLY_ABILITIES,
    SHOES_ABILITIES,
    STANDARD_ABILITIES,
)
from splatnlp.viz.cluster_labels import abbrev

logger = logging.getLogger(__name__)
_AP_SUFFIX_RE = re.compile(r"_(\d+)$")


@dataclass(frozen=True)
class AbilityToken:
    name: str
    family: str
    min_ap: int
    main_only: bool

    def __repr__(self) -> str:
        return (
            f"{abbrev(self.name)}({self.min_ap})"
            if not self.main_only
            else self.name
        )

    @staticmethod
    def from_vocab_entry(token: str) -> "AbilityToken":
        if token in MAIN_ONLY_ABILITIES:
            # Main-only abilities inherently provide 10 AP when on a main slot.
            # Their min_ap in the token reflects this fixed value.
            return AbilityToken(
                name=token, family=token, min_ap=10, main_only=True
            )
        m = _AP_SUFFIX_RE.search(token)
        if not m:
            raise ValueError(
                f"Token '{token}' not recognised as STANDARD or MAIN-ONLY."
            )
        min_ap = int(m.group(1))
        family = token[: m.start()]
        return AbilityToken(
            name=token, family=family, min_ap=min_ap, main_only=False
        )


@dataclass
class Build:
    """
    Represents a concrete, legal gear build.
    The 'mains' dictionary must always contain 'head', 'clothes', and 'shoes' as keys.
    The value for a main slot can be an ability family string or None if empty.
    """

    # Class constants for validation
    GEAR_SLOTS: ClassVar[tuple[str, ...]] = ("head", "clothes", "shoes")
    MAX_MAIN_SLOTS: ClassVar[int] = 3
    MAX_SUB_SLOTS_TOTAL: ClassVar[int] = 9
    MAX_TOTAL_AP: ClassVar[int] = 57  # 3 mains * 10 AP + 9 subs * 3 AP

    mains: dict[str, str | None]  # Ability family or None
    subs: dict[str, int]  # Key: ability family, Value: count of subs

    total_ap: int = 0  # Will be calculated in __post_init__

    def __post_init__(self):
        """
        Calculates total AP and validates the build upon instantiation.
        Raises ValueError if the build is invalid.
        """
        # Ensure mains dict structure is as expected before calculating achieved_ap
        if (
            not all(slot in self.mains for slot in self.GEAR_SLOTS)
            or len(self.mains) != self.MAX_MAIN_SLOTS
        ):
            raise ValueError(
                f"Mains dictionary must contain exactly {self.MAX_MAIN_SLOTS} keys: "
                f"{', '.join(self.GEAR_SLOTS)}. Got: {self.mains.keys()}"
            )

        # Calculate achieved_ap based on the current mains and subs
        current_achieved_ap = self._calculate_achieved_ap()
        self.total_ap = sum(current_achieved_ap.values())

        logger.debug(
            f"New build attempt: Mains={self.mains}, Subs={self.subs}, "
            f"AchievedAP={current_achieved_ap}, TotalAP={self.total_ap}"
        )

        if not self._validate():
            # Log details of why validation failed before raising error
            # This could be expanded with more specific error messages from _validate
            logger.error(
                f"Invalid build configuration: Mains={self.mains}, Subs={self.subs}"
            )
            raise ValueError("Invalid build configuration based on game rules.")

    def _calculate_achieved_ap(self) -> dict[str, int]:
        """Helper method to calculate achieved AP based on current mains and subs."""
        out: dict[str, int] = {}
        for ability_family_on_main in self.mains.values():
            if ability_family_on_main is not None:
                out[ability_family_on_main] = (
                    out.get(ability_family_on_main, 0) + 10
                )

        for sub_ability_family, count in self.subs.items():
            if count > 0:  # Only consider abilities actually present in subs
                out[sub_ability_family] = out.get(sub_ability_family, 0) + (
                    count * 3
                )
        return out

    @property
    def achieved_ap(self) -> dict[str, int]:
        """
        Provides a dynamically calculated dictionary of achieved AP per ability family.
        This ensures it's always up-to-date if Build were mutable (though it's not designed to be post-init).
        """
        return self._calculate_achieved_ap()

    def _validate(self) -> bool:
        """
        Validates the build against game rules.
        Assumes self.total_ap and current_achieved_ap have been calculated.
        """
        # 1. Check total AP limit
        if self.total_ap > self.MAX_TOTAL_AP:
            logger.debug(
                f"Validation failed: Total AP {self.total_ap} exceeds {self.MAX_TOTAL_AP}."
            )
            return False

        # 2. Check main slot assignments and main-only ability rules
        # (Already checked for structure in __post_init__)
        for gear_slot, assigned_ability_family in self.mains.items():
            if assigned_ability_family is None:
                continue  # Empty main slot is valid

            if assigned_ability_family in MAIN_ONLY_ABILITIES:
                expected_slot_for_this_ability = (
                    CANONICAL_MAIN_ONLY_ABILITIES.get(assigned_ability_family)
                )
                if expected_slot_for_this_ability is None:
                    logger.debug(
                        f"Validation failed: Main-only ability {assigned_ability_family} "
                        "has no canonical slot defined."
                    )
                    return False
                if expected_slot_for_this_ability != gear_slot:
                    logger.debug(
                        f"Validation failed: Main-only ability {assigned_ability_family} "
                        f"is on {gear_slot} but expected on {expected_slot_for_this_ability}."
                    )
                    return False
            # No explicit check here for standard abilities on wrong main slots,
            # as standard abilities can go on any main slot.

        # 3. Check total number of sub-slots used
        if sum(self.subs.values()) > self.MAX_SUB_SLOTS_TOTAL:
            logger.debug(
                f"Validation failed: Total sub-slots {sum(self.subs.values())} "
                f"exceeds {self.MAX_SUB_SLOTS_TOTAL}."
            )
            return False

        # 4. Ensure no negative sub counts (though dataclass type hints should help)
        if any(count < 0 for count in self.subs.values()):
            logger.debug("Validation failed: Negative sub-slot count found.")
            return False

        return True

    def disallowed_abilities(self) -> list[str]:
        """
        Determines which abilities cannot be added to this build.
        Used by the beam search to prune next possible tokens.
        """
        disallowed: list[str] = []

        # Main-only family exclusivity: if a main slot has a main-only ability,
        # other main-only abilities for that SAME SLOT TYPE are disallowed.
        active_main_only_slots: dict[str, str] = {}  # e.g. {"head": "comeback"}
        for gear_type, ability_family in self.mains.items():
            if ability_family in MAIN_ONLY_ABILITIES:
                active_main_only_slots[gear_type] = ability_family

        if "head" in active_main_only_slots:
            disallowed.extend(
                x
                for x in HEADGEAR_ABILITIES
                if x != active_main_only_slots["head"]
            )
        if "clothes" in active_main_only_slots:
            disallowed.extend(
                x
                for x in CLOTHING_ABILITIES
                if x != active_main_only_slots["clothes"]
            )
        if "shoes" in active_main_only_slots:
            disallowed.extend(
                x
                for x in SHOES_ABILITIES
                if x != active_main_only_slots["shoes"]
            )

        # If all 3 main slots are filled by main-only abilities,
        # then ALL other main-only abilities are disallowed.
        if len(active_main_only_slots) == self.MAX_MAIN_SLOTS:
            all_placed_main_onlys = set(active_main_only_slots.values())
            disallowed.extend(
                x for x in MAIN_ONLY_ABILITIES if x not in all_placed_main_onlys
            )

        # When gear is already "full" in terms of AP potential (3 mains + 9 subs = 57 AP),
        # no more abilities can be meaningfully added to contribute AP.
        # This check is more about AP capacity.
        # A simpler check: if all main slots are non-None and total subs are 9.
        mains_full = all(
            self.mains[slot] is not None for slot in self.GEAR_SLOTS
        )
        subs_full = sum(self.subs.values()) == self.MAX_SUB_SLOTS_TOTAL

        if mains_full and subs_full:
            logger.debug(
                "Build is full (3 mains, 9 subs). All other abilities disallowed."
            )
            # Disallow all abilities not already part of the build's AP contribution.
            # This is a bit broad; beam search might still want to consider higher AP versions
            # of existing standard abilities if that's a separate mechanism.
            # For now, let's assume if full, nothing new.
            current_families_in_build = set(self.achieved_ap.keys())
            disallowed.extend(
                token_family
                for token_family in (*MAIN_ONLY_ABILITIES, *STANDARD_ABILITIES)
                if token_family not in current_families_in_build
            )

        logger.debug(
            f"Disallowed for next step: {', '.join(sorted(list(set(disallowed))))}"
        )
        return list(set(disallowed))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Build):
            return False

        # Check main-only abilities match exactly in their slots
        for slot, value in self.mains.items():
            if value in MAIN_ONLY_ABILITIES and value != other.mains[slot]:
                return False

        # Compare main ability counts
        if Counter(self.mains.values()) != Counter(other.mains.values()):
            return False

        # Compare sub ability counts
        return self.subs == other.subs
