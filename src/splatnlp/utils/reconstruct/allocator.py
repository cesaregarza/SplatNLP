import logging
import math
from itertools import combinations
from typing import Any, Mapping, Optional

from splatnlp.utils.constants import CANONICAL_MAIN_ONLY_ABILITIES
from splatnlp.utils.reconstruct.classes import AbilityToken, Build

logger = logging.getLogger(__name__)


class Allocator:
    """
    Allocates a set of capstone abilities to a Splatoon gear build.
    The goal is to satisfy all min_ap requirements of the capstones
    while minimizing the total AP of the resulting build (i.e., "tightest fit").
    This version can assign multiple main slots to a single standard ability if optimal.
    """

    def __init__(
        self, main_only_slots_override: Optional[Mapping[str, str]] = None
    ):
        self._main_only_slots_map: dict[str, str] = dict(
            CANONICAL_MAIN_ONLY_ABILITIES
        )
        if main_only_slots_override:
            self._main_only_slots_map.update(main_only_slots_override)

        # Memoization cache for the recursive solver
        self._memo_recursive_assign: dict[Any, None] = (
            {}
        )  # Key: state tuple, Value: (not used, just for presence)

    def _solve_standard_mains_recursively(
        self,
        standard_capstones: list[
            AbilityToken
        ],  # The list of all standard capstones to process
        cap_idx: int,  # Current index in standard_capstones being processed
        available_main_slots: tuple[
            str, ...
        ],  # Tuple of available physical main slot names (e.g., ('clothes', 'shoes'))
        current_mains_config: dict[
            str, Optional[str]
        ],  # Accumulates main assignments (head:X, clothes:Y, shoes:Z)
        best_build_info: list[
            Optional[tuple[int, dict[str, Optional[str]], dict[str, int]]]
        ],  # Mutable list to store the best result
    ):
        """
        Recursive helper to find the optimal assignment of standard abilities to main slots.
        Modifies best_build_info[0] if a better valid build is found.
        """
        state_key = (
            cap_idx,
            available_main_slots,  # Already sorted tuple
            frozenset(
                current_mains_config.items()
            ),  # Represents assignments so far
        )
        if state_key in self._memo_recursive_assign:
            return

        # Base Case: All standard capstones have been considered for main slot assignment
        if cap_idx == len(standard_capstones):
            # current_mains_config now reflects a complete assignment hypothesis for mains
            # (including main-onlys from initial setup and standard abilities from recursion)

            # Calculate AP provided by all abilities on main slots
            ap_from_all_mains: dict[str, int] = {}  # family -> AP from mains
            for ability_family in current_mains_config.values():
                if ability_family is not None:
                    ap_from_all_mains[ability_family] = (
                        ap_from_all_mains.get(ability_family, 0) + 10
                    )

            subs_needed_for_all_std_caps: dict[str, int] = {}
            current_total_sub_slots_used = 0
            all_min_ap_requirements_met = True

            # Check main-only capstones (already included in ap_from_all_mains if placed)
            # Their min_ap is 10, which is met if they are on a main.

            # Calculate subs for standard capstones and check if min_ap is met
            for std_token in standard_capstones:
                ap_from_mains_for_this_token = ap_from_all_mains.get(
                    std_token.family, 0
                )

                remaining_ap_needed = (
                    std_token.min_ap - ap_from_mains_for_this_token
                )
                num_subs_for_this_token = 0
                if remaining_ap_needed > 0:
                    num_subs_for_this_token = math.ceil(
                        remaining_ap_needed / 3.0
                    )

                if num_subs_for_this_token > 0:
                    subs_needed_for_all_std_caps[std_token.family] = (
                        subs_needed_for_all_std_caps.get(std_token.family, 0)
                        + num_subs_for_this_token
                    )
                    current_total_sub_slots_used += num_subs_for_this_token

                # Check if this specific std_token's min_ap is met
                if (
                    ap_from_mains_for_this_token + (num_subs_for_this_token * 3)
                    < std_token.min_ap
                ):
                    all_min_ap_requirements_met = False
                    break

            if (
                not all_min_ap_requirements_met
                or current_total_sub_slots_used > Build.MAX_SUB_SLOTS_TOTAL
            ):
                self._memo_recursive_assign[state_key] = (
                    None  # Mark as processed
                )
                return

            # Try to construct and validate the build
            try:
                final_subs_map = {
                    fam: count
                    for fam, count in subs_needed_for_all_std_caps.items()
                    if count > 0
                }
                # current_mains_config is the complete mains picture for this path
                candidate_build = Build(
                    mains=dict(current_mains_config), subs=final_subs_map
                )  # Use a copy

                if (
                    best_build_info[0] is None
                    or candidate_build.total_ap < best_build_info[0][0]
                ):  # type: ignore
                    best_build_info[0] = (
                        candidate_build.total_ap,
                        dict(current_mains_config),
                        final_subs_map,
                    )
            except ValueError:
                pass  # Build validation failed (e.g., total AP > 57, main-only on wrong slot from initial)

            self._memo_recursive_assign[state_key] = None  # Mark as processed
            return

        # Recursive Step: Consider current standard_capstones[cap_idx]
        current_std_token = standard_capstones[cap_idx]

        # Option 1: Assign 0 main slots to current_std_token
        # Pass current_mains_config as is, since no mains are used by this token here
        self._solve_standard_mains_recursively(
            standard_capstones,
            cap_idx + 1,
            available_main_slots,
            current_mains_config,
            best_build_info,
        )

        # Option 2: Try assigning 1, 2, or 3 main slots to current_std_token
        # Max mains one ability family would ever "sensibly" take is related to its min_ap.
        # For min_ap=20, 2 mains is sensible. For min_ap=10, 1 main.
        # We can try up to min(len(available_main_slots), 3)
        max_mains_to_attempt_for_token = min(len(available_main_slots), 3)

        for num_mains_to_assign_to_token in range(
            1, max_mains_to_attempt_for_token + 1
        ):
            # Choose 'num_mains_to_assign_to_token' slots from 'available_main_slots'
            for main_slots_combo_for_token in combinations(
                available_main_slots, num_mains_to_assign_to_token
            ):
                # Create a new mains_config for this path
                next_mains_config = dict(current_mains_config)

                # Assign the current standard token's family to these chosen main slots
                for slot in main_slots_combo_for_token:
                    next_mains_config[slot] = (
                        current_std_token.family
                    )  # Overwrites if slot was None

                remaining_available_slots_list = list(
                    set(available_main_slots) - set(main_slots_combo_for_token)
                )

                self._solve_standard_mains_recursively(
                    standard_capstones,
                    cap_idx + 1,  # Move to the next standard capstone
                    tuple(
                        sorted(remaining_available_slots_list)
                    ),  # New set of available slots
                    next_mains_config,  # Updated mains configuration
                    best_build_info,
                )

        self._memo_recursive_assign[state_key] = None  # Mark as processed

    def allocate(
        self, capstones: Mapping[str, AbilityToken]
    ) -> Optional[Build]:
        self._memo_recursive_assign.clear()  # Clear memo for each new allocation call

        main_only_capstones: list[AbilityToken] = []
        standard_capstones_dict: dict[str, AbilityToken] = {}

        for token in capstones.values():
            if token.main_only:
                main_only_capstones.append(token)
            else:
                if (
                    token.family not in standard_capstones_dict
                    or token.min_ap
                    > standard_capstones_dict[token.family].min_ap
                ):
                    standard_capstones_dict[token.family] = token

        standard_capstones_list: list[AbilityToken] = list(
            standard_capstones_dict.values()
        )

        # Initial mains configuration with only Nones
        initial_mains_config: dict[str, Optional[str]] = {
            slot: None for slot in Build.GEAR_SLOTS
        }

        # Phase 0: Place main-only abilities
        for token in main_only_capstones:
            slot = self._main_only_slots_map.get(token.family)
            if slot is None:
                logger.debug(
                    f"Main-only capstone {token.name} has no defined slot."
                )
                return None
            if initial_mains_config[slot] is not None:
                logger.debug(
                    f"Slot conflict for main-only ability {token.name} on slot {slot}."
                )
                return None
            initial_mains_config[slot] = token.family

        # List of physical main gear slots still available for standard abilities
        free_main_slots_list: list[str] = [
            s for s, fam in initial_mains_config.items() if fam is None
        ]

        # best_build_info is a list containing one item: the best tuple or None
        # This allows the recursive helper to modify it.
        # Structure: [ Optional[(total_ap, mains_dict, subs_dict)] ]
        best_build_info_container: list[
            Optional[tuple[int, dict[str, Optional[str]], dict[str, int]]]
        ] = [None]

        # Start the recursive process for standard abilities
        # The initial call passes the mains config which already has main-onlys placed.
        self._solve_standard_mains_recursively(
            standard_capstones_list,
            0,  # Start with the first standard capstone
            tuple(
                sorted(free_main_slots_list)
            ),  # Available slots for standard abilities
            initial_mains_config,  # Mains config with main-onlys placed
            best_build_info_container,
        )

        # If a best build was found by the recursive solver
        if best_build_info_container[0] is not None:
            _total_ap, best_mains, best_subs = best_build_info_container[0]
            try:
                # Ensure the final mains dict has all gear slots, even if some are None
                final_mains_for_build = {
                    slot: best_mains.get(slot) for slot in Build.GEAR_SLOTS
                }
                return Build(mains=final_mains_for_build, subs=best_subs)
            except ValueError as e:
                logger.error(
                    f"Failed to reconstruct previously validated best build. Error: {e}. Mains: {best_mains}, Subs: {best_subs}"
                )
                return None

        logger.debug(
            "No valid build configuration found for the given capstones."
        )
        return None
