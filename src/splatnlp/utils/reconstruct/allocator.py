"""Allocate predicted abilities to a legal Splatoon gear build.

This module defines :class:`Allocator` for the reconstruction pipeline. It
searches possible assignments of capstone abilities to gear slots, choosing
between main and sub slots while respecting game rules. The allocator
minimises total Ability Points (AP) while meeting each requested ability's
minimum requirement.

See ``README.md`` in this directory for a full description of the strategy.
"""

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
    This version can assign multiple main slots to a single standard ability if
    optimal. Now also returns the penalty score (actual_ap - required_ap for
    each token).
    """

    def __init__(
        self, main_only_slots_override: Optional[Mapping[str, str]] = None
    ):
        self._main_only_slots_map: dict[str, str] = dict(
            CANONICAL_MAIN_ONLY_ABILITIES
        )
        if main_only_slots_override:
            self._main_only_slots_map.update(main_only_slots_override)

        self._memo_recursive_assign: dict[Any, None] = {}

    def _solve_standard_mains_recursively(
        self,
        standard_capstones: list[AbilityToken],
        cap_idx: int,
        available_main_slots: tuple[str, ...],
        current_mains_config: dict[str, Optional[str]],
        best_build_info: list[
            Optional[
                tuple[
                    int,
                    float,  # priority score
                    dict[str, Optional[str]],
                    dict[str, int],
                    int,
                ]
            ]
        ],
        capstone_family_to_token: dict[str, AbilityToken],
        priority: dict[str, float],
    ):
        """
        Recursive helper to find the optimal assignment of standard abilities to
        main slots. Modifies best_build_info[0] if a better valid build is
        found.

        Parameters
        ----------
        standard_capstones : list[AbilityToken]
            List of standard ability tokens to consider for main slots
        cap_idx : int
            Current index in standard_capstones
        available_main_slots : tuple[str, ...]
            List of main slots still available
        current_mains_config : dict[str, Optional[str]]
            Current assignment of abilities to main slots
        best_build_info : list[Optional[tuple]]
            Container for the best build found so far
        capstone_family_to_token : dict[str, AbilityToken]
            Mapping from family names to their tokens
        priority : dict[str, float]
            Mapping from family names to their priority scores (higher is better)
        """
        state_key = (
            cap_idx,
            available_main_slots,
            frozenset(current_mains_config.items()),
        )
        if state_key in self._memo_recursive_assign:
            return

        # Base Case: All standard capstones have been considered for main slot
        # assignment
        if cap_idx == len(standard_capstones):
            # Calculate AP provided by all abilities on main slots
            ap_from_all_mains: dict[str, int] = {}
            for ability_family in current_mains_config.values():
                if ability_family is not None:
                    ap_from_all_mains[ability_family] = (
                        ap_from_all_mains.get(ability_family, 0) + 10
                    )

            subs_needed_for_all_std_caps: dict[str, int] = {}
            current_total_sub_slots_used = 0
            all_min_ap_requirements_met = True

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
                self._memo_recursive_assign[state_key] = None
                return

            # Try to construct and validate the build
            try:
                final_subs_map = {
                    fam: count
                    for fam, count in subs_needed_for_all_std_caps.items()
                    if count > 0
                }
                candidate_build = Build(
                    mains=dict(current_mains_config), subs=final_subs_map
                )

                # Compute total penalty (sum of actual_ap - required_ap for all
                # tokens)
                total_penalty = 0
                # For all capstones (main-only and standard)
                for fam, token in capstone_family_to_token.items():
                    # Compute actual_ap for this family in this build
                    ap_from_mains = 0
                    for slot, slot_fam in current_mains_config.items():
                        if slot_fam == fam:
                            ap_from_mains += 10
                    ap_from_subs = final_subs_map.get(fam, 0) * 3
                    actual_ap = ap_from_mains + ap_from_subs
                    total_penalty += actual_ap - token.min_ap

                # Calculate priority score for main slots
                priority_score = sum(
                    priority.get(fam, 0.0)
                    for fam in current_mains_config.values()
                    if fam is not None
                )

                if (
                    best_build_info[0] is None
                    or candidate_build.total_ap < best_build_info[0][0]
                    or (
                        candidate_build.total_ap == best_build_info[0][0]
                        and priority_score > best_build_info[0][1]
                    )
                ):
                    best_build_info[0] = (
                        candidate_build.total_ap,
                        priority_score,
                        dict(current_mains_config),
                        final_subs_map,
                        total_penalty,
                    )
            except ValueError:
                pass  # Build validation failed

            self._memo_recursive_assign[state_key] = None
            return

        # Recursive Step: Consider current standard_capstones[cap_idx]
        current_std_token = standard_capstones[cap_idx]

        # Option 1: Assign 0 main slots to current_std_token
        self._solve_standard_mains_recursively(
            standard_capstones,
            cap_idx + 1,
            available_main_slots,
            current_mains_config,
            best_build_info,
            capstone_family_to_token,
            priority,
        )

        # Option 2: Try assigning 1, 2, or 3 main slots to current_std_token
        max_mains_to_attempt_for_token = min(len(available_main_slots), 3)

        for num_mains_to_assign_to_token in range(
            1, max_mains_to_attempt_for_token + 1
        ):
            for main_slots_combo_for_token in combinations(
                available_main_slots, num_mains_to_assign_to_token
            ):
                next_mains_config = dict(current_mains_config)
                for slot in main_slots_combo_for_token:
                    next_mains_config[slot] = current_std_token.family

                remaining_available_slots_list = list(
                    set(available_main_slots) - set(main_slots_combo_for_token)
                )

                self._solve_standard_mains_recursively(
                    standard_capstones,
                    cap_idx + 1,
                    tuple(sorted(remaining_available_slots_list)),
                    next_mains_config,
                    best_build_info,
                    capstone_family_to_token,
                    priority,
                )

        self._memo_recursive_assign[state_key] = None

    def allocate(
        self,
        capstones: Mapping[str, AbilityToken],
        priority: dict[str, float] | None = None,
    ) -> Optional[tuple[Build, int]]:
        """
        Allocate abilities to gear slots, considering priorities for main slots.

        Parameters
        ----------
        capstones : Mapping[str, AbilityToken]
            Mapping from token strings to their AbilityToken objects
        priority : dict[str, float] | None
            Optional mapping from family names to priority scores (higher is
            better). Used to break ties when multiple valid builds have the same
            total AP.

        Returns
        -------
        Optional[tuple[Build, int]]
            The best valid build found and its penalty score, or (None, None) if
            no valid build could be formed
        """
        self._memo_recursive_assign.clear()
        if priority is None:
            priority = {}

        main_only_capstones: list[AbilityToken] = []
        standard_capstones_dict: dict[str, AbilityToken] = {}

        # For penalty calculation, we need a mapping from family to the "most
        # demanding" token
        capstone_family_to_token: dict[str, AbilityToken] = {}

        for token in capstones.values():
            if token.main_only:
                main_only_capstones.append(token)
                capstone_family_to_token[token.family] = token
            else:
                if (
                    token.family not in standard_capstones_dict
                    or token.min_ap
                    > standard_capstones_dict[token.family].min_ap
                ):
                    standard_capstones_dict[token.family] = token
                # For penalty, always keep the highest min_ap token for each
                # family
                if (
                    token.family not in capstone_family_to_token
                    or token.min_ap
                    > capstone_family_to_token[token.family].min_ap
                ):
                    capstone_family_to_token[token.family] = token

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
                return None, None
            if initial_mains_config[slot] is not None:
                logger.debug(
                    f"Slot conflict for main-only ability {token.name} on slot "
                    f"{slot}."
                )
                return None, None
            initial_mains_config[slot] = token.family

        # List of physical main gear slots still available for standard
        # abilities
        free_main_slots_list: list[str] = [
            s for s, fam in initial_mains_config.items() if fam is None
        ]

        # best_build_info is a list containing one item: the best tuple or None
        # Structure: [ Optional[
        # (total_ap, priority_score, mains_dict, subs_dict, penalty_per_token)
        # ] ]
        best_build_info_container: list[
            Optional[
                tuple[
                    int,
                    float,
                    dict[str, Optional[str]],
                    dict[str, int],
                    int,
                ]
            ]
        ] = [None]

        # Start the recursive process for standard abilities
        self._solve_standard_mains_recursively(
            standard_capstones_list,
            0,
            tuple(sorted(free_main_slots_list)),
            initial_mains_config,
            best_build_info_container,
            capstone_family_to_token,
            priority,
        )

        # If a best build was found by the recursive solver
        if best_build_info_container[0] is not None:
            _total_ap, _priority_score, best_mains, best_subs, total_penalty = (
                best_build_info_container[0]
            )
            try:
                final_mains_for_build = {
                    slot: best_mains.get(slot) for slot in Build.GEAR_SLOTS
                }
                build = Build(mains=final_mains_for_build, subs=best_subs)
                return build, total_penalty
            except ValueError as e:
                logger.error(
                    "Failed to reconstruct previously validated best build. "
                    f"Error: {e}. Mains: {best_mains}, Subs: {best_subs}"
                )
                return None, None

        logger.debug(
            "No valid build configuration found for the given capstones."
        )
        return None, None
