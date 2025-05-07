"""
This module is designed to reconstruct optimal and legal Splatoon-style gear builds
based on a predictive model and defined game mechanics. The primary function,
`reconstruct_build`, employs a beam search algorithm to explore combinations of
"capstone" abilities. An "Allocator" class then attempts to materialize these
capstone sets into concrete, valid gear configurations.

Core Requirements and Logic:

1.  **Goal**:
    To determine the single best, legal gear build for a given weapon,
    potentially influenced by an initial set of abilities (context). A "build"
    consists of three gear pieces (head, clothes, shoes), each with one main
    ability and up to three sub-ability slots (total 9 sub-slots).

2.  **Input Components**:
    -   `predict` function: A model that, given a current sequence of abilities
        (context) and a weapon ID, predicts scores or probabilities for potential
        next abilities.
    -   `weapon_id`: Identifier for the weapon for which the build is generated.
    -   `initial_context`: A sequence of ability names to start the build process.
    -   `vocab_tokens`: A comprehensive list of all possible ability tokens recognized
        by the system (e.g., "swim_speed_up_3", "stealth_jump").
    -   Game Constants: Imported definitions for main-only abilities, abilities
        exclusive to certain gear types, maximum sub-slots, etc.

3.  **Beam Search (`reconstruct_build`)**:
    -   The system iteratively builds up a set of "capstone" abilities using beam search.
    -   A capstone `AbilityToken` represents a target ability. For standard abilities
        (e.g., Ink Saver Main, Swim Speed Up), it specifies a `family` (e.g.,
        "ink_saver_main") and a `min_ap` (minimum Ability Points to achieve).
        For main-only abilities (e.g., Stealth Jump), it represents the specific ability.
    -   The search expands states in the beam by adding new capstone abilities
        suggested by the `predict` function.
    -   The search avoids adding duplicate standard ability families or duplicate
        main-only ability names but allows a standard ability family's `min_ap`
        to be updated if a higher-AP token for that family is chosen.

4.  **Build Allocation and Legality (`Allocator` class)**:
    -   For each set of capstones explored by the beam search, the `Allocator`
        attempts to assign them to a legal gear configuration.
    -   **Main-Only Abilities**:
        - Must occupy main ability slots.
        - All main-only abilities have canonical gear piece assignments (e.g.,
          "Stealth Jump" on "shoes", "Comeback" on "head"). These preferences are
          honored.
        - Conflicts (e.g., trying to place two head-only abilities, or no free
          main slots for a required main-only ability) render the capstone set
          unbuildable by the allocator.
    -   **Standard Abilities**:
        - Can be satisfied using main slots (contributing 10 AP) or sub-slots
          (each contributing 3 AP).
        - The `Allocator` must ensure that the `achieved_ap` for each standard
          ability family meets or exceeds its `min_ap` from the capstone token.
    -   **Allocator's Optimal Assignment Strategy**:
        - After placing main-only abilities, the `Allocator` uses a combinatorial
          approach to assign standard capstone abilities to the remaining main slots.
        - It iterates through the number of standard abilities (`k`) to place on
          main slots, and for each `k`, through all combinations of `k` standard
          ability families.
        - For each valid combination (where all `min_ap` are met using the chosen
          mains and necessary subs within the 9-sub-slot limit):
            - A score is calculated. This score prioritizes:
                1. Filling more main slots with *any* capstone abilities (main-only
                   or standard).
                2. Achieving a higher total sum of AP for all *requested capstone abilities*.
        - The `Allocator` selects the configuration (assignment of abilities to
          mains and subs) that achieves the highest score.
    -   **Constraint Adherence**:
        - The total number of sub-slots used must not exceed `MAX_SUB_SLOTS` (9).
        - The `Allocator` only uses abilities present in the input `capstones`. It
          does not introduce new abilities to simply fill empty gear slots if those
          abilities were not part of the requested capstone set.

5.  **Output**:
    -   The `reconstruct_build` function returns a single `Build` object
        representing the best legal gear configuration found.
    -   "Best" is primarily determined by the cumulative `log_score` of the
        capstone sequence in the beam search that led to a valid build.
    -   The returned `Build` contains the assignment of ability families to main gear
        slots, the count of sub-slots per ability family, and the total AP
        achieved for each family.

6.  **Data Representation**:
    -   `AbilityToken`: Stores `name`, `family`, `min_ap`, and `main_only` status.
        Main-only tokens have `min_ap=0` as their value is fixed at 10 AP when
        placed, requiring no further AP via subs from their own token.
    -   `Build`: Encapsulates the `mains` (slot -> family), `subs` (family -> count),
        and `achieved_ap` (family -> AP) of a concrete gear setup.
    -   `BeamState`: Tracks a hypothesis in the beam search, including the current
        set of `capstones`, its `log_score`, and any materialized `Build`.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass, field
from itertools import combinations
from typing import Callable, Mapping, Optional, Sequence

from splatnlp.utils.constants import (
    BUCKET_THRESHOLDS,
    CLOTHING_ABILITIES,
    HEADGEAR_ABILITIES,
    MAIN_ONLY_ABILITIES,
    NULL,
    SHOES_ABILITIES,
    SPECIAL_TOKENS,
    STANDARD_ABILITIES,
)
from splatnlp.viz.cluster_labels import abbrev

logger = logging.getLogger(__name__)

_AP_SUFFIX_RE = re.compile(r"_(\d+)$")

CANONICAL_MAIN_ONLY_ABILITIES = {
    **{name: "head" for name in HEADGEAR_ABILITIES},
    **{name: "clothes" for name in CLOTHING_ABILITIES},
    **{name: "shoes" for name in SHOES_ABILITIES},
}

TOKEN_BONUS = math.log(2.0)


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
            return AbilityToken(
                name=token, family=token, min_ap=0, main_only=True
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
    """Concrete, *legal* build."""

    mains: dict[str, str]
    subs: dict[str, int]
    achieved_ap: dict[str, int]

    def __post_init__(self):
        logger.debug(
            f"New build: {self.mains}, {self.subs}, {self.achieved_ap}"
        )

    def disallowed_abilities(self) -> list[str]:
        disallowed: list[str] = []

        for geartype, ability in self.mains.items():
            if ability in STANDARD_ABILITIES or ability is None:
                continue
            logger.debug("Checking %s for %s", geartype, ability)
            if geartype == "head":
                disallowed.extend(x for x in HEADGEAR_ABILITIES if x != ability)
            elif geartype == "clothes":
                disallowed.extend(x for x in CLOTHING_ABILITIES if x != ability)
            elif geartype == "shoes":
                disallowed.extend(x for x in SHOES_ABILITIES if x != ability)

        # When gear is already full (3 mains + 9 subs) nothing else fits
        total_slots = len(self.mains) + sum(self.subs.values())
        if total_slots == 12:
            logger.debug("Twelve slots already filled")
            disallowed.extend(
                x
                for x in (*MAIN_ONLY_ABILITIES, *STANDARD_ABILITIES)
                if x not in self.mains.values() and x not in self.subs
            )
        logger.debug("Disallowed: %s", ", ".join(disallowed))
        return disallowed


class Allocator:
    """Greedy but optimal (for ≤3 slots) ability allocator."""

    MAX_SUB_SLOTS = 9
    SLOTS = ("head", "clothes", "shoes")

    def __init__(self, main_only_slots: Mapping[str, str] | None = None):
        base = dict(CANONICAL_MAIN_ONLY_ABILITIES)
        self._main_only_slots = {**base, **(main_only_slots or {})}

    def allocate(
        self, capstones: Mapping[str, AbilityToken]
    ) -> Optional[Build]:
        """Return a *legal* ``Build`` or ``None`` if impossible."""
        mains: dict[str, Optional[str]] = {s: None for s in self.SLOTS}
        achieved: dict[str, int] = {}
        subs: dict[str, int] = {}

        # ── Phase 0 – place main‑only abilities exactly where they belong
        for tok in capstones.values():
            if not tok.main_only:
                continue
            slot = self._main_only_slots.get(tok.name)
            if slot is None or mains[slot] is not None:
                return None  # slot clash (duplicate main‑only)
            mains[slot] = tok.name
            achieved[tok.name] = 10  # main‑only mains grant 10 AP

        # ── Phase 1 – satisfy standard‑ability AP targets
        std_caps = {
            t.family: t.min_ap for t in capstones.values() if not t.main_only
        }
        if not std_caps and all(v is not None for v in mains.values()):
            return Build(mains, subs, achieved)  # only main‑onlys – done

        free_slots = [s for s, v in mains.items() if v is None]
        best: Optional[tuple[int, int, tuple[str, ...], dict[str, int]]] = None

        for k in range(len(free_slots) + 1):
            for mains_families in combinations(std_caps, k):
                cur_ap = dict(achieved)
                for fam in mains_families:
                    cur_ap[fam] = 10

                subs_needed: dict[str, int] = {}
                for fam, need in std_caps.items():
                    have = cur_ap.get(fam, 0)
                    req_subs = max(0, math.ceil((need - have) / 3))
                    subs_needed[fam] = req_subs
                    cur_ap[fam] = have + req_subs * 3

                total_subs = sum(subs_needed.values())
                if total_subs > self.MAX_SUB_SLOTS:
                    continue

                wasted = sum(cur_ap[f] - std_caps[f] for f in std_caps)
                key = (total_subs, wasted)
                if best is None or key < best[:2]:
                    best = (*key, mains_families, subs_needed)

        if best is None:
            return None  # impossible capstone set

        _, _, mains_families, subs_needed = best
        slot_iter = iter(free_slots)
        for fam in mains_families:
            mains[next(slot_iter)] = fam

        subs.update({fam: n for fam, n in subs_needed.items() if n})

        # achieved AP accounting
        for fam in mains_families:
            achieved[fam] = 10
        for fam, n in subs.items():
            achieved[fam] = achieved.get(fam, 0) + n * 3

        return Build(mains=mains, subs=subs, achieved_ap=achieved)


@dataclass
class BeamState:
    capstones: dict[str, AbilityToken]
    log_score: float
    build: Optional[Build] = None

    def key(self) -> tuple[tuple[str, str], ...]:
        return tuple(sorted((k, v.name) for k, v in self.capstones.items()))

    def __post_init__(self):
        logger.debug(
            "New state: %s, score %.3f", self.capstones, self.log_score
        )


def reconstruct_build(
    predict: Callable[[Sequence[str], str], Mapping[str, float]],
    weapon_id: str,
    initial_context: Sequence[str],
    *,
    vocab_tokens: Sequence[str],
    beam_width: int = 6,
    acceptance_threshold: float = 0.80,
    min_candidates: int = 2,
    max_iterations: int = 12,
    gear_slot_for_main_only: Mapping[str, str] | None = None,
) -> Optional[Build]:
    """Beam‑search over capstone tokens, returning the best *legal* build."""

    logger.info("Starting build search for weapon '%s'", weapon_id)
    if not initial_context:
        initial_context = [NULL]

    token_objects = {
        t: AbilityToken.from_vocab_entry(t)
        for t in vocab_tokens
        if t not in SPECIAL_TOKENS
    }

    allocator = Allocator(gear_slot_for_main_only)
    beam: list[BeamState] = [BeamState(capstones={}, log_score=0.0)]

    for iteration in range(max_iterations):
        logger.debug("Iteration %d/%d", iteration + 1, max_iterations)
        next_candidates: list[BeamState] = []

        for state in beam:
            # If we already have a complete legal build, keep it unchanged
            if state.build and all(
                v is not None for v in state.build.mains.values()
            ):
                next_candidates.append(state)
                continue

            # Ensure ``state.build`` exists so we can generate a disallowed list
            if state.build is None:
                state.build = allocator.allocate(state.capstones)
            disallowed = (
                state.build.disallowed_abilities() if state.build else []
            )

            context = [tok.name for tok in state.capstones.values()] or list(
                initial_context
            )
            logits = predict(context, weapon_id)

            ranked = sorted(
                (
                    (token_objects[name], p)
                    for name, p in logits.items()
                    if name not in SPECIAL_TOKENS
                    and name not in state.capstones
                    and name not in disallowed
                ),
                key=lambda x: x[1],
                reverse=True,
            )
            logger.debug("Ranked: %d abilities", len(ranked))
            logger.debug("Skipped: %d abilities", len(disallowed))
            logger.debug("Skipped abilities: %s", ", ".join(disallowed))

            # high‑confidence tokens + top‑k fallback
            final_expansion: list[tuple[AbilityToken, float]] = [
                (t, p) for t, p in ranked if p >= acceptance_threshold
            ]
            if len(final_expansion) < min_candidates:
                final_expansion.extend(
                    ranked[len(final_expansion) : min_candidates]
                )

            # ── Expand each selected token
            for tok, prob in final_expansion:
                new_caps = dict(state.capstones)
                key = tok.name if tok.main_only else tok.family
                new_caps[key] = tok
                new_score = (
                    state.log_score + math.log(max(prob, 1e-9)) + TOKEN_BONUS
                )

                cand = BeamState(
                    capstones=new_caps,
                    log_score=new_score,
                )
                cand.build = allocator.allocate(new_caps)
                if cand.build is None:
                    continue  # cannot ever fit

                next_candidates.append(cand)

        # ── Beam pruning: unique by capstone set, keep best ``beam_width``
        unique: dict[tuple[tuple[str, str], ...], BeamState] = {}
        for st in next_candidates:
            k = st.key()
            if k not in unique or st.log_score > unique[k].log_score:
                unique[k] = st
        beam = sorted(
            unique.values(),
            key=lambda s: s.log_score / max(1, len(s.capstones)),
            reverse=True,
        )[:beam_width]
        if not beam:
            logger.debug("Search collapsed - no valid states after pruning")
            return None

    # ── pick best fully‑specified legal build
    legal_states = [st for st in beam if st.build]
    if not legal_states:
        logger.debug("No legal states found in final beam")
        return None
    best = max(legal_states, key=lambda s: s.log_score)
    logger.debug("Best build log-score %.3f", best.log_score)
    return best.build
