import math
from dataclasses import dataclass
from typing import Optional

from splatnlp.utils.constants import TOKEN_BONUS
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


@dataclass
class BeamState:
    """Represents a partial set of requested capstones in a multi-label
    scenario.
    """

    capstones: dict[str, AbilityToken]
    log_prob: float


def reconstruct_build(
    predict_fn,
    weapon_id: str,
    initial_context: list[str],
    allocator: Allocator,
    beam_size: int = 5,
    max_steps: int = 6,
    token_bonus: float = TOKEN_BONUS,
    alpha: float = 0.1,
) -> Optional[Build]:
    """
    Multi-label beam search: at each step, expand each state by considering
    new tokens predicted by the model. We do NOT rely on <END> tokens. We
    simply run up to `max_steps` expansions and keep track of the best build
    seen so far (highest (log_prob - alpha * penalty)).

    Parameters
    ----------
    predict_fn : Callable
        A function taking (current_tokens, weapon_id) -> List of (token, logp).
    weapon_id : str
        The ID of the weapon for which we're generating a build.
    initial_context : list[str]
        Starting tokens. Possibly empty or partially filled with user
        preferences.
    allocator : Allocator
        The object that tries to build a valid gear config from a set of
        capstones.
    beam_size : int
        Number of states to keep at each step.
    max_steps : int
        Maximum expansions before we stop.
    token_bonus : float
        Bonus added to the log probability each time we add a new token.
    alpha : float
        Weight for the penalty term in final scoring.

    Returns
    -------
    Optional[Build]
        The best valid Build found, or None if no valid build could be formed.
    """

    # 1) Convert initial_context into an initial set of capstones
    initial_capstones: dict[str, AbilityToken] = {}
    for tok in initial_context:
        cap = AbilityToken.from_vocab_entry(tok)
        if cap.main_only:
            initial_capstones[cap.family] = cap
        else:
            existing = initial_capstones.get(cap.family)
            if existing is None or cap.min_ap > existing.min_ap:
                initial_capstones[cap.family] = cap

    # Start our beam with one state
    beam: list[BeamState] = [
        BeamState(capstones=initial_capstones, log_prob=0.0)
    ]

    # 2) Track the best build so far
    best_final_score = -math.inf
    best_build: Optional[Build] = None

    # 3) Run expansions for up to max_steps
    for step in range(max_steps):
        candidates: list[BeamState] = []

        for state in beam:
            # Expand using the model
            current_tokens = list(state.capstones.keys())
            next_tokens_with_scores = predict_fn(current_tokens, weapon_id)

            # If your model returns many expansions, you might limit them:
            # next_tokens_with_scores = next_tokens_with_scores[:top_k]

            for next_token, lp in next_tokens_with_scores:
                # Convert to AbilityToken
                try:
                    next_cap = AbilityToken.from_vocab_entry(next_token)
                except ValueError:
                    continue  # skip invalid

                # If it's main-only, ensure we don't duplicate
                new_caps = dict(state.capstones)
                if next_cap.main_only:
                    if next_cap.family in new_caps:
                        # Already added => skip
                        continue
                    new_caps[next_cap.family] = next_cap
                else:
                    # For standard, keep higher min_ap if we already have that
                    # family
                    existing = new_caps.get(next_cap.family)
                    if existing is None or next_cap.min_ap > existing.min_ap:
                        new_caps[next_cap.family] = next_cap

                # Add a token bonus to encourage adding
                new_log_prob = state.log_prob + lp + token_bonus

                # Create the new candidate
                new_state = BeamState(capstones=new_caps, log_prob=new_log_prob)
                candidates.append(new_state)

            # Also consider "not adding anything new" as a candidate,
            # to allow the beam state to carry over if no expansions are
            # beneficial. The log_prob doesn't change, but the state is
            # effectively re-added.
            candidates.append(state)

        # 4) Sort by log_prob descending, take the top beam_size
        candidates.sort(key=lambda s: s.log_prob, reverse=True)
        beam = candidates[:beam_size]

        # 5) Attempt to allocate each state in the beam and see if it yields a
        # better build
        for st in beam:
            build, penalty = allocator.allocate(st.capstones)
            if build is not None:
                final_score = st.log_prob - alpha * penalty
                if final_score > best_final_score:
                    best_final_score = final_score
                    best_build = build

    # 6) After max_steps expansions, we have our best_build
    return best_build
