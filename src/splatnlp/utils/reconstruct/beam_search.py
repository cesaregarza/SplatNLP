import math
from dataclasses import dataclass
from typing import Optional

from splatnlp.utils.constants import NULL, TOKEN_BONUS
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


@dataclass
class BeamState:
    """
    Represents a partial set of requested capstones in a multi-label scenario.
    Keys in `capstones` are the exact token strings. Values are the
    corresponding AbilityToken objects.
    """

    capstones: dict[str, AbilityToken]
    log_prob: float
    family_logp: dict[
        str, float
    ]  # Maps family names to their best log probability


def reconstruct_build(
    predict_fn,
    weapon_id: str,
    initial_context: list[str],
    allocator: Allocator,
    beam_size: int = 5,
    max_steps: int = 6,
    token_bonus: float = TOKEN_BONUS,
    alpha: float = 0.1,
    top_k: int = 1,
) -> Optional[list[Build]]:
    """
    Multi-label beam search: at each step, expand each state by considering
    new tokens predicted by the model. We do NOT rely on <END> tokens. We
    simply run up to `max_steps` expansions and keep track of the best build
    seen so far (highest (log_prob - alpha * penalty)).

    Parameters
    ----------
    predict_fn : Callable
        A function taking (current_tokens, weapon_id) -> Dict[str, float],
        mapping each possible next token to a log probability.
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
    top_k : int
        Number of top predictions to return.

    Returns
    -------
    Optional[list[Build]]
        The `k` best valid Builds found, or None if no valid build could be
        formed.
    """

    # 1) Convert initial_context into an initial set of capstones
    #    Dictionary key = exact token string, value = AbilityToken
    initial_capstones: dict[str, AbilityToken] = {}
    initial_family_logp: dict[str, float] = {}

    for tok in initial_context:
        cap = AbilityToken.from_vocab_entry(tok)
        if cap.main_only:
            # If the exact token is not already in the dict, add it
            if tok not in initial_capstones:
                initial_capstones[tok] = cap
                initial_family_logp[cap.family] = (
                    0.0  # Default high priority for user-specified tokens
                )
        else:
            # For standard abilities, if we already have the same family with
            # lower AP, replace it. Otherwise, just add.
            existing_tokens_for_family = [
                k
                for k, v in initial_capstones.items()
                if v.family == cap.family
            ]
            # Check if there's an existing token with strictly less min_ap
            replaced = False
            for old_token_key in existing_tokens_for_family:
                old_cap = initial_capstones[old_token_key]
                if cap.min_ap > old_cap.min_ap:
                    # Replace the old token
                    del initial_capstones[old_token_key]
                    initial_capstones[tok] = cap
                    initial_family_logp[cap.family] = (
                        0.0  # Default high priority for user-specified tokens
                    )
                    replaced = True
                    break
            if not replaced and not existing_tokens_for_family:
                initial_capstones[tok] = cap
                initial_family_logp[cap.family] = (
                    0.0  # Default high priority for user-specified tokens
                )

    # Start our beam with one state
    beam: list[BeamState] = [
        BeamState(
            capstones=initial_capstones,
            log_prob=0.0,
            family_logp=initial_family_logp,
        )
    ]

    # 2) Track the best build so far
    top_candidates: list[tuple[float, Build]] = []

    # 3) Run expansions for up to max_steps
    for step in range(max_steps):
        candidates: list[BeamState] = []

        for state in beam:
            # Expand using the model
            current_tokens = list(state.capstones.keys())
            if len(current_tokens) == 0:
                current_tokens = [
                    NULL
                ]  # Some marker if your model expects at least one token

            # predict_fn returns a dict: {token: logp, ...}
            next_tokens_with_scores = predict_fn(current_tokens, weapon_id)

            # Sort tokens by log probability to prioritize higher probability
            # tokens
            for next_token, lp in sorted(
                next_tokens_with_scores.items(),
                key=lambda kv: kv[1],
                reverse=True,
            ):
                # Convert to AbilityToken
                try:
                    next_cap = AbilityToken.from_vocab_entry(next_token)
                except ValueError:
                    continue  # skip invalid

                # Build a new dictionary of capstones from the old state
                new_caps = dict(state.capstones)
                new_family_logp = dict(state.family_logp)

                if next_cap.main_only:
                    # If we already have this EXACT main-only token, skip
                    if next_token in new_caps:
                        continue
                    # Otherwise add it
                    new_caps[next_token] = next_cap
                    new_family_logp[next_cap.family] = max(
                        new_family_logp.get(next_cap.family, -math.inf), lp
                    )
                else:
                    # For standard abilities, see if we have an existing token
                    # for the same family.
                    existing_tokens_for_family = [
                        k
                        for k, v in new_caps.items()
                        if v.family == next_cap.family
                    ]
                    if not existing_tokens_for_family:
                        # No token for this family => just add
                        new_caps[next_token] = next_cap
                        new_family_logp[next_cap.family] = max(
                            new_family_logp.get(next_cap.family, -math.inf), lp
                        )
                    else:
                        # There's at least one existing token for this family.
                        # We only keep the new token if min_ap is strictly
                        # higher than all existing ones.
                        # If so, remove the older tokens for that family and add
                        # the new one.
                        can_replace = False
                        for old_token_key in existing_tokens_for_family:
                            old_cap = new_caps[old_token_key]
                            if next_cap.min_ap > old_cap.min_ap:
                                # Remove them
                                del new_caps[old_token_key]
                                can_replace = True
                        if can_replace:
                            # Add the new higher-min-ap token
                            new_caps[next_token] = next_cap
                            new_family_logp[next_cap.family] = max(
                                new_family_logp.get(next_cap.family, -math.inf),
                                lp,
                            )
                        else:
                            # If we didn't replace anything, that means
                            # next_cap.min_ap <= old_cap.min_ap, so it's not an
                            # improvement. Skip it.
                            continue

                # Add a token bonus to encourage adding
                new_log_prob = state.log_prob + lp + token_bonus

                # Create the new candidate
                new_state = BeamState(
                    capstones=new_caps,
                    log_prob=new_log_prob,
                    family_logp=new_family_logp,
                )
                candidates.append(new_state)

            # Also consider "not adding anything new" as a candidate,
            # to allow the beam state to carry over if no expansions are
            # beneficial.
            candidates.append(state)

        # 4) Sort by log_prob descending, take the top beam_size
        candidates.sort(key=lambda s: s.log_prob, reverse=True)
        beam = candidates[:beam_size]

        # 5) Attempt to allocate each state in the beam and see if it yields
        # a better build
        for st in beam:
            build, penalty = allocator.allocate(
                st.capstones, priority=st.family_logp
            )
            if build is not None:
                final_score = st.log_prob - alpha * penalty
                # Only add if we don't already have an equivalent build
                if not any(b == build for _, b in top_candidates):
                    top_candidates.append((final_score, build))

    # 6) After max_steps expansions, we have our best_build
    if not top_candidates:
        return None

    top_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k_candidates = top_candidates[:top_k]
    return [b for _, b in top_k_candidates]
