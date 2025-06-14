import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, overload

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


@dataclass
class TraceFrame:
    """One snapshot of the beam‑search loop."""

    step: int
    partial_caps: Mapping[str, AbilityToken]
    logits: Mapping[str, float]
    activations: Optional[Sequence[float]] = None
    beam_rank: int = 0

    def to_dict(self) -> dict[str, Any]:
        return dict(
            step=self.step,
            beam_rank=self.beam_rank,
            partial_caps=list(self.partial_caps.keys()),
            logits=self.logits,
            activations=(
                self.activations.tolist()
                if self.activations is not None
                else None
            ),
        )


def greedy_closure(
    predict_fn,
    weapon_id: str,
    capstones: dict[str, AbilityToken],
    threshold: float = 0.5,
) -> dict[str, AbilityToken]:
    """
    Greedily add all tokens that exceed the threshold and improve the build.
    Returns when no more tokens can be added.
    """
    while True:
        # Run model on current context (or NULL if empty)
        current_tokens = list(capstones.keys()) or [NULL]
        raw_predictions = predict_fn(current_tokens, weapon_id)

        # Handle case where predict_fn returns (predictions, activations)
        if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
            probs = raw_predictions[0]
        else:
            probs = raw_predictions

        # Find tokens to add
        to_add = []
        for tok, p in probs.items():
            if p < threshold:
                continue

            try:
                next_cap = AbilityToken.from_vocab_entry(tok)
            except ValueError:
                continue  # skip invalid tokens

            # Check if we should add this token
            if next_cap.main_only:
                # For main-only, only add if not already present
                if tok not in capstones:
                    to_add.append(tok)
            else:
                # For standard abilities, check if we should add/replace
                existing_tokens = [
                    k
                    for k, v in capstones.items()
                    if v.family == next_cap.family
                ]
                if not existing_tokens:
                    # No token for this family => add
                    to_add.append(tok)
                else:
                    # Only add if min_ap is higher than existing
                    should_add = False
                    for old_tok in existing_tokens:
                        old_cap = capstones[old_tok]
                        if next_cap.min_ap > old_cap.min_ap:
                            should_add = True
                            break
                    if should_add:
                        to_add.append(tok)

        if not to_add:
            break

        # Add all selected tokens at once
        for tok in to_add:
            cap = AbilityToken.from_vocab_entry(tok)
            if cap.main_only:
                capstones[tok] = cap
            else:
                # Remove any existing tokens of same family with lower min_ap
                existing_tokens = [
                    k
                    for k, v in capstones.items()
                    if v.family == cap.family and v.min_ap < cap.min_ap
                ]
                for old_tok in existing_tokens:
                    del capstones[old_tok]
                capstones[tok] = cap

    return capstones


@overload
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
    record_traces: Literal[False] = False,
) -> Optional[list[Build]]: ...


@overload
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
    record_traces: Literal[True] = True,
) -> tuple[Optional[list[Build]], Optional[list[TraceFrame]]]: ...


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
    record_traces: bool = False,
) -> (
    Optional[list[Build]]
    | tuple[Optional[list[Build]], Optional[list[TraceFrame]]]
):
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
    record_traces : bool
        Whether to record the trace of the beam search.

    Returns
    -------
    Optional[list[Build]]
        The `k` best valid Builds found, or None if no valid build could be
        formed.
    Optional[list[TraceFrame]]
        The trace of the beam search, or None if tracing is disabled.
    """
    # 1) Convert initial_context into an initial set of capstones
    initial_capstones: dict[str, AbilityToken] = {}
    for tok in initial_context:
        cap = AbilityToken.from_vocab_entry(tok)
        initial_capstones[tok] = cap

    # 2) Run greedy closure to get initial state
    initial_capstones = greedy_closure(predict_fn, weapon_id, initial_capstones)

    # Start our beam with one state
    beam: list[BeamState] = [
        BeamState(
            capstones=initial_capstones,
            log_prob=0.0,
            family_logp={},  # No need for family_logp in new approach
        )
    ]

    # 3) Track the best build so far and tracing storage
    top_candidates: list[tuple[float, Build]] = []
    trace: list[TraceFrame] = [] if record_traces else None

    # 4) Run beam search for refinements
    for step in range(max_steps):
        candidates: list[BeamState] = []

        for state in beam:
            # Get predictions
            current_tokens = list(state.capstones.keys()) or [NULL]
            raw_predictions = predict_fn(current_tokens, weapon_id)

            # Handle case where predict_fn returns (predictions, activations)
            if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
                probs, activations = raw_predictions
            else:
                probs = raw_predictions
                activations = None

            # Sort tokens by log probability
            for next_token, lp in sorted(
                probs.items(),
                key=lambda kv: kv[1],
                reverse=True,
            ):
                try:
                    next_cap = AbilityToken.from_vocab_entry(next_token)
                except ValueError:
                    continue  # skip invalid

                # Build a new dictionary of capstones from the old state
                new_caps = dict(state.capstones)

                if next_cap.main_only:
                    # If we already have this EXACT main-only token, skip
                    if next_token in new_caps:
                        continue
                    # Otherwise add it
                    new_caps[next_token] = next_cap
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
                    else:
                        # There's at least one existing token for this family.
                        # We only keep the new token if min_ap is strictly
                        # higher than all existing ones.
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
                    family_logp={},  # No need for family_logp in new approach
                )
                candidates.append(new_state)

            # Also consider "not adding anything new" as a candidate,
            # to allow the beam state to carry over if no expansions are
            # beneficial.
            candidates.append(state)

        # Record the trace
        if record_traces:
            trace.append(
                TraceFrame(
                    step=step,
                    partial_caps=state.capstones,
                    logits=probs,
                    activations=activations,  # Now properly handling activations
                    beam_rank=beam.index(state),
                )
            )

        # Sort by log_prob descending, take the top beam_size
        candidates.sort(key=lambda s: s.log_prob, reverse=True)
        beam = candidates[:beam_size]

        # Attempt to allocate each state in the beam and see if it yields
        # a better build
        for st in beam:
            build, penalty = allocator.allocate(st.capstones)
            if build is not None:
                final_score = st.log_prob - alpha * penalty
                # Only add if we don't already have an equivalent build
                if not any(b == build for _, b in top_candidates):
                    top_candidates.append((final_score, build))

    # After max_steps expansions, we have our best_build
    if not top_candidates:
        return (None, trace) if record_traces else None

    top_candidates.sort(key=lambda x: x[0], reverse=True)
    top_k_candidates = top_candidates[:top_k]
    result_builds = [b for _, b in top_k_candidates]
    return (result_builds, trace) if record_traces else result_builds
