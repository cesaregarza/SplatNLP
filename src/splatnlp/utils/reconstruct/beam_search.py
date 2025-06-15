import math
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, overload

from splatnlp.utils.constants import NULL, TOKEN_BONUS
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


def is_saturated(build: Build | None) -> bool:
    """Check if a build has reached maximum AP capacity."""
    return build is not None and build.total_ap >= Build.MAX_TOTAL_AP


@dataclass
class BeamState:
    """
    Represents a partial set of requested capstones in a multi-label scenario.
    Keys in `capstones` are the exact token strings. Values are the
    corresponding AbilityToken objects.
    """

    capstones: dict[str, AbilityToken]
    log_prob: float
    family_logp: dict[str, float] = field(default_factory=dict)
    saturated: bool = False  # Whether this state has reached max AP capacity
    path_id: str = "0"  # Hierarchical path identifier for tracing


@dataclass
class TraceFrame:
    """One snapshot of the beam‑search loop."""

    step: int
    partial_caps: Mapping[str, AbilityToken]
    logits: Mapping[str, float]
    activations: Optional[Sequence[float]] = None
    beam_rank: int = 0
    build_ap: int | None = None  # Total AP of the allocated build
    penalty: int | None = None  # Penalty score from allocation
    saturated: bool = False  # Whether this state is saturated
    build: Build | None = None  # The actual allocated build
    path_id: str = "0"  # Hierarchical path identifier

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
            build_ap=self.build_ap,
            penalty=self.penalty,
            saturated=self.saturated,
            build=str(self.build) if self.build is not None else None,
            path_id=self.path_id,
        )


def greedy_closure(
    predict_fn,
    weapon_id: str,
    capstones: dict[str, AbilityToken],
    allocator: Allocator,
    threshold: float = 0.5,
    record_trace: bool = False,
    trace: list[TraceFrame] = None,
    step: int = 0,
) -> tuple[dict[str, AbilityToken], int]:
    """
    Greedily add all tokens that exceed the threshold and improve the build.
    Returns when no more tokens can be added.
    Returns a tuple of (capstones, step) where step is the updated step counter.
    """
    while True:
        # Run model on current context (or NULL if empty)
        current_tokens = list(capstones.keys()) or [NULL]
        raw_predictions = predict_fn(current_tokens, weapon_id)

        # Handle case where predict_fn returns (predictions, activations)
        if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
            probs, activations = raw_predictions
        else:
            probs = raw_predictions
            activations = None

        # Try to allocate the current state
        build, penalty = allocator.allocate(capstones)
        saturated = is_saturated(build) if build is not None else False

        # Record trace if requested
        if record_trace and trace is not None:
            trace.append(
                TraceFrame(
                    step=step,
                    partial_caps=dict(capstones),  # Make immutable copy
                    logits=dict(probs),  # Make immutable copy
                    activations=(
                        activations.clone()  # Use clone() for PyTorch tensors
                        if activations is not None
                        else None
                    ),
                    beam_rank=0,  # Always rank 0 in greedy closure
                    build_ap=build.total_ap if build is not None else None,
                    penalty=penalty,
                    saturated=saturated,
                    build=build,
                    path_id="0",  # Greedy closure always uses path "0"
                )
            )

        # If state is saturated, we're done
        if saturated:
            break

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

        step += 1

    return capstones, step


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
    greed_threshold: float = 0.5,
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
    greed_threshold: float = 0.5,
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
    greed_threshold: float = 0.5,
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
    greed_threshold : float
        Probability threshold for greedy exploitation vs branching.
        Tokens with probability >= greed_threshold are added greedily,
        while tokens below this threshold create new branches.

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

    # Initialize trace if requested
    trace: list[TraceFrame] = [] if record_traces else None

    # 2) Run greedy closure to get initial state
    global_step = 0
    initial_capstones, global_step = greedy_closure(
        predict_fn,
        weapon_id,
        initial_capstones,
        allocator=allocator,
        threshold=greed_threshold,
        record_trace=record_traces,
        trace=trace,
        step=global_step,
    )

    # Start our beam with one state
    beam: list[BeamState] = [
        BeamState(
            capstones=initial_capstones,
            log_prob=0.0,
            family_logp={},  # No need for family_logp in new approach
            saturated=False,  # Initialize saturated flag
            path_id="0",  # Initial state is path "0"
        )
    ]

    # 3) Track the best build so far
    # Store (score, build, path_id) tuples
    top_candidates: list[tuple[float, Build, str]] = []

    # 4) Run beam search for refinements
    prev_beam_signatures: set[frozenset[str]] = set()

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

            # Try to allocate the current state
            build, penalty = allocator.allocate(state.capstones)
            if build is not None:
                state.saturated = is_saturated(build)
            else:
                state.saturated = False

            # If state is saturated, keep it but don't expand
            if state.saturated:
                candidates.append(state)
                continue

            # Split tokens into confident and uncertain
            confident = [
                (t, lp) for t, lp in probs.items() if lp >= greed_threshold
            ]
            uncertain = [
                (t, lp) for t, lp in probs.items() if lp < greed_threshold
            ]

            # ---- greedy exploit phase --------------------------------------------
            for next_token, lp in confident:
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

                # Add a token bonus to encourage adding, with decay
                effective_bonus = token_bonus / (
                    1 + step
                )  # Simple 1/(t+1) decay
                new_log_prob = state.log_prob + lp + effective_bonus

                # Create the new candidate
                new_state = BeamState(
                    capstones=new_caps,
                    log_prob=new_log_prob,
                    family_logp={},  # No need for family_logp in new approach
                    saturated=False,  # Will be set after allocation
                    path_id=state.path_id,  # Keep parent's path for confident tokens
                )
                candidates.append(new_state)

            # ---- branching phase --------------------------------------------------
            for rank, (next_token, lp) in enumerate(
                sorted(uncertain, key=lambda x: x[1], reverse=True)
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

                # Add a token bonus to encourage adding, with decay
                effective_bonus = token_bonus / (
                    1 + step
                )  # Simple 1/(t+1) decay
                new_log_prob = state.log_prob + lp + effective_bonus

                # Create the new candidate with a new path ID
                new_state = BeamState(
                    capstones=new_caps,
                    log_prob=new_log_prob,
                    family_logp={},  # No need for family_logp in new approach
                    saturated=False,  # Will be set after allocation
                    path_id=f"{state.path_id}.{rank}",  # New branch with rank
                )
                candidates.append(new_state)

            # Also consider "not adding anything new" as a candidate,
            # to allow the beam state to carry over if no expansions are
            # beneficial.
            candidates.append(state)

        # Record the trace with allocation information
        if record_traces:
            trace.append(
                TraceFrame(
                    step=global_step,  # Use global step counter
                    partial_caps=dict(state.capstones),  # Make immutable copy
                    logits=dict(probs),  # Make immutable copy
                    activations=(
                        activations.clone()  # Use clone() for PyTorch tensors
                        if activations is not None
                        else None
                    ),
                    beam_rank=beam.index(state),
                    build_ap=build.total_ap if build is not None else None,
                    penalty=penalty,
                    saturated=state.saturated,
                    build=build,
                    path_id=state.path_id,
                )
            )

        # Sort by log_prob descending, take the top beam_size
        candidates.sort(key=lambda s: s.log_prob, reverse=True)
        beam = candidates[:beam_size]

        # Check for early stopping - if all beams are saturated, we're done
        if all(st.saturated for st in beam):
            break  # Nothing else can be added anywhere

        # Check for early stopping - if beam hasn't changed, we're done
        current_signatures = {frozenset(st.capstones.keys()) for st in beam}
        if current_signatures == prev_beam_signatures:
            break  # Early convergence
        prev_beam_signatures = current_signatures

        # Attempt to allocate each state in the beam and see if it yields
        # a better build
        for st in beam:
            build, penalty = allocator.allocate(st.capstones)
            if build is not None:
                final_score = st.log_prob - alpha * penalty
                # Only add if we don't already have an equivalent build
                if not any(b == build for _, b, _ in top_candidates):
                    top_candidates.append((final_score, build, st.path_id))

        global_step += 1

    # After max_steps expansions, we have our best_build
    if not top_candidates:
        return (None, trace) if record_traces else None

    # Sort by path_id first, then by score
    top_candidates.sort(key=lambda x: (x[0], x[2]), reverse=True)
    top_k_candidates = top_candidates[:top_k]
    result_builds = [b for _, b, _ in top_k_candidates]
    return (result_builds, trace) if record_traces else result_builds
