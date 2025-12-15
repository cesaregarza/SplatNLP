from dataclasses import dataclass, field
from typing import Any, Literal, Mapping, Optional, Sequence, overload

from splatnlp.utils.constants import BUCKET_THRESHOLDS, NULL, TOKEN_BONUS
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


def expand_capstones_to_context(
    capstones: dict[str, AbilityToken],
) -> list[str]:
    """
    Expand capstone tokens to include all prerequisite tiers.

    Training data uses cumulative tokenization where a build with 12 AP SSU
    is represented as [swim_speed_up_3, swim_speed_up_6, swim_speed_up_12].
    This function converts a dict of capstones (highest tier only) to the
    full token list the model expects.

    Parameters
    ----------
    capstones : dict[str, AbilityToken]
        Mapping from token strings to AbilityToken objects (highest tier only)

    Returns
    -------
    list[str]
        Full list of tokens including all prerequisite tiers
    """
    if not capstones:
        return [NULL]

    tokens = []
    for token_str, cap in capstones.items():
        if cap.main_only:
            tokens.append(token_str)
        else:
            for threshold in BUCKET_THRESHOLDS:
                if threshold <= cap.min_ap:
                    tokens.append(f"{cap.family}_{threshold}")

    return tokens if tokens else [NULL]


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
    ]  # Maps family names to their best observed probability (priority)
    trace: list["TraceFrame"] = field(default_factory=list)


@dataclass
class TraceFrame:
    """One snapshot of the beam‑search loop."""

    step: int
    partial_caps: Mapping[str, AbilityToken]
    logits: Mapping[str, float]
    activations: Optional[Sequence[float]] = None
    beam_rank: int = 0
    build: Optional[Build] = None

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
            build=self.build.to_dict() if self.build is not None else None,
        )


def greedy_closure(
    predict_fn,
    weapon_id: str,
    capstones: dict[str, AbilityToken],
    allocator: Allocator,
    threshold: float = 0.5,
    token_bonus: float = TOKEN_BONUS,
    *,
    record_traces: bool = False,
    start_step: int = 0,
):
    """
    Greedily add all tokens that exceed ``threshold`` and improve ``capstones``.
    Each iteration is recorded in ``traces`` when ``record_traces`` is ``True``.
    The returned ``step`` is the global step index of the last greedy addition.
    """
    step = start_step
    traces: list[TraceFrame] | None = [] if record_traces else None
    log_prob_accum = 0.0
    family_logp: dict[str, float] = {}
    while True:
        # Run model on current context (expanded to include prerequisite tiers)
        current_tokens = expand_capstones_to_context(capstones)
        raw_predictions = predict_fn(current_tokens, weapon_id)

        # Handle case where predict_fn returns (predictions, activations)
        if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
            probs, activations = raw_predictions
        else:
            probs = raw_predictions
            activations = None

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

        if record_traces:
            assert traces is not None
            # Try to allocate a build for this state
            build, _ = allocator.allocate(capstones, priority=family_logp)
            traces.append(
                TraceFrame(
                    step=step,
                    partial_caps=dict(capstones),
                    logits=dict(probs),
                    activations=activations,
                    beam_rank=0,
                    build=build,
                )
            )

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
            p = float(probs.get(tok, 0.0))
            if p > 0:
                log_prob_accum += p + token_bonus
                prev = family_logp.get(cap.family, float("-inf"))
                family_logp[cap.family] = max(prev, p)
        step += 1

    return (
        (capstones, step, traces, log_prob_accum, family_logp)
        if record_traces
        else (capstones, log_prob_accum, family_logp)
    )


@overload
def reconstruct_build(
    predict_fn,
    weapon_id: str,
    initial_context: list[str],
    allocator: Allocator,
    beam_size: int = 5,
    max_steps: int | None = 6,
    token_bonus: float = TOKEN_BONUS,
    alpha: float = 0.1,
    top_k: int = 1,
    record_traces: Literal[False] = False,
    min_new_token_prob: float = 0.01,
) -> Optional[list[Build]]: ...


@overload
def reconstruct_build(
    predict_fn,
    weapon_id: str,
    initial_context: list[str],
    allocator: Allocator,
    beam_size: int = 5,
    max_steps: int | None = 6,
    token_bonus: float = TOKEN_BONUS,
    alpha: float = 0.1,
    top_k: int = 1,
    record_traces: Literal[True] = True,
    min_new_token_prob: float = 0.01,
) -> tuple[Optional[list[Build]], Optional[list[list[TraceFrame]]]]: ...


def reconstruct_build(
    predict_fn,
    weapon_id: str,
    initial_context: list[str],
    allocator: Allocator,
    beam_size: int = 5,
    max_steps: int | None = 6,
    token_bonus: float = TOKEN_BONUS,
    alpha: float = 0.1,
    top_k: int = 1,
    record_traces: bool = False,
    min_new_token_prob: float = 0.01,
) -> (
    Optional[list[Build]]
    | tuple[Optional[list[Build]], Optional[list[list[TraceFrame]]]]
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
        mapping each possible next token to a probability in [0, 1].
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
    max_steps : int | None
        Maximum expansions before we stop. If None, run until the beam stops
        changing (with an internal safety cap).
    token_bonus : float
        Bonus added to the log probability each time we add a new token.
    alpha : float
        Weight for the penalty term in final scoring.
    top_k : int
        Number of top predictions to return.
    record_traces : bool
        Whether to record the trace of the beam search.

    Notes
    -----
    The global ``step`` counter starts at ``0`` and continues from the
    greedy closure into the beam phase. Each beam state keeps its own trace
    history so the final returned traces correspond to the builds actually
    produced.

    Returns
    -------
    Optional[list[Build]]
        The `k` best valid Builds found, or None if no valid build could be
        formed.
    Optional[list[list[TraceFrame]]]
        Traces for each returned build, or None if tracing is disabled.
    """
    # 1) Convert initial_context into an initial set of capstones
    initial_capstones: dict[str, AbilityToken] = {}
    for tok in initial_context:
        if tok == "<NULL>":
            continue
        cap = AbilityToken.from_vocab_entry(tok)
        initial_capstones[tok] = cap
    has_real_user_tokens = any(tok != "<NULL>" for tok in initial_context)
    family_logp: dict[str, float] = {}

    # Track the best build so far and tracing storage
    top_candidates: list[dict[str, Any]] = []

    def already_seen(build: Build) -> bool:
        return any(cand["build"] == build for cand in top_candidates)

    # Evaluate the raw user-provided context before any greedy expansion so a
    # valid starting build is always considered.
    if has_real_user_tokens:
        user_build, user_penalty = allocator.allocate(
            dict(initial_capstones), priority=family_logp
        )
        if user_build is not None:
            top_candidates.append(
                {
                    "score": 0.0,
                    "penalty": (
                        user_penalty
                        if user_penalty is not None
                        else float("inf")
                    ),
                    "build": user_build,
                    "trace": [],
                }
            )

    # 2) Run greedy closure to get initial state
    if record_traces:
        (
            initial_capstones,
            step,
            greedy_trace,
            greedy_log_prob,
            family_logp,
        ) = greedy_closure(
            predict_fn,
            weapon_id,
            initial_capstones,
            allocator,
            record_traces=True,
            start_step=0,
            token_bonus=token_bonus,
        )
    else:
        initial_capstones, greedy_log_prob, family_logp = greedy_closure(
            predict_fn,
            weapon_id,
            initial_capstones,
            allocator,
            token_bonus=token_bonus,
        )
        step = 0
        greedy_trace = []

    # Start our beam with one state
    beam: list[BeamState] = [
        BeamState(
            capstones=initial_capstones,
            log_prob=greedy_log_prob,
            family_logp=family_logp,
            trace=list(greedy_trace),
        )
    ]

    # Evaluate the greedy state before any expansions so a valid starting build
    # cannot be pruned from the beam (e.g., when the context already fills all
    # 57 AP and any added token would make allocation impossible).
    initial_build, initial_penalty = allocator.allocate(
        initial_capstones, priority=family_logp
    )
    if initial_build is not None:
        initial_score = greedy_log_prob
        if not already_seen(initial_build):
            top_candidates.append(
                {
                    "score": initial_score,
                    "penalty": (
                        initial_penalty
                        if initial_penalty is not None
                        else float("inf")
                    ),
                    "build": initial_build,
                    "trace": list(greedy_trace),
                }
            )

    # 4) Run beam search for refinements
    # Continue step numbering from the greedy phase
    current_step = step + 1
    adaptive_cap = 30  # safety guard when max_steps is None
    max_iterations = max_steps if max_steps is not None else adaptive_cap
    prev_signatures = {
        tuple(sorted(initial_capstones.keys()))
    }  # track unique token sets
    for _ in range(max_iterations):
        candidates: list[BeamState] = []

        for rank, state in enumerate(beam):
            # Get predictions (expanded to include prerequisite tiers)
            current_tokens = expand_capstones_to_context(state.capstones)
            raw_predictions = predict_fn(current_tokens, weapon_id)

            # Handle case where predict_fn returns (predictions, activations)
            if isinstance(raw_predictions, tuple) and len(raw_predictions) == 2:
                probs, activations = raw_predictions
            else:
                probs = raw_predictions
                activations = None

            frame = None

            # Sort tokens by probability
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
                    # For standard abilities, see if we have an existing
                    # token for the same family.
                    existing_tokens_for_family = [
                        k
                        for k, v in new_caps.items()
                        if v.family == next_cap.family
                    ]
                    if not existing_tokens_for_family:
                        # No token for this family => just add
                        new_caps[next_token] = next_cap
                    else:
                        # There's at least one existing token for this
                        # family. We only keep the new token if min_ap is
                        # strictly higher than all existing ones.
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
                            # next_cap.min_ap <= old_cap.min_ap, so it's
                            # not an improvement. Skip it.
                            continue

                # Add a token bonus to encourage adding (keep raw prob space)
                new_log_prob = state.log_prob + lp + token_bonus
                new_family_logp = dict(state.family_logp)
                if lp > 0:
                    prev_lp = new_family_logp.get(
                        next_cap.family, float("-inf")
                    )
                    new_family_logp[next_cap.family] = max(prev_lp, float(lp))

                if record_traces and frame is None:
                    # Try to allocate a build for this state
                    build, _ = allocator.allocate(
                        state.capstones, priority=state.family_logp
                    )
                    frame = TraceFrame(
                        step=current_step,
                        partial_caps=dict(state.capstones),
                        logits=dict(probs),
                        activations=activations,
                        beam_rank=rank,
                        build=build,
                    )

                new_state = BeamState(
                    capstones=new_caps,
                    log_prob=new_log_prob,
                    family_logp=new_family_logp,
                    trace=state.trace
                    + ([frame] if record_traces and frame is not None else []),
                )
                candidates.append(new_state)

            # Also consider "not adding anything new" as a candidate,
            # to allow the beam state to carry over if no expansions are
            # beneficial.
            if record_traces and frame is None:
                # Try to allocate a build for this state
                build, _ = allocator.allocate(
                    state.capstones, priority=state.family_logp
                )
                frame = TraceFrame(
                    step=current_step,
                    partial_caps=dict(state.capstones),
                    logits=dict(probs),
                    activations=activations,
                    beam_rank=rank,
                    build=build,
                )

            candidates.append(
                BeamState(
                    capstones=dict(state.capstones),
                    log_prob=state.log_prob,
                    family_logp=dict(state.family_logp),
                    trace=state.trace
                    + ([frame] if record_traces and frame is not None else []),
                )
            )

        # Sort by log_prob descending, take the top beam_size
        candidates.sort(key=lambda s: s.log_prob, reverse=True)
        beam = candidates[:beam_size]
        current_step += 1

        # Create a final frame for each state in the beam to record the end of this step
        if record_traces:
            for state in beam:
                current_tokens = expand_capstones_to_context(state.capstones)
                raw_predictions = predict_fn(current_tokens, weapon_id)
                if (
                    isinstance(raw_predictions, tuple)
                    and len(raw_predictions) == 2
                ):
                    probs, activations = raw_predictions
                else:
                    probs = raw_predictions
                    activations = None

                # Try to allocate a build for this state
                build, _ = allocator.allocate(
                    state.capstones, priority=state.family_logp
                )

                final_frame = TraceFrame(
                    step=current_step
                    - 1,  # Use previous step number since we already incremented
                    partial_caps=dict(state.capstones),
                    logits=dict(probs),
                    activations=activations,
                    beam_rank=0,  # Use 0 since we don't know the rank at this point
                    build=build,
                )
                state.trace.append(final_frame)

        # Attempt to allocate each state in the beam and see if it yields
        # a better build
        for st in beam:
            build, penalty = allocator.allocate(
                st.capstones, priority=st.family_logp
            )
            if build is not None:
                final_score = st.log_prob
                if not already_seen(build):
                    top_candidates.append(
                        {
                            "score": final_score,
                            "penalty": (
                                penalty if penalty is not None else float("inf")
                            ),
                            "build": build,
                            "trace": st.trace,
                        }
                    )

        # Early stop if beam signatures stopped changing
        current_signatures = {tuple(sorted(st.capstones.keys())) for st in beam}
        any_new = any(sig not in prev_signatures for sig in current_signatures)
        prev_signatures.update(current_signatures)
        if not any_new:
            break

    # After max_steps expansions, we have our best_build
    if not top_candidates:
        return (None, None) if record_traces else None

    # Prefer fullest builds first, then lowest penalty, then model score.
    top_candidates.sort(
        key=lambda c: (
            -c["build"].total_ap,
            c["penalty"],
            -c["score"],
        )
    )
    top_k_candidates = top_candidates[:top_k]
    result_builds = [cand["build"] for cand in top_k_candidates]
    traces_out = [cand["trace"] for cand in top_k_candidates]
    return (result_builds, traces_out) if record_traces else result_builds
