"""Branchiness metrics for SetCompletionModel outputs.

These metrics are designed to capture "mode competition" at a branching point,
not just uncertainty reduction (entropy-drop IG).

We work with a *fixed* candidate support F computed at a greedy-closure branch
point, typically using ``stack_policy="next-tier"`` so stacking ladders do not
dominate the candidate set.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

import torch

from splatnlp.mechinterp.analysis.information_gain import (
    StackPolicy,
    add_token_to_capstones,
    allowed_tokens_from_vocab,
    normalize_weights,
    stacking_family_levels,
    support_tokens_for_state,
    tokens_to_capstones,
)
from splatnlp.model.models import SetCompletionModel
from splatnlp.utils.constants import NULL
from splatnlp.utils.infer import build_predict_abilities_batch
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import (
    expand_capstones_to_context,
    greedy_closure,
)

SuppressionWeight = Literal["p0", "uniform"]


def _shannon_entropy_list(probs: list[float], *, log_base: float) -> float:
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be > 0 and != 1")
    inv_log_base = 1.0 / math.log(log_base)
    h = 0.0
    for p in probs:
        p = float(p)
        if p <= 0.0:
            continue
        h -= p * (math.log(p) * inv_log_base)
    return float(h)


def _jensen_shannon_divergence(
    p: list[float],
    q: list[float],
    *,
    log_base: float,
) -> float:
    if len(p) != len(q):
        raise ValueError("p and q must have the same length")
    if not p:
        return 0.0
    m = [0.5 * (pi + qi) for pi, qi in zip(p, q)]
    h_m = _shannon_entropy_list(m, log_base=log_base)
    h_p = _shannon_entropy_list(p, log_base=log_base)
    h_q = _shannon_entropy_list(q, log_base=log_base)
    return float(h_m - 0.5 * (h_p + h_q))


@dataclass(frozen=True)
class TokenShiftRow:
    token: str
    p0: float
    js_shift: float
    expected_js: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TokenSuppressionRow:
    token: str
    p0: float
    suppression: float
    expected_suppression: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeaponBranchinessResult:
    weapon_id: str
    log_base: float
    base_context: list[str]
    greedy_threshold: float
    omit_conditioned: bool
    stack_policy: StackPolicy
    branch_context: list[str]
    support0: list[str]
    h0: float
    suppression_top_m: int
    suppression_weight: SuppressionWeight
    suppression_score: float
    suppression_rows: list[TokenSuppressionRow]
    branchiness_js: float
    js_rows: list[TokenShiftRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weapon_id": self.weapon_id,
            "log_base": self.log_base,
            "base_context": list(self.base_context),
            "greedy_threshold": self.greedy_threshold,
            "omit_conditioned": self.omit_conditioned,
            "stack_policy": self.stack_policy,
            "branch_context": list(self.branch_context),
            "support0": list(self.support0),
            "support0_len": int(len(self.support0)),
            "h0": self.h0,
            "suppression_top_m": self.suppression_top_m,
            "suppression_weight": self.suppression_weight,
            "suppression_score": self.suppression_score,
            "suppression_rows": [r.to_dict() for r in self.suppression_rows],
            "branchiness_js": self.branchiness_js,
            "js_rows": [r.to_dict() for r in self.js_rows],
        }


@dataclass(frozen=True)
class FixedSupportDistributions:
    branch_context: list[str]
    support0: list[str]
    p0: dict[str, float]
    conditional_probs: dict[str, dict[str, float]]


def fixed_support_distributions_for_weapon(
    *,
    model: SetCompletionModel,
    weapon_id: str,
    vocab: Mapping[str, int],
    weapon_vocab: Mapping[str, int],
    batch_size: int = 64,
    base_context: list[str] | None = None,
    greedy_threshold: float = 0.7,
    omit_conditioned: bool = True,
    stack_policy: StackPolicy = "next-tier",
    device: torch.device | None = None,
) -> FixedSupportDistributions:
    """Compute p0 and q_t distributions on a fixed candidate support.

    Returns a base distribution p0 over the fixed support F computed at the
    greedy-closure branching point, plus conditional distributions q_t over
    F\\{t} after conditioning on each t in F.
    """
    if base_context is None:
        base_context = [NULL]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if greedy_threshold <= 0.0 or greedy_threshold >= 1.0:
        raise ValueError("greedy_threshold must be in (0, 1)")
    if device is None:
        device = torch.device("cpu")

    allowed_tokens = allowed_tokens_from_vocab(vocab)
    levels_by_family = stacking_family_levels(allowed_tokens)

    predict_batch = build_predict_abilities_batch(
        dict(vocab),
        dict(weapon_vocab),
        pad_token="<PAD>",
        hook=None,
        device=device,
        output_type="dict",
    )

    def predict_one(
        current_tokens: list[str], weapon_id: str
    ) -> dict[str, float]:
        return predict_batch(model, [list(current_tokens)], weapon_id)[0]

    initial_capstones = tokens_to_capstones(base_context)
    allocator = Allocator()
    branch_capstones, _, _ = greedy_closure(
        predict_one,
        str(weapon_id),
        dict(initial_capstones),
        allocator,
        threshold=float(greedy_threshold),
    )
    branch_context = expand_capstones_to_context(branch_capstones)

    conditioned = set(branch_context)
    conditioned.discard(NULL)

    support0 = support_tokens_for_state(
        allowed_tokens=list(allowed_tokens),
        conditioned_tokens=conditioned,
        omit_conditioned=omit_conditioned,
        stack_policy=stack_policy,
        levels_by_family=levels_by_family,
    )

    if not support0:
        return FixedSupportDistributions(
            branch_context=list(branch_context),
            support0=[],
            p0={},
            conditional_probs={},
        )

    base_probs_raw = predict_batch(model, [list(branch_context)], weapon_id)[0]
    p0 = normalize_weights(base_probs_raw, support=support0)

    contexts: list[list[str]] = []
    for tok in support0:
        capstones_with_tok = add_token_to_capstones(branch_capstones, tok)
        contexts.append(expand_capstones_to_context(capstones_with_tok))

    conditional_probs: dict[str, dict[str, float]] = {}
    for start in range(0, len(support0), batch_size):
        chunk_tokens = support0[start : start + batch_size]
        chunk_contexts = contexts[start : start + batch_size]
        chunk_raw = predict_batch(model, chunk_contexts, weapon_id)
        for tok, probs_raw in zip(chunk_tokens, chunk_raw):
            support_minus = [t for t in support0 if t != tok]
            conditional_probs[tok] = normalize_weights(
                probs_raw, support=support_minus
            )

    return FixedSupportDistributions(
        branch_context=list(branch_context),
        support0=list(support0),
        p0=dict(p0),
        conditional_probs=conditional_probs,
    )


def branchiness_for_weapon(
    *,
    model: SetCompletionModel,
    weapon_id: str,
    vocab: Mapping[str, int],
    weapon_vocab: Mapping[str, int],
    batch_size: int = 64,
    log_base: float = 2.0,
    base_context: list[str] | None = None,
    greedy_threshold: float = 0.7,
    omit_conditioned: bool = True,
    stack_policy: StackPolicy = "next-tier",
    suppression_top_m: int = 8,
    suppression_weight: SuppressionWeight = "p0",
    js_top_k: int = 15,
    device: torch.device | None = None,
) -> WeaponBranchinessResult:
    """Compute suppression-based and JS-shift branchiness at a branch point."""
    if base_context is None:
        base_context = [NULL]
    if suppression_top_m <= 0:
        raise ValueError("suppression_top_m must be > 0")

    dists = fixed_support_distributions_for_weapon(
        model=model,
        weapon_id=str(weapon_id),
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        batch_size=int(batch_size),
        base_context=list(base_context),
        greedy_threshold=float(greedy_threshold),
        omit_conditioned=bool(omit_conditioned),
        stack_policy=stack_policy,
        device=device,
    )

    branch_context = dists.branch_context
    support0 = dists.support0
    p0 = dists.p0
    conditional_probs = dists.conditional_probs

    if len(support0) <= 1:
        return WeaponBranchinessResult(
            weapon_id=str(weapon_id),
            log_base=float(log_base),
            base_context=list(base_context),
            greedy_threshold=float(greedy_threshold),
            omit_conditioned=bool(omit_conditioned),
            stack_policy=stack_policy,
            branch_context=list(branch_context),
            support0=list(support0),
            h0=0.0,
            suppression_top_m=int(suppression_top_m),
            suppression_weight=suppression_weight,
            suppression_score=0.0,
            suppression_rows=[],
            branchiness_js=0.0,
            js_rows=[],
        )

    h0 = _shannon_entropy_list(
        [float(p0[t]) for t in support0],
        log_base=float(log_base),
    )

    js_shift: dict[str, float] = {}
    for tok in support0:
        denom = 1.0 - float(p0[tok])
        if denom <= 0.0:
            js_shift[tok] = 0.0
            continue
        support_minus = [t for t in support0 if t != tok]
        q_minus = [float(p0[t]) / denom for t in support_minus]
        q_cond = [float(conditional_probs[tok][t]) for t in support_minus]
        js_shift[tok] = _jensen_shannon_divergence(
            q_cond,
            q_minus,
            log_base=float(log_base),
        )

    js_rows = [
        TokenShiftRow(
            token=tok,
            p0=float(p0[tok]),
            js_shift=float(js_shift[tok]),
            expected_js=float(p0[tok] * js_shift[tok]),
        )
        for tok in support0
    ]
    js_rows.sort(key=lambda r: r.expected_js, reverse=True)
    branchiness_js = float(sum(r.expected_js for r in js_rows))
    if js_top_k > 0:
        js_rows_out = js_rows[: int(js_top_k)]
    else:
        js_rows_out = []

    top_m = min(int(suppression_top_m), len(support0))
    top_tokens = sorted(support0, key=lambda t: p0[t], reverse=True)[:top_m]
    suppression_by_tok: dict[str, float] = {}
    for tok in top_tokens:
        score = 0.0
        for other in top_tokens:
            if other == tok:
                continue
            w = float(p0[other]) if suppression_weight == "p0" else 1.0
            delta = float(p0[other]) - float(conditional_probs[tok][other])
            if delta > 0.0:
                score += w * delta
        suppression_by_tok[tok] = float(score)

    suppression_rows = [
        TokenSuppressionRow(
            token=tok,
            p0=float(p0[tok]),
            suppression=float(suppression_by_tok[tok]),
            expected_suppression=float(p0[tok] * suppression_by_tok[tok]),
        )
        for tok in top_tokens
    ]
    suppression_rows.sort(key=lambda r: r.expected_suppression, reverse=True)
    suppression_score = float(
        sum(r.expected_suppression for r in suppression_rows)
    )

    return WeaponBranchinessResult(
        weapon_id=str(weapon_id),
        log_base=float(log_base),
        base_context=list(base_context),
        greedy_threshold=float(greedy_threshold),
        omit_conditioned=bool(omit_conditioned),
        stack_policy=stack_policy,
        branch_context=list(branch_context),
        support0=list(support0),
        h0=float(h0),
        suppression_top_m=int(top_m),
        suppression_weight=suppression_weight,
        suppression_score=float(suppression_score),
        suppression_rows=suppression_rows,
        branchiness_js=float(branchiness_js),
        js_rows=js_rows_out,
    )
