"""Expected depth delta after conditioning at a branch point.

Given a weapon and a branch-point context C (typically after greedy closure),
we consider adding one candidate token t, then performing another greedy
closure until the next branching point. The resulting increase in context
length is the depth delta Δd(t).

We compute the expectation E_t~q[Δd(t)] under the candidate distribution q
(usually the normalized model probabilities over a candidate support).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Mapping

from splatnlp.mechinterp.analysis.information_gain import (
    add_token_to_capstones,
    tokens_to_capstones,
)
from splatnlp.utils.constants import NULL
from splatnlp.utils.reconstruct.beam_search import expand_capstones_to_context
from splatnlp.utils.reconstruct.classes import AbilityToken

PredictBatchFn = Callable[[list[list[str]], str], list[Mapping[str, float]]]


def greedy_closure_batched(
    predict_batch_fn: PredictBatchFn,
    *,
    weapon_id: str,
    capstones_batch: list[dict[str, AbilityToken]],
    threshold: float,
) -> list[dict[str, AbilityToken]]:
    """Batched version of `greedy_closure` for many parallel states.

    This intentionally mirrors `splatnlp.utils.reconstruct.beam_search.greedy_closure`
    semantics, but runs model inference in batches to reduce overhead.
    """
    if threshold <= 0.0 or threshold >= 1.0:
        raise ValueError("threshold must be in (0, 1)")

    capstones_batch = [dict(c) for c in capstones_batch]
    active = list(range(len(capstones_batch)))

    while active:
        contexts = [
            expand_capstones_to_context(capstones_batch[i]) for i in active
        ]
        probs_batch = predict_batch_fn(contexts, str(weapon_id))

        next_active: list[int] = []
        for idx, probs in zip(active, probs_batch):
            capstones = capstones_batch[idx]

            to_add: list[str] = []
            for tok, p in probs.items():
                if float(p) < float(threshold):
                    continue
                try:
                    next_cap = AbilityToken.from_vocab_entry(tok)
                except ValueError:
                    continue

                if next_cap.main_only:
                    if tok not in capstones:
                        to_add.append(tok)
                    continue

                existing = [
                    k
                    for k, v in capstones.items()
                    if v.family == next_cap.family
                ]
                if not existing:
                    to_add.append(tok)
                    continue

                should_add = any(
                    next_cap.min_ap > capstones[old_tok].min_ap
                    for old_tok in existing
                )
                if should_add:
                    to_add.append(tok)

            if not to_add:
                continue

            for tok in to_add:
                cap = AbilityToken.from_vocab_entry(tok)
                if cap.main_only:
                    capstones[tok] = cap
                    continue

                existing_tokens = [
                    k
                    for k, v in capstones.items()
                    if v.family == cap.family and v.min_ap < cap.min_ap
                ]
                for old_tok in existing_tokens:
                    del capstones[old_tok]
                capstones[tok] = cap

            capstones_batch[idx] = capstones
            next_active.append(idx)

        if not next_active:
            break
        active = next_active

    return capstones_batch


@dataclass(frozen=True)
class TokenDepthDeltaRow:
    token: str
    p0: float
    depth_after: int
    delta_depth: int
    expected_delta_depth: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeaponDepthDeltaResult:
    weapon_id: str
    weapon_label: str
    greedy_threshold: float
    branch_context: list[str]
    depth_before: int
    expected_delta_depth: float
    expected_depth_after: float
    ig: float
    ig_per_expected_delta_depth: float | None
    tokens: list[TokenDepthDeltaRow]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weapon_id": self.weapon_id,
            "weapon_label": self.weapon_label,
            "greedy_threshold": self.greedy_threshold,
            "branch_context": list(self.branch_context),
            "depth_before": self.depth_before,
            "expected_delta_depth": self.expected_delta_depth,
            "expected_depth_after": self.expected_depth_after,
            "information_gain": self.ig,
            "ig_per_expected_delta_depth": self.ig_per_expected_delta_depth,
            "tokens": [t.to_dict() for t in self.tokens],
        }


def _depth(tokens: list[str]) -> int:
    """Count "depth" as the number of non-NULL tokens.

    We treat the `<NULL>` sentinel as depth 0, so selecting any new token
    guarantees Δdepth >= 1.
    """
    return int(sum(1 for t in tokens if t != NULL))


def expected_depth_delta_for_weapon(
    *,
    predict_batch_fn: PredictBatchFn,
    weapon_id: str,
    weapon_label: str,
    greedy_threshold: float,
    branch_context: list[str],
    support0: list[str],
    p0: Mapping[str, float],
    information_gain: float,
    top_k_tokens: int = 15,
) -> WeaponDepthDeltaResult:
    if greedy_threshold <= 0.0 or greedy_threshold >= 1.0:
        raise ValueError("greedy_threshold must be in (0, 1)")

    depth_before = _depth(branch_context)
    if not support0:
        return WeaponDepthDeltaResult(
            weapon_id=str(weapon_id),
            weapon_label=str(weapon_label),
            greedy_threshold=float(greedy_threshold),
            branch_context=list(branch_context),
            depth_before=depth_before,
            expected_delta_depth=0.0,
            expected_depth_after=float(depth_before),
            ig=float(information_gain),
            ig_per_expected_delta_depth=None,
            tokens=[],
        )

    total = float(sum(float(p0.get(t, 0.0)) for t in support0))
    if total <= 0.0:
        raise ValueError("p0 has non-positive mass on support0")
    q = {t: float(p0.get(t, 0.0)) / total for t in support0}

    branch_capstones = tokens_to_capstones(branch_context)
    capstones_with_tok = [
        add_token_to_capstones(branch_capstones, tok) for tok in support0
    ]

    closed = greedy_closure_batched(
        predict_batch_fn,
        weapon_id=str(weapon_id),
        capstones_batch=capstones_with_tok,
        threshold=float(greedy_threshold),
    )

    rows: list[TokenDepthDeltaRow] = []
    exp_delta = 0.0
    for tok, caps in zip(support0, closed):
        depth_after = _depth(expand_capstones_to_context(caps))
        delta = int(depth_after - depth_before)
        contrib = float(q[tok] * float(delta))
        exp_delta += contrib
        rows.append(
            TokenDepthDeltaRow(
                token=tok,
                p0=float(q[tok]),
                depth_after=depth_after,
                delta_depth=delta,
                expected_delta_depth=contrib,
            )
        )

    rows.sort(key=lambda r: r.expected_delta_depth, reverse=True)
    if top_k_tokens > 0:
        rows_out = rows[: int(top_k_tokens)]
    else:
        rows_out = []

    denom = float(exp_delta)
    ratio = float(information_gain) / denom if denom > 0.0 else None

    return WeaponDepthDeltaResult(
        weapon_id=str(weapon_id),
        weapon_label=str(weapon_label),
        greedy_threshold=float(greedy_threshold),
        branch_context=list(branch_context),
        depth_before=depth_before,
        expected_delta_depth=float(exp_delta),
        expected_depth_after=float(depth_before) + float(exp_delta),
        ig=float(information_gain),
        ig_per_expected_delta_depth=ratio,
        tokens=rows_out,
    )
