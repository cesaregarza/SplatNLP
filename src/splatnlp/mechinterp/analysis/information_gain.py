"""Information gain utilities for SetCompletionModel outputs.

The SetCompletionModel produces independent sigmoid probabilities per token,
not a categorical distribution. For entropy-based analyses we first normalize
the non-special token probabilities into a categorical distribution.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Literal, Mapping

import torch

from splatnlp.model.models import SetCompletionModel
from splatnlp.utils.constants import NULL
from splatnlp.utils.infer import build_predict_abilities_batch
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import (
    expand_capstones_to_context,
    greedy_closure,
)
from splatnlp.utils.reconstruct.classes import AbilityToken

StackPolicy = Literal["none", "next-tier", "max-tier"]


def allowed_tokens_from_vocab(vocab: Mapping[str, int]) -> list[str]:
    """Return vocabulary tokens eligible for entropy calculations."""
    return [t for t in vocab.keys() if not t.startswith("<")]


def normalize_weights(
    weights: Mapping[str, float], *, support: Iterable[str]
) -> dict[str, float]:
    """Normalize positive weights into a categorical distribution."""
    tokens = list(support)
    total = sum(float(weights[t]) for t in tokens)
    if total <= 0:
        raise ValueError("Cannot normalize: total weight is <= 0")
    return {t: float(weights[t]) / total for t in tokens}


def shannon_entropy(
    probs: Mapping[str, float], *, log_base: float = 2.0
) -> float:
    """Compute Shannon entropy for a categorical distribution.

    Args:
        probs: Mapping of outcome -> probability, must sum to ~1.
        log_base: Log base for the entropy units (2 -> bits, e -> nats).
    """
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be > 0 and != 1")

    inv_log_base = 1.0 / math.log(log_base)
    h = 0.0
    for p in probs.values():
        p = float(p)
        if p <= 0.0:
            continue
        h -= p * (math.log(p) * inv_log_base)
    return float(h)


def entropy_from_probs_tensor(
    probs: torch.Tensor,
    *,
    log_base: float = 2.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Compute Shannon entropy row-wise for a categorical distribution tensor.

    Args:
        probs: Tensor of shape (batch, n) whose rows sum to 1.
        log_base: Log base for entropy units (2 -> bits, e -> nats).
        eps: Clamp minimum to avoid log(0) when probabilities underflow to 0.
    """
    if log_base <= 0 or log_base == 1:
        raise ValueError("log_base must be > 0 and != 1")
    if probs.ndim != 2:
        raise ValueError("probs must have shape (batch, n)")

    denom = float(math.log(log_base))
    if denom == 0.0:
        raise ValueError("Invalid log_base")

    p = probs.to(dtype=torch.float64)
    p = torch.clamp(p, min=float(eps))
    h_nats = -(p * torch.log(p)).sum(dim=1)
    return h_nats / denom


def tokens_to_capstones(tokens: Iterable[str]) -> dict[str, AbilityToken]:
    capstones: dict[str, AbilityToken] = {}
    for tok in tokens:
        if tok == NULL:
            continue
        try:
            cap = AbilityToken.from_vocab_entry(tok)
        except ValueError:
            continue

        if cap.main_only:
            capstones[tok] = cap
            continue

        existing = [k for k, v in capstones.items() if v.family == cap.family]
        if not existing:
            capstones[tok] = cap
            continue

        best_old = max(capstones[k].min_ap for k in existing)
        if cap.min_ap <= best_old:
            continue

        for old_tok in existing:
            del capstones[old_tok]
        capstones[tok] = cap
    return capstones


def add_token_to_capstones(
    capstones: Mapping[str, AbilityToken], tok: str
) -> dict[str, AbilityToken]:
    next_cap = AbilityToken.from_vocab_entry(tok)
    updated = dict(capstones)

    if next_cap.main_only:
        updated.setdefault(tok, next_cap)
        return updated

    existing = [k for k, v in updated.items() if v.family == next_cap.family]
    if not existing:
        updated[tok] = next_cap
        return updated

    best_old = max(updated[k].min_ap for k in existing)
    if next_cap.min_ap <= best_old:
        return updated

    for old_tok in existing:
        del updated[old_tok]
    updated[tok] = next_cap
    return updated


def stacking_family_levels(tokens: Iterable[str]) -> dict[str, list[int]]:
    levels_by_family: dict[str, set[int]] = defaultdict(set)
    for tok in tokens:
        try:
            cap = AbilityToken.from_vocab_entry(tok)
        except ValueError:
            continue
        if cap.main_only:
            continue
        levels_by_family[cap.family].add(int(cap.min_ap))
    return {fam: sorted(levels) for fam, levels in levels_by_family.items()}


def support_tokens_for_state(
    *,
    allowed_tokens: list[str],
    conditioned_tokens: set[str],
    omit_conditioned: bool,
    stack_policy: StackPolicy,
    levels_by_family: Mapping[str, list[int]],
) -> list[str]:
    if omit_conditioned:
        base_support = [
            t for t in allowed_tokens if t not in conditioned_tokens
        ]
    else:
        base_support = list(allowed_tokens)

    if stack_policy == "none":
        return base_support

    current_level: dict[str, int] = {}
    for tok in conditioned_tokens:
        try:
            cap = AbilityToken.from_vocab_entry(tok)
        except ValueError:
            continue
        if cap.main_only:
            continue
        current_level[cap.family] = max(
            int(cap.min_ap), current_level.get(cap.family, 0)
        )

    out: list[str] = []
    for tok in base_support:
        try:
            cap = AbilityToken.from_vocab_entry(tok)
        except ValueError:
            out.append(tok)
            continue

        if cap.main_only:
            out.append(tok)
            continue

        levels = levels_by_family.get(cap.family)
        if not levels:
            out.append(tok)
            continue

        if stack_policy == "next-tier":
            current = current_level.get(cap.family, 0)
            next_level = next((lvl for lvl in levels if lvl > current), None)
            if next_level is None:
                continue
            if tok == f"{cap.family}_{int(next_level)}":
                out.append(tok)
            continue

        if stack_policy == "max-tier":
            max_level = int(levels[-1])
            if tok == f"{cap.family}_{max_level}":
                out.append(tok)
            continue

        raise ValueError(f"Unknown stack_policy: {stack_policy}")

    return out


def information_gain_matrix(
    *,
    model: SetCompletionModel,
    vocab: Mapping[str, int],
    weapon_vocab: Mapping[str, int],
    weapon_ids: list[str] | None = None,
    pair_batch_size: int = 1024,
    log_base: float = 2.0,
    base_context: list[str] | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compute IG matrices for many weapons efficiently.

    Returns a dict with:
      - weapon_ids: list[str]
      - tokens: list[str] (non-<...> vocab tokens, sorted)
      - p0: float matrix (n_weapons, n_tokens)
      - h0: float vector (n_weapons,)
      - h1: float matrix (n_weapons, n_tokens)
      - expected_h1: float vector (n_weapons,)
      - information_gain: float vector (n_weapons,)
    """
    if base_context is None:
        base_context = [NULL]
    if device is None:
        device = torch.device("cpu")
    if pair_batch_size <= 0:
        raise ValueError("pair_batch_size must be > 0")

    tokens = sorted(allowed_tokens_from_vocab(vocab))
    if not tokens:
        raise ValueError("No allowed tokens in vocab")

    if weapon_ids is None:
        weapon_ids = sorted(weapon_vocab.keys())
    else:
        weapon_ids = list(weapon_ids)
    if not weapon_ids:
        raise ValueError("weapon_ids must be non-empty")

    pad_id = int(vocab["<PAD>"])
    token_ids = [int(vocab[t]) for t in tokens]
    token_ids_t = torch.tensor(token_ids, device=device, dtype=torch.long)

    # Baseline distribution p0 per weapon from base_context.
    base_ids = [int(vocab[t]) for t in base_context]
    base_tokens = torch.tensor(base_ids, device=device, dtype=torch.long)
    input_tokens = base_tokens.unsqueeze(0).repeat(len(weapon_ids), 1)
    input_weapons = torch.tensor(
        [int(weapon_vocab[w]) for w in weapon_ids],
        device=device,
        dtype=torch.long,
    ).unsqueeze(1)
    key_padding_mask = input_tokens == pad_id

    with torch.no_grad():
        outputs = model(
            input_tokens, input_weapons, key_padding_mask=key_padding_mask
        )
        probs0 = torch.sigmoid(outputs)

    weights0 = probs0.index_select(1, token_ids_t).to(dtype=torch.float64)
    denom0 = weights0.sum(dim=1, keepdim=True)
    if torch.any(denom0 <= 0):
        raise ValueError("Encountered non-positive normalization mass for p0")
    p0 = weights0 / denom0
    h0 = entropy_from_probs_tensor(p0, log_base=log_base)

    # Precompute token contexts (expanded prerequisite tiers) as vocab IDs.
    token_context_ids: list[list[int]] = []
    for tok in tokens:
        cap = AbilityToken.from_vocab_entry(tok)
        ctx_tokens = expand_capstones_to_context({tok: cap})
        token_context_ids.append([int(vocab[t]) for t in ctx_tokens])
    max_ctx_len = max(len(ctx) for ctx in token_context_ids)
    token_context_matrix = torch.full(
        (len(tokens), max_ctx_len),
        pad_id,
        device=device,
        dtype=torch.long,
    )
    for idx, ctx in enumerate(token_context_ids):
        if not ctx:
            continue
        token_context_matrix[idx, : len(ctx)] = torch.tensor(
            ctx, device=device, dtype=torch.long
        )

    n_weapons = len(weapon_ids)
    n_tokens = len(tokens)
    h1 = torch.empty((n_weapons, n_tokens), dtype=torch.float64)

    weapon_idx = torch.arange(n_weapons, device=device).repeat_interleave(
        n_tokens
    )
    token_idx = torch.arange(n_tokens, device=device).repeat(n_weapons)
    total_pairs = int(weapon_idx.numel())

    for start in range(0, total_pairs, pair_batch_size):
        end = min(total_pairs, start + pair_batch_size)
        wi = weapon_idx[start:end]
        ti = token_idx[start:end]

        batch_tokens = token_context_matrix.index_select(0, ti)
        batch_weapons = input_weapons.index_select(0, wi)
        batch_mask = batch_tokens == pad_id
        with torch.no_grad():
            outputs = model(
                batch_tokens,
                batch_weapons,
                key_padding_mask=batch_mask,
            )
            probs1 = torch.sigmoid(outputs)

        weights1 = probs1.index_select(1, token_ids_t).to(dtype=torch.float64)
        denom1 = weights1.sum(dim=1, keepdim=True)
        if torch.any(denom1 <= 0):
            raise ValueError(
                "Encountered non-positive normalization mass for p1"
            )
        p1 = weights1 / denom1
        h1_batch = entropy_from_probs_tensor(p1, log_base=log_base)
        h1[wi.cpu(), ti.cpu()] = h1_batch.detach().cpu()

    expected_h1 = (p0.cpu() * h1).sum(dim=1)
    information_gain = h0.cpu() - expected_h1

    return {
        "weapon_ids": list(weapon_ids),
        "tokens": list(tokens),
        "p0": p0.cpu(),
        "h0": h0.cpu(),
        "h1": h1,
        "expected_h1": expected_h1,
        "information_gain": information_gain,
    }


@dataclass(frozen=True)
class TokenInfoGainRow:
    token: str
    p0: float
    h1: float
    expected_h1: float
    ig_contribution: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WeaponInfoGainResult:
    weapon_id: str
    log_base: float
    base_context: list[str]
    greedy_threshold: float
    omit_conditioned: bool
    stack_policy: StackPolicy
    branch_context: list[str]
    h0: float
    expected_h1: float
    information_gain: float
    tokens: list[TokenInfoGainRow]
    family_p0_mass: dict[str, float]
    family_ig_contribution: dict[str, float]
    family_ig_contribution_pct: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "weapon_id": self.weapon_id,
            "log_base": self.log_base,
            "base_context": list(self.base_context),
            "greedy_threshold": self.greedy_threshold,
            "omit_conditioned": self.omit_conditioned,
            "stack_policy": self.stack_policy,
            "branch_context": list(self.branch_context),
            "h0": self.h0,
            "expected_h1": self.expected_h1,
            "information_gain": self.information_gain,
            "tokens": [t.to_dict() for t in self.tokens],
            "family_p0_mass": dict(self.family_p0_mass),
            "family_ig_contribution": dict(self.family_ig_contribution),
            "family_ig_contribution_pct": dict(self.family_ig_contribution_pct),
        }


def information_gain_for_weapon(
    *,
    model: SetCompletionModel,
    weapon_id: str,
    vocab: Mapping[str, int],
    weapon_vocab: Mapping[str, int],
    batch_size: int = 64,
    log_base: float = 2.0,
    base_context: list[str] | None = None,
    tokens: list[str] | None = None,
    greedy_threshold: float = 0.7,
    omit_conditioned: bool = True,
    stack_policy: StackPolicy = "none",
    device: torch.device | None = None,
) -> WeaponInfoGainResult:
    """Compute H0 - E[H1] for a weapon from a <NULL> (or provided) context.

    We first greedy-close the base context (using ``greedy_threshold``) until
    reaching a branching point, then compute entropy on the remaining (i.e.,
    unconditioned) tokens when ``omit_conditioned`` is True.
    """
    if base_context is None:
        base_context = [NULL]
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if greedy_threshold <= 0.0 or greedy_threshold >= 1.0:
        raise ValueError("greedy_threshold must be in (0, 1)")
    if device is None:
        device = torch.device("cpu")

    allowed_tokens = (
        tokens if tokens is not None else allowed_tokens_from_vocab(vocab)
    )
    if not allowed_tokens:
        raise ValueError("No allowed tokens provided")

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
        return WeaponInfoGainResult(
            weapon_id=str(weapon_id),
            log_base=float(log_base),
            base_context=list(base_context),
            greedy_threshold=float(greedy_threshold),
            omit_conditioned=bool(omit_conditioned),
            stack_policy=stack_policy,
            branch_context=list(branch_context),
            h0=0.0,
            expected_h1=0.0,
            information_gain=0.0,
            tokens=[],
            family_p0_mass={},
            family_ig_contribution={},
            family_ig_contribution_pct={},
        )

    base_probs_raw = predict_batch(model, [list(branch_context)], weapon_id)[0]
    p0 = normalize_weights(base_probs_raw, support=support0)
    h0 = shannon_entropy(p0, log_base=log_base)

    contexts: list[list[str]] = []
    conditioned_sets: list[set[str]] = []
    for tok in support0:
        capstones_with_tok = add_token_to_capstones(branch_capstones, tok)
        ctx = expand_capstones_to_context(capstones_with_tok)
        contexts.append(ctx)
        if omit_conditioned:
            ctx_set = set(ctx)
            ctx_set.discard(NULL)
            conditioned_sets.append(ctx_set)
        else:
            conditioned_sets.append(set())

    h1_by_token: dict[str, float] = {}
    for start in range(0, len(support0), batch_size):
        chunk_tokens = support0[start : start + batch_size]
        chunk_contexts = contexts[start : start + batch_size]
        chunk_conditioned = conditioned_sets[start : start + batch_size]
        chunk_probs = predict_batch(model, chunk_contexts, weapon_id)
        for tok, probs_raw, cond in zip(
            chunk_tokens, chunk_probs, chunk_conditioned
        ):
            support1 = support_tokens_for_state(
                allowed_tokens=list(allowed_tokens),
                conditioned_tokens=cond,
                omit_conditioned=omit_conditioned,
                stack_policy=stack_policy,
                levels_by_family=levels_by_family,
            )
            if not support1:
                h1_by_token[tok] = 0.0
                continue
            p1 = normalize_weights(probs_raw, support=support1)
            h1_by_token[tok] = shannon_entropy(p1, log_base=log_base)

    expected_h1 = sum(p0[tok] * h1_by_token[tok] for tok in support0)
    ig = h0 - expected_h1

    rows: list[TokenInfoGainRow] = []
    for tok in support0:
        p = float(p0[tok])
        h1 = float(h1_by_token[tok])
        rows.append(
            TokenInfoGainRow(
                token=tok,
                p0=p,
                h1=h1,
                expected_h1=p * h1,
                ig_contribution=p * (h0 - h1),
            )
        )

    rows.sort(key=lambda r: r.ig_contribution, reverse=True)

    family_p0_mass: dict[str, float] = defaultdict(float)
    family_ig_contribution: dict[str, float] = defaultdict(float)
    for row in rows:
        try:
            cap = AbilityToken.from_vocab_entry(row.token)
            family = cap.family
        except ValueError:
            family = row.token
        family_p0_mass[family] += float(row.p0)
        family_ig_contribution[family] += float(row.ig_contribution)

    ig_total = float(ig)
    family_ig_pct: dict[str, float] = (
        {
            family: float(value) / ig_total
            for family, value in family_ig_contribution.items()
        }
        if ig_total != 0.0
        else {}
    )

    return WeaponInfoGainResult(
        weapon_id=str(weapon_id),
        log_base=float(log_base),
        base_context=list(base_context),
        greedy_threshold=float(greedy_threshold),
        omit_conditioned=bool(omit_conditioned),
        stack_policy=stack_policy,
        branch_context=list(branch_context),
        h0=float(h0),
        expected_h1=float(expected_h1),
        information_gain=float(ig),
        tokens=rows,
        family_p0_mass=dict(family_p0_mass),
        family_ig_contribution=dict(family_ig_contribution),
        family_ig_contribution_pct=dict(family_ig_pct),
    )
