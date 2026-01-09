from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import orjson
from scipy import stats

from splatnlp.eval.sendou_baseline import (
    ap_to_slot_items,
    load_sendou_builds,
    split_builds_by_tier,
    tokens_to_capstones,
)
from splatnlp.serve.tokenize import tokenize_build
from splatnlp.utils.constants import BUCKET_THRESHOLDS, NULL
from splatnlp.utils.reconstruct import Allocator

MAX_TOTAL_AP = 57
MAX_AP_L1 = MAX_TOTAL_AP * 2

RefBuild = tuple[dict[str, int], Counter[str]]
RefByWeapon = dict[str, list[RefBuild]]

_THRESHOLD_TO_INDEX = {int(t): int(i) for i, t in enumerate(BUCKET_THRESHOLDS)}


def _normalize_ap_dict(ap: dict[str, int] | None) -> dict[str, int] | None:
    if ap is None:
        return None
    return {str(k): int(v) for k, v in ap.items() if int(v) != 0}


def _ap_l1_diff(a: dict[str, int], b: dict[str, int]) -> int:
    keys = set(a) | set(b)
    return int(sum(abs(int(a.get(k, 0)) - int(b.get(k, 0))) for k in keys))


def _accuracy_from_ap(a: dict[str, int], b: dict[str, int]) -> float:
    return 1.0 - float(_ap_l1_diff(a, b)) / float(MAX_AP_L1)


def _slot_counter(abilities_ap: dict[str, int]) -> Counter[str]:
    return Counter(ap_to_slot_items(abilities_ap))


def _multiset_is_subset(need: Counter[str], have: Counter[str]) -> bool:
    return not (need - have)


def _completion_slot_accuracy_from_counters(
    *,
    truth: Counter[str],
    observed: Counter[str],
    pred: Counter[str],
) -> float:
    missing = truth - observed
    denom = sum(missing.values())
    if denom <= 0:
        return 1.0

    added = pred - observed
    correct = sum((added & missing).values())
    return float(correct) / float(denom)


def _observed_slot_recall_from_counters(
    *, observed: Counter[str], pred: Counter[str]
) -> float:
    denom = sum(observed.values())
    if denom <= 0:
        return 1.0
    correct = sum((pred & observed).values())
    return float(correct) / float(denom)


def _context_violation_from_counters(
    *, observed: Counter[str], pred: Counter[str] | None
) -> float:
    if pred is None:
        return 1.0 if sum(observed.values()) > 0 else 0.0
    return 0.0 if _multiset_is_subset(observed, pred) else 1.0


def _token_multiset(tokens: list[str]) -> Counter[str]:
    return Counter(str(t) for t in tokens if str(t) != NULL)


def _canonicalize_context_tokens(
    tokens: list[str],
    *,
    allocator: Allocator,
) -> list[str]:
    capstones = tokens_to_capstones(tokens)
    build, _ = allocator.allocate(capstones, priority={})
    if build is None:
        return list(tokens) if tokens else [NULL]
    return tokenize_build(build.achieved_ap)


def _edit_cost_against_lock(
    *,
    lock_tokens: list[str],
    pred_tokens: list[str],
) -> tuple[float, float]:
    """
    Measures how much a prediction *violates* an immutable lock context.

    This is intentionally asymmetric: additions are free (completion), while
    removals or tier downshifts of locked abilities incur cost.
    """
    lock_caps = tokens_to_capstones(lock_tokens)
    pred_caps = tokens_to_capstones(pred_tokens)

    lock_main_only = {c.family for c in lock_caps.values() if c.main_only}
    pred_main_only = {c.family for c in pred_caps.values() if c.main_only}

    lock_standard = {
        c.family: c.min_ap for c in lock_caps.values() if not c.main_only
    }
    pred_standard = {
        c.family: c.min_ap for c in pred_caps.values() if not c.main_only
    }

    removed_main_only = lock_main_only - pred_main_only
    removed_standard = set(lock_standard) - set(pred_standard)

    swap_cost = 0.0
    for a, b in [("run_speed_up", "swim_speed_up")]:
        if (
            a in removed_standard
            and b in pred_standard
            and b not in lock_standard
        ):
            removed_standard.remove(a)
            swap_cost += 3.0
        if (
            b in removed_standard
            and a in pred_standard
            and a not in lock_standard
        ):
            removed_standard.remove(b)
            swap_cost += 3.0

    tier_cost = 0.0
    for family in set(lock_standard) & set(pred_standard):
        lock_min = int(lock_standard[family])
        pred_min = int(pred_standard[family])
        if pred_min >= lock_min:
            continue
        lock_idx = _THRESHOLD_TO_INDEX.get(lock_min)
        pred_idx = _THRESHOLD_TO_INDEX.get(pred_min)
        if lock_idx is None or pred_idx is None:
            continue
        delta = abs(int(pred_idx) - int(lock_idx))
        tier_cost += float(delta) * 2.0

    cross_family = (
        1.0 if (removed_main_only or removed_standard or swap_cost > 0) else 0.0
    )

    cost = 0.0
    cost += float(len(removed_main_only)) * 5.0
    cost += float(len(removed_standard)) * 5.0
    cost += swap_cost
    cost += tier_cost
    return cost, cross_family


def _load_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def _signature(weapon_token: str, abilities_ap: dict[str, int]) -> tuple:
    return (
        weapon_token,
        tuple(sorted((str(k), int(v)) for k, v in abilities_ap.items())),
    )


def _completion_slot_accuracy(
    case: dict[str, Any], pred_ap: dict[str, int] | None
) -> float:
    if pred_ap is None:
        return 0.0

    truth = Counter(ap_to_slot_items(dict(case["truth_abilities_ap"])))
    observed = Counter(ap_to_slot_items(dict(case["masked_abilities_ap"])))
    pred = Counter(ap_to_slot_items(dict(pred_ap)))
    return _completion_slot_accuracy_from_counters(
        truth=truth,
        observed=observed,
        pred=pred,
    )


def _bootstrap_ci_mean(
    values: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    ci: float,
) -> tuple[float, float, float]:
    if values.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape {values.shape}")
    if len(values) == 0:
        raise ValueError("Cannot bootstrap empty array")

    n = len(values)
    idx = rng.integers(0, n, size=(n_resamples, n), endpoint=False)
    means = values[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        float(values.mean()),
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    )


def _bootstrap_ci_mean_diff(
    a: np.ndarray,
    b: np.ndarray,
    *,
    rng: np.random.Generator,
    n_resamples: int,
    ci: float,
) -> tuple[float, float, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    return _bootstrap_ci_mean(diff, rng=rng, n_resamples=n_resamples, ci=ci)


def _paired_tests(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: {a.shape} vs {b.shape}")
    diff = a - b
    if np.allclose(diff, 0.0):
        return {"ttest_p": 1.0, "wilcoxon_p": 1.0, "paired_d": 0.0}

    ttest = stats.ttest_rel(a, b, alternative="two-sided")
    try:
        wil = stats.wilcoxon(
            diff,
            alternative="two-sided",
            zero_method="zsplit",
            mode="approx",
        )
        wil_p = float(wil.pvalue)
    except ValueError:
        wil_p = 1.0

    denom = float(np.std(diff, ddof=1))
    paired_d = float(diff.mean() / denom) if denom > 0 else 0.0
    return {
        "ttest_p": float(ttest.pvalue),
        "wilcoxon_p": wil_p,
        "paired_d": paired_d,
    }


@dataclass(frozen=True)
class MethodSummary:
    method: str
    mask: int
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairwiseSummary:
    a: str
    b: str
    mask: int
    metric: str
    n: int
    mean_diff: float
    ci_low: float
    ci_high: float
    ttest_p: float
    wilcoxon_p: float
    paired_d: float


def analyze_compare_file(
    path: Path,
    *,
    metric_names: list[str],
    seed: int,
    n_resamples: int,
    ci: float,
    split_overlap: bool,
    csv_path: Path | None,
    weapon_vocab_path: Path | None,
) -> dict[str, Any]:
    payload = _load_json(path)
    meta = payload.get("meta") or {}
    cases = payload.get("cases") or []
    results = payload.get("results") or {}

    if (
        not isinstance(meta, dict)
        or not isinstance(cases, list)
        or not isinstance(results, dict)
    ):
        raise ValueError(f"Unexpected file shape for {path}")

    case_by_id = {int(c["case_id"]): c for c in cases}
    allocator = Allocator()
    case_cache: dict[int, dict[str, Any]] = {}
    for cid, c in case_by_id.items():
        truth_ap = _normalize_ap_dict(dict(c["truth_abilities_ap"])) or {}
        truth_slots = _slot_counter(truth_ap)

        observed_ap = _normalize_ap_dict(dict(c["masked_abilities_ap"])) or {}
        observed_slots = _slot_counter(observed_ap)

        context_tokens = list(c.get("context_tokens") or [])
        context_tokens_canon = _canonicalize_context_tokens(
            context_tokens,
            allocator=allocator,
        )

        case_cache[cid] = {
            "weapon_token": str(c["weapon_token"]),
            "truth_ap": truth_ap,
            "truth_slots": truth_slots,
            "observed_slots": observed_slots,
            "context_tokens": context_tokens,
            "context_tokens_canon": context_tokens_canon,
            "rt_token_violation": float(
                _token_multiset(context_tokens)
                != _token_multiset(context_tokens_canon)
            ),
        }

    overlap_by_case_id: dict[int, bool] | None = None
    ref_by_weapon: RefByWeapon | None = None
    needs_ref_set = any(m.startswith("tier1_set_") for m in metric_names)
    needs_coherence = any(
        m
        in (
            "top1_pmi_frankenstein_penalty",
            "top1_nn_jaccard_family",
        )
        for m in metric_names
    )
    needs_builds = split_overlap or needs_ref_set or needs_coherence

    train_builds = []
    eval_builds = []

    if needs_builds:
        csv_path_eff = csv_path or Path(
            str(meta.get("csv", "test_data/abilities-with-weapons.csv"))
        )
        weapon_vocab_eff = weapon_vocab_path or Path(
            str(
                meta.get(
                    "weapon_vocab",
                    "saved_models/dataset_v0_2_full/weapon_vocab.json",
                )
            )
        )
        weapon_vocab = _load_json(weapon_vocab_eff)
        builds = load_sendou_builds(csv_path_eff, weapon_vocab=weapon_vocab)
        train_builds, eval_builds = split_builds_by_tier(
            builds,
            train_tiers=list(meta.get("train_tiers", [2, 3])),
            eval_tiers=list(meta.get("eval_tiers", [1])),
        )

    if split_overlap:
        train_sigs = {
            _signature(r.weapon_token, r.abilities_ap) for r in train_builds
        }
        overlap_by_case_id = {}
        for cid, c in case_by_id.items():
            overlap_by_case_id[cid] = (
                _signature(
                    str(c["weapon_token"]), dict(c["truth_abilities_ap"])
                )
                in train_sigs
            )

    if needs_ref_set:
        ref_by_weapon = defaultdict(list)
        for row in eval_builds:
            ap = _normalize_ap_dict(dict(row.abilities_ap)) or {}
            slots = _slot_counter(ap)
            ref_by_weapon[str(row.weapon_token)].append((ap, slots))

    ref_family_sets_by_weapon: dict[str, list[set[str]]] | None = None
    pmi_cooccur_by_weapon: dict[str, dict[str, Any]] | None = None
    if needs_coherence:
        ref_family_sets_by_weapon = defaultdict(list)
        for row in eval_builds:
            ref_family_sets_by_weapon[str(row.weapon_token)].append(
                {
                    str(k)
                    for k, v in dict(row.abilities_ap).items()
                    if int(v) != 0
                }
            )

        eps = 1e-12
        n_by_weapon: Counter[str] = Counter()
        fam_counts_by_weapon: dict[str, Counter[str]] = defaultdict(Counter)
        pair_counts_by_weapon: dict[str, Counter[tuple[str, str]]] = (
            defaultdict(Counter)
        )

        for row in train_builds:
            weapon = str(row.weapon_token)
            fams = sorted(
                {
                    str(k)
                    for k, v in dict(row.abilities_ap).items()
                    if int(v) != 0
                }
            )
            if not fams:
                continue
            n_by_weapon[weapon] += 1
            fam_counts_by_weapon[weapon].update(fams)
            pair_counts_by_weapon[weapon].update(combinations(fams, 2))

        pmi_cooccur_by_weapon = {}
        for weapon, n in n_by_weapon.items():
            denom = float(n)
            p_a = {
                fam: float(cnt) / denom
                for fam, cnt in fam_counts_by_weapon[weapon].items()
            }
            p_ab = {
                pair: float(cnt) / denom
                for pair, cnt in pair_counts_by_weapon[weapon].items()
            }
            pmi_cooccur_by_weapon[weapon] = {
                "p_a": p_a,
                "p_ab": p_ab,
                "eps": eps,
            }

    rng = np.random.default_rng(seed)

    method_rows_by_case_id: dict[str, dict[int, dict[str, Any]]] = {}
    for method, method_payload in results.items():
        rows = (
            method_payload.get("rows")
            if isinstance(method_payload, dict)
            else None
        )
        if not isinstance(rows, list):
            continue
        method_rows_by_case_id[str(method)] = {
            int(r["case_id"]): r for r in rows
        }

    methods = sorted(method_rows_by_case_id.keys())
    if not methods:
        raise ValueError(f"No methods found in {path}")

    masks = sorted({int(c["ability_mask"]) for c in cases})

    out: dict[str, Any] = {"path": str(path), "meta": meta}
    out["methods"] = methods
    out["masks"] = masks
    out["metrics"] = metric_names

    all_ids = list(case_by_id.keys())
    group_specs: list[tuple[str, list[int]]] = [("all", all_ids)]
    if overlap_by_case_id is not None:
        overlap_ids = [cid for cid, flag in overlap_by_case_id.items() if flag]
        non_overlap_ids = [
            cid for cid, flag in overlap_by_case_id.items() if not flag
        ]
        group_specs = [("all", all_ids), ("no_overlap", non_overlap_ids)]
        group_specs.append(("overlap", overlap_ids))

    if "full" in method_rows_by_case_id and "ultra" in method_rows_by_case_id:
        full_rows = method_rows_by_case_id["full"]
        ultra_rows = method_rows_by_case_id["ultra"]
        divergent = []
        for cid in all_ids:
            full_row = full_rows.get(cid)
            ultra_row = ultra_rows.get(cid)
            if full_row is None or ultra_row is None:
                continue
            full_ap = _normalize_ap_dict(
                full_row.get("predicted_top1_achieved_ap")
            )
            ultra_ap = _normalize_ap_dict(
                ultra_row.get("predicted_top1_achieved_ap")
            )
            if full_ap != ultra_ap:
                divergent.append(cid)
        if divergent:
            divergent_set = set(divergent)
            group_specs.append(("full_ultra_divergent", divergent))
            if overlap_by_case_id is not None:
                group_specs.append(
                    (
                        "full_ultra_divergent_no_overlap",
                        [
                            cid
                            for cid in non_overlap_ids
                            if cid in divergent_set
                        ],
                    )
                )
                group_specs.append(
                    (
                        "full_ultra_divergent_overlap",
                        [cid for cid in overlap_ids if cid in divergent_set],
                    )
                )

    for group_name, group_case_ids in group_specs:
        group_summaries: list[MethodSummary] = []
        group_pairwise: list[PairwiseSummary] = []
        n_by_mask: dict[int, int] = {}

        for mask in masks:
            mask_case_ids = [
                cid
                for cid in group_case_ids
                if int(case_by_id[cid]["ability_mask"]) == mask
            ]
            if not mask_case_ids:
                continue

            n_by_mask[int(mask)] = len(mask_case_ids)

            for metric in metric_names:
                values_by_method: dict[str, np.ndarray] = {}
                for method in methods:
                    arr: list[float] = []
                    for cid in mask_case_ids:
                        row = method_rows_by_case_id[method].get(cid)
                        if row is None:
                            arr.append(0.0)
                            continue

                        cache = case_cache[cid]
                        truth_ap: dict[str, int] = cache["truth_ap"]
                        truth_slots: Counter[str] = cache["truth_slots"]
                        observed_slots: Counter[str] = cache["observed_slots"]

                        pred_best_ap = _normalize_ap_dict(
                            row.get("predicted_best_achieved_ap")
                        )
                        pred_top1_ap = _normalize_ap_dict(
                            row.get("predicted_top1_achieved_ap")
                        )
                        pred_best_slots = (
                            None
                            if pred_best_ap is None
                            else _slot_counter(pred_best_ap)
                        )
                        pred_top1_slots = (
                            None
                            if pred_top1_ap is None
                            else _slot_counter(pred_top1_ap)
                        )

                        if metric == "best_accuracy":
                            arr.append(float(row.get("best_accuracy", 0.0)))
                        elif metric == "top1_best_accuracy":
                            if pred_top1_ap is None:
                                arr.append(0.0)
                            else:
                                arr.append(
                                    _accuracy_from_ap(truth_ap, pred_top1_ap)
                                )
                        elif metric == "exact_hit":
                            arr.append(float(row.get("exact_hit", 0.0)))
                        elif metric == "top1_exact_hit":
                            arr.append(1.0 if pred_top1_ap == truth_ap else 0.0)
                        elif metric == "completion_slot_acc":
                            arr.append(
                                _completion_slot_accuracy(
                                    case_by_id[cid],
                                    row.get("predicted_best_achieved_ap"),
                                )
                            )
                        elif metric == "top1_completion_slot_acc":
                            if pred_top1_slots is None:
                                arr.append(0.0)
                            else:
                                arr.append(
                                    _completion_slot_accuracy_from_counters(
                                        truth=truth_slots,
                                        observed=observed_slots,
                                        pred=pred_top1_slots,
                                    )
                                )
                        elif metric == "top1_observed_slot_recall":
                            if pred_top1_slots is None:
                                arr.append(0.0)
                            else:
                                arr.append(
                                    _observed_slot_recall_from_counters(
                                        observed=observed_slots,
                                        pred=pred_top1_slots,
                                    )
                                )
                        elif metric in (
                            "top1_context_violation",
                            "top1_edit_chance",
                        ):
                            arr.append(
                                _context_violation_from_counters(
                                    observed=observed_slots,
                                    pred=pred_top1_slots,
                                )
                            )
                        elif metric == "rt_token_violation":
                            arr.append(
                                float(cache.get("rt_token_violation", 0.0))
                            )
                        elif metric in (
                            "top1_edit_cost",
                            "top1_cross_family_edit",
                        ):
                            lock_tokens = list(
                                cache.get("context_tokens_canon") or []
                            )
                            if pred_top1_ap is None:
                                pred_tokens = [NULL]
                            else:
                                pred_tokens = tokenize_build(pred_top1_ap)

                            cost, cross = _edit_cost_against_lock(
                                lock_tokens=lock_tokens,
                                pred_tokens=pred_tokens,
                            )
                            arr.append(
                                float(
                                    cost
                                    if metric == "top1_edit_cost"
                                    else cross
                                )
                            )
                        elif metric == "top1_nn_jaccard_family":
                            if ref_family_sets_by_weapon is None:
                                raise ValueError(
                                    "top1_nn_jaccard_family requires eval builds."
                                )
                            if pred_top1_ap is None:
                                arr.append(0.0)
                                continue
                            refs = ref_family_sets_by_weapon.get(
                                cache["weapon_token"], []
                            )
                            if not refs:
                                arr.append(0.0)
                                continue

                            fams_pred = {
                                str(k)
                                for k, v in pred_top1_ap.items()
                                if int(v) != 0
                            }
                            if not fams_pred:
                                arr.append(0.0)
                                continue
                            best = 0.0
                            for fams_ref in refs:
                                inter = len(fams_pred & fams_ref)
                                union = len(fams_pred | fams_ref)
                                if union <= 0:
                                    continue
                                best = max(best, float(inter) / float(union))
                            arr.append(float(best))
                        elif metric == "top1_pmi_frankenstein_penalty":
                            if pmi_cooccur_by_weapon is None:
                                raise ValueError(
                                    "top1_pmi_frankenstein_penalty requires train builds."
                                )
                            if pred_top1_ap is None:
                                arr.append(0.0)
                                continue
                            fams = sorted(
                                {
                                    str(k)
                                    for k, v in pred_top1_ap.items()
                                    if int(v) != 0
                                }
                            )
                            if len(fams) < 2:
                                arr.append(0.0)
                                continue

                            stats_w = pmi_cooccur_by_weapon.get(
                                cache["weapon_token"]
                            )
                            if stats_w is None:
                                arr.append(0.0)
                                continue
                            p_a = stats_w["p_a"]
                            p_ab = stats_w["p_ab"]
                            eps = float(stats_w["eps"])
                            penalty = 0.0
                            for a, b in combinations(fams, 2):
                                pa = float(p_a.get(a, 0.0))
                                pb = float(p_a.get(b, 0.0))
                                pab = float(p_ab.get((a, b), 0.0))
                                denom = (pa + eps) * (pb + eps)
                                if denom <= 0:
                                    continue
                                pmi = math.log((pab + eps) / denom)
                                if pmi < 0:
                                    penalty += -pmi
                            arr.append(float(penalty))
                        elif metric.startswith("tier1_set_"):
                            if ref_by_weapon is None:
                                raise ValueError(
                                    "tier1_set_* metrics require eval builds."
                                )

                            refs = ref_by_weapon.get(cache["weapon_token"], [])
                            consistent_refs: list[RefBuild] = []
                            for ref_ap, ref_slots in refs:
                                if _multiset_is_subset(
                                    observed_slots, ref_slots
                                ):
                                    consistent_refs.append((ref_ap, ref_slots))
                            if not consistent_refs:
                                arr.append(0.0)
                                continue

                            if metric == "tier1_set_best_accuracy":
                                if pred_best_ap is None:
                                    arr.append(0.0)
                                else:
                                    best = max(
                                        _accuracy_from_ap(pred_best_ap, ref_ap)
                                        for ref_ap, _ in consistent_refs
                                    )
                                    arr.append(float(best))
                            elif metric == "tier1_set_best_accuracy_top1":
                                if pred_top1_ap is None:
                                    arr.append(0.0)
                                else:
                                    best = max(
                                        _accuracy_from_ap(pred_top1_ap, ref_ap)
                                        for ref_ap, _ in consistent_refs
                                    )
                                    arr.append(float(best))
                            elif metric == "tier1_set_completion_slot_acc":
                                if pred_best_slots is None:
                                    arr.append(0.0)
                                else:
                                    best = max(
                                        _completion_slot_accuracy_from_counters(
                                            truth=ref_slots,
                                            observed=observed_slots,
                                            pred=pred_best_slots,
                                        )
                                        for _, ref_slots in consistent_refs
                                    )
                                    arr.append(float(best))
                            elif metric == "tier1_set_completion_slot_acc_top1":
                                if pred_top1_slots is None:
                                    arr.append(0.0)
                                else:
                                    best = max(
                                        _completion_slot_accuracy_from_counters(
                                            truth=ref_slots,
                                            observed=observed_slots,
                                            pred=pred_top1_slots,
                                        )
                                        for _, ref_slots in consistent_refs
                                    )
                                    arr.append(float(best))
                            elif metric == "tier1_set_exact_hit":
                                if pred_best_slots is None:
                                    arr.append(0.0)
                                else:
                                    hit = any(
                                        pred_best_slots == ref_slots
                                        for _, ref_slots in consistent_refs
                                    )
                                    arr.append(1.0 if hit else 0.0)
                            elif metric == "tier1_set_exact_hit_top1":
                                if pred_top1_slots is None:
                                    arr.append(0.0)
                                else:
                                    hit = any(
                                        pred_top1_slots == ref_slots
                                        for _, ref_slots in consistent_refs
                                    )
                                    arr.append(1.0 if hit else 0.0)
                            else:
                                raise ValueError(f"Unknown metric: {metric}")
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                    values_by_method[method] = np.array(arr, dtype=np.float64)

                for method, vals in values_by_method.items():
                    mean, lo, hi = _bootstrap_ci_mean(
                        vals,
                        rng=rng,
                        n_resamples=n_resamples,
                        ci=ci,
                    )
                    group_summaries.append(
                        MethodSummary(
                            method=method,
                            mask=mask,
                            metric=metric,
                            n=len(vals),
                            mean=mean,
                            ci_low=lo,
                            ci_high=hi,
                        )
                    )

                for i, a in enumerate(methods):
                    for b in methods[i + 1 :]:
                        av = values_by_method[a]
                        bv = values_by_method[b]
                        mean, lo, hi = _bootstrap_ci_mean_diff(
                            av,
                            bv,
                            rng=rng,
                            n_resamples=n_resamples,
                            ci=ci,
                        )
                        tests = _paired_tests(av, bv)
                        group_pairwise.append(
                            PairwiseSummary(
                                a=a,
                                b=b,
                                mask=mask,
                                metric=metric,
                                n=len(av),
                                mean_diff=mean,
                                ci_low=lo,
                                ci_high=hi,
                                **tests,
                            )
                        )

        out[group_name] = {
            "n_by_mask": dict(n_by_mask),
            "summaries": [s.__dict__ for s in group_summaries],
            "pairwise": [p.__dict__ for p in group_pairwise],
        }

    return out


def _print_report(report: dict[str, Any], *, ci: float) -> None:
    path = report["path"]
    methods = report["methods"]
    masks = report["masks"]
    metrics = report["metrics"]
    meta = report.get("meta", {})
    print(f"\n=== {path} ===")
    if isinstance(meta, dict):
        train_tiers = meta.get("train_tiers")
        eval_tiers = meta.get("eval_tiers")
        top_k = meta.get("top_k")
        print(
            f"train_tiers={train_tiers} eval_tiers={eval_tiers} top_k={top_k}"
        )
    print(f"methods={methods} masks={masks} metrics={metrics}")

    for group in [
        "all",
        "no_overlap",
        "overlap",
        "full_ultra_divergent",
        "full_ultra_divergent_no_overlap",
        "full_ultra_divergent_overlap",
    ]:
        if group not in report:
            continue
        print(f"\n[{group}]")
        summaries = report[group]["summaries"]
        pairwise = report[group]["pairwise"]

        by_key = {(s["mask"], s["metric"], s["method"]): s for s in summaries}
        for metric in metrics:
            print(f"\nmetric={metric} (mean, {int(ci*100)}% CI)")
            for mask in masks:
                parts = []
                for method in methods:
                    s = by_key.get((mask, metric, method))
                    if s is None:
                        continue
                    parts.append(
                        f"{method}={s['mean']:.4f} "
                        f"[{s['ci_low']:.4f},{s['ci_high']:.4f}]"
                    )
                joined = "  ".join(parts)
                print(f" mask={mask}: {joined}")

        print("\npairwise mean diffs (A-B) w/ p-values")
        for metric in metrics:
            print(f"\nmetric={metric}")
            for mask in masks:
                rows = [
                    p
                    for p in pairwise
                    if int(p["mask"]) == int(mask)
                    and str(p["metric"]) == metric
                ]
                rows.sort(
                    key=lambda r: abs(float(r["mean_diff"])), reverse=True
                )
                for r in rows:
                    print(
                        f" mask={mask} {r['a']}-{r['b']} "
                        f"diff={r['mean_diff']:.4f} "
                        f"[{r['ci_low']:.4f},{r['ci_high']:.4f}] "
                        f"ttest_p={r['ttest_p']:.2e} "
                        f"wilcoxon_p={r['wilcoxon_p']:.2e}"
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in",
        dest="inputs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more sendou_compare_*.json files.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["best_accuracy", "completion_slot_acc"],
        choices=[
            "best_accuracy",
            "completion_slot_acc",
            "exact_hit",
            "top1_best_accuracy",
            "top1_completion_slot_acc",
            "top1_exact_hit",
            "top1_observed_slot_recall",
            "top1_context_violation",
            "top1_edit_chance",
            "rt_token_violation",
            "top1_edit_cost",
            "top1_cross_family_edit",
            "top1_pmi_frankenstein_penalty",
            "top1_nn_jaccard_family",
            "tier1_set_best_accuracy",
            "tier1_set_best_accuracy_top1",
            "tier1_set_completion_slot_acc",
            "tier1_set_completion_slot_acc_top1",
            "tier1_set_exact_hit",
            "tier1_set_exact_hit_top1",
        ],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument(
        "--split-overlap",
        action="store_true",
        help="Report stats for cases whose exact (weapon, build) appears in "
        "the train tiers vs not.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Sendou CSV (defaults to meta['csv']).",
    )
    parser.add_argument(
        "--weapon-vocab",
        type=Path,
        default=None,
        help="weapon_vocab.json (defaults to meta['weapon_vocab']).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path (writes combined report).",
    )
    args = parser.parse_args()

    combined: dict[str, Any] = {
        "inputs": [str(p) for p in args.inputs],
        "reports": [],
    }

    for path in args.inputs:
        report = analyze_compare_file(
            path,
            metric_names=list(args.metrics),
            seed=int(args.seed),
            n_resamples=int(args.bootstrap),
            ci=float(args.ci),
            split_overlap=bool(args.split_overlap),
            csv_path=args.csv,
            weapon_vocab_path=args.weapon_vocab,
        )
        combined["reports"].append(report)
        _print_report(report, ci=float(args.ci))

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(
            orjson.dumps(
                combined,
                option=orjson.OPT_INDENT_2 | orjson.OPT_NON_STR_KEYS,
            )
        )
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
