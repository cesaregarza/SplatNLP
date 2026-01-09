from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import orjson
import pandas as pd
import torch

from splatnlp.eval.sendou_baseline import (
    ap_to_slot_items,
    compare_builds,
    evaluate_top_k,
    load_sendou_builds,
    mask_abilities,
    slot_items_to_ap,
    slots_to_build,
    split_builds_by_tier,
)
from splatnlp.eval.sendou_mode_baseline import (
    build_mode_index,
    choose_mode_completion,
)
from splatnlp.model.models import SetCompletionModel
from splatnlp.serve.tokenize import tokenize_build
from splatnlp.utils.constants import (
    CANONICAL_MAIN_ONLY_ABILITIES,
    MAIN_ONLY_ABILITIES,
    NULL,
    STANDARD_ABILITIES,
)
from splatnlp.utils.infer import build_predict_abilities_batch_multiweapon
from splatnlp.utils.reconstruct import Allocator, reconstruct_builds_batched
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _load_model(
    checkpoint: Path,
    *,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    device: torch.device,
) -> SetCompletionModel:
    model = SetCompletionModel(
        len(vocab),
        len(weapon_vocab),
        32,
        512,
        len(vocab),
        num_layers=3,
        num_heads=8,
        num_inducing_points=32,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=vocab["<PAD>"],
    )
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("ability_mask")
        .agg(
            exact_hit_rate=("exact_hit", "mean"),
            avg_correct_out=("num_correct_out", "mean"),
            avg_correct_best=("num_correct_best", "mean"),
            avg_top1_accuracy=("accuracy", "mean"),
            avg_best_accuracy=("best_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.to_dict(orient="records")


def _truth_build(row: dict[str, Any]) -> Build:
    return slots_to_build(ap_to_slot_items(dict(row["truth_abilities_ap"])))


def build_eval_cases(
    eval_builds: list,
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
    nested_masks: bool,
) -> tuple[list[dict[str, Any]], dict[int, Build]]:
    rng = torch.Generator().manual_seed(seed)

    truth_builds: dict[tuple[int, int, int], Build] = {}
    for row in eval_builds:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_builds:
            continue
        truth_builds[key] = slots_to_build(ap_to_slot_items(row.abilities_ap))

    cases: list[dict[str, Any]] = []
    truth_by_case_id: dict[int, Build] = {}
    case_id = 0

    rng_py = random.Random(
        int(torch.randint(0, 2**31 - 1, (1,), generator=rng))
    )

    masks = sorted(int(m) for m in masks)
    if nested_masks:
        candidates = list(eval_builds)
        if limit_per_mask is not None:
            candidates = rng_py.sample(
                candidates, min(int(limit_per_mask), len(candidates))
            )

        for row in candidates:
            slot_items = ap_to_slot_items(row.abilities_ap)
            row_seed = (
                int(seed) * 1_000_003
                + int(row.build_id) * 97_999
                + int(row.plus_tier) * 1_009
                + int(row.weapon_id)
            ) & 0xFFFFFFFF
            rng_row = random.Random(int(row_seed))
            rng_row.shuffle(slot_items)

            truth = truth_builds[(row.build_id, row.plus_tier, row.weapon_id)]

            for mask in masks:
                if int(mask) > len(slot_items):
                    raise ValueError(
                        f"Mask={mask} exceeds slot count={len(slot_items)} "
                        f"for build_id={row.build_id}"
                    )
                kept = slot_items[int(mask) :]
                masked_abilities = slot_items_to_ap(list(kept))
                context_tokens = tokenize_build(masked_abilities)

                cases.append(
                    {
                        "case_id": case_id,
                        "build_id": row.build_id,
                        "plus_tier": row.plus_tier,
                        "weapon_id": row.weapon_id,
                        "weapon_token": row.weapon_token,
                        "ability_mask": int(mask),
                        "truth_abilities_ap": dict(row.abilities_ap),
                        "masked_abilities_ap": dict(masked_abilities),
                        "context_tokens": list(context_tokens),
                        "truth_achieved_ap": dict(truth.achieved_ap),
                    }
                )
                truth_by_case_id[case_id] = truth
                case_id += 1
    else:
        for mask in masks:
            candidates = list(eval_builds)
            if limit_per_mask is not None:
                candidates = rng_py.sample(
                    candidates, min(int(limit_per_mask), len(candidates))
                )

            for row in candidates:
                masked_abilities = mask_abilities(
                    row.abilities_ap, int(mask), rng_py
                )
                context_tokens = tokenize_build(masked_abilities)
                truth = truth_builds[
                    (row.build_id, row.plus_tier, row.weapon_id)
                ]

                cases.append(
                    {
                        "case_id": case_id,
                        "build_id": row.build_id,
                        "plus_tier": row.plus_tier,
                        "weapon_id": row.weapon_id,
                        "weapon_token": row.weapon_token,
                        "ability_mask": int(mask),
                        "truth_abilities_ap": dict(row.abilities_ap),
                        "masked_abilities_ap": dict(masked_abilities),
                        "context_tokens": list(context_tokens),
                        "truth_achieved_ap": dict(truth.achieved_ap),
                    }
                )
                truth_by_case_id[case_id] = truth
                case_id += 1

    return cases, truth_by_case_id


def _capstones_from_tokens(tokens: Iterable[str]) -> dict[str, AbilityToken]:
    capstones: dict[str, AbilityToken] = {}
    for tok in tokens:
        if tok == NULL:
            continue
        try:
            capstones[tok] = AbilityToken.from_vocab_entry(tok)
        except ValueError:
            continue
    return capstones


def _canonicalize_context_tokens(
    tokens: list[str],
    *,
    allocator: Allocator,
) -> list[str]:
    capstones = _capstones_from_tokens(tokens)
    build, _ = allocator.allocate(capstones, priority={})
    if build is None:
        return list(tokens) if tokens else [NULL]
    return tokenize_build(build.achieved_ap)


def _try_add_token(
    capstones: dict[str, AbilityToken],
    token: str,
) -> bool:
    try:
        cap = AbilityToken.from_vocab_entry(token)
    except ValueError:
        return False

    if cap.main_only:
        if token in capstones:
            return False
        capstones[token] = cap
        return True

    existing = [k for k, v in capstones.items() if v.family == cap.family]
    if not existing:
        capstones[token] = cap
        return True

    improves = any(cap.min_ap > capstones[old].min_ap for old in existing)
    if not improves:
        return False

    for old in existing:
        if capstones[old].min_ap < cap.min_ap:
            del capstones[old]
    capstones[token] = cap
    return True


def _is_full_build(build: Build) -> bool:
    mains_full = all(build.mains[slot] is not None for slot in Build.GEAR_SLOTS)
    subs_full = sum(build.subs.values()) == Build.MAX_SUB_SLOTS_TOTAL
    return mains_full and subs_full


def run_random_completion(
    cases: list[dict[str, Any]],
    truth_by_case_id: Mapping[int, Build],
    *,
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    t0 = time.time()

    standard_families = list(STANDARD_ABILITIES)
    main_only_families = list(MAIN_ONLY_ABILITIES)

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        ability_mask = int(case["ability_mask"])
        masked_abilities_ap = dict(case["masked_abilities_ap"])
        context_items = ap_to_slot_items(masked_abilities_ap)

        rng = random.Random((int(seed) * 1_000_003 + case_id) & 0xFFFFFFFF)

        context_mains = [it for it in context_items if it.endswith("_main")]
        context_subs = [it for it in context_items if it.endswith("_sub")]

        missing_mains = 3 - len(context_mains)
        missing_subs = 9 - len(context_subs)
        if missing_mains < 0 or missing_subs < 0:
            raise ValueError(
                f"Invalid masked build for case_id={case_id}: "
                f"mains={len(context_mains)}, subs={len(context_subs)}"
            )
        if missing_mains + missing_subs != ability_mask:
            raise ValueError(
                f"Mask mismatch for case_id={case_id}: "
                f"expected={ability_mask}, "
                f"missing={missing_mains + missing_subs}"
            )

        taken_slots = set()
        main_families = [m[:-5] for m in context_mains]
        for family in main_families:
            if family in CANONICAL_MAIN_ONLY_ABILITIES:
                taken_slots.add(CANONICAL_MAIN_ONLY_ABILITIES[family])

        for _ in range(missing_mains):
            available_main_only = [
                fam
                for fam in main_only_families
                if CANONICAL_MAIN_ONLY_ABILITIES[fam] not in taken_slots
            ]
            candidates = standard_families + available_main_only
            if not candidates:
                raise ValueError(
                    f"No candidates to fill mains for case_id={case_id}"
                )
            family = rng.choice(candidates)
            main_families.append(family)
            if family in CANONICAL_MAIN_ONLY_ABILITIES:
                taken_slots.add(CANONICAL_MAIN_ONLY_ABILITIES[family])

        sub_families = [s[:-4] for s in context_subs]
        for _ in range(missing_subs):
            sub_families.append(rng.choice(standard_families))

        pred_slot_items = [f"{fam}_main" for fam in main_families] + [
            f"{fam}_sub" for fam in sub_families
        ]
        pred_build = slots_to_build(pred_slot_items)

        truth = truth_by_case_id[case_id]
        rows.append(_row_from_pred_builds(case, truth, [pred_build]))

    dt = time.time() - t0
    return rows, dt


def run_conditional_baseline(
    cases: list[dict[str, Any]],
    truth_by_case_id: Mapping[int, Build],
    *,
    train_builds: list,
) -> tuple[list[dict[str, Any]], float]:
    t0 = time.time()
    mode_builds, candidates = build_mode_index(train_builds)

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        weapon_token = str(case["weapon_token"])
        masked_abilities_ap = dict(case["masked_abilities_ap"])
        context_slots = Counter(ap_to_slot_items(masked_abilities_ap))
        pred_build = choose_mode_completion(
            weapon_token,
            context_slots,
            mode_builds=mode_builds,
            candidates=candidates,
            conditional=True,
        )

        truth = truth_by_case_id[case_id]
        rows.append(_row_from_pred_builds(case, truth, [pred_build]))

    dt = time.time() - t0
    return rows, dt


def oneshot_capstones_threshold(
    *,
    probs: Mapping[str, float],
    context_tokens: list[str],
    threshold: float,
) -> tuple[dict[str, AbilityToken], dict[str, float]]:
    capstones = _capstones_from_tokens(context_tokens)
    family_logp: dict[str, float] = {}

    for tok, cap in capstones.items():
        p = float(probs.get(tok, 0.0))
        family_logp[cap.family] = max(family_logp.get(cap.family, 0.0), p)

    for tok, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        if float(p) < float(threshold):
            break
        changed = _try_add_token(capstones, tok)
        if changed:
            cap = AbilityToken.from_vocab_entry(tok)
            family_logp[cap.family] = max(
                family_logp.get(cap.family, 0.0), float(p)
            )

    return capstones, family_logp


def oneshot_rank_until_full(
    *,
    probs: Mapping[str, float],
    context_tokens: list[str],
    allocator: Allocator,
) -> tuple[Build | None, dict[str, AbilityToken], dict[str, float], int]:
    capstones = _capstones_from_tokens(context_tokens)
    family_logp: dict[str, float] = {}

    for tok, cap in capstones.items():
        p = float(probs.get(tok, 0.0))
        family_logp[cap.family] = max(family_logp.get(cap.family, 0.0), p)

    build, _ = allocator.allocate(capstones, priority=family_logp)
    if build is not None and _is_full_build(build):
        return build, capstones, family_logp, 0

    added = 0
    for tok, p in sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])):
        proposed = dict(capstones)
        changed = _try_add_token(proposed, tok)
        if not changed:
            continue

        cap = AbilityToken.from_vocab_entry(tok)
        proposed_family_logp = dict(family_logp)
        proposed_family_logp[cap.family] = max(
            proposed_family_logp.get(cap.family, 0.0), float(p)
        )

        proposed_build, _ = allocator.allocate(
            proposed,
            priority=proposed_family_logp,
        )
        if proposed_build is None:
            continue

        capstones = proposed
        family_logp = proposed_family_logp
        build = proposed_build
        added += 1

        if _is_full_build(build):
            break

    return build, capstones, family_logp, added


def oneshot_capstones_topk(
    *,
    probs: Mapping[str, float],
    context_tokens: list[str],
    top_k: int,
) -> tuple[dict[str, AbilityToken], dict[str, float]]:
    capstones = _capstones_from_tokens(context_tokens)
    family_logp: dict[str, float] = {}

    for tok, cap in capstones.items():
        p = float(probs.get(tok, 0.0))
        family_logp[cap.family] = max(family_logp.get(cap.family, 0.0), p)

    picked = 0
    for tok, p in sorted(probs.items(), key=lambda kv: kv[1], reverse=True):
        if picked >= int(top_k):
            break
        changed = _try_add_token(capstones, tok)
        if not changed:
            continue
        cap = AbilityToken.from_vocab_entry(tok)
        family_logp[cap.family] = max(
            family_logp.get(cap.family, 0.0), float(p)
        )
        picked += 1

    return capstones, family_logp


def _row_from_pred_builds(
    case: dict[str, Any],
    truth: Build,
    pred_builds: list[Build],
) -> dict[str, Any]:
    metrics = evaluate_top_k(truth, pred_builds)

    top1_ap = pred_builds[0].achieved_ap if pred_builds else None
    best_ap = None
    best_acc = float("-inf")
    for pred in pred_builds:
        acc = compare_builds(truth, pred)["accuracy"]
        if acc > best_acc:
            best_acc = acc
            best_ap = pred.achieved_ap

    return {
        "case_id": int(case["case_id"]),
        "ability_mask": int(case["ability_mask"]),
        "n_predictions": int(len(pred_builds)),
        "predicted_top1_achieved_ap": (
            None if top1_ap is None else dict(top1_ap)
        ),
        "predicted_best_achieved_ap": (
            None if best_ap is None else dict(best_ap)
        ),
        **{k: float(v) for k, v in metrics.items()},
    }


def run_decoder_variant(
    cases: list[dict[str, Any]],
    truth_by_case_id: Mapping[int, Build],
    *,
    allocator: Allocator,
    predict_batch_fn,
    beam_size: int,
    max_steps: int | None,
    top_k: int,
    greedy_threshold: float,
    case_batch_size: int,
    context_field: str = "context_tokens",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch_start in range(0, len(cases), int(case_batch_size)):
        batch_cases = cases[batch_start : batch_start + int(case_batch_size)]
        weapon_ids = [str(c["weapon_token"]) for c in batch_cases]
        contexts = [
            list(c.get(context_field) or c["context_tokens"])
            for c in batch_cases
        ]
        batch_pred = reconstruct_builds_batched(
            predict_batch_fn=predict_batch_fn,
            weapon_ids=weapon_ids,
            initial_contexts=contexts,
            allocator=allocator,
            beam_size=int(beam_size),
            max_steps=max_steps,
            top_k=int(top_k),
            greedy_threshold=float(greedy_threshold),
        )
        for case, pred_builds_opt in zip(batch_cases, batch_pred):
            case_id = int(case["case_id"])
            truth = truth_by_case_id[case_id]
            pred_builds = pred_builds_opt or []
            rows.append(_row_from_pred_builds(case, truth, pred_builds))
    return rows


def run_oneshot_variant(
    cases: list[dict[str, Any]],
    truth_by_case_id: Mapping[int, Build],
    *,
    allocator: Allocator,
    probs_by_case_id: Mapping[int, Mapping[str, float]],
    variant: str,
    greedy_threshold: float,
    top_k: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        truth = truth_by_case_id[case_id]
        probs = probs_by_case_id[case_id]

        extras: dict[str, Any] = {}
        if variant == "rank_until_full":
            build, capstones, family_logp, added = oneshot_rank_until_full(
                probs=probs,
                context_tokens=list(case["context_tokens"]),
                allocator=allocator,
            )
            pred_builds = [] if build is None else [build]
            extras.update(
                {
                    "oneshot_added_tokens": int(added),
                    "oneshot_is_full": (
                        0.0
                        if build is None
                        else float(1.0 if _is_full_build(build) else 0.0)
                    ),
                    "oneshot_total_ap": (
                        None if build is None else int(build.total_ap)
                    ),
                    "oneshot_capstones": sorted(list(capstones.keys())),
                }
            )
        elif variant == "threshold":
            capstones, family_logp = oneshot_capstones_threshold(
                probs=probs,
                context_tokens=list(case["context_tokens"]),
                threshold=float(greedy_threshold),
            )
            build, _ = allocator.allocate(capstones, priority=family_logp)
            pred_builds = [] if build is None else [build]
        elif variant == "topk_fixed":
            capstones, family_logp = oneshot_capstones_topk(
                probs=probs,
                context_tokens=list(case["context_tokens"]),
                top_k=int(top_k),
            )
            build, _ = allocator.allocate(capstones, priority=family_logp)
            pred_builds = [] if build is None else [build]
        else:
            raise ValueError(f"Unknown oneshot variant: {variant}")

        row = _row_from_pred_builds(case, truth, pred_builds)
        row.update(extras)
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("test_data/abilities-with-weapons.csv"),
    )
    parser.add_argument(
        "--vocab",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/vocab.json"),
    )
    parser.add_argument(
        "--weapon-vocab",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/weapon_vocab.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saved_models/dataset_v0_2_super/clean_slate.pth"),
    )
    parser.add_argument(
        "--train-tiers",
        type=int,
        nargs="+",
        default=[2, 3],
    )
    parser.add_argument("--eval-tiers", type=int, nargs="+", default=[1])
    parser.add_argument("--masks", type=int, nargs="+", default=[6])
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--nested-masks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate nested masks per build (shared permutation across masks).",
    )
    parser.add_argument(
        "--include-strict-lock",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include decoder variants that start from canonicalized context tokens.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
    )
    parser.add_argument(
        "--case-batch-size",
        type=int,
        default=64,
        help="Batch size for decoder (beam search) variants.",
    )
    parser.add_argument("--greedy-threshold", type=float, default=0.5)
    parser.add_argument(
        "--decoder-max-steps",
        type=int,
        default=8,
        help="Beam search max steps for decoder variants (-1 for uncapped).",
    )
    parser.add_argument(
        "--oneshot-topk",
        type=int,
        default=8,
        help="How many one-shot tokens to add for the fixed top-k baseline.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON path (defaults to tmp_results/...).",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    weapon_vocab = _load_json(args.weapon_vocab)
    vocab = _load_json(args.vocab)

    builds = load_sendou_builds(args.csv, weapon_vocab=weapon_vocab)
    train_builds, eval_builds = split_builds_by_tier(
        builds,
        train_tiers=list(args.train_tiers),
        eval_tiers=list(args.eval_tiers),
    )
    if not train_builds:
        raise ValueError(f"No training builds for tiers={args.train_tiers}")
    if not eval_builds:
        raise ValueError(f"No eval builds for tiers={args.eval_tiers}")

    cases, truth_by_case_id = build_eval_cases(
        eval_builds,
        masks=list(args.masks),
        limit_per_mask=(int(args.limit) if int(args.limit) > 0 else None),
        seed=int(args.seed),
        nested_masks=bool(args.nested_masks),
    )

    print(
        f"Loaded {len(builds)} builds "
        f"(train={len(train_builds)}, eval={len(eval_builds)})"
    )
    print(f"Evaluating {len(cases)} cases for masks={args.masks}")

    model = _load_model(
        args.checkpoint,
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        device=device,
    )
    predict_batch_multi_factory = build_predict_abilities_batch_multiweapon(
        vocab,
        weapon_vocab,
        pad_token="<PAD>",
        hook=None,
        device=device,
        output_type="dict",
    )

    def predict_batch_multi_fn(
        token_batches: list[list[str]],
        weapon_ids: list[str],
    ) -> list[dict[str, float]]:
        return predict_batch_multi_factory(model, token_batches, weapon_ids)

    allocator = Allocator()

    out_path = (
        args.out
        if args.out is not None
        else Path("tmp_results")
        / (
            "sendou_decoder_ablations_"
            f"mask{'-'.join(str(m) for m in args.masks)}_"
            f"seed{int(args.seed)}.json"
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "csv": str(args.csv),
        "vocab": str(args.vocab),
        "weapon_vocab": str(args.weapon_vocab),
        "checkpoint": str(args.checkpoint),
        "train_tiers": list(args.train_tiers),
        "eval_tiers": list(args.eval_tiers),
        "masks": list(args.masks),
        "limit": int(args.limit),
        "seed": int(args.seed),
        "nested_masks": bool(args.nested_masks),
        "include_strict_lock": bool(args.include_strict_lock),
        "device": str(args.device),
        "greedy_threshold": float(args.greedy_threshold),
        "case_batch_size": int(args.case_batch_size),
        "decoder_max_steps": (
            None
            if int(args.decoder_max_steps) < 0
            else int(args.decoder_max_steps)
        ),
        "oneshot_topk_fixed": int(args.oneshot_topk),
    }

    decoder_max_steps = (
        None if int(args.decoder_max_steps) < 0 else int(args.decoder_max_steps)
    )

    results: dict[str, Any] = {}

    if args.include_strict_lock:
        print("\nCanonicalizing contexts for strict-lock variants...")
        for case in cases:
            case["context_tokens_lock"] = _canonicalize_context_tokens(
                list(case["context_tokens"]),
                allocator=allocator,
            )

    # 0) Non-model baselines (fast).
    baseline_variants: list[tuple[str, dict[str, Any]]] = [
        ("conditional", {"variant": "conditional"}),
        ("random", {"variant": "random"}),
    ]
    for name, cfg in baseline_variants:
        print(f"\nRunning {name} ({cfg})...")
        t0 = time.time()
        if cfg["variant"] == "conditional":
            rows, dt = run_conditional_baseline(
                cases,
                truth_by_case_id,
                train_builds=train_builds,
            )
        elif cfg["variant"] == "random":
            rows, dt = run_random_completion(
                cases,
                truth_by_case_id,
                seed=int(args.seed),
            )
        else:
            raise ValueError(f"Unknown baseline variant: {cfg['variant']}")

        results[name] = {
            "summary": _summarize_rows(rows),
            "rows": rows,
            "wall_time_s": float(dt),
        }
        _ = t0
        print(f"{name}: wall_time_s={dt:.2f}")

    # 1) Decoder variants.
    decoder_variants: list[tuple[str, dict[str, Any]]] = [
        (
            "decoder_greedy_only",
            {
                "beam_size": 1,
                "max_steps": 0,
                "top_k": 1,
                "context": "context_tokens",
            },
        ),
        (
            "decoder_beam1",
            {
                "beam_size": 1,
                "max_steps": decoder_max_steps,
                "top_k": 1,
                "context": "context_tokens",
            },
        ),
        (
            "decoder_beam3",
            {
                "beam_size": 3,
                "max_steps": decoder_max_steps,
                "top_k": 1,
                "context": "context_tokens",
            },
        ),
    ]
    if args.include_strict_lock:
        decoder_variants.extend(
            [
                (
                    "decoder_beam1_strict_lock",
                    {
                        "beam_size": 1,
                        "max_steps": decoder_max_steps,
                        "top_k": 1,
                        "context": "context_tokens_lock",
                    },
                ),
                (
                    "decoder_beam3_strict_lock",
                    {
                        "beam_size": 3,
                        "max_steps": decoder_max_steps,
                        "top_k": 1,
                        "context": "context_tokens_lock",
                    },
                ),
            ]
        )
    for name, cfg in decoder_variants:
        print(f"\nRunning {name} ({cfg})...")
        t0 = time.time()
        rows = run_decoder_variant(
            cases,
            truth_by_case_id,
            allocator=allocator,
            predict_batch_fn=predict_batch_multi_fn,
            beam_size=int(cfg["beam_size"]),
            max_steps=cfg["max_steps"],
            top_k=int(cfg["top_k"]),
            greedy_threshold=float(args.greedy_threshold),
            case_batch_size=int(args.case_batch_size),
            context_field=str(cfg.get("context", "context_tokens")),
        )
        dt = time.time() - t0
        results[name] = {
            "summary": _summarize_rows(rows),
            "rows": rows,
            "wall_time_s": float(dt),
        }
        print(f"{name}: wall_time_s={dt:.2f}")

    # 2) One-shot baselines (single model pass, then allocate).
    print("\nComputing one-shot probabilities...")
    probs_by_case_id: dict[int, Mapping[str, float]] = {}
    for batch_start in range(0, len(cases), int(args.case_batch_size)):
        batch_cases = cases[
            batch_start : batch_start + int(args.case_batch_size)
        ]
        contexts_batch = [list(c["context_tokens"]) for c in batch_cases]
        weapon_ids_batch = [str(c["weapon_token"]) for c in batch_cases]
        probs_batch = predict_batch_multi_fn(contexts_batch, weapon_ids_batch)
        probs_by_case_id.update(
            {
                int(c["case_id"]): probs
                for c, probs in zip(batch_cases, probs_batch)
            }
        )

    oneshot_variants: list[tuple[str, dict[str, Any]]] = [
        ("oneshot_rank_until_full", {"variant": "rank_until_full"}),
        ("oneshot_threshold", {"variant": "threshold"}),
        (
            "oneshot_topk_fixed",
            {"variant": "topk_fixed", "top_k": int(args.oneshot_topk)},
        ),
    ]
    for name, cfg in oneshot_variants:
        print(f"\nRunning {name} ({cfg})...")
        t0 = time.time()
        rows = run_oneshot_variant(
            cases,
            truth_by_case_id,
            allocator=allocator,
            probs_by_case_id=probs_by_case_id,
            variant=str(cfg["variant"]),
            greedy_threshold=float(args.greedy_threshold),
            top_k=int(cfg.get("top_k", int(args.oneshot_topk))),
        )
        dt = time.time() - t0
        results[name] = {
            "summary": _summarize_rows(rows),
            "rows": rows,
            "wall_time_s": float(dt),
        }
        print(f"{name}: wall_time_s={dt:.2f}")

    payload = {"meta": meta, "cases": cases, "results": results}
    out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
