from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import orjson
import pandas as pd
import torch

from splatnlp.eval.sendou_baseline import (
    ap_to_slot_items,
    compare_builds,
    compute_token_priors,
    evaluate_top_k,
    load_sendou_builds,
    make_weapon_prior_predict_fn,
    mask_abilities,
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
    STANDARD_ABILITIES,
)
from splatnlp.utils.infer import build_predict_abilities
from splatnlp.utils.reconstruct import Allocator, reconstruct_build
from splatnlp.utils.reconstruct.classes import Build


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _load_model(
    checkpoint: Path, vocab: dict[str, int], weapon_vocab: dict[str, int]
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
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return model


def _default_out_path(
    *,
    train_tiers: list[int],
    eval_tiers: list[int],
    masks: list[int],
    limit: int,
    seed: int,
    beam_size: int,
    max_steps: int,
    top_k: int,
) -> Path:
    tiers_train = "-".join(str(t) for t in train_tiers)
    tiers_eval = "-".join(str(t) for t in eval_tiers)
    masks_str = "-".join(str(m) for m in masks)
    return Path("tmp_results") / (
        "sendou_compare_"
        f"train{tiers_train}_eval{tiers_eval}_"
        f"masks{masks_str}_limit{limit}_seed{seed}_"
        f"beam{beam_size}_steps{max_steps}_top{top_k}.json"
    )


def build_eval_cases(
    builds_eval: list,
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[int, Build]]:
    rng = torch.Generator().manual_seed(seed)

    truth_builds: dict[tuple[int, int, int], Build] = {}
    for row in builds_eval:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_builds:
            continue
        truth_builds[key] = slots_to_build(ap_to_slot_items(row.abilities_ap))

    cases: list[dict[str, Any]] = []
    truth_by_case_id: dict[int, Build] = {}
    case_id = 0

    # Use Python's random via torch-generated entropy for determinism across
    # environments (torch generator is stable under our seed).
    import random

    rng_py = random.Random(
        int(torch.randint(0, 2**31 - 1, (1,), generator=rng))
    )

    for mask in masks:
        candidates = list(builds_eval)
        if limit_per_mask is not None:
            candidates = rng_py.sample(
                candidates, min(limit_per_mask, len(candidates))
            )

        for row in candidates:
            masked_abilities = mask_abilities(row.abilities_ap, mask, rng_py)
            context_tokens = tokenize_build(masked_abilities)
            truth = truth_builds[(row.build_id, row.plus_tier, row.weapon_id)]

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


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    df = pd.DataFrame(rows)
    summary = (
        df.groupby("ability_mask")
        .agg(
            exact_hit_rate=("exact_hit", "mean"),
            avg_correct_out=("num_correct_out", "mean"),
            avg_correct_best=("num_correct_best", "mean"),
            avg_best_accuracy=("best_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary.to_dict(orient="records")


def eval_conditional_mode(
    cases: list[dict[str, Any]],
    truth_by_case_id: dict[int, Build],
    *,
    train_builds: list,
    conditional: bool,
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
            conditional=conditional,
        )
        truth = truth_by_case_id[case_id]
        metrics = evaluate_top_k(truth, [pred_build])
        rows.append(
            {
                "case_id": case_id,
                "ability_mask": int(case["ability_mask"]),
                "n_predictions": 1,
                "predicted_top1_achieved_ap": dict(pred_build.achieved_ap),
                "predicted_best_achieved_ap": dict(pred_build.achieved_ap),
                **{k: float(v) for k, v in metrics.items()},
            }
        )

    dt = time.time() - t0
    return rows, dt


def eval_model(
    cases: list[dict[str, Any]],
    truth_by_case_id: dict[int, Build],
    *,
    model: SetCompletionModel,
    vocab: dict[str, int],
    weapon_vocab: dict[str, int],
    beam_size: int,
    max_steps: int,
    top_k: int,
    include_predictions: bool,
) -> tuple[list[dict[str, Any]], float]:
    t0 = time.time()
    allocator = Allocator()

    predict_factory = build_predict_abilities(
        vocab,
        weapon_vocab,
        pad_token="<PAD>",
        hook=None,
        device=torch.device("cpu"),
        output_type="dict",
    )

    def predict_fn(tokens: list[str], weapon_id: str) -> dict[str, float]:
        return predict_factory(model, tokens, weapon_id)

    rows: list[dict[str, Any]] = []
    for case in cases:
        case_id = int(case["case_id"])
        weapon_token = str(case["weapon_token"])
        context_tokens = list(case["context_tokens"])

        pred_builds = (
            reconstruct_build(
                predict_fn=predict_fn,
                weapon_id=weapon_token,
                initial_context=context_tokens,
                allocator=allocator,
                beam_size=beam_size,
                max_steps=max_steps,
                top_k=top_k,
            )
            or []
        )
        truth = truth_by_case_id[case_id]
        metrics = evaluate_top_k(truth, pred_builds)

        top1_ap = pred_builds[0].achieved_ap if pred_builds else None
        best_ap = None
        best_acc = float("-inf")
        for pred in pred_builds:
            acc = compare_builds(truth, pred)["accuracy"]
            if acc > best_acc:
                best_acc = acc
                best_ap = pred.achieved_ap

        row: dict[str, Any] = {
            "case_id": case_id,
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
        if include_predictions:
            row["predicted_builds"] = [b.to_dict() for b in pred_builds]
        rows.append(row)

    dt = time.time() - t0
    return rows, dt


def eval_random_completion(
    cases: list[dict[str, Any]],
    truth_by_case_id: dict[int, Build],
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

        rng = random.Random((int(seed) * 1_000_003 + case_id) & 0xFFFFFFFF)
        context_items = ap_to_slot_items(masked_abilities_ap)

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
        metrics = evaluate_top_k(truth, [pred_build])
        rows.append(
            {
                "case_id": case_id,
                "ability_mask": ability_mask,
                "n_predictions": 1,
                "predicted_top1_achieved_ap": dict(pred_build.achieved_ap),
                "predicted_best_achieved_ap": dict(pred_build.achieved_ap),
                **{k: float(v) for k, v in metrics.items()},
            }
        )

    dt = time.time() - t0
    return rows, dt


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
        "--full-checkpoint",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/model.pth"),
    )
    parser.add_argument(
        "--ultra-checkpoint",
        type=Path,
        default=Path("saved_models/dataset_v0_2_super/clean_slate.pth"),
    )
    parser.add_argument(
        "--train-tiers",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Tiers to build conditional baseline from.",
    )
    parser.add_argument(
        "--eval-tiers",
        type=int,
        nargs="+",
        default=[1],
        help="Tiers to evaluate on.",
    )
    parser.add_argument(
        "--masks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--methods",
        type=str,
        nargs="+",
        default=["full", "ultra", "conditional"],
        choices=["full", "ultra", "conditional", "weapon_prior", "random"],
    )
    parser.add_argument(
        "--include-predictions",
        action="store_true",
        help="Include predicted build dicts for model methods (large).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output JSON file (defaults to tmp_results/...).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file, if present.",
    )
    parser.add_argument(
        "--flush-every",
        type=int,
        default=25,
        help="Write progress every N cases per method.",
    )
    args = parser.parse_args()

    out_path = (
        args.out
        if args.out is not None
        else _default_out_path(
            train_tiers=list(args.train_tiers),
            eval_tiers=list(args.eval_tiers),
            masks=list(args.masks),
            limit=int(args.limit),
            seed=int(args.seed),
            beam_size=int(args.beam_size),
            max_steps=int(args.max_steps),
            top_k=int(args.top_k),
        )
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    meta = {
        "csv": str(args.csv),
        "vocab": str(args.vocab),
        "weapon_vocab": str(args.weapon_vocab),
        "full_checkpoint": str(args.full_checkpoint),
        "ultra_checkpoint": str(args.ultra_checkpoint),
        "train_tiers": list(args.train_tiers),
        "eval_tiers": list(args.eval_tiers),
        "masks": list(args.masks),
        "limit": int(args.limit),
        "seed": int(args.seed),
        "beam_size": int(args.beam_size),
        "max_steps": int(args.max_steps),
        "top_k": int(args.top_k),
        "methods": list(args.methods),
        "include_predictions": bool(args.include_predictions),
    }

    existing: dict[str, Any] | None = None
    if args.resume and out_path.exists():
        existing = orjson.loads(out_path.read_bytes())
        existing_meta = existing.get("meta")
        if existing_meta != meta:
            if not isinstance(existing_meta, dict):
                raise ValueError(
                    f"Cannot resume: meta mismatch in {out_path}. "
                    "Provide a new --out or remove --resume."
                )
            existing_norm = dict(existing_meta)
            existing_norm.pop("methods", None)
            meta_norm = dict(meta)
            meta_norm.pop("methods", None)
            if existing_norm != meta_norm:
                raise ValueError(
                    f"Cannot resume: meta mismatch in {out_path}. "
                    "Provide a new --out or remove --resume."
                )
        if isinstance(existing_meta, dict):
            existing_methods = existing_meta.get("methods")
            if isinstance(existing_methods, list):
                merged: list[str] = []
                seen: set[str] = set()
                for method in [*existing_methods, *list(args.methods)]:
                    if not isinstance(method, str):
                        continue
                    if method in seen:
                        continue
                    seen.add(method)
                    merged.append(method)
                if merged:
                    meta["methods"] = merged

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

    print(
        f"Loaded {len(builds)} builds "
        f"(train={len(train_builds)}, eval={len(eval_builds)})"
    )

    cases: list[dict[str, Any]]
    truth_by_case_id: dict[int, Build]
    if existing is not None and "cases" in existing and "results" in existing:
        cases = list(existing["cases"])
        _, truth_by_case_id = build_eval_cases(
            eval_builds,
            masks=[1],
            limit_per_mask=1,
            seed=args.seed,
        )
        truth_by_case_id = {}
        for case in cases:
            key = (
                int(case["build_id"]),
                int(case["plus_tier"]),
                int(case["weapon_id"]),
            )
            truth_by_case_id[int(case["case_id"])] = slots_to_build(
                ap_to_slot_items(dict(case["truth_abilities_ap"]))
            )
        results = dict(existing["results"])
    else:
        cases, truth_by_case_id = build_eval_cases(
            eval_builds,
            masks=list(args.masks),
            limit_per_mask=(int(args.limit) if int(args.limit) > 0 else None),
            seed=int(args.seed),
        )
        results: dict[str, Any] = {}

    def flush() -> None:
        payload = {
            "meta": meta,
            "cases": cases,
            "results": results,
        }
        out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"Wrote {out_path}")

    if existing is None:
        flush()

    for method in args.methods:
        if method in results and len(results[method].get("rows", [])) == len(
            cases
        ):
            print(f"Skipping {method}: already complete in {out_path}")
            continue

        print(f"Running {method} on {len(cases)} cases...")
        method_rows: list[dict[str, Any]] = []

        if (
            existing is not None
            and method in results
            and results[method].get("rows")
        ):
            method_rows = list(results[method]["rows"])

        done_ids = {int(r["case_id"]) for r in method_rows}
        remaining_cases = [
            c for c in cases if int(c["case_id"]) not in done_ids
        ]

        if method == "conditional":
            t0 = time.time()
            rows, dt = eval_conditional_mode(
                remaining_cases,
                truth_by_case_id,
                train_builds=train_builds,
                conditional=True,
            )
            method_rows.extend(rows)
            dt_total = dt
            _ = t0
        elif method == "random":
            rows, dt_total = eval_random_completion(
                remaining_cases,
                truth_by_case_id,
                seed=int(args.seed),
            )
            method_rows.extend(rows)
        else:
            predict_fn = None

            if method == "weapon_prior":
                priors_by_weapon, global_prior = compute_token_priors(
                    train_builds,
                    vocab=vocab,
                    mix_global=0.1,
                )
                predict_fn = make_weapon_prior_predict_fn(
                    priors_by_weapon=priors_by_weapon,
                    global_prior=global_prior,
                    vocab=vocab,
                )
            else:
                ckpt = (
                    args.full_checkpoint
                    if method == "full"
                    else args.ultra_checkpoint
                )
                model = _load_model(
                    ckpt, vocab=vocab, weapon_vocab=weapon_vocab
                )

                predict_factory = build_predict_abilities(
                    vocab,
                    weapon_vocab,
                    pad_token="<PAD>",
                    hook=None,
                    device=torch.device("cpu"),
                    output_type="dict",
                )

                def predict_fn(
                    tokens: list[str], weapon_id: str
                ) -> dict[str, float]:
                    return predict_factory(model, tokens, weapon_id)

            t_start = time.time()
            allocator = Allocator()

            for i, case in enumerate(remaining_cases, 1):
                case_id = int(case["case_id"])
                weapon_token = str(case["weapon_token"])
                context_tokens = list(case["context_tokens"])

                pred_builds = (
                    reconstruct_build(
                        predict_fn=predict_fn,
                        weapon_id=weapon_token,
                        initial_context=context_tokens,
                        allocator=allocator,
                        beam_size=int(args.beam_size),
                        max_steps=int(args.max_steps),
                        top_k=int(args.top_k),
                    )
                    or []
                )
                truth = truth_by_case_id[case_id]
                metrics = evaluate_top_k(truth, pred_builds)

                top1_ap = pred_builds[0].achieved_ap if pred_builds else None
                best_ap = None
                best_acc = float("-inf")
                for pred in pred_builds:
                    acc = compare_builds(truth, pred)["accuracy"]
                    if acc > best_acc:
                        best_acc = acc
                        best_ap = pred.achieved_ap

                row: dict[str, Any] = {
                    "case_id": case_id,
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
                if args.include_predictions:
                    row["predicted_builds"] = [b.to_dict() for b in pred_builds]
                method_rows.append(row)

                if int(args.flush_every) > 0 and i % int(args.flush_every) == 0:
                    results[method] = {
                        "rows": method_rows,
                    }
                    flush()
                    print(f"{method}: {i}/{len(remaining_cases)} cases")

            dt_total = time.time() - t_start

        results[method] = {
            "summary": summarize_rows(method_rows),
            "rows": method_rows,
            "wall_time_s": float(dt_total),
        }
        flush()

    print("Done.")


if __name__ == "__main__":
    main()
