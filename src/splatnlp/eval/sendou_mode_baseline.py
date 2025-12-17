from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import orjson
import pandas as pd

from splatnlp.eval.sendou_baseline import (
    ap_to_slot_items,
    evaluate_top_k,
    load_sendou_builds,
    mask_abilities,
    slots_to_build,
    split_builds_by_tier,
)
from splatnlp.utils.reconstruct.classes import Build


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _abilities_key(abilities_ap: dict[str, int]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((str(k), int(v)) for k, v in abilities_ap.items()))


def build_mode_index(
    builds: list,
) -> tuple[dict[str, Build], dict[str, list[tuple[int, Counter[str], Build]]]]:
    counts_by_weapon: dict[str, Counter[tuple[tuple[str, int], ...]]] = (
        defaultdict(Counter)
    )
    build_by_key: dict[tuple[tuple[str, int], ...], Build] = {}
    slot_counts_by_key: dict[tuple[tuple[str, int], ...], Counter[str]] = {}

    for row in builds:
        key = _abilities_key(row.abilities_ap)
        counts_by_weapon[row.weapon_token][key] += 1
        if key not in build_by_key:
            slot_items = ap_to_slot_items(row.abilities_ap)
            build_by_key[key] = slots_to_build(slot_items)
            slot_counts_by_key[key] = Counter(slot_items)

    mode_builds: dict[str, Build] = {}
    candidates: dict[str, list[tuple[int, Counter[str], Build]]] = {}

    for weapon_token, counts in counts_by_weapon.items():
        (mode_key, mode_count) = max(counts.items(), key=lambda x: x[1])
        _ = mode_count
        mode_builds[weapon_token] = build_by_key[mode_key]

        weapon_candidates: list[tuple[int, Counter[str], Build]] = []
        for key, count in counts.items():
            weapon_candidates.append(
                (int(count), slot_counts_by_key[key], build_by_key[key])
            )
        weapon_candidates.sort(key=lambda x: x[0], reverse=True)
        candidates[weapon_token] = weapon_candidates

    return mode_builds, candidates


def choose_mode_completion(
    weapon_token: str,
    context_slots: Counter[str],
    *,
    mode_builds: dict[str, Build],
    candidates: dict[str, list[tuple[int, Counter[str], Build]]],
    conditional: bool,
) -> Build:
    if not conditional:
        return mode_builds[weapon_token]

    for _, slot_counts, build in candidates.get(weapon_token, []):
        if not (context_slots - slot_counts):
            return build

    return mode_builds[weapon_token]


def run_eval(
    builds: list,
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
    conditional: bool,
    mode_builds: dict[str, Build],
    candidates: dict[str, list[tuple[int, Counter[str], Build]]],
) -> pd.DataFrame:
    rng = random.Random(seed)

    truth_by_key: dict[tuple[int, int, int], Build] = {}
    for row in builds:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_by_key:
            continue
        truth_slots = ap_to_slot_items(row.abilities_ap)
        truth_by_key[key] = slots_to_build(truth_slots)

    eval_rows: list[dict[str, float]] = []
    for mask in masks:
        candidates_rows = list(builds)
        if limit_per_mask is not None:
            candidates_rows = rng.sample(
                candidates_rows, min(limit_per_mask, len(candidates_rows))
            )

        for row in candidates_rows:
            masked_abilities = mask_abilities(row.abilities_ap, mask, rng)
            context_slots = Counter(ap_to_slot_items(masked_abilities))
            pred_build = choose_mode_completion(
                row.weapon_token,
                context_slots,
                mode_builds=mode_builds,
                candidates=candidates,
                conditional=conditional,
            )

            truth = truth_by_key[(row.build_id, row.plus_tier, row.weapon_id)]
            metrics = evaluate_top_k(truth, [pred_build])
            metrics["ability_mask"] = float(mask)
            eval_rows.append(metrics)

    eval_df = pd.DataFrame(eval_rows)
    return (
        eval_df.groupby("ability_mask")
        .agg(
            exact_hit_rate=("exact_hit", "mean"),
            avg_correct_out=("num_correct_out", "mean"),
            avg_correct_best=("num_correct_best", "mean"),
            avg_best_accuracy=("best_accuracy", "mean"),
        )
        .reset_index()
    )


def run_eval_detailed(
    builds: list,
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
    conditional: bool,
    mode_builds: dict[str, Build],
    candidates: dict[str, list[tuple[int, Counter[str], Build]]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rng = random.Random(seed)

    truth_by_key: dict[tuple[int, int, int], Build] = {}
    for row in builds:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_by_key:
            continue
        truth_slots = ap_to_slot_items(row.abilities_ap)
        truth_by_key[key] = slots_to_build(truth_slots)

    eval_rows: list[dict[str, object]] = []
    for mask in masks:
        candidates_rows = list(builds)
        if limit_per_mask is not None:
            candidates_rows = rng.sample(
                candidates_rows, min(limit_per_mask, len(candidates_rows))
            )

        for row in candidates_rows:
            masked_abilities = mask_abilities(row.abilities_ap, mask, rng)
            context_slots = Counter(ap_to_slot_items(masked_abilities))
            pred_build = choose_mode_completion(
                row.weapon_token,
                context_slots,
                mode_builds=mode_builds,
                candidates=candidates,
                conditional=conditional,
            )

            truth = truth_by_key[(row.build_id, row.plus_tier, row.weapon_id)]
            metrics = evaluate_top_k(truth, [pred_build])

            rec: dict[str, object] = {
                "build_id": row.build_id,
                "plus_tier": row.plus_tier,
                "weapon_id": row.weapon_id,
                "weapon_token": row.weapon_token,
                "ability_mask": int(mask),
                "truth_abilities_ap": dict(row.abilities_ap),
                "masked_abilities_ap": dict(masked_abilities),
                "truth_achieved_ap": dict(truth.achieved_ap),
                "predicted_achieved_ap": dict(pred_build.achieved_ap),
                **{k: float(v) for k, v in metrics.items()},
            }
            eval_rows.append(rec)

    eval_df = pd.DataFrame(eval_rows)
    summary = (
        eval_df.groupby("ability_mask")
        .agg(
            exact_hit_rate=("exact_hit", "mean"),
            avg_correct_out=("num_correct_out", "mean"),
            avg_correct_best=("num_correct_best", "mean"),
            avg_best_accuracy=("best_accuracy", "mean"),
        )
        .reset_index()
    )
    return summary, eval_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("test_data/abilities-with-weapons.csv"),
    )
    parser.add_argument(
        "--weapon-vocab",
        type=Path,
        default=Path("saved_models/dataset_v0_2_full/weapon_vocab.json"),
    )
    parser.add_argument(
        "--masks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Number of slots to drop (e.g. 1 2 3 6 9).",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditional",
        action="store_true",
        help="Choose the most common completion consistent with the observed "
        "slots (multiset containment).",
    )
    parser.add_argument(
        "--train-tiers",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Tiers to build the mode baseline from (defaults to 2 3).",
    )
    parser.add_argument(
        "--eval-tiers",
        type=int,
        nargs="+",
        default=[1],
        help="Tiers to evaluate on (defaults to 1).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write detailed rows + summary to this JSON file.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable writing the output JSON file.",
    )
    args = parser.parse_args()

    weapon_vocab = _load_json(args.weapon_vocab)

    builds = load_sendou_builds(args.csv, weapon_vocab=weapon_vocab)
    train_builds, eval_builds = split_builds_by_tier(
        builds, train_tiers=args.train_tiers, eval_tiers=args.eval_tiers
    )
    print(
        f"Loaded {len(builds)} builds from {args.csv} "
        f"(train={len(train_builds)}, eval={len(eval_builds)})"
    )
    if not train_builds:
        raise ValueError(f"No training builds for tiers={args.train_tiers}")
    if not eval_builds:
        raise ValueError(f"No eval builds for tiers={args.eval_tiers}")

    mode_builds, candidates = build_mode_index(train_builds)

    t0 = time.time()
    summary, rows = run_eval_detailed(
        eval_builds,
        masks=args.masks,
        limit_per_mask=(args.limit if args.limit > 0 else None),
        seed=args.seed,
        conditional=args.conditional,
        mode_builds=mode_builds,
        candidates=candidates,
    )
    dt = time.time() - t0
    print(summary.to_string(index=False))
    print(f"wall_time_s={dt:.2f}")

    if not args.no_cache:
        out_path = args.out
        if out_path is None:
            tiers_train = "-".join(str(t) for t in args.train_tiers)
            tiers_eval = "-".join(str(t) for t in args.eval_tiers)
            masks = "-".join(str(m) for m in args.masks)
            out_path = Path("tmp_results") / (
                "sendou_mode_baseline_"
                f"cond{int(bool(args.conditional))}_"
                f"train{tiers_train}_eval{tiers_eval}_"
                f"masks{masks}_limit{args.limit}_seed{args.seed}.json"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "method": "mode_baseline",
                "conditional": bool(args.conditional),
                "csv": str(args.csv),
                "weapon_vocab": str(args.weapon_vocab),
                "train_tiers": list(args.train_tiers),
                "eval_tiers": list(args.eval_tiers),
                "masks": list(args.masks),
                "limit": int(args.limit),
                "seed": int(args.seed),
                "n_builds_total": int(len(builds)),
                "n_builds_train": int(len(train_builds)),
                "n_builds_eval": int(len(eval_builds)),
                "wall_time_s": float(dt),
            },
            "summary": summary.to_dict(orient="records"),
            "rows": rows,
        }
        out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
