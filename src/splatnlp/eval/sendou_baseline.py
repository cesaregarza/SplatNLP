from __future__ import annotations

import argparse
import random
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import orjson
import pandas as pd

from splatnlp.serve.tokenize import tokenize_build
from splatnlp.utils.constants import (
    BUCKET_THRESHOLDS,
    CANONICAL_MAIN_ONLY_ABILITIES,
    MAIN_ONLY_ABILITIES,
    NULL,
    STANDARD_ABILITIES,
)
from splatnlp.utils.reconstruct import Allocator, reconstruct_build
from splatnlp.utils.reconstruct.classes import AbilityToken, Build

SENDOU_ABILITY_MAP: dict[str, str] = {
    # Main-only abilities
    "AD": "ability_doubler",
    "CB": "comeback",
    "DR": "drop_roller",
    "H": "haunt",
    "LDE": "last_ditch_effort",
    "NS": "ninja_squid",
    "OG": "opening_gambit",
    "OS": "object_shredder",
    "RP": "respawn_punisher",
    "SJ": "stealth_jump",
    "T": "tenacity",
    "TI": "thermal_ink",
    # Standard abilities
    "IA": "intensify_action",
    "IRU": "ink_recovery_up",
    "ISM": "ink_saver_main",
    "ISS": "ink_saver_sub",
    "RES": "ink_resistance_up",
    "QR": "quick_respawn",
    "QSJ": "quick_super_jump",
    "RSU": "run_speed_up",
    "SCU": "special_charge_up",
    "SPU": "special_power_up",
    "SRU": "sub_resistance_up",
    "SS": "special_saver",
    "SSU": "swim_speed_up",
    # Historical / alternate codes (seen in some exports)
    "BRU": "ink_saver_main",
    "ISC": "ink_saver_sub",
    "MQR": "quick_respawn",
    "QSR": "quick_super_jump",
    "BRU2": "ink_saver_main",
    "BRU3": "ink_saver_main",
}


@dataclass(frozen=True)
class SendouBuildRow:
    build_id: int
    plus_tier: int
    weapon_id: int
    weapon_token: str
    abilities_ap: dict[str, int]


def _load_json(path: Path) -> dict:
    return orjson.loads(path.read_bytes())


def _normalize_sendou_csv(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "buildid": "build_id",
        "slotIndex": "slot_index",
        "weaponSplId": "weapon_id",
        "tier": "plus_tier",
    }
    missing = set(rename) - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected Sendou CSV columns: {sorted(df.columns)}")
    df = df.rename(columns=rename)
    df["ability"] = df["ability"].map(SENDOU_ABILITY_MAP).fillna(df["ability"])

    allowed = set(MAIN_ONLY_ABILITIES) | set(STANDARD_ABILITIES)
    unknown = sorted(set(df["ability"].unique()) - allowed)
    if unknown:
        raise ValueError(
            "Sendou CSV contains unknown abilities after mapping: "
            + ", ".join(unknown)
        )

    df["slot_index"] = df["slot_index"].astype(int)
    df["weapon_id"] = df["weapon_id"].astype(int)
    df["build_id"] = df["build_id"].astype(int)
    df["plus_tier"] = df["plus_tier"].astype(int)

    # slot_index: 0=main, 1..3=subs (per gear piece)
    df["ap"] = df["slot_index"].eq(0).mul(7).add(3)
    return df


def _load_weapon_reference_ids(path: Path) -> dict[int, int]:
    if not path.exists():
        return {}

    data = orjson.loads(path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f"Unexpected weapon_info.json shape: {type(data)}")

    out: dict[int, int] = {}
    for key, value in data.items():
        try:
            weapon_id = int(key)
        except (TypeError, ValueError):
            continue
        if not isinstance(value, dict):
            continue
        ref = value.get("reference_id")
        if isinstance(ref, int):
            out[weapon_id] = ref
    return out


def load_sendou_builds(
    csv_path: Path,
    weapon_vocab: dict[str, int],
    *,
    weapon_info_path: Path | None = None,
    dedupe_after_reference: bool = True,
) -> list[SendouBuildRow]:
    df = pd.read_csv(csv_path)
    df = _normalize_sendou_csv(df)

    base_df = (
        df.groupby(["build_id", "plus_tier", "weapon_id", "ability"])["ap"]
        .sum()
        .reset_index()
        .groupby(["build_id", "plus_tier", "weapon_id"])[["ability", "ap"]]
        .apply(lambda x: dict(zip(x["ability"], x["ap"])))
        .reset_index()
        .rename(columns={0: "abilities_ap"})
    )

    weapon_info_path = weapon_info_path or Path("docs/weapon_info.json")
    reference_ids = _load_weapon_reference_ids(weapon_info_path)
    if reference_ids:
        mapped_weapon_ids = base_df["weapon_id"].map(
            lambda wid: reference_ids.get(int(wid), int(wid))
        )
        n_mapped = int((mapped_weapon_ids != base_df["weapon_id"]).sum())
        if n_mapped:
            print(
                f"Mapping {n_mapped} builds to reference weapon IDs "
                f"using {weapon_info_path}"
            )
        base_df["weapon_id"] = mapped_weapon_ids

    base_df["weapon_token"] = base_df["weapon_id"].apply(
        lambda wid: f"weapon_id_{int(wid)}"
    )
    keep = base_df["weapon_token"].isin(set(weapon_vocab))
    dropped = int((~keep).sum())
    if dropped:
        print(
            f"Dropping {dropped} builds with unknown weapon IDs (not in weapon_vocab)"
        )
    base_df = base_df[keep].reset_index(drop=True)

    rows: list[SendouBuildRow] = []
    seen: dict[tuple[int, int, int], dict[str, int]] = {}
    dropped_dupes = 0
    for rec in base_df.to_dict(orient="records"):
        key = (
            int(rec["build_id"]),
            int(rec["plus_tier"]),
            int(rec["weapon_id"]),
        )
        abilities_ap = dict(rec["abilities_ap"])
        if dedupe_after_reference and key in seen:
            if abilities_ap != seen[key]:
                raise ValueError(
                    "Conflicting abilities for deduped key "
                    f"{key}: {abilities_ap} vs {seen[key]}"
                )
            dropped_dupes += 1
            continue
        seen[key] = abilities_ap
        rows.append(
            SendouBuildRow(
                build_id=key[0],
                plus_tier=key[1],
                weapon_id=key[2],
                weapon_token=str(rec["weapon_token"]),
                abilities_ap=abilities_ap,
            )
        )
    if dedupe_after_reference and dropped_dupes:
        print(
            f"Collapsed {dropped_dupes} duplicate builds after weapon mapping"
        )
    return rows


def ap_to_slot_items(abilities_ap: dict[str, int]) -> list[str]:
    sub_ap_values = {(3 * x) % 10: x for x in range(0, 10)}
    slots: list[str] = []
    for family, ap in abilities_ap.items():
        num_subs = sub_ap_values[ap % 10]
        num_mains = (ap - num_subs * 3) // 10
        slots.extend([f"{family}_sub"] * num_subs)
        slots.extend([f"{family}_main"] * num_mains)
    return slots


def slot_items_to_ap(slot_items: list[str]) -> dict[str, int]:
    out: dict[str, int] = defaultdict(int)
    for item in slot_items:
        if item.endswith("_sub"):
            out[item[:-4]] += 3
        elif item.endswith("_main"):
            out[item[:-5]] += 10
        else:
            raise ValueError(f"Unknown slot item: {item}")
    return dict(out)


def mask_abilities(
    abilities_ap: dict[str, int], num_masks: int, rng: random.Random
) -> dict[str, int]:
    slot_items = ap_to_slot_items(abilities_ap)
    rng.shuffle(slot_items)
    kept = slot_items[num_masks:]
    return slot_items_to_ap(kept)


def abilities_to_capstone_tokens(abilities_ap: dict[str, int]) -> list[str]:
    tokens: list[str] = []
    for family, ap in abilities_ap.items():
        if family in MAIN_ONLY_ABILITIES:
            if ap > 0:
                tokens.append(family)
            continue
        thresholds = [t for t in BUCKET_THRESHOLDS if t <= ap]
        if thresholds:
            tokens.append(f"{family}_{max(thresholds)}")
    return tokens or [NULL]


def tokens_to_capstones(tokens: list[str]) -> dict[str, AbilityToken]:
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

        existing_tokens = [
            k for k, v in capstones.items() if v.family == cap.family
        ]
        replaced = False
        for old_tok in existing_tokens:
            if cap.min_ap > capstones[old_tok].min_ap:
                del capstones[old_tok]
                capstones[tok] = cap
                replaced = True
        if not existing_tokens or replaced:
            capstones[tok] = cap
    return capstones


def slots_to_build(slot_items: list[str]) -> Build:
    mains: dict[str, str | None] = {s: None for s in Build.GEAR_SLOTS}
    subs: dict[str, int] = defaultdict(int)

    main_families: list[str] = []
    for item in slot_items:
        if item.endswith("_sub"):
            subs[item[:-4]] += 1
        elif item.endswith("_main"):
            main_families.append(item[:-5])
        else:
            raise ValueError(f"Unknown slot item: {item}")

    # Place main-only abilities on their canonical slots first.
    remaining_standard_mains: list[str] = []
    for family in main_families:
        if family in CANONICAL_MAIN_ONLY_ABILITIES:
            slot = CANONICAL_MAIN_ONLY_ABILITIES[family]
            if mains[slot] is not None and mains[slot] != family:
                raise ValueError(f"Main-only slot conflict on {slot}: {family}")
            mains[slot] = family
        else:
            remaining_standard_mains.append(family)

    for family in remaining_standard_mains:
        for slot in Build.GEAR_SLOTS:
            if mains[slot] is None:
                mains[slot] = family
                break
        else:
            raise ValueError("Too many main slots while building truth Build")

    return Build(mains=mains, subs=dict(subs))


def compare_builds(true_build: Build, pred_build: Build) -> dict[str, float]:
    truth_ms = Counter(ap_to_slot_items(true_build.achieved_ap))
    pred_ms = Counter(ap_to_slot_items(pred_build.achieved_ap))
    overlap = truth_ms & pred_ms
    correct = sum(overlap.values())
    missing = sum((truth_ms - pred_ms).values())
    extra = sum((pred_ms - truth_ms).values())

    all_keys = set(true_build.achieved_ap) | set(pred_build.achieved_ap)
    ap_diff = sum(
        abs(true_build.achieved_ap.get(k, 0) - pred_build.achieved_ap.get(k, 0))
        for k in all_keys
    )
    return {
        "correct": float(correct),
        "missing": float(missing),
        "extra": float(extra),
        "accuracy": 1 - ap_diff / (57 * 2),
    }


def evaluate_top_k(
    true_build: Build, pred_builds: list[Build]
) -> dict[str, float]:
    out = {
        "exact_hit": 0.0,
        "closest_hit_rank": 0.0,
        "num_correct_best": 0.0,
        "num_correct_out": 0.0,
        "num_missing": 0.0,
        "num_extra": 0.0,
        "accuracy": 0.0,
        "best_accuracy": 0.0,
    }
    for i, pred_build in enumerate(pred_builds or []):
        metrics = compare_builds(true_build, pred_build)
        out["exact_hit"] += (
            1.0 if pred_build.achieved_ap == true_build.achieved_ap else 0.0
        )
        out["num_correct_best"] = max(
            out["num_correct_best"], metrics["correct"]
        )
        out["best_accuracy"] = max(out["best_accuracy"], metrics["accuracy"])
        out["accuracy"] = metrics["accuracy"]
        if i == 0:
            out["num_correct_out"] = metrics["correct"]
        if out["num_correct_best"] == metrics["correct"]:
            out["num_missing"] = metrics["missing"]
            out["num_extra"] = metrics["extra"]
            out["closest_hit_rank"] = float(i)
        if out["exact_hit"]:
            break
    return out


def compute_token_priors(
    builds: list[SendouBuildRow],
    vocab: dict[str, int],
    *,
    mix_global: float = 0.0,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    vocab_tokens = list(vocab.keys())
    allowed = {t for t in vocab_tokens if not t.startswith("<")}

    global_counts: Counter[str] = Counter()
    counts_by_weapon: dict[str, Counter[str]] = defaultdict(Counter)
    n_by_weapon: Counter[str] = Counter()

    for row in builds:
        tokens = tokenize_build(row.abilities_ap)
        tokens = [t for t in tokens if t in allowed]
        global_counts.update(tokens)
        counts_by_weapon[row.weapon_token].update(tokens)
        n_by_weapon[row.weapon_token] += 1

    n_total = sum(n_by_weapon.values())
    if n_total == 0:
        raise ValueError("No builds left after weapon filtering.")

    global_prior = {t: global_counts[t] / n_total for t in allowed}

    priors_by_weapon: dict[str, dict[str, float]] = {}
    for weapon_token, counts in counts_by_weapon.items():
        denom = n_by_weapon[weapon_token]
        prior = {t: counts[t] / denom for t in allowed if counts[t] > 0}
        priors_by_weapon[weapon_token] = prior

    if mix_global > 0.0:
        mixed: dict[str, dict[str, float]] = {}
        for weapon_token, prior in priors_by_weapon.items():
            merged: dict[str, float] = {}
            for t in allowed:
                pw = prior.get(t, 0.0)
                pg = global_prior.get(t, 0.0)
                p = (1.0 - mix_global) * pw + mix_global * pg
                if p > 0:
                    merged[t] = p
            mixed[weapon_token] = merged
        priors_by_weapon = mixed

    return priors_by_weapon, global_prior


def make_weapon_prior_predict_fn(
    priors_by_weapon: dict[str, dict[str, float]],
    global_prior: dict[str, float],
    vocab: dict[str, int],
):
    allowed = [t for t in vocab.keys() if not t.startswith("<")]

    def dense(prior: dict[str, float]) -> dict[str, float]:
        # Keep iteration deterministic for reproducibility.
        return {t: float(prior.get(t, 0.0)) for t in allowed}

    priors_dense = {w: dense(p) for w, p in priors_by_weapon.items()}
    global_dense = dense(global_prior)

    def predict_fn(
        current_tokens: list[str], weapon_id: str
    ) -> dict[str, float]:
        _ = current_tokens  # context-independent baseline
        return priors_dense.get(weapon_id, global_dense)

    return predict_fn


def split_builds_by_weapon(
    builds: list[SendouBuildRow],
    *,
    eval_frac: float,
    seed: int,
) -> tuple[list[SendouBuildRow], list[SendouBuildRow]]:
    if eval_frac <= 0.0:
        return builds, builds

    if not (0.0 < eval_frac < 1.0):
        raise ValueError(f"eval_frac must be in (0, 1), got {eval_frac}")

    rng = random.Random(seed)

    train: list[SendouBuildRow] = []
    eval_rows: list[SendouBuildRow] = []

    by_weapon: dict[str, list[SendouBuildRow]] = defaultdict(list)
    for row in builds:
        by_weapon[row.weapon_token].append(row)

    for _, rows in by_weapon.items():
        if len(rows) < 2:
            train.extend(rows)
            continue

        n_eval = max(1, int(round(len(rows) * eval_frac)))
        n_eval = min(n_eval, len(rows) - 1)
        picked = rng.sample(rows, n_eval)
        picked_keys = {(r.build_id, r.plus_tier, r.weapon_id) for r in picked}
        eval_rows.extend(picked)
        train.extend(
            [
                r
                for r in rows
                if (r.build_id, r.plus_tier, r.weapon_id) not in picked_keys
            ]
        )

    return train, eval_rows


def split_builds_by_tier(
    builds: list[SendouBuildRow],
    *,
    train_tiers: list[int],
    eval_tiers: list[int],
) -> tuple[list[SendouBuildRow], list[SendouBuildRow]]:
    train_set = {int(t) for t in train_tiers}
    eval_set = {int(t) for t in eval_tiers}
    train = [b for b in builds if b.plus_tier in train_set]
    eval_rows = [b for b in builds if b.plus_tier in eval_set]
    return train, eval_rows


def run_eval(
    builds: list[SendouBuildRow],
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
    beam_size: int,
    max_steps: int,
    top_k: int,
    predict_fn,
) -> pd.DataFrame:
    rng = random.Random(seed)
    allocator = Allocator()

    truth_by_key: dict[tuple[int, int, int], Build] = {}
    for row in builds:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_by_key:
            continue
        truth_slots = ap_to_slot_items(row.abilities_ap)
        truth_by_key[key] = slots_to_build(truth_slots)

    eval_rows: list[dict[str, float]] = []
    for mask in masks:
        candidates = list(builds)
        if limit_per_mask is not None:
            candidates = rng.sample(
                candidates, min(limit_per_mask, len(candidates))
            )

        for row in candidates:
            masked_abilities = mask_abilities(row.abilities_ap, mask, rng)
            context_tokens = tokenize_build(masked_abilities)

            pred_builds = (
                reconstruct_build(
                    predict_fn=predict_fn,
                    weapon_id=row.weapon_token,
                    initial_context=context_tokens,
                    allocator=allocator,
                    beam_size=beam_size,
                    max_steps=max_steps,
                    top_k=top_k,
                )
                or []
            )
            truth = truth_by_key[(row.build_id, row.plus_tier, row.weapon_id)]
            metrics = evaluate_top_k(truth, pred_builds)
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
    builds: list[SendouBuildRow],
    *,
    masks: list[int],
    limit_per_mask: int | None,
    seed: int,
    beam_size: int,
    max_steps: int,
    top_k: int,
    predict_fn,
    include_predictions: bool = False,
    include_context: bool = True,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rng = random.Random(seed)
    allocator = Allocator()

    truth_by_key: dict[tuple[int, int, int], Build] = {}
    for row in builds:
        key = (row.build_id, row.plus_tier, row.weapon_id)
        if key in truth_by_key:
            continue
        truth_slots = ap_to_slot_items(row.abilities_ap)
        truth_by_key[key] = slots_to_build(truth_slots)

    eval_rows: list[dict[str, object]] = []
    for mask in masks:
        candidates = list(builds)
        if limit_per_mask is not None:
            candidates = rng.sample(
                candidates, min(limit_per_mask, len(candidates))
            )

        for row in candidates:
            masked_abilities = mask_abilities(row.abilities_ap, mask, rng)
            context_tokens = tokenize_build(masked_abilities)

            pred_builds = (
                reconstruct_build(
                    predict_fn=predict_fn,
                    weapon_id=row.weapon_token,
                    initial_context=context_tokens,
                    allocator=allocator,
                    beam_size=beam_size,
                    max_steps=max_steps,
                    top_k=top_k,
                )
                or []
            )
            truth = truth_by_key[(row.build_id, row.plus_tier, row.weapon_id)]
            metrics = evaluate_top_k(truth, pred_builds)

            top1_ap = pred_builds[0].achieved_ap if pred_builds else None
            best_ap = None
            best_acc = float("-inf")
            for pred in pred_builds:
                acc = compare_builds(truth, pred)["accuracy"]
                if acc > best_acc:
                    best_acc = acc
                    best_ap = pred.achieved_ap

            rec: dict[str, object] = {
                "build_id": row.build_id,
                "plus_tier": row.plus_tier,
                "weapon_id": row.weapon_id,
                "weapon_token": row.weapon_token,
                "ability_mask": int(mask),
                "n_predictions": len(pred_builds),
                "truth_achieved_ap": dict(truth.achieved_ap),
                "predicted_top1_achieved_ap": (
                    None if top1_ap is None else dict(top1_ap)
                ),
                "predicted_best_achieved_ap": (
                    None if best_ap is None else dict(best_ap)
                ),
                **{k: float(v) for k, v in metrics.items()},
            }
            if include_context:
                rec["truth_abilities_ap"] = dict(row.abilities_ap)
                rec["masked_abilities_ap"] = dict(masked_abilities)
                rec["context_tokens"] = list(context_tokens)
            if include_predictions:
                rec["predicted_builds"] = [b.to_dict() for b in pred_builds]
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
        "--masks",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        help="Number of slots to drop (e.g. 1 2 3 6 9).",
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument(
        "--mix-global",
        type=float,
        default=0.1,
        help="Interpolate weapon prior with global prior.",
    )
    parser.add_argument(
        "--train-tiers",
        type=int,
        nargs="+",
        default=[2, 3],
        help="Tiers to compute priors from (defaults to 2 3).",
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
        "--include-predictions",
        action="store_true",
        help="Include predicted build dicts (can make output large).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable writing the output JSON file.",
    )
    args = parser.parse_args()

    weapon_vocab = _load_json(args.weapon_vocab)
    vocab = _load_json(args.vocab)

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

    priors_by_weapon, global_prior = compute_token_priors(
        train_builds, vocab=vocab, mix_global=args.mix_global
    )
    predict_fn = make_weapon_prior_predict_fn(
        priors_by_weapon=priors_by_weapon,
        global_prior=global_prior,
        vocab=vocab,
    )

    t0 = time.time()
    summary, rows = run_eval_detailed(
        eval_builds,
        masks=args.masks,
        limit_per_mask=(args.limit if args.limit > 0 else None),
        seed=args.seed,
        beam_size=args.beam_size,
        max_steps=args.max_steps,
        top_k=args.top_k,
        predict_fn=predict_fn,
        include_predictions=bool(args.include_predictions),
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
            mix = str(args.mix_global).replace(".", "p")
            out_path = Path("tmp_results") / (
                "sendou_weapon_prior_"
                f"train{tiers_train}_eval{tiers_eval}_"
                f"masks{masks}_limit{args.limit}_seed{args.seed}_"
                f"beam{args.beam_size}_steps{args.max_steps}_"
                f"top{args.top_k}_mix{mix}.json"
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "method": "weapon_prior",
                "csv": str(args.csv),
                "vocab": str(args.vocab),
                "weapon_vocab": str(args.weapon_vocab),
                "train_tiers": list(args.train_tiers),
                "eval_tiers": list(args.eval_tiers),
                "masks": list(args.masks),
                "limit": int(args.limit),
                "seed": int(args.seed),
                "beam_size": int(args.beam_size),
                "max_steps": int(args.max_steps),
                "top_k": int(args.top_k),
                "mix_global": float(args.mix_global),
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
