"""CLI for measuring information gain from conditioning on one token.

Computes:
  IG = H0 - E_t~p0[ H1(t) ]

By default, this first greedy-closes from the starting context to a branching
point (``--greedy-threshold``) and omits entropy contribution from tokens that
are already in the conditioned context.

Example (Dapple Dualies):
    poetry run python -m splatnlp.mechinterp.cli.info_gain_cli \
      --weapon-id weapon_id_5000 \
      --checkpoint saved_models/dataset_v0_2_super/clean_slate.pth \
      --out tmp_results/info_gain_dapple_ultra.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson
import torch

from splatnlp.mechinterp.analysis.information_gain import (
    information_gain_for_weapon,
    information_gain_matrix,
)
from splatnlp.mechinterp.skill_helpers.context_loader import (
    _get_weapon_id_to_name,
)
from splatnlp.model.models import SetCompletionModel


def _load_json(path: Path) -> dict[str, int]:
    return json.loads(path.read_text())


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


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--weapon-id",
        type=str,
        help="Weapon token id like weapon_id_5000 (Dapple Dualies).",
    )
    group.add_argument(
        "--all-weapons",
        action="store_true",
        help="Compute information gain for every weapon in weapon_vocab.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("saved_models/dataset_v0_2_super/clean_slate.pth"),
        help="Model checkpoint to load.",
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
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--greedy-threshold",
        type=float,
        default=0.7,
        help="Greedy-closure accept threshold for reaching a branching point.",
    )
    parser.add_argument(
        "--stack-policy",
        type=str,
        default="none",
        choices=["none", "next-tier", "max-tier"],
        help="Candidate-set policy for stacking-tier abilities.",
    )
    parser.add_argument(
        "--include-conditioned",
        action="store_true",
        help="Include already-conditioned tokens in entropy calculations.",
    )
    parser.add_argument(
        "--use-matrix",
        action="store_true",
        help=(
            "Use the fast matrix implementation (ignores greedy closure and "
            "conditioning omission)."
        ),
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help=(
            "Write a larger JSON including aligned per-token vectors "
            "(p0/h1/expected_h1_per_token/ig_contribution)."
        ),
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=1024,
        help="Batch size for (weapon, token) pairs in --all-weapons mode.",
    )
    parser.add_argument(
        "--log-base",
        type=float,
        default=2.0,
        help="Entropy log base (2 -> bits, e -> nats).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=15,
        help="How many tokens to print in summaries.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional JSON output path.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but not available.")

    vocab = _load_json(args.vocab)
    weapon_vocab = _load_json(args.weapon_vocab)
    model = _load_model(
        args.checkpoint, vocab=vocab, weapon_vocab=weapon_vocab, device=device
    )

    if args.all_weapons:
        omit_conditioned = not bool(args.include_conditioned)
        top_n = max(0, int(args.top_n))
        id_to_name = _get_weapon_id_to_name()
        tokens = sorted(t for t in vocab.keys() if not t.startswith("<"))
        tok_to_idx = {t: i for i, t in enumerate(tokens)}

        if args.use_matrix:
            matrices = information_gain_matrix(
                model=model,
                vocab=vocab,
                weapon_vocab=weapon_vocab,
                weapon_ids=None,
                pair_batch_size=int(args.pair_batch_size),
                log_base=float(args.log_base),
                base_context=None,
                device=device,
            )
            tokens = matrices["tokens"]
            weapon_ids: list[str] = matrices["weapon_ids"]
            p0 = matrices["p0"]
            h0 = matrices["h0"]
            h1 = matrices["h1"]
            expected_h1 = matrices["expected_h1"]
            information_gain = matrices["information_gain"]

            weapons_out = []
            for i, weapon_id in enumerate(weapon_ids):
                numeric_id = weapon_id.split("_")[-1]
                weapon_label = id_to_name.get(
                    numeric_id, f"Weapon {numeric_id}"
                )
                contrib = p0[i] * (h0[i] - h1[i])
                if top_n:
                    k = min(top_n, contrib.numel())
                    vals, idx = torch.topk(contrib, k=k)
                    top_contrib = [
                        {
                            "token": tokens[int(j)],
                            "ig_contribution": float(vals[n]),
                            "p0": float(p0[i, int(j)]),
                            "h1": float(h1[i, int(j)]),
                        }
                        for n, j in enumerate(idx.tolist())
                    ]
                else:
                    top_contrib = []

                weapons_out.append(
                    {
                        "weapon_id": weapon_id,
                        "weapon_label": weapon_label,
                        "h0": float(h0[i]),
                        "expected_h1": float(expected_h1[i]),
                        "information_gain": float(information_gain[i]),
                        "top_ig_contributions": top_contrib,
                    }
                )
        else:
            weapon_ids = sorted(weapon_vocab.keys())
            weapons_out = []
            for weapon_id in weapon_ids:
                numeric_id = weapon_id.split("_")[-1]
                weapon_label = id_to_name.get(
                    numeric_id, f"Weapon {numeric_id}"
                )
                result = information_gain_for_weapon(
                    model=model,
                    weapon_id=str(weapon_id),
                    vocab=vocab,
                    weapon_vocab=weapon_vocab,
                    batch_size=int(args.batch_size),
                    log_base=float(args.log_base),
                    greedy_threshold=float(args.greedy_threshold),
                    omit_conditioned=omit_conditioned,
                    stack_policy=str(args.stack_policy),
                    device=device,
                )
                out_row: dict[str, object] = {
                    "weapon_id": weapon_id,
                    "weapon_label": weapon_label,
                    "h0": float(result.h0),
                    "expected_h1": float(result.expected_h1),
                    "information_gain": float(result.information_gain),
                    "branch_context": list(result.branch_context),
                    "branch_context_len": int(len(result.branch_context)),
                    "stack_policy": result.stack_policy,
                    "omit_conditioned": result.omit_conditioned,
                    "family_ig_contribution": dict(
                        result.family_ig_contribution
                    ),
                    "family_ig_contribution_pct": dict(
                        result.family_ig_contribution_pct
                    ),
                    "top_ig_contributions": [
                        r.to_dict() for r in result.tokens[:top_n]
                    ],
                }

                if args.detailed:
                    p0_vec: list[float] = [0.0] * len(tokens)
                    h1_vec: list[float | None] = [None] * len(tokens)
                    expected_h1_vec: list[float] = [0.0] * len(tokens)
                    ig_contrib_vec: list[float] = [0.0] * len(tokens)
                    for row in result.tokens:
                        idx = tok_to_idx.get(row.token)
                        if idx is None:
                            continue
                        p0_vec[idx] = float(row.p0)
                        h1_vec[idx] = float(row.h1)
                        expected_h1_vec[idx] = float(row.expected_h1)
                        ig_contrib_vec[idx] = float(row.ig_contribution)

                    out_row.update(
                        {
                            "support0_len": int(len(result.tokens)),
                            "p0": p0_vec,
                            "h1": h1_vec,
                            "expected_h1_per_token": expected_h1_vec,
                            "ig_contribution": ig_contrib_vec,
                        }
                    )

                weapons_out.append(out_row)

        weapons_out.sort(key=lambda w: w["information_gain"], reverse=True)
        print("Top weapons by information_gain:")
        for row in weapons_out[: min(10, len(weapons_out))]:
            print(
                f"{row['weapon_id']:14s} {row['weapon_label'][:28]:28s} "
                f"IG={row['information_gain']:.6f} "
                f"H0={row['h0']:.6f} E[H1]={row['expected_h1']:.6f}"
            )

        out_path = (
            args.out
            if args.out is not None
            else Path("tmp_results") / "info_gain_all_weapons_ultra.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "checkpoint": str(args.checkpoint),
                "vocab": str(args.vocab),
                "weapon_vocab": str(args.weapon_vocab),
                "device": str(args.device),
                "base_context": ["<NULL>"],
                "mode": "matrix" if bool(args.use_matrix) else "greedy_closure",
                "greedy_threshold": float(args.greedy_threshold),
                "omit_conditioned": omit_conditioned,
                "stack_policy": str(args.stack_policy),
                "detailed": bool(args.detailed),
                "use_matrix": bool(args.use_matrix),
                "pair_batch_size": int(args.pair_batch_size),
                "batch_size": int(args.batch_size),
                "log_base": float(args.log_base),
            },
            "tokens": tokens,
            "weapons": weapons_out,
        }
        out_path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"\nWrote {out_path}")
        return

    assert args.weapon_id is not None
    omit_conditioned = not bool(args.include_conditioned)
    result = information_gain_for_weapon(
        model=model,
        weapon_id=str(args.weapon_id),
        vocab=vocab,
        weapon_vocab=weapon_vocab,
        batch_size=int(args.batch_size),
        log_base=float(args.log_base),
        greedy_threshold=float(args.greedy_threshold),
        omit_conditioned=omit_conditioned,
        stack_policy=str(args.stack_policy),
        device=device,
    )

    print(
        f"weapon_id={result.weapon_id} log_base={result.log_base:g} "
        f"H0={result.h0:.6f} E[H1]={result.expected_h1:.6f} "
        f"IG={result.information_gain:.6f}"
    )
    print(
        f"branch_context_len={len(result.branch_context)} "
        f"greedy_threshold={result.greedy_threshold:g} "
        f"omit_conditioned={result.omit_conditioned} "
        f"stack_policy={result.stack_policy}"
    )

    top_n = max(0, int(args.top_n))
    if top_n:
        print("\nTop IG contributions:")
        for row in result.tokens[:top_n]:
            print(
                f"{row.token:25s} contrib={row.ig_contribution:.6f} "
                f"p0={row.p0:.4f} H1={row.h1:.4f}"
            )

        print("\nTop p0 tokens:")
        by_p0 = sorted(result.tokens, key=lambda r: r.p0, reverse=True)
        for row in by_p0[:top_n]:
            print(f"{row.token:25s} p0={row.p0:.4f} H1={row.h1:.4f}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "weapon_id": str(args.weapon_id),
                "checkpoint": str(args.checkpoint),
                "vocab": str(args.vocab),
                "weapon_vocab": str(args.weapon_vocab),
                "device": str(args.device),
                "batch_size": int(args.batch_size),
                "base_context": ["<NULL>"],
                "mode": "greedy_closure",
                "greedy_threshold": float(args.greedy_threshold),
                "omit_conditioned": omit_conditioned,
                "stack_policy": str(args.stack_policy),
                "log_base": float(args.log_base),
            },
            "result": result.to_dict(),
        }
        args.out.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
