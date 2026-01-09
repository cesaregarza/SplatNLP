from __future__ import annotations

import argparse
import json
from pathlib import Path

import orjson
import torch

from splatnlp.model_embeddings.io import load_json
from splatnlp.model_embeddings.model_loader import build_model, resolve_model_params
from splatnlp.model_embeddings.trajectory import build_predictors
from splatnlp.utils.constants import TOKEN_BONUS
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build


def _parse_tokens(value: str | None) -> list[str]:
    if not value:
        return []
    stripped = value.strip()
    if stripped.startswith("["):
        return json.loads(stripped)
    return [tok.strip() for tok in stripped.split(",") if tok.strip()]


def _normalize_weapon_id(
    weapon_id: str, weapon_vocab: dict[str, int]
) -> str:
    if weapon_id in weapon_vocab:
        return weapon_id
    if weapon_id.isdigit():
        token = f"weapon_id_{weapon_id}"
        if token in weapon_vocab:
            return token
    raise ValueError(f"Unknown weapon identifier: {weapon_id}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run beam search and capture pooled intent embeddings."
    )
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--weapon-vocab-path", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--model-params", type=str)
    parser.add_argument("--device", type=str)

    parser.add_argument("--weapon-id", type=str, required=True)
    parser.add_argument("--initial-tokens", type=str)
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=6)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--greedy-threshold", type=float, default=0.5)
    parser.add_argument("--min-new-token-prob", type=float, default=0.01)
    parser.add_argument("--token-bonus", type=float, default=TOKEN_BONUS)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false"
    )
    parser.set_defaults(normalize=True)

    parser.add_argument("--embedding-dim", type=int)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    parser.add_argument("--num-inducing-points", type=int)
    parser.add_argument(
        "--use-layer-norm", dest="use_layer_norm", action="store_true"
    )
    parser.add_argument(
        "--no-layer-norm", dest="use_layer_norm", action="store_false"
    )
    parser.set_defaults(use_layer_norm=None)
    parser.add_argument("--dropout", type=float)

    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    vocab = load_json(args.vocab_path)
    weapon_vocab = load_json(args.weapon_vocab_path)

    pad_token_id = vocab.get("<PAD>")
    if pad_token_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary")

    checkpoint_path = Path(args.model_checkpoint)
    params = resolve_model_params(
        checkpoint_path,
        args.model_params,
        {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_inducing_points": args.num_inducing_points,
            "use_layer_norm": args.use_layer_norm,
            "dropout": args.dropout,
        },
    )

    model = build_model(
        params,
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        pad_token_id=pad_token_id,
    )
    model.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )
    model.to(torch_device)
    model.eval()

    weapon_id = _normalize_weapon_id(args.weapon_id, weapon_vocab)
    initial_tokens = _parse_tokens(args.initial_tokens)

    allocator = Allocator()
    predictor = build_predictors(
        model,
        vocab,
        weapon_vocab,
        device=torch_device,
        normalize=args.normalize,
    )

    try:
        builds, traces = reconstruct_build(
            predict_fn=predictor.predict,
            predict_batch_fn=predictor.predict_batch,
            weapon_id=weapon_id,
            initial_context=initial_tokens,
            allocator=allocator,
            beam_size=args.beam_size,
            max_steps=args.max_steps,
            top_k=args.top_k,
            greedy_threshold=args.greedy_threshold,
            min_new_token_prob=args.min_new_token_prob,
            token_bonus=args.token_bonus,
            alpha=args.alpha,
            record_traces=True,
        )
    finally:
        predictor.close()

    output = {
        "weapon_id": weapon_id,
        "initial_tokens": initial_tokens,
        "builds": None
        if builds is None
        else [build.to_dict() for build in builds],
        "traces": None
        if traces is None
        else [[frame.to_dict() for frame in trace] for trace in traces],
        "config": {
            "beam_size": args.beam_size,
            "max_steps": args.max_steps,
            "top_k": args.top_k,
            "greedy_threshold": args.greedy_threshold,
            "min_new_token_prob": args.min_new_token_prob,
            "token_bonus": args.token_bonus,
            "alpha": args.alpha,
            "normalize": args.normalize,
        },
        "model_params": params,
    }

    output_path = Path(args.output)
    output_path.write_bytes(orjson.dumps(output, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
