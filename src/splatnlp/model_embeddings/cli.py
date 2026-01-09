from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import torch

from splatnlp.model_embeddings.extract import extract_embeddings
from splatnlp.model_embeddings.io import load_json, load_tokenized_data
from splatnlp.model_embeddings.model_loader import build_model, resolve_model_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract SetCompletionModel pooled embeddings."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--weapon-vocab-path", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-params", type=str)
    parser.add_argument("--table-name", type=str)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--limit", type=int)

    parser.add_argument("--device", type=str)
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

    vocab = load_json(args.vocab_path)
    weapon_vocab = load_json(args.weapon_vocab_path)
    df = load_tokenized_data(args.data_path, table_name=args.table_name)

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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = extract_embeddings(
        model,
        df,
        vocab,
        weapon_vocab,
        output_dir,
        batch_size=args.batch_size,
        device=device,
        normalize=args.normalize,
        limit=args.limit,
    )

    run_config = {
        "data_path": args.data_path,
        "vocab_path": args.vocab_path,
        "weapon_vocab_path": args.weapon_vocab_path,
        "model_checkpoint": args.model_checkpoint,
        "model_params": args.model_params,
        "table_name": args.table_name,
        "batch_size": args.batch_size,
        "limit": args.limit,
        "device": device,
        "normalize": args.normalize,
        "resolved_model_params": params,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    config_path = output_dir / "run_config.json"
    config_path.write_bytes(orjson.dumps(run_config, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
