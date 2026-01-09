from __future__ import annotations

import argparse
from pathlib import Path

import orjson
import torch

from splatnlp.model_embeddings.harness import (
    build_embedding_dataloader,
    extract_training_embeddings,
)
from splatnlp.model_embeddings.io import load_json, load_tokenized_data
from splatnlp.model_embeddings.model_loader import build_model, resolve_model_params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embeddings from training-style masked contexts."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--weapon-vocab-path", type=str, required=True)
    parser.add_argument("--model-checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--model-params", type=str)
    parser.add_argument("--table-name", type=str)

    parser.add_argument("--num-masks-per-set", type=int, default=5)
    parser.add_argument("--skew-factor", type=float, default=1.2)
    parser.add_argument("--include-null", action="store_true")
    parser.add_argument("--limit", type=int)

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
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
    if args.limit is not None:
        df = df.head(args.limit)

    pad_token_id = vocab.get("<PAD>")
    if pad_token_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary")
    null_token_id = vocab.get("<NULL>") if args.include_null else None

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

    dataloader = build_embedding_dataloader(
        df,
        vocab_size=len(vocab),
        pad_token_id=pad_token_id,
        num_instances_per_set=args.num_masks_per_set,
        skew_factor=args.skew_factor,
        null_token_id=null_token_id,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=args.shuffle,
        pin_memory=device == "cuda",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = extract_training_embeddings(
        model,
        dataloader,
        pad_token_id=pad_token_id,
        output_dir=str(output_dir),
        device=device,
        normalize=args.normalize,
    )

    run_config = {
        "data_path": args.data_path,
        "vocab_path": args.vocab_path,
        "weapon_vocab_path": args.weapon_vocab_path,
        "model_checkpoint": args.model_checkpoint,
        "model_params": args.model_params,
        "table_name": args.table_name,
        "num_masks_per_set": args.num_masks_per_set,
        "skew_factor": args.skew_factor,
        "include_null": args.include_null,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle": args.shuffle,
        "device": device,
        "normalize": args.normalize,
        "limit": args.limit,
        "resolved_model_params": params,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    config_path = output_dir / "run_config.json"
    config_path.write_bytes(orjson.dumps(run_config, option=orjson.OPT_INDENT_2))


if __name__ == "__main__":
    main()
