from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import orjson
import torch
import torch.distributed as dist

from splatnlp.model_embeddings.harness import (
    build_embedding_dataloader,
    extract_training_embeddings,
)
from splatnlp.model_embeddings.io import load_json, load_tokenized_data
from splatnlp.model_embeddings.model_loader import (
    build_model,
    load_checkpoint,
    resolve_model_params,
)


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
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--normalize", dest="normalize", action="store_true")
    parser.add_argument(
        "--no-normalize", dest="normalize", action="store_false"
    )
    parser.set_defaults(normalize=True)
    parser.add_argument("--log-interval", type=int, default=50)

    parser.add_argument(
        "--wandb-log",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb-project", type=str, default="splatnlp-embeddings"
    )
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--wandb-tags", type=str, nargs="*")

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

    rank = 0
    world_size = 1
    if args.distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed extraction requires CUDA")
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    logging.basicConfig(
        level=logging.INFO if rank == 0 else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger = logging.getLogger(__name__)

    device = args.device
    if args.distributed:
        device = "cuda"
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    vocab = load_json(args.vocab_path)
    weapon_vocab = load_json(args.weapon_vocab_path)
    df = load_tokenized_data(args.data_path, table_name=args.table_name)
    if args.limit is not None:
        df = df.head(args.limit)
    full_rows = len(df)
    if args.distributed and world_size > 1:
        df = df.iloc[rank::world_size].reset_index(drop=True)

    pad_token_id = vocab.get("<PAD>")
    if pad_token_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary")
    null_token_id = vocab.get("<NULL>") if args.include_null else None

    params = resolve_model_params(
        args.model_checkpoint,
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
    state_dict = load_checkpoint(args.model_checkpoint, map_location="cpu")
    model.load_state_dict(state_dict)

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

    output_root = Path(args.output_dir)
    output_dir = (
        output_root / f"rank_{rank:02d}"
        if args.distributed
        else output_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    instances_per_set = args.num_masks_per_set + (
        1 if args.include_null else 0
    )
    shard_total = len(df) * instances_per_set
    global_total = full_rows * instances_per_set

    wandb_run = None
    if args.wandb_log:
        try:
            import wandb
            from wandb.util import generate_id
        except Exception as exc:  # pragma: no cover - optional dep runtime
            raise RuntimeError(
                "Weights & Biases logging requested but wandb is unavailable."
            ) from exc

        if args.distributed and rank != 0:
            wandb_run = None
        else:
            run_name = (
                args.wandb_run_name or f"embeddings_{generate_id()}"
            )
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config={
                    **vars(args),
                    "world_size": world_size,
                    "full_rows": full_rows,
                    "shard_rows": len(df),
                    "instances_per_set": instances_per_set,
                    "resolved_model_params": params,
                },
                name=run_name,
                tags=args.wandb_tags,
                dir=str(output_root),
            )
            logger.info("Weights & Biases run initialised: %s", wandb.run.url)

    progress_callback = None
    if wandb_run:
        def _progress_callback(
            step: int, seen: int, metrics: dict[str, float]
        ) -> None:
            payload = {
                "rows_processed": seen,
                "rows_per_sec": metrics["rows_per_sec"],
                "elapsed_sec": metrics["elapsed_sec"],
                "embedding_norm_mean": metrics["embedding_norm_mean"],
                "batch_size": metrics["batch_size"],
                "progress": seen / shard_total if shard_total else 1.0,
            }
            if world_size > 1:
                global_seen = min(seen * world_size, global_total)
                payload["rows_processed_global"] = global_seen
                payload["progress_global"] = (
                    global_seen / global_total if global_total else 1.0
                )
            wandb_run.log(payload, step=seen)

        progress_callback = _progress_callback

    outputs = extract_training_embeddings(
        model,
        dataloader,
        pad_token_id=pad_token_id,
        output_dir=str(output_dir),
        device=device,
        normalize=args.normalize,
        progress_callback=progress_callback,
        log_interval=args.log_interval,
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
        "distributed": args.distributed,
        "rank": rank,
        "world_size": world_size,
        "full_rows": full_rows,
        "shard_rows": len(df),
        "instances_per_set": instances_per_set,
        "log_interval": args.log_interval,
        "resolved_model_params": params,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    config_path = output_dir / "run_config.json"
    config_path.write_bytes(
        orjson.dumps(run_config, option=orjson.OPT_INDENT_2)
    )

    if args.distributed and dist.is_initialized():
        dist.barrier()
        if rank == 0:
            manifest = {
                "distributed": True,
                "world_size": world_size,
                "output_root": str(output_root),
                "instances_per_set": instances_per_set,
                "full_rows": full_rows,
                "global_total": global_total,
                "normalize": args.normalize,
                "model_checkpoint": args.model_checkpoint,
                "shards": [
                    {
                        "rank": shard_rank,
                        "output_dir": str(
                            output_root / f"rank_{shard_rank:02d}"
                        ),
                        "embeddings": str(
                            output_root
                            / f"rank_{shard_rank:02d}"
                            / "embeddings.npy"
                        ),
                        "metadata": str(
                            output_root
                            / f"rank_{shard_rank:02d}"
                            / "metadata.jsonl"
                        ),
                    }
                    for shard_rank in range(world_size)
                ],
            }
            manifest_path = output_root / "manifest.json"
            manifest_path.write_bytes(
                orjson.dumps(manifest, option=orjson.OPT_INDENT_2)
            )

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    main()
