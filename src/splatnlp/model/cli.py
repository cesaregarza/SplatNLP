import argparse
import io
import os
import sqlite3

import boto3
import orjson
import pandas as pd
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast

from splatnlp.model.config import TrainingConfig
from splatnlp.model.evaluation import test_model
from splatnlp.model.models import SetCompletionModel
from splatnlp.model.training_loop import train_model
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.utils.constants import PAD


def load_vocab(vocab_path):
    if vocab_path.startswith(("http://", "https://", "s3://")):
        if vocab_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = vocab_path[5:].split("/", 1)
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        else:
            import requests

            response = requests.get(vocab_path)
            content = response.content
        return orjson.loads(content)
    else:
        with open(vocab_path, "rb") as f:
            return orjson.loads(f.read())


def load_data(data_path, table_name=None):
    if data_path.startswith(("http://", "https://", "s3://")):
        if data_path.startswith("s3://"):
            s3 = boto3.client("s3")
            bucket, key = data_path[5:].split("/", 1)
            response = s3.get_object(Bucket=bucket, Key=key)
            content = response["Body"].read()
        else:
            import requests

            response = requests.get(data_path)
            content = response.content
        return pd.read_csv(io.BytesIO(content), sep="\t", header=0)
    elif data_path.endswith(".db") or data_path.endswith(".sqlite"):
        conn = sqlite3.connect(data_path)
        if table_name:
            return pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        else:
            return pd.read_sql_query("SELECT * FROM data", conn)
    else:
        return pd.read_csv(data_path, sep="\t")


def main():
    parser = argparse.ArgumentParser(description="Train a SetCompletionModel")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the tokenized dataset (local file, S3 path, or SQLite DB)",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file (local file or S3 path)",
    )
    parser.add_argument(
        "--weapon_vocab_path",
        type=str,
        required=True,
        help="Path to the weapon vocabulary JSON file (local file or S3 path)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--embedding_dim",
        type=int,
        default=32,
        help="Dimension of token embeddings",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Dimension of hidden layers"
    )
    parser.add_argument(
        "--num_layers", type=int, default=3, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=8, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=32,
        help="Number of inducing points",
    )
    parser.add_argument(
        "--use_layer_norm",
        type=bool,
        default=False,
        help="Use layer normalization",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.3, help="Dropout rate"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="Learning rate"
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01, help="Weight decay"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=1.0,
        help="Fraction of the dataset to use for training",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        help="Table name for SQLite database (optional)",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="Enable verbose output",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Number of epochs with no improvement after which training will be stopped",
    )
    parser.add_argument(
        "--clip_grad_norm",
        type=float,
        default=1.0,
        help="Maximum norm of gradients for gradient clipping",
    )
    parser.add_argument(
        "--scheduler_factor",
        type=float,
        default=0.1,
        help="Factor by which the learning rate will be reduced",
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=2,
        help="Number of epochs with no improvement after which learning rate will be reduced",
    )
    parser.add_argument(
        "--use_mixed_precision",
        type=bool,
        default=False,
        help="Enable mixed precision training for H100 GPUs",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--metric_update_interval",
        type=int,
        default=1,
        help="Interval for updating metrics during training and validation",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable Distributed Data Parallel (DDP)",
    )
    parser.add_argument(
        "--num_masks_per_set",
        type=int,
        default=5,
        help="Number of masked instances to generate per set (default: 5)",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1.  Initialise DDP  (one process per GPU, launched by torchrun)
    # ------------------------------------------------------------------
    if args.distributed:
        dist.init_process_group(backend="nccl", init_method="env://")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        # Limit all prints & tqdm to rank‑0 only
        if local_rank != 0:

            def _silence(*a, **kw):
                pass

            import builtins

            import tqdm as _tqdm

            builtins.print = _silence
            _tqdm.tqdm = lambda *a, **k: a[0]  # no‑op iterator
    else:
        local_rank = 0

    if args.verbose:
        print("Loading data and vocabulary...")

    # Load data and vocabulary
    df = load_data(args.data_path, args.table_name)
    if args.verbose:
        print(f"Loaded {len(df)} rows from {args.data_path}")
        print(f"Columns: {df.columns}")
        print(f"Heads: {df.head()}")

    df["ability_tags"] = df["ability_tags"].apply(orjson.loads)
    vocab = load_vocab(args.vocab_path)
    weapon_vocab = load_vocab(args.weapon_vocab_path)

    if args.verbose:
        print("Generating datasets...")

    # Generate datasets
    train_df, val_df, test_df = generate_tokenized_datasets(
        df, frac=args.fraction
    )

    if args.verbose:
        print("Generating dataloaders...")

    # Generate dataloaders
    train_dl, val_dl, test_dl = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=len(vocab),
        pad_token_id=vocab[PAD],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False,
        persistent_workers=(
            True if args.device == "cuda" and args.num_workers > 0 else False
        ),
        distributed=args.distributed,
        num_instances_per_set=args.num_masks_per_set,
    )

    if args.verbose:
        print("Creating model...")

    # Create model
    model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(vocab),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
        pad_token_id=vocab[PAD],
    )

    # Create training config
    config = TrainingConfig(
        num_epochs=args.num_epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        device=args.device,
        distributed=args.distributed,
    )

    if args.verbose:
        print("Starting model training...")

    # Train model
    scaler = GradScaler() if args.use_mixed_precision else None
    metrics_history, trained_model = train_model(
        model,
        train_dl,
        val_dl,
        config,
        vocab,
        verbose=args.verbose,
        scaler=scaler,
        metric_update_interval=args.metric_update_interval,
        ddp=args.distributed,
    )

    if args.verbose:
        print("Evaluating model...")

    test_model(
        model=model,
        test_dl=test_dl,
        config=config,
        vocab=vocab,
        pad_token=PAD,
        verbose=args.verbose,
        ddp=args.distributed,
    )

    if args.verbose:
        print("Saving model, metrics, and parameters...")

    # Save model, metrics, and parameters (only on rank-0 in distributed mode)
    if (not args.distributed) or (torch.distributed.get_rank() == 0):
        os.makedirs(args.output_dir, exist_ok=True)
        torch.save(
            trained_model.state_dict(),
            os.path.join(args.output_dir, "model.pth"),
        )
        with open(
            os.path.join(args.output_dir, "metrics_history.json"), "wb"
        ) as f:
            f.write(
                orjson.dumps(metrics_history, option=orjson.OPT_SERIALIZE_NUMPY)
            )

        # Save model parameters
        model_params = {
            "vocab_size": len(vocab),
            "weapon_vocab_size": len(weapon_vocab),
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "output_dim": len(vocab),
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_inducing_points": args.num_inducing_points,
            "use_layer_norm": args.use_layer_norm,
            "dropout": args.dropout,
            "pad_token_id": vocab[PAD],
            "use_mixed_precision": args.use_mixed_precision,
        }
        with open(os.path.join(args.output_dir, "model_params.json"), "w") as f:
            orjson.dumps(model_params, f)

        print(f"Model, metrics, and parameters saved in {args.output_dir}")


if __name__ == "__main__":
    main()
