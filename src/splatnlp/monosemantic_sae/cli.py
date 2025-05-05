import argparse
import io
import os
import sqlite3

import boto3
import orjson
import pandas as pd
import torch
from torch.cuda.amp import GradScaler

from splatnlp.model.config import TrainingConfig
from splatnlp.model.models import SetCompletionModel
from splatnlp.model.training_loop import train_model
from splatnlp.monosemantic_sae.models import ModifiedSetCompletionModel
from splatnlp.monosemantic_sae.training_loop import train_autoencoder
from splatnlp.preprocessing.constants import PAD
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)


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
    parser = argparse.ArgumentParser(
        description="Train an Autoencoder for SetCompletionModel or a standard SetCompletionModel"
    )
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
        "--pretrained_model_path",
        type=str,
        help="Path to the pretrained SetCompletionModel for autoencoder training",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the trained model",
    )
    parser.add_argument(
        "--autoencoder_dim",
        type=int,
        default=64,
        help="Dimension of the autoencoder's hidden layer",
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
        "--embedding_dim",
        type=int,
        default=32,
        help="Dimension of the embedding layer",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=512,
        help="Dimension of the hidden layer",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=32,
        help="Number of inducing points for attention",
    )
    parser.add_argument(
        "--use_layer_norm",
        type=bool,
        default=True,
        help="Use layer normalization",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="Dropout rate",
    )
    parser.add_argument(
        "--metric_update_interval",
        type=int,
        default=100,
        help="Interval for updating metrics during training",
    )

    args = parser.parse_args()

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
    train_dl, val_dl, _ = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=len(vocab),
        pad_token_id=vocab[PAD],
        batch_size=args.batch_size,
        num_workers=args.num_workers,
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
    )

    scaler = GradScaler() if args.use_mixed_precision else None

    if args.pretrained_model_path:
        if args.verbose:
            print("Loading pretrained model for autoencoder training...")

        # Create modified model with autoencoder using from_pretrained method
        model = ModifiedSetCompletionModel.from_pretrained(
            pretrained_model_path=args.pretrained_model_path,
            autoencoder_dim=args.autoencoder_dim,
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

        if args.verbose:
            print("Starting autoencoder training...")

        # Train autoencoder
        metrics_history, trained_model = train_autoencoder(
            model,
            train_dl,
            val_dl,
            config,
            vocab,
            pad_token=PAD,
            verbose=args.verbose,
            scaler=scaler,
        )
    else:
        if args.verbose:
            print("Creating model for standard training...")

        # Create standard model
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

        if args.verbose:
            print("Starting model training...")

        # Train model
        metrics_history, trained_model = train_model(
            model,
            train_dl,
            val_dl,
            config,
            vocab,
            verbose=args.verbose,
            scaler=scaler,
            metric_update_interval=args.metric_update_interval,
        )

    if args.verbose:
        print("Saving model, metrics, and parameters...")

    # Save model, metrics, and parameters
    os.makedirs(args.output_dir, exist_ok=True)

    if args.pretrained_model_path:
        torch.save(
            trained_model.state_dict(),
            os.path.join(args.output_dir, "autoencoder_model.pth"),
        )
        # Save autoencoder-specific parameters
        model_params = {
            "autoencoder_dim": args.autoencoder_dim,
            "use_mixed_precision": args.use_mixed_precision,
        }
    else:
        torch.save(
            trained_model.state_dict(),
            os.path.join(args.output_dir, "model.pth"),
        )
        # Save standard model parameters
        model_params = {
            "embedding_dim": args.embedding_dim,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "num_inducing_points": args.num_inducing_points,
            "use_layer_norm": args.use_layer_norm,
            "dropout": args.dropout,
            "use_mixed_precision": args.use_mixed_precision,
        }

    with open(os.path.join(args.output_dir, "metrics_history.json"), "wb") as f:
        f.write(
            orjson.dumps(metrics_history, option=orjson.OPT_SERIALIZE_NUMPY)
        )

    with open(os.path.join(args.output_dir, "model_params.json"), "w") as f:
        orjson.dumps(model_params, f)

    print(f"Model, metrics, and parameters saved in {args.output_dir}")


if __name__ == "__main__":
    main()
