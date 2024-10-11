import argparse
import os

import orjson
import pandas as pd
import torch

from splatnlp.model.config import TrainingConfig
from splatnlp.model.models import SetCompletionModel
from splatnlp.model.training_loop import train_model
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)


def load_vocab(vocab_path):
    with open(vocab_path, "rb") as f:
        return orjson.loads(f.read())


def main():
    parser = argparse.ArgumentParser(description="Train a SetCompletionModel")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to the tokenized dataset",
    )
    parser.add_argument(
        "--vocab_path",
        type=str,
        required=True,
        help="Path to the vocabulary JSON file",
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
        default=128,
        help="Dimension of token embeddings",
    )
    parser.add_argument(
        "--hidden_dim", type=int, default=256, help="Dimension of hidden layers"
    )
    parser.add_argument(
        "--output_dim",
        type=int,
        default=128,
        help="Dimension of output representation",
    )
    parser.add_argument(
        "--num_layers", type=int, default=2, help="Number of transformer layers"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_inducing_points",
        type=int,
        default=32,
        help="Number of inducing points",
    )
    parser.add_argument(
        "--use_layer_norm", action="store_true", help="Use layer normalization"
    )
    parser.add_argument(
        "--dropout", type=float, default=0.1, help="Dropout rate"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="Learning rate"
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

    args = parser.parse_args()

    # Load data and vocabulary
    df = pd.read_csv(args.data_path)
    vocab = load_vocab(args.vocab_path)

    # Generate datasets
    train_df, val_df, test_df = generate_tokenized_datasets(
        df, frac=args.fraction
    )

    # Generate dataloaders
    train_dl, val_dl, _ = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=len(vocab),
        pad_token_id=vocab["<PAD>"],
        batch_size=args.batch_size,
    )

    # Create model
    model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(
            vocab
        ),  # Assuming weapon vocab is the same as ability vocab
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
        pad_token_id=vocab["<PAD>"],
    )

    # Create training config
    config = TrainingConfig(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_epochs=args.num_epochs,
        device=args.device,
    )

    # Train model
    metrics_history, trained_model = train_model(
        model, train_dl, val_dl, config, vocab
    )

    # Save model and metrics
    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(
        trained_model.state_dict(), os.path.join(args.output_dir, "model.pth")
    )
    with open(os.path.join(args.output_dir, "metrics_history.json"), "wb") as f:
        f.write(orjson.dumps(metrics_history))

    print(f"Model and metrics saved in {args.output_dir}")


if __name__ == "__main__":
    main()
