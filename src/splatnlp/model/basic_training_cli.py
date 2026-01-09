from __future__ import annotations

import argparse
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import orjson
import torch

from splatnlp.model.config import TrainingConfig
from splatnlp.model.evaluation import test_model
from splatnlp.model.models import SetCompletionModel
from splatnlp.model.training_loop import train_model
from splatnlp.model_embeddings.io import load_json, load_tokenized_data
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.utils.constants import PAD


@dataclass(frozen=True)
class ModelConfig:
    embedding_dim: int = 32
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    num_inducing_points: int = 32
    use_layer_norm: bool = True
    dropout: float = 0.0


@dataclass(frozen=True)
class DataConfig:
    data_path: str
    vocab_path: str
    weapon_vocab_path: str
    output_dir: str
    table_name: str | None = None
    max_rows: int | None = None
    fraction: float = 0.1
    random_state: int | None = None
    validation_size: float = 0.1
    test_size: float = 0.2


@dataclass(frozen=True)
class RunConfig:
    num_epochs: int = 5
    batch_size: int = 1024
    num_workers: int = 4
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    clip_grad_norm: float = 1.0
    scheduler_factor: float = 0.1
    scheduler_patience: int = 2
    patience: int = 3
    use_mixed_precision: bool = False
    num_masks_per_set: int = 5
    skew_factor: float = 1.2
    include_null: bool = False
    metric_update_interval: int = 1
    device: str = "cpu"
    verbose: bool = True


def _write_json(path: Path, payload: dict) -> None:
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))


def run_basic_training(
    data_config: DataConfig,
    model_config: ModelConfig,
    run_config: RunConfig,
) -> None:
    logger = logging.getLogger(__name__)
    vocab = load_json(data_config.vocab_path)
    weapon_vocab = load_json(data_config.weapon_vocab_path)

    logger.info(
        "Loading tokenized data from %s (max_rows=%s)",
        data_config.data_path,
        data_config.max_rows,
    )
    df = load_tokenized_data(
        data_config.data_path,
        table_name=data_config.table_name,
        max_rows=data_config.max_rows,
    )
    logger.info("Loaded %d rows", len(df))
    train_df, val_df, test_df = generate_tokenized_datasets(
        df,
        frac=data_config.fraction,
        random_state=data_config.random_state,
        validation_size=data_config.validation_size,
        test_size=data_config.test_size,
    )
    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    pad_token_id = vocab.get(PAD)
    if pad_token_id is None:
        raise ValueError("'<PAD>' token missing from vocabulary")
    null_token_id = vocab.get("<NULL>") if run_config.include_null else None

    train_dl, val_dl, test_dl = generate_dataloaders(
        train_df,
        val_df,
        test_df,
        vocab_size=len(vocab),
        pad_token_id=pad_token_id,
        batch_size=run_config.batch_size,
        num_workers=run_config.num_workers,
        pin_memory=run_config.device == "cuda",
        persistent_workers=(
            run_config.device == "cuda" and run_config.num_workers > 0
        ),
        num_instances_per_set=run_config.num_masks_per_set,
        skew_factor=run_config.skew_factor,
        null_token_id=null_token_id,
        shuffle_train=True,
    )

    model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=model_config.embedding_dim,
        hidden_dim=model_config.hidden_dim,
        output_dim=len(vocab),
        num_layers=model_config.num_layers,
        num_heads=model_config.num_heads,
        num_inducing_points=model_config.num_inducing_points,
        use_layer_norm=model_config.use_layer_norm,
        dropout=model_config.dropout,
        pad_token_id=pad_token_id,
    )

    train_cfg = TrainingConfig(
        num_epochs=run_config.num_epochs,
        patience=run_config.patience,
        learning_rate=run_config.learning_rate,
        weight_decay=run_config.weight_decay,
        clip_grad_norm=run_config.clip_grad_norm,
        scheduler_factor=run_config.scheduler_factor,
        scheduler_patience=run_config.scheduler_patience,
        device=run_config.device,
        distributed=False,
    )

    scaler = torch.amp.GradScaler() if run_config.use_mixed_precision else None
    logger.info(
        "Training on %s for %d epochs (batch_size=%d, masks_per_set=%d)",
        run_config.device,
        run_config.num_epochs,
        run_config.batch_size,
        run_config.num_masks_per_set,
    )
    train_start = time.time()
    metrics_history, trained_model = train_model(
        model,
        train_dl,
        val_dl,
        train_cfg,
        vocab,
        verbose=run_config.verbose,
        scaler=scaler,
        metric_update_interval=run_config.metric_update_interval,
        ddp=False,
    )
    logger.info("Training completed in %.2fs", time.time() - train_start)

    logger.info("Running test evaluation")
    test_metrics = test_model(
        model=trained_model,
        test_dl=test_dl,
        config=train_cfg,
        vocab=vocab,
        pad_token=PAD,
        verbose=run_config.verbose,
        ddp=False,
    )

    output_dir = Path(data_config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Saving artifacts to %s", output_dir)
    torch.save(trained_model.state_dict(), output_dir / "model.pth")
    _write_json(output_dir / "metrics_history.json", metrics_history)
    _write_json(output_dir / "test_metrics.json", test_metrics)

    model_params = {
        "vocab_size": len(vocab),
        "weapon_vocab_size": len(weapon_vocab),
        **asdict(model_config),
        "pad_token_id": pad_token_id,
    }
    _write_json(output_dir / "model_params.json", model_params)

    _write_json(output_dir / "data_config.json", asdict(data_config))
    _write_json(output_dir / "run_config.json", asdict(run_config))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Basic training for SetCompletionModel."
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--vocab-path", type=str, required=True)
    parser.add_argument("--weapon-vocab-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--table-name", type=str)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--fraction", type=float, default=0.1)
    parser.add_argument("--random-state", type=int)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.2)

    parser.add_argument("--embedding-dim", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-inducing-points", type=int, default=32)
    parser.add_argument(
        "--use-layer-norm", dest="use_layer_norm", action="store_true"
    )
    parser.add_argument(
        "--no-layer-norm", dest="use_layer_norm", action="store_false"
    )
    parser.set_defaults(use_layer_norm=True)
    parser.add_argument("--dropout", type=float, default=0.0)

    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--scheduler-factor", type=float, default=0.1)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--use-mixed-precision", action="store_true")
    parser.add_argument("--num-masks-per-set", type=int, default=5)
    parser.add_argument("--skew-factor", type=float, default=1.2)
    parser.add_argument("--include-null", action="store_true")
    parser.add_argument("--metric-update-interval", type=int, default=1)
    parser.add_argument("--device", type=str)
    parser.add_argument(
        "--verbose", dest="verbose", action="store_true", default=True
    )
    parser.add_argument(
        "--quiet", dest="verbose", action="store_false"
    )

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data_config = DataConfig(
        data_path=args.data_path,
        vocab_path=args.vocab_path,
        weapon_vocab_path=args.weapon_vocab_path,
        output_dir=args.output_dir,
        table_name=args.table_name,
        max_rows=args.max_rows,
        fraction=args.fraction,
        random_state=args.random_state,
        validation_size=args.validation_size,
        test_size=args.test_size,
    )
    model_config = ModelConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        use_layer_norm=args.use_layer_norm,
        dropout=args.dropout,
    )
    run_config = RunConfig(
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        clip_grad_norm=args.clip_grad_norm,
        scheduler_factor=args.scheduler_factor,
        scheduler_patience=args.scheduler_patience,
        patience=args.patience,
        use_mixed_precision=args.use_mixed_precision,
        num_masks_per_set=args.num_masks_per_set,
        skew_factor=args.skew_factor,
        include_null=args.include_null,
        metric_update_interval=args.metric_update_interval,
        device=device,
        verbose=args.verbose,
    )

    run_basic_training(data_config, model_config, run_config)


if __name__ == "__main__":
    main()
