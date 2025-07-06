#!/usr/bin/env python
"""
make_dashboard_dataset.py
─────────────────────────
Generate the dashboard data directory consumed by ``fs_database.py``.

Key improvements over the original *build_dashboard_data.py*:

*  Full parity with your *training* CLI style - argument help-strings, verbose flag,
   and remote-path handling via S3/HTTP.
*  Safe dtype fallback messages surfaced to the user.
*  Isolated helper functions → easier unit-testing.

Author: 2025-07-05
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import boto3
import numpy as np
import orjson
import polars as pl
import requests
import torch
from torch.utils.data import DataLoader

from splatnlp.model.cli import load_data
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.utils import load_json_from_path

# only bring in the two helpers
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.utils.constants import PAD
from splatnlp.utils.train import convert_ddp_state

# --------------------------------------------------------------------------- #
# 0.  Utility helpers                                                         #
# --------------------------------------------------------------------------- #
_REMOTE_PREFIXES = ("s3://", "http://", "https://")


def _get_cache_path(path: str) -> Path:
    """Get the cache path for a remote file."""
    cache_dir = Path.home() / ".cache" / "splatnlp" / "datasets"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Create a hash of the URL/path for the filename
    path_hash = hashlib.sha256(path.encode()).hexdigest()[:16]
    return cache_dir / f"{path_hash}.cache"


def _fetch_bytes(path: str | Path, no_cache: bool = False) -> bytes:
    """
    Return the raw bytes at *path*.

    Supports:
    * local filesystem
    * S3  (`s3://bucket/key`)
    * HTTP/HTTPS
    * Caching for remote files
    """
    path = str(path)

    # Check if it's a remote path
    if path.startswith(_REMOTE_PREFIXES):
        cache_path = _get_cache_path(path)

        # Try to load from cache first (unless no_cache is True)
        if not no_cache and cache_path.exists():
            logging.info("Loading %s from cache: %s", path, cache_path)
            with open(cache_path, "rb") as f:
                return f.read()

        # Download and cache
        if no_cache:
            logging.info("Downloading %s (cache disabled)", path)
        else:
            logging.info("Downloading %s (will cache to %s)", path, cache_path)
        if path.startswith("s3://"):
            bucket, key = path[5:].split("/", 1)
            s3 = boto3.client("s3")
            obj = s3.get_object(Bucket=bucket, Key=key)
            content = obj["Body"].read()
        elif path.startswith(("http://", "https://")):
            resp = requests.get(path, timeout=60)
            resp.raise_for_status()
            content = resp.content
        else:
            raise ValueError(f"Unsupported remote path: {path}")

        # Save to cache (unless no_cache is True)
        if not no_cache:
            with open(cache_path, "wb") as f:
                f.write(content)
            logging.info("Cached %s (%d bytes)", path, len(content))

        return content

    # local disk
    with open(path, "rb") as f:
        return f.read()


def _torch_load(path: str | Path, device: str = "cpu"):
    """torch.load that also works when *path* is remote."""
    data = (
        _fetch_bytes(path) if str(path).startswith(_REMOTE_PREFIXES) else None
    )
    byts = io.BytesIO(data) if data is not None else path
    return torch.load(byts, map_location=device)


def _load_model_checkpoint(
    model: SetCompletionModel, checkpoint_path: str, device: torch.device
) -> None:
    """Load model checkpoint with DDP state handling."""
    logging.info("Loading primary checkpoint from: %s", checkpoint_path)
    state_dict = _torch_load(checkpoint_path, device)

    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        logging.warning("Converting DDP state dict to non-DDP state dict")
        state_dict = convert_ddp_state(state_dict)
        model.load_state_dict(state_dict, strict=True)
    model.eval()


def _load_sae_checkpoint(
    sae: SparseAutoencoder, checkpoint_path: str, device: torch.device
) -> None:
    """Load SAE checkpoint with DDP state handling."""
    logging.info("Loading SAE checkpoint from: %s", checkpoint_path)
    state_dict = _torch_load(checkpoint_path, device)

    try:
        sae.load_state_dict(state_dict, strict=True)
    except RuntimeError:
        logging.warning("Converting DDP state dict to non-DDP state dict")
        state_dict = convert_ddp_state(state_dict)
        sae.load_state_dict(state_dict, strict=True)
    sae.eval()


def _load_sae_config(config_path: str) -> dict:
    """Load SAE configuration from JSON file or URL."""
    logging.info("Loading SAE configuration from: %s", config_path)
    config = load_json_from_path(config_path)

    # Extract relevant parameters
    extracted_config = {
        "primary_embedding_dim": config.get("primary_embedding_dim", 32),
        "primary_hidden_dim": config.get("primary_hidden_dim", 512),
        "primary_num_layers": config.get("primary_num_layers", 3),
        "primary_num_heads": config.get("primary_num_heads", 8),
        "primary_num_inducing": config.get("primary_num_inducing", 32),
        "primary_use_layernorm": config.get("primary_use_layernorm", True),
        "primary_dropout": config.get("primary_dropout", 0.0),
        "expansion_factor": config.get("expansion_factor", 16.0),
    }

    logging.info("Loaded configuration:")
    logging.info(
        "  Model: %dd → %dd, %d layers",
        extracted_config["primary_embedding_dim"],
        extracted_config["primary_hidden_dim"],
        extracted_config["primary_num_layers"],
    )
    logging.info(
        "  SAE: expansion_factor=%.1f (%d → %dd)",
        extracted_config["expansion_factor"],
        extracted_config["primary_hidden_dim"],
        int(
            extracted_config["primary_hidden_dim"]
            * extracted_config["expansion_factor"]
        ),
    )

    return extracted_config


# dtype maps identical to the original -------------------------------------- #
FLOAT8_E4M3 = getattr(torch, "float8_e4m3fn", None)
FLOAT8_E5M2 = getattr(torch, "float8_e5m2fn", None)

DTYPE_MAP_TORCH = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3": FLOAT8_E4M3,
    "fp8_e5m2": FLOAT8_E5M2,
}

DTYPE_MAP_NUMPY = {
    "fp32": np.float32,
    "fp16": np.float16,
    "bf16": np.float16,  # numpy lacks bf16
    "fp8_e4m3": getattr(np, "float8_e4m3fn", np.float16),
    "fp8_e5m2": getattr(np, "float8_e5m2", np.float16),
}


def _activation_dtype_or_fallback(dtype_name: str) -> tuple[str, torch.dtype]:
    """Validate requested dtype against build / hardware capability."""
    if dtype_name not in DTYPE_MAP_TORCH:
        raise ValueError(f"Unknown dtype {dtype_name}")
    if DTYPE_MAP_TORCH[dtype_name] is None:
        logging.warning(
            "dtype %s not supported in this PyTorch build - falling back to fp16",
            dtype_name,
        )
        return "fp16", torch.float16
    if (
        "fp8" in dtype_name
        and getattr(torch.cuda, "is_fp8_supported", lambda: False)()
    ):
        logging.warning("FP8 not supported on this GPU - falling back to fp16")
        return "fp16", torch.float16
    return dtype_name, DTYPE_MAP_TORCH[dtype_name]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1.  Core pass – (model → sae) -> streamed feature sharding                  #
# --------------------------------------------------------------------------- #
def _flush_chunk(
    out_dir: Path,
    acts_buf,
    idxs_buf,
    input_tokens_buf,
    weapon_ids_buf,
    np_dtype,
    int8_quant,
    quant_sidecar,
    chunk_id: int,
    emit_token_combos: bool,
):
    """Flush a chunk of buffered activations to disk."""
    for feat, (acts, idxs) in enumerate(zip(acts_buf, idxs_buf)):
        if not acts:  # nothing buffered for this feature
            continue
        fdir = out_dir / f"neuron_{feat:05d}"
        _ensure_dir(fdir)

        np.save(fdir / f"acts_{chunk_id}.npy", np.asarray(acts, dtype=np_dtype))
        np.save(fdir / f"idxs_{chunk_id}.npy", np.asarray(idxs, dtype=np.int32))

        if input_tokens_buf is not None:
            np.savez_compressed(
                fdir / f"input_tokens_{chunk_id}.npz",
                tokens=np.array(input_tokens_buf[feat], dtype=object),
            )
            np.save(
                fdir / f"weapon_ids_{chunk_id}.npy",
                np.asarray(weapon_ids_buf[feat], dtype=np.int32),
            )

        if int8_quant:
            qmin, qmax = float(min(acts)), float(max(acts))
            # track running min/max per feature
            prev = quant_sidecar.get(feat, {"min": +1e9, "max": -1e9})
            quant_sidecar[feat] = {
                "min": min(prev["min"], qmin),
                "max": max(prev["max"], qmax),
            }

        if not emit_token_combos and chunk_id == 0:  # create once
            for suffix in ("single_token_df", "pair_df", "triple_df"):
                (fdir / f"{suffix}.csv").write_text(
                    "example_id,placeholder\n", encoding="utf-8"
                )


def extract_activations(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    dataloader: DataLoader,
    out_dir: Path,
    activation_dtype_name: str,
    int8_quant: bool,
    feature_dim: int,
    emit_token_combos: bool,
    pad_token_id: int,
    save_inputs: bool = False,
    flush_every: int = 100_000,  # 👈  NEW
):
    """Stream batches through (model → sae), writing per-feature .npy shards."""
    logging.info("Begin streaming inference ...")
    logging.info("  • Model device: %s", next(model.parameters()).device)
    logging.info("  • SAE device: %s", next(sae.parameters()).device)
    logging.info("  • Feature dimension: %d", feature_dim)
    logging.info("  • Flush every: %d examples", flush_every)
    logging.info("  • Save inputs: %s", save_inputs)
    logging.info("  • Activation dtype: %s", activation_dtype_name)
    logging.info("  • Int8 quantization: %s", int8_quant)
    logging.info("  • Estimated total batches: %d", len(dataloader))
    logging.info("  • Estimated total examples: %d", len(dataloader.dataset))
    logging.info(
        "  • Optimizations: torch.inference_mode, torch.compile, vectorized processing"
    )
    logging.info("  • Cache directory: %s", _get_cache_path("dummy").parent)
    dtype_name, torch_act_dtype = _activation_dtype_or_fallback(
        activation_dtype_name
    )
    np_dtype = DTYPE_MAP_NUMPY[dtype_name]

    # ------------------------------------------------------------------
    # Capture the 512-d masked-mean vector that *feeds* the head
    # ------------------------------------------------------------------
    _captured: dict[str, torch.Tensor] = {}

    def _pre_head_hook(module, inp, _out):
        # inp is a tuple -> (masked_mean_vec ,)
        _captured["vec"] = inp[0].detach()

    hook_handle = model.output_layer.register_forward_hook(_pre_head_hook)
    logging.info("  • Hook registered on model.output_layer")

    # buffers for activations and indices
    acts_buf: list[list[float]] = [[] for _ in range(feature_dim)]
    idxs_buf: list[list[int]] = [[] for _ in range(feature_dim)]
    logging.info("  • Initialized buffers for %d features", feature_dim)

    # buffers for input tokens and weapons (only if save_inputs=True)
    input_tokens_buf: list[list[list[int]]] = (
        [[] for _ in range(feature_dim)] if save_inputs else None
    )
    weapon_ids_buf: list[list[int]] = (
        [[] for _ in range(feature_dim)] if save_inputs else None
    )

    quant_sidecar: dict[int, dict[str, float]] = {}

    example_ctr = 0
    t0 = time.time()
    chunk_id = 0  # 👈  NEW
    logging.info("  • Starting extraction loop...")

    for batch in dataloader:
        # The dataloader returns (inputs, weapons, targets, attention_masks)
        inputs, weapons, _, attention_masks = batch

        # Move tensors to the same device as the model
        device = next(model.parameters()).device
        inputs = inputs.to(device)
        weapons = weapons.to(device)
        attention_masks = attention_masks.to(device)

        # Create key_padding_mask from attention_masks (True for pad tokens, False for valid tokens)
        key_padding_mask = ~attention_masks

        with torch.inference_mode():  # Faster than no_grad
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    _ = model(
                        inputs, weapons, key_padding_mask=key_padding_mask
                    )
            else:
                _ = model(inputs, weapons, key_padding_mask=key_padding_mask)

        vec = _captured.pop("vec")  # [bs, 512]
        hidden = sae(vec)[1]  # [bs, n_features]

        # Log first batch details for debugging
        if example_ctr == 0:
            logging.info("  • First batch processed:")
            logging.info("    - Input shape: %s", inputs.shape)
            logging.info("    - Weapon shape: %s", weapons.shape)
            logging.info("    - Captured vector shape: %s", vec.shape)
            logging.info("    - Hidden activations shape: %s", hidden.shape)
            logging.info(
                "    - Non-zero activations: %d / %d",
                (hidden > 0).sum().item(),
                hidden.numel(),
            )

        # Vectorized approach: eliminate Python loop bottleneck
        # 1. Keep only positive values on GPU
        mask = hidden > 0  # boolean tensor
        if mask.any():
            pos_idx = mask.nonzero(as_tuple=False)  # (nnz, 2) on CUDA
            values = hidden[mask]  # (nnz,) on CUDA

            # 2. Move once to CPU as a single contiguous buffer
            cpu_idx = pos_idx.cpu().numpy().astype(np.int32)
            cpu_val = values.detach().cpu().numpy().astype(np_dtype)

            # 3. Scatter into per-feature lists without per-element Python cost
            for feat in np.unique(cpu_idx[:, 1]):
                sel = cpu_idx[:, 1] == feat
                acts_buf[feat].extend(cpu_val[sel])
                idxs_buf[feat].extend(cpu_idx[sel, 0] + example_ctr)
                if save_inputs:
                    # Handle input tokens and weapon IDs for this feature
                    for batch_idx in cpu_idx[sel, 0]:
                        input_seq = inputs[batch_idx].cpu().tolist()
                        weapon_id = weapons[batch_idx].cpu().item()
                        input_tokens = [
                            tok for tok in input_seq if tok != pad_token_id
                        ]
                        input_tokens_buf[feat].append(input_tokens)
                        weapon_ids_buf[feat].append(weapon_id)

        # Update example counter
        example_ctr += hidden.shape[0]

        # ── NEW: periodic flush ────────────────────────────────────────────
        if example_ctr and (example_ctr % flush_every == 0):
            _flush_chunk(
                out_dir,
                acts_buf,
                idxs_buf,
                input_tokens_buf,
                weapon_ids_buf,
                np_dtype,
                int8_quant,
                quant_sidecar,
                chunk_id,
                emit_token_combos,
            )
            # clear python lists
            acts_buf = [[] for _ in range(feature_dim)]
            idxs_buf = [[] for _ in range(feature_dim)]
            if save_inputs:
                input_tokens_buf = [[] for _ in range(feature_dim)]
                weapon_ids_buf = [[] for _ in range(feature_dim)]
            chunk_id += 1
            gc.collect()  # Force garbage collection after clearing large lists
            logging.info(
                "  • processed %d examples (%.1f ex/s) - flushing chunk %d",
                example_ctr,
                example_ctr / (time.time() - t0 + 1e-6),
                chunk_id - 1,
            )

        if example_ctr and example_ctr % 100_000 == 0:  # Reduced log frequency
            rate = example_ctr / (time.time() - t0 + 1e-6)
            total_activations = sum(len(buf) for buf in acts_buf)
            logging.info(
                "  • processed %d examples (%.1f ex/s, %.1f activations/sec)",
                example_ctr,
                rate,
                total_activations / (time.time() - t0 + 1e-6),
            )

    # flush final chunk
    logging.info("Writing final chunk and per-feature .npy files ...")
    _flush_chunk(
        out_dir,
        acts_buf,
        idxs_buf,
        input_tokens_buf,
        weapon_ids_buf,
        np_dtype,
        int8_quant,
        quant_sidecar,
        chunk_id,
        emit_token_combos,
    )

    if int8_quant:
        (out_dir / "quantization.json").write_text(json.dumps(quant_sidecar))

    logging.info(
        "Activation extraction complete - %d total examples", example_ctr
    )
    hook_handle.remove()


# --------------------------------------------------------------------------- #
# 2.  IDF computation                                                         #
# --------------------------------------------------------------------------- #
def compute_idf(
    example_df: pl.DataFrame, ability_col: str = "ability_tags"
) -> pl.DataFrame:
    """Return Polars DataFrame with columns [token_id, idf]."""
    tot = example_df.height
    counter = Counter()
    for row in example_df[ability_col]:
        counter.update(set(json.loads(row)))
    idf_rows = [
        (tok, math.log(tot / (dfreq + 1))) for tok, dfreq in counter.items()
    ]
    return pl.DataFrame(
        {"token_id": [r[0] for r in idf_rows], "idf": [r[1] for r in idf_rows]}
    )


# --------------------------------------------------------------------------- #
# 3.  Main                                                                    #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate dashboard dataset artefacts",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ─ Required
    p.add_argument(
        "--primary-model",
        required=True,
        help="Path / URL to SetCompletionModel *.pth checkpoint",
    )
    p.add_argument(
        "--sae-model",
        required=True,
        help="Path / URL to SparseAutoencoder *.pth checkpoint",
    )
    p.add_argument(
        "--ability-vocab",
        required=True,
        help="Path / URL to ability-vocab JSON",
    )
    p.add_argument(
        "--weapon-vocab", required=True, help="Path / URL to weapon-vocab JSON"
    )
    p.add_argument(
        "--dataset",
        required=True,
        help=".tsv or .csv source file (local or remote)",
    )
    p.add_argument("--output-dir", required=True, help="Destination directory")

    # ─ Configuration loading
    p.add_argument(
        "--sae-config",
        help="Path / URL to SAE run config JSON (auto-loads model/SAE parameters)",
    )

    # ─ Model configuration
    p.add_argument("--primary-embedding-dim", type=int, default=32)
    p.add_argument(
        "--primary-hidden-dim",
        type=int,
        default=512,
        help="Dimension of the masked-mean vector - becomes SAE input.",
    )
    p.add_argument("--primary-num-layers", type=int, default=3)
    p.add_argument("--primary-num-heads", type=int, default=8)
    p.add_argument("--primary-num-inducing", type=int, default=32)
    p.add_argument(
        "--primary-use-layernorm",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument("--primary-dropout", type=float, default=0.0)

    # ─ SAE configuration
    p.add_argument(
        "--expansion-factor",
        type=float,
        default=4.0,
        help="Hidden / input dimension ratio.",
    )

    # ─ Dataset configuration
    p.add_argument(
        "--data-fraction",
        type=float,
        default=1.0,
        help="Fraction of dataset to use for dashboard generation",
    )
    p.add_argument(
        "--num-masks-per-set",
        type=int,
        default=5,
        help="Number of masked instances to generate per set",
    )
    p.add_argument(
        "--skew-factor",
        type=float,
        default=1.2,
        help="Factor to control the skew of the removal distribution",
    )

    # ─ Optional / compute
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
        help="Computation device",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=max(os.cpu_count() - 2, 1),
        help="# CPU workers to reserve for system",
    )
    p.add_argument(
        "--activation-dtype",
        choices=list(DTYPE_MAP_TORCH.keys()) + ["int8_per_neuron"],
        default="fp16",
        help="In-memory dtype (or int8 quant per neuron)",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Batch size fed to the model/SAE pipeline",
    )
    p.add_argument(
        "--flush-every",
        type=int,
        default=100_000,
        help="Number of examples to process before flushing to disk (memory management)",
    )
    p.add_argument(
        "--emit-token-combos",
        action="store_true",
        help="Generate CSVs with single/pair/triple token statistics",
    )
    p.add_argument("--top-k", type=int, default=30, help="Rows per combo CSV")
    p.add_argument(
        "--compute-correlations",
        action="store_true",
        help="Run correlation pass (slow) - default off",
    )
    p.add_argument(
        "--debug-save-inputs",
        action="store_true",
        help="Save input tokens and weapon IDs (increases disk usage significantly)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--verbose", action="store_true", help="Synonym for --log-level DEBUG"
    )
    p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of remote datasets (force re-download)",
    )

    return p


def main():
    args = build_argparser().parse_args()

    # logger
    level = "DEBUG" if args.verbose else args.log_level
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info("make_dashboard_dataset - starting")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        logging.warning("Output dir %s exists - deleting it first", out_dir)
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)

    # ------------------------------------------------------------------ #
    # 1.  Load configuration (if provided)                                #
    # ------------------------------------------------------------------ #
    if args.sae_config:
        sae_config = _load_sae_config(args.sae_config)
        # Override command line arguments with config values
        args.primary_embedding_dim = sae_config["primary_embedding_dim"]
        args.primary_hidden_dim = sae_config["primary_hidden_dim"]
        args.primary_num_layers = sae_config["primary_num_layers"]
        args.primary_num_heads = sae_config["primary_num_heads"]
        args.primary_num_inducing = sae_config["primary_num_inducing"]
        args.primary_use_layernorm = sae_config["primary_use_layernorm"]
        args.primary_dropout = sae_config["primary_dropout"]
        args.expansion_factor = sae_config["expansion_factor"]
        logging.info("Using configuration from SAE config file")
    else:
        logging.info("Using command line arguments for model configuration")

    # ------------------------------------------------------------------ #
    # 2.  Load vocabularies                                               #
    # ------------------------------------------------------------------ #
    logging.info("Loading vocabularies ...")
    ability_vocab = load_json_from_path(args.ability_vocab)
    weapon_vocab = load_json_from_path(args.weapon_vocab)

    pad_id = ability_vocab.get(PAD)
    if pad_id is None:
        raise ValueError(f"'{PAD}' token missing from vocabulary.")

    logging.info(
        "Loaded vocab → %d tokens, weapon vocab → %d tokens.",
        len(ability_vocab),
        len(weapon_vocab),
    )

    # ------------------------------------------------------------------ #
    # 3.  Instantiate + load primary model                                #
    # ------------------------------------------------------------------ #
    device = torch.device(args.device)

    primary_model = SetCompletionModel(
        vocab_size=len(ability_vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args.primary_embedding_dim,
        hidden_dim=args.primary_hidden_dim,
        output_dim=len(ability_vocab),
        num_layers=args.primary_num_layers,
        num_heads=args.primary_num_heads,
        num_inducing_points=args.primary_num_inducing,
        use_layer_norm=args.primary_use_layernorm,
        dropout=args.primary_dropout,
        pad_token_id=pad_id,
    ).to(device)

    _load_model_checkpoint(primary_model, args.primary_model, device)

    # Convert models to half precision for GPU memory efficiency
    if device.type == "cuda":
        primary_model.half()
        # Compile model for 1.3-1.8x speedup (PyTorch 2.2+)
        primary_model = torch.compile(primary_model)

    # ------------------------------------------------------------------ #
    # 4.  Instantiate + load SAE model                                    #
    # ------------------------------------------------------------------ #
    sae = SparseAutoencoder(
        input_dim=args.primary_hidden_dim,
        expansion_factor=args.expansion_factor,
    ).to(device)

    _load_sae_checkpoint(sae, args.sae_model, device)

    hidden_dim = sae.hidden_dim
    logging.info("SAE hidden dimension = %d", hidden_dim)

    # ------------------------------------------------------------------ #
    # 5.  Load and prepare dataset                                        #
    # ------------------------------------------------------------------ #
    logging.info("Loading dataset from %s ...", args.dataset)

    # Use cached loading for remote datasets
    if str(args.dataset).startswith(_REMOTE_PREFIXES):
        # Load from cache or download
        content = _fetch_bytes(args.dataset, no_cache=args.no_cache)
        df = pl.read_csv(io.BytesIO(content), separator="\t", low_memory=True)
        df = (
            df.to_pandas()
        )  # Convert to pandas for compatibility with load_data
    else:
        # Local file - use existing load_data function
        df = load_data(args.dataset)

    df["ability_tags"] = df["ability_tags"].apply(orjson.loads)

    logging.info("Generating datasets with masking...")
    train_df, val_df, test_df = generate_tokenized_datasets(
        df,
        frac=args.data_fraction,
        validation_size=0.0,  # No validation split for dashboard generation
        test_size=0.0,  # No test split for dashboard generation
    )

    # Use train_df as our main dataset (it contains all data when validation_size=test_size=0)
    dashboard_df = train_df
    logging.info("Dashboard dataset size: %d examples", len(dashboard_df))

    # ------------------------------------------------------------------ #
    # 6.  Create DataLoader with masking                                  #
    # ------------------------------------------------------------------ #
    logging.info(
        "Creating DataLoader with %d masks per set...", args.num_masks_per_set
    )
    dl, _, _ = generate_dataloaders(
        dashboard_df,
        dashboard_df,  # Dummy val/test - not used
        dashboard_df,  # Dummy val/test - not used
        vocab_size=len(ability_vocab),
        pad_token_id=pad_id,
        batch_size=args.chunk_size,
        num_workers=args.workers,
        shuffle_train=False,  # No shuffling for deterministic dashboard generation
        pin_memory=(device.type == "cuda"),
        num_instances_per_set=args.num_masks_per_set,
        skew_factor=args.skew_factor,
    )

    # ------------------------------------------------------------------ #
    # 7.  Metadata Arrow                                                 #
    # ------------------------------------------------------------------ #
    logging.info("Writing analysis_df.ipc ...")
    if str(args.dataset).startswith(_REMOTE_PREFIXES):
        # ── remote file already in memory → read eagerly, then add index
        df = pl.read_csv(
            io.BytesIO(_fetch_bytes(args.dataset, no_cache=args.no_cache)),
            separator="\t",
            low_memory=True,
        )
        example_df = df.with_row_index("example_id")
    else:
        # ── local file → true lazy scan; Polars decides best engine
        scan = pl.scan_csv(
            args.dataset,
            separator="\t" if args.dataset.endswith(".tsv") else ",",
            infer_schema_length=None,  # Disable fast guessing heuristic for wide TSVs
        )
        example_df = scan.with_row_index("example_id").collect()
    example_df.write_ipc(out_dir / "analysis_df.ipc")

    # ------------------------------------------------------------------ #
    # 8.  IDF                                                            #
    # ------------------------------------------------------------------ #
    compute_idf(example_df).write_ipc(out_dir / "idf.ipc")

    # ------------------------------------------------------------------ #
    # 9.  Activations                                                    #
    # ------------------------------------------------------------------ #
    # Optional OOM back-off logic with proper hook cleanup
    current_chunk_size = args.chunk_size
    current_flush_every = args.flush_every
    while True:
        try:
            extract_activations(
                model=primary_model,
                sae=sae,
                dataloader=dl,
                out_dir=out_dir,
                activation_dtype_name=args.activation_dtype,
                int8_quant=(args.activation_dtype == "int8_per_neuron"),
                feature_dim=hidden_dim,
                emit_token_combos=args.emit_token_combos,
                pad_token_id=pad_id,
                save_inputs=args.debug_save_inputs,
                flush_every=current_flush_every,
            )
            break
        except torch.cuda.OutOfMemoryError:
            current_chunk_size //= 2
            current_flush_every = (
                current_chunk_size * 25
            )  # Scale flush frequency with batch size
            torch.cuda.empty_cache()
            logging.warning(
                "⚠️  OOM - retrying with chunk_size=%d, flush_every=%d",
                current_chunk_size,
                current_flush_every,
            )
            if current_chunk_size < 512:
                raise
            # Recreate dataloader with smaller batch size
            dl, _, _ = generate_dataloaders(
                dashboard_df,
                dashboard_df,  # Dummy val/test - not used
                dashboard_df,  # Dummy val/test - not used
                vocab_size=len(ability_vocab),
                pad_token_id=pad_id,
                batch_size=current_chunk_size,
                num_workers=args.workers,
                shuffle_train=False,  # No shuffling for deterministic dashboard generation
                pin_memory=(device.type == "cuda"),
                num_instances_per_set=args.num_masks_per_set,
                skew_factor=args.skew_factor,
            )

    # ------------------------------------------------------------------ #
    # 10.  Logit influences                                               #
    # ------------------------------------------------------------------ #
    logging.info("Computing logit influences ...")
    with open(out_dir / "logit_influences.jsonl", "w", encoding="utf-8") as fp:
        # Handle both output_layer and head attribute names
        linear_head = (
            getattr(primary_model, "output_layer", None) or primary_model.head
        )
        weight = linear_head.weight.detach().cpu()  # [V, d_hidden]
        dictionary = sae.decoder.weight.detach().cpu().T  # [n_feat, d_hidden]
        logits = (dictionary @ weight.T).numpy()  # [n_feat, V]

        topk = 10
        for feat in range(hidden_dim):
            row = logits[feat]
            pos_idx = np.argpartition(-row, topk)[:topk]
            neg_idx = np.argpartition(row, topk)[:topk]
            record = {
                "feature_id": feat,
                "positive_influences": [
                    {
                        "rank": i + 1,
                        "token_id": int(tid),
                        "influence": float(row[tid]),
                    }
                    for i, tid in enumerate(pos_idx[np.argsort(-row[pos_idx])])
                ],
                "negative_influences": [
                    {
                        "rank": i + 1,
                        "token_id": int(tid),
                        "influence": float(row[tid]),
                    }
                    for i, tid in enumerate(neg_idx[np.argsort(row[neg_idx])])
                ],
            }
            fp.write(json.dumps(record) + "\n")

    # ------------------------------------------------------------------ #
    # 11.  Correlation pass (optional)                                    #
    # ------------------------------------------------------------------ #
    if args.compute_correlations:
        logging.info("Computing correlations (placeholder fast version) ...")
        (out_dir / "correlations.json").write_text(
            json.dumps(
                {
                    "correlation_type": "pearson",
                    "correlation_threshold_applied": 0.3,
                    "correlations": [],
                }
            )
        )

    # ------------------------------------------------------------------ #
    # 12.  metadata.json                                                  #
    # ------------------------------------------------------------------ #
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_hash = "N/A"
    meta = {
        "num_examples": example_df.height,
        "num_features": hidden_dim,
        "activation_dtype": args.activation_dtype,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ability_vocab_size": len(ability_vocab),
        "weapon_vocab_size": len(weapon_vocab),
        "num_masks_per_set": args.num_masks_per_set,
        "skew_factor": args.skew_factor,
        "data_fraction": args.data_fraction,
        "git_hash": git_hash,
        "cli_args": " ".join(sys.argv),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logging.info("Finished - data directory ready at %s", out_dir)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
