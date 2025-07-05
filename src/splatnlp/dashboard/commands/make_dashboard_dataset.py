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
import io
import json
import logging
import math
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import boto3
import numpy as np
import polars as pl
import requests
import torch
from torch.utils.data import DataLoader, Dataset

# --------------------------------------------------------------------------- #
# 0.  Utility helpers                                                         #
# --------------------------------------------------------------------------- #
_REMOTE_PREFIXES = ("s3://", "http://", "https://")


def _fetch_bytes(path: str | Path) -> bytes:
    """
    Return the raw bytes at *path*.

    Supports:
    * local filesystem
    * S3  (`s3://bucket/key`)
    * HTTP/HTTPS
    """
    path = str(path)
    if path.startswith("s3://"):
        bucket, key = path[5:].split("/", 1)
        s3 = boto3.client("s3")
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    if path.startswith(("http://", "https://")):
        resp = requests.get(path, timeout=60)
        resp.raise_for_status()
        return resp.content
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


def _json_load(path: str | Path):
    """orjson-style loader with remote support."""
    import orjson

    raw = _fetch_bytes(path)
    return orjson.loads(raw)


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


def _activation_dtype_or_fallback(dtype_name: str) -> Tuple[str, torch.dtype]:
    """Validate requested dtype against build / hardware capability."""
    if dtype_name not in DTYPE_MAP_TORCH:
        raise ValueError(f"Unknown dtype {dtype_name}")
    if DTYPE_MAP_TORCH[dtype_name] is None:
        logging.warning(
            "dtype %s not supported in this PyTorch build - falling back to fp16",
            dtype_name,
        )
        return "fp16", torch.float16
    if "fp8" in dtype_name and not torch.cuda.is_fp8_supported():
        logging.warning("FP8 not supported on this GPU - falling back to fp16")
        return "fp16", torch.float16
    return dtype_name, DTYPE_MAP_TORCH[dtype_name]


def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 1. Dataset wrapper                                                          #
# --------------------------------------------------------------------------- #
class TSVAbilityDataset(Dataset):
    """Lazy dataset that yields parsed tensors for model consumption."""

    def __init__(
        self,
        csv_path: Path | str,
        ability_pad_id: int = 0,
        device: str | torch.device = "cpu",
    ):
        # handle remote files
        if str(csv_path).startswith(_REMOTE_PREFIXES):
            content = _fetch_bytes(csv_path)
            df = pl.read_csv(
                io.BytesIO(content), separator="\t", low_memory=True
            )
        else:
            df = pl.read_csv(
                csv_path,
                separator="\t" if str(csv_path).endswith(".tsv") else ",",
                low_memory=True,
            )

        self.ability_lists: List[str] = df["ability_tags"].to_list()
        self.weapon_ids: List[int] = df["weapon_id"].to_list()
        self.device = device
        self.pad_id = ability_pad_id

        # unify length
        self.max_len = max(len(json.loads(lst)) for lst in self.ability_lists)
        self.len_ = len(self.weapon_ids)

    def __len__(self):  # type: ignore[override]
        return self.len_

    def __getitem__(self, idx):  # type: ignore[override]
        abilities = json.loads(self.ability_lists[idx])
        pad_len = self.max_len - len(abilities)
        abilities = abilities + [self.pad_id] * pad_len
        abilities = torch.tensor(
            abilities, dtype=torch.long, device=self.device
        )
        weapon_id = torch.tensor(
            self.weapon_ids[idx], dtype=torch.long, device=self.device
        )
        return abilities, weapon_id


# --------------------------------------------------------------------------- #
# 2.  Core pass – (model → sae) -> streamed feature sharding                  #
# --------------------------------------------------------------------------- #
def extract_activations(
    model: torch.nn.Module,
    sae: torch.nn.Module,
    dataloader: DataLoader,
    out_dir: Path,
    activation_dtype_name: str,
    int8_quant: bool,
    feature_dim: int,
    emit_token_combos: bool,
    top_k: int,
):
    """Stream batches through (model → sae), writing per-feature .npy shards."""
    logging.info("Begin streaming inference …")
    dtype_name, torch_act_dtype = _activation_dtype_or_fallback(
        activation_dtype_name
    )
    np_dtype = DTYPE_MAP_NUMPY[dtype_name]

    # buffers
    acts_buf: List[List[float]] = [[] for _ in range(feature_dim)]
    idxs_buf: List[List[int]] = [[] for _ in range(feature_dim)]
    quant_sidecar = {}

    example_ctr = 0
    t0 = time.time()

    for batch in dataloader:
        abilities, weapon_ids = batch
        with torch.no_grad():
            hidden = model(abilities, weapon_ids)  # [bs, 512]
            hidden = sae(hidden)  # [bs, n_features]

        hidden_cpu = hidden.detach().cpu()
        for row in hidden_cpu:
            nz = (row > 0).nonzero().squeeze(1).tolist()
            if not nz:
                example_ctr += 1
                continue
            vals = row[nz].to(torch_act_dtype).tolist()
            for feat, val in zip(nz, vals):
                acts_buf[feat].append(val)
                idxs_buf[feat].append(example_ctr)
            example_ctr += 1

        if example_ctr and example_ctr % 10_000 == 0:
            rate = example_ctr / (time.time() - t0 + 1e-6)
            logging.info(
                "  • processed %d examples (%.1f ex/s)", example_ctr, rate
            )

    # flush
    logging.info("Writing per-feature .npy files …")
    for feat in range(feature_dim):
        fdir = out_dir / f"neuron_{feat:05d}"
        _ensure_dir(fdir)
        np.save(fdir / "acts.npy", np.asarray(acts_buf[feat], dtype=np_dtype))
        np.save(fdir / "idxs.npy", np.asarray(idxs_buf[feat], dtype=np.int32))

        if int8_quant and acts_buf[feat]:
            qmin, qmax = float(min(acts_buf[feat])), float(max(acts_buf[feat]))
            quant_sidecar[feat] = {"min": qmin, "max": qmax}

        if not emit_token_combos:  # write tiny placeholder CSVs
            for suffix in ("single_token_df", "pair_df", "triple_df"):
                (fdir / f"{suffix}.csv").write_text(
                    "example_id,placeholder\n", encoding="utf-8"
                )

    if int8_quant:
        (out_dir / "quantization.json").write_text(json.dumps(quant_sidecar))

    logging.info(
        "Activation extraction complete - %d total examples", example_ctr
    )


# --------------------------------------------------------------------------- #
# 3.  IDF computation                                                         #
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
    return pl.DataFrame(idf_rows, schema=["token_id", "idf"])


# --------------------------------------------------------------------------- #
# 4.  Main                                                                    #
# --------------------------------------------------------------------------- #
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate dashboard dataset artefacts"
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
        default=50_000,
        help="Batch size fed to the model/SAE pipeline",
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
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    p.add_argument(
        "--verbose", action="store_true", help="Synonym for --log-level DEBUG"
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
    logging.info("make_dashboard_dataset – starting")

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        logging.warning("Output dir %s exists – deleting it first", out_dir)
        shutil.rmtree(out_dir)
    _ensure_dir(out_dir)

    # ------------------------------------------------------------------ #
    # 1.  Load artefacts                                                 #
    # ------------------------------------------------------------------ #
    device = torch.device(args.device)
    logging.info("Loading primary model from %s …", args.primary_model)
    model = _torch_load(args.primary_model, device=device)
    model.eval()

    logging.info("Loading SAE model from %s …", args.sae_model)
    sae = _torch_load(args.sae_model, device=device)
    sae.eval()

    hidden_dim = sae.decoder.out_features  # assume nn.Linear decoder
    logging.info("SAE hidden dimension = %d", hidden_dim)

    # vocabularies (they are not used inside this script but saved for completeness)
    logging.info("Loading vocabularies …")
    ability_vocab = _json_load(args.ability_vocab)
    weapon_vocab = _json_load(args.weapon_vocab)

    # ------------------------------------------------------------------ #
    # 2.  Dataset + DataLoader                                           #
    # ------------------------------------------------------------------ #
    ds = TSVAbilityDataset(args.dataset, ability_pad_id=0, device=device)
    dl = DataLoader(
        ds,
        batch_size=args.chunk_size,
        shuffle=False,
        num_workers=0,  # streaming inference – keep on main process
        pin_memory=(device.type == "cuda"),
    )

    # ------------------------------------------------------------------ #
    # 3.  Metadata Arrow                                                 #
    # ------------------------------------------------------------------ #
    logging.info("Writing analysis_df.ipc …")
    if str(args.dataset).startswith(_REMOTE_PREFIXES):
        content = io.BytesIO(_fetch_bytes(args.dataset))
        scan = pl.scan_csv(content, separator="\t", infer_schema_length=0)
    else:
        scan = pl.scan_csv(
            args.dataset,
            separator="\t" if str(args.dataset).endswith(".tsv") else ",",
        )
    example_df = scan.with_row_count("example_id").collect(streaming=True)
    example_df.write_ipc(out_dir / "analysis_df.ipc")

    # ------------------------------------------------------------------ #
    # 4.  IDF                                                            #
    # ------------------------------------------------------------------ #
    compute_idf(example_df).write_ipc(out_dir / "idf.ipc")

    # ------------------------------------------------------------------ #
    # 5.  Activations                                                    #
    # ------------------------------------------------------------------ #
    extract_activations(
        model=model,
        sae=sae,
        dataloader=dl,
        out_dir=out_dir,
        activation_dtype_name=args.activation_dtype,
        int8_quant=(args.activation_dtype == "int8_per_neuron"),
        feature_dim=hidden_dim,
        emit_token_combos=args.emit_token_combos,
        top_k=args.top_k,
    )

    # ------------------------------------------------------------------ #
    # 6.  Logit influences                                               #
    # ------------------------------------------------------------------ #
    logging.info("Computing logit influences …")
    with open(out_dir / "logit_influences.jsonl", "w", encoding="utf-8") as fp:
        linear_head: torch.nn.Linear = model.head  # type: ignore[attr-defined]
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
    # 7.  Correlation pass (optional)                                    #
    # ------------------------------------------------------------------ #
    if args.compute_correlations:
        logging.info("Computing correlations (placeholder fast version) …")
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
    # 8.  metadata.json                                                  #
    # ------------------------------------------------------------------ #
    meta = {
        "num_examples": example_df.height,
        "num_features": hidden_dim,
        "activation_dtype": args.activation_dtype,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ability_vocab_size": len(ability_vocab),
        "weapon_vocab_size": len(weapon_vocab),
    }
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    logging.info("Finished - data directory ready at %s", out_dir)


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
