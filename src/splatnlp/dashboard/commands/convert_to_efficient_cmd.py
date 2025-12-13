#!/usr/bin/env python3
"""
Convert joblib batch files to efficient Parquet + Zarr v3 format.

This module provides a ``BatchConverter`` class which can ingest batches of
serialized objects produced with ``joblib`` and persist them to a combination
of Parquet files (for tabular metadata) and Zarr v3 arrays (for dense
numerical data).  Compared to the legacy implementation which targeted
Zarr v2, this version adopts the modern v3 API: compressors are supplied via
the ``compressors`` keyword argument, codecs are imported from
``zarr.codecs`` or ``numcodecs.zarr3`` as appropriate, and array resizing
and appending make use of the built‑in ``append()`` convenience method.

The converter reads each input joblib file, extracts tabular data and
high‑dimensional arrays, writes the tabular portion to Parquet using
``polars``, and writes the numerical arrays into per‑batch Zarr stores with
appropriate chunking and compression.  Embedded arrays such as model inputs
and reconstructions are appended to global Zarr stores, avoiding a costly
resize and copy operation for each batch.  Metadata about the conversion is
recorded as JSON for later inspection.
"""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import polars as pl
import zarr
from tqdm import tqdm

try:
    # Import additional codecs from numcodecs if available (e.g. LZ4)
    from numcodecs.zarr3 import LZ4  # type: ignore
except Exception:
    LZ4 = None  # Fallback if LZ4 is not present


class BatchConverter:
    """Convert joblib batches to Parquet + Zarr v3 format."""

    def __init__(
        self,
        source_dir: Path,
        target_dir: Path,
        chunk_size: int = 1000,
        compression: str = "zstd",
        compression_level: int = 3,
    ) -> None:
        """
        Initialise the converter.

        Parameters
        ----------
        source_dir:
            Directory containing ``*.joblib`` files to be converted.
        target_dir:
            Directory into which Parquet files and Zarr stores will be written.
        chunk_size:
            The number of rows per chunk along the first axis when writing
            Zarr arrays.
        compression:
            A symbolic name selecting the compression algorithm for Zarr
            arrays.  Supported values include ``"zstd"``, ``"blosc"``,
            ``"lz4"`` (via ``numcodecs``) and ``"gzip"``.
        compression_level:
            A positive integer controlling the compression level passed to
            the codec.  Higher values generally yield smaller files at the
            expense of CPU time.
        """
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.chunk_size = chunk_size
        self.compression = compression.lower() if compression else "zstd"
        self.compression_level = compression_level

        # Create output directories
        self.metadata_dir = self.target_dir / "metadata"
        self.activations_dir = self.target_dir / "activations"
        self.embeddings_dir = self.target_dir / "embeddings"

        for dir_path in [
            self.metadata_dir,
            self.activations_dir,
            self.embeddings_dir,
        ]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Discover input files
        self.batch_files = sorted(self.source_dir.glob("*.joblib"))
        if not self.batch_files:
            raise ValueError(f"No joblib files found in {source_dir}")

    # ------------------------------------------------------------------
    # Internal helpers
    def _get_compressor(self) -> Optional[Any]:
        """Return an appropriate Zarr v3 codec instance based on settings.

        The Zarr v3 API expects compressors to be provided via the
        ``compressors`` keyword argument rather than the legacy ``compressor``
        argument.  Each supported algorithm is mapped onto an instance of
        ``zarr.codecs.*Codec`` or ``numcodecs.zarr3`` where required.  If
        compression is disabled or an unrecognised value is supplied, this
        method returns ``None``.
        """
        c = self.compression
        lvl = self.compression_level
        if c in {"blosc", "zstd"}:
            # The Blosc wrapper can use the zstd codec internally.  For
            # consistency with the old behaviour, we default to bit‑shuffle
            # filtering when using Blosc.
            if c == "blosc":
                return zarr.codecs.BloscCodec(
                    cname="zstd",
                    clevel=lvl,
                    shuffle=zarr.codecs.BloscShuffle.bitshuffle,
                )
            # For pure Zstandard compression use the dedicated codec
            return zarr.codecs.ZstdCodec(level=lvl)
        if c == "gzip":
            return zarr.codecs.GzipCodec(level=lvl)
        if c == "lz4":
            # Attempt to use the dedicated LZ4 codec provided by numcodecs.zarr3
            if LZ4 is not None:
                # ``acceleration`` is inversely related to compression level: higher
                # acceleration trades off ratio for speed.  We map level 1–9 to
                # acceleration values to mirror the intuitive meaning of
                # ``compression_level``.
                try:
                    return LZ4(acceleration=max(1, lvl))  # type: ignore
                except Exception:
                    pass
            # Fallback: use Blosc with LZ4 as the underlying algorithm
            return zarr.codecs.BloscCodec(cname="lz4", clevel=lvl)
        # Disable compression
        return None

    # ------------------------------------------------------------------
    def _create_zarr_array(
        self,
        path: Path,
        shape: tuple[int, ...],
        dtype: np.dtype,
        chunks: tuple[int, ...],
    ) -> zarr.core.array.Array:
        """Create a new persistent Zarr v3 array with the configured compressor.

        This helper wraps ``zarr.create_array`` and ensures that the new array
        uses the v3 format, appropriate chunking and optional compression.  If
        ``path`` already exists, any previous contents will be silently
        overwritten.
        """
        compressor = self._get_compressor()
        # Convert Path to string: zarr expects a string path for stores
        return zarr.create_array(
            store=str(path),
            shape=shape,
            dtype=dtype,
            chunks=chunks,
            compressors=compressor,
            overwrite=True,
        )

    # ------------------------------------------------------------------
    def convert_batch(
        self, batch_file: Path, batch_index: int
    ) -> Dict[str, Any]:
        """Convert a single joblib batch into Parquet and Zarr outputs.

        Parameters
        ----------
        batch_file:
            The path to a ``*.joblib`` file containing the batch data.
        batch_index:
            The numeric index of the batch, used to generate output filenames
            and to assign a unique global sample index.

        Returns
        -------
        dict
            A dictionary containing statistics about the conversion, such as
            input size, number of samples, and compression ratio.
        """
        print(f"\nConverting {batch_file.name}…")
        stats: Dict[str, Any] = {}

        # Load joblib data
        print("  Loading joblib…")
        batch_data: Dict[str, Any] = joblib.load(batch_file)
        df: pd.DataFrame = batch_data["analysis_df_records"]
        activations: np.ndarray = batch_data["all_sae_hidden_activations"]
        metadata: Dict[str, Any] = batch_data.get("metadata", {})

        stats["input_size_mb"] = batch_file.stat().st_size / (1024**2)
        stats["n_samples"] = len(df)
        stats["n_features"] = activations.shape[1]

        # ------------------------------------------------------------------
        # Step 1: Convert DataFrame to Parquet
        print("  Converting DataFrame to Parquet…")

        # Extract simple columns (lists and scalars) from the DataFrame; multi
        # dimensional arrays are handled separately below
        ability_tokens = df["ability_input_tokens"].tolist()
        weapon_ids = df["weapon_id_token"].values
        is_null = df["is_null_token"].values
        weapon_names = df["weapon_name"].fillna("").values

        # Extract embedded arrays from DataFrame columns.  We use np.stack to
        # produce contiguous 2‑D arrays rather than storing ragged arrays in
        # Parquet, which would be inefficient.
        sae_input_arrays = np.stack(df["sae_input"].values)
        sae_recon_arrays = np.stack(df["sae_recon"].values)
        model_logits_arrays = np.stack(df["model_logits"].values)

        # Compose a clean Polars DataFrame for Parquet without large arrays
        clean_df = pl.DataFrame(
            {
                "batch_id": [batch_index] * len(df),
                "sample_id": list(range(len(df))),
                "ability_tokens": ability_tokens,
                "weapon_id_token": weapon_ids,
                "is_null_token": is_null,
                "weapon_name": weapon_names,
                # Provide a global index for cross‑batch lookups.  Note: we
                # compute the range based on the batch index and the number of
                # samples.
                "global_index": list(
                    range(batch_index * len(df), (batch_index + 1) * len(df))
                ),
            }
        )

        parquet_path = self.metadata_dir / f"batch_{batch_index:04d}.parquet"
        clean_df.write_parquet(
            parquet_path,
            compression=self.compression,
            compression_level=self.compression_level,
        )
        stats["parquet_size_mb"] = parquet_path.stat().st_size / (1024**2)

        # ------------------------------------------------------------------
        # Step 2: Save activations as Zarr v3
        print("  Converting activations to Zarr…")
        activations_path = (
            self.activations_dir / f"batch_{batch_index:04d}.zarr"
        )
        # Create a new array with appropriate chunking; we map the first axis to
        # chunks of ``chunk_size`` rows and the second axis is unchunked so
        # entire feature vectors reside in a single chunk along that dimension.
        z_activations = self._create_zarr_array(
            activations_path,
            shape=activations.shape,
            dtype=activations.dtype,
            chunks=(self.chunk_size, activations.shape[1]),
        )
        # Write the activations in slices to avoid loading the entire array at
        # once.  This loop writes contiguous blocks of rows.
        for start in range(0, len(activations), self.chunk_size):
            end = min(start + self.chunk_size, len(activations))
            z_activations[start:end] = activations[start:end]
        # Attach metadata to the array
        z_activations.attrs["batch_index"] = batch_index
        z_activations.attrs["n_samples"] = len(activations)
        z_activations.attrs["n_features"] = activations.shape[1]
        # Copy any other metadata fields from the input batch
        z_activations.attrs.update(metadata)

        # Compute size on disk for reporting
        stats["zarr_activations_size_mb"] = sum(
            f.stat().st_size for f in activations_path.rglob("*")
        ) / (1024**2)

        # ------------------------------------------------------------------
        # Step 3: Append embedded arrays to global Zarr stores
        print("  Appending embedded arrays…")
        self._append_to_global_array("sae_input", sae_input_arrays, batch_index)
        self._append_to_global_array("sae_recon", sae_recon_arrays, batch_index)
        self._append_to_global_array(
            "model_logits", model_logits_arrays, batch_index
        )

        # Tidy up to free memory between batches
        del batch_data, df, activations
        gc.collect()

        # Calculate compression ratio for this batch
        total_output_mb = (
            stats["parquet_size_mb"] + stats["zarr_activations_size_mb"]
        )
        stats["compression_ratio"] = (
            stats["input_size_mb"] / total_output_mb
            if total_output_mb
            else float("inf")
        )
        print(
            f"  ✓ Converted: {stats['input_size_mb']:.1f} MB -> {total_output_mb:.1f} MB "
            f"(compression ratio: {stats['compression_ratio']:.2f}x)"
        )
        return stats

    # ------------------------------------------------------------------
    def _append_to_global_array(
        self, name: str, data: np.ndarray, batch_index: int
    ) -> None:
        """Append data to a persistent global array in the embeddings directory.

        If the array already exists, its shape will be extended along the
        first axis and the new data appended using ``zarr.Array.append``.  If
        the array does not yet exist, a new one will be created with
        appropriate chunking and optional compression.  Metadata attributes
        recording the start and end indices of the appended block are stored
        on the array itself.
        """
        array_path = self.embeddings_dir / f"{name}.zarr"
        # Determine whether we are appending to an existing array or creating a new one
        if array_path.exists():
            # Open the existing array in appendable mode.  ``mode='a'`` allows
            # reading and writing without truncating the existing data.
            z = zarr.open(str(array_path), mode="a")  # type: ignore
            # Record the starting index before appending
            start_idx = z.shape[0]
            # Append along the first axis.  The return value is the new shape.
            new_shape = z.append(data)
            end_idx = new_shape[0]
        else:
            # Create a new array using the helper.  Use a chunk size of up to
            # 10,000 along the first axis as a reasonable default for large
            # embeddings; the second dimension (features) is kept as a single
            # chunk so that entire vectors are co‑located on disk.
            chunk0 = 10000 if data.ndim > 1 else min(10000, data.shape[0])
            chunks = (chunk0, data.shape[1]) if data.ndim > 1 else (chunk0,)
            z = self._create_zarr_array(
                array_path, shape=data.shape, dtype=data.dtype, chunks=chunks
            )
            # Initialise the array with the first batch of data
            z[:] = data
            start_idx = 0
            end_idx = data.shape[0]
        # Update metadata attributes with start and end indices for this batch
        z.attrs[f"batch_{batch_index:04d}_start"] = start_idx
        z.attrs[f"batch_{batch_index:04d}_end"] = end_idx

    # ------------------------------------------------------------------
    def convert_all(self) -> Dict[str, Any]:
        """Convert all discovered batch files and summarise the results.

        Iterates over every joblib file in the source directory and invokes
        :meth:`convert_batch` on each.  After all batches are processed, a
        summary JSON file containing aggregate statistics and parameters is
        written to the target directory.

        Returns
        -------
        dict
            A nested dictionary capturing per‑batch and overall statistics.
        """
        print(f"\nConverting {len(self.batch_files)} batch files…")
        print(f"Output directory: {self.target_dir}")
        all_stats: list[Dict[str, Any]] = []
        for idx, batch_file in enumerate(
            tqdm(self.batch_files, desc="Converting")
        ):
            stats = self.convert_batch(batch_file, idx)
            all_stats.append(stats)

        # Aggregate statistics across all batches
        total_stats = {
            "n_batches": len(self.batch_files),
            "total_samples": int(sum(s["n_samples"] for s in all_stats)),
            "total_input_size_mb": float(
                sum(s["input_size_mb"] for s in all_stats)
            ),
            "total_output_size_mb": float(
                sum(
                    s["parquet_size_mb"] + s["zarr_activations_size_mb"]
                    for s in all_stats
                )
            ),
            "average_compression_ratio": float(
                np.mean([s["compression_ratio"] for s in all_stats])
            ),
            "chunk_size": self.chunk_size,
            "compression": self.compression,
            "compression_level": self.compression_level,
            "batch_stats": all_stats,
        }
        # Write out a JSON summary
        metadata_path = self.target_dir / "conversion_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(total_stats, f, indent=2)
        # Print human‑readable summary
        print("\n" + "=" * 80)
        print("CONVERSION COMPLETE")
        print("=" * 80)
        print(f"Total samples: {total_stats['total_samples']:,}")
        print(f"Input size: {total_stats['total_input_size_mb']:.1f} MB")
        print(f"Output size: {total_stats['total_output_size_mb']:.1f} MB")
        print(
            f"Compression ratio: {total_stats['average_compression_ratio']:.2f}x"
        )
        print(f"Metadata saved to: {metadata_path}")
        return total_stats


def main() -> None:
    """Entrypoint for running the batch conversion as a script."""
    # Modify these paths as necessary to point to your input and output
    # locations.  The defaults assume an ``e:`` drive on Windows; adjust
    # accordingly when running on other platforms.
    source_dir = Path("/mnt/e/activations_ultra/batches")
    target_dir = Path("/mnt/e/activations_ultra_efficient")
    converter = BatchConverter(
        source_dir=source_dir,
        target_dir=target_dir,
        chunk_size=1000,
        compression="zstd",
        compression_level=3,
    )
    converter.convert_all()


if __name__ == "__main__":
    main()
