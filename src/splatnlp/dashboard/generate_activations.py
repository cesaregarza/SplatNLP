"""Generate and cache activations for dashboard visualization."""

import argparse
import gc
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import h5py
import joblib
import numpy as np
import orjson
import pandas as pd
import torch
from tqdm import tqdm

from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.monosemantic_sae.utils import setup_hook
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.utils.constants import PAD

logger = logging.getLogger(__name__)


def load_vocab(vocab_path: str) -> Dict:
    """Load vocabulary from file or S3."""
    if vocab_path.startswith(("http://", "https://", "s3://")):
        if vocab_path.startswith("s3://"):
            import boto3

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


def load_data(data_path: str) -> pd.DataFrame:
    """Load data from CSV file."""
    df = pd.read_csv(data_path, sep="\t", header=0)
    if isinstance(df["ability_tags"].iloc[0], str):
        df["ability_tags"] = df["ability_tags"].apply(orjson.loads)
    return df


def generate_null_token_activations(
    primary_model: torch.nn.Module,
    sae_model: torch.nn.Module,
    weapon_vocab: Dict[str, int],
    vocab: Dict[str, int],
    pad_token_id: int,  # noqa: F841
    device: str,
    hook_target: str = "masked_mean",
) -> List[Dict]:
    """Generate activations for null tokens with each weapon.

    Args:
        primary_model: The primary transformer model
        sae_model: The sparse autoencoder model
        weapon_vocab: Weapon vocabulary mapping
        vocab: Token vocabulary mapping
        pad_token_id: Padding token ID
        device: Device to run on (cuda/cpu)
        hook_target: Target layer for hook

    Returns:
        List of activation records for null tokens
    """
    logger.info(
        f"Generating null token activations for {len(weapon_vocab)} weapons..."
    )

    # Get null token ID
    null_token_id = vocab.get("<NULL>")
    if null_token_id is None:
        logger.warning(
            "No <NULL> token found in vocabulary, skipping null activations"
        )
        return []

    # Setup hook
    hook, hook_handle = setup_hook(primary_model, target=hook_target)

    records = []

    # Process each weapon
    with torch.no_grad():
        for weapon_name, weapon_id in tqdm(
            weapon_vocab.items(), desc="Null token activations"
        ):
            # Create input: just the null token
            inputs = torch.tensor(
                [[null_token_id]], dtype=torch.long, device=device
            )
            weapons = torch.tensor([weapon_id], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(inputs, dtype=torch.bool)
            key_padding = ~attention_mask

            # Forward pass
            hook.clear_activations()
            logits = primary_model(
                inputs, weapons, key_padding_mask=key_padding
            )
            primary_acts = hook.get_and_clear()
            sae_recon, sae_hidden = sae_model(primary_acts.float())

            # Store record
            records.append(
                {
                    "ability_input_tokens": [null_token_id],
                    "weapon_id_token": weapon_id,
                    "weapon_name": weapon_name,
                    "is_null_token": True,
                    "sae_input": primary_acts[0].cpu().numpy(),
                    "sae_hidden": sae_hidden[0].cpu().numpy(),
                    "sae_recon": sae_recon[0].cpu().numpy(),
                    "model_logits": logits[0].cpu().numpy(),
                }
            )

    # Clean up hook
    hook_handle.remove()

    logger.info(f"Generated {len(records)} null token activation records")
    return records


def collect_activations(
    primary_model: torch.nn.Module,
    sae_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    hook_target: str = "masked_mean",
) -> List[Dict]:
    """Collect activations for all examples in dataloader.

    Args:
        primary_model: The primary transformer model
        sae_model: The sparse autoencoder model
        dataloader: DataLoader with examples
        device: Device to run on (cuda/cpu)
        hook_target: Target layer for hook

    Returns:
        List of activation records
    """
    logger.info(
        f"Collecting activations for {len(dataloader.dataset)} examples..."
    )

    hook, hook_handle = setup_hook(primary_model, target=hook_target)

    records = []
    with torch.no_grad():
        for inputs, weapons, _, attention_masks in tqdm(
            dataloader, desc="Collecting activations"
        ):
            inputs = inputs.to(device)
            weapons = weapons.to(device)
            key_padding = ~attention_masks.to(device)

            hook.clear_activations()
            logits = primary_model(
                inputs, weapons, key_padding_mask=key_padding
            )
            primary_acts = hook.get_and_clear()
            sae_recon, sae_hidden = sae_model(primary_acts.float())

            batch_size = inputs.size(0)
            for i in range(batch_size):
                raw_ability_tokens = (
                    inputs[i][attention_masks[i]].cpu().tolist()
                )
                records.append(
                    {
                        "ability_input_tokens": raw_ability_tokens,
                        "weapon_id_token": int(weapons[i].item()),
                        "is_null_token": False,
                        "sae_input": primary_acts[i].cpu().numpy(),
                        "sae_hidden": sae_hidden[i].cpu().numpy(),
                        "sae_recon": sae_recon[i].cpu().numpy(),
                        "model_logits": logits[i].cpu().numpy(),
                    }
                )

    # Clean up hook
    hook_handle.remove()

    logger.info(f"Collected activations for {len(records)} records")
    return records


def generate_dashboard_activations(args: argparse.Namespace):
    """Generate and save activations for dashboard.

    Args:
        args: Command line arguments
    """
    # Setup logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load vocabularies
    logger.info("Loading vocabularies...")
    vocab = load_vocab(args.vocab_path)
    weapon_vocab = load_vocab(args.weapon_vocab_path)
    logger.info(
        f"Loaded vocab size: {len(vocab)}, weapon vocab size: {len(weapon_vocab)}"
    )

    pad_token_id = vocab.get(PAD, vocab.get("<PAD>"))

    # Load models
    logger.info("Loading primary model...")
    primary_model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(vocab),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=pad_token_id,
    )
    primary_model.load_state_dict(
        torch.load(
            args.primary_model_checkpoint, map_location=torch.device("cpu")
        )
    )
    primary_model.to(device)
    primary_model.eval()
    logger.info("Primary model loaded and set to eval mode")

    logger.info("Loading SAE model...")
    sae_model = SparseAutoencoder(
        input_dim=args.hidden_dim,
        expansion_factor=args.sae_expansion_factor,
    )
    sae_model.load_state_dict(
        torch.load(args.sae_model_checkpoint, map_location=torch.device("cpu"))
    )
    sae_model.to(device)
    sae_model.eval()
    logger.info("SAE model loaded and set to eval mode")

    # Check if cache exists and skip if not forced
    if os.path.exists(args.output_path) and not args.force:
        logger.info(f"Cache already exists at {args.output_path}")
        logger.info("Use --force to regenerate")
        return

    # Load and prepare data
    logger.info(f"Loading data from {args.data_path}...")
    df_full = load_data(args.data_path)
    logger.info(f"Loaded {len(df_full)} rows of data")

    # Create train split
    train_df, _, _ = generate_tokenized_datasets(
        df=df_full,
        frac=args.fraction,
        random_state=args.random_state,
        validation_size=0.0,
        test_size=0.0,
    )
    logger.info(f"Using {len(train_df)} rows for activation generation")

    # Create dataloader
    dataloader, _, _ = generate_dataloaders(
        train_set=train_df,
        validation_set=train_df,
        test_set=train_df,
        vocab_size=len(vocab),
        pad_token_id=pad_token_id,
        num_instances_per_set=1,
        skew_factor=1.2,
        null_token_id=None,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect regular activations
    records = collect_activations(
        primary_model=primary_model,
        sae_model=sae_model,
        dataloader=dataloader,
        device=device,
        hook_target=args.hook_target,
    )

    # Generate null token activations
    null_records = generate_null_token_activations(
        primary_model=primary_model,
        sae_model=sae_model,
        weapon_vocab=weapon_vocab,
        vocab=vocab,
        pad_token_id=pad_token_id,
        device=device,
        hook_target=args.hook_target,
    )

    # Combine all records
    all_records = records + null_records
    logger.info(
        f"Total records: {len(all_records)} ({len(records)} regular + {len(null_records)} null)"
    )

    # Convert to DataFrame
    analysis_df = pd.DataFrame(all_records)

    # Extract hidden activations array
    all_sae_hidden_activations = np.stack(
        analysis_df["sae_hidden"].to_list(), axis=0
    )
    logger.info(
        f"Shape of all_sae_hidden_activations: {all_sae_hidden_activations.shape}"
    )

    # Save cache
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving activations cache to {output_path}...")
    joblib.dump(
        {
            "analysis_df_records": analysis_df,
            "all_sae_hidden_activations": all_sae_hidden_activations,
            "metadata": {
                "num_regular": len(records),
                "num_null": len(null_records),
                "fraction": args.fraction,
                "vocab_size": len(vocab),
                "weapon_vocab_size": len(weapon_vocab),
            },
        },
        output_path,
    )

    logger.info("Activation generation complete!")

    # Print summary statistics
    logger.info("\nSummary:")
    logger.info(f"  Regular examples: {len(records)}")
    logger.info(f"  Null token examples: {len(null_records)}")
    logger.info(f"  Total examples: {len(all_records)}")
    logger.info(
        f"  Features per example: {all_sae_hidden_activations.shape[1]}"
    )
    logger.info(f"  Output saved to: {output_path}")


def collect_activations_chunk(
    primary_model: torch.nn.Module,
    sae_model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    hook_target: str = "masked_mean",
) -> Tuple[List[Dict], np.ndarray]:
    """Collect activations for chunk without storing heavy arrays in memory."""
    logger.info(f"Collecting activations for {len(dataloader.dataset)} examples...")

    hook, hook_handle = setup_hook(primary_model, target=hook_target)
    records = []
    hidden_activations = []

    with torch.no_grad():
        for inputs, weapons, _, attention_masks in tqdm(dataloader, desc="Collecting activations"):
            inputs = inputs.to(device)
            weapons = weapons.to(device)
            key_padding = ~attention_masks.to(device)

            hook.clear_activations()
            logits = primary_model(inputs, weapons, key_padding_mask=key_padding)
            primary_acts = hook.get_and_clear()
            sae_recon, sae_hidden = sae_model(primary_acts.float())

            batch_size = inputs.size(0)
            for i in range(batch_size):
                raw_ability_tokens = inputs[i][attention_masks[i]].cpu().tolist()
                records.append({
                    "ability_input_tokens": raw_ability_tokens,
                    "weapon_id_token": int(weapons[i].item()),
                    "is_null_token": False,
                })
                hidden_activations.append(sae_hidden[i].cpu().numpy())

    hook_handle.remove()
    hidden_activations = np.stack(hidden_activations, axis=0)
    
    logger.info(f"Collected activations for {len(records)} records")
    return records, hidden_activations


def save_chunked_activations(
    chunk_dir: Path,
    output_path: Path,
    metadata_df: pd.DataFrame,
    chunk_files: List[str],
    null_hidden_activations: np.ndarray,
    metadata: dict,
) -> None:
    """Save activations in a memory-efficient format using HDF5."""
    logger.info("Creating activation cache with HDF5 backend...")
    
    # Create HDF5 file
    h5_path = output_path.with_suffix('.h5')
    
    # Determine dimensions
    total_records = len(metadata_df)
    if chunk_files:
        data = np.load(chunk_files[0])
        feature_dim = data["hidden_activations"].shape[1]
        data.close()
    elif len(null_hidden_activations) > 0:
        feature_dim = null_hidden_activations.shape[1]
    else:
        raise ValueError("No data to process")
    
    logger.info(f"Creating HDF5 file with shape ({total_records}, {feature_dim})")
    
    # Create HDF5 and copy data
    with h5py.File(h5_path, 'w') as h5f:
        dset = h5f.create_dataset(
            'activations',
            shape=(total_records, feature_dim),
            dtype='float32',
            chunks=(min(1000, total_records), feature_dim),
            compression='gzip',
            compression_opts=1
        )
        
        current_idx = 0
        
        # Copy chunk data
        for chunk_file in tqdm(chunk_files, desc="Copying chunks to HDF5"):
            data = np.load(chunk_file)
            chunk_data = data["hidden_activations"]
            chunk_size = chunk_data.shape[0]
            dset[current_idx:current_idx + chunk_size] = chunk_data
            current_idx += chunk_size
            data.close()
            gc.collect()
        
        # Add null activations
        if len(null_hidden_activations) > 0:
            dset[current_idx:current_idx + len(null_hidden_activations)] = null_hidden_activations
    
    # Save metadata
    metadata_path = output_path.with_suffix('.metadata.joblib')
    joblib.dump({
        "analysis_df_records": metadata_df,
        "activation_h5_path": str(h5_path),
        "metadata": metadata,
    }, metadata_path, compress=3)
    
    # Create pointer file for compatibility
    with open(output_path, 'wb') as f:
        pickle.dump({
            "_loader_version": "streaming_v1",
            "metadata_path": str(metadata_path),
            "h5_path": str(h5_path),
            "shape": (total_records, feature_dim),
        }, f)
    
    logger.info(f"Cache saved: {output_path} (metadata), {h5_path} (data)")


def generate_dashboard_activations_chunked(args: argparse.Namespace):
    """Generate activations with memory-efficient chunking."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Load vocabularies
    logger.info("Loading vocabularies...")
    vocab = load_vocab(args.vocab_path)
    weapon_vocab = load_vocab(args.weapon_vocab_path)
    pad_token_id = vocab.get(PAD, vocab.get("<PAD>"))

    # Load models
    logger.info("Loading models...")
    primary_model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        output_dim=len(vocab),
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        num_inducing_points=args.num_inducing_points,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=pad_token_id,
    )
    primary_model.load_state_dict(torch.load(args.primary_model_checkpoint, map_location="cpu"))
    primary_model.to(device).eval()

    sae_model = SparseAutoencoder(
        input_dim=args.hidden_dim,
        expansion_factor=args.sae_expansion_factor,
    )
    sae_model.load_state_dict(torch.load(args.sae_model_checkpoint, map_location="cpu"))
    sae_model.to(device).eval()

    # Check existing cache
    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        logger.info(f"Cache already exists at {output_path}. Use --force to regenerate")
        return

    # Load and sample data
    logger.info(f"Loading data from {args.data_path}...")
    df_full = load_data(args.data_path)
    sampled_df = df_full.sample(frac=args.fraction, random_state=args.random_state, replace=False).reset_index(drop=True)
    logger.info(f"Sampled {len(sampled_df)} rows ({args.fraction:.1%} of {len(df_full)})")

    # Setup chunk storage
    chunk_dir = Path(args.chunk_storage_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    if args.force:
        for old_chunk in chunk_dir.glob("chunk_*.npz"):
            old_chunk.unlink()

    # Process chunks
    rows_per_chunk = int(len(df_full) * args.chunk_size)
    num_chunks = int(np.ceil(len(sampled_df) / rows_per_chunk))
    
    all_records_metadata = []
    chunk_files = []
    
    for chunk_idx in range(num_chunks):
        chunk_start = chunk_idx * rows_per_chunk
        chunk_end = min((chunk_idx + 1) * rows_per_chunk, len(sampled_df))
        
        logger.info(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (rows {chunk_start}-{chunk_end})")
        
        chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
        chunk_files.append(str(chunk_file))
        
        if chunk_file.exists() and not args.force:
            logger.info("Chunk exists, skipping...")
            data = np.load(chunk_file)
            num_records = int(data["num_records"])
            data.close()
            all_records_metadata.extend([{"is_null_token": False} for _ in range(num_records)])
            continue
        
        # Process chunk
        chunk_df = sampled_df.iloc[chunk_start:chunk_end].copy()
        chunk_loader, _, _ = generate_dataloaders(
            train_set=chunk_df,
            validation_set=chunk_df,
            test_set=chunk_df,
            vocab_size=len(vocab),
            pad_token_id=pad_token_id,
            num_instances_per_set=1,
            skew_factor=1.2,
            null_token_id=None,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
        )
        
        chunk_records, chunk_activations = collect_activations_chunk(
            primary_model, sae_model, chunk_loader, device, args.hook_target
        )
        
        np.savez_compressed(chunk_file, hidden_activations=chunk_activations, num_records=len(chunk_records))
        all_records_metadata.extend(chunk_records)
        
        del chunk_df, chunk_loader, chunk_records, chunk_activations
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # Generate null token activations
    logger.info("\nGenerating null token activations...")
    null_records = generate_null_token_activations(
        primary_model, sae_model, weapon_vocab, vocab, pad_token_id, device, args.hook_target
    )
    
    null_hidden_activations = np.stack([r["sae_hidden"] for r in null_records], axis=0) if null_records else np.empty((0, args.hidden_dim * int(args.sae_expansion_factor)))
    null_metadata = [{k: v for k, v in r.items() if k not in ["sae_input", "sae_hidden", "sae_recon", "model_logits"]} for r in null_records]
    all_records_metadata.extend(null_metadata)

    # Save everything
    logger.info(f"\nTotal records: {len(all_records_metadata)}")
    analysis_df = pd.DataFrame(all_records_metadata)
    
    metadata = {
        "num_regular": len(all_records_metadata) - len(null_records),
        "num_null": len(null_records),
        "fraction": args.fraction,
        "chunk_size": args.chunk_size,
        "vocab_size": len(vocab),
        "weapon_vocab_size": len(weapon_vocab),
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_chunked_activations(chunk_dir, output_path, analysis_df, chunk_files, null_hidden_activations, metadata)
    
    logger.info("Activation generation complete!")


def main():
    """Main entry point for activation generation."""
    parser = argparse.ArgumentParser(
        description="Generate activations for dashboard visualization",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model paths
    parser.add_argument(
        "--primary-model-checkpoint",
        type=str,
        required=True,
        help="Path to primary model checkpoint",
    )
    parser.add_argument(
        "--sae-model-checkpoint",
        type=str,
        required=True,
        help="Path to SAE model checkpoint",
    )

    # Vocabulary paths
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to vocabulary JSON file",
    )
    parser.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to weapon vocabulary JSON file",
    )

    # Data configuration
    parser.add_argument(
        "--data-path",
        type=str,
        required=True,
        help="Path to tokenized data CSV",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Fraction of data to use for activation generation",
    )
    parser.add_argument(
        "--chunk-size",
        type=float,
        default=0.01,
        help="Fraction of data to process per chunk (default: 0.01 = 1%)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for data sampling",
    )

    # Model configuration
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension of primary model",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension of primary model",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=3,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads",
    )
    parser.add_argument(
        "--num-inducing-points",
        type=int,
        default=32,
        help="Number of inducing points for PMA",
    )
    parser.add_argument(
        "--sae-expansion-factor",
        type=float,
        default=4.0,
        help="SAE expansion factor",
    )

    # Processing configuration
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--hook-target",
        type=str,
        default="masked_mean",
        help="Target layer for activation hook",
    )

    # Output configuration
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save activation cache",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration even if cache exists",
    )

    args = parser.parse_args()
    
    # Use chunked version for large datasets
    if args.fraction > args.chunk_size:
        logger.info(f"Using chunked processing (fraction {args.fraction} > chunk_size {args.chunk_size})")
        args.chunk_storage_dir = "/mnt/e/activations"
        generate_dashboard_activations_chunked(args)
    else:
        generate_dashboard_activations(args)


if __name__ == "__main__":
    main()
