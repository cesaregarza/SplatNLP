"""Command for generating and caching activations for dashboard visualization."""

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
    if not df.empty and isinstance(df["ability_tags"].iloc[0], str): # check if not empty
        df["ability_tags"] = df["ability_tags"].apply(orjson.loads)
    return df


def generate_null_token_activations(
    primary_model: torch.nn.Module,
    sae_model: torch.nn.Module,
    weapon_vocab: Dict[str, int],
    vocab: Dict[str, int],
    pad_token_id: int,
    device: str,
    hook_target: str = "masked_mean",
) -> List[Dict]:
    """Generate activations for null tokens with each weapon."""
    logger.info(
        f"Generating null token activations for {len(weapon_vocab)} weapons..."
    )
    null_token_id = vocab.get("<NULL>")
    if null_token_id is None:
        logger.warning(
            "No <NULL> token found in vocabulary, skipping null activations"
        )
        return []

    hook, hook_handle = setup_hook(primary_model, target=hook_target)
    records = []
    with torch.no_grad():
        for weapon_name, weapon_id in tqdm(
            weapon_vocab.items(), desc="Null token activations"
        ):
            inputs = torch.tensor(
                [[null_token_id]], dtype=torch.long, device=device
            )
            weapons = torch.tensor([weapon_id], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(inputs, dtype=torch.bool)
            key_padding = ~attention_mask

            hook.clear_activations()
            logits = primary_model(
                inputs, weapons, key_padding_mask=key_padding
            )
            primary_acts = hook.get_and_clear()
            sae_recon, sae_hidden = sae_model(primary_acts.float())

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
    """Collect activations for all examples in dataloader."""
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
    hook_handle.remove()
    logger.info(f"Collected activations for {len(records)} records")
    return records


def _generate_dashboard_activations_no_chunk(args: argparse.Namespace):
    """Internal function for non-chunked activation generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading vocabularies...")
    vocab = load_vocab(args.vocab_path)
    weapon_vocab = load_vocab(args.weapon_vocab_path)
    logger.info(
        f"Loaded vocab size: {len(vocab)}, weapon vocab size: {len(weapon_vocab)}"
    )
    pad_token_id = vocab.get(PAD, vocab.get("<PAD>"))

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
    primary_model.to(device).eval()
    logger.info("Primary model loaded and set to eval mode")

    logger.info("Loading SAE model...")
    sae_model = SparseAutoencoder(
        input_dim=args.hidden_dim,
        expansion_factor=args.sae_expansion_factor,
    )
    sae_model.load_state_dict(
        torch.load(args.sae_model_checkpoint, map_location=torch.device("cpu"))
    )
    sae_model.to(device).eval()
    logger.info("SAE model loaded and set to eval mode")

    output_path = Path(args.output_path) # Define output_path here
    if output_path.exists() and not args.force:
        logger.info(f"Cache already exists at {args.output_path}")
        logger.info("Use --force to regenerate")
        return

    logger.info(f"Loading data from {args.data_path}...")
    df_full = load_data(args.data_path)
    logger.info(f"Loaded {len(df_full)} rows of data")

    train_df, _, _ = generate_tokenized_datasets(
        df=df_full,
        frac=args.fraction,
        random_state=args.random_state,
        validation_size=0.0,
        test_size=0.0,
    )
    logger.info(f"Using {len(train_df)} rows for activation generation")

    dataloader, _, _ = generate_dataloaders(
        train_set=train_df,
        validation_set=train_df, # Using train_df for validation set as per original
        test_set=train_df, # Using train_df for test set as per original
        vocab_size=len(vocab),
        pad_token_id=pad_token_id,
        num_instances_per_set=1,
        skew_factor=1.2,
        null_token_id=None, # As per original
        batch_size=args.batch_size,
        shuffle=False, # As per original
        drop_last=False, # As per original
    )

    records = collect_activations(
        primary_model=primary_model,
        sae_model=sae_model,
        dataloader=dataloader,
        device=device,
        hook_target=args.hook_target,
    )

    null_records = generate_null_token_activations(
        primary_model=primary_model,
        sae_model=sae_model,
        weapon_vocab=weapon_vocab,
        vocab=vocab,
        pad_token_id=pad_token_id, # pad_token_id is passed here
        device=device,
        hook_target=args.hook_target,
    )

    all_records = records + null_records
    logger.info(
        f"Total records: {len(all_records)} ({len(records)} regular + {len(null_records)} null)"
    )

    analysis_df = pd.DataFrame(all_records)
    all_sae_hidden_activations = np.stack(
        analysis_df["sae_hidden"].to_list(), axis=0
    ) if all_records else np.empty((0, args.hidden_dim * int(args.sae_expansion_factor))) # handle empty all_records
    
    logger.info(
        f"Shape of all_sae_hidden_activations: {all_sae_hidden_activations.shape}"
    )

    # output_path was defined earlier
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
    logger.info("\nSummary:")
    logger.info(f"  Regular examples: {len(records)}")
    logger.info(f"  Null token examples: {len(null_records)}")
    logger.info(f"  Total examples: {len(all_records)}")
    if all_sae_hidden_activations.ndim == 2: # Check ndim before accessing shape[1]
        logger.info(
            f"  Features per example: {all_sae_hidden_activations.shape[1]}"
        )
    else:
        logger.info("  Features per example: N/A (no activations generated)")
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
    hidden_activations_array = np.empty((0,0)) # Default for empty
    if hidden_activations: # only stack if list is not empty
        # Get expected feature dimension from sae_model if possible
        feature_dim = sae_model.b_dec.shape[0] if hasattr(sae_model, 'b_dec') and sae_model.b_dec is not None else (sae_model.W_enc.shape[1] if hasattr(sae_model, 'W_enc') and sae_model.W_enc is not None else -1)
        if feature_dim != -1:
            hidden_activations_array = np.stack(hidden_activations, axis=0)
        else: # Fallback if feature_dim cannot be determined
            hidden_activations_array = np.array(hidden_activations) # Might result in object array if shapes are inconsistent

    logger.info(f"Collected activations for {len(records)} records, hidden_array_shape: {hidden_activations_array.shape}")
    return records, hidden_activations_array


def save_chunked_activations(
    chunk_dir: Path, # Added type hint for clarity
    output_path: Path,
    metadata_df: pd.DataFrame,
    chunk_files: List[str],
    null_hidden_activations: np.ndarray,
    metadata: dict,
    sae_output_dim: int # Added for robustness when no data in chunks
) -> None:
    """Save activations in a memory-efficient format using HDF5."""
    logger.info("Creating activation cache with HDF5 backend...")
    
    h5_path = output_path.with_suffix('.h5')
    
    total_records = len(metadata_df)
    feature_dim = -1

    if chunk_files:
        for chunk_file_path_str in chunk_files:
            try:
                with np.load(chunk_file_path_str) as data:
                    if "hidden_activations" in data and data["hidden_activations"].ndim == 2 and data["hidden_activations"].shape[0] > 0:
                        feature_dim = data["hidden_activations"].shape[1]
                        break
            except Exception as e:
                logger.warning(f"Could not read shape from chunk {chunk_file_path_str}: {e}")
    
    if feature_dim == -1 and len(null_hidden_activations) > 0 and null_hidden_activations.ndim == 2 and null_hidden_activations.shape[0] > 0:
         feature_dim = null_hidden_activations.shape[1]
    
    if feature_dim == -1 : # If still undetermined
        feature_dim = sae_output_dim # Use passed sae_output_dim
        logger.info(f"Feature dimension undetermined from data, using sae_output_dim: {feature_dim}")


    if total_records == 0 and feature_dim == 0 : # No records and no features, likely an issue
         logger.warning("No records and no features. Saving metadata only for empty cache.")
         # Fall through to metadata saving, HDF5 will be empty if created
    
    logger.info(f"Target HDF5 file: {h5_path} with shape ({total_records}, {feature_dim})")
    
    # Create HDF5 and copy data
    # Ensure chunks are appropriate for possibly empty data
    h5_chunks = None
    if total_records > 0 and feature_dim > 0:
        h5_chunks = (min(1000, total_records), feature_dim)

    with h5py.File(h5_path, 'w') as h5f:
        # Create dataset only if there's data to store, or if we want an empty placeholder
        dset = None
        if total_records > 0 and feature_dim > 0:
            dset = h5f.create_dataset(
                'activations',
                shape=(total_records, feature_dim),
                dtype='float32',
                chunks=h5_chunks,
                compression='gzip',
                compression_opts=1
            )
        else: # Create an empty dataset if total_records or feature_dim is 0
             dset = h5f.create_dataset('activations', shape=(total_records, feature_dim), dtype='float32')


        current_idx = 0
        
        for chunk_file in tqdm(chunk_files, desc="Copying chunks to HDF5"):
            try:
                with np.load(chunk_file) as data:
                    chunk_data = data.get("hidden_activations")
                    if chunk_data is not None and chunk_data.ndim == 2 and chunk_data.shape[0] > 0:
                        if chunk_data.shape[1] == feature_dim: # Check consistency
                            chunk_size = chunk_data.shape[0]
                            if dset is not None: # Ensure dset exists
                                dset[current_idx:current_idx + chunk_size] = chunk_data
                            current_idx += chunk_size
                        else:
                            logger.warning(f"Inconsistent feature dimension in chunk {chunk_file}. Expected {feature_dim}, got {chunk_data.shape[1]}. Skipping.")
                    elif chunk_data is not None and chunk_data.shape[0] == 0:
                        logger.info(f"Skipping empty data in chunk: {chunk_file}")
                    # else: logger.warning(f"Invalid or missing 'hidden_activations' in chunk: {chunk_file}") # Can be noisy
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_file}: {e}")
            gc.collect()
        
        if len(null_hidden_activations) > 0 and null_hidden_activations.ndim == 2 and null_hidden_activations.shape[0] > 0:
            if null_hidden_activations.shape[1] == feature_dim: # Check consistency
                if dset is not None: # Ensure dset exists
                    dset[current_idx:current_idx + len(null_hidden_activations)] = null_hidden_activations
            else:
                 logger.warning(f"Inconsistent feature dimension in null_hidden_activations. Expected {feature_dim}, got {null_hidden_activations.shape[1]}. Skipping.")
    
    metadata_path = output_path.with_suffix('.metadata.joblib')
    joblib.dump({
        "analysis_df_records": metadata_df,
        "activation_h5_path": str(h5_path) if (total_records > 0 and feature_dim > 0) else None, # Store None if no data
        "metadata": metadata,
    }, metadata_path, compress=3)
    
    with open(output_path, 'wb') as f:
        pickle.dump({
            "_loader_version": "streaming_v1" if (total_records > 0 and feature_dim > 0) else "streaming_v1_empty",
            "metadata_path": str(metadata_path),
            "h5_path": str(h5_path) if (total_records > 0 and feature_dim > 0) else None,
            "shape": (total_records, feature_dim),
        }, f)
    
    logger.info(f"Cache saved: {output_path} (pointer), {metadata_path} (metadata), {h5_path if (total_records > 0 and feature_dim > 0) else '(no data H5)'} (data)")


def _generate_dashboard_activations_chunked(args: argparse.Namespace):
    """Internal function for chunked activation generation."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Loading vocabularies...")
    vocab = load_vocab(args.vocab_path)
    weapon_vocab = load_vocab(args.weapon_vocab_path)
    pad_token_id = vocab.get(PAD, vocab.get("<PAD>"))

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
    sae_output_dim = args.hidden_dim * int(args.sae_expansion_factor)


    output_path = Path(args.output_path)
    if output_path.exists() and not args.force:
        logger.info(f"Cache already exists at {output_path}. Use --force to regenerate")
        return

    logger.info(f"Loading data from {args.data_path}...")
    df_full = load_data(args.data_path)
    
    args.fraction = max(0.0, min(1.0, args.fraction))
    if args.fraction == 0.0:
        logger.warning("Fraction is 0, no data will be sampled for regular activations.")
        sampled_df = pd.DataFrame(columns=df_full.columns if not df_full.empty else None)
    else:
        sampled_df = df_full.sample(frac=args.fraction, random_state=args.random_state, replace=False).reset_index(drop=True) if not df_full.empty else pd.DataFrame()
    logger.info(f"Sampled {len(sampled_df)} rows ({args.fraction:.1%} of {len(df_full)})")
    
    chunk_storage_dir = getattr(args, 'chunk_storage_dir', '/mnt/e/activations_default_cmd_chunked')
    chunk_dir = Path(chunk_storage_dir)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    if args.force:
        for old_chunk in chunk_dir.glob("chunk_*.npz"):
            logger.info(f"Deleting old chunk: {old_chunk}")
            old_chunk.unlink()

    all_records_metadata = []
    chunk_files = []
    
    args.chunk_size = max(0.0001, min(args.chunk_size, args.fraction if args.fraction > 0 else 1.0))

    if len(sampled_df) > 0 :
        rows_per_chunk = max(1, int(len(df_full) * args.chunk_size)) if len(df_full) > 0 else 1 # handle empty df_full
        num_chunks = int(np.ceil(len(sampled_df) / rows_per_chunk)) if rows_per_chunk > 0 else 0
        
        for chunk_idx in range(num_chunks):
            chunk_start = chunk_idx * rows_per_chunk
            chunk_end = min((chunk_idx + 1) * rows_per_chunk, len(sampled_df))
            
            logger.info(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (rows {chunk_start}-{chunk_end})")
            
            chunk_file = chunk_dir / f"chunk_{chunk_idx:04d}.npz"
            chunk_files.append(str(chunk_file))
            
            if chunk_file.exists() and not args.force:
                logger.info(f"Chunk {chunk_file} exists, attempting to load metadata...")
                try:
                    with np.load(chunk_file) as data:
                        num_records = int(data["num_records"])
                    all_records_metadata.extend([{"is_null_token": False} for _ in range(num_records)])
                    logger.info(f"Loaded metadata for {num_records} records from existing chunk.")
                    continue
                except Exception as e:
                    logger.warning(f"Could not load existing chunk {chunk_file}: {e}. Will attempt to regenerate.")
            
            chunk_df_current = sampled_df.iloc[chunk_start:chunk_end].copy()
            if len(chunk_df_current) == 0:
                logger.info(f"Skipping empty chunk {chunk_idx + 1}/{num_chunks}")
                np.savez_compressed(chunk_file, hidden_activations=np.empty((0, sae_output_dim)), num_records=0)
                continue

            chunk_loader, _, _ = generate_dataloaders(
                train_set=chunk_df_current, validation_set=chunk_df_current, test_set=chunk_df_current,
                vocab_size=len(vocab), pad_token_id=pad_token_id,
                num_instances_per_set=1, skew_factor=1.2, null_token_id=None,
                batch_size=args.batch_size, shuffle=False, drop_last=False,
            )
            
            chunk_records, chunk_activations = collect_activations_chunk(
                primary_model, sae_model, chunk_loader, device, args.hook_target
            )
            
            # Ensure chunk_activations has the correct second dimension if it's empty
            if chunk_activations.ndim == 2 and chunk_activations.shape[0] == 0 and chunk_activations.shape[1] == 0:
                 np.savez_compressed(chunk_file, hidden_activations=np.empty((0, sae_output_dim)), num_records=len(chunk_records))
            else:
                 np.savez_compressed(chunk_file, hidden_activations=chunk_activations, num_records=len(chunk_records))
            all_records_metadata.extend(chunk_records)
            
            del chunk_df_current, chunk_loader, chunk_records, chunk_activations
            gc.collect()
            if device == "cuda": torch.cuda.empty_cache()
    else:
        logger.info("No regular data to process based on fraction and sampling.")

    logger.info("\nGenerating null token activations...")
    null_records_full = generate_null_token_activations(
        primary_model, sae_model, weapon_vocab, vocab, pad_token_id, device, args.hook_target
    )
    
    if null_records_full:
        null_hidden_activations = np.stack([r["sae_hidden"] for r in null_records_full], axis=0)
        null_metadata_records = [{k: v for k, v in r.items() if k not in ["sae_input", "sae_hidden", "sae_recon", "model_logits"]} for r in null_records_full]
    else:
        null_hidden_activations = np.empty((0, sae_output_dim)) 
        null_metadata_records = []

    all_records_metadata.extend(null_metadata_records)

    logger.info(f"\nTotal records for metadata: {len(all_records_metadata)}")
    analysis_df = pd.DataFrame(all_records_metadata)
    
    metadata = {
        "num_regular": len(all_records_metadata) - len(null_metadata_records),
        "num_null": len(null_metadata_records),
        "fraction": args.fraction,
        "chunk_size": args.chunk_size,
        "vocab_size": len(vocab),
        "weapon_vocab_size": len(weapon_vocab),
        "sae_output_dim": sae_output_dim # Store this for later loading if needed
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_chunked_activations(chunk_dir, output_path, analysis_df, chunk_files, null_hidden_activations, metadata, sae_output_dim)
    
    logger.info("Activation generation command complete!")


def generate_activations_command(args: argparse.Namespace):
    """Main command function for generating activations."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logger.info("Executing generate_activations_command")

    effective_chunk_size = getattr(args, 'chunk_size', 0.01)
    if effective_chunk_size <= 0:
        logger.warning("args.chunk_size not set or invalid, defaulting to 0.01.")
        effective_chunk_size = 0.01
    
    effective_fraction = getattr(args, 'fraction', 0.0)
    if effective_fraction < 0:
        logger.warning("args.fraction not set or invalid, defaulting to 0.0.")
        effective_fraction = 0.0
    
    # Ensure sae_output_dim is available for empty cache scenario
    sae_output_dim = args.hidden_dim * int(args.sae_expansion_factor)


    if effective_fraction > 0 and effective_fraction > effective_chunk_size :
        logger.info(f"Using chunked processing (fraction {effective_fraction} > chunk_size {effective_chunk_size})")
        if not hasattr(args, 'chunk_storage_dir') or args.chunk_storage_dir is None:
            args.chunk_storage_dir = "/mnt/e/activations_cmd_default_chunked_main"
            logger.info(f"args.chunk_storage_dir not provided, defaulting to {args.chunk_storage_dir}")
        _generate_dashboard_activations_chunked(args)
    elif effective_fraction > 0:
        logger.info(f"Using non-chunked processing (fraction {effective_fraction})")
        _generate_dashboard_activations_no_chunk(args)
    else: # effective_fraction is 0 or invalid
        logger.info("Skipping activation generation as fraction is <= 0.")
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not output_path.exists() or args.force:
            logger.info("Creating empty cache structure because fraction is 0 and (cache missing or --force).")
            vocab = load_vocab(args.vocab_path)
            weapon_vocab = load_vocab(args.weapon_vocab_path)
            
            empty_analysis_df = pd.DataFrame(columns=["ability_input_tokens", "weapon_id_token", "is_null_token"])
            empty_hidden_activations = np.empty((0, sae_output_dim))
            
            metadata_payload = {
                "num_regular": 0, "num_null": 0, "fraction": effective_fraction,
                "vocab_size": len(vocab), "weapon_vocab_size": len(weapon_vocab),
                "sae_output_dim": sae_output_dim
            }
            if hasattr(args, 'chunk_size'): metadata_payload["chunk_size"] = effective_chunk_size

            chunk_storage_dir = getattr(args, 'chunk_storage_dir', '/mnt/e/activations_default_empty_cmd_main')
            chunk_dir_path = Path(chunk_storage_dir)
            chunk_dir_path.mkdir(parents=True, exist_ok=True)

            save_chunked_activations(
                chunk_dir=chunk_dir_path, output_path=output_path,
                metadata_df=empty_analysis_df, chunk_files=[], 
                null_hidden_activations=empty_hidden_activations, 
                metadata=metadata_payload, sae_output_dim=sae_output_dim
            )
            logger.info(f"Empty cache created at {output_path}")
        else:
            logger.info(f"Cache already exists at {output_path}, fraction is 0, and --force not used. No action taken.")
