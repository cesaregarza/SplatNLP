import logging
import os
import sys
from pathlib import Path

import joblib

# --- Set up project paths ---
project_root = Path(__file__).resolve().parent.parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()

# --- Load all required data/models (reuse notebook logic, but minimal) ---
import orjson
import pandas as pd
import torch

# --- Import dashboard app ---
from splatnlp.dashboard.app import app

# Project-specific imports
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder
from splatnlp.preprocessing.datasets.generate_datasets import (
    generate_dataloaders,
    generate_tokenized_datasets,
)
from splatnlp.preprocessing.transform.mappings import generate_maps

# --- Paths (edit as needed) ---
PRIMARY_MODEL_CHECKPOINT = "saved_models/dataset_v0_2_full/model.pth"
SAE_MODEL_PATH = "run_20250429_023422"
SAE_MODEL_CHECKPOINT = f"saved_models/dataset_v0_2_full/sae_runs/{SAE_MODEL_PATH}/sae_model_final.pth"
VOCAB_PATH = "saved_models/dataset_v0_2_full/vocab.json"
WEAPON_VOCAB_PATH = "saved_models/dataset_v0_2_full/weapon_vocab.json"
DATA_CSV = str(project_root / "test_data/tokenized/tokenized_data.csv")
FRACTION = 0.1  # Increased from 0.01 to 0.1 for more data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Load vocabs ---
logger.info("Loading vocabularies...")
with open(VOCAB_PATH, "rb") as f:
    vocab = orjson.loads(f.read())
with open(WEAPON_VOCAB_PATH, "rb") as f:
    weapon_vocab = orjson.loads(f.read())
inv_vocab = {v: k for k, v in vocab.items()}
inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}
logger.info(
    f"Loaded vocab size: {len(vocab)}, weapon vocab size: {len(weapon_vocab)}"
)

pad_token_id = vocab.get("<PAD>")

# --- Load models ---
logger.info("Loading primary model...")
primary_model = SetCompletionModel(
    vocab_size=len(vocab),
    weapon_vocab_size=len(weapon_vocab),
    embedding_dim=32,
    hidden_dim=512,
    output_dim=len(vocab),
    num_layers=3,
    num_heads=8,
    num_inducing_points=32,
    use_layer_norm=True,
    dropout=0.0,
    pad_token_id=pad_token_id,
)
primary_model.load_state_dict(
    torch.load(PRIMARY_MODEL_CHECKPOINT, map_location=torch.device("cpu"))
)
primary_model.to(DEVICE)
primary_model.eval()
logger.info("Primary model loaded and set to eval mode.")

logger.info("Loading SAE model...")
sae_model = SparseAutoencoder(
    input_dim=512,
    expansion_factor=4.0,
    l1_coefficient=0.0,
    target_usage=0.0,
    usage_coeff=0.0,
)
sae_model.load_state_dict(
    torch.load(SAE_MODEL_CHECKPOINT, map_location=torch.device("cpu"))
)
sae_model.to(DEVICE)
sae_model.eval()
logger.info("SAE model loaded and set to eval mode.")

# --- Load data ---
logger.info(f"Loading data from {DATA_CSV} ...")
df_full = pd.read_csv(DATA_CSV, sep="\t", header=0)
if isinstance(df_full["ability_tags"].iloc[0], str):
    df_full["ability_tags"] = df_full["ability_tags"].apply(orjson.loads)
logger.info(f"Loaded {len(df_full)} rows of data.")

train_df, _, _ = generate_tokenized_datasets(
    df=df_full,
    frac=FRACTION,
    random_state=42,
    validation_size=0.0,
    test_size=0.0,
)
logger.info(f"Tokenized training set: {len(train_df)} rows.")

analysis_loader, _, _ = generate_dataloaders(
    train_set=train_df,
    validation_set=train_df,
    test_set=train_df,
    vocab_size=len(vocab),
    pad_token_id=pad_token_id,
    num_instances_per_set=1,
    skew_factor=1.2,
    null_token_id=None,
    batch_size=64,
    shuffle=False,
    drop_last=False,
)
logger.info(
    f"Analysis DataLoader ready with {len(analysis_loader.dataset)} samples."
)

# --- Collect activations (minimal, for dashboard) ---
logger.info("Collecting activations for dashboard...")
ACTIVATIONS_CACHE = "/mnt/e/splatgpt/activations/analysis_cache.joblib"

# --- Try to load cached activations ---
if os.path.exists(ACTIVATIONS_CACHE):
    logger.info(f"Loading cached activations from {ACTIVATIONS_CACHE} ...")
    cache = joblib.load(ACTIVATIONS_CACHE)
    analysis_df_records = cache["analysis_df_records"]
    all_sae_hidden_activations = cache["all_sae_hidden_activations"]
    logger.info(
        f"Loaded cached activations: {all_sae_hidden_activations.shape}"
    )
    logger.info(f"Analysis DataFrame shape: {analysis_df_records.shape}")
    logger.info(
        f"Analysis DataFrame columns: {analysis_df_records.columns.tolist()}"
    )
    logger.info(f"Sample of analysis_df_records:\n{analysis_df_records.head()}")
else:
    logger.info("No cache found, collecting activations for dashboard...")
    from collections import defaultdict

    import numpy as np
    from tqdm import tqdm

    from splatnlp.monosemantic_sae.utils import setup_hook

    HOOK_TARGET = "masked_mean"
    hook, hook_handle = setup_hook(primary_model, target=HOOK_TARGET)

    records = []
    with torch.no_grad():
        for inputs, weapons, _, attention_masks in tqdm(
            analysis_loader, desc="Collecting activations", unit="batch"
        ):
            inputs = inputs.to(DEVICE)
            weapons = weapons.to(DEVICE)
            key_padding = ~attention_masks.to(DEVICE)
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
                        "sae_input": primary_acts[i].cpu().numpy(),
                        "sae_hidden": sae_hidden[i].cpu().numpy(),
                        "sae_recon": sae_recon[i].cpu().numpy(),
                        "model_logits": logits[i].cpu().numpy(),
                    }
                )
    logger.info(f"Collected activations for {len(records)} records.")
    analysis_df_records = pd.DataFrame(records)
    all_sae_hidden_activations = np.stack(
        analysis_df_records["sae_hidden"].to_list(), axis=0
    )
    logger.info(
        f"Shape of all_sae_hidden_activations: {all_sae_hidden_activations.shape}"
    )
    logger.info(f"Analysis DataFrame shape: {analysis_df_records.shape}")
    logger.info(
        f"Analysis DataFrame columns: {analysis_df_records.columns.tolist()}"
    )
    logger.info(f"Sample of analysis_df_records:\n{analysis_df_records.head()}")
    logger.info(f"Saving activations cache to {ACTIVATIONS_CACHE} ...")
    joblib.dump(
        {
            "analysis_df_records": analysis_df_records,
            "all_sae_hidden_activations": all_sae_hidden_activations,
        },
        ACTIVATIONS_CACHE,
    )

# --- Make data available to dashboard ---
from types import SimpleNamespace

from splatnlp.dashboard.components.feature_names import FeatureNamesManager

# Create feature names manager
feature_names_manager = FeatureNamesManager()
logger.info(f"Loaded {len(feature_names_manager.feature_names)} named features")

DASHBOARD_CONTEXT = SimpleNamespace(
    vocab=vocab,
    inv_vocab=inv_vocab,
    weapon_vocab=weapon_vocab,
    inv_weapon_vocab=inv_weapon_vocab,
    primary_model=primary_model,
    sae_model=sae_model,
    analysis_df_records=analysis_df_records,
    all_sae_hidden_activations=all_sae_hidden_activations,
    feature_names_manager=feature_names_manager,
)

# Optionally, monkeypatch or inject this context into the dashboard app
import splatnlp.dashboard.app as dashboard_app

setattr(dashboard_app, "DASHBOARD_CONTEXT", DASHBOARD_CONTEXT)

logger.info("Starting Dash dashboard server...")
# --- Run the dashboard ---
app.run(debug=True)
