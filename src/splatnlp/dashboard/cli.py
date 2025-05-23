import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace  # Ensure SimpleNamespace is imported

import joblib
import orjson
import pandas as pd
import torch

# --- Set up project paths ---
# This assumes cli.py is in src/splatnlp/dashboard/
# So project_root is Path(__file__).resolve().parent.parent.parent
project_root = Path(__file__).resolve().parent.parent.parent
# src_path should point to the 'src' directory itself.
# If cli.py is in src/splatnlp/dashboard, then src_path is project_root.
# This is a bit confusing; typically, if your package is splatnlp,
# and it's inside 'src', then 'src' is what you add to PYTHONPATH.
# Let's adjust based on the typical structure where 'src' contains 'splatnlp'.
# If __file__ is src/splatnlp/dashboard/cli.py, then project_root (Path(__file__).parent.parent.parent) is 'src'.
# And the actual root of the project (containing 'src', 'tests', etc.) is project_root.parent.
# However, the original script had:
# project_root = Path(__file__).resolve().parent.parent # This assumes cli.py is two levels deep from project root
# src_path = project_root / "src" # This assumes 'src' is a subdir of project_root
# This setup is unusual if cli.py is already in src/splatnlp/...
# Let's assume a standard layout:
# project_actual_root/
#   src/
#     splatnlp/
#       dashboard/
#         cli.py
# In this case, project_actual_root should be added to sys.path if we want to run `python src/splatnlp/dashboard/cli.py`
# Or, if running as `python -m splatnlp.dashboard.cli`, Python handles paths.
# For robustness if script is run directly:
current_script_path = Path(__file__).resolve()
# src/splatnlp/dashboard/cli.py -> src/splatnlp/dashboard -> src/splatnlp -> src
project_src_dir = current_script_path.parent.parent.parent
if str(project_src_dir) not in sys.path:
    sys.path.append(str(project_src_dir))
# Now imports like `from splatnlp.dashboard.app import app` should work.


import h5py  # Add this import

# --- Configure logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# --- Import dashboard app ---
from splatnlp.dashboard.app import DASHBOARD_CONTEXT as app_context_ref
from splatnlp.dashboard.app import app
from splatnlp.dashboard.components.feature_names import FeatureNamesManager

# Project-specific imports
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --- Main data loading and context creation function ---
def load_dashboard_data(args_ns: argparse.Namespace):  # Use Namespace type hint
    logger.info("Loading vocabularies...")
    with open(args_ns.vocab_path, "rb") as f:  # Use args_ns
        vocab = orjson.loads(f.read())
    with open(args_ns.weapon_vocab_path, "rb") as f:  # Use args_ns
        weapon_vocab = orjson.loads(f.read())
    inv_vocab = {v: k for k, v in vocab.items()}
    inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}
    logger.info(
        f"Loaded vocab size: {len(vocab)}, weapon vocab size: {len(weapon_vocab)}"
    )

    pad_token_id = vocab.get("<PAD>")

    logger.info("Loading primary model...")
    primary_model = SetCompletionModel(
        vocab_size=len(vocab),
        weapon_vocab_size=len(weapon_vocab),
        embedding_dim=args_ns.primary_embedding_dim,
        hidden_dim=args_ns.primary_hidden_dim,
        output_dim=len(vocab),
        num_layers=args_ns.primary_num_layers,
        num_heads=args_ns.primary_num_heads,
        num_inducing_points=args_ns.primary_num_inducing,
        use_layer_norm=True,
        dropout=0.0,
        pad_token_id=pad_token_id,
    )
    primary_model.load_state_dict(
        torch.load(
            args_ns.primary_model_checkpoint, map_location=torch.device("cpu")
        )
    )
    primary_model.to(DEVICE)
    primary_model.eval()
    logger.info("Primary model loaded and set to eval mode.")

    logger.info("Loading SAE model...")
    sae_model = SparseAutoencoder(
        input_dim=args_ns.primary_hidden_dim,
        expansion_factor=args_ns.sae_expansion_factor,
    )
    sae_model.load_state_dict(
        torch.load(
            args_ns.sae_model_checkpoint, map_location=torch.device("cpu")
        )
    )
    sae_model.to(DEVICE)
    sae_model.eval()
    logger.info("SAE model loaded and set to eval mode.")

    token_activations_accessor = None
    if args_ns.token_activations_path:  # Check if path is provided
        if os.path.exists(args_ns.token_activations_path):
            try:
                token_activations_accessor = h5py.File(
                    args_ns.token_activations_path, "r"
                )
                logger.info(
                    f"Successfully loaded token activations HDF5 file from {args_ns.token_activations_path}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to load token activations HDF5 file from {args_ns.token_activations_path}: {e}"
                )
                token_activations_accessor = None  # Ensure it's None on failure
        else:
            logger.warning(
                f"Token activations path provided ({args_ns.token_activations_path}) but file not found."
            )
            # token_activations_accessor remains None

    if os.path.exists(args_ns.activations_cache_path):
        logger.info(
            f"Loading cached activations from {args_ns.activations_cache_path} ..."
        )
        cache = joblib.load(args_ns.activations_cache_path)
        analysis_df_records_data = cache["analysis_df_records"]
        all_sae_hidden_activations = cache["all_sae_hidden_activations"]

        if isinstance(analysis_df_records_data, list):
            analysis_df_records = pd.DataFrame(analysis_df_records_data)
        elif isinstance(analysis_df_records_data, pd.DataFrame):
            analysis_df_records = analysis_df_records_data
        else:
            logger.error(
                "Cached 'analysis_df_records' is not a list of dicts or a DataFrame."
            )
            raise ValueError(
                "Cached 'analysis_df_records' is not a list of dicts or a DataFrame."
            )
        logger.info(
            f"Loaded cached activations: {all_sae_hidden_activations.shape}"
        )
    else:
        logger.error(
            f"Activations cache not found at {args_ns.activations_cache_path}."
        )
        logger.info(
            "Please generate the activations cache first using a separate script."
        )
        raise FileNotFoundError(
            f"Activations cache not found: {args_ns.activations_cache_path}. Pre-generate this file."
        )

    # Create feature names manager
    feature_names_manager = FeatureNamesManager()
    logger.info(
        f"Loaded {len(feature_names_manager.feature_names)} named features"
    )

    # Create a new SimpleNamespace object for the data to be returned
    # This is distinct from the app_context_ref which is the global context in app.py
    dashboard_context_data = SimpleNamespace(
        vocab=vocab,
        inv_vocab=inv_vocab,
        weapon_vocab=weapon_vocab,
        inv_weapon_vocab=inv_weapon_vocab,
        primary_model=primary_model,
        sae_model=sae_model,
        analysis_df_records=analysis_df_records,  # This is now a DataFrame
        all_sae_hidden_activations=all_sae_hidden_activations,
        token_activations_accessor=token_activations_accessor,  # Add this
        feature_names_manager=feature_names_manager,
        device=DEVICE,
    )
    return dashboard_context_data


def main():
    parser = argparse.ArgumentParser(
        description="Run the SplatNLP SAE Feature Dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,  # Provides default values in help messages
    )

    # --- Required Paths ---
    parser.add_argument(
        "--primary-model-checkpoint",
        type=str,
        required=True,
        help="Path to the primary model's .pth checkpoint file.",
    )
    parser.add_argument(
        "--sae-model-checkpoint",
        type=str,
        required=True,
        help="Path to the SAE model's .pth checkpoint file.",
    )
    parser.add_argument(
        "--vocab-path",
        type=str,
        required=True,
        help="Path to the main vocabulary JSON file (e.g., 'vocab.json').",
    )
    parser.add_argument(
        "--weapon-vocab-path",
        type=str,
        required=True,
        help="Path to the weapon vocabulary JSON file (e.g., 'weapon_vocab.json').",
    )
    parser.add_argument(
        "--activations-cache-path",
        type=str,
        required=True,
        help="Path to the mandatory joblib cache file. This file must contain 'analysis_df_records' (Pandas DataFrame or list of dicts) and 'all_sae_hidden_activations' (NumPy array).",
    )

    # --- Optional Paths ---
    parser.add_argument(
        "--token-activations-path",
        type=str,
        default=None,
        help="Optional path to an HDF5 file containing per-token primary model hidden state activations. "
        "This enables detailed projection tooltips for input tokens in 'Top Activating Examples' and 'Subsampled Intervals Grid'. "
        "Datasets within the HDF5 file should be named by example index (e.g., '0', '1', ...) and each should store a (sequence_length, hidden_dimension) NumPy array.",
    )

    # --- Primary Model Configuration (matching SetCompletionModel defaults if possible) ---
    parser.add_argument(
        "--primary-embedding-dim",
        type=int,
        default=32,
        help="Embedding dimension for the primary model.",
    )
    parser.add_argument(
        "--primary-hidden-dim",
        type=int,
        default=512,
        help="Hidden dimension for the primary model. This also serves as the input dimension for the SAE.",
    )
    parser.add_argument(
        "--primary-num-layers",
        type=int,
        default=3,
        help="Number of layers in the primary model's transformer encoder.",
    )
    parser.add_argument(
        "--primary-num-heads",
        type=int,
        default=8,
        help="Number of attention heads in the primary model.",
    )
    parser.add_argument(
        "--primary-num-inducing",
        type=int,
        default=32,
        help="Number of inducing points for PMA in the primary model.",
    )

    # --- SAE Model Configuration (matching SparseAutoencoder defaults if possible) ---
    parser.add_argument(
        "--sae-expansion-factor",
        type=float,
        default=4.0,
        help="SAE expansion factor, determines SAE hidden_dim relative to input_dim (primary_hidden_dim * expansion_factor).",
    )

    # --- Server Configuration ---
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address for the Dash dashboard server.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port number for the Dash dashboard server.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run Dash server in debug mode, enabling hot-reloading and verbose error messages.",
    )

    args = parser.parse_args()

    dashboard_data_obj = load_dashboard_data(args)

    # Populate the DASHBOARD_CONTEXT in app.py by setting attributes on the imported SimpleNamespace
    # This works because app_context_ref is a reference to the DASHBOARD_CONTEXT SimpleNamespace in app.py
    app_context_ref.vocab = dashboard_data_obj.vocab
    app_context_ref.inv_vocab = dashboard_data_obj.inv_vocab
    app_context_ref.weapon_vocab = dashboard_data_obj.weapon_vocab
    app_context_ref.inv_weapon_vocab = dashboard_data_obj.inv_weapon_vocab
    app_context_ref.primary_model = dashboard_data_obj.primary_model
    app_context_ref.sae_model = dashboard_data_obj.sae_model
    # Ensure analysis_df_records in the context is the DataFrame
    app_context_ref.analysis_df_records = dashboard_data_obj.analysis_df_records
    app_context_ref.all_sae_hidden_activations = (
        dashboard_data_obj.all_sae_hidden_activations
    )
    app_context_ref.token_activations_accessor = (
        dashboard_data_obj.token_activations_accessor
    )  # Add this line
    app_context_ref.feature_names_manager = (
        dashboard_data_obj.feature_names_manager
    )
    app_context_ref.device = dashboard_data_obj.device

    logger.info(
        f"Populated DASHBOARD_CONTEXT in app.py. Type of analysis_df_records: {type(app_context_ref.analysis_df_records)}"
    )
    if app_context_ref.token_activations_accessor:
        logger.info(
            f"Token activations accessor loaded: {app_context_ref.token_activations_accessor.filename}"
        )
    else:
        logger.info(
            "Token activations accessor not loaded (path not provided or file error)."
        )
    logger.info("Starting Dash dashboard server...")
    app.run_server(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
