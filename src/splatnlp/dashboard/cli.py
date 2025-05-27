#!/usr/bin/env python3
"""Command-line interface for the SplatNLP SAE Feature Dashboard."""

import argparse
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import orjson
import torch

from splatnlp.dashboard.app import DASHBOARD_CONTEXT as app_context_ref
from splatnlp.dashboard.app import app
from splatnlp.dashboard.components.feature_labels import FeatureLabelsManager
from splatnlp.model.models import SetCompletionModel
from splatnlp.monosemantic_sae.models import SparseAutoencoder

logger = logging.getLogger(__name__)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_dashboard_data(args_ns: argparse.Namespace):
    logger.info("Loading vocabularies...")
    with open(args_ns.vocab_path, "rb") as f:
        vocab = orjson.loads(f.read())
    with open(args_ns.weapon_vocab_path, "rb") as f:
        weapon_vocab = orjson.loads(f.read())
    inv_vocab = {v: k for k, v in vocab.items()}
    inv_weapon_vocab = {v: k for k, v in weapon_vocab.items()}
    pad_token_id = vocab.get("<PAD>")

    # Load models only if needed for dynamic tooltips
    primary_model = None
    sae_model = None
    if args_ns.enable_dynamic_tooltips:
        logger.info("Loading models for dynamic tooltips...")
        primary_model = SetCompletionModel(
            vocab_size=len(vocab), weapon_vocab_size=len(weapon_vocab),
            embedding_dim=args_ns.primary_embedding_dim, hidden_dim=args_ns.primary_hidden_dim,
            output_dim=len(vocab), num_layers=args_ns.primary_num_layers,
            num_heads=args_ns.primary_num_heads, num_inducing_points=args_ns.primary_num_inducing,
            use_layer_norm=True, dropout=0.0, pad_token_id=pad_token_id,
        )
        primary_model.load_state_dict(torch.load(args_ns.primary_model_checkpoint, map_location=torch.device("cpu")))
        primary_model.to(DEVICE).eval()

        sae_model = SparseAutoencoder(
            input_dim=args_ns.primary_hidden_dim, expansion_factor=args_ns.sae_expansion_factor,
        )
        sae_model.load_state_dict(torch.load(args_ns.sae_model_checkpoint, map_location=torch.device("cpu")))
        sae_model.to(DEVICE).eval()

    # Initialize database-backed context or fallback to precomputed analytics
    precomputed_analytics = None
    db_context = None
    
    if hasattr(args_ns, 'database_path') and args_ns.database_path:
        logger.info(f"Using database-backed context from {args_ns.database_path}")
        try:
            from splatnlp.dashboard.database_manager import DashboardDatabase, DatabaseBackedContext
            db_manager = DashboardDatabase(args_ns.database_path)
            db_context = DatabaseBackedContext(db_manager)
            
            # Get database info
            db_info = db_manager.get_database_info()
            logger.info(f"Database loaded: {db_info}")
            
        except Exception as e:
            logger.error(f"Failed to load database: {e}")
            logger.info("Falling back to precomputed analytics...")
            db_context = None
    
    if db_context is None:
        # Fallback to precomputed analytics
        analytics_path = getattr(args_ns, 'precomputed_analytics_path', None)
        if analytics_path and os.path.exists(analytics_path):
            logger.info(f"Loading precomputed analytics from {analytics_path}")
            try:
                precomputed_analytics = joblib.load(analytics_path)
                logger.info(f"Successfully loaded precomputed analytics. Found data for {len(precomputed_analytics.get('features', []))} features.")
            except Exception as e:
                logger.error(f"Failed to load precomputed analytics file: {e}")
                raise
        else:
            logger.warning("No database or precomputed analytics available")

    feature_labels_manager = FeatureLabelsManager()

    dashboard_context_data = SimpleNamespace(
        vocab=vocab, inv_vocab=inv_vocab,
        weapon_vocab=weapon_vocab, inv_weapon_vocab=inv_weapon_vocab,
        primary_model=primary_model,
        sae_model=sae_model,
        precomputed_analytics=precomputed_analytics,
        db_context=db_context,
        feature_labels_manager=feature_labels_manager,
        device=DEVICE,
    )
    return dashboard_context_data

def main():
    parser = argparse.ArgumentParser(
        description="Run the SplatNLP SAE Feature Dashboard.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Data source (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--database-path", type=str,
                          help="Path to dashboard database (SQLite file)")
    data_group.add_argument("--precomputed-analytics-path", type=str,
                          help="Path to the precomputed dashboard analytics file (e.g., .joblib)")
    
    # Required arguments
    parser.add_argument("--vocab-path", type=str, required=True,
                      help="Path to vocabulary JSON file")
    parser.add_argument("--weapon-vocab-path", type=str, required=True,
                      help="Path to weapon vocabulary JSON file")
    
    # Optional arguments for dynamic tooltips
    parser.add_argument("--enable-dynamic-tooltips", action="store_true",
                      help="Enable dynamic tooltip computation (requires model checkpoints)")
    parser.add_argument("--primary-model-checkpoint", type=str,
                      help="Path to primary model checkpoint (required if --enable-dynamic-tooltips)")
    parser.add_argument("--sae-model-checkpoint", type=str,
                      help="Path to SAE model checkpoint (required if --enable-dynamic-tooltips)")
    
    # Model configuration (only needed if dynamic tooltips enabled)
    parser.add_argument("--primary-embedding-dim", type=int, default=32)
    parser.add_argument("--primary-hidden-dim", type=int, default=512)
    parser.add_argument("--primary-num-layers", type=int, default=3)
    parser.add_argument("--primary-num-heads", type=int, default=8)
    parser.add_argument("--primary-num-inducing", type=int, default=32)
    parser.add_argument("--sae-expansion-factor", type=float, default=4.0)
    
    # Server configuration
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    
    # Validate arguments
    if args.enable_dynamic_tooltips:
        if not args.primary_model_checkpoint or not args.sae_model_checkpoint:
            parser.error("--primary-model-checkpoint and --sae-model-checkpoint are required when --enable-dynamic-tooltips is set")
    
    dashboard_data_obj = load_dashboard_data(args)

    # Populate the DASHBOARD_CONTEXT
    app_context_ref.vocab = dashboard_data_obj.vocab
    app_context_ref.inv_vocab = dashboard_data_obj.inv_vocab
    app_context_ref.weapon_vocab = dashboard_data_obj.weapon_vocab
    app_context_ref.inv_weapon_vocab = dashboard_data_obj.inv_weapon_vocab
    app_context_ref.primary_model = dashboard_data_obj.primary_model
    app_context_ref.sae_model = dashboard_data_obj.sae_model
    app_context_ref.precomputed_analytics = dashboard_data_obj.precomputed_analytics
    app_context_ref.db_context = dashboard_data_obj.db_context
    app_context_ref.feature_labels_manager = dashboard_data_obj.feature_labels_manager
    app_context_ref.device = dashboard_data_obj.device

    if dashboard_data_obj.db_context:
        logger.info("Populated DASHBOARD_CONTEXT with database-backed context")
    elif dashboard_data_obj.precomputed_analytics:
        logger.info(
            f"Populated DASHBOARD_CONTEXT with precomputed analytics. "
            f"Number of features in analytics: {len(app_context_ref.precomputed_analytics.get('features', []))}"
        )
    else:
        logger.warning("No data source available - dashboard may not function properly")
    
    if args.enable_dynamic_tooltips:
        logger.info("Dynamic tooltips enabled - models loaded and ready")
    else:
        logger.info("Dynamic tooltips disabled - using only precomputed data")
    
    logger.info("Starting Dash dashboard server...")
    app.run(host=args.host, port=args.port, debug=args.debug)

if __name__ == "__main__":
    main()
