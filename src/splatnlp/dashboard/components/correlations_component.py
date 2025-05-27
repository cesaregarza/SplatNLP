from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import h5py
import joblib
from dash import Input, Output, State, callback, dcc, html
from scipy.stats import pearsonr

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None

# Global cache for precomputed correlations
_precomputed_correlations = None


def load_precomputed_correlations():
    """Load precomputed correlations if available."""
    global _precomputed_correlations
    
    if _precomputed_correlations is not None:
        return _precomputed_correlations
    
    # Try to load precomputed correlations
    h5_path = Path("/root/dev/SplatNLP/correlations_sae.h5")
    joblib_path = Path("/root/dev/SplatNLP/correlations_sae.joblib")
    
    if h5_path.exists():
        try:
            # Load from HDF5 for efficiency
            _precomputed_correlations = {}
            with h5py.File(h5_path, 'r') as f:
                _precomputed_correlations['indices'] = f['correlation_indices'][:]
                _precomputed_correlations['values'] = f['correlation_values'][:]
                _precomputed_correlations['n_features'] = f.attrs['n_features']
                _precomputed_correlations['top_k'] = f.attrs['top_k']
                _precomputed_correlations['min_correlation'] = f.attrs['min_correlation']
            return _precomputed_correlations
        except Exception as e:
            print(f"Error loading precomputed correlations from HDF5: {e}")
    
    elif joblib_path.exists():
        try:
            # Fallback to joblib
            data = joblib.load(joblib_path)
            _precomputed_correlations = {
                'indices': data['correlation_indices'],
                'values': data['correlation_values'],
                'n_features': data['n_features'],
                'top_k': data['top_k'],
                'min_correlation': data['min_correlation']
            }
            return _precomputed_correlations
        except Exception as e:
            print(f"Error loading precomputed correlations from joblib: {e}")
    
    return None

correlations_component = html.Div(
    id="correlations-content",
    children=[
        html.H4("Feature Correlations", className="mb-3"),
        dcc.Loading(
            id="loading-correlations",
            type="default",
            children=[
                html.Div(
                    id="sae-feature-correlations-display", className="mb-3"
                ),
                html.Hr(),
                html.Div(id="token-logit-correlations-display"),
            ],
            className="mb-2",
        ),
        html.P(id="correlations-error-message", style={"color": "red"}),
    ],
    className="mb-4",
)


def calculate_top_correlated_features(
    all_sae_acts, selected_feature_id, top_n=5
):
    # First try to use precomputed correlations
    precomputed = load_precomputed_correlations()
    if precomputed is not None:
        try:
            # Get precomputed correlations for this feature
            if selected_feature_id < precomputed['n_features']:
                indices = precomputed['indices'][selected_feature_id]
                values = precomputed['values'][selected_feature_id]
                
                # Filter valid correlations (indices >= 0)
                valid_mask = indices >= 0
                valid_indices = indices[valid_mask]
                valid_values = values[valid_mask]
                
                # Create correlation list
                correlations = []
                for idx, val in zip(valid_indices[:top_n], valid_values[:top_n]):
                    correlations.append({
                        "feature_id": int(idx),
                        "correlation": float(val)
                    })
                
                return correlations, None
        except Exception as e:
            print(f"Error using precomputed correlations: {e}")
            # Fall back to computing on the fly
    
    # On-the-fly calculation part removed. If precomputed file cache is not found,
    # this function will return (None, "Precomputed correlation file not found and on-the-fly calculation disabled.")
    # The main callback will then rely on db_context.
    return None, "Precomputed correlation file not found; on-the-fly calculation has been disabled."

# The function calculate_top_token_logit_correlations was removed as it's no longer used by 
# update_correlations_display when db_context is available. Logit influences are fetched from DB.

@callback(
    [
        Output("sae-feature-correlations-display", "children"),
        Output("token-logit-correlations-display", "children"),
        Output("correlations-error-message", "children"),
    ],
    [Input("feature-dropdown", "value")],
    # State("feature-dropdown", "value") # Not strictly needed if context is accessed via import
)
def update_correlations_display(selected_feature_id):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT
    import logging # Ensure logger is available if used
    logger = logging.getLogger(__name__)


    if selected_feature_id is None:
        return [], [], "Select an SAE feature to view correlations."
    
    if DASHBOARD_CONTEXT is None or not hasattr(DASHBOARD_CONTEXT, 'db_context') or DASHBOARD_CONTEXT.db_context is None:
        # This check ensures db_context exists. If not, it's a setup issue from cli.py for the 'run' command.
        logger.warning("CorrelationsComponent: Dashboard context or DB context not available.")
        return [], [], "Error: Database context not available. Ensure data is loaded correctly."

    db_context = DASHBOARD_CONTEXT.db_context
    # inv_vocab is still needed if get_logit_influences returns token_ids that need mapping,
    # but the current db_manager.get_logit_influences already returns token_name.
    # inv_vocab = DASHBOARD_CONTEXT.inv_vocab 

    error_message_parts = []
    sae_corr_display_children = []
    token_logit_influences_display_children = [] # Renamed for clarity

    try:
        # 1. Correlated SAE Features from DB
        # First, try the hardcoded file cache, then db_context if that's the desired logic.
        # For this refactor, we prioritize db_context.
        # The `load_precomputed_correlations()` attempts to load from specific file paths.
        # We can retain this as a potential separate source if needed, but primary data should be from DB.
        
        top_sae_features_from_db = db_context.get_feature_correlations(selected_feature_id, limit=5)
        
        sae_corr_display_children.append(html.H5("Top Correlated SAE Features (from DB)", className="mb-2"))
        if top_sae_features_from_db:
            for item in top_sae_features_from_db:
                # Ensure keys match what db_context.get_feature_correlations returns.
                # It returns: {'feature_id': feature_b, 'correlation': correlation}
                sae_corr_display_children.append(
                    html.P(
                        f"Feature {item['feature_id']}: {item['correlation']:.3f}",
                        className="ms-3",
                    )
                )
        else:
            sae_corr_display_children.append(html.P("No SAE feature correlations found in database."))

        # 2. Top Logit Influences from DB
        # This replaces the old "Token-Logit Correlations" which were calculated on the fly.
        # The database stores precomputed logit influences.
        # The limit in get_logit_influences applies per type (positive/negative)
        top_logit_influences_data = db_context.get_logit_influences(selected_feature_id, limit=5) 

        token_logit_influences_display_children.append(html.H5("Top Logit Influences (from DB)", className="mb-2"))
        
        positive_influences = top_logit_influences_data.get("positive", [])
        negative_influences = top_logit_influences_data.get("negative", [])

        if positive_influences or negative_influences:
            if positive_influences:
                token_logit_influences_display_children.append(html.Strong("Positive Influences:", className="ms-3"))
                for item in positive_influences:
                    token_logit_influences_display_children.append(
                        html.P(
                            f"Token '{item.get('token_name', 'N/A')}' (ID: {item.get('token_id', 'N/A')}): Influence {item.get('influence_value', 0.0):.3f} (Rank: {item.get('rank', 'N/A')})",
                            className="ms-4",
                        )
                    )
            if negative_influences:
                token_logit_influences_display_children.append(html.Strong("Negative Influences:", className="ms-3 mt-2"))
                for item in negative_influences:
                    token_logit_influences_display_children.append(
                        html.P(
                            f"Token '{item.get('token_name', 'N/A')}' (ID: {item.get('token_id', 'N/A')}): Influence {item.get('influence_value', 0.0):.3f} (Rank: {item.get('rank', 'N/A')})",
                            className="ms-4",
                        )
                    )
        else:
            token_logit_influences_display_children.append(html.P("No logit influence data found in database for this feature.", className="ms-3"))

        final_error_message = " | ".join(error_message_parts) if error_message_parts else ""
        return sae_corr_display_children, token_logit_influences_display_children, final_error_message

    except Exception as e:
        logger.error(f"Error in update_correlations_display while using db_context: {e}", exc_info=True)
        return [], [], f"An unexpected error occurred: {str(e)}"
