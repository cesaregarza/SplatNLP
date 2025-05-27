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
    
    # Fall back to computing correlations on the fly
    if not isinstance(all_sae_acts, np.ndarray):
        return [], "SAE activations are not a NumPy array."
    if selected_feature_id >= all_sae_acts.shape[1]:
        return (
            [],
            f"Selected feature ID {selected_feature_id} out of range for SAE activations (max: {all_sae_acts.shape[1]-1}).",
        )

    target_feature_vector = all_sae_acts[:, selected_feature_id]
    correlations = []

    for i in range(all_sae_acts.shape[1]):
        if i == selected_feature_id:
            continue
        other_feature_vector = all_sae_acts[:, i]

        # Check for zero variance, pearsonr will error or return NaN
        # Also ensure vectors are long enough for correlation
        if target_feature_vector.size < 2 or other_feature_vector.size < 2:
            corr = 0
        elif (
            np.std(target_feature_vector) == 0
            or np.std(other_feature_vector) == 0
        ):
            corr = 0
        else:
            # pearsonr can fail if inputs are perfectly correlated or anti-correlated in a way that leads to issues,
            # or if there are NaNs/infs not caught by np.std.
            try:
                corr, _ = pearsonr(target_feature_vector, other_feature_vector)
            except ValueError:
                corr = 0
        correlations.append(
            {"feature_id": i, "correlation": corr if np.isfinite(corr) else 0}
        )

    sorted_correlations = sorted(
        correlations, key=lambda x: abs(x["correlation"]), reverse=True
    )
    return sorted_correlations[:top_n], None


def calculate_top_token_logit_correlations(
    all_sae_acts, analysis_df, selected_feature_id, inv_vocab, top_n=10
):
    if not isinstance(all_sae_acts, np.ndarray):
        return [], "SAE activations are not a NumPy array."
    if selected_feature_id >= all_sae_acts.shape[1]:
        return (
            [],
            f"Selected feature ID {selected_feature_id} out of range for SAE activations (max: {all_sae_acts.shape[1]-1}).",
        )

    target_feature_activations = all_sae_acts[:, selected_feature_id]

    if not isinstance(analysis_df, pd.DataFrame):
        return [], "Analysis data is not a Pandas DataFrame."
    if "model_logits" not in analysis_df.columns:
        return [], "model_logits column not found in analysis_df."
    if analysis_df.empty:
        return [], "Analysis data is empty."

    # Convert model_logits column to a 2D NumPy array
    try:
        # Check if elements are already numpy arrays
        if isinstance(analysis_df["model_logits"].iloc[0], np.ndarray):
            all_model_logits = np.stack(analysis_df["model_logits"].values)
        # Check if elements are lists or Series (which can be converted to lists)
        elif isinstance(analysis_df["model_logits"].iloc[0], (list, pd.Series)):
            all_model_logits = np.array(analysis_df["model_logits"].to_list())
        else:
            return (
                [],
                f"Unsupported type for model_logits: {type(analysis_df['model_logits'].iloc[0])}.",
            )
    except Exception as e:
        return [], f"Error converting model_logits to NumPy array: {str(e)}"

    if all_model_logits.ndim != 2:  # Expected shape (num_examples, vocab_size)
        return (
            [],
            f"Expected model_logits to be 2D, but got shape {all_model_logits.shape} after conversion.",
        )
    if all_model_logits.shape[0] != target_feature_activations.shape[0]:
        return (
            [],
            f"Mismatch in number of examples: SAE activations ({target_feature_activations.shape[0]}) vs model_logits ({all_model_logits.shape[0]}).",
        )

    correlations = []
    vocab_size = all_model_logits.shape[1]

    for i in range(vocab_size):
        logit_values_for_token_i = all_model_logits[:, i]

        if (
            target_feature_activations.size < 2
            or logit_values_for_token_i.size < 2
        ):
            corr = 0
        elif (
            np.std(target_feature_activations) == 0
            or np.std(logit_values_for_token_i) == 0
        ):
            corr = 0
        else:
            try:
                corr, _ = pearsonr(
                    target_feature_activations, logit_values_for_token_i
                )
            except ValueError:
                corr = 0

        # Ensure token_id 'i' is correctly used for inv_vocab
        token_name = inv_vocab.get(
            str(i), inv_vocab.get(int(i), f"TokenID_{i}")
        )
        correlations.append(
            {
                "token_name": token_name,
                "token_id": i,
                "correlation": corr if np.isfinite(corr) else 0,
            }
        )

    sorted_correlations = sorted(
        correlations, key=lambda x: abs(x["correlation"]), reverse=True
    )
    return sorted_correlations[:top_n], None


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

    if selected_feature_id is None:
        return [], [], "Select an SAE feature to view correlations."
    if DASHBOARD_CONTEXT is None:
        return (
            [],
            [],
            "Error: DASHBOARD_CONTEXT is None. Ensure data is loaded.",
        )

    error_message_parts = []
    sae_corr_display_children = []  # Changed to avoid conflict with outer var
    token_corr_display_children = []  # Changed to avoid conflict

    try:
        all_sae_acts = DASHBOARD_CONTEXT.all_sae_hidden_activations
        analysis_df = DASHBOARD_CONTEXT.analysis_df_records  # Use analysis_df
        inv_vocab = DASHBOARD_CONTEXT.inv_vocab

        # Safeguard check for analysis_df type
        if not isinstance(analysis_df, pd.DataFrame):
            return (
                [],
                [],
                "Error: analysis_df_records is not a DataFrame as expected.",
            )

        # Other critical data (all_sae_acts, inv_vocab) are assumed to be correct from cli.py
        # or will raise errors in helper functions if types are wrong.

        # 1. Correlated SAE Features
        top_sae_features, err = calculate_top_correlated_features(
            all_sae_acts, selected_feature_id
        )
        if err:
            error_message_parts.append(f"SAE Feature Corr Error: {err}")
        else:
            # Check if using precomputed correlations
            precomputed = load_precomputed_correlations()
            correlation_label = "Top Correlated SAE Features (Pearson"
            if precomputed is not None:
                correlation_label += ", precomputed):"
            else:
                correlation_label += "):"
            
            sae_corr_display_children.append(
                html.H5(
                    correlation_label, className="mb-2"
                )
            )
            if top_sae_features:
                for item in top_sae_features:
                    sae_corr_display_children.append(
                        html.P(
                            f"Feature {item['feature_id']}: {item['correlation']:.3f}",
                            className="ms-3",
                        )
                    )
            else:
                sae_corr_display_children.append(
                    html.P(
                        "No significant SAE feature correlations found or calculable."
                    )
                )

        # 2. Token-Logit Correlations
        # Pass the converted analysis_df to the helper
        top_token_corrs, err = calculate_top_token_logit_correlations(
            all_sae_acts, analysis_df, selected_feature_id, inv_vocab
        )
        if err:
            error_message_parts.append(f"Token-Logit Corr Error: {err}")
        else:
            token_corr_display_children.append(
                html.H5(
                    "Top Correlated Output Tokens (Logit Value vs SAE Feature Activation):",
                    className="mb-2",
                )
            )
            if top_token_corrs:
                for item in top_token_corrs:
                    token_corr_display_children.append(
                        html.P(
                            f"Token '{item['token_name']}' (ID: {item['token_id']}): {item['correlation']:.3f}",
                            className="ms-3",
                        )
                    )
            else:
                token_corr_display_children.append(
                    html.P(
                        "No significant token-logit correlations found or calculable."
                    )
                )

        final_error_message = (
            " | ".join(error_message_parts) if error_message_parts else ""
        )
        return (
            sae_corr_display_children,
            token_corr_display_children,
            final_error_message,
        )

    except Exception as e:
        return (
            [],
            [],
            f"An unexpected error occurred in correlations component: {str(e)}",
        )
