from dash import html, dcc, callback, Input, Output, State
import numpy as np
import pandas as pd
import random

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None

def format_example_for_interval(record, inv_vocab, inv_weapon_vocab, feature_activation_value, model_logits):
    if record is None:
        return html.P("Example data not found.")

    ability_token_ids = record.get("ability_input_tokens", [])
    weapon_id_token = record.get("weapon_id_token")

    if not isinstance(ability_token_ids, list):
        ability_token_ids = [ability_token_ids] if pd.notna(ability_token_ids) else []


    ability_names = [inv_vocab.get(str(tid), inv_vocab.get(int(tid), f"ID_{tid}")) for tid in ability_token_ids]
    weapon_name = inv_weapon_vocab.get(str(weapon_id_token), inv_weapon_vocab.get(int(weapon_id_token), f"WeaponID_{weapon_id_token}"))
    
    top_pred_str = "N/A"
    if model_logits is not None:
        if not isinstance(model_logits, np.ndarray):
            try:
                model_logits = np.array(model_logits)
            except Exception:
                model_logits = None

        if isinstance(model_logits, np.ndarray) and model_logits.size > 0:
            if model_logits.ndim > 1:
                model_logits = model_logits.squeeze()
            if model_logits.ndim == 1 and model_logits.size > 0:
                 try:
                    top_idx = np.argmax(model_logits)
                    token_name = inv_vocab.get(str(top_idx), inv_vocab.get(int(top_idx), f"Token_ID_{top_idx}"))
                    score = model_logits[top_idx]
                    top_pred_str = f"{token_name} ({score:.2f})"
                 except IndexError:
                    top_pred_str = "Error getting prediction"
            elif model_logits.ndim == 0: 
                 top_idx = model_logits.item()
                 token_name = inv_vocab.get(str(top_idx), inv_vocab.get(int(top_idx), f"Token_ID_{top_idx}"))
                 score = top_idx
                 top_pred_str = f"{token_name} ({score:.2f})"


    return html.Div([
        html.Strong(f"Weapon: {weapon_name}"),
        html.P(f"Inputs: {', '.join(ability_names) if ability_names else 'N/A'}"),
        html.P(f"SAE Feature Activation: {feature_activation_value:.4f}"),
        html.P(f"Top Prediction: {top_pred_str}")
    ], style={"border": "1px solid #eee", "padding": "5px", "margin-bottom": "5px", "font-size": "0.9em"})


intervals_grid_component = html.Div(id="intervals-grid-content", children=[
    html.H4("Subsampled Intervals Grid for SAE Feature Activations", className="mb-3"),
    dcc.Loading(
        id="loading-intervals-grid",
        type="default",
        children=html.Div(id="intervals-grid-display"),
        className="mb-2"
    ),
    html.P(id="intervals-grid-error-message", style={"color": "red"})
], className="mb-4")

@callback(
    [Output("intervals-grid-display", "children"),
     Output("intervals-grid-error-message", "children")],
    [Input("feature-dropdown", "value")],
)
def update_intervals_grid(selected_feature_id):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return [], "Select an SAE feature to view the intervals grid."
    if DASHBOARD_CONTEXT is None:
        return [], "Error: DASHBOARD_CONTEXT is None. Ensure data is loaded."

    error_message = ""
    try:
        all_sae_acts = DASHBOARD_CONTEXT.all_sae_hidden_activations
        analysis_data_source = DASHBOARD_CONTEXT.analysis_df_records
        inv_vocab = DASHBOARD_CONTEXT.inv_vocab
        inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab

        # Validate data types
        if not isinstance(all_sae_acts, np.ndarray):
            raise ValueError("SAE activations (all_sae_hidden_activations) are not a NumPy array.")
        
        if isinstance(analysis_data_source, list) and all(isinstance(item, dict) for item in analysis_data_source):
            analysis_df = pd.DataFrame(analysis_data_source)
        elif isinstance(analysis_data_source, pd.DataFrame):
            analysis_df = analysis_data_source
        else:
            raise ValueError("Analysis records (analysis_df_records) are not a Pandas DataFrame or a convertible list of dicts.")

        if not isinstance(inv_vocab, dict):
            raise ValueError("Inverse vocabulary (inv_vocab) is not a dictionary.")
        if not isinstance(inv_weapon_vocab, dict):
            raise ValueError("Inverse weapon vocabulary (inv_weapon_vocab) is not a dictionary.")

        if selected_feature_id >= all_sae_acts.shape[1]:
            return [], f"Error: Feature ID {selected_feature_id} is out of range for SAE activations (max: {all_sae_acts.shape[1]-1})."

        feature_activations = all_sae_acts[:, selected_feature_id]
        
        min_act = np.min(feature_activations)
        max_act = np.max(feature_activations)

        # Handle case where all activations are the same or very close
        if np.isclose(min_act, max_act) or max_act == min_act:
            max_act = min_act + 1e-6 if min_act == 0 else min_act * (1 + 1e-6) if min_act > 0 else min_act * (1 - 1e-6)
            if np.isclose(min_act, max_act):
                max_act = min_act + 1e-6

        num_intervals = 10
        intervals = np.linspace(min_act, max_act, num_intervals + 1)
        
        if intervals[-1] < max_act:
             intervals[-1] = max_act


        grid_layout_children = []
        examples_per_interval = 3

        for i in range(num_intervals):
            lower_bound = intervals[i]
            upper_bound = intervals[i+1]

            if upper_bound < lower_bound: upper_bound = lower_bound

            if i == num_intervals - 1:
                condition = (feature_activations >= lower_bound) & (feature_activations <= upper_bound)
            else:
                condition = (feature_activations >= lower_bound) & (feature_activations < upper_bound)
            
            indices_in_interval = np.where(condition)[0]
            
            interval_title = f"Interval {i+1}: [{lower_bound:.3f}, {upper_bound:.3f}) - {len(indices_in_interval)} examples"
            if i == num_intervals -1:
                 interval_title = f"Interval {i+1}: [{lower_bound:.3f}, {upper_bound:.3f}] - {len(indices_in_interval)} examples"

            interval_div_children = [html.H6(interval_title)]

            if len(indices_in_interval) > 0:
                num_to_sample = min(len(indices_in_interval), examples_per_interval)
                if num_to_sample > 0:
                    sampled_indices = random.sample(list(indices_in_interval), num_to_sample)
                    
                    for example_idx_np_int in sampled_indices:
                        example_idx = int(example_idx_np_int)
                        if example_idx >= len(analysis_df):
                            interval_div_children.append(html.P(f"Warning: Index {example_idx} out of bounds for analysis_df (len: {len(analysis_df)})."))
                            continue
                        record = analysis_df.iloc[example_idx]
                        activation_value_for_example = feature_activations[example_idx]
                        model_logits_for_example = record.get("model_logits")
                        
                        if isinstance(model_logits_for_example, list):
                            model_logits_for_example = np.array(model_logits_for_example)

                        interval_div_children.append(
                            format_example_for_interval(record, inv_vocab, inv_weapon_vocab, activation_value_for_example, model_logits_for_example)
                        )
            else:
                interval_div_children.append(html.P("No examples in this interval."))
            
            grid_layout_children.append(html.Div(interval_div_children, className="interval-section", style={'margin-bottom': '15px', 'padding': '10px', 'border': '1px solid #ddd', 'border-radius': '5px'}))

        if not grid_layout_children and not error_message:
             error_message = "Could not generate interval grid. Feature might have no activations or too few distinct values."

        return grid_layout_children, error_message

    except Exception as e:
        return [], f"An error occurred in update_intervals_grid: {str(e)}"
