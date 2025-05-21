from dash import html, dcc, callback, Input, Output, State
import dash_ag_grid as dag
import numpy as np
import pandas as pd

# App context will be monkey-patched by the run script
# DASHBOARD_CONTEXT = None

top_examples_component = html.Div(id="top-examples-content", children=[
    html.H4("Top Activating Examples for SAE Feature", className="mb-3"),
    dcc.Loading(
        id="loading-top-examples",
        type="default",
        children=dag.AgGrid(
            id="top-examples-grid",
            rowData=[],
            columnDefs=[],
            defaultColDef={"sortable": True, "resizable": True, "filter": True, "minWidth": 150, "wrapText": True, "autoHeight": True},
            dashGridOptions={"domLayout": "autoHeight"},
            style={"width": "100%"}
        ),
        className="mb-2"
    ),
    html.P(id="top-examples-error-message", style={"color": "red"})
], className="mb-4")

def get_top_k_predictions(logits, inv_vocab, k=5):
    if logits is None:
        return "Logits not available"
    
    if not isinstance(logits, np.ndarray):
        try:
            logits = np.array(logits)
        except Exception:
            return "Logits could not be converted to array"

    if logits.ndim == 0:
        logits = np.array([logits.item()])
    elif logits.ndim > 1:
        squeezed_logits = np.squeeze(logits)
        if squeezed_logits.ndim == 1:
            logits = squeezed_logits
        else:
            return f"Logits have an unhandled shape: {logits.shape}"

    actual_k = min(k, len(logits))
    if actual_k == 0:
        return "No logits to process"

    top_k_indices = np.argsort(logits)[-actual_k:][::-1]
    predictions = []
    for idx in top_k_indices:
        token_name = inv_vocab.get(str(idx), inv_vocab.get(int(idx), f"Token_ID_{idx}"))
        score = logits[idx]
        predictions.append(f"{token_name} ({score:.2f})")
    return ", ".join(predictions)

@callback(
    [Output("top-examples-grid", "rowData"),
     Output("top-examples-grid", "columnDefs"),
     Output("top-examples-error-message", "children")],
    [Input("feature-dropdown", "value")],
)
def update_top_examples_grid(selected_feature_id):
    from splatnlp.dashboard.app import DASHBOARD_CONTEXT

    if selected_feature_id is None:
        return [], [{"field": "Message", "valueGetter": "'Select an SAE feature.'"}], ""
    if DASHBOARD_CONTEXT is None:
        return [], [{"field": "Error", "valueGetter": "'Dashboard context not loaded.'"}], "Error: DASHBOARD_CONTEXT is None. Ensure data is loaded."

    error_message = ""
    try:
        all_sae_acts = DASHBOARD_CONTEXT.all_sae_hidden_activations
        analysis_records_list = DASHBOARD_CONTEXT.analysis_df_records
        inv_vocab = DASHBOARD_CONTEXT.inv_vocab
        inv_weapon_vocab = DASHBOARD_CONTEXT.inv_weapon_vocab

        # Basic type checks for critical data
        if not isinstance(all_sae_acts, np.ndarray):
            raise ValueError("SAE activations (all_sae_hidden_activations) are not a NumPy array.")
        if not isinstance(analysis_records_list, list):
            raise ValueError("Analysis records (analysis_df_records) are not a list.")
        if analysis_records_list and not isinstance(analysis_records_list[0], dict):
            raise ValueError("Elements of analysis_df_records are not dictionaries.")
        if not isinstance(inv_vocab, dict):
            raise ValueError("Inverse vocabulary (inv_vocab) is not a dictionary.")
        if not isinstance(inv_weapon_vocab, dict):
            raise ValueError("Inverse weapon vocabulary (inv_weapon_vocab) is not a dictionary.")
            
        if selected_feature_id >= all_sae_acts.shape[1]:
            err_msg = f"Error: Feature ID {selected_feature_id} is out of range for SAE activations (max: {all_sae_acts.shape[1]-1})."
            return [], [{"field": "Error", "valueGetter": f"'{err_msg}'"}], err_msg

        feature_activations = all_sae_acts[:, selected_feature_id]
        
        top_n = 20
        num_examples = len(feature_activations)
        actual_top_n = min(top_n, num_examples)
        
        if actual_top_n == 0:
            return [], [{"field": "Message", "valueGetter": "'No examples to display.'"}], "No examples found."

        sorted_indices_all = np.argsort(feature_activations)
        top_indices = sorted_indices_all[-actual_top_n:][::-1]

        grid_data = []
        for rank, example_idx in enumerate(top_indices):
            example_idx_int = int(example_idx)

            if example_idx_int >= len(analysis_records_list):
                error_message += f"Warning: Example index {example_idx_int} out of bounds for analysis_records_list (len: {len(analysis_records_list)}). Skipping. "
                continue

            record = analysis_records_list[example_idx_int]
            
            ability_token_ids = record.get("ability_input_tokens", [])
            weapon_id_token = record.get("weapon_id_token")
            sae_feature_activation_value = feature_activations[example_idx_int]
            model_logits = record.get("model_logits")

            if not isinstance(ability_token_ids, list):
                ability_token_ids = [ability_token_ids] if pd.notna(ability_token_ids) else []


            ability_names = [inv_vocab.get(str(tid), inv_vocab.get(int(tid), f"ID_{tid}")) for tid in ability_token_ids]
            weapon_name = inv_weapon_vocab.get(str(weapon_id_token), inv_weapon_vocab.get(int(weapon_id_token), f"WeaponID_{weapon_id_token}"))

            top_preds_str = get_top_k_predictions(model_logits, inv_vocab, k=5)

            grid_data.append({
                "Rank": rank + 1,
                "Weapon": weapon_name,
                "Input Abilities": ", ".join(ability_names) if ability_names else "N/A",
                "SAE Feature Activation": f"{sae_feature_activation_value:.4f}",
                "Top Predicted Abilities": top_preds_str,
                "Original Index": example_idx_int
            })
        
        column_defs = [
            {"field": "Rank", "maxWidth": 80, "sortable": True},
            {"field": "Weapon", "sortable": True, "filter": True},
            {"field": "Input Abilities", "wrapText": True, "autoHeight": True, "flex": 2, "sortable": False},
            {"field": "SAE Feature Activation", "sortable": True,
             "valueFormatter": {"function": "params.value"},
             "cellDataType": "number"},
            {"field": "Top Predicted Abilities", "wrapText": True, "autoHeight": True, "flex": 2, "sortable": False},
            {"field": "Original Index", "maxWidth": 120, "sortable": True, "cellDataType": "number"},
        ]
        
        if not grid_data and not error_message:
            error_message = "No examples found for this feature after filtering."
            if not error_message:
                 return [], [{"field": "Message", "valueGetter": "'No examples found for this feature.'"}], ""
            
        return grid_data, column_defs, error_message

    except Exception as e:
        err_msg_user = f"An error occurred: {str(e)}"
        return [], [{"field": "Error", "valueGetter": f"'{err_msg_user}'"}], err_msg_user
