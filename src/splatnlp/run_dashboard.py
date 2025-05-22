import os
import sys
from pathlib import Path
from typing import Optional

import h5py
import numpy as np
import pandas as pd
import requests
import torch

from splatnlp.dashboard.app import DASHBOARD_CONTEXT, app

# Add the project root to the Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from splatnlp.data.tokenizer import SplatTokenizer
from splatnlp.models.sae import SAE
from splatnlp.preprocessing.weapon_data import fetch_weapon_data, build_weapon_maps


# def fetch_weapon_data() -> dict[str, str]:
#     """Fetch and process weapon data from the Splatoon API."""
#     TRANSLATION_URL = "https://splat.top/api/game_translation"
#     FINAL_REF_URL = "https://splat.top/api/weapon_info"
#
#     INFO_JSON = requests.get(FINAL_REF_URL).json()
#     TRANSLATION_JSON = requests.get(TRANSLATION_URL).json()["USen"][
#         "WeaponName_Main"
#     ]
#
#     NAME_TO_REF = {
#         value: key
#         for key, value in TRANSLATION_JSON.items()
#         if any(
#             key.endswith(f"_{suffix}")
#             for suffix in ["00", "01", "O", "H", "Oct"]
#         )
#     }
#
#     REF_TO_ID = {
#         value["class"] + "_" + value["kit"]: key
#         for key, value in INFO_JSON.items()
#     }
#
#     NAME_TO_DATA = {
#         key: INFO_JSON[REF_TO_ID[NAME_TO_REF[key]]]
#         for key, value in TRANSLATION_JSON.items()
#         if value in REF_TO_ID
#     }
#
#     JSON_NAME_TO_ID = {
#         key: REF_TO_ID[NAME_TO_REF[key]]
#         for key in NAME_TO_REF
#         if NAME_TO_REF[key] in REF_TO_ID
#     }
#     JSON_ID_TO_NAME = {
#         f"weapon_id_{value}": key for key, value in JSON_NAME_TO_ID.items()
#     }
#
#     return JSON_ID_TO_NAME


def main() -> None:
    # Set up paths
    base_dir = Path("/root/dev/SplatNLP")
    model_path = base_dir / "models" / "sae_model.pt"
    activations_cache_path = (
        base_dir / "data" / "processed" / "activations_cache.h5"
    )
    token_acts_cache_path = (
        base_dir / "data" / "processed" / "token_acts_cache.h5"
    )
    analysis_df_path = base_dir / "data" / "processed" / "analysis_df.parquet"

    # Load the SAE model
    print("Loading SAE model...")
    sae_model = SAE.load(model_path)
    sae_model.eval()

    # Load the tokenizer
    print("Loading tokenizer...")
    tokenizer = SplatTokenizer()

    # Load the activations cache
    print("Loading activations cache...")
    with h5py.File(activations_cache_path, "r") as f:
        all_sae_hidden_activations = f["activations"][:]

    # Load token activations if available
    print("Loading token activations...")
    token_activations_accessor: Optional[h5py.File] = None
    if os.path.exists(token_acts_cache_path):
        token_activations_accessor = h5py.File(token_acts_cache_path, "r")
        print("Token activations loaded successfully.")
    else:
        print(
            "Warning: Token activations cache not found at",
            token_acts_cache_path,
        )

    # Load analysis DataFrame
    print("Loading analysis DataFrame...")
    analysis_df = pd.read_parquet(analysis_df_path)

    # Fetch weapon data
    print("Fetching weapon data using splatnlp.preprocessing.weapon_data...")
    info_json, translation_json = fetch_weapon_data()

    # Build weapon maps using the proper function
    print("Building weapon maps using splatnlp.preprocessing.weapon_data...")
    name_to_ref_map, ref_to_id_map, name_to_data_map, id_to_name_map = build_weapon_maps(info_json, translation_json)

    # Set up the dashboard context
    print("Setting up dashboard context...")
    DASHBOARD_CONTEXT.sae_model = sae_model
    DASHBOARD_CONTEXT.tokenizer = tokenizer
    DASHBOARD_CONTEXT.all_sae_hidden_activations = all_sae_hidden_activations
    DASHBOARD_CONTEXT.token_activations_accessor = token_activations_accessor
    DASHBOARD_CONTEXT.analysis_df_records = analysis_df
    DASHBOARD_CONTEXT.inv_vocab = tokenizer.id_to_token
    DASHBOARD_CONTEXT.inv_weapon_vocab = id_to_name_map # Use the new map
    # DASHBOARD_CONTEXT.json_weapon_id_to_name = json_weapon_id_to_name # This is now superseded by inv_weapon_vocab

    # Run the dashboard
    print("Starting dashboard server...")
    app.run_server(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
