"""Weapon data fetching and mapping utilities."""

from typing import Dict, Tuple

import requests


def fetch_weapon_data() -> Tuple[Dict, Dict]:
    """
    Fetch weapon data from the Splatoon API.

    Returns:
        Tuple of (weapon_info, weapon_translations)
    """
    TRANSLATION_URL = "https://splat.top/api/game_translation"
    FINAL_REF_URL = "https://splat.top/api/weapon_info"

    info_json = requests.get(FINAL_REF_URL).json()
    translation_json = requests.get(TRANSLATION_URL).json()["USen"][
        "WeaponName_Main"
    ]

    return info_json, translation_json


def build_weapon_maps(
    info_json: Dict,
    translation_json: Dict,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict], Dict[str, str]]:
    """
    Build weapon name and data mappings from API responses.

    Args:
        info_json: Weapon info JSON from API. Keys are weapon IDs (e.g., "0", "10").
                   Values are dicts with weapon data (e.g., {"class": "Shooter_Short", "kit": "00_0"}).
        translation_json: Weapon translations JSON from API. Keys are reference strings
                          (e.g., "WeaponName_Main_Shooter_Short_00"), values are translated names.

    Returns:
        Tuple of (name_to_ref_map, ref_to_id_map, name_to_data_map, id_to_name_map) mappings.
    """
    # Build name -> reference mapping (e.g., {"Sploosh-o-matic": "WeaponName_Main_SplashShooter_00"})
    # These are the primary translated names we want to use as keys elsewhere.
    name_to_ref_map = {
        value: key
        for key, value in translation_json.items()
        if any(
            key.endswith(f"_{suffix}")
            # Common suffixes for base weapon kits in translation keys
            for suffix in ["00", "01", "O", "H", "Oct", "XP"] 
        )
    }

    # Build reference string (class + kit) -> ID mapping (e.g., {"Shooter_Short_00_0": "0"})
    # This helps link info_json entries back to a consistent reference format if needed,
    # but primary linking will be done by iterating info_json.
    ref_to_id_map = {
        value.get("class", "") + "_" + value.get("kit", ""): str(key)
        for key, value in info_json.items()
        if value.get("class") and value.get("kit")
    }

    name_to_data_map: Dict[str, Dict] = {}
    id_to_name_map: Dict[str, str] = {}

    # Filter translation_json once for relevant reference keys used for name lookup
    # These are keys from translation_json that typically represent a base weapon.
    ref_keys_in_translation = {
        key: value
        for key, value in translation_json.items()
        # Ensure we are only picking keys that are likely to be main weapon names
        if any(key.startswith(prefix) for prefix in ["WeaponName_Main_", "ExtName_Main_"]) and \
           any(key.endswith(suffix) for suffix in ["00", "01", "O", "H", "Oct", "XP"])
    }
    
    # Iterate through info_json (keyed by weapon_id) to build name_to_data_map and id_to_name_map
    for weapon_id, weapon_data in info_json.items():
        if not isinstance(weapon_data, dict):
            continue # Skip if weapon_data is not a dictionary

        class_name = weapon_data.get("class")
        kit_name_full = weapon_data.get("kit") # e.g., "00_0", "01_0", "Custom_0"
        reference_kit_val = weapon_data.get("reference_kit") # e.g., "WeaponName_Main_Shooter_Short_00"

        translated_name = None

        if class_name and kit_name_full:
            # Try to guess the translation key pattern
            # Pattern 1: Based on class and the first part of kit_name_full
            kit_name_base = kit_name_full.split("_")[0] # "00" from "00_0", "Custom" from "Custom_0"
            
            # Common prefixes for translation keys
            possible_prefixes = ["WeaponName_Main_", "ExtName_Main_"]
            for prefix in possible_prefixes:
                key_guess1 = f"{prefix}{class_name}_{kit_name_base}"
                if key_guess1 in ref_keys_in_translation:
                    translated_name = ref_keys_in_translation[key_guess1]
                    break
            
            if not translated_name and reference_kit_val and reference_kit_val in ref_keys_in_translation:
                # Pattern 2: Using reference_kit directly if it's a valid key in our filtered translations
                translated_name = ref_keys_in_translation[reference_kit_val]

        if translated_name:
            # Ensure weapon_id is stored as string, matching keys in info_json
            current_weapon_id_str = str(weapon_id)
            
            # Populate id_to_name_map
            # Check for conflicts: if a name is already mapped, prefer shorter weapon_id (e.g. "0" over "12345")
            # or if the existing id is a placeholder and the new one is more specific.
            # This handles cases where multiple weapon_ids might resolve to the same translated name due to data variations.
            if translated_name not in name_to_data_map or \
               (current_weapon_id_str not in id_to_name_map.get(translated_name, "") and \
                len(current_weapon_id_str) < len(id_to_name_map.get(translated_name, "Z"*10))): # "Z"*10 as a long string
                 id_to_name_map[current_weapon_id_str] = translated_name
                 name_to_data_map[translated_name] = {
                    k: weapon_data.get(k)
                    for k in ["sub", "special", "class", "reference_kit"]
                 }
        # else:
            # Optionally log or handle cases where a translated name couldn't be found for a weapon_id
            # print(f"Warning: Could not find translated name for weapon_id {weapon_id}")


    # Filter name_to_ref_map to only include names that we successfully found data for
    # This ensures consistency between name_to_ref_map and name_to_data_map keys.
    filtered_name_to_ref_map = {
        name: ref for name, ref in name_to_ref_map.items() if name in name_to_data_map
    }
    
    # Also, ensure id_to_name_map's names are present in the filtered_name_to_ref_map
    # This means that every id in id_to_name_map points to a name that has a valid ref and data.
    final_id_to_name_map = {
        id_val: name for id_val, name in id_to_name_map.items() if name in filtered_name_to_ref_map
    }
    final_name_to_data_map = {
        name: data for name, data in name_to_data_map.items() if name in filtered_name_to_ref_map
    }


    return filtered_name_to_ref_map, ref_to_id_map, final_name_to_data_map, final_id_to_name_map
