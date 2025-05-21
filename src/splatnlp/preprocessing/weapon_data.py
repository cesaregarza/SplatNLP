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
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict]]:
    """
    Build weapon name and data mappings from API responses.

    Args:
        info_json: Weapon info JSON from API
        translation_json: Weapon translations JSON from API

    Returns:
        Tuple of (name_to_ref, ref_to_id, name_to_data) mappings
    """
    # Build name -> reference mapping
    name_to_ref = {
        value: key
        for key, value in translation_json.items()
        if any(
            key.endswith(f"_{suffix}")
            for suffix in ["00", "01", "O", "H", "Oct"]
        )
    }

    # Build reference -> ID mapping
    ref_to_id = {
        value["class"] + "_" + value["kit"]: key
        for key, value in info_json.items()
    }

    # Build name -> data mapping
    name_to_id = {
        key: ref_to_id[NAME_TO_REF[key]]
        for key in NAME_TO_REF
        if NAME_TO_REF[key] in REF_TO_ID
    }

    name_to_data = {
        key: INFO_JSON[str(value)]
        for key, value in name_to_id.items()
        if value in INFO_JSON
    }

    # Filter data to just the fields we care about
    name_to_data = {
        key: {
            k: v
            for k, v in value.items()
            if k in ["sub", "special", "class", "reference_kit"]
        }
        for key, value in name_to_data.items()
    }

    return name_to_ref, ref_to_id, name_to_data
