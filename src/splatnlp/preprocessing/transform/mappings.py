from functools import lru_cache

import requests

API_REF_URL = "https://stat.ink/api/v3/weapon"
TRANSLATION_URL = "https://splat.top/api/game_translation"
FINAL_REF_URL = "https://splat.top/api/weapon_info"
BASE_IMAGE_URL = (
    "https://splat-top.nyc3.cdn.digitaloceanspaces.com/assets/weapon_flat/"
    "Path_Wst_%s.png"
)


@lru_cache(maxsize=None)
def generate_maps() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Generate mappings for weapon data.

    This function fetches data from various APIs and generates three mappings:
    1. key_to_id: Maps stat.ink weapon keys to their corresponding IDs.
    2. id_to_name: Maps weapon IDs to their names.
    3. id_to_url: Maps weapon IDs to their image URLs.

    Returns:
        tuple[dict[str, str], dict[str, str], dict[str, str]]: A tuple
        containing three dictionaries: key_to_id, id_to_name, and id_to_url.
    """
    api_ref = requests.get(API_REF_URL).json()
    translation = requests.get(TRANSLATION_URL).json()
    final_ref = requests.get(FINAL_REF_URL).json()

    name_to_ref = {
        value: key
        for key, value in translation["USen"]["WeaponName_Main"].items()
        if any(
            key.endswith(f"_{suffix}")
            for suffix in ["00", "01", "O", "H", "Oct"]
        )
    }

    ref_to_id = {
        value["class"] + "_" + value["kit"]: key
        for key, value in final_ref.items()
    }

    reference_kits = {
        str(key): str(value["reference_id"]) for key, value in final_ref.items()
    }

    key_to_id = {
        weapon["key"]: reference_kits.get(
            ref_to_id.get(name_to_ref.get(weapon["name"]["en_US"], ""), ""), ""
        )
        for weapon in api_ref
    }

    id_to_name = {
        ref_to_id[value]: key
        for key, value in name_to_ref.items()
        if value in ref_to_id
    }

    id_to_url = {
        key: BASE_IMAGE_URL % (value["class"] + "_" + value["kit"])
        for key, value in final_ref.items()
    }

    return key_to_id, id_to_name, id_to_url


@lru_cache(maxsize=None)
def get_all_ids() -> list[str]:
    """Get all weapon IDs.

    Returns:
        list[str]: A list of all weapon IDs.
    """
    weapon_ref = requests.get(FINAL_REF_URL).json()
    return list(
        {
            k: v for k, v in weapon_ref.items() if k == str(v["reference_id"])
        }.keys()
    )
