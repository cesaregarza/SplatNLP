import re
from functools import lru_cache
from pathlib import Path

from splatnlp.preprocessing.transform.mappings import generate_maps

SPECIAL = {
    "ink_recovery_up": "iru",
    "ink_resistance_up": "res",
    "sub_power_up": "bpu",
}

_WORD_NUM_RE = re.compile(r"^([a-z_]+?)(?:_(\d+))?$")


def abbrev(token: str) -> str:
    """Abbreviate a token into a shortened form.

    Examples:
        quick_respawn_29 → qr29
        ink_recovery_up → iru

    Args:
        token (str): The token string to abbreviate.

    Returns:
        str: Abbreviated form of the token.
    """
    m = _WORD_NUM_RE.match(token)
    if not m:
        return token
    base, lvl = m.groups()
    short = SPECIAL.get(base) or "".join(w[0] for w in base.split("_"))
    return f"{short}{lvl or ''}"


def shorten(tokens: list[str], keep: int = 3, sep: str = ", ") -> str:
    """Abbreviate a list of tokens and retain only the first few entries.

    Args:
        tokens (list[str]): List of token strings to abbreviate.
        keep (int, optional): Number of tokens from the start of the list to
            retain. Defaults to 3.
        sep (str, optional): Separator to use between abbreviated tokens.
            Defaults to ", " (comma and space).

    Returns:
        str: String of abbreviated tokens, separated by `sep`.
    """
    return sep.join(abbrev(t) for t in tokens[:keep])


@lru_cache(maxsize=1)
def _build_weapon_maps(vocab_dir: Path):
    """Build and cache weapon ID mapping dictionaries from vocabulary files.

    Args:
        vocab_dir (Path): Directory containing the weapon vocabulary JSON file.

    Returns:
        tuple: A tuple containing four dictionaries:
            - name_to_id (dict): Mapping from weapon names to IDs.
            - id_to_name (dict): Mapping from IDs to weapon names.
            - id_to_url (dict): Mapping from IDs to URLs associated with
                weapons.
            - internal_to_name (dict): Mapping from internal weapon vocabulary
                identifiers to weapon names.
    """
    _, id_to_name, id_to_url = generate_maps()
    from splatnlp.embeddings.load import load_vocab_json

    weapon_vocab = load_vocab_json(vocab_dir)
    name_to_id = {v: int(k) for k, v in id_to_name.items()}
    name_to_internal = {
        n: weapon_vocab.get(f"weapon_id_{wid}")
        for n, wid in name_to_id.items()
        if f"weapon_id_{wid}" in weapon_vocab
    }
    internal_to_name = {v: k for k, v in name_to_internal.items()}
    return name_to_id, id_to_name, id_to_url, internal_to_name
