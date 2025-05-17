import pandas as pd

from splatnlp.preprocessing.datasets.tokenize import tokenize
from splatnlp.utils.constants import PAD


def test_tokenize_with_and_without_mappings():
    df = pd.DataFrame(
        {
            "ability_tags": [
                "comeback ninja_squid weapon_id_0",
                "stealth_jump comeback weapon_id_1",
            ],
            "weapon_id": ["0", "1"],
        }
    )

    out_df, ability_to_id, weapon_to_id = tokenize(df)

    assert (
        out_df["ability_tags"]
        .apply(lambda x: all(isinstance(i, int) for i in x))
        .all()
    )
    assert ability_to_id[PAD] == len(ability_to_id) - 1

    out_df2, ability_to_id2, weapon_to_id2 = tokenize(
        df, ability_to_id, weapon_to_id
    )

    assert out_df2.equals(out_df)
    assert ability_to_id2 == ability_to_id
    assert weapon_to_id2 == weapon_to_id
