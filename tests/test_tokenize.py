import pandas as pd

from splatnlp.preprocessing.datasets.tokenize import tokenize
from splatnlp.utils.constants import PAD


def make_df_one():
    return pd.DataFrame(
        {
            "ability_tags": ["a b ", "b c "],
            "weapon_id": ["w1", "w2"],
        }
    )


def make_df_two():
    return pd.DataFrame(
        {
            "ability_tags": ["c d ", "a d "],
            "weapon_id": ["w1", "w3"],
        }
    )


def assert_int_lists(series: pd.Series) -> None:
    for value in series:
        assert isinstance(value, list)
        assert all(isinstance(i, int) for i in value)


def test_tokenize_with_and_without_mappings():
    df1 = make_df_one()
    out1, ability_map1, weapon_map1 = tokenize(df1)

    assert_int_lists(out1["ability_tags"])
    assert ability_map1[PAD] == len(ability_map1) - 1
    assert weapon_map1 == {"weapon_id_w1": 0, "weapon_id_w2": 1}

    df2 = make_df_two()
    out2, ability_map2, weapon_map2 = tokenize(df2, ability_map1, weapon_map1)

    assert_int_lists(out2["ability_tags"])
    assert ability_map2[PAD] == len(ability_map2) - 1
    assert ability_map2["d"] == 3
    assert weapon_map2 == {
        "weapon_id_w1": 0,
        "weapon_id_w2": 1,
        "weapon_id_w3": 2,
    }
    assert out2["weapon_id"].tolist() == [0, 2]
