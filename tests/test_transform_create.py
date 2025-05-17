import pandas as pd
import pytest

from splatnlp.preprocessing.transform.create import (
    add_columns,
    compute_ability_to_id,
    create_weapon_df,
)


def make_raw_df():
    data = {}
    for team in "AB":
        for player in range(1, 5):
            data[f"{team}{player}-weapon"] = [f"{team}{player}_w"]
            data[f"{team}{player}-abilities"] = [f"{team}{player}_abilities"]
    data.update({
        "period": [1],
        "game-ver": ["1.0"],
        "lobby": ["open"],
        "mode": ["sz"],
        "win": ["A"],
    })
    return pd.DataFrame(data)


def test_create_weapon_df_splits_players_and_merges_metadata():
    raw_df = make_raw_df()
    weapon_df = create_weapon_df(raw_df)
    assert len(weapon_df) == 8
    expected_cols = {
        "weapon",
        "abilities",
        "player_no",
        "team",
        "period",
        "game-ver",
        "lobby",
        "mode",
        "win",
    }
    assert expected_cols.issubset(weapon_df.columns)
    row = weapon_df[(weapon_df["team"] == "A") & (weapon_df["player_no"] == 1)].iloc[0]
    assert row["weapon"] == "A1_w"
    assert row["abilities"] == "A1_abilities"
    assert row["period"] == 1
    assert row["game-ver"] == "1.0"


def test_add_columns_creates_expected_fields(monkeypatch):
    raw_df = create_weapon_df(make_raw_df())
    mapping = {f"A{p}_w": str(p) for p in range(1, 5)}
    mapping.update({f"B{p}_w": str(4 + p) for p in range(1, 5)})

    def fake_maps():
        return mapping, {}, {}

    monkeypatch.setattr(
        "splatnlp.preprocessing.transform.create.generate_maps", fake_maps
    )

    df = add_columns(raw_df.copy())
    assert (df["weapon_id"] == df["weapon"].map(mapping)).all()
    for _, row in df.iterrows():
        expected_hash = str(compute_ability_to_id(row["abilities"])) + row["weapon_id"]
        assert row["ability_hash"] == expected_hash
    assert df[df["team"] == "A"]["win"].unique().tolist() == [True]
    assert df[df["team"] == "B"]["win"].unique().tolist() == [False]


def test_compute_ability_to_id_is_deterministic():
    val1 = compute_ability_to_id("abc")
    val2 = compute_ability_to_id("abc")
    assert val1 == val2

