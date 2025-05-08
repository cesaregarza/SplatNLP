import pytest

from splatnlp.utils.constants import (
    CLOTHING_ABILITIES,
    HEADGEAR_ABILITIES,
    MAIN_ONLY_ABILITIES,
    SHOES_ABILITIES,
    STANDARD_ABILITIES,
)
from splatnlp.utils.reconstruct.classes import AbilityToken, Build


def test_ability_token_creation():
    # Test main-only ability
    token = AbilityToken.from_vocab_entry("ninja_squid")
    assert token.name == "ninja_squid"
    assert token.family == "ninja_squid"
    assert token.min_ap == 10
    assert token.main_only is True

    # Test standard ability with AP
    token = AbilityToken.from_vocab_entry("ink_saver_main_3")
    assert token.name == "ink_saver_main_3"
    assert token.family == "ink_saver_main"
    assert token.min_ap == 3
    assert token.main_only is False


def test_ability_token_repr():
    # Test main-only ability representation
    token = AbilityToken(
        name="ninja_squid", family="ninja_squid", min_ap=0, main_only=True
    )
    assert repr(token) == "ninja_squid"

    # Test standard ability representation
    token = AbilityToken(
        name="ink_saver_main_3",
        family="ink_saver_main",
        min_ap=3,
        main_only=False,
    )
    assert repr(token) == "ism3(3)"


def test_ability_token_invalid():
    with pytest.raises(
        ValueError, match="Token 'invalid_token' not recognised"
    ):
        AbilityToken.from_vocab_entry("invalid_token")


def test_build_initialization():
    # Test valid initialization
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "ink_saver_main",
            "shoes": None,
        },
        subs={"ink_saver_main": 2, "swim_speed_up": 1},
    )
    assert build.mains == {
        "head": "comeback",
        "clothes": "ink_saver_main",
        "shoes": None,
    }
    assert build.subs == {"ink_saver_main": 2, "swim_speed_up": 1}
    assert build.total_ap == 29

    # Test invalid initialization - missing required slots
    with pytest.raises(
        ValueError, match="Mains dictionary must contain exactly 3 keys"
    ):
        Build(
            mains={"head": "comeback", "clothes": None},  # Missing shoes
            subs={},
        )

    # Test invalid initialization - extra slots
    with pytest.raises(
        ValueError, match="Mains dictionary must contain exactly 3 keys"
    ):
        Build(
            mains={
                "head": "comeback",
                "clothes": None,
                "shoes": None,
                "extra": None,
            },
            subs={},
        )


def test_build_disallowed_abilities_with_main_only():
    build = Build(
        mains={"head": "comeback", "clothes": None, "shoes": None},
        subs={},
    )
    disallowed = build.disallowed_abilities()
    assert "comeback" not in disallowed
    assert "tenacity" in disallowed
    assert "ninja_squid" not in disallowed
    assert all(
        ability in disallowed
        for ability in HEADGEAR_ABILITIES
        if ability != "comeback"
    )


def test_build_disallowed_abilities_with_full_gear():
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "haunt",
            "shoes": "drop_roller",
        },
        subs={ability: 1 for ability in STANDARD_ABILITIES[:9]},
    )
    disallowed = build.disallowed_abilities()
    assert len(disallowed) > 0
    assert all(
        ability in disallowed
        for ability in STANDARD_ABILITIES
        if ability not in build.mains.values() and ability not in build.subs
    )
    assert not any(
        ability in (*build.mains.values(), *build.subs.keys())
        for ability in disallowed
    )


def test_build_disallowed_abilities_with_nearly_full_gear():
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "haunt",
            "shoes": "drop_roller",
        },
        subs={ability: 1 for ability in STANDARD_ABILITIES[:8]},
    )
    disallowed = build.disallowed_abilities()
    assert len(disallowed) > 0
    assert not all(
        ability in disallowed
        for ability in STANDARD_ABILITIES
        if ability not in build.mains.values() and ability not in build.subs
    )
    assert not any(
        ability in (*build.mains.values(), *build.subs.keys())
        for ability in disallowed
    )


def test_build_with_clothing_abilities():
    build = Build(
        mains={"head": None, "clothes": "haunt", "shoes": None},
        subs={},
    )
    disallowed = build.disallowed_abilities()
    assert "haunt" not in disallowed
    assert all(
        ability in disallowed
        for ability in CLOTHING_ABILITIES
        if ability != "haunt"
    )


def test_build_with_shoes_abilities():
    build = Build(
        mains={"head": None, "clothes": None, "shoes": "object_shredder"},
        subs={},
    )
    disallowed = build.disallowed_abilities()
    assert "object_shredder" not in disallowed
    assert all(
        ability in disallowed
        for ability in SHOES_ABILITIES
        if ability != "object_shredder"
    )


def test_build_total_ap_calculation():
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "haunt",
            "shoes": "ink_saver_main",
        },
        subs={"ink_saver_main": 2, "swim_speed_up": 1},
    )
    assert build.total_ap == 39


def test_build_achieved_ap_calculation():
    # Test with main-only abilities
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "ninja_squid",
            "shoes": "stealth_jump",
        },
        subs={},
    )
    ap = build.achieved_ap
    assert ap["comeback"] == 10
    assert ap["ninja_squid"] == 10
    assert ap["stealth_jump"] == 10
    assert len(ap) == 3

    # Test with standard abilities and subs
    build = Build(
        mains={
            "head": "ink_saver_main",
            "clothes": None,
            "shoes": None,
        },
        subs={"ink_saver_main": 2, "swim_speed_up": 1},
    )
    ap = build.achieved_ap
    assert ap["ink_saver_main"] == 16  # 10 from main + 2*3 from subs
    assert ap["swim_speed_up"] == 3
    assert len(ap) == 2

    # Test with empty slots
    build = Build(
        mains={"head": None, "clothes": None, "shoes": None},
        subs={},
    )
    ap = build.achieved_ap
    assert len(ap) == 0


def test_build_validation():
    # Test valid build with main-only abilities in correct slots
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "ninja_squid",
            "shoes": "stealth_jump",
        },
        subs={},
    )
    assert build._validate() is True

    # Test invalid build - main-only ability in wrong slot
    with pytest.raises(ValueError, match="Invalid build configuration"):
        Build(
            mains={
                "head": "ninja_squid",  # ninja_squid should be on clothes
                "clothes": None,
                "shoes": None,
            },
            subs={},
        )

    # Test invalid build - too many AP
    with pytest.raises(ValueError, match="Invalid build configuration"):
        Build(
            mains={
                "head": "ink_saver_main",
                "clothes": "ink_saver_main",
                "shoes": "ink_saver_main",
            },
            subs={"ink_saver_main": 9, "swim_speed_up": 1},
        )

    # Test invalid build - too many subs
    with pytest.raises(ValueError, match="Invalid build configuration"):
        Build(
            mains={"head": None, "clothes": None, "shoes": None},
            subs={"ink_saver_main": 10},
        )


def test_build_disallowed_abilities():
    # Test with main-only abilities
    build = Build(
        mains={
            "head": "comeback",
            "clothes": None,
            "shoes": None,
        },
        subs={},
    )
    disallowed = build.disallowed_abilities()
    assert "comeback" not in disallowed
    assert "last_ditch_effort" in disallowed  # Other headgear ability
    assert "ninja_squid" not in disallowed
    assert all(
        ability in disallowed
        for ability in HEADGEAR_ABILITIES
        if ability != "comeback"
    )

    # Test with full build (3 mains + 9 subs)
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "ninja_squid",
            "shoes": "stealth_jump",
        },
        subs={ability: 3 for ability in STANDARD_ABILITIES[:3]},
    )
    disallowed = build.disallowed_abilities()
    assert len(disallowed) > 0
    assert all(
        ability in disallowed
        for ability in STANDARD_ABILITIES
        if ability not in build.subs
    )
    assert all(
        ability in disallowed
        for ability in MAIN_ONLY_ABILITIES
        if ability not in build.mains.values()
    )

    # Test with nearly full build
    build = Build(
        mains={
            "head": "comeback",
            "clothes": "ninja_squid",
            "shoes": "stealth_jump",
        },
        subs={ability: 2 for ability in STANDARD_ABILITIES[:3]},
    )
    disallowed = build.disallowed_abilities()
    assert len(disallowed) > 0
    assert not all(
        ability in disallowed
        for ability in STANDARD_ABILITIES
        if ability not in build.subs
    )
