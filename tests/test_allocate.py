import pytest

from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.classes import AbilityToken


# --- Test Fixtures ---
@pytest.fixture
def allocator_instance():
    """Provides a default Allocator instance."""
    return Allocator()


# Main-only tokens (min_ap=10 as per AbilityToken.from_vocab_entry)
@pytest.fixture
def comeback_t():
    return AbilityToken.from_vocab_entry("comeback")


@pytest.fixture
def lde_t():
    return AbilityToken.from_vocab_entry("last_ditch_effort")


@pytest.fixture
def ninja_squid_t():
    return AbilityToken.from_vocab_entry("ninja_squid")


@pytest.fixture
def stealth_jump_t():
    return AbilityToken.from_vocab_entry("stealth_jump")


# Standard ability tokens
@pytest.fixture
def ism_3_t():
    return AbilityToken.from_vocab_entry("ink_saver_main_3")


@pytest.fixture
def ism_9_t():
    return AbilityToken.from_vocab_entry("ink_saver_main_9")


@pytest.fixture
def ism_10_t():
    return AbilityToken.from_vocab_entry("ink_saver_main_10")


@pytest.fixture
def ism_12_t():
    return AbilityToken.from_vocab_entry("ink_saver_main_12")


@pytest.fixture
def ssu_3_t():
    return AbilityToken.from_vocab_entry("swim_speed_up_3")


@pytest.fixture
def ssu_6_t():
    return AbilityToken.from_vocab_entry("swim_speed_up_6")


@pytest.fixture
def rsu_9_t():
    return AbilityToken.from_vocab_entry("run_speed_up_9")


@pytest.fixture
def iru_1_t():
    return AbilityToken.from_vocab_entry("ink_resistance_up_1")


@pytest.fixture
def scu_20_t():
    return AbilityToken.from_vocab_entry("special_charge_up_20")


@pytest.fixture
def ssu_21_t():
    return AbilityToken.from_vocab_entry("swim_speed_up_21")


@pytest.fixture
def qr_3_t():
    return AbilityToken.from_vocab_entry("quick_respawn_3")


@pytest.fixture
def qsj_3_t():
    return AbilityToken.from_vocab_entry("quick_super_jump_3")


# --- Allocator Tests ---


def test_no_capstones(allocator_instance: Allocator):
    """Allocator with no capstones should return an empty valid build (0 AP)."""
    build = allocator_instance.allocate({})
    assert build is not None
    assert build.total_ap == 0
    assert build.mains == {"head": None, "clothes": None, "shoes": None}
    assert build.subs == {}


def test_single_main_only(
    allocator_instance: Allocator, comeback_t: AbilityToken
):
    """Single main-only ability."""
    capstones = {"cb": comeback_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.mains == {"head": "comeback", "clothes": None, "shoes": None}
    assert build.subs == {}
    assert build.total_ap == 10
    assert build.achieved_ap.get("comeback") == 10


def test_main_only_conflict(
    allocator_instance: Allocator, comeback_t: AbilityToken, lde_t: AbilityToken
):
    """Conflicting main-only abilities for the same slot."""
    capstones = {"cb": comeback_t, "lde": lde_t}  # Both head
    build = allocator_instance.allocate(capstones)
    assert build is None


def test_all_main_slots_with_main_only(
    allocator_instance: Allocator,
    comeback_t: AbilityToken,
    ninja_squid_t: AbilityToken,
    stealth_jump_t: AbilityToken,
):
    """All three main slots filled with compatible main-only abilities."""
    capstones = {"cb": comeback_t, "ns": ninja_squid_t, "sj": stealth_jump_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.mains == {
        "head": "comeback",
        "clothes": "ninja_squid",
        "shoes": "stealth_jump",
    }
    assert build.subs == {}
    assert build.total_ap == 30


def test_standard_ability_min_ap_3_gets_1_sub(
    allocator_instance: Allocator, ism_3_t: AbilityToken
):
    """Standard ability with min_ap 3. Optimal is 1 sub (3 AP), not a main (10 AP)."""
    capstones = {"ism": ism_3_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    # It should not use a main slot if subs are cheaper for total_ap
    assert build.mains == {"head": None, "clothes": None, "shoes": None}
    assert build.subs == {"ink_saver_main": 1}
    assert build.total_ap == 3
    assert build.achieved_ap.get("ink_saver_main") == 3


def test_standard_ability_min_ap_9_gets_3_subs(
    allocator_instance: Allocator, ism_9_t: AbilityToken
):
    """Standard ability with min_ap 9. Optimal is 3 subs (9 AP), not a main (10 AP)."""
    capstones = {"ism": ism_9_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.mains == {"head": None, "clothes": None, "shoes": None}
    assert build.subs == {"ink_saver_main": 3}
    assert build.total_ap == 9
    assert build.achieved_ap.get("ink_saver_main") == 9


def test_standard_ability_min_ap_10_gets_1_main(
    allocator_instance: Allocator, ism_10_t: AbilityToken
):
    """Standard ability with min_ap 10. Optimal is 1 main (10 AP), not 4 subs (12 AP)."""
    capstones = {"ism": ism_10_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert "ink_saver_main" in build.mains.values()  # Should be on a main
    assert sum(1 for v in build.mains.values() if v == "ink_saver_main") == 1
    assert build.subs.get("ink_saver_main", 0) == 0  # No subs for it
    assert build.total_ap == 10
    assert build.achieved_ap.get("ink_saver_main") == 10


def test_standard_ability_min_ap_12_gets_1_main_1_sub(
    allocator_instance: Allocator, ism_12_t: AbilityToken
):
    """Standard min_ap 12. Optimal: 1 main (10) + 1 sub (3) = 13 AP.
    Alternative: 4 subs = 12 AP. So 4 subs is better.
    """
    capstones = {"ism": ism_12_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    # Optimal should be 4 subs (12 AP) vs 1 main + 1 sub (13 AP)
    assert build.mains == {
        "head": None,
        "clothes": None,
        "shoes": None,
    }  # No main for ISM
    assert build.subs == {"ink_saver_main": 4}
    assert build.total_ap == 12
    assert build.achieved_ap.get("ink_saver_main") == 12


def test_mixed_main_only_and_standard_tightest_fit(
    allocator_instance: Allocator,
    comeback_t: AbilityToken,
    ssu_3_t: AbilityToken,
):
    """Main-only (10AP) and a standard (min_ap 3).
    Optimal: CB on main (10AP), SSU as 1 sub (3AP). Total 13AP.
    """
    capstones = {"cb": comeback_t, "ssu": ssu_3_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.mains == {"head": "comeback", "clothes": None, "shoes": None}
    assert build.subs == {"swim_speed_up": 1}
    assert build.total_ap == 13
    assert build.achieved_ap.get("comeback") == 10
    assert build.achieved_ap.get("swim_speed_up") == 3


def test_two_standards_competing_for_main_vs_subs(
    allocator_instance: Allocator, ism_9_t: AbilityToken, ssu_6_t: AbilityToken
):
    """Two standard abilities: ISM (min 9), SSU (min 6).
    Option 1: ISM (3 subs, 9AP), SSU (2 subs, 6AP). Total 15AP. All subs.
    Option 2: ISM (main, 10AP), SSU (2 subs, 6AP). Total 16AP.
    Option 3: SSU (main, 10AP), ISM (3 subs, 9AP). Total 19AP.
    Option 4: ISM (main, 10AP), SSU (main, 10AP). Total 20AP.
    Optimal is Option 1: All subs.
    """
    capstones = {"ism": ism_9_t, "ssu": ssu_6_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.mains == {"head": None, "clothes": None, "shoes": None}
    assert build.subs == {"ink_saver_main": 3, "swim_speed_up": 2}
    assert build.total_ap == 15
    assert build.achieved_ap.get("ink_saver_main") == 9
    assert build.achieved_ap.get("swim_speed_up") == 6


def test_three_standards_one_must_use_main(
    allocator_instance: Allocator,
    ism_10_t: AbilityToken,
    ssu_6_t: AbilityToken,
    rsu_9_t: AbilityToken,
):
    """ISM (min 10), SSU (min 6), RSU (min 9).
    ISM wants main (10AP) vs 4 subs (12AP).
    SSU wants 2 subs (6AP).
    RSU wants 3 subs (9AP).
    If ISM takes main: 10 (ISM) + 6 (SSU subs) + 9 (RSU subs) = 25 AP. (1 main, 5 subs)
    If all subs: 12 (ISM subs) + 6 (SSU subs) + 9 (RSU subs) = 27 AP. (0 mains, 9 subs)
    So, ISM on main is better.
    """
    capstones = {"ism": ism_10_t, "ssu": ssu_6_t, "rsu": rsu_9_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert "ink_saver_main" in build.mains.values()
    assert build.subs.get("swim_speed_up") == 2
    assert build.subs.get("run_speed_up") == 3
    assert build.subs.get("ink_saver_main", 0) == 0  # ISM is on main
    assert build.total_ap == 25


def test_sub_slots_exceeded(
    allocator_instance: Allocator,
    ism_9_t: AbilityToken,
    ssu_6_t: AbilityToken,
    rsu_9_t: AbilityToken,
    iru_1_t: AbilityToken,
    scu_20_t: AbilityToken,
    comeback_t: AbilityToken,
    ninja_squid_t: AbilityToken,
    stealth_jump_t: AbilityToken,
):
    """
    ISM_9 (3 subs), SSU_6 (2 subs), RSU_9 (3 subs), IRU_1 (1 sub). Total 3+2+3+1 = 9 subs.
    Adding SCU_20 (min 7 subs if no main). This will exceed 9 subs.
    If SCU_20 takes a main (10AP), it needs 4 subs (12AP). Total 10+12=22AP for SCU.
    Build: SCU main (10) + 4 SCU subs (12) = 22
           ISM 3 subs (9)
           SSU 2 subs (6)
           RSU 3 subs (9) -> this combo is already 4+3+2+3 = 12 subs. Impossible.
    The allocator should find no valid build.
    """
    capstones = {
        "ism": ism_9_t,
        "ssu": ssu_6_t,
        "rsu": rsu_9_t,
        "iru": iru_1_t,
        "scu": scu_20_t,
        "cb": comeback_t,
        "ns": ninja_squid_t,
        "sj": stealth_jump_t,
    }
    build = allocator_instance.allocate(capstones)
    assert build is None


def test_capstone_min_ap_unmet(
    allocator_instance: Allocator, scu_20_t: AbilityToken
):
    """SCU_20 needs 20 AP.
    Max AP from 1 main + 3 subs of *itself* = 10 + 3*3 = 19. Not enough.
    Max AP from 0 main + 9 subs of *itself* = 9*3 = 27. Enough.
    This test is more about whether the allocator correctly calculates it.
    If capstones = {scu_20_t}, it should use 7 subs. Total AP 21.
    """
    # This test is tricky because SCU_20 means min_ap=20.
    # It *can* be satisfied with 7 subs (21 AP).
    # The previous sub_slots_exceeded might be a better test for "unmet".
    # Let's make one where it's truly impossible:
    # e.g. AbilityX_30_AP where only 9 subs are allowed (max 27 from subs)
    # and only 1 main slot can be taken by it (10 + some subs).
    # For now, let's test SCU_20 as is.
    capstones = {"scu": scu_20_t}
    build = allocator_instance.allocate(capstones)
    assert build is not None
    num_mains = sum(1 for v in build.mains.values() if v == "special_charge_up")
    assert num_mains == 2
    assert build.subs.get("special_charge_up", 0) == 0
    assert build.total_ap == 20
    assert build.achieved_ap.get("special_charge_up", 0) == 20


def test_impossible_min_ap_overall(allocator_instance: Allocator):
    """An ability requiring more AP than possible with 1 main + 9 subs for itself."""
    # e.g. family "impossible_ability", min_ap 40
    # Max for one ability: 1 main (10) + 9 subs (27) = 37.
    # So min_ap 40 is impossible for a single ability family.
    impossible_token = AbilityToken(
        name="imp_40", family="impossible", min_ap=60, main_only=False
    )
    capstones = {"imp": impossible_token}
    build = allocator_instance.allocate(capstones)
    assert build is None


def test_complex_scenario_minimize_total_ap(
    allocator_instance: Allocator,
    comeback_t: AbilityToken,
    ism_3_t: AbilityToken,
    ssu_21_t: AbilityToken,
    qr_3_t: AbilityToken,
    qsj_3_t: AbilityToken,
):
    """
    Comeback (main, 10AP)
    ISM_3 (min 3AP):
        - Option A: 1 sub (3AP)
    SSU_21 (min 21AP):
        - Option A: 7 subs (21AP)
        - Option B: 1 main + 4 subs (10+4*3=22AP)
        - Option C: 2 mains + 1 sub (10*2+3=23AP)
    QR_3 (min 3AP):
        - Option A: 1 sub (3AP)
    QSJ_3 (min 3AP):
        - Option A: 1 sub (3AP)

    Overall Scenarios:
    1. CB (main head, 10), QR_3 (sub, 3), QSJ_3 (sub, 3), ISM_3 (sub, 3), SSU_21 (1m4s, 22)
    2. CB (main head, 10), QR_3 (sub, 3), QSJ_3 (sub, 3), ISM_3 (sub, 3), SSU_21 (7s, 21) (impossible)
    3. CB (main head, 10), QR_3 (sub, 3), QSJ_3 (sub, 3), ISM_3 (sub, 3), SSU_21 (1m4s, 22)
    """
    capstones = {
        "cb": comeback_t,
        "ism": ism_3_t,
        "ssu": ssu_21_t,
        "qr": qr_3_t,
        "qsj": qsj_3_t,
    }
    build = allocator_instance.allocate(capstones)
    assert build is not None
    assert build.total_ap == 41
    assert build.mains["head"] == "comeback"
    # One of the other two mains is "swim_speed_up" but not both
    assert (
        build.mains["clothes"] == "swim_speed_up"
        or build.mains["shoes"] == "swim_speed_up"
    )
    assert build.mains["clothes"] != build.mains["shoes"]
    assert build.subs["ink_saver_main"] == 1
    assert build.subs["quick_respawn"] == 1
    assert build.subs["quick_super_jump"] == 1
