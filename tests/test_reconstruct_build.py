import pytest

from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build


def noop_predict(tokens, weapon_id):
    """Predict function that returns no additional tokens."""
    return {}


def test_conflicting_initial_context_returns_none():
    allocator = Allocator()
    build = reconstruct_build(
        noop_predict,
        "w",
        ["comeback", "last_ditch_effort"],
        allocator,
        beam_size=2,
        max_steps=1,
    )
    assert build is None


def test_conflicting_predictions_select_one_head():
    def predict_fn(tokens, weapon_id):
        if "comeback" not in tokens and "last_ditch_effort" not in tokens:
            return {"comeback": 0.0, "last_ditch_effort": 0.0}
        if "comeback" in tokens and "last_ditch_effort" not in tokens:
            return {"last_ditch_effort": 0.0}
        if "last_ditch_effort" in tokens and "comeback" not in tokens:
            return {"comeback": 0.0}
        return {}

    allocator = Allocator()
    builds = reconstruct_build(
        predict_fn, "w", [], allocator, beam_size=2, max_steps=2
    )
    assert builds
    build = builds[0]
    assert build.mains["head"] in {"comeback", "last_ditch_effort"}


def test_exceed_sub_slots_returns_none():
    allocator = Allocator()
    initial_tokens = [
        "comeback",
        "ninja_squid",
        "stealth_jump",
        "ink_saver_main_9",
        "run_speed_up_9",
        "swim_speed_up_9",
        "quick_respawn_9",
    ]
    build = reconstruct_build(
        noop_predict,
        "w",
        initial_tokens,
        allocator,
        beam_size=2,
        max_steps=1,
    )
    assert build is None


def test_full_build_not_pruned_when_expansions_are_invalid():
    """
    When the initial context already consumes all 57 AP, any additional token
    makes allocation impossible. The valid starting build should still be
    returned rather than being pruned by the beam.
    """

    def suggest_extra_token(tokens, weapon_id):
        # Always suggest adding another ability (which would exceed 57 AP)
        return {"ink_saver_main_3": 1.0}

    allocator = Allocator()
    builds = reconstruct_build(
        suggest_extra_token,
        "w",
        ["special_charge_up_57"],
        allocator,
        beam_size=1,
        max_steps=1,
    )
    assert builds
    build = builds[0]
    assert build.total_ap == 57
    # Special Charge Up should stay on all mains with 9 subs to hit 57 AP
    assert set(build.mains.values()) == {"special_charge_up"}
    assert build.subs.get("special_charge_up") == 9
