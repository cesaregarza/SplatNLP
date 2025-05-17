import pytest

from splatnlp.utils.reconstruct.beam_search import reconstruct_build
from splatnlp.utils.reconstruct.allocator import Allocator


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
    build = reconstruct_build(predict_fn, "w", [], allocator, beam_size=2, max_steps=2)
    assert build is not None
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

