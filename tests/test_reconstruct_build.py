import pytest

from splatnlp.utils.constants import NULL
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import (
    reconstruct_build,
    reconstruct_builds_batched,
)
from splatnlp.utils.reconstruct.classes import AbilityToken


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
        # Return one token above greedy threshold (0.5), other below
        # This ensures greedy closure adds one, then beam search explores
        if "comeback" not in tokens and "last_ditch_effort" not in tokens:
            # Return comeback high enough for greedy, lde below threshold
            return {"comeback": 0.6, "last_ditch_effort": 0.3}
        # After adding one, the other stays below threshold
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


def test_allocator_prefers_high_ap_over_priority_bias():
    """
    When sub slots are exhausted, low-AP abilities should not be promoted to
    mains even if their priority score is higher than stronger abilities.
    """

    caps = {
        "run_speed_up_21": AbilityToken.from_vocab_entry("run_speed_up_21"),
        "ink_saver_sub_21": AbilityToken.from_vocab_entry("ink_saver_sub_21"),
        "swim_speed_up_6": AbilityToken.from_vocab_entry("swim_speed_up_6"),
        "ink_recovery_up_3": AbilityToken.from_vocab_entry("ink_recovery_up_3"),
    }
    allocator = Allocator()

    # Bias priority toward ink recovery up; AP should still win.
    priority = {"ink_recovery_up": 100.0}
    build, _ = allocator.allocate(caps, priority=priority)
    assert build is not None
    assert "ink_recovery_up" not in build.mains.values()
    assert "run_speed_up" in build.mains.values()
    assert "ink_saver_sub" in build.mains.values()


def test_reconstruct_build_uses_predict_batch_fn():
    single_calls: list[list[str]] = []
    batch_calls: list[int] = []

    def logic(tokens: list[str]) -> dict[str, float]:
        key = tuple(tok for tok in tokens if tok != NULL)
        if not key:
            return {"ink_saver_main_3": 0.6}
        if key == ("ink_saver_main_3",):
            return {"run_speed_up_3": 0.2, "swim_speed_up_3": 0.19}
        if "run_speed_up_3" in key and "swim_speed_up_3" not in key:
            return {"swim_speed_up_3": 0.18}
        if "swim_speed_up_3" in key and "run_speed_up_3" not in key:
            return {"run_speed_up_3": 0.17}
        return {}

    def predict_fn(tokens, weapon_id):
        single_calls.append(list(tokens))
        return logic(tokens)

    def predict_batch_fn(token_batches, weapon_id):
        batch_calls.append(len(token_batches))
        return [logic(tokens) for tokens in token_batches]

    allocator = Allocator()
    builds = reconstruct_build(
        predict_fn=predict_fn,
        predict_batch_fn=predict_batch_fn,
        weapon_id="w",
        initial_context=[],
        allocator=allocator,
        beam_size=2,
        max_steps=2,
        top_k=1,
    )
    assert builds
    assert len(single_calls) == 2  # greedy closure only
    assert any(n > 1 for n in batch_calls)


def test_reconstruct_builds_batched_matches_individual_calls():
    def predict_single(tokens: list[str], weapon_id: str) -> dict[str, float]:
        key = tuple(tok for tok in tokens if tok != NULL)
        if weapon_id == "w1":
            if not key:
                return {"ink_saver_main_3": 0.6}
            if key == ("ink_saver_main_3",):
                return {"run_speed_up_6": 0.7}
            return {}
        if weapon_id == "w2":
            if not key:
                return {"swim_speed_up_6": 0.6}
            if key == ("swim_speed_up_6",):
                return {"stealth_jump": 0.7}
            return {}
        return {}

    def predict_fn(tokens: list[str], weapon_id: str) -> dict[str, float]:
        return predict_single(tokens, weapon_id)

    def predict_batch_fn(
        token_batches: list[list[str]],
        weapon_id: str,
    ) -> list[dict[str, float]]:
        return [predict_single(tokens, weapon_id) for tokens in token_batches]

    def predict_batch_multi_fn(
        token_batches: list[list[str]],
        weapon_ids: list[str],
    ) -> list[dict[str, float]]:
        return [
            predict_single(tokens, wid)
            for tokens, wid in zip(token_batches, weapon_ids)
        ]

    allocator = Allocator()
    weapon_ids = ["w1", "w2", "w3"]
    contexts = [[], [], ["comeback", "last_ditch_effort"]]

    expected = [
        reconstruct_build(
            predict_fn=predict_fn,
            predict_batch_fn=predict_batch_fn,
            weapon_id=wid,
            initial_context=ctx,
            allocator=allocator,
            beam_size=2,
            max_steps=2,
            top_k=1,
        )
        for wid, ctx in zip(weapon_ids, contexts)
    ]

    got = reconstruct_builds_batched(
        predict_batch_fn=predict_batch_multi_fn,
        weapon_ids=weapon_ids,
        initial_contexts=contexts,
        allocator=allocator,
        beam_size=2,
        max_steps=2,
        top_k=1,
    )

    assert got == expected
