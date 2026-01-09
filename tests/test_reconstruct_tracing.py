from splatnlp.utils.constants import NULL
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build


def predict(tokens, weapon_id):
    key = tuple(tok for tok in tokens if tok != NULL)
    if not key:
        return {"ink_saver_main_3": 0.6}
    if key == ("ink_saver_main_3",):
        return {"run_speed_up_6": 0.7}
    return {}


def predict_batch(token_batches, weapon_id):
    return [predict(tokens, weapon_id) for tokens in token_batches]


def test_reconstruct_build_returns_traces():
    allocator = Allocator()
    result_builds, traces = reconstruct_build(
        predict_fn=predict,
        predict_batch_fn=predict_batch,
        weapon_id="w",
        initial_context=[],
        allocator=allocator,
        beam_size=1,
        max_steps=1,
        record_traces=True,
    )
    assert result_builds is not None
    assert traces is not None
    assert len(result_builds) == 1
    assert len(traces) == 1
    trace = traces[0]
    # Check the final trace frame has the expected tokens (step count may vary)
    assert set(trace[-1].partial_caps.keys()) == {
        "ink_saver_main_3",
        "run_speed_up_6",
    }
