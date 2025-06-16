import pytest

from splatnlp.utils.constants import NULL
from splatnlp.utils.reconstruct.beam_search import greedy_closure


def simple_predict(tokens, weapon_id):
    key = tuple(tok for tok in tokens if tok != NULL)
    if not key:
        return {"ink_saver_main_3": 0.6}
    if key == ("ink_saver_main_3",):
        return {"run_speed_up_6": 0.7}
    return {}


def test_greedy_closure_tracing_steps():
    caps, step, traces = greedy_closure(
        simple_predict, "w", {}, record_traces=True, start_step=0
    )
    assert set(caps.keys()) == {"ink_saver_main_3", "run_speed_up_6"}
    assert step == 2
    assert [t.step for t in traces] == [0, 1, 2]
    assert list(traces[0].partial_caps.keys()) == []
    assert set(traces[1].partial_caps.keys()) == {"ink_saver_main_3"}
    assert set(traces[2].partial_caps.keys()) == {
        "ink_saver_main_3",
        "run_speed_up_6",
    }

