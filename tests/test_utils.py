import pandas as pd
import pytest
import torch

from splatnlp.utils.infer import build_predict_abilities
from splatnlp.utils.reconstruct.allocator import Allocator
from splatnlp.utils.reconstruct.beam_search import reconstruct_build


class DummyModel:
    def __init__(self, outputs):
        self.outputs = outputs

    def eval(self):
        pass

    def __call__(self, ability_tokens, weapon_tokens, key_padding_mask=None):
        return torch.tensor([self.outputs], dtype=torch.float32)


class DummyHook:
    def __init__(self):
        self.bypass = False
        self.no_change = False


@pytest.mark.parametrize("output_type", ["df", "dict", "list"])
def test_build_predict_abilities(output_type):
    vocab = {"<PAD>": 0, "tok1": 1, "tok2": 2}
    weapon_vocab = {"w": 0}
    model = DummyModel([0.0, -1.0, 1.0])
    predictor = build_predict_abilities(
        vocab, weapon_vocab, pad_token="<PAD>", output_type=output_type
    )
    result = predictor(model, ["tok1"], "w")

    if output_type == "df":
        assert isinstance(result, pd.DataFrame)
        assert list(result.index) == ["tok2", "<PAD>", "tok1"]
    elif output_type == "dict":
        assert list(result.keys()) == ["<PAD>", "tok1", "tok2"]
    else:
        assert result == ["<PAD>", "tok1", "tok2"]


def test_build_predict_abilities_hook():
    vocab = {"<PAD>": 0, "tok1": 1, "tok2": 2}
    weapon_vocab = {"w": 0}
    model = DummyModel([0.0, -1.0, 1.0])
    hook = DummyHook()
    predictor = build_predict_abilities(
        vocab, weapon_vocab, pad_token="<PAD>", hook=hook
    )
    predictor(model, ["tok1"], "w", bypass=True, no_change=True)
    assert hook.bypass is True
    assert hook.no_change is True


def test_reconstruct_build_basic():
    def predict_fn(tokens, weapon_id):
        # Use probabilities above threshold (0.5) for tokens to be added
        if "comeback" not in tokens:
            return {"comeback": 0.6}
        elif "swim_speed_up_3" not in tokens:
            return {"swim_speed_up_3": 0.6}
        return {}

    allocator = Allocator()
    builds = reconstruct_build(
        predict_fn, "weapon", [], allocator, beam_size=2, max_steps=2
    )

    assert builds is not None
    build = builds[0]
    assert build.mains["head"] == "comeback"
    assert build.subs.get("swim_speed_up") == 1
