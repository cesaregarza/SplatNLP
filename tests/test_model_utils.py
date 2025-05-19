import numpy as np
import pytest
import torch

from splatnlp.model.utils import (
    create_multi_hot_targets,
    update_epoch_metrics,
    update_progress_bar,
)


class DummyPBar:
    def __init__(self):
        self.postfix = None

    def set_postfix(self, postfix):
        self.postfix = postfix


def test_create_multi_hot_targets_simple():
    vocab = {"<PAD>": 0, "a": 1, "b": 2, "c": 3}
    batch_targets = torch.tensor(
        [[1, 2, 0], [2, 3, 3]], dtype=torch.long
    )
    result = create_multi_hot_targets(batch_targets, vocab, torch.device("cpu"))
    expected = torch.tensor(
        [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]]
    )
    assert torch.allclose(result, expected)


def test_update_epoch_metrics_and_progress_bar():
    metrics = {"loss": 2.0}
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 0])

    update_epoch_metrics(metrics, y_true, y_pred)

    pbar = DummyPBar()
    update_progress_bar(pbar, metrics, total_batches=2)

    assert pytest.approx(metrics["f1"], 0.01) == 0.75
    assert pytest.approx(metrics["precision"], 0.01) == 0.75
    assert pytest.approx(metrics["recall"], 0.01) == 0.75
    assert pbar.postfix == {
        "Loss": "1.000",
        "F1": f"{metrics['f1']:.3f}",
        "Precision": f"{metrics['precision']:.3f}",
        "Recall": f"{metrics['recall']:.3f}",
    }
