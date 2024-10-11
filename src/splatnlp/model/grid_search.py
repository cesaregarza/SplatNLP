from itertools import product
from math import prod
from typing import Any

import torch
from tqdm import tqdm

from splatnlp.model.config import TrainingConfig
from splatnlp.model.evaluation import test_model
from splatnlp.model.training_loop import train_model


def grid_search(
    model_class: type,
    train_dl: torch.utils.data.DataLoader,
    val_dl: torch.utils.data.DataLoader,
    test_dl: torch.utils.data.DataLoader,
    config: TrainingConfig,
    vocab: dict[str, int],
    param_grid: dict[str, list[Any]],
    fixed_params: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float], torch.nn.Module]:
    best_model_params = None
    best_test_metrics = None
    best_model = None
    best_val_f1 = 0

    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    total_combinations = prod([len(values) for values in param_values])
    pbar = tqdm(total=total_combinations, desc="Grid Search")

    for params in product(*param_values):
        current_model_params = dict(zip(param_names, params))
        model_params = {**fixed_params, **current_model_params}

        print("Searching model params:", current_model_params)

        model = model_class(**model_params)
        _, trained_model = train_model(
            model, train_dl, val_dl, config, vocab, verbose=False
        )

        test_metrics = test_model(
            trained_model, test_dl, config, vocab, verbose=False
        )
        print("Test Metrics:", test_metrics)

        if test_metrics["f1"] > best_val_f1:
            print("New best model found!")
            best_val_f1 = test_metrics["f1"]
            best_model_params = model_params
            best_test_metrics = test_metrics
            best_model = trained_model

        pbar.update(1)
        pbar.set_postfix({"Best F1": f"{best_val_f1:.4f}"})

    pbar.close()
    return best_model_params, best_test_metrics, best_model
