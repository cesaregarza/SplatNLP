"""Experiment runners for mechanistic interpretability."""

from splatnlp.mechinterp.experiments.base import (
    ExperimentRunner,
    get_runner_for_type,
)

__all__ = [
    "ExperimentRunner",
    "get_runner_for_type",
]
