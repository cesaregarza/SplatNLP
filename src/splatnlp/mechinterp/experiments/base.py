"""Base experiment runner and factory.

This module provides the abstract base class for experiment runners
and a factory function to get the appropriate runner for an experiment type.
"""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime

from splatnlp.mechinterp.schemas.experiment_results import (
    Aggregates,
    DiagnosticInfo,
    ExperimentResult,
)
from splatnlp.mechinterp.schemas.experiment_specs import (
    ExperimentSpec,
    ExperimentType,
)
from splatnlp.mechinterp.skill_helpers.context_loader import MechInterpContext

logger = logging.getLogger(__name__)


class ExperimentRunner(ABC):
    """Abstract base class for experiment runners.

    Subclasses implement specific experiment types by overriding
    the `_run_experiment` method.
    """

    # Human-readable name for this runner
    name: str = "base"

    # Experiment types this runner handles
    handles_types: list[ExperimentType] = []

    def run(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
    ) -> ExperimentResult:
        """Execute an experiment and return results.

        Args:
            spec: Experiment specification
            ctx: Context with database and vocabularies

        Returns:
            ExperimentResult with aggregates, tables, and diagnostics
        """
        logger.info(
            f"Starting {self.name} experiment for feature {spec.feature_id}"
        )
        started_at = datetime.now()
        start_time = time.time()

        # Initialize result
        result = ExperimentResult(
            spec_id=spec.id,
            spec_path=spec.to_filename(),
            feature_id=spec.feature_id,
            experiment_type=spec.type.value,
            started_at=started_at,
        )

        try:
            # Validate constraints
            self._validate_constraints(spec, ctx, result)

            # Run the actual experiment
            self._run_experiment(spec, ctx, result)

            result.success = True

        except Exception as e:
            logger.error(f"Experiment failed: {e}", exc_info=True)
            result.success = False
            result.error_message = str(e)
            result.diagnostics.warnings.append(f"Experiment failed: {e}")

        # Finalize timing
        result.completed_at = datetime.now()
        result.duration_seconds = time.time() - start_time

        logger.info(
            f"Completed {self.name} in {result.duration_seconds:.1f}s "
            f"(success={result.success})"
        )

        return result

    @abstractmethod
    def _run_experiment(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Execute the experiment logic.

        Subclasses must implement this to perform the actual experiment.
        Results should be written to the `result` object in place.

        Args:
            spec: Experiment specification
            ctx: Context with database and vocabularies
            result: Result object to populate
        """
        pass

    def _validate_constraints(
        self,
        spec: ExperimentSpec,
        ctx: MechInterpContext,
        result: ExperimentResult,
    ) -> None:
        """Validate experiment constraints.

        Override in subclasses to add experiment-specific validation.
        """
        from splatnlp.mechinterp.schemas.glossary import DOMAIN_CONSTRAINTS

        for constraint_id in spec.constraints:
            if constraint_id not in DOMAIN_CONSTRAINTS:
                result.diagnostics.warnings.append(
                    f"Unknown constraint: {constraint_id}"
                )

    def _check_relu_floor(
        self,
        activations: list[float],
        threshold: float = 0.1,
    ) -> tuple[bool, float]:
        """Check if activations are near the ReLU floor.

        Args:
            activations: List of activation values
            threshold: Activation threshold to consider "floor"

        Returns:
            Tuple of (floor_detected, floor_rate)
        """
        if not activations:
            return False, 0.0

        n_floor = sum(1 for a in activations if a < threshold)
        rate = n_floor / len(activations)
        detected = rate > 0.5  # More than half at floor

        return detected, rate


# Registry of experiment runners
_RUNNER_REGISTRY: dict[ExperimentType, type[ExperimentRunner]] = {}


def register_runner(
    runner_class: type[ExperimentRunner],
) -> type[ExperimentRunner]:
    """Decorator to register an experiment runner.

    Usage:
        @register_runner
        class MyRunner(ExperimentRunner):
            handles_types = [ExperimentType.MY_TYPE]
            ...
    """
    for exp_type in runner_class.handles_types:
        _RUNNER_REGISTRY[exp_type] = runner_class
        logger.debug(f"Registered {runner_class.name} for {exp_type}")
    return runner_class


def get_runner_for_type(experiment_type: ExperimentType) -> ExperimentRunner:
    """Get the appropriate runner for an experiment type.

    Args:
        experiment_type: Type of experiment

    Returns:
        Instance of the appropriate ExperimentRunner subclass

    Raises:
        ValueError: If no runner is registered for the type
    """
    # Import runners to trigger registration
    _import_runners()

    if experiment_type not in _RUNNER_REGISTRY:
        available = list(_RUNNER_REGISTRY.keys())
        raise ValueError(
            f"No runner registered for {experiment_type}. "
            f"Available: {available}"
        )

    runner_class = _RUNNER_REGISTRY[experiment_type]
    return runner_class()


def _import_runners() -> None:
    """Import all runner modules to trigger registration."""
    # pylint: disable=import-outside-toplevel,unused-import
    try:
        from splatnlp.mechinterp.experiments import family_sweep
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import itemset_mining
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import minimal_cores
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import interactions
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import validation
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import weapon_sweep
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import interference
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import token_influence
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import kit_sweep
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import binary_presence
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import core_coverage
    except ImportError:
        pass

    try:
        from splatnlp.mechinterp.experiments import decoder_output
    except ImportError:
        pass


def list_available_runners() -> dict[str, list[str]]:
    """List all available experiment runners.

    Returns:
        Dict mapping runner names to experiment types they handle
    """
    _import_runners()

    runners: dict[str, list[str]] = {}
    for exp_type, runner_class in _RUNNER_REGISTRY.items():
        name = runner_class.name
        if name not in runners:
            runners[name] = []
        runners[name].append(exp_type.value)

    return runners
