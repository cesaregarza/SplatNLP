"""I/O utilities for research state persistence.

This module handles reading and writing research state to the
artifact storage directory at /mnt/e/mechinterp_runs/.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Literal

from splatnlp.mechinterp.schemas.research_state import ResearchState

logger = logging.getLogger(__name__)

# Base path for all mechinterp artifacts
RUNS_BASE_PATH = Path("/mnt/e/mechinterp_runs")

# Subdirectories
STATE_DIR = RUNS_BASE_PATH / "state"
SPECS_DIR = RUNS_BASE_PATH / "specs"
RESULTS_DIR = RUNS_BASE_PATH / "results"
FIGURES_DIR = RUNS_BASE_PATH / "figures"
NOTES_DIR = RUNS_BASE_PATH / "notes"


def ensure_dirs() -> None:
    """Ensure all artifact directories exist."""
    for d in [STATE_DIR, SPECS_DIR, RESULTS_DIR, FIGURES_DIR, NOTES_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def get_state_path(
    feature_id: int,
    model_type: Literal["full", "ultra"] = "ultra",
) -> Path:
    """Get the path to a feature's state file.

    Args:
        feature_id: SAE feature ID
        model_type: Model type for namespacing

    Returns:
        Path to the state JSON file
    """
    return STATE_DIR / f"feature_{feature_id}_{model_type}.json"


def get_notes_path(
    feature_id: int,
    model_type: Literal["full", "ultra"] = "ultra",
) -> Path:
    """Get the path to a feature's notes file.

    Args:
        feature_id: SAE feature ID
        model_type: Model type for namespacing

    Returns:
        Path to the notes Markdown file
    """
    return NOTES_DIR / f"feature_{feature_id}_{model_type}.md"


def get_spec_path(spec_id: str, feature_id: int, experiment_type: str) -> Path:
    """Get the path for an experiment spec file.

    Args:
        spec_id: Unique spec identifier (typically timestamp)
        feature_id: Feature being analyzed
        experiment_type: Type of experiment

    Returns:
        Path to the spec JSON file
    """
    type_short = experiment_type.replace("_", "-")
    return SPECS_DIR / f"{spec_id}__f{feature_id}__{type_short}.json"


def get_result_path(spec_id: str) -> Path:
    """Get the path for an experiment result file.

    Args:
        spec_id: Spec identifier (matches the spec file)

    Returns:
        Path to the result JSON file
    """
    return RESULTS_DIR / f"{spec_id}__result.json"


def state_exists(
    feature_id: int,
    model_type: Literal["full", "ultra"] = "ultra",
) -> bool:
    """Check if a state file exists for a feature.

    Args:
        feature_id: SAE feature ID
        model_type: Model type

    Returns:
        True if state file exists
    """
    return get_state_path(feature_id, model_type).exists()


def load_state(
    feature_id: int,
    model_type: Literal["full", "ultra"] = "ultra",
) -> ResearchState | None:
    """Load research state from disk.

    Args:
        feature_id: SAE feature ID
        model_type: Model type

    Returns:
        ResearchState if file exists, None otherwise
    """
    path = get_state_path(feature_id, model_type)
    if not path.exists():
        logger.info(f"No state file found at {path}")
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)
        state = ResearchState.model_validate(data)
        logger.info(f"Loaded state for feature {feature_id} from {path}")
        return state
    except Exception as e:
        logger.error(f"Failed to load state from {path}: {e}")
        return None


def save_state(state: ResearchState) -> Path:
    """Save research state to disk.

    Args:
        state: ResearchState to save

    Returns:
        Path where state was saved
    """
    ensure_dirs()
    path = get_state_path(state.feature_id, state.model_type)

    # Update timestamp
    state.updated_at = datetime.now()

    try:
        with open(path, "w") as f:
            f.write(state.model_dump_json(indent=2))
        logger.info(f"Saved state for feature {state.feature_id} to {path}")
        return path
    except Exception as e:
        logger.error(f"Failed to save state to {path}: {e}")
        raise


def create_initial_state(
    feature_id: int,
    model_type: Literal["full", "ultra"] = "ultra",
    feature_label: str | None = None,
) -> ResearchState:
    """Create a new research state for a feature.

    Args:
        feature_id: SAE feature ID
        model_type: Model type
        feature_label: Optional human label for the feature

    Returns:
        Newly created ResearchState (not yet saved)
    """
    state = ResearchState(
        feature_id=feature_id,
        model_type=model_type,
        feature_label=feature_label,
    )
    state.add_history(
        action="state_created",
        details=f"Initial research state created for feature {feature_id}",
    )
    return state


def list_states(
    model_type: Literal["full", "ultra"] | None = None,
) -> list[tuple[int, str, Path]]:
    """List all existing state files.

    Args:
        model_type: Filter by model type (None = all)

    Returns:
        List of (feature_id, model_type, path) tuples
    """
    ensure_dirs()
    results = []

    for path in STATE_DIR.glob("feature_*_*.json"):
        # Parse filename: feature_{id}_{model}.json
        parts = path.stem.split("_")
        if len(parts) >= 3:
            try:
                fid = int(parts[1])
                mtype = parts[2]
                if model_type is None or mtype == model_type:
                    results.append((fid, mtype, path))
            except ValueError:
                continue

    return sorted(results, key=lambda x: (x[1], x[0]))


def list_specs(feature_id: int | None = None) -> list[Path]:
    """List experiment spec files.

    Args:
        feature_id: Filter by feature ID (None = all)

    Returns:
        List of spec file paths, sorted by timestamp
    """
    ensure_dirs()
    pattern = f"*__f{feature_id}__*.json" if feature_id else "*.json"
    return sorted(SPECS_DIR.glob(pattern))


def list_results(feature_id: int | None = None) -> list[Path]:
    """List experiment result files.

    Args:
        feature_id: Filter by feature ID (None = all)

    Returns:
        List of result file paths, sorted by timestamp
    """
    ensure_dirs()
    if feature_id:
        # Results are named {spec_id}__result.json, need to check inside
        results = []
        for path in RESULTS_DIR.glob("*__result.json"):
            try:
                with open(path) as f:
                    data = json.load(f)
                if data.get("feature_id") == feature_id:
                    results.append(path)
            except Exception:
                continue
        return sorted(results)
    return sorted(RESULTS_DIR.glob("*__result.json"))
