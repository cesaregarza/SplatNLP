"""Dashboard commands for activation processing and analysis."""

from splatnlp.dashboard.commands import precompute

__all__ = [
    "generate_activations_cmd",
    "convert_to_efficient_cmd",
    "create_examples_storage_cmd",
    "analyze_activations_cmd",
    "run_dashboard_cmd",
    "precompute",  # Expose the precompute submodule
]
