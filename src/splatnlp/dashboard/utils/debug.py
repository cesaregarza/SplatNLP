"""
Debug utilities for the dashboard components.
"""

import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class OperationProfiler:
    """Context manager and decorator for profiling operation execution time."""

    def __init__(self, operation_name: str):
        """Initialize profiler with operation name.

        Args:
            operation_name: Name of the operation for logging
        """
        self.operation_name = operation_name
        self.start_time = None

    def __enter__(self) -> None:
        """Start timing when entering context."""
        self.start_time = time.time()

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Log execution time when exiting context."""
        end_time = time.time()
        duration = end_time - self.start_time
        logger.info(
            f"Operation '{self.operation_name}' took {duration:.3f} seconds"
        )

    def __call__(self, func: Callable[..., T]) -> Callable[..., T]:
        """Allow using the profiler as a decorator.

        Args:
            func: Function to profile

        Returns:
            Decorated function that logs execution time
        """

        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            with self:
                return func(*args, **kwargs)

        return wrapper


# For backward compatibility
def profile_operation(operation_name: str) -> OperationProfiler:
    """Create an OperationProfiler instance.

    Args:
        operation_name: Name of the operation for logging

    Returns:
        OperationProfiler instance that can be used as decorator or context manager
    """
    return OperationProfiler(operation_name)
