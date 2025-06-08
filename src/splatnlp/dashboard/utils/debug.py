"""
Debug utilities for the dashboard components.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


def profile_operation(
    operation_name: str,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to profile operation execution time.

    Args:
        operation_name: Name of the operation for logging

    Returns:
        Decorated function that logs execution time
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            duration = end_time - start_time
            logger.info(
                f"Operation '{operation_name}' took {duration:.3f} seconds"
            )
            return result

        return wrapper

    return decorator
