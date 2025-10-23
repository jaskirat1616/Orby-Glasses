"""
Production Error Handler
Graceful error recovery for reliable operation.
"""

import logging
import traceback
from typing import Optional, Callable
from functools import wraps


class ErrorHandler:
    """Production-grade error handling and recovery."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """Initialize error handler."""
        self.logger = logger or logging.getLogger(__name__)
        self.error_counts = {}
        self.max_retries = 3

    def safe_execute(self, func: Callable, *args, default=None, **kwargs):
        """
        Safely execute a function with error handling.

        Args:
            func: Function to execute
            *args: Function arguments
            default: Default return value on error
            **kwargs: Function keyword arguments

        Returns:
            Function result or default value
        """
        func_name = func.__name__
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"Error in {func_name}: {e}")
            self.error_counts[func_name] = self.error_counts.get(func_name, 0) + 1
            return default

    def resilient(self, default=None, max_retries=3):
        """
        Decorator for resilient functions.

        Args:
            default: Default return value on failure
            max_retries: Maximum retry attempts
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                for attempt in range(max_retries):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        if attempt == max_retries - 1:
                            self.logger.error(
                                f"Failed {func.__name__} after {max_retries} attempts: {e}"
                            )
                            return default
                        self.logger.warning(
                            f"Retry {attempt + 1}/{max_retries} for {func.__name__}: {e}"
                        )
            return wrapper
        return decorator

    def get_error_stats(self):
        """Get error statistics."""
        return self.error_counts.copy()
