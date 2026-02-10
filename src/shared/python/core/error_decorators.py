"""Error handling decorators and context managers for DRY error handling.

This module provides reusable error handling patterns to eliminate
repeated try-except-log blocks throughout the codebase.

Usage:
    from src.shared.python.error_decorators import log_errors, handle_import_error

    @log_errors("Failed to process data")
    def process_data(data):
        # Your code here
        pass

    @handle_import_error(default_value=None)
    def optional_import():
        import optional_module
        return optional_module
"""

from __future__ import annotations

import functools
import sys
from collections.abc import Callable
from typing import Any, TypeVar, cast

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def log_errors(
    message: str = "Operation failed",
    reraise: bool = True,
    default_return: Any = None,
) -> Callable[[F], F]:
    """Decorator to log errors with a custom message.

    Args:
        message: Error message to log
        reraise: Whether to reraise the exception after logging
        default_return: Value to return if exception occurs and reraise=False

    Returns:
        Decorated function

    Example:
        @log_errors("Failed to load model")
        def load_model(path):
            return Model.load(path)
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"{message}: {e}", exc_info=True)
                if reraise:
                    raise
                return default_return

        return cast(F, wrapper)

    return decorator


def handle_import_error(
    module_name: str | None = None,
    default_value: Any = None,
    log_warning: bool = True,
) -> Callable[[F], F]:
    """Decorator to handle import errors gracefully.

    Args:
        module_name: Name of module being imported (for logging)
        default_value: Value to return if import fails
        log_warning: Whether to log a warning on import failure

    Returns:
        Decorated function

    Example:
        @handle_import_error("optional_module", default_value=None)
        def try_import():
            import optional_module
            return optional_module
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return func(*args, **kwargs)
            except ImportError as e:
                if log_warning:
                    mod_name = module_name or func.__name__
                    logger.warning(f"Failed to import {mod_name}: {e}")
                return default_value

        return cast(F, wrapper)

    return decorator


def retry_on_error(
    max_attempts: int = 3,
    exceptions: tuple = (Exception,),
    delay: float = 0.0,
) -> Callable[[F], F]:
    """Decorator to retry function on error.

    Args:
        max_attempts: Maximum number of retry attempts
        exceptions: Tuple of exception types to catch
        delay: Delay between retries in seconds

    Returns:
        Decorated function

    Example:
        @retry_on_error(max_attempts=3, exceptions=(IOError,))
        def read_file(path):
            return open(path).read()
    """
    import time

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exception = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        logger.warning(
                            f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {e}"
                        )
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed for {func.__name__}"
                        )
            raise last_exception  # type: ignore

        return cast(F, wrapper)

    return decorator


class ErrorContext:
    """Context manager for consistent error handling.

    Example:
        with ErrorContext("Loading configuration"):
            config = load_config()
    """

    def __init__(
        self,
        operation: str,
        reraise: bool = True,
        log_success: bool = False,
    ):
        """Initialize error context.

        Args:
            operation: Description of the operation
            reraise: Whether to reraise exceptions
            log_success: Whether to log successful completion
        """
        self.operation = operation
        self.reraise = reraise
        self.log_success = log_success

    def __enter__(self) -> ErrorContext:
        """Enter context."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """Exit context and handle errors."""
        if exc_type is not None:
            logger.error(f"{self.operation} failed: {exc_val}", exc_info=True)
            return not self.reraise  # Suppress exception if not reraising
        elif self.log_success:
            logger.info(f"{self.operation} completed successfully")
        return False


def validate_args(**validators: Callable[[Any], bool]) -> Callable[[F], F]:
    """Decorator to validate function arguments.

    Args:
        **validators: Keyword arguments mapping parameter names to validation functions

    Returns:
        Decorated function

    Example:
        @validate_args(x=lambda x: x > 0, y=lambda y: isinstance(y, str))
        def process(x, y):
            pass
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get function signature
            import inspect

            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()

            # Validate arguments
            for param_name, validator in validators.items():
                if param_name in bound_args.arguments:
                    value = bound_args.arguments[param_name]
                    if not validator(value):
                        raise ValueError(
                            f"Validation failed for parameter '{param_name}' with value {value}"
                        )

            return func(*args, **kwargs)

        return cast(F, wrapper)

    return decorator


# Convenience functions for common patterns


def safe_import(module_name: str, package: str | None = None) -> Any | None:
    """Safely import a module, returning None if import fails.

    Args:
        module_name: Name of module to import
        package: Package name for relative imports

    Returns:
        Imported module or None if import fails

    Example:
        mujoco = safe_import("mujoco")
        if mujoco is not None:
            # Use mujoco
            pass
    """
    try:
        return __import__(module_name, fromlist=[""], level=0)
    except ImportError as e:
        logger.debug(f"Failed to import {module_name}: {e}")
        return None


def check_module_available(module_name: str) -> bool:
    """Check if a module is available for import.

    Args:
        module_name: Name of module to check

    Returns:
        True if module can be imported, False otherwise

    Example:
        if check_module_available("mujoco"):
            import mujoco
    """
    return module_name in sys.modules or safe_import(module_name) is not None
