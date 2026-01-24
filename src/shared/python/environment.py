"""Centralized environment variable management for the Golf Modeling Suite.

This module consolidates common patterns for reading and validating
environment variables across the codebase, addressing DRY violations
identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.environment import (
        get_env,
        get_env_bool,
        get_env_int,
        get_secret_key,
        get_database_url,
        get_environment,
    )

    # Get environment variable with default
    port = get_env_int("PORT", default=8000)

    # Get boolean environment variable
    debug = get_env_bool("DEBUG", default=False)

    # Get API secret key
    key = get_secret_key(required=True)

    # Check current environment
    env = get_environment()  # "development", "staging", or "production"
"""

from __future__ import annotations

import os
from typing import TypeVar

from src.shared.python.error_utils import ConfigurationError

T = TypeVar("T")


class EnvironmentError(ConfigurationError):
    """Raised when an environment variable is missing or invalid."""

    def __init__(
        self,
        var_name: str,
        reason: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ):
        super().__init__(
            config_key=var_name,
            reason=reason or "Environment variable not set or invalid",
            expected=expected,
            actual=actual,
        )
        self.var_name = var_name


def get_env(
    name: str,
    default: str | None = None,
    *,
    required: bool = False,
    strip: bool = True,
) -> str | None:
    """Get an environment variable value.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        required: If True, raise EnvironmentError if not set and no default.
        strip: If True, strip whitespace from value.

    Returns:
        The environment variable value, default, or None.

    Raises:
        EnvironmentError: If required and not set with no default.

    Example:
        >>> api_url = get_env("API_URL", default="http://localhost:8000")
    """
    value = os.environ.get(name)

    if value is not None:
        return value.strip() if strip else value

    if default is not None:
        return default

    if required:
        raise EnvironmentError(name, "Required environment variable not set")

    return None


def get_env_bool(
    name: str,
    default: bool = False,
) -> bool:
    """Get a boolean environment variable.

    Recognizes: true/false, yes/no, 1/0, on/off (case-insensitive).

    Args:
        name: Environment variable name.
        default: Default value if not set.

    Returns:
        Boolean value.

    Example:
        >>> debug = get_env_bool("DEBUG", default=False)
    """
    value = os.environ.get(name)

    if value is None:
        return default

    value = value.strip().lower()

    if value in ("true", "yes", "1", "on"):
        return True
    if value in ("false", "no", "0", "off", ""):
        return False

    # If not recognized, return default
    return default


def get_env_int(
    name: str,
    default: int | None = None,
    *,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    """Get an integer environment variable.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Integer value or None.

    Raises:
        EnvironmentError: If value is not a valid integer or out of range.

    Example:
        >>> port = get_env_int("PORT", default=8000, min_value=1, max_value=65535)
    """
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        int_value = int(value.strip())
    except ValueError as e:
        raise EnvironmentError(
            name,
            f"Invalid integer value: {value!r}",
            expected="integer",
            actual=value,
        ) from e

    if min_value is not None and int_value < min_value:
        raise EnvironmentError(
            name,
            f"Value {int_value} below minimum {min_value}",
            expected=f">= {min_value}",
            actual=str(int_value),
        )

    if max_value is not None and int_value > max_value:
        raise EnvironmentError(
            name,
            f"Value {int_value} above maximum {max_value}",
            expected=f"<= {max_value}",
            actual=str(int_value),
        )

    return int_value


def get_env_float(
    name: str,
    default: float | None = None,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    """Get a float environment variable.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        min_value: Minimum allowed value.
        max_value: Maximum allowed value.

    Returns:
        Float value or None.

    Raises:
        EnvironmentError: If value is not a valid float or out of range.

    Example:
        >>> timeout = get_env_float("TIMEOUT", default=30.0, min_value=0.0)
    """
    value = os.environ.get(name)

    if value is None:
        return default

    try:
        float_value = float(value.strip())
    except ValueError as e:
        raise EnvironmentError(
            name,
            f"Invalid float value: {value!r}",
            expected="float",
            actual=value,
        ) from e

    if min_value is not None and float_value < min_value:
        raise EnvironmentError(
            name,
            f"Value {float_value} below minimum {min_value}",
            expected=f">= {min_value}",
            actual=str(float_value),
        )

    if max_value is not None and float_value > max_value:
        raise EnvironmentError(
            name,
            f"Value {float_value} above maximum {max_value}",
            expected=f"<= {max_value}",
            actual=str(float_value),
        )

    return float_value


def get_env_list(
    name: str,
    default: list[str] | None = None,
    *,
    separator: str = ",",
    strip_items: bool = True,
    filter_empty: bool = True,
) -> list[str]:
    """Get a list environment variable.

    Args:
        name: Environment variable name.
        default: Default value if not set.
        separator: Item separator (default: comma).
        strip_items: Strip whitespace from each item.
        filter_empty: Remove empty items.

    Returns:
        List of string values.

    Example:
        >>> hosts = get_env_list("ALLOWED_HOSTS", default=["localhost"])
    """
    value = os.environ.get(name)

    if value is None:
        return default if default is not None else []

    items = value.split(separator)

    if strip_items:
        items = [item.strip() for item in items]

    if filter_empty:
        items = [item for item in items if item]

    return items


def get_environment() -> str:
    """Get the current deployment environment.

    Reads from ENVIRONMENT env var, defaulting to "development".

    Returns:
        One of: "development", "staging", "production" (normalized to lowercase).

    Example:
        >>> env = get_environment()
        >>> if env == "production":
        ...     # Enable production settings
    """
    env = os.environ.get("ENVIRONMENT", "development").lower()

    # Normalize common variants
    if env in ("dev", "local"):
        return "development"
    if env in ("stage", "test", "testing"):
        return "staging"
    if env in ("prod", "live"):
        return "production"

    return env


def is_production() -> bool:
    """Check if running in production environment.

    Returns:
        True if ENVIRONMENT is "production".
    """
    return get_environment() == "production"


def is_development() -> bool:
    """Check if running in development environment.

    Returns:
        True if ENVIRONMENT is "development".
    """
    return get_environment() == "development"


def get_secret_key(*, required: bool = False) -> str | None:
    """Get the API secret key.

    Checks GOLF_API_SECRET_KEY and falls back to SECRET_KEY.

    Args:
        required: If True, raise EnvironmentError if not set in production.

    Returns:
        Secret key string or None.

    Raises:
        EnvironmentError: If required and not set in production.

    Example:
        >>> key = get_secret_key(required=True)
    """
    key = os.environ.get("GOLF_API_SECRET_KEY") or os.environ.get("SECRET_KEY")

    if key:
        return key

    if required or is_production():
        raise EnvironmentError(
            "GOLF_API_SECRET_KEY",
            "Secret key is required for production",
        )

    return None


def get_database_url(default: str = "sqlite:///golf.db") -> str:
    """Get the database URL.

    Args:
        default: Default database URL (SQLite).

    Returns:
        Database URL string.

    Example:
        >>> db_url = get_database_url()
    """
    return os.environ.get("DATABASE_URL", default)


def get_admin_password() -> str | None:
    """Get the admin password.

    Returns:
        Admin password or None if not set.
    """
    return os.environ.get("GOLF_ADMIN_PASSWORD")


def get_api_host(default: str = "0.0.0.0") -> str:
    """Get the API host address.

    Args:
        default: Default host address.

    Returns:
        Host address string.
    """
    return os.environ.get("GOLF_API_HOST", default)


def get_api_port(default: int = 8000) -> int:
    """Get the API port number.

    Args:
        default: Default port number.

    Returns:
        Port number.
    """
    return (
        get_env_int("GOLF_API_PORT", default=default, min_value=1, max_value=65535)
        or default
    )


def get_log_level(default: str = "INFO") -> str:
    """Get the logging level.

    Args:
        default: Default log level.

    Returns:
        Log level string (uppercase).
    """
    level = os.environ.get("LOG_LEVEL", default).upper()
    valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
    return level if level in valid_levels else default


def require_env(name: str) -> str:
    """Get a required environment variable or raise error.

    Args:
        name: Environment variable name.

    Returns:
        Environment variable value.

    Raises:
        EnvironmentError: If not set.

    Example:
        >>> api_key = require_env("API_KEY")
    """
    result = get_env(name, required=True)
    if result is None:
        raise EnvironmentError(name, "Required environment variable not set")
    return result
