"""Centralized environment variable management for the Golf Modeling Suite.

This module consolidates common patterns for reading and validating
environment variables across the codebase, addressing DRY violations
identified in Pragmatic Programmer reviews.

Usage:
    from src.shared.python.config.environment import (
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
    realtime_host = get_realtime_host()
    realtime_port = get_realtime_port()
"""

from __future__ import annotations

import functools
import os
from typing import TypeVar

from src.shared.python.core.error_utils import ConfigurationError

T = TypeVar("T")


class EnvironmentError(ConfigurationError):
    """Raised when an environment variable is missing or invalid."""

    def __init__(
        self,
        var_name: str,
        reason: str | None = None,
        expected: str | None = None,
        actual: str | None = None,
    ) -> None:
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
    if not (name is not None):
        raise ValueError("name must be provided")
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
    if not (name is not None):
        raise ValueError("name must be provided")
    value = os.environ.get(name)

    if value is None:
        return default if default is not None else []

    items = value.split(separator)

    if strip_items:
        items = [item.strip() for item in items]

    if filter_empty:
        items = [item for item in items if item]

    return items


@functools.cache
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


def get_database_pool_size(default: int = 5) -> int:
    """Get the database connection-pool size for non-SQLite engines."""
    return get_env_int("GOLF_DB_POOL_SIZE", default=default, min_value=1) or default


def get_database_pool_recycle(default: int = 300) -> int:
    """Get the database pool recycle window in seconds for non-SQLite engines."""
    value = get_env_int("GOLF_DB_POOL_RECYCLE", default=default, min_value=0)
    return default if value is None else value


def get_database_pool_pre_ping(default: bool = True) -> bool:
    """Get whether the database pool should pre-ping connections."""
    return get_env_bool("GOLF_DB_POOL_PRE_PING", default=default)


def get_auth_cache_ttl_seconds(default: int = 300) -> int:
    """Get the auth-cache TTL in seconds.

    Operators tuning multi-worker deployments can shorten this window to
    bound stale-credential exposure or lengthen it to reduce bcrypt load.
    Reads ``GOLF_AUTH_CACHE_TTL_SECONDS``; falls back to the supplied
    default (5 minutes) when unset or invalid.
    """
    value = get_env_int("GOLF_AUTH_CACHE_TTL_SECONDS", default=default, min_value=1)
    return default if value is None else value


def get_auth_cache_max_entries(default: int = 10_000) -> int:
    """Get the maximum number of entries kept in the auth cache.

    Once the cache is full, the oldest entry is evicted FIFO-style. Larger
    values reduce bcrypt churn for high-cardinality API-key fleets at the
    cost of memory per worker. Reads ``GOLF_AUTH_CACHE_MAX_ENTRIES``.
    """
    value = get_env_int("GOLF_AUTH_CACHE_MAX_ENTRIES", default=default, min_value=1)
    return default if value is None else value


def get_admin_password() -> str | None:
    """Get the admin password.

    Returns:
        Admin password or None if not set.
    """
    return os.environ.get("GOLF_ADMIN_PASSWORD")


def get_api_host(default: str = "127.0.0.1") -> str:
    """Get the API host address.

    Uses 127.0.0.1 (localhost) by default for security.
    Set GOLF_API_HOST=0.0.0.0 for Docker or when external access is needed.

    Args:
        default: Default host address (127.0.0.1 for localhost-only access).

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


def get_realtime_host(default: str = "127.0.0.1") -> str:
    """Get the realtime WebSocket host address.

    Uses loopback by default so the autostarted realtime server stays
    local-only unless the deployment explicitly opts into a broader bind
    address.

    Args:
        default: Default host address.

    Returns:
        Host address string.
    """
    return os.environ.get("GOLF_REALTIME_HOST", default)


def get_realtime_port(default: int = 8765) -> int:
    """Get the realtime WebSocket port number.

    Args:
        default: Default port number.

    Returns:
        Port number.
    """
    return (
        get_env_int(
            "GOLF_REALTIME_PORT",
            default=default,
            min_value=1,
            max_value=65535,
        )
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


@functools.cache
def is_docker() -> bool:
    """Check if running inside a Docker container.

    Returns:
        True if running in Docker.
    """
    if os.path.exists("/.dockerenv"):
        return True
    cgroup_path = "/proc/self/cgroup"
    if not os.path.isfile(cgroup_path):
        return False
    # Issue #5911: ``open(...)`` was previously inlined into ``any(...)``,
    # leaking the file handle when ``any()`` short-circuited. Use ``with``.
    with open(cgroup_path, encoding="utf-8") as handle:
        return any("docker" in line for line in handle)


@functools.cache
def is_wsl() -> bool:
    """Check if running in Windows Subsystem for Linux (WSL).

    Returns:
        True if running in WSL.
    """
    if os.name != "posix":
        return False
    try:
        with open("/proc/version") as f:
            return "microsoft" in f.read().lower()
    except FileNotFoundError:
        return False


# ─── Golf Suite Specific Accessors ─────────────────────────────


def get_golf_port(default: int = 8000) -> int:
    """Get the Golf Suite server port.

    Reads from GOLF_PORT env var (distinct from GOLF_API_PORT).

    Args:
        default: Default port number.

    Returns:
        Port number.
    """
    return (
        get_env_int("GOLF_PORT", default=default, min_value=1, max_value=65535)
        or default
    )


def get_golf_suite_mode(default: str = "remote") -> str:
    """Get the Golf Suite operating mode.

    Defaults to ``"remote"`` (auth-required) when ``GOLF_SUITE_MODE`` is not
    set.  Set ``GOLF_SUITE_MODE=local`` explicitly to enable auth bypass for
    local development — do not rely on the absence of the variable.

    Args:
        default: Default mode (``"remote"``).

    Returns:
        Mode string (e.g., ``"local"``, ``"remote"``).
    """
    return get_env("GOLF_SUITE_MODE", default=default) or default


def is_auth_disabled() -> bool:
    """Check if authentication is disabled.

    Auth is disabled when ``GOLF_SUITE_MODE=local`` or
    ``GOLF_AUTH_DISABLED=true``.

    Returns:
        True if authentication checks should be skipped.
    """
    return get_golf_suite_mode() == "local" or get_env_bool(
        "GOLF_AUTH_DISABLED", default=False
    )


def get_golf_ui_dist() -> str | None:
    """Get the path to the Golf UI distribution directory.

    Returns:
        Path string or None if not set.
    """
    return get_env("GOLF_UI_DIST")


def is_browser_suppressed() -> bool:
    """Check if auto-opening a browser is suppressed.

    Returns:
        True if ``GOLF_NO_BROWSER=true``.
    """
    return get_env_bool("GOLF_NO_BROWSER", default=False)


def is_headless() -> bool:
    """Check if running in headless mode.

    Returns:
        True if ``HEADLESS=true`` or no DISPLAY on POSIX.
    """
    if get_env_bool("HEADLESS", default=False):
        return True
    return bool(os.name == "posix" and not os.environ.get("DISPLAY"))


def get_display(default: str = ":0") -> str:
    """Get the X11 DISPLAY value.

    Args:
        default: Default display (``":0"``).

    Returns:
        DISPLAY string.
    """
    return get_env("DISPLAY", default=default) or default


def get_dbc_level(default: str = "") -> str:
    """Get the DbC enforcement level string.

    Returns:
        Raw DBC_LEVEL env value (lowercase, stripped).
    """
    return (get_env("DBC_LEVEL", default=default) or default).lower().strip()
