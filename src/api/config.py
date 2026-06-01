"""API configuration defaults and environment overrides.

Configuration precedence (highest to lowest):
  1. Environment variables (ALLOWED_HOSTS, CORS_ORIGINS, API_HOST, API_PORT)
  2. Defaults defined in this module

For server host/port the canonical accessors live in
``src.shared.python.config.environment`` (``get_api_host``, ``get_api_port``).
This module re-exports them for backward compatibility with existing callers
in ``src/api/server.py``.

See also:
  - ``src/config/interim_config.yaml`` -- YAML defaults for CORS origins,
    trusted hosts, rate-limiting (not loaded at runtime; documents intent)
  - ``src/shared/python/config/environment.py`` -- canonical env-var accessors
  - ``src/shared/python/security/env_validator.py`` -- startup validator
"""

from __future__ import annotations

from pydantic import ValidationError

from src.shared.python.config.typed_settings import (
    DEFAULT_ALLOWED_HOSTS as _SETTINGS_DEFAULT_ALLOWED_HOSTS,
)
from src.shared.python.config.typed_settings import (
    DEFAULT_CORS_ORIGINS as _SETTINGS_DEFAULT_CORS_ORIGINS,
)
from src.shared.python.config.typed_settings import (
    DEFAULT_SERVER_HOST as _SETTINGS_DEFAULT_SERVER_HOST,
)
from src.shared.python.config.typed_settings import (
    DEFAULT_SERVER_PORT as _SETTINGS_DEFAULT_SERVER_PORT,
)
from src.shared.python.config.typed_settings import get_settings

MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_UPLOAD_SIZE_MB = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)

HSTS_MAX_AGE_SECONDS = 31536000
DEFAULT_PAGINATION_LIMIT = 100
MAX_POSE_DATA_ENTRIES = 100

VALID_ESTIMATOR_TYPES = {"mediapipe", "openpose", "movenet"}
VALID_EXPORT_FORMATS = {"json"}

MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0
DEFAULT_CONFIDENCE = 0.5

# Re-exported from the canonical typed settings module so there is a single
# source of truth for these defaults (issue #6565).
DEFAULT_ALLOWED_HOSTS = _SETTINGS_DEFAULT_ALLOWED_HOSTS
DEFAULT_CORS_ORIGINS = _SETTINGS_DEFAULT_CORS_ORIGINS


def get_allowed_hosts() -> list[str]:
    """Return allowed hosts with environment overrides.

    Delegates to the canonical :class:`Settings` (env var ``ALLOWED_HOSTS``).
    """
    return get_settings().allowed_hosts


def get_cors_origins() -> list[str]:
    """Return CORS origins with environment overrides.

    Delegates to the canonical :class:`Settings` (env var ``CORS_ORIGINS``).
    """
    origins = get_settings().cors_origins
    if "*" in origins:
        raise ValueError(
            "CORS_ORIGINS must not contain '*' when credentials are enabled (fail-closed)"
        )
    return origins


# Server configuration
# NOTE: These functions read API_HOST / API_PORT (legacy env var names).
# The canonical env vars defined in src.shared.python.config.environment are
# GOLF_API_HOST and GOLF_API_PORT.  A future cleanup task should migrate callers
# to the shared accessors (get_api_host / get_api_port) and retire API_HOST /
# API_PORT -- see issue #2068.
DEFAULT_SERVER_HOST = _SETTINGS_DEFAULT_SERVER_HOST
DEFAULT_SERVER_PORT = _SETTINGS_DEFAULT_SERVER_PORT


def get_server_host() -> str:
    """Get server host from environment or default.

    Environment Variable:
        API_HOST: Server bind address (default: 127.0.0.1)

    Returns:
        Server host address.
    """
    try:
        return get_settings().server_host
    except ValidationError as e:
        import os

        port_str = os.environ.get("API_PORT", str(DEFAULT_SERVER_PORT))
        if "API_PORT" in str(e) or "server_port" in str(e):
            raise ValueError(f"Invalid API_PORT value: {port_str!r}") from e
        raise


def get_server_port() -> int:
    """Get server port from environment or default.

    Environment Variable:
        API_PORT: Server port (default: 8000)

    Returns:
        Server port number.

    Raises:
        ValueError: If API_PORT is not a valid integer or out of range.
    """
    import os

    # Preserve the legacy ``ValueError`` (not pydantic's ``ValidationError``)
    # and the exact message that includes the raw env string.
    port_str = os.environ.get("API_PORT", str(DEFAULT_SERVER_PORT))
    try:
        return get_settings().server_port
    except ValidationError as e:
        raise ValueError(f"Invalid API_PORT value: {port_str!r}") from e
