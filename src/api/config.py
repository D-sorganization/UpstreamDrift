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

import os

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

DEFAULT_ALLOWED_HOSTS = [
    "localhost",
    "127.0.0.1",
    "testserver",  # For FastAPI TestClient
    "*.golfmodelingsuite.com",
]

DEFAULT_CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8080",
    "https://app.golfmodelingsuite.com",
]


def get_allowed_hosts() -> list[str]:
    """Return allowed hosts with environment overrides."""
    env_value = os.getenv("ALLOWED_HOSTS")
    if env_value:
        return [host.strip() for host in env_value.split(",") if host.strip()]
    return DEFAULT_ALLOWED_HOSTS.copy()


def get_cors_origins() -> list[str]:
    """Return CORS origins with environment overrides."""
    env_value = os.getenv("CORS_ORIGINS")
    if env_value:
        return [origin.strip() for origin in env_value.split(",") if origin.strip()]
    return DEFAULT_CORS_ORIGINS.copy()


# Server configuration
# NOTE: These functions read API_HOST / API_PORT (legacy env var names).
# The canonical env vars defined in src.shared.python.config.environment are
# GOLF_API_HOST and GOLF_API_PORT.  A future cleanup task should migrate callers
# to the shared accessors (get_api_host / get_api_port) and retire API_HOST /
# API_PORT -- see issue #2068.
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 8000


def get_server_host() -> str:
    """Get server host from environment or default.

    Environment Variable:
        API_HOST: Server bind address (default: 127.0.0.1)

    Returns:
        Server host address.
    """
    return os.getenv("API_HOST", DEFAULT_SERVER_HOST)


def get_server_port() -> int:
    """Get server port from environment or default.

    Environment Variable:
        API_PORT: Server port (default: 8000)

    Returns:
        Server port number.

    Raises:
        ValueError: If API_PORT is not a valid integer or out of range.
    """
    port_str = os.getenv("API_PORT", str(DEFAULT_SERVER_PORT))
    try:
        port = int(port_str)
        if not (1 <= port <= 65535):
            raise ValueError(f"Invalid API_PORT value: {port_str!r}")
        return port
    except ValueError as e:
        raise ValueError(f"Invalid API_PORT value: {port_str!r}") from e
