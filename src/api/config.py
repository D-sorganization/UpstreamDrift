"""API configuration defaults and environment overrides."""

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
