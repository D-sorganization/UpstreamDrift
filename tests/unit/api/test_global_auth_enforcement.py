"""Unit tests for global route authentication enforcement (Issue #6636).

Verifies that:
1. In cloud mode (remote), non-public routes (like engines, datasets, physics)
   require a valid authentication Bearer token and return 401 otherwise.
2. Health and observability endpoints (core health, readyz) are public and do not require auth.
3. In local mode (local), all routes bypass authentication.
"""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest
from fastapi.testclient import TestClient


def get_cloud_env() -> dict[str, str]:
    return {
        "GOLF_SUITE_MODE": "remote",
        "GOLF_AUTH_DISABLED": "false",
        "ENVIRONMENT": "development",
        "API_PORT": "8000",
    }


def get_local_env() -> dict[str, str]:
    return {
        "GOLF_SUITE_MODE": "local",
        "GOLF_AUTH_DISABLED": "true",
        "ENVIRONMENT": "development",
        "API_PORT": "8000",
    }


@pytest.fixture
def clean_import_app():
    """Import the main FastAPI app under clean env configurations."""
    import sys

    # Clear cached modules to force route registration with current env settings
    modules_to_reload = [
        "src.shared.python.config.environment",
        "src.shared.python.config.typed_settings",
        "src.api.config",
        "src.api.route_registry",
        "src.api.server",
    ]
    cached_states = {}
    for mod in modules_to_reload:
        if mod in sys.modules:
            cached_states[mod] = sys.modules[mod]
            del sys.modules[mod]

    # Reload the environment config first to clear functools caches
    from src.shared.python.config import environment as env_mod

    env_mod.get_environment.cache_clear()

    try:
        from src.api.server import app

        yield app
    finally:
        # Restore sys.modules to prevent breaking other tests
        for mod, value in cached_states.items():
            sys.modules[mod] = value
        env_mod.get_environment.cache_clear()


def test_cloud_mode_requires_auth(clean_import_app) -> None:
    """In cloud mode, non-public endpoints return 401 when unauthenticated."""
    with (
        patch.dict(os.environ, get_cloud_env()),
        TestClient(clean_import_app) as client,
    ):
        # Non-public endpoints must fail with 401
        response = client.get("/api/v1/engines")
        assert response.status_code == 401
        assert "not authenticated" in response.json()["detail"].lower()

        response = client.get("/api/v1/tools/data-explorer/datasets")
        assert response.status_code == 401

        # Public endpoints must pass with 200/503
        response = client.get("/health")
        assert response.status_code == 200

        response = client.get("/readyz")
        # readyz returns 200 or 503 depending on engine warmup, but never 401
        assert response.status_code in (200, 503)


def test_local_mode_bypasses_auth(clean_import_app) -> None:
    """In local mode, all endpoints bypass auth (no 401)."""
    with (
        patch.dict(os.environ, get_local_env()),
        TestClient(clean_import_app) as client,
    ):
        # Non-public endpoints must not return 401
        response = client.get("/api/v1/engines")
        assert response.status_code != 401

        response = client.get("/api/v1/tools/data-explorer/datasets")
        assert response.status_code != 401

        response = client.get("/health")
        assert response.status_code == 200
