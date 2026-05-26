"""Parity test fixtures.

Provides FastAPI test client and physics engine fixtures for
engine-vs-API consistency testing.
"""

from __future__ import annotations

from collections.abc import Generator
from typing import Any

import pytest

_PARITY_API_IMPORT_ERROR: ImportError | None = None

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError as exc:
    TestClient = Any  # type: ignore[assignment,misc]
    app = None  # type: ignore[assignment]
    _PARITY_API_IMPORT_ERROR = exc


@pytest.fixture(scope="module")
def client() -> Generator[TestClient, None, None]:
    """FastAPI test client with full app lifespan."""
    if app is None:
        detail = (
            str(_PARITY_API_IMPORT_ERROR) if _PARITY_API_IMPORT_ERROR else "unknown"
        )
        pytest.skip(f"API server deps not available: {detail}")
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pendulum_engine() -> Any:
    """Fresh PendulumPhysicsEngine instance."""
    from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
        PendulumPhysicsEngine,
    )

    engine = PendulumPhysicsEngine()
    return engine
