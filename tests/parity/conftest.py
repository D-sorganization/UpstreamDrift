"""Parity test fixtures.

Provides FastAPI test client and physics engine fixtures for
engine-vs-API consistency testing.
"""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient
    from src.api.server import app
except ImportError as _exc:
    pytest.skip(f"API server deps not available: {_exc}", allow_module_level=True)


@pytest.fixture(scope="module")
def client():
    """FastAPI test client with full app lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def pendulum_engine():
    """Fresh PendulumPhysicsEngine instance."""
    from src.engines.physics_engines.pendulum.python.pendulum_physics_engine import (
        PendulumPhysicsEngine,
    )

    engine = PendulumPhysicsEngine()
    return engine
