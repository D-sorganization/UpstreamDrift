"""Tests for src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine."""

from __future__ import annotations

import pytest

from src.engines.physics_engines.pinocchio.python import (
    pinocchio_physics_engine as engine_module,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def bindings_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model the module as imported without the Pinocchio bindings."""
    monkeypatch.delattr(engine_module, "pin", raising=False)


@pytest.mark.usefixtures("bindings_unavailable")
def test_load_from_path_without_bindings_is_a_missing_requirement() -> None:
    """Loading must raise ImportError (a requirement), never NameError (#12988)."""
    engine = engine_module.PinocchioPhysicsEngine()

    with pytest.raises(ImportError, match="Pinocchio Python bindings"):
        engine._load_from_path_impl("model.urdf")

    assert engine.model is None


@pytest.mark.usefixtures("bindings_unavailable")
def test_load_from_string_without_bindings_is_a_missing_requirement() -> None:
    """String loading enforces the same precondition as path loading."""
    engine = engine_module.PinocchioPhysicsEngine()

    with pytest.raises(ImportError, match="Pinocchio Python bindings"):
        engine._load_from_string_impl("<robot name='r'/>", "urdf")

    assert engine.model is None
