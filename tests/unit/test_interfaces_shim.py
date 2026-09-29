"""Tests for interfaces shim module (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.interfaces import PhysicsEngine, RecorderInterface


class TestInterfacesShim:
    def test_physics_engine_importable(self) -> None:
        assert PhysicsEngine is not None

    def test_recorder_interface_importable(self) -> None:
        assert RecorderInterface is not None

    def test_both_in_all(self) -> None:
        import src.shared.python.interfaces as m

        assert "PhysicsEngine" in m.__all__
        assert "RecorderInterface" in m.__all__
