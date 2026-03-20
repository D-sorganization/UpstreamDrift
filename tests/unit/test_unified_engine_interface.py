"""Tests for engine_core.unified_engine_interface (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.engine_core.unified_engine_interface import (
    EngineManager,
    EngineType,
    UnifiedEngineInterface,
    create_unified_interface,
)


class TestEngineType:
    def test_pendulum_type(self) -> None:
        assert EngineType.PENDULUM.value == "pendulum"

    def test_golf_swing_type(self) -> None:
        assert EngineType.GOLF_SWING_PENDULUM.value == "golf_swing_pendulum"

    def test_has_multiple_types(self) -> None:
        assert len(list(EngineType)) >= 3


class TestEngineManager:
    def test_construction(self) -> None:
        em = EngineManager()
        assert em is not None

    def test_has_get_available_engines(self) -> None:
        em = EngineManager()
        assert hasattr(em, "get_available_engines")


class TestCreateUnifiedInterface:
    def test_returns_unified_engine_interface(self) -> None:
        ui = create_unified_interface()
        assert isinstance(ui, UnifiedEngineInterface)

    def test_get_available_engines(self) -> None:
        ui = create_unified_interface()
        engines = ui.get_available_engines()
        assert isinstance(engines, list)

    def test_no_engine_loaded_initially(self) -> None:
        ui = create_unified_interface()
        assert ui.current_engine is None

    def test_has_engine_manager(self) -> None:
        ui = create_unified_interface()
        assert ui.engine_manager is not None
