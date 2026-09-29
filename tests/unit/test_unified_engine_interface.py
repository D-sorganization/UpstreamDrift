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
    def test_unified_engine_interface_construction(self) -> None:
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

    def test_unified_engine_interface_has_engine_manager(self) -> None:
        ui = create_unified_interface()
        assert ui.engine_manager is not None


class TestIssue2500EngineStringLoading:
    """Issue #2500: load_engine(str) must resolve lowercase enum values."""

    def test_engine_type_enum_values_are_lowercase(self) -> None:
        """EngineType enum values should be lowercase strings."""
        for member in EngineType:
            assert member.value == member.value.lower(), (
                f"EngineType.{member.name}.value must be lowercase, got {member.value!r}"
            )

    def test_load_engine_lowercase_string_invokes_internal_load(self) -> None:
        """load_engine('mujoco') must reach _load_engine with EngineType.MUJOCO."""
        from unittest.mock import MagicMock

        ui = create_unified_interface()
        mock_load = MagicMock()
        ui.engine_manager._load_engine = mock_load
        mock_engine = MagicMock()
        ui.engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        ui.load_engine("mujoco")

        mock_load.assert_called_once()
        called_with = mock_load.call_args[0][0]
        assert called_with == EngineType.MUJOCO, (
            f"_load_engine must be called with EngineType.MUJOCO, got {called_with!r}. "
            "String conversion must use .lower(), not .upper()."
        )

    def test_load_engine_uppercase_string_also_resolves(self) -> None:
        """load_engine('MUJOCO') must also reach _load_engine (case-insensitive)."""
        from unittest.mock import MagicMock

        ui = create_unified_interface()
        mock_load = MagicMock()
        ui.engine_manager._load_engine = mock_load
        mock_engine = MagicMock()
        ui.engine_manager.get_active_physics_engine = MagicMock(
            return_value=mock_engine
        )

        ui.load_engine("MUJOCO")

        mock_load.assert_called_once()
        called_with = mock_load.call_args[0][0]
        assert called_with == EngineType.MUJOCO, (
            f"_load_engine must be called with EngineType.MUJOCO, got {called_with!r}."
        )
