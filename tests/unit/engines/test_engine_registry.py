"""Tests for src.shared.python.engine_core.engine_registry (Issues #1949, #1744)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from src.shared.python.engine_core.engine_registry import (
    EngineRegistration,
    EngineRegistry,
    EngineStatus,
    EngineType,
    get_registry,
)

# ---------------------------------------------------------------------------
# EngineType enum
# ---------------------------------------------------------------------------


class TestEngineType:
    def test_mujoco_exists(self) -> None:
        assert EngineType.MUJOCO

    def test_pinocchio_exists(self) -> None:
        assert EngineType.PINOCCHIO

    def test_drake_exists(self) -> None:
        assert EngineType.DRAKE

    def test_engine_registry_values_are_strings(self) -> None:
        for member in EngineType:
            assert isinstance(member.value, str)

    def test_all_distinct(self) -> None:
        values = [m.value for m in EngineType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# EngineStatus enum
# ---------------------------------------------------------------------------


class TestEngineRegistryStatus:
    def test_available_exists(self) -> None:
        assert EngineStatus.AVAILABLE

    def test_error_exists(self) -> None:
        assert EngineStatus.ERROR

    def test_all_distinct(self) -> None:
        values = [m.value for m in EngineStatus]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# EngineRegistration dataclass
# ---------------------------------------------------------------------------


class TestEngineRegistration:
    def _make_factory(self) -> MagicMock:
        return MagicMock()

    def test_requires_binary_defaults_empty(self) -> None:
        reg = EngineRegistration(
            engine_type=EngineType.MUJOCO, factory=self._make_factory()
        )
        assert reg.requires_binary == []

    def test_probe_class_defaults_none(self) -> None:
        reg = EngineRegistration(
            engine_type=EngineType.MUJOCO, factory=self._make_factory()
        )
        assert reg.probe_class is None

    def test_registration_path_defaults_none(self) -> None:
        reg = EngineRegistration(
            engine_type=EngineType.DRAKE, factory=self._make_factory()
        )
        assert reg.registration_path is None

    def test_stores_engine_type(self) -> None:
        reg = EngineRegistration(
            engine_type=EngineType.PINOCCHIO, factory=self._make_factory()
        )
        assert reg.engine_type == EngineType.PINOCCHIO

    def test_accepts_path(self) -> None:
        reg = EngineRegistration(
            engine_type=EngineType.MUJOCO,
            factory=self._make_factory(),
            registration_path=Path("/tmp/fake"),
        )
        assert reg.registration_path == Path("/tmp/fake")


# ---------------------------------------------------------------------------
# EngineRegistry
# ---------------------------------------------------------------------------


class TestEngineRegistry:
    def _make_registration(
        self, engine_type: EngineType = EngineType.MUJOCO
    ) -> EngineRegistration:
        return EngineRegistration(engine_type=engine_type, factory=MagicMock())

    def test_starts_empty(self) -> None:
        registry = EngineRegistry()
        assert registry.all_types() == []

    def test_register_adds_engine(self) -> None:
        registry = EngineRegistry()
        registry.register(self._make_registration(EngineType.MUJOCO))
        assert EngineType.MUJOCO in registry.all_types()

    def test_get_returns_registration(self) -> None:
        registry = EngineRegistry()
        reg = self._make_registration(EngineType.DRAKE)
        registry.register(reg)
        result = registry.get(EngineType.DRAKE)
        assert result is reg

    def test_get_missing_returns_none(self) -> None:
        registry = EngineRegistry()
        assert registry.get(EngineType.MUJOCO) is None

    def test_register_multiple(self) -> None:
        registry = EngineRegistry()
        registry.register(self._make_registration(EngineType.MUJOCO))
        registry.register(self._make_registration(EngineType.PINOCCHIO))
        assert len(registry.all_types()) == 2

    def test_re_register_overwrites(self) -> None:
        registry = EngineRegistry()
        reg1 = self._make_registration(EngineType.MUJOCO)
        reg2 = self._make_registration(EngineType.MUJOCO)
        registry.register(reg1)
        registry.register(reg2)
        assert registry.get(EngineType.MUJOCO) is reg2
        assert len(registry.all_types()) == 1

    def test_all_types_returns_list(self) -> None:
        registry = EngineRegistry()
        assert isinstance(registry.all_types(), list)

    def test_register_none_raises(self) -> None:
        registry = EngineRegistry()
        with pytest.raises((AssertionError, TypeError, ValueError)):
            registry.register(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# get_registry — module-level singleton
# ---------------------------------------------------------------------------


class TestGetRegistry:
    def test_returns_engine_registry(self) -> None:
        result = get_registry()
        assert isinstance(result, EngineRegistry)

    def test_returns_same_instance(self) -> None:
        assert get_registry() is get_registry()
