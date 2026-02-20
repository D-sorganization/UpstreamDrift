"""Tests for the engine plugin registry and discovery system.

TDD: These tests define the contract for the plugin registry before
implementation. They cover:
- Thread-safe registration/unregistration
- Entry-point based plugin discovery
- Engine lifecycle protocol (shutdown)
- Plugin metadata
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.shared.python.engine_core.engine_registry import (
    EngineRegistration,
    EngineType,
)
from src.shared.python.engine_core.plugin_registry import (
    EngineLifecycle,
    EnginePluginMetadata,
    PluginRegistry,
    discover_entry_point_plugins,
)


# ---------------------------------------------------------------------------
# Stub engine for testing
# ---------------------------------------------------------------------------


class _StubEngine:
    """Minimal engine that satisfies PhysicsEngine-like interface for tests."""

    def __init__(self) -> None:
        self._shutdown = False

    @property
    def model_name(self) -> str:
        return "stub"

    def load_from_path(self, path: str) -> None:
        pass

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        pass

    def reset(self) -> None:
        pass

    def step(self, dt: float | None = None) -> None:
        pass

    def forward(self) -> None:
        pass

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(2), np.zeros(2)

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        pass

    def set_control(self, u: np.ndarray) -> None:
        pass

    def get_time(self) -> float:
        return 0.0

    def compute_mass_matrix(self) -> np.ndarray:
        return np.eye(2)

    def compute_bias_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_gravity_forces(self) -> np.ndarray:
        return np.zeros(2)

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        return qacc

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        return None

    def compute_drift_acceleration(self) -> np.ndarray:
        return np.zeros(2)

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        return tau

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(2)

    @property
    def engine_type(self) -> str:
        return "stub"

    def save_checkpoint(self) -> Any:
        return {}

    def restore_checkpoint(self, checkpoint: Any) -> None:
        pass

    def shutdown(self) -> None:
        self._shutdown = True


def _stub_factory() -> _StubEngine:
    return _StubEngine()


# ---------------------------------------------------------------------------
# PluginRegistry tests
# ---------------------------------------------------------------------------


class TestPluginRegistryBasics:
    """Test basic registration and retrieval."""

    def test_register_and_get(self) -> None:
        registry = PluginRegistry()
        reg = EngineRegistration(
            engine_type=EngineType.PENDULUM, factory=_stub_factory
        )
        registry.register(reg)
        assert registry.get(EngineType.PENDULUM) is reg

    def test_get_missing_returns_none(self) -> None:
        registry = PluginRegistry()
        assert registry.get(EngineType.DRAKE) is None

    def test_all_types(self) -> None:
        registry = PluginRegistry()
        reg = EngineRegistration(
            engine_type=EngineType.MUJOCO, factory=_stub_factory
        )
        registry.register(reg)
        assert EngineType.MUJOCO in registry.all_types()

    def test_register_overwrites(self) -> None:
        registry = PluginRegistry()
        reg1 = EngineRegistration(
            engine_type=EngineType.PENDULUM, factory=_stub_factory
        )
        reg2 = EngineRegistration(
            engine_type=EngineType.PENDULUM, factory=_stub_factory
        )
        registry.register(reg1)
        registry.register(reg2)
        assert registry.get(EngineType.PENDULUM) is reg2


class TestPluginRegistryUnregister:
    """Test unregistration support."""

    def test_unregister_existing(self) -> None:
        registry = PluginRegistry()
        reg = EngineRegistration(
            engine_type=EngineType.PENDULUM, factory=_stub_factory
        )
        registry.register(reg)
        result = registry.unregister(EngineType.PENDULUM)
        assert result is True
        assert registry.get(EngineType.PENDULUM) is None

    def test_unregister_missing_returns_false(self) -> None:
        registry = PluginRegistry()
        result = registry.unregister(EngineType.DRAKE)
        assert result is False

    def test_unregister_removes_from_all_types(self) -> None:
        registry = PluginRegistry()
        reg = EngineRegistration(
            engine_type=EngineType.MUJOCO, factory=_stub_factory
        )
        registry.register(reg)
        registry.unregister(EngineType.MUJOCO)
        assert EngineType.MUJOCO not in registry.all_types()


class TestPluginRegistryThreadSafety:
    """Test thread-safe operations."""

    def test_concurrent_register(self) -> None:
        registry = PluginRegistry()
        errors: list[Exception] = []

        def register_engine(engine_type: EngineType) -> None:
            try:
                reg = EngineRegistration(
                    engine_type=engine_type, factory=_stub_factory
                )
                registry.register(reg)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=register_engine, args=(et,))
            for et in [EngineType.MUJOCO, EngineType.DRAKE, EngineType.PINOCCHIO]
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(registry.all_types()) == 3

    def test_concurrent_read_write(self) -> None:
        registry = PluginRegistry()
        reg = EngineRegistration(
            engine_type=EngineType.PENDULUM, factory=_stub_factory
        )
        registry.register(reg)
        errors: list[Exception] = []

        def reader() -> None:
            try:
                for _ in range(100):
                    registry.get(EngineType.PENDULUM)
                    registry.all_types()
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                for i in range(100):
                    new_reg = EngineRegistration(
                        engine_type=EngineType.PENDULUM, factory=_stub_factory
                    )
                    registry.register(new_reg)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        threads.append(threading.Thread(target=writer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ---------------------------------------------------------------------------
# EnginePluginMetadata tests
# ---------------------------------------------------------------------------


class TestEnginePluginMetadata:
    """Test plugin metadata dataclass."""

    def test_metadata_creation(self) -> None:
        meta = EnginePluginMetadata(
            name="test-engine",
            version="1.0.0",
            engine_type=EngineType.PENDULUM,
            author="Test Author",
            description="A test engine",
        )
        assert meta.name == "test-engine"
        assert meta.version == "1.0.0"
        assert meta.engine_type is EngineType.PENDULUM

    def test_metadata_defaults(self) -> None:
        meta = EnginePluginMetadata(
            name="minimal",
            version="0.1.0",
            engine_type=EngineType.MUJOCO,
        )
        assert meta.author == ""
        assert meta.description == ""
        assert meta.requires == []

    def test_register_with_metadata(self) -> None:
        registry = PluginRegistry()
        meta = EnginePluginMetadata(
            name="test-engine",
            version="1.0.0",
            engine_type=EngineType.PENDULUM,
        )
        registry.register_plugin(
            engine_type=EngineType.PENDULUM,
            factory=_stub_factory,
            metadata=meta,
        )
        assert registry.get(EngineType.PENDULUM) is not None
        assert registry.get_metadata(EngineType.PENDULUM) is meta

    def test_get_metadata_missing_returns_none(self) -> None:
        registry = PluginRegistry()
        assert registry.get_metadata(EngineType.DRAKE) is None


# ---------------------------------------------------------------------------
# EngineLifecycle tests
# ---------------------------------------------------------------------------


class TestEngineLifecycle:
    """Test engine lifecycle management."""

    def test_lifecycle_tracks_engine(self) -> None:
        lifecycle = EngineLifecycle()
        engine = _StubEngine()
        lifecycle.track(EngineType.PENDULUM, engine)
        assert lifecycle.is_active(EngineType.PENDULUM)

    def test_lifecycle_shutdown_single(self) -> None:
        lifecycle = EngineLifecycle()
        engine = _StubEngine()
        lifecycle.track(EngineType.PENDULUM, engine)
        lifecycle.shutdown(EngineType.PENDULUM)
        assert engine._shutdown is True
        assert not lifecycle.is_active(EngineType.PENDULUM)

    def test_lifecycle_shutdown_all(self) -> None:
        lifecycle = EngineLifecycle()
        engine1 = _StubEngine()
        engine2 = _StubEngine()
        lifecycle.track(EngineType.PENDULUM, engine1)
        lifecycle.track(EngineType.MUJOCO, engine2)
        lifecycle.shutdown_all()
        assert engine1._shutdown is True
        assert engine2._shutdown is True
        assert not lifecycle.is_active(EngineType.PENDULUM)
        assert not lifecycle.is_active(EngineType.MUJOCO)

    def test_lifecycle_shutdown_missing_is_noop(self) -> None:
        lifecycle = EngineLifecycle()
        lifecycle.shutdown(EngineType.DRAKE)  # Should not raise

    def test_lifecycle_engine_without_shutdown(self) -> None:
        """Engines without shutdown() method should still be cleanable."""
        lifecycle = EngineLifecycle()
        engine = MagicMock(spec=[])  # No shutdown method
        lifecycle.track(EngineType.MUJOCO, engine)
        lifecycle.shutdown(EngineType.MUJOCO)
        assert not lifecycle.is_active(EngineType.MUJOCO)


# ---------------------------------------------------------------------------
# Entry-point discovery tests
# ---------------------------------------------------------------------------


class TestEntryPointDiscovery:
    """Test entry-point based plugin discovery."""

    def test_discover_with_no_plugins(self) -> None:
        """When no entry points are installed, should return empty list."""
        with patch(
            "src.shared.python.engine_core.plugin_registry.entry_points",
            return_value=[],
        ):
            plugins = discover_entry_point_plugins()
            assert plugins == []

    def test_discover_with_valid_plugin(self) -> None:
        """Should load and return valid plugin registrations."""
        mock_ep = MagicMock()
        mock_ep.name = "test_engine"
        mock_ep.load.return_value = {
            "engine_type": EngineType.PENDULUM,
            "factory": _stub_factory,
            "metadata": EnginePluginMetadata(
                name="test",
                version="1.0.0",
                engine_type=EngineType.PENDULUM,
            ),
        }

        with patch(
            "src.shared.python.engine_core.plugin_registry.entry_points",
            return_value=[mock_ep],
        ):
            plugins = discover_entry_point_plugins()
            assert len(plugins) == 1
            assert plugins[0]["engine_type"] is EngineType.PENDULUM

    def test_discover_skips_broken_plugin(self) -> None:
        """Should skip plugins that fail to load without crashing."""
        mock_ep = MagicMock()
        mock_ep.name = "broken_engine"
        mock_ep.load.side_effect = ImportError("missing dependency")

        with patch(
            "src.shared.python.engine_core.plugin_registry.entry_points",
            return_value=[mock_ep],
        ):
            plugins = discover_entry_point_plugins()
            assert plugins == []
