"""Discovery/selection availability guards for EngineManager.

Covers issues #6880 (JaxSim may not be selected/loaded when the ``jaxsim``
package is absent) and #6884 (discovery must not report runtime-backed adapter
directories as ``AVAILABLE`` when the runtime dependency is missing).

These tests exercise the guard logic directly and do **not** require ``jax`` or
``jaxsim`` to be installed.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.shared.python.engine_core import engine_availability, engine_manager
from src.shared.python.engine_core.engine_availability import (
    EngineStatus as AvailabilityStatus,
)
from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
)


def _make_engine_dirs(root: Path, *engine_names: str) -> None:
    """Create ``src/engines/physics_engines/<name>`` directories under root."""
    base = root / "src" / "engines" / "physics_engines"
    for name in engine_names:
        (base / name).mkdir(parents=True, exist_ok=True)


def test_runtime_backed_engine_not_available_without_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6884: a present adapter dir must not be AVAILABLE when runtime missing."""
    monkeypatch.setattr(
        engine_manager,
        "get_runtime_engine_status",
        lambda name: AvailabilityStatus.NOT_INSTALLED,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_engine_dirs(root, "mujoco", "drake")
        manager = EngineManager(suite_root=root)
        assert EngineType.MUJOCO not in manager.get_available_engines()
        assert EngineType.DRAKE not in manager.get_available_engines()
        assert manager.get_engine_status(EngineType.MUJOCO) == EngineStatus.UNAVAILABLE


def test_runtime_backed_engine_available_when_package_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6884: adapter dir + importable runtime => AVAILABLE."""
    monkeypatch.setattr(
        engine_manager,
        "get_runtime_engine_status",
        lambda name: AvailabilityStatus.AVAILABLE,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_engine_dirs(root, "mujoco")
        manager = EngineManager(suite_root=root)
        assert EngineType.MUJOCO in manager.get_available_engines()
        assert manager.get_engine_status(EngineType.MUJOCO) == EngineStatus.AVAILABLE


def test_jaxsim_not_available_when_package_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6880: JaxSim adapter dir present but jaxsim package absent => not available."""

    def _status(name: str) -> AvailabilityStatus:
        if name == "jaxsim.api":
            return AvailabilityStatus.NOT_INSTALLED
        return AvailabilityStatus.AVAILABLE

    monkeypatch.setattr(engine_manager, "get_runtime_engine_status", _status)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_engine_dirs(root, "jaxsim")
        manager = EngineManager(suite_root=root)
        assert EngineType.JAXSIM not in manager.get_available_engines()
        assert manager.get_engine_status(EngineType.JAXSIM) == EngineStatus.UNAVAILABLE


def test_switch_to_jaxsim_fails_when_package_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6880: switch_engine(JAXSIM) returns False and leaves no active engine."""
    monkeypatch.setattr(
        engine_manager,
        "get_runtime_engine_status",
        lambda name: AvailabilityStatus.NOT_INSTALLED,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_engine_dirs(root, "jaxsim")
        manager = EngineManager(suite_root=root)
        assert manager.switch_engine(EngineType.JAXSIM) is False
        assert manager.active_physics_engine is None


def test_path_only_engines_unaffected_by_runtime_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Engines without a runtime Python dependency stay path/probe-gated.

    Pendulum-family engines have no importable runtime package, so their
    availability must depend on source presence alone even when the runtime
    layer reports NOT_INSTALLED for everything.
    """
    monkeypatch.setattr(
        engine_manager,
        "get_runtime_engine_status",
        lambda name: AvailabilityStatus.NOT_INSTALLED,
    )
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "src" / "engines" / "pendulum_models").mkdir(parents=True)
        manager = EngineManager(suite_root=root)
        assert EngineType.PENDULUM in manager.get_available_engines()


def test_runtime_status_helper_maps_engine_types() -> None:
    """The runtime-status helper resolves real EngineType values without error."""
    # Pure-rigid engines without a dedicated importable package map to None and
    # are treated as path-gated (no runtime requirement).
    assert engine_manager.runtime_dependency_name(EngineType.MUJOCO) == "mujoco"
    assert engine_manager.runtime_dependency_name(EngineType.JAXSIM) == "jaxsim.api"
    assert engine_manager.runtime_dependency_name(EngineType.PENDULUM) is None
    # Sanity: the resolved names are accepted by the availability layer.
    assert engine_availability.get_engine_status("mujoco") in AvailabilityStatus


def test_jaxsim_not_available_when_api_submodule_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#6887: partial jaxsim install (jaxsim ok, jaxsim.api missing) => not available.

    Discovery must probe the same surface the loader requires (jaxsim.api),
    so an incomplete install is caught before switch_engine, not during.
    """

    def _status(name: str) -> AvailabilityStatus:
        # jaxsim top-level importable; jaxsim.api is not
        if name == "jaxsim.api":
            return AvailabilityStatus.NOT_INSTALLED
        return AvailabilityStatus.AVAILABLE

    monkeypatch.setattr(engine_manager, "get_runtime_engine_status", _status)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _make_engine_dirs(root, "jaxsim")
        manager = EngineManager(suite_root=root)
        assert EngineType.JAXSIM not in manager.get_available_engines()
        assert manager.get_engine_status(EngineType.JAXSIM) == EngineStatus.UNAVAILABLE
