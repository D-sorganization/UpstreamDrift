"""Lazy engine discovery must never import heavy runtimes (#8934).

``EngineManager._discover_engines`` used to really import
pydrake.all/jaxsim/mujoco/pinocchio/opensim/myosuite via the deep availability
probe, costing 8-25 s and >1 GB RSS on every launcher AND API start. Discovery
now uses a metadata-only ``importlib.util.find_spec`` presence probe
(:func:`engine_availability.is_dependency_present`); the real import is
deferred to ``EngineManager._ensure_runtime_importable`` on the actual launch
path.
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import pytest

from src.shared.python.engine_core import engine_availability, engine_manager
from src.shared.python.engine_core.engine_availability import (
    is_dependency_present,
    reset_engine_status_cache,
)
from src.shared.python.engine_core.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
    runtime_dependency_name,
)

pytestmark = pytest.mark.unit

# Top-level packages of every runtime-backed engine dependency. Importing any
# of these during discovery is the regression under test.
_HEAVY_MODULES = ("pydrake", "pinocchio", "jaxsim", "opensim", "myosuite")


@pytest.fixture(autouse=True)
def _clean_probe_caches() -> None:
    reset_engine_status_cache()
    yield
    reset_engine_status_cache()


def _make_engine_dirs(root: Path, *engine_names: str) -> None:
    base = root / "src" / "engines" / "physics_engines"
    for name in engine_names:
        (base / name).mkdir(parents=True, exist_ok=True)


def _build_manager(tmp_path: Path) -> EngineManager:
    _make_engine_dirs(tmp_path, "mujoco", "drake", "pinocchio", "jaxsim")
    return EngineManager(suite_root=tmp_path)


class TestDiscoveryNeverImportsRuntimes:
    """DbC postcondition: the presence probe adds nothing to sys.modules."""

    def test_discovery_does_not_import_heavy_runtimes(self, tmp_path: Path) -> None:
        before = set(sys.modules)
        _build_manager(tmp_path)
        newly_imported = set(sys.modules) - before
        leaked = {
            mod
            for mod in newly_imported
            for heavy in _HEAVY_MODULES
            if mod == heavy or mod.startswith(f"{heavy}.")
        }
        assert not leaked, f"discovery imported heavy runtimes: {sorted(leaked)}"

    def test_presence_probe_postcondition_no_import(self) -> None:
        # jaxsim is not expected to be importable here, but the contract is
        # unconditional: probing must not add the module to sys.modules.
        for dep in ("jaxsim.api", "opensim", "myosuite", "drake"):
            top = engine_availability._PRESENCE_SPEC_OVERRIDES.get(dep, dep).partition(
                "."
            )[0]
            was_imported = top in sys.modules
            is_dependency_present(dep)
            assert (top in sys.modules) == was_imported

    def test_deep_probe_not_called_during_discovery(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(name: str) -> None:
            raise AssertionError(f"deep probe invoked during discovery for {name!r}")

        monkeypatch.setattr(engine_manager, "get_runtime_engine_status", _boom)
        _build_manager(tmp_path)  # must not raise


class TestAvailabilityMatchesFindSpec:
    """Discovery availability must equal find_spec truth per dependency."""

    @pytest.mark.parametrize(
        ("engine_type", "spec_name"),
        [
            (EngineType.MUJOCO, "mujoco"),
            (EngineType.DRAKE, "pydrake"),
            (EngineType.PINOCCHIO, "pinocchio"),
            (EngineType.JAXSIM, "jaxsim"),
        ],
    )
    def test_status_matches_find_spec(
        self, tmp_path: Path, engine_type: EngineType, spec_name: str
    ) -> None:
        manager = _build_manager(tmp_path)
        try:
            expected_present = importlib.util.find_spec(spec_name) is not None
        except (ImportError, ValueError):
            expected_present = False
        expected = (
            EngineStatus.AVAILABLE if expected_present else EngineStatus.UNAVAILABLE
        )
        assert manager.get_engine_status(engine_type) == expected

    def test_is_dependency_present_maps_drake_to_pydrake(self) -> None:
        try:
            expected = importlib.util.find_spec("pydrake") is not None
        except (ImportError, ValueError):
            expected = False
        assert is_dependency_present("drake") == expected

    def test_is_dependency_present_absent_package(self) -> None:
        assert is_dependency_present("definitely_not_a_real_package_8934") is False

    def test_is_dependency_present_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="import_name"):
            is_dependency_present("")


class TestDeepImportDeferredToLaunch:
    """The real import (with its error handling) runs only at launch."""

    def test_load_engine_runs_deep_probe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        manager = _build_manager(tmp_path)
        probed: list[str] = []

        def _fake_status(name: str) -> engine_availability.EngineStatus:
            probed.append(name)
            return engine_availability.EngineStatus.NOT_INSTALLED

        monkeypatch.setattr(engine_manager, "get_runtime_engine_status", _fake_status)
        from src.shared.python.core.error_utils import EngineLaunchError

        with pytest.raises(EngineLaunchError):
            manager._load_engine(EngineType.DRAKE)
        assert probed == [runtime_dependency_name(EngineType.DRAKE)]
        assert manager.get_engine_status(EngineType.DRAKE) == EngineStatus.ERROR


class TestColdStartBudget:
    def test_discovery_completes_within_budget(self, tmp_path: Path) -> None:
        """Full discovery over all engines must be fast (< 2 s).

        Generous budget: on this machine post-fix discovery is ~10-200 ms
        (metadata probes only); the 2 s ceiling absorbs slow CI file systems
        while still failing hard if a real runtime import (3-8 s for
        pydrake.all alone) sneaks back onto the discovery path.
        """
        _build_manager(tmp_path)  # warm loader imports outside the budget
        reset_engine_status_cache()
        start = time.perf_counter()
        _build_manager(tmp_path)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"engine discovery took {elapsed:.2f}s (budget 2 s)"
