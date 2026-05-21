"""Coverage for src.engines.loaders — engines not covered elsewhere.

Mocks heavy engine dependencies (MuJoCo, OpenSim, MyoSuite, pydrake, MATLAB).
Pure-Python — no real physics engines spun up.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.shared.python.engine_core.engine_probes as engine_probes_mod
from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.engine_core.engine_probes import EngineProbe
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.interfaces import PhysicsEngine

_PROBE_SPEC = [
    "is_available",
    "diagnostic_message",
    "get_fix_instructions",
    "details",
]


def _probe_mock(available: bool = True) -> MagicMock:
    cls = MagicMock()
    instance = MagicMock(spec=EngineProbe)
    cls.return_value = instance
    result = MagicMock(spec=_PROBE_SPEC)
    result.is_available.return_value = available
    result.diagnostic_message = "missing"
    result.get_fix_instructions.return_value = "install it"
    instance.probe.return_value = result
    return cls


# ---------------------------------------------------------------------------
# LOADER_MAP completeness
# ---------------------------------------------------------------------------


def test_loader_map_has_all_engine_types() -> None:
    from src.engines.loaders import LOADER_MAP

    expected = {
        EngineType.MUJOCO,
        EngineType.DRAKE,
        EngineType.PINOCCHIO,
        EngineType.OPENSIM,
        EngineType.MYOSIM,
        EngineType.PENDULUM,
        EngineType.GOLF_SWING_PENDULUM,
        EngineType.PUTTING_GREEN,
        EngineType.MATLAB_3D,
    }
    assert expected.issubset(set(LOADER_MAP))


# ---------------------------------------------------------------------------
# OpenSim loader
# ---------------------------------------------------------------------------


def test_load_opensim_engine_success(tmp_path: Path) -> None:
    from src.engines.loaders import load_opensim_engine

    engine = MagicMock(spec=PhysicsEngine)
    engine_cls = MagicMock(return_value=engine)

    opensim_mod = MagicMock(spec=["OpenSimPhysicsEngine"])
    opensim_mod.OpenSimPhysicsEngine = engine_cls

    with (
        patch.dict(
            sys.modules,
            {
                "src.engines.physics_engines.opensim.python.opensim_physics_engine": opensim_mod,  # noqa: E501
            },
        ),
        patch.object(engine_probes_mod, "OpenSimProbe", _probe_mock(True)),
    ):
        result = load_opensim_engine(tmp_path)
    assert result is engine


def test_load_opensim_engine_probe_fails(tmp_path: Path) -> None:
    from src.engines.loaders import load_opensim_engine

    engine_cls = MagicMock()
    opensim_mod = MagicMock(spec=["OpenSimPhysicsEngine"])
    opensim_mod.OpenSimPhysicsEngine = engine_cls
    with (
        patch.dict(
            sys.modules,
            {
                "src.engines.physics_engines.opensim.python.opensim_physics_engine": opensim_mod,  # noqa: E501
            },
        ),
        patch.object(engine_probes_mod, "OpenSimProbe", _probe_mock(False)),
        pytest.raises(GolfModelingError, match="OpenSim (not ready|requirements)"),
    ):
        load_opensim_engine(tmp_path)


# ---------------------------------------------------------------------------
# MyoSim loader
# ---------------------------------------------------------------------------


def test_load_myosim_engine_success(tmp_path: Path) -> None:
    from src.engines.loaders import load_myosim_engine

    engine = MagicMock(spec=PhysicsEngine)
    engine_cls = MagicMock(return_value=engine)
    myosim_mod = MagicMock(spec=["MyoSuitePhysicsEngine"])
    myosim_mod.MyoSuitePhysicsEngine = engine_cls

    with (
        patch.dict(
            sys.modules,
            {
                "src.engines.physics_engines.myosuite.python.myosuite_physics_engine": myosim_mod,  # noqa: E501
            },
        ),
        patch.object(engine_probes_mod, "MyoSimProbe", _probe_mock(True)),
    ):
        result = load_myosim_engine(tmp_path)
    assert result is engine


# ---------------------------------------------------------------------------
# Pendulum loader
# ---------------------------------------------------------------------------


def test_load_pendulum_engine_success(tmp_path: Path) -> None:
    from src.engines.loaders import load_pendulum_engine

    engine = MagicMock(spec=PhysicsEngine)
    engine_cls = MagicMock(return_value=engine)
    pend_mod = MagicMock(spec=["PendulumPhysicsEngine"])
    pend_mod.PendulumPhysicsEngine = engine_cls
    with patch.dict(
        sys.modules,
        {
            "src.engines.physics_engines.pendulum.python.pendulum_physics_engine": pend_mod,  # noqa: E501
        },
    ):
        result = load_pendulum_engine(tmp_path)
    assert result is engine


def test_load_pendulum_engine_import_error_raises(tmp_path: Path) -> None:
    from src.engines.loaders import load_pendulum_engine

    real_import = __import__

    def fake_import(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if "pendulum_physics_engine" in name:
            raise ImportError("nope")
        return real_import(name, *a, **kw)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(GolfModelingError, match="Pendulum engine not found"),
    ):
        load_pendulum_engine(tmp_path)


# ---------------------------------------------------------------------------
# Golf Swing Pendulum loader
# ---------------------------------------------------------------------------


def test_load_golf_swing_pendulum_success(tmp_path: Path) -> None:
    from src.engines.loaders import load_golf_swing_pendulum_engine

    engine = MagicMock(spec=PhysicsEngine)
    engine_cls = MagicMock(return_value=engine)
    engine_cls.is_available = MagicMock(return_value=True)
    mod = MagicMock(spec=["GolfSwingPendulumEngine"])
    mod.GolfSwingPendulumEngine = engine_cls

    with patch.dict(
        sys.modules,
        {
            "src.engines.physics_engines.pendulum.python.golf_swing_physics_engine": mod,  # noqa: E501
        },
    ):
        result = load_golf_swing_pendulum_engine(tmp_path)
    assert result is engine


def test_load_golf_swing_pendulum_unavailable_raises(tmp_path: Path) -> None:
    from src.engines.loaders import load_golf_swing_pendulum_engine

    engine_cls = MagicMock()
    engine_cls.is_available = MagicMock(return_value=False)
    mod = MagicMock(spec=["GolfSwingPendulumEngine"])
    mod.GolfSwingPendulumEngine = engine_cls

    with (
        patch.dict(
            sys.modules,
            {
                "src.engines.physics_engines.pendulum.python.golf_swing_physics_engine": mod,  # noqa: E501
            },
        ),
        pytest.raises(GolfModelingError, match="vendor/ud-tools"),
    ):
        load_golf_swing_pendulum_engine(tmp_path)


# ---------------------------------------------------------------------------
# Putting Green loader
# ---------------------------------------------------------------------------


def test_load_putting_green_success(tmp_path: Path) -> None:
    from src.engines.loaders import load_putting_green_engine

    sim = MagicMock()
    sim_cls = MagicMock(return_value=sim)
    mod = MagicMock(spec=["PuttingGreenSimulator"])
    mod.PuttingGreenSimulator = sim_cls
    with patch.dict(
        sys.modules,
        {
            "src.engines.physics_engines.putting_green.python.simulator": mod,
        },
    ):
        result = load_putting_green_engine(tmp_path)
    assert result is sim


def test_load_putting_green_import_error_raises(tmp_path: Path) -> None:
    from src.engines.loaders import load_putting_green_engine

    real_import = __import__

    def fake_import(name: str, *a, **kw):  # type: ignore[no-untyped-def]
        if "putting_green.python.simulator" in name:
            raise ImportError("nope")
        return real_import(name, *a, **kw)

    with (
        patch("builtins.__import__", side_effect=fake_import),
        pytest.raises(GolfModelingError, match="Putting Green engine not found"),
    ):
        load_putting_green_engine(tmp_path)


# ---------------------------------------------------------------------------
# MATLAB_3D loader edge cases
# ---------------------------------------------------------------------------


def test_load_matlab_3d_requires_path_type() -> None:
    from src.engines.loaders import load_matlab_3d_engine

    with pytest.raises(TypeError, match="suite_root must be a Path"):
        load_matlab_3d_engine("not-a-path")  # type: ignore[arg-type]


def test_load_matlab_3d_model_missing_returns_unloaded(tmp_path: Path) -> None:
    """When the default .slx file does not exist the loader emits a warning
    and returns the adapter unloaded."""
    from src.engines.loaders import load_matlab_3d_engine

    engine = load_matlab_3d_engine(tmp_path)
    assert engine is not None


def test_load_matlab_3d_load_error_propagates(tmp_path: Path) -> None:
    """If load_from_path raises, loader wraps it as GolfModelingError."""
    from src.engines.loaders import DEFAULT_MATLAB_3D_SLX_RELPATH

    model_path = tmp_path / DEFAULT_MATLAB_3D_SLX_RELPATH
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text("<fake slx>")

    fake_adapter_instance = MagicMock(spec=PhysicsEngine)
    fake_adapter_instance.load_from_path.side_effect = RuntimeError("boom")
    fake_adapter_cls = MagicMock(return_value=fake_adapter_instance)

    fake_simscape_mod = MagicMock(spec=["SimscapeAdapter"])
    fake_simscape_mod.SimscapeAdapter = fake_adapter_cls

    with (
        patch.dict(sys.modules, {"src.engines.simscape": fake_simscape_mod}),
        pytest.raises(GolfModelingError, match="failed to load default model"),
    ):
        from src.engines.loaders import load_matlab_3d_engine

        load_matlab_3d_engine(tmp_path)


def test_ensure_engine_loaded_rejects_none() -> None:
    from src.engines.loaders import _ensure_engine_loaded

    with pytest.raises(GolfModelingError, match="DbC postcondition"):
        _ensure_engine_loaded(None, "Test")  # type: ignore[arg-type]
