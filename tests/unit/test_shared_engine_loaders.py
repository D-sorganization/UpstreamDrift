"""Unit tests for engine loaders.

Tests both the canonical location (src.engines.loaders) and the
backward-compatible shim (src.shared.python.engine_loaders).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import src.shared.python.engine_core.engine_probes as engine_probes_mod
from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.engine_core.engine_loaders import (
    LOADER_MAP,
    load_drake_engine,
    load_mujoco_engine,
    load_pinocchio_engine,
)
from src.shared.python.engine_core.engine_probes import EngineProbe
from src.shared.python.engine_core.engine_registry import EngineType
from src.shared.python.engine_core.interfaces import PhysicsEngine

_PROBE_RESULT_SPEC = [
    "is_available",
    "diagnostic_message",
    "get_fix_instructions",
    "details",
]


@pytest.fixture
def mock_suite_root(tmp_path: Path) -> Path:
    """Create a mock suite root structure."""
    return tmp_path


@pytest.mark.parametrize(
    "engine_type",
    [
        EngineType.MUJOCO,
        EngineType.DRAKE,
        EngineType.PINOCCHIO,
        EngineType.OPENSIM,
        EngineType.MYOSIM,
    ],
    ids=["mujoco", "drake", "pinocchio", "opensim", "myosim"],
)
def test_loader_map(engine_type) -> None:
    """Verify that LOADER_MAP contains the engine type."""
    assert engine_type in LOADER_MAP


def test_loader_map_from_canonical_location() -> None:
    """Verify LOADER_MAP is importable from the canonical location."""
    from src.engines.loaders import LOADER_MAP as canonical_map

    assert EngineType.MUJOCO in canonical_map
    assert canonical_map is LOADER_MAP  # Same object, not a copy


def _make_probe_mock(*, available: bool = True) -> MagicMock:
    """Create a mock probe class whose instance.probe() returns a result mock."""
    mock_probe_cls = MagicMock()
    mock_probe = MagicMock(spec=EngineProbe)
    mock_probe_cls.return_value = mock_probe
    mock_result = MagicMock(spec=_PROBE_RESULT_SPEC)
    mock_result.is_available.return_value = available
    if not available:
        mock_result.diagnostic_message = "Not installed"
        mock_result.get_fix_instructions.return_value = "Install it"
    mock_probe.probe.return_value = mock_result
    return mock_probe_cls


@pytest.mark.serial
def test_load_mujoco_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of MuJoCo engine."""
    mock_engine = MagicMock(spec=PhysicsEngine)
    mock_engine_cls = MagicMock(return_value=mock_engine)

    mock_physics_mod = MagicMock(spec=["MuJoCoPhysicsEngine"])
    mock_physics_mod.MuJoCoPhysicsEngine = mock_engine_cls

    mock_probe_cls = _make_probe_mock(available=True)

    modules_patch = {
        "mujoco": MagicMock(),
        "src.engines.physics_engines.mujoco": MagicMock(),
        "src.engines.physics_engines.mujoco.python": MagicMock(),
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf": MagicMock(),
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine": mock_physics_mod,
    }

    with (
        patch.dict(sys.modules, modules_patch),
        patch.object(engine_probes_mod, "MuJoCoProbe", mock_probe_cls),
    ):
        engine = load_mujoco_engine(mock_suite_root)

        assert engine == mock_engine
        mock_probe_cls.assert_called_once_with(mock_suite_root)
        mock_engine_cls.assert_called_once()


@patch.dict(sys.modules, {"mujoco": MagicMock()})
def test_load_mujoco_engine_not_available(mock_suite_root: Path) -> None:
    """Test MuJoCo engine loading when probe fails."""
    mock_probe_cls = _make_probe_mock(available=False)

    with patch.object(engine_probes_mod, "MuJoCoProbe", mock_probe_cls):
        # Error may be "MuJoCo not ready" (if engine module imports succeed and probe fails)
        # or "MuJoCo requirements not met" (if engine module import fails)
        with pytest.raises(
            GolfModelingError, match="MuJoCo (not ready|requirements not met)"
        ):
            load_mujoco_engine(mock_suite_root)


@pytest.mark.serial
def test_load_drake_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of Drake engine."""
    mock_engine = MagicMock(spec=PhysicsEngine)
    mock_engine_cls = MagicMock(return_value=mock_engine)

    mock_drake_mod = MagicMock(spec=["DrakePhysicsEngine"])
    mock_drake_mod.DrakePhysicsEngine = mock_engine_cls

    mock_probe_cls = _make_probe_mock(available=True)

    modules_patch = {
        "pydrake": MagicMock(),
        "src.engines.physics_engines.drake": MagicMock(),
        "src.engines.physics_engines.drake.python": MagicMock(),
        "src.engines.physics_engines.drake.python.drake_physics_engine": mock_drake_mod,
    }

    with (
        patch.dict(sys.modules, modules_patch),
        patch.object(engine_probes_mod, "DrakeProbe", mock_probe_cls),
    ):
        engine = load_drake_engine(mock_suite_root)

        assert engine == mock_engine_cls.return_value
        mock_probe_cls.assert_called_once_with(mock_suite_root)


@pytest.mark.serial
def test_load_pinocchio_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of Pinocchio engine."""
    mock_engine = MagicMock(spec=PhysicsEngine)
    mock_engine_cls = MagicMock(return_value=mock_engine)

    mock_pin_mod = MagicMock(spec=["PinocchioPhysicsEngine"])
    mock_pin_mod.PinocchioPhysicsEngine = mock_engine_cls

    mock_probe_cls = _make_probe_mock(available=True)

    modules_patch = {
        "pinocchio": MagicMock(),
        "src.engines.physics_engines.pinocchio": MagicMock(),
        "src.engines.physics_engines.pinocchio.python": MagicMock(),
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine": mock_pin_mod,
    }

    with (
        patch.dict(sys.modules, modules_patch),
        patch.object(engine_probes_mod, "PinocchioProbe", mock_probe_cls),
    ):
        engine = load_pinocchio_engine(mock_suite_root)

        assert engine == mock_engine_cls.return_value
        mock_probe_cls.assert_called_once_with(mock_suite_root)
