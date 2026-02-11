"""Unit tests for engine loaders.

Tests both the canonical location (src.engines.loaders) and the
backward-compatible shim (src.shared.python.engine_loaders).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.data_io.common_utils import GolfModelingError
from src.shared.python.engine_core.engine_loaders import (
    LOADER_MAP,
    load_drake_engine,
    load_mujoco_engine,
    load_pinocchio_engine,
)
from src.shared.python.engine_core.engine_registry import EngineType


@pytest.fixture
def mock_suite_root(tmp_path: Path) -> Path:
    """Create a mock suite root structure."""
    return tmp_path


def test_loader_map() -> None:
    """Verify that LOADER_MAP contains all engine types."""
    assert EngineType.MUJOCO in LOADER_MAP
    assert EngineType.DRAKE in LOADER_MAP
    assert EngineType.PINOCCHIO in LOADER_MAP
    assert EngineType.OPENSIM in LOADER_MAP
    assert EngineType.MYOSIM in LOADER_MAP


def test_loader_map_from_canonical_location() -> None:
    """Verify LOADER_MAP is importable from the canonical location."""
    from src.engines.loaders import LOADER_MAP as canonical_map

    assert EngineType.MUJOCO in canonical_map
    assert canonical_map is LOADER_MAP  # Same object, not a copy


@patch.dict(
    sys.modules,
    {
        "mujoco": MagicMock(),
        "src.engines.physics_engines.mujoco": MagicMock(),
        "src.engines.physics_engines.mujoco.python": MagicMock(),
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf": MagicMock(),
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine": MagicMock(),
    },
)
def test_load_mujoco_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of MuJoCo engine."""
    with (
        patch(
            "src.shared.python.engine_core.engine_probes.MuJoCoProbe"
        ) as mock_probe_cls,
        patch(
            "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.MuJoCoPhysicsEngine"
        ) as mock_engine_cls,
    ):
        # Setup Probe
        mock_probe = mock_probe_cls.return_value
        mock_result = MagicMock()
        mock_result.is_available.return_value = True
        mock_probe.probe.return_value = mock_result

        # Setup Engine
        mock_engine = mock_engine_cls.return_value

        # Run
        engine = load_mujoco_engine(mock_suite_root)

        # Verify
        assert engine == mock_engine
        mock_probe_cls.assert_called_once_with(mock_suite_root)
        mock_engine_cls.assert_called_once()


@patch.dict(sys.modules, {"mujoco": MagicMock()})
def test_load_mujoco_engine_not_available(mock_suite_root: Path) -> None:
    """Test MuJoCo engine loading when probe fails."""
    with patch(
        "src.shared.python.engine_core.engine_probes.MuJoCoProbe"
    ) as mock_probe_cls:
        # Setup Probe to fail
        mock_probe = mock_probe_cls.return_value
        mock_result = MagicMock()
        mock_result.is_available.return_value = False
        mock_result.diagnostic_message = "Not installed"
        mock_result.get_fix_instructions.return_value = "Install it"
        mock_probe.probe.return_value = mock_result

        # Run - matches actual error message from engine_loaders
        # Error may be "MuJoCo not ready" (if engine module imports succeed and probe fails)
        # or "MuJoCo requirements not met" (if engine module import fails)
        with pytest.raises(
            GolfModelingError, match="MuJoCo (not ready|requirements not met)"
        ):
            load_mujoco_engine(mock_suite_root)


@patch.dict(
    sys.modules,
    {
        "pydrake": MagicMock(),
        "src.engines.physics_engines.drake": MagicMock(),
        "src.engines.physics_engines.drake.python": MagicMock(),
        "src.engines.physics_engines.drake.python.drake_physics_engine": MagicMock(),
    },
)
def test_load_drake_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of Drake engine."""
    with (
        patch(
            "src.shared.python.engine_core.engine_probes.DrakeProbe"
        ) as mock_probe_cls,
        patch(
            "src.engines.physics_engines.drake.python.drake_physics_engine.DrakePhysicsEngine"
        ) as mock_engine_cls,
    ):
        # Setup Probe
        mock_probe = mock_probe_cls.return_value
        mock_result = MagicMock()
        mock_result.is_available.return_value = True
        mock_probe.probe.return_value = mock_result

        # Run
        engine = load_drake_engine(mock_suite_root)

        assert engine == mock_engine_cls.return_value


@patch.dict(
    sys.modules,
    {
        "pinocchio": MagicMock(),
        "src.engines.physics_engines.pinocchio": MagicMock(),
        "src.engines.physics_engines.pinocchio.python": MagicMock(),
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine": MagicMock(),
    },
)
def test_load_pinocchio_engine_success(mock_suite_root: Path) -> None:
    """Test successful loading of Pinocchio engine."""
    with (
        patch(
            "src.shared.python.engine_core.engine_probes.PinocchioProbe"
        ) as mock_probe_cls,
        patch(
            "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine.PinocchioPhysicsEngine"
        ) as mock_engine_cls,
    ):
        # Setup Probe
        mock_probe = mock_probe_cls.return_value
        mock_result = MagicMock()
        mock_result.is_available.return_value = True
        mock_probe.probe.return_value = mock_result

        # Run
        engine = load_pinocchio_engine(mock_suite_root)

        assert engine == mock_engine_cls.return_value
