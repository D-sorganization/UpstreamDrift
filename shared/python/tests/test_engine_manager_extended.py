import shutil
import sys
from unittest.mock import MagicMock, patch

import pytest

from shared.python.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
    GolfModelingError,
)

# --- Fixtures ---


@pytest.fixture
def mock_suite_root(tmp_path):
    """Create a mock suite root directory structure."""
    root = tmp_path / "golf_suite"
    root.mkdir()

    engines = root / "engines"
    engines.mkdir()

    physics = engines / "physics_engines"
    physics.mkdir()

    (physics / "mujoco").mkdir()
    (physics / "drake").mkdir()
    (physics / "pinocchio").mkdir()
    (physics / "opensim").mkdir()
    (physics / "myosim").mkdir()

    simscape = engines / "Simscape_Multibody_Models"
    simscape.mkdir()
    (simscape / "2D_Golf_Model").mkdir()
    (simscape / "3D_Golf_Model").mkdir()

    (engines / "pendulum_models").mkdir()

    return root


@pytest.fixture
def engine_manager(mock_suite_root):
    """Initialize EngineManager with mock root."""
    # Patch probes globally for the lifetime of the fixture
    # This ensures new instances created inside methods are also mocked
    with (
        patch("shared.python.engine_probes.MuJoCoProbe") as MockMuJoCo,
        patch("shared.python.engine_probes.DrakeProbe") as MockDrake,
        patch("shared.python.engine_probes.PinocchioProbe") as MockPinocchio,
        patch("shared.python.engine_probes.OpenSimProbe") as MockOpenSim,
        patch("shared.python.engine_probes.MyoSimProbe") as MockMyoSim,
        patch("shared.python.engine_probes.MatlabProbe") as MockMatlab,
        patch("shared.python.engine_probes.PendulumProbe") as MockPendulum,
    ):

        manager = EngineManager(mock_suite_root)

        # Store mocks in manager for tests to access if needed (though not standard)
        manager._mocks = {  # type: ignore[attr-defined]
            EngineType.MUJOCO: MockMuJoCo,
            EngineType.DRAKE: MockDrake,
            EngineType.PINOCCHIO: MockPinocchio,
            EngineType.OPENSIM: MockOpenSim,
            EngineType.MYOSIM: MockMyoSim,
            EngineType.MATLAB_2D: MockMatlab,
            EngineType.PENDULUM: MockPendulum,
        }

        yield manager


# --- Tests ---


def test_initialization(engine_manager, mock_suite_root):
    assert engine_manager.suite_root == mock_suite_root
    assert engine_manager.current_engine is None
    assert len(engine_manager.engine_status) > 0


def test_discover_engines(engine_manager):
    # All directories were created in fixture, so all should be available
    for status in engine_manager.engine_status.values():
        assert status == EngineStatus.AVAILABLE


def test_discover_engines_missing(mock_suite_root):
    # Remove a directory
    shutil.rmtree(mock_suite_root / "engines" / "physics_engines" / "mujoco")

    with (
        patch("shared.python.engine_probes.MuJoCoProbe"),
        patch("shared.python.engine_probes.DrakeProbe"),
        patch("shared.python.engine_probes.PinocchioProbe"),
        patch("shared.python.engine_probes.OpenSimProbe"),
        patch("shared.python.engine_probes.MyoSimProbe"),
        patch("shared.python.engine_probes.MatlabProbe"),
        patch("shared.python.engine_probes.PendulumProbe"),
    ):
        manager = EngineManager(mock_suite_root)

    assert manager.get_engine_status(EngineType.MUJOCO) == EngineStatus.UNAVAILABLE


def test_switch_engine_unknown(engine_manager):
    assert engine_manager.switch_engine("unknown_engine") is False


def test_switch_engine_unavailable(engine_manager):
    engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.UNAVAILABLE
    assert engine_manager.switch_engine(EngineType.MUJOCO) is False


def test_switch_engine_success(engine_manager):
    with patch.object(engine_manager, "_load_engine") as mock_load:
        result = engine_manager.switch_engine(EngineType.MUJOCO)
        assert result is True
        mock_load.assert_called_once_with(EngineType.MUJOCO)
        assert engine_manager.current_engine == EngineType.MUJOCO


def test_switch_engine_failure(engine_manager):
    with patch.object(
        engine_manager, "_load_engine", side_effect=GolfModelingError("Load failed")
    ):
        result = engine_manager.switch_engine(EngineType.MUJOCO)
        assert result is False
        assert engine_manager.engine_status[EngineType.MUJOCO] == EngineStatus.ERROR


def test_load_engine_no_loader(engine_manager):
    # Mock the registry to return None (no registration)
    from shared.python.engine_registry import get_registry
    registry = get_registry()
    
    with patch.object(registry, 'get', return_value=None):
        with pytest.raises(GolfModelingError):
            engine_manager._load_engine(EngineType.MUJOCO)


def test_load_mujoco_engine_success(engine_manager):
    # Test using the current registry-based approach
    engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE
    
    mock_mujoco_module = MagicMock()
    
    with patch.dict(sys.modules, {"mujoco": mock_mujoco_module}):
        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.MuJoCoPhysicsEngine"
        ) as mock_engine_class:
            mock_engine_instance = MagicMock()
            mock_engine_class.return_value = mock_engine_instance
            
            result = engine_manager.switch_engine(EngineType.MUJOCO)
            
            assert result is True
            assert engine_manager.active_physics_engine is not None
            assert engine_manager.engine_status[EngineType.MUJOCO] == EngineStatus.LOADED


def test_load_mujoco_engine_probe_fail(engine_manager):
    # Test probe failure through switch_engine
    engine_manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE
    
    with patch("shared.python.engine_probes.MuJoCoProbe") as mock_probe_class:
        mock_probe = MagicMock()
        mock_probe.probe.return_value.is_available.return_value = False
        mock_probe.probe.return_value.diagnostic_message = "Not ready"
        mock_probe_class.return_value = mock_probe
        
        # This should fail during the loading process
        with patch("shared.python.engine_loaders.load_mujoco_engine") as mock_loader:
            mock_loader.side_effect = GolfModelingError("MuJoCo not ready")
            
            result = engine_manager.switch_engine(EngineType.MUJOCO)
            assert result is False
            assert engine_manager.engine_status[EngineType.MUJOCO] == EngineStatus.ERROR


def test_load_drake_engine_success(engine_manager):
    # Test using the current registry-based approach
    engine_manager.engine_status[EngineType.DRAKE] = EngineStatus.AVAILABLE
    
    mock_pydrake_module = MagicMock()
    
    with patch.dict(sys.modules, {"pydrake": mock_pydrake_module}):
        with patch(
            "engines.physics_engines.drake.python.drake_physics_engine.DrakePhysicsEngine"
        ) as mock_engine_class:
            mock_engine_instance = MagicMock()
            mock_engine_class.return_value = mock_engine_instance
            
            result = engine_manager.switch_engine(EngineType.DRAKE)
            
            assert result is True
            assert engine_manager.active_physics_engine is not None
            assert engine_manager.engine_status[EngineType.DRAKE] == EngineStatus.LOADED


def test_load_pinocchio_engine_success(engine_manager):
    # Test using the current registry-based approach
    engine_manager.engine_status[EngineType.PINOCCHIO] = EngineStatus.AVAILABLE
    
    mock_pinocchio_module = MagicMock()
    
    with patch.dict(sys.modules, {"pinocchio": mock_pinocchio_module}):
        with patch(
            "engines.physics_engines.pinocchio.python.pinocchio_physics_engine.PinocchioPhysicsEngine"
        ) as mock_engine_class:
            mock_engine_instance = MagicMock()
            mock_engine_class.return_value = mock_engine_instance
            
            result = engine_manager.switch_engine(EngineType.PINOCCHIO)
            
            assert result is True
            assert engine_manager.active_physics_engine is not None
            assert engine_manager.engine_status[EngineType.PINOCCHIO] == EngineStatus.LOADED


def test_load_matlab_engine_success(engine_manager):
    mock_matlab = MagicMock()
    mock_matlab_engine = MagicMock()
    mock_matlab.engine = mock_matlab_engine

    with patch.dict(
        sys.modules, {"matlab": mock_matlab, "matlab.engine": mock_matlab_engine}
    ):
        with patch("pathlib.Path.exists", return_value=True):
            engine_manager._load_matlab_engine(EngineType.MATLAB_2D)
            assert engine_manager._matlab_engine is not None


def test_load_pendulum_engine(engine_manager):
    engine_manager._load_pendulum_engine()


def test_cleanup(engine_manager):
    engine_manager._matlab_engine = MagicMock()
    engine_manager.cleanup()
    assert engine_manager._matlab_engine is None
    assert engine_manager.current_engine is None


def test_get_engine_info(engine_manager):
    info = engine_manager.get_engine_info()
    assert "available_engines" in info
    assert "engine_status" in info


def test_validate_engine_configuration(engine_manager):
    assert engine_manager.validate_engine_configuration(EngineType.MUJOCO) is False
    (engine_manager.engine_paths[EngineType.MUJOCO] / "python").mkdir()
    assert engine_manager.validate_engine_configuration(EngineType.MUJOCO) is True


def test_probe_all_engines(engine_manager):
    engine_manager.probe_all_engines()
    assert len(engine_manager.probe_results) == len(EngineType)


def test_get_diagnostic_report(engine_manager):
    # Mock probes to have deterministic output
    for mock_cls in engine_manager._mocks.values():  # type: ignore[attr-defined]
        mock_instance = mock_cls.return_value
        mock_instance.probe.return_value.is_available.return_value = True
        mock_instance.probe.return_value.status = EngineStatus.AVAILABLE
        mock_instance.probe.return_value.version = "1.0.0"
        mock_instance.probe.return_value.missing_dependencies = []
        mock_instance.probe.return_value.diagnostic_message = "Ready"
        mock_instance.probe.return_value.engine_name = "MockEngine"

    # Set specific name
    engine_manager._mocks[  # type: ignore[attr-defined]
        EngineType.MUJOCO
    ].return_value.probe.return_value.engine_name = "mujoco"

    report = engine_manager.get_diagnostic_report()
    assert "Engine Readiness Report" in report
    assert "MUJOCO" in report
