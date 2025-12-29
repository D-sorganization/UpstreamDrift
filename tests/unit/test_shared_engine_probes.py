import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shared.python.engine_probes import (
    DrakeProbe,
    EngineProbe,
    EngineProbeResult,
    MatlabProbe,
    MuJoCoProbe,
    PendulumProbe,
    PinocchioProbe,
    ProbeStatus,
)


# Test base classes and data structures
def test_engine_probe_result():
    res = EngineProbeResult(
        engine_name="TestEngine",
        status=ProbeStatus.AVAILABLE,
        version="1.0",
        missing_dependencies=[],
        diagnostic_message="All good",
    )
    assert res.is_available()
    assert res.get_fix_instructions() == "Engine is available"

    res_missing = EngineProbeResult(
        engine_name="TestEngine",
        status=ProbeStatus.MISSING_BINARY,
        version=None,
        missing_dependencies=["bin"],
        diagnostic_message="Missing binary",
    )
    assert not res_missing.is_available()
    assert "Install TestEngine binaries" in res_missing.get_fix_instructions()


def test_engine_probe_base():
    probe = EngineProbe("Base", Path("."))
    with pytest.raises(NotImplementedError):
        probe.probe()


# Test MuJoCoProbe
def test_mujoco_probe_success(tmp_path):
    # Mock suite root structure
    engine_dir = tmp_path / "engines/physics_engines/mujoco"
    (engine_dir / "python/mujoco_humanoid_golf").mkdir(parents=True)
    (engine_dir / "python/humanoid_launcher.py").touch()
    (engine_dir / "assets").mkdir()
    (engine_dir / "assets/model.xml").touch()

    with patch.dict(sys.modules, {"mujoco": MagicMock(__version__="3.0")}):
        probe = MuJoCoProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.AVAILABLE
        assert result.version == "3.0"


def test_mujoco_probe_missing_package(tmp_path):
    with patch.dict(sys.modules):
        if "mujoco" in sys.modules:
            del sys.modules["mujoco"]
        # Also need to ensure it can't be imported from environment if present
        with patch(
            "builtins.__import__", side_effect=ImportError("No module named mujoco")
        ):
            probe = MuJoCoProbe(tmp_path)
            result = probe.probe()
            assert result.status == ProbeStatus.NOT_INSTALLED


def test_mujoco_probe_dll_error(tmp_path):
    with patch.dict(sys.modules):
        # Mock import raising OSError
        with patch("builtins.__import__", side_effect=OSError("DLL load failed")):
            probe = MuJoCoProbe(tmp_path)
            result = probe.probe()
            assert result.status == ProbeStatus.MISSING_BINARY


def test_mujoco_probe_missing_assets(tmp_path):
    # Setup directory but no assets
    engine_dir = tmp_path / "engines/physics_engines/mujoco"
    (engine_dir / "python/mujoco_humanoid_golf").mkdir(parents=True)
    (engine_dir / "python/humanoid_launcher.py").touch()

    with patch.dict(sys.modules, {"mujoco": MagicMock(__version__="3.0")}):
        probe = MuJoCoProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.MISSING_ASSETS
        assert "assets" in result.diagnostic_message


# Test DrakeProbe
def test_drake_probe_success(tmp_path):
    engine_dir = tmp_path / "engines/physics_engines/drake"
    (engine_dir / "python/src").mkdir(parents=True)
    (engine_dir / "python/src/golf_gui.py").touch()

    mock_drake = MagicMock(__version__="1.0")

    # Need to mock pydrake and pydrake.multibody
    with patch.dict(
        sys.modules,
        {
            "pydrake": mock_drake,
            "pydrake.multibody": MagicMock(),
            "socket": MagicMock(),
        },
    ):
        probe = DrakeProbe(tmp_path)
        # Mock socket to succeed on port 7000
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.bind.return_value = None
            result = probe.probe()
            assert result.status == ProbeStatus.AVAILABLE


def test_drake_probe_port_blocked(tmp_path):
    engine_dir = tmp_path / "engines/physics_engines/drake"
    (engine_dir / "python/src").mkdir(parents=True)
    (engine_dir / "python/src/golf_gui.py").touch()

    with patch.dict(
        sys.modules,
        {"pydrake": MagicMock(__version__="1.0"), "pydrake.multibody": MagicMock()},
    ):
        probe = DrakeProbe(tmp_path)
        # Mock socket to always fail binding
        with patch("socket.socket") as mock_socket:
            mock_socket.return_value.bind.side_effect = OSError("Address in use")
            result = probe.probe()
            assert result.status == ProbeStatus.CONFIGURATION_ERROR


def test_drake_probe_missing_module(tmp_path):
    with patch.dict(sys.modules, {"pydrake": MagicMock()}):
        # Mock import pydrake.multibody failing
        with patch(
            "builtins.__import__",
            side_effect=ImportError("No module named pydrake.multibody"),
        ):
            pass

    # Let's try patching the module lookup
    with patch.dict(sys.modules):
        if "pydrake.multibody" in sys.modules:
            del sys.modules["pydrake.multibody"]

        probe = DrakeProbe(tmp_path)
        with patch("builtins.__import__", side_effect=ImportError):
            res = probe.probe()
            assert res.status == ProbeStatus.NOT_INSTALLED


# Test PinocchioProbe
def test_pinocchio_probe_success(tmp_path):
    engine_dir = tmp_path / "engines/physics_engines/pinocchio"
    (engine_dir / "python/pinocchio_golf").mkdir(parents=True)

    with patch.dict(sys.modules, {"pinocchio": MagicMock(__version__="2.0")}):
        probe = PinocchioProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.AVAILABLE


def test_pinocchio_probe_missing_dir(tmp_path):
    with patch.dict(sys.modules, {"pinocchio": MagicMock(__version__="2.0")}):
        probe = PinocchioProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.MISSING_ASSETS
        assert "engine directory" in result.missing_dependencies


# Test PendulumProbe
def test_pendulum_probe_success(tmp_path):
    engine_dir = tmp_path / "engines/pendulum_models"
    (engine_dir / "python/src").mkdir(parents=True)
    (engine_dir / "python/src/constants.py").touch()
    (engine_dir / "python/src/pendulum_solver.py").touch()

    probe = PendulumProbe(tmp_path)
    result = probe.probe()
    assert result.status == ProbeStatus.AVAILABLE


def test_pendulum_probe_missing(tmp_path):
    probe = PendulumProbe(tmp_path)
    result = probe.probe()
    assert result.status == ProbeStatus.MISSING_ASSETS


# Test MatlabProbe
def test_matlab_probe_success(tmp_path):
    engine_dir = tmp_path / "engines/Simscape_Multibody_Models/2D_Golf_Model"
    engine_dir.mkdir(parents=True)
    (engine_dir / "model.slx").touch()

    mock_matlab = MagicMock()
    mock_matlab_engine = MagicMock()

    with patch.dict(
        sys.modules, {"matlab": mock_matlab, "matlab.engine": mock_matlab_engine}
    ):
        probe = MatlabProbe(tmp_path, is_3d=False)
        result = probe.probe()
        assert result.status == ProbeStatus.AVAILABLE


def test_matlab_probe_missing_files(tmp_path):
    engine_dir = tmp_path / "engines/Simscape_Multibody_Models/3D_Golf_Model"
    engine_dir.mkdir(parents=True)

    mock_matlab = MagicMock()
    mock_matlab_engine = MagicMock()

    with patch.dict(
        sys.modules, {"matlab": mock_matlab, "matlab.engine": mock_matlab_engine}
    ):
        probe = MatlabProbe(tmp_path, is_3d=True)
        result = probe.probe()
        assert result.status == ProbeStatus.MISSING_ASSETS
        assert "Simulink/MATLAB files" in result.missing_dependencies


def test_matlab_probe_not_installed(tmp_path):
    with patch.dict(sys.modules):
        if "matlab.engine" in sys.modules:
            del sys.modules["matlab.engine"]
        with patch("builtins.__import__", side_effect=ImportError):
            probe = MatlabProbe(tmp_path)
            result = probe.probe()
            assert result.status == ProbeStatus.NOT_INSTALLED


# Test OpenSimProbe
def test_opensim_probe_success(tmp_path):
    engine_dir = tmp_path / "engines/physics_engines/opensim"
    (engine_dir / "python/opensim_golf").mkdir(parents=True)

    from shared.python.engine_probes import OpenSimProbe

    with patch.dict(sys.modules, {"opensim": MagicMock(__version__="4.0")}):
        probe = OpenSimProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.AVAILABLE
        assert result.version == "4.0"


def test_opensim_probe_missing_dir(tmp_path):
    from shared.python.engine_probes import OpenSimProbe

    with patch.dict(sys.modules, {"opensim": MagicMock(__version__="4.0")}):
        probe = OpenSimProbe(tmp_path)
        result = probe.probe()
        assert result.status == ProbeStatus.MISSING_ASSETS
        assert "engine directory" in result.missing_dependencies


def test_opensim_probe_not_installed(tmp_path):
    from shared.python.engine_probes import OpenSimProbe

    with patch.dict(sys.modules):
        if "opensim" in sys.modules:
            del sys.modules["opensim"]
        with patch("builtins.__import__", side_effect=ImportError):
            probe = OpenSimProbe(tmp_path)
            result = probe.probe()
            assert result.status == ProbeStatus.NOT_INSTALLED
