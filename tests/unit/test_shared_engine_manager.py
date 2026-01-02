"""Unit tests for shared engine manager."""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from shared.python.engine_manager import (
    EngineManager,
    EngineStatus,
    EngineType,
    GolfModelingError,
)


class TestEngineManager(unittest.TestCase):
    """Test cases for EngineManager."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_root = Path("/mock/root")

        # Patch engine probes individually
        self.mujoco_patcher = patch("shared.python.engine_probes.MuJoCoProbe")
        self.mock_mujoco_probe_cls = self.mujoco_patcher.start()
        self.mock_mujoco_probe_cls.return_value.probe.return_value.is_available.return_value = (
            True
        )

        self.drake_patcher = patch("shared.python.engine_probes.DrakeProbe")
        self.mock_drake_probe_cls = self.drake_patcher.start()
        self.mock_drake_probe_cls.return_value.probe.return_value.is_available.return_value = (
            True
        )

        self.pinocchio_patcher = patch("shared.python.engine_probes.PinocchioProbe")
        self.mock_pinocchio_probe_cls = self.pinocchio_patcher.start()

        self.pendulum_patcher = patch("shared.python.engine_probes.PendulumProbe")
        self.mock_pendulum_probe_cls = self.pendulum_patcher.start()

        self.matlab_patcher = patch("shared.python.engine_probes.MatlabProbe")
        self.mock_matlab_probe_cls = self.matlab_patcher.start()

        # Patch setup_logging
        self.logging_patcher = patch("shared.python.engine_manager.setup_logging")
        self.logging_patcher.start()

    def tearDown(self):
        """Tear down test fixtures."""
        self.mujoco_patcher.stop()
        self.drake_patcher.stop()
        self.pinocchio_patcher.stop()
        self.pendulum_patcher.stop()
        self.matlab_patcher.stop()
        self.logging_patcher.stop()

    def test_initialization_discovery(self):
        """Test that engines are discovered correctly."""
        with patch.object(EngineManager, "_discover_engines") as mock_discover:
            manager = EngineManager(self.mock_root)
            mock_discover.assert_called_once()

        manager = EngineManager(self.mock_root)

        path_mock_mujoco = MagicMock(spec=Path)
        path_mock_mujoco.exists.return_value = True

        path_mock_drake = MagicMock(spec=Path)
        path_mock_drake.exists.return_value = False

        manager.engine_paths = {
            EngineType.MUJOCO: path_mock_mujoco,
            EngineType.DRAKE: path_mock_drake,
        }

        manager._discover_engines()

        self.assertEqual(
            manager.engine_status[EngineType.MUJOCO], EngineStatus.AVAILABLE
        )
        self.assertEqual(
            manager.engine_status[EngineType.DRAKE], EngineStatus.UNAVAILABLE
        )

    def test_switch_engine_success(self):
        """Test successful engine switch."""
        manager = EngineManager(self.mock_root)
        manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

        with patch.object(manager, "_load_engine") as mock_load:
            success = manager.switch_engine(EngineType.MUJOCO)

            self.assertTrue(success)
            self.assertEqual(manager.current_engine, EngineType.MUJOCO)
            mock_load.assert_called_with(EngineType.MUJOCO)

    def test_switch_engine_unavailable(self):
        """Test switching to unavailable engine."""
        manager = EngineManager(self.mock_root)
        manager.engine_status[EngineType.MUJOCO] = EngineStatus.UNAVAILABLE

        success = manager.switch_engine(EngineType.MUJOCO)
        self.assertFalse(success)

    def test_switch_engine_failure(self):
        """Test handling of engine loading failure."""
        manager = EngineManager(self.mock_root)
        manager.engine_status[EngineType.MUJOCO] = EngineStatus.AVAILABLE

        with patch.object(
            manager, "_load_engine", side_effect=GolfModelingError("Fail")
        ):
            success = manager.switch_engine(EngineType.MUJOCO)
            self.assertFalse(success)
            self.assertEqual(
                manager.engine_status[EngineType.MUJOCO], EngineStatus.ERROR
            )

    def test_load_mujoco_engine_details(self):
        """Test detailed steps of loading MuJoCo engine."""
        manager = EngineManager(self.mock_root)
        manager.engine_paths[EngineType.MUJOCO] = Path("/mock/mujoco")

        # Configure probe specifically for this test
        self.mock_mujoco_probe_cls.return_value.probe.return_value.is_available.return_value = (
            True
        )

        with patch(
            "engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine.mujoco"
        ) as mock_mujoco_pkg:
            mock_mujoco_pkg.__version__ = "3.2.3"
            mock_mujoco_pkg.MjModel.from_xml_path.return_value = MagicMock()

            with (
                patch("pathlib.Path.exists", return_value=True),
                patch("pathlib.Path.glob", return_value=[Path("model.xml")]),
            ):
                manager._load_mujoco_engine()
                self.assertEqual(manager._mujoco_module, mock_mujoco_pkg)

    def test_get_engine_info(self):
        """Test information retrieval."""
        manager = EngineManager(self.mock_root)
        manager.engine_status = {EngineType.MUJOCO: EngineStatus.AVAILABLE}

        info = manager.get_engine_info()
        self.assertIn("mujoco", info["available_engines"])

    def test_validate_engine_configuration(self):
        """Test configuration validation."""
        manager = EngineManager(self.mock_root)
        manager.engine_status = {EngineType.MUJOCO: EngineStatus.AVAILABLE}

        with patch("pathlib.Path.exists", return_value=True):
            self.assertTrue(manager.validate_engine_configuration(EngineType.MUJOCO))
