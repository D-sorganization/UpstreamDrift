"""Tests for drake_gui_app.py."""

from unittest.mock import MagicMock, patch

import pytest

from src.engines.physics_engines.drake.python.src.drake_gui_app import (
    DrakeSimApp,
    HAS_QT,
)

if not HAS_QT:
    pytest.skip("Skipping PyQt tests since PyQt6 is not available", allow_module_level=True)


class TestDrakeSimApp:
    """Tests for DrakeSimApp."""

    @patch("src.engines.physics_engines.drake.python.src.drake_gui_app.SimulationGUIBase.__init__")
    @patch.object(DrakeSimApp, "_scan_urdf_models")
    @patch.object(DrakeSimApp, "_init_simulation")
    @patch.object(DrakeSimApp, "_setup_ui")
    @patch.object(DrakeSimApp, "_sync_kinematic_sliders")
    @patch("src.engines.physics_engines.drake.python.src.drake_gui_app.QtCore.QTimer")
    def test_init(
        self,
        mock_timer_cls: MagicMock,
        mock_sync: MagicMock,
        mock_setup_ui: MagicMock,
        mock_init_sim: MagicMock,
        mock_scan: MagicMock,
        mock_base_init: MagicMock,
    ) -> None:
        """Test application initialization."""
        mock_timer = MagicMock()
        mock_timer_cls.return_value = mock_timer
        
        app = DrakeSimApp()
        
        mock_base_init.assert_called_once()
        mock_scan.assert_called_once()
        mock_init_sim.assert_called_once()
        mock_setup_ui.assert_called_once()
        mock_sync.assert_called_once()
        
        mock_timer_cls.assert_called_once()
        mock_timer.timeout.connect.assert_called_once()
        mock_timer.start.assert_called_once_with(1)  # TIME_STEP_S * MS_PER_SECOND = 0.001 * 1000 = 1

    @patch.object(DrakeSimApp, "__init__", return_value=None)
    def test_get_joint_names(self, mock_init: MagicMock) -> None:
        """Test getting joint names."""
        app = DrakeSimApp()
        
        # Test with no plant
        app.plant = None
        assert app.get_joint_names() == []
        
        # Test with plant
        mock_plant = MagicMock()
        mock_plant.num_joints.return_value = 2
        
        mock_joint1 = MagicMock()
        mock_joint1.name.return_value = "j1"
        mock_joint1.num_velocities.return_value = 1
        
        mock_joint2 = MagicMock()
        mock_joint2.name.return_value = "j2"
        mock_joint2.num_velocities.return_value = 0
        
        def get_joint(idx):
            if idx == 0:
                return mock_joint1
            return mock_joint2
            
        mock_plant.get_joint.side_effect = get_joint
        app.plant = mock_plant
        
        with patch("src.engines.physics_engines.drake.python.src.drake_gui_app.JointIndex", side_effect=lambda x: x):
            names = app.get_joint_names()
        
        assert names == ["j1"]

    @patch.object(DrakeSimApp, "__init__", return_value=None)
    def test_scan_urdf_models(self, mock_init: MagicMock, tmp_path) -> None:
        """Test scanning for URDF models."""
        app = DrakeSimApp()
        app.available_models = []
        
        # Create dummy urdf files
        urdf_dir = tmp_path / "urdf"
        urdf_dir.mkdir()
        (urdf_dir / "test_model_1.urdf").touch()
        (urdf_dir / "another_model.urdf").touch()
        
        # Patch Path so that we mock the directory paths
        with patch("src.engines.physics_engines.drake.python.src.drake_gui_app.Path") as mock_path:
            # We want the docker_shared path to be our tmp_path
            mock_path.return_value.exists.return_value = True
            mock_path.return_value = urdf_dir
            
            # Since Path() is called multiple times, we need to handle it properly
            # In the code, docker_shared = Path("/shared/urdf")
            # We just mock the whole Path class to return our tmp_path when called with '/shared/urdf'
            def path_side_effect(arg):
                if arg == "/shared/urdf":
                    return urdf_dir
                if str(arg) == "__file__":
                    # Just return a dummy path for __file__
                    dummy = MagicMock()
                    dummy.parents = [MagicMock()] * 6
                    return dummy
                return tmp_path / str(arg)
                
            mock_path.side_effect = path_side_effect
            
            app._scan_urdf_models()
            
        # The exact parsing might be slightly tricky to mock due to pathlib
        # If it found models, available_models will have > 0 items
        assert len(app.available_models) > 0

    @patch.object(DrakeSimApp, "__init__", return_value=None)
    def test_update_status(self, mock_init: MagicMock) -> None:
        """Test updating status bar."""
        app = DrakeSimApp()
        app.statusBar = MagicMock()
        mock_status_bar = MagicMock()
        app.statusBar.return_value = mock_status_bar
        
        app._update_status("Testing 123")
        
        mock_status_bar.showMessage.assert_called_once_with("Testing 123")

    @patch.object(DrakeSimApp, "__init__", return_value=None)
    def test_update_status_none_msg(self, mock_init: MagicMock) -> None:
        """Test updating status bar with None msg."""
        app = DrakeSimApp()
        with pytest.raises(ValueError, match="message must be provided"):
            app._update_status(None)  # type: ignore
