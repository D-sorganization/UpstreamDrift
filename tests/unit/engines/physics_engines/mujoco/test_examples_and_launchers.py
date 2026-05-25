"""Tests for MuJoCo Python examples and launchers."""

import sys
from unittest.mock import MagicMock, patch

import numpy as np

from src.engines.physics_engines.mujoco.python.examples import (
    example_featherstone_algorithms,
    example_screw_theory,
)


def test_example_featherstone():
    """Test the featherstone algorithm example."""
    # Run the main function. We mock out the actual functions if they are heavy,
    # but since it's just numpy math in the example, we can mock out the `aba`, `crba`, etc.
    # to avoid needing the full implementations if they depend on external things.
    with (
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_featherstone_algorithms.crba"
        ) as mock_crba,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_featherstone_algorithms.rnea"
        ) as mock_rnea,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_featherstone_algorithms.aba"
        ) as mock_aba,
        patch("numpy.linalg.solve") as mock_solve,
    ):
        mock_crba.return_value = np.eye(2)
        mock_rnea.return_value = np.array([0.0, 0.0])
        mock_aba.return_value = np.array([0.0, 0.0])

        example_featherstone_algorithms.main()

        mock_crba.assert_called()
        mock_rnea.assert_called()
        mock_aba.assert_called()


def test_example_screw_theory():
    """Test the screw theory example."""
    with (
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_screw_theory.screw_axis"
        ) as mock_sa,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_screw_theory.exponential_map"
        ) as mock_exp,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_screw_theory.logarithmic_map"
        ) as mock_log,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_screw_theory.screw_to_transform"
        ) as mock_s2t,
        patch(
            "src.engines.physics_engines.mujoco.python.examples.example_screw_theory.adjoint_transform"
        ) as mock_adj,
    ):
        mock_sa.return_value = np.zeros(6)
        mock_exp.return_value = np.eye(4)
        mock_log.return_value = (np.zeros(6), 0.0)
        mock_s2t.return_value = np.eye(4)
        mock_adj.return_value = np.eye(6)

        example_screw_theory.main()

        mock_sa.assert_called()
        mock_exp.assert_called()
        mock_log.assert_called()
        mock_s2t.assert_called()
        mock_adj.assert_called()


def test_golf_suite_launcher_init():
    """Test UpstreamDriftLauncher initialization."""
    from PyQt6.QtWidgets import QApplication
    from src.engines.physics_engines.mujoco.python import golf_suite_launcher

    app = QApplication.instance() or QApplication(sys.argv)
    launcher = golf_suite_launcher.UpstreamDriftLauncher()

    assert launcher.mujoco_path.name == "advanced_gui.py"
    assert hasattr(launcher, "btn_mujoco")
    assert hasattr(launcher, "btn_drake")
    assert hasattr(launcher, "btn_pinocchio")


def test_golf_suite_launcher_methods():
    """Test UpstreamDriftLauncher launch methods."""
    from PyQt6.QtWidgets import QApplication
    from src.engines.physics_engines.mujoco.python import golf_suite_launcher

    app = QApplication.instance() or QApplication(sys.argv)
    launcher = golf_suite_launcher.UpstreamDriftLauncher()

    # Test generic launch script
    with patch("subprocess.Popen") as mock_popen:
        # With non-existent path
        mock_path = MagicMock()
        mock_path.exists.return_value = False
        with patch.object(
            golf_suite_launcher.QtWidgets.QMessageBox, "critical"
        ) as mock_msg:
            launcher._launch_script("Test", mock_path, mock_path)
            mock_popen.assert_not_called()
            mock_msg.assert_called()

        # With existent path
        mock_path.exists.return_value = True
        launcher._launch_script("Test", mock_path, mock_path)
        mock_popen.assert_called_once()

    # Test mujoco launch
    with patch.object(launcher, "_launch_script") as mock_launch:
        launcher._launch_mujoco()
        mock_launch.assert_called_once()

    # Test drake launch (docker)
    with patch("subprocess.Popen") as mock_popen:
        launcher._launch_drake()
        mock_popen.assert_called_once()

    # Test pinocchio launch
    with patch.object(launcher, "_launch_script") as mock_launch:
        launcher._launch_pinocchio()
        mock_launch.assert_called_once()
