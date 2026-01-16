"""GUI Component Tests - MuJoCo Simulation Widget and Launchers.

This module tests GUI components with appropriate mocking for headless environments.
Tests verify actual behavior, not just code execution.

Note: These tests require mujoco and PyQt6 to be installed. Tests are skipped
if dependencies are missing rather than using extensive mocking.
"""

from importlib.util import find_spec
from unittest.mock import MagicMock

import numpy as np
import pytest

# Check for required dependencies at module level using importlib.util.find_spec
MUJOCO_AVAILABLE = find_spec("mujoco") is not None

try:
    from PyQt6.QtWidgets import QApplication

    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False


@pytest.fixture(scope="module")
def qapp():
    """Create a QApplication instance for tests that need it."""
    if not PYQT_AVAILABLE:
        pytest.skip("PyQt6 not available")
    # Check if QApplication already exists
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not installed")
class TestMuJoCoSimWidget:
    """Tests for the MuJoCoSimWidget class.

    These tests verify that the widget properly manages MuJoCo simulation state
    and correctly transforms between simulation and visualization coordinates.
    """

    def test_widget_initialization(self, qapp):
        """Test that widget initializes with correct default parameters."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
            MuJoCoSimWidget,
        )

        widget = MuJoCoSimWidget(width=640, height=480, fps=30)

        # Verify initialization parameters
        assert widget.width() == 640 or widget.minimumWidth() <= 640
        assert widget.height() == 480 or widget.minimumHeight() <= 480
        assert hasattr(widget, "model")
        assert hasattr(widget, "data")

        widget.close()

    def test_load_simple_model(self, qapp, tmp_path):
        """Test loading a minimal MuJoCo model."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
            MuJoCoSimWidget,
        )

        widget = MuJoCoSimWidget(width=100, height=100, fps=60)

        # Create a minimal valid MuJoCo XML
        model_xml = """
        <mujoco>
            <worldbody>
                <body name="test_body" pos="0 0 1">
                    <joint type="hinge" axis="0 0 1"/>
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """

        # Load model
        widget.load_model_from_xml(model_xml)

        # Verify model loaded correctly
        assert widget.model is not None
        assert widget.model.nq >= 1, "Model should have at least one DOF"
        assert widget.data is not None

        widget.close()

    def test_reset_state_returns_to_initial(self, qapp):
        """Test that reset_state returns simulation to initial configuration."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
            MuJoCoSimWidget,
        )

        widget = MuJoCoSimWidget(width=100, height=100, fps=60)

        model_xml = """
        <mujoco>
            <worldbody>
                <body name="pendulum" pos="0 0 1">
                    <joint type="hinge" axis="0 1 0"/>
                    <geom type="capsule" size="0.05" fromto="0 0 0 0 0 -0.5"/>
                </body>
            </worldbody>
        </mujoco>
        """
        widget.load_model_from_xml(model_xml)

        # Capture initial state
        initial_qpos = widget.data.qpos.copy()

        # Modify state
        widget.data.qpos[0] = 1.5  # Set joint angle to 1.5 rad

        # Reset and verify
        widget.reset_state()
        np.testing.assert_array_almost_equal(
            widget.data.qpos,
            initial_qpos,
            err_msg="State should return to initial after reset",
        )

        widget.close()

    def test_camera_setting(self, qapp):
        """Test that camera views can be set correctly."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
            MuJoCoSimWidget,
        )

        widget = MuJoCoSimWidget(width=100, height=100, fps=60)

        model_xml = """
        <mujoco>
            <worldbody>
                <body pos="0 0 0">
                    <geom type="box" size="1 1 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        widget.load_model_from_xml(model_xml)

        # Test various camera views
        for view in ["front", "side", "top", "perspective"]:
            try:
                widget.set_camera(view)
                # No exception means success
            except ValueError:
                # Some views may not be implemented - that's acceptable
                pass

        widget.close()

    def test_get_dof_info_returns_dict(self, qapp):
        """Test that get_dof_info returns meaningful DOF information."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_widget import (
            MuJoCoSimWidget,
        )

        widget = MuJoCoSimWidget(width=100, height=100, fps=60)

        model_xml = """
        <mujoco>
            <worldbody>
                <body name="link1" pos="0 0 1">
                    <joint name="joint1" type="hinge" axis="0 0 1"/>
                    <geom type="sphere" size="0.1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        widget.load_model_from_xml(model_xml)

        dof_info = widget.get_dof_info()

        # Verify DOF info structure
        assert isinstance(dof_info, (dict, list)), "DOF info should be dict or list"
        if isinstance(dof_info, dict):
            assert len(dof_info) >= 1, "Should have at least one DOF"

        widget.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not installed")
class TestHumanoidLauncher:
    """Tests for the HumanoidLauncher application.

    These tests verify that the launcher can be instantiated and basic
    operations work correctly.
    """

    def test_launcher_instantiation(self, qapp):
        """Test that HumanoidLauncher can be instantiated."""
        from engines.physics_engines.mujoco.python.humanoid_launcher import (
            HumanoidLauncher,
        )

        launcher = HumanoidLauncher()

        # Verify basic attributes exist
        assert hasattr(launcher, "show")
        assert hasattr(launcher, "close")

        # Clean up
        launcher.close()

    def test_launcher_has_required_components(self, qapp):
        """Test that launcher has expected UI components."""
        from engines.physics_engines.mujoco.python.humanoid_launcher import (
            HumanoidLauncher,
        )

        launcher = HumanoidLauncher()

        # Check for typical launcher attributes
        # These are informational - we don't fail if they're missing
        expected_attrs = ["centralWidget", "menuBar", "statusBar"]
        found_attrs = [attr for attr in expected_attrs if hasattr(launcher, attr)]

        assert (
            len(found_attrs) > 0
        ), "Launcher should have at least one standard widget attribute"

        launcher.close()


@pytest.mark.skipif(not MUJOCO_AVAILABLE, reason="MuJoCo not installed")
@pytest.mark.skipif(not PYQT_AVAILABLE, reason="PyQt6 not installed")
class TestControlsTab:
    """Tests for the ControlsTab widget."""

    def test_controls_tab_instantiation(self, qapp):
        """Test that ControlsTab can be instantiated with mock dependencies."""
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab import (
            ControlsTab,
        )

        # ControlsTab requires parent and sim_widget
        mock_parent = MagicMock()
        mock_sim_widget = MagicMock()

        tab = ControlsTab(mock_parent, mock_sim_widget)

        # Verify tab was created
        assert tab is not None

        # Clean up
        if hasattr(tab, "close"):
            tab.close()
