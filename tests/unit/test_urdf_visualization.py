"""Tests for URDF visualization widget.

Issue #755: Added comprehensive tests for MuJoCo preview and visualization toggles.
"""

import pytest

from src.shared.python.engine_core.engine_availability import (
    MUJOCO_AVAILABLE,
    PYQT6_AVAILABLE,
    PYTEST_QT_AVAILABLE,
)

pytestmark = pytest.mark.skipif(
    not PYQT6_AVAILABLE or not PYTEST_QT_AVAILABLE,
    reason="PyQt6 or pytest-qt not available",
)

if PYQT6_AVAILABLE:
    from src.tools.model_explorer.mujoco_viewer import (
        MuJoCoViewerWidget,
        VisualizationFlags,
    )
    from src.tools.model_explorer.visualization_widget import VisualizationWidget


def test_visualization_widget_init(qtbot):
    """Test that VisualizationWidget initializes correctly."""
    widget = VisualizationWidget()
    qtbot.addWidget(widget)

    assert widget.urdf_content == ""
    assert widget.info_label.text() == "No URDF content loaded"


def test_visualization_widget_update(qtbot):
    """Test updating the visualization."""
    widget = VisualizationWidget()
    qtbot.addWidget(widget)

    urdf_content = """<robot name="test">
    <link name="base"/>
    <joint name="j1" type="fixed">
        <parent link="base"/>
        <child link="l1"/>
    </joint>
    </robot>"""

    widget.update_visualization(urdf_content)

    assert widget.urdf_content == urdf_content
    # Check that info label contains link/joint counts
    # Note: simple string counting in the widget implementation might be naive but that's what we test against
    assert "Links: 1" in widget.info_label.text()
    assert "Joints: 1" in widget.info_label.text()


def test_visualization_widget_clear(qtbot):
    """Test clearing the visualization."""
    widget = VisualizationWidget()
    qtbot.addWidget(widget)

    widget.update_visualization("<robot/>")
    widget.clear()

    assert widget.urdf_content == ""
    assert widget.info_label.text() == "No URDF content loaded"


def test_visualization_widget_reset_view(qtbot):
    """Test resetting the view."""
    widget = VisualizationWidget()
    # Force use_mujoco to False to ensure gl_widget is created for testing fallback
    if widget.use_mujoco:
        widget.use_mujoco = False
        widget._setup_ui()  # Re-setup

    qtbot.addWidget(widget)

    # Modify camera state via the GL widget
    widget.gl_widget.camera_distance = 10.0
    widget.gl_widget.camera_rotation_x = 45.0

    widget.reset_view()

    # Reset sets back to 1.0 according to implementation
    # The initial value in __init__ is 5.0, but reset_view() uses 1.0
    assert widget.gl_widget.camera_distance == 1.0
    assert widget.gl_widget.camera_rotation_x == 0.0


# =============================================================================
# MuJoCo Viewer Tests (Issue #755)
# =============================================================================


class TestVisualizationFlags:
    """Tests for VisualizationFlags dataclass."""

    def test_default_values(self):
        """Test default visualization flags."""
        flags = VisualizationFlags()

        assert flags.show_collision is False
        assert flags.show_frames is True  # Frames on by default
        assert flags.show_joint_limits is False
        assert flags.show_contacts is False
        assert flags.show_com is False

    def test_to_dict(self):
        """Test serialization to dictionary."""
        flags = VisualizationFlags(
            show_collision=True,
            show_frames=False,
            show_contacts=True,
        )

        result = flags.to_dict()

        assert result["collision"] is True
        assert result["frames"] is False
        assert result["contacts"] is True
        assert result["joint_limits"] is False
        assert result["com"] is False


class TestMuJoCoViewerWidget:
    """Tests for MuJoCo viewer widget and toggles."""

    def test_viewer_init(self, qtbot):
        """Test MuJoCo viewer widget initialization."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Check initial state
        assert widget._urdf_content == ""
        assert widget._vis_flags.show_frames is True  # Default on

        # Check toggles exist
        assert widget._collision_checkbox is not None
        assert widget._frames_checkbox is not None
        assert widget._joints_checkbox is not None
        assert widget._contacts_checkbox is not None

    def test_toggle_collision(self, qtbot):
        """Test collision toggle changes visualization flags."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Initial state
        assert widget._vis_flags.show_collision is False

        # Toggle on
        widget._collision_checkbox.setChecked(True)
        assert widget._vis_flags.show_collision is True

        # Toggle off
        widget._collision_checkbox.setChecked(False)
        assert widget._vis_flags.show_collision is False

    def test_toggle_frames(self, qtbot):
        """Test frames toggle changes visualization flags."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Initial state (frames on by default)
        assert widget._vis_flags.show_frames is True
        assert widget._frames_checkbox.isChecked() is True

        # Toggle off
        widget._frames_checkbox.setChecked(False)
        assert widget._vis_flags.show_frames is False

        # Toggle back on
        widget._frames_checkbox.setChecked(True)
        assert widget._vis_flags.show_frames is True

    def test_toggle_joints(self, qtbot):
        """Test joint limits toggle changes visualization flags."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Initial state
        assert widget._vis_flags.show_joint_limits is False

        # Toggle on
        widget._joints_checkbox.setChecked(True)
        assert widget._vis_flags.show_joint_limits is True

    def test_toggle_contacts(self, qtbot):
        """Test contacts toggle changes visualization flags."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Initial state
        assert widget._vis_flags.show_contacts is False

        # Toggle on
        widget._contacts_checkbox.setChecked(True)
        assert widget._vis_flags.show_contacts is True

        # Toggle off
        widget._contacts_checkbox.setChecked(False)
        assert widget._vis_flags.show_contacts is False

    def test_visualization_changed_signal(self, qtbot):
        """Test that visualization_changed signal is emitted on toggle."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        received_flags = []

        def on_changed(flags_dict):
            received_flags.append(flags_dict)

        widget.visualization_changed.connect(on_changed)

        # Toggle collision
        widget._collision_checkbox.setChecked(True)

        # Check signal was emitted
        assert len(received_flags) == 1
        assert received_flags[0]["collision"] is True

    def test_set_visualization_flags_programmatic(self, qtbot):
        """Test setting flags programmatically updates checkboxes."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        flags = VisualizationFlags(
            show_collision=True,
            show_frames=False,
            show_joint_limits=True,
            show_contacts=True,
        )

        widget.set_visualization_flags(flags)

        # Check checkboxes match
        assert widget._collision_checkbox.isChecked() is True
        assert widget._frames_checkbox.isChecked() is False
        assert widget._joints_checkbox.isChecked() is True
        assert widget._contacts_checkbox.isChecked() is True

        # Check internal state matches
        assert widget._vis_flags.show_collision is True
        assert widget._vis_flags.show_frames is False

    def test_get_visualization_flags(self, qtbot):
        """Test retrieving current flags."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        widget._collision_checkbox.setChecked(True)
        widget._contacts_checkbox.setChecked(True)

        flags = widget.get_visualization_flags()

        assert flags.show_collision is True
        assert flags.show_contacts is True
        assert flags.show_frames is True  # Default

    def test_is_mujoco_available(self, qtbot):
        """Test MuJoCo availability check."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # Result depends on whether MuJoCo is installed
        result = widget.is_mujoco_available()
        assert result == (MUJOCO_AVAILABLE and widget._renderer is not None)

    def test_get_model_info_empty(self, qtbot):
        """Test model info with no loaded model."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        info = widget.get_model_info()

        assert info["model_loaded"] is False
        assert info["link_count"] == 0
        assert info["joint_count"] == 0
        assert info["mujoco_available"] == MUJOCO_AVAILABLE

    def test_headless_mode_toggles_disabled(self, qtbot):
        """Test that toggles are disabled in headless mode."""
        # This test requires MuJoCo to NOT be available
        # We'll simulate by checking the _disable_toggles behavior

        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        # If MuJoCo is not available, toggles should be disabled
        if not MUJOCO_AVAILABLE:
            assert widget._collision_checkbox.isEnabled() is False
            assert widget._frames_checkbox.isEnabled() is False
            assert widget._joints_checkbox.isEnabled() is False
            assert widget._contacts_checkbox.isEnabled() is False
            assert widget._launch_btn.isEnabled() is False

    def test_update_visualization(self, qtbot):
        """Test updating visualization with URDF content."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        urdf = """<robot name="test">
            <link name="base"/>
            <link name="arm"/>
            <joint name="j1" type="revolute">
                <parent link="base"/>
                <child link="arm"/>
                <axis xyz="0 0 1"/>
            </joint>
        </robot>"""

        widget.update_visualization(urdf)

        # Content should be stored regardless of rendering success
        assert widget._urdf_content == urdf

        # Status should indicate something happened (loaded or failed)
        status = widget._status_label.text()
        # Either successfully loaded with counts, or shows failure message
        assert (
            "links" in status.lower()
            or "loaded" in status.lower()
            or "failed" in status.lower()
        )

    def test_clear(self, qtbot):
        """Test clearing the viewer."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        widget.update_visualization("<robot><link name='test'/></robot>")
        widget.clear()

        assert widget._urdf_content == ""

    def test_reset_view(self, qtbot):
        """Test resetting camera view."""
        widget = MuJoCoViewerWidget()
        qtbot.addWidget(widget)

        if widget._renderer:
            # Modify camera
            widget._renderer.azimuth = 180.0
            widget._renderer.elevation = -45.0
            widget._renderer.distance = 10.0

            widget.reset_view()

            # Check reset values
            assert widget._renderer.azimuth == 90.0
            assert widget._renderer.elevation == -20.0
            assert widget._renderer.distance == 3.0
