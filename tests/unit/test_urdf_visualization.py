"""Tests for URDF visualization widget."""

# Skip tests if running headless without X11/Wayland, unless using offscreen platform
# But usually pytest-qt handles this if env var is set.
# The project has explicit skipping for GUI tests in some places, but let's try standard approach.

from tools.urdf_generator.visualization_widget import VisualizationWidget


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
    qtbot.addWidget(widget)

    # Modify camera state via the GL widget
    widget.gl_widget.camera_distance = 10.0
    widget.gl_widget.camera_rotation_x = 45.0

    widget.reset_view()

    assert widget.gl_widget.camera_distance == 1.0
    assert widget.gl_widget.camera_rotation_x == 0.0
