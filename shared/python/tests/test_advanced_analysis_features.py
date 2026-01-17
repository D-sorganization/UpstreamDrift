"""Tests for Advanced Analysis Features (Wavelet, Swing Plane, Correlation)."""

import numpy as np
import pytest

from shared.python.dashboard.advanced_analysis import (
    AdvancedAnalysisDialog,
    CorrelationTab,
    SwingPlaneTab,
    WaveletTab,
)


class MockRecorder:
    def get_time_series(self, key: str):
        t = np.linspace(0, 1, 100)

        # Base signal
        base = np.sin(2 * np.pi * 10 * t)

        if key == "club_head_position":
            # 3D Spiral for Swing Plane
            x = np.cos(2 * np.pi * t)
            y = np.sin(2 * np.pi * t)
            z = t
            data = np.column_stack([x, y, z])
        elif key in [
            "joint_positions",
            "joint_velocities",
            "joint_torques",
            "actuator_forces",
        ]:
            # 2D Array (N, 2)
            data = np.column_stack([base, base * 0.5])
        elif key in [
            "club_head_speed",
            "kinetic_energy",
            "total_energy",
            "control_accel",
        ]:
            # Scalar (N,) or (N, 1)
            data = base * 10.0
        elif key in ["left_foot_force", "right_foot_force"]:
            # 3D vector (N, 3)
            data = np.column_stack([base, base, base])
        else:
            # Default
            data = base

        return t, data


@pytest.fixture
def app(qapp):
    return qapp


@pytest.fixture
def recorder():
    return MockRecorder()


def test_wavelet_tab(recorder, qtbot):
    tab = WaveletTab(recorder)
    qtbot.addWidget(tab)

    # Check initial plot (pcolormesh creates a collection)
    assert len(tab.ax.collections) > 0

    # Change w0
    tab.spin_w0.setValue(10.0)
    # Should re-plot
    assert len(tab.ax.collections) > 0


def test_swing_plane_tab(recorder, qtbot):
    tab = SwingPlaneTab(recorder)
    qtbot.addWidget(tab)

    # It creates two subplots.
    # We can check that the figure has axes.
    assert len(tab.canvas.fig.axes) >= 2

    # The first axis is 3D, so it should have collections (scatter) or patch (surface)
    ax3d = tab.canvas.fig.axes[0]
    # Matplotlib 3D axes add collections for scatter and polygons for surface
    assert len(ax3d.collections) > 0 or len(ax3d.patches) > 0


def test_correlation_tab(recorder, qtbot):
    tab = CorrelationTab(recorder)
    qtbot.addWidget(tab)

    # Initial plot might not run until update_plot is called manually or by init?
    # In my code: "# Do not update automatically on init to save time, or do it once"
    # But I see I commented it out: "# self.update_plot()" in the code?
    # Wait, let me check the code I wrote.
    # Ah, I see: `btn_refresh.clicked.connect(self.update_plot)` and `# self.update_plot()`
    # So it doesn't run on init.

    # Trigger update
    tab.update_plot()

    # Check if image is plotted (imshow creates an image)
    assert len(tab.ax.images) > 0


def test_advanced_analysis_dialog_features(recorder, qtbot):
    dlg = AdvancedAnalysisDialog(None, recorder)
    qtbot.addWidget(dlg)

    # Check that new tabs are present
    assert dlg.tabs.count() == 6
    assert isinstance(dlg.tab_wavelet, WaveletTab)
    assert isinstance(dlg.tab_swing_plane, SwingPlaneTab)
    assert isinstance(dlg.tab_correlation, CorrelationTab)
