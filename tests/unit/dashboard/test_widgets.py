from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from PyQt6.QtWidgets import QApplication

from src.shared.python.dashboard.widgets import (
    ControlPanel,
    FrequencyAnalysisDialog,
    LivePlotWidget,
)
from src.shared.python.engine_core.interfaces import RecorderInterface


class MockRecorder(RecorderInterface):
    """Mock recorder for widget testing."""

    def __init__(self) -> None:
        """Initialize mock recorder with random joint data."""
        self.engine = None
        self.data = {
            "joint_positions": (np.arange(100) * 0.1, np.random.rand(100, 3)),
            "joint_velocities": (np.arange(100) * 0.1, np.random.rand(100, 3)),
        }
        self.config: dict[str, object] = {}

    def get_time_series(self, key) -> tuple:
        """Return time series data for the given key."""
        return self.data.get(key, (np.array([]), None))

    def get_induced_acceleration_series(self, src_idx) -> tuple:
        """Return empty induced acceleration series."""
        return np.array([]), None

    def set_analysis_config(self, config) -> None:
        """Set the analysis configuration."""
        self.config = config

    def start(self) -> None:
        """Start recording."""

    def stop(self) -> None:
        """Stop recording."""

    def reset(self) -> None:
        """Reset recorder state."""

    def record_step(self) -> None:
        """Record a single step."""

    def compute_analysis_post_hoc(self) -> None:
        """Compute post-hoc analysis."""

    def get_counterfactual_series(self, key) -> tuple:
        """Return empty counterfactual series."""
        return np.array([]), None

    def get_data_dict(self) -> dict:
        """Return the data dictionary."""
        # Fix: return the data dict so widget can initialize metrics if it uses this (it doesn't actually use this for init, it uses hardcoded keys, but good practice)
        return self.data


@pytest.fixture
def app(qapp) -> QApplication:
    """Provide the QApplication instance."""
    return qapp


@pytest.fixture
def recorder() -> MockRecorder:
    """Create a MockRecorder instance."""
    return MockRecorder()


def test_live_plot_widget_init(app, recorder) -> None:
    """Test LivePlotWidget initializes correctly."""
    widget = LivePlotWidget(recorder)
    assert widget.current_key == "joint_positions"
    assert widget.lbl_stats.text().startswith("Mean:")


def test_live_plot_widget_stats_update(app, recorder) -> None:
    """Test LivePlotWidget stats label updates on plot refresh."""
    widget = LivePlotWidget(recorder)
    widget.update_plot()

    # Check if stats label updated
    text = widget.lbl_stats.text()
    assert "Mean:" in text
    assert "Std:" in text
    # Should not be 0.00 since we have random data
    assert text != "Mean: 0.00 | Std: 0.00 | Min: 0.00 | Max: 0.00"


def test_live_plot_widget_xy_mode(app, recorder) -> None:
    """Test LivePlotWidget X-Y scatter plot mode."""
    widget = LivePlotWidget(recorder)

    # Enable X-Y mode
    widget.chk_xy.setChecked(True)
    assert widget.chk_compare.isChecked()  # Should auto-enable compare

    # Set comparison
    widget.combo_compare.setCurrentText("Joint Velocities")

    # Update plot
    widget.update_plot()

    # Verify plot title or axes labels
    title = widget.ax.get_title()
    assert "(X)" in title
    assert "(Y)" in title
    assert "Joint Positions" in widget.ax.get_xlabel()
    assert "Joint Velocities" in widget.ax.get_ylabel()

    # Verify ax2 is None (removed)
    assert widget.ax2 is None


def test_live_plot_widget_freq_analysis(app, recorder) -> None:
    """Test LivePlotWidget frequency analysis dialog launch."""
    widget = LivePlotWidget(recorder)

    # Mock QDialog.exec to prevent blocking
    # Also mock compute_psd to avoid needing full signal processing
    with (
        patch.object(FrequencyAnalysisDialog, "exec", return_value=None) as mock_exec,
        patch(
            "shared.python.dashboard.widgets.compute_psd",
            return_value=(np.linspace(0, 10, 10), np.random.rand(3, 10)),
        ),
    ):
        widget.show_freq_analysis()
        mock_exec.assert_called_once()
        # Verify psd called?
        # widget.show_freq_analysis instantiates FrequencyAnalysisDialog which calls compute_psd


def test_frequency_analysis_dialog(app) -> None:
    """Test FrequencyAnalysisDialog initialization."""
    data = np.random.rand(100, 2)
    fs = 10.0

    # Mock compute_psd
    with patch(
        "shared.python.dashboard.widgets.compute_psd",
        return_value=(np.linspace(0, 10, 10), np.random.rand(2, 10)),
    ):
        dlg = FrequencyAnalysisDialog(None, data, fs, "Test Data")
        assert dlg.ax is not None


# =============================================================================
# ControlPanel Tests
# =============================================================================


@pytest.fixture
def control_panel(qapp) -> ControlPanel:
    """Fixture for ControlPanel widget."""
    return ControlPanel()


def test_control_panel_init(control_panel) -> None:
    """Test ControlPanel initializes with all buttons."""
    assert control_panel.btn_start is not None
    assert control_panel.btn_pause is not None
    assert control_panel.btn_stop is not None
    assert control_panel.btn_reset is not None


def test_control_panel_start_signal(control_panel, qtbot) -> None:
    """Test start button emits start_requested signal."""
    with qtbot.waitSignal(control_panel.start_requested, timeout=1000):
        control_panel.btn_start.click()


def test_control_panel_pause_signal(control_panel, qtbot) -> None:
    """Test pause button emits pause_requested signal."""
    with qtbot.waitSignal(control_panel.pause_requested, timeout=1000):
        control_panel.btn_pause.click()


def test_control_panel_stop_signal(control_panel, qtbot) -> None:
    """Test stop button emits stop_requested signal."""
    with qtbot.waitSignal(control_panel.stop_requested, timeout=1000):
        control_panel.btn_stop.click()


def test_control_panel_reset_signal(control_panel, qtbot) -> None:
    """Test reset button emits reset_requested signal."""
    with qtbot.waitSignal(control_panel.reset_requested, timeout=1000):
        control_panel.btn_reset.click()


def test_control_panel_pause_checkable(control_panel) -> None:
    """Test pause button is checkable (toggle state)."""
    assert control_panel.btn_pause.isCheckable()
    control_panel.btn_pause.click()
    assert control_panel.btn_pause.isChecked()
    control_panel.btn_pause.click()
    assert not control_panel.btn_pause.isChecked()


def test_control_panel_tooltips(control_panel) -> None:
    """Test buttons have appropriate tooltips with shortcuts."""
    assert "Ctrl+R" in control_panel.btn_start.toolTip()
    assert "Space" in control_panel.btn_pause.toolTip()
    assert "S" in control_panel.btn_stop.toolTip()
    assert "R" in control_panel.btn_reset.toolTip()


def test_control_panel_accessibility(control_panel) -> None:
    """Test ControlPanel has accessibility attributes."""
    assert control_panel.accessibleName() == "Simulation Control Panel"
    assert "Controls for starting" in control_panel.accessibleDescription()


# =============================================================================
# LivePlotWidget Additional Tests
# =============================================================================


def test_live_plot_widget_metric_switching(app, recorder) -> None:
    """Test switching between different metrics."""
    widget = LivePlotWidget(recorder)

    # Switch to velocities
    widget.combo.setCurrentText("Joint Velocities")
    assert widget.current_key == "joint_velocities"

    # Switch back to positions
    widget.combo.setCurrentText("Joint Positions")
    assert widget.current_key == "joint_positions"


def test_live_plot_widget_plot_mode_norm(app, recorder) -> None:
    """Test 'Norm' plot mode (magnitude of vector)."""
    widget = LivePlotWidget(recorder)
    widget.mode_combo.setCurrentText("Norm")
    widget.update_plot()
    # Should have exactly one line (the norm)
    lines = widget.ax.get_lines()
    assert len(lines) == 1


def test_live_plot_widget_empty_data(app) -> None:
    """Test widget handles empty data gracefully."""

    class EmptyRecorder(MockRecorder):
        """MockRecorder that always returns empty data."""

        def get_time_series(self, key: str) -> tuple:
            """Return empty arrays for any key."""
            return np.array([]), None

    empty_recorder = EmptyRecorder()
    widget = LivePlotWidget(empty_recorder)
    widget.update_plot()  # Should not crash

    # Stats should show zeros or N/A
    text = widget.lbl_stats.text()
    assert "Mean:" in text


def test_live_plot_widget_single_dimension(app, recorder) -> None:
    """Test 'Single Dimension' plot mode with dimension selector."""
    widget = LivePlotWidget(recorder)
    widget.mode_combo.setCurrentText("Single Dimension")

    # Set specific dimension and update
    widget.dim_spin.setValue(1)
    widget.update_plot()

    # Should have exactly one line
    lines = widget.ax.get_lines()
    assert len(lines) == 1


def test_live_plot_widget_snapshot(app, recorder, tmp_path, monkeypatch) -> None:
    """Test snapshot capture functionality."""
    from PyQt6 import QtWidgets

    widget = LivePlotWidget(recorder)

    # Mock file dialog to return a temp path
    snapshot_path = tmp_path / "snapshot.png"
    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getSaveFileName",
        lambda *args, **kwargs: (str(snapshot_path), "PNG (*.png)"),
    )

    # Trigger snapshot
    widget.copy_snapshot()

    # Verify file was created
    assert snapshot_path.exists()
