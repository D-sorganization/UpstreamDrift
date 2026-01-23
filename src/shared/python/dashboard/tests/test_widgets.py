from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.dashboard.widgets import FrequencyAnalysisDialog, LivePlotWidget
from src.shared.python.interfaces import RecorderInterface


class MockRecorder(RecorderInterface):
    def __init__(self):
        self.engine = None
        self.data = {
            "joint_positions": (np.arange(100) * 0.1, np.random.rand(100, 3)),
            "joint_velocities": (np.arange(100) * 0.1, np.random.rand(100, 3)),
        }
        self.config = {}

    def get_time_series(self, key):
        return self.data.get(key, (np.array([]), None))

    def get_induced_acceleration_series(self, src_idx):
        return np.array([]), None

    def set_analysis_config(self, config):
        self.config = config

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        pass

    def record_step(self):
        pass

    def compute_analysis_post_hoc(self):
        pass

    def get_counterfactual_series(self, key):
        return np.array([]), None

    def get_data_dict(self):
        # Fix: return the data dict so widget can initialize metrics if it uses this (it doesn't actually use this for init, it uses hardcoded keys, but good practice)
        return self.data


@pytest.fixture
def app(qapp):
    return qapp


@pytest.fixture
def recorder():
    return MockRecorder()


def test_live_plot_widget_init(app, recorder):
    widget = LivePlotWidget(recorder)
    assert widget.current_key == "joint_positions"
    assert widget.lbl_stats.text().startswith("Mean:")


def test_live_plot_widget_stats_update(app, recorder):
    widget = LivePlotWidget(recorder)
    widget.update_plot()

    # Check if stats label updated
    text = widget.lbl_stats.text()
    assert "Mean:" in text
    assert "Std:" in text
    # Should not be 0.00 since we have random data
    assert text != "Mean: 0.00 | Std: 0.00 | Min: 0.00 | Max: 0.00"


def test_live_plot_widget_xy_mode(app, recorder):
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


def test_live_plot_widget_freq_analysis(app, recorder):
    widget = LivePlotWidget(recorder)

    # Mock QDialog.exec to prevent blocking
    # Also mock compute_psd to avoid needing full signal processing
    with patch.object(FrequencyAnalysisDialog, "exec", return_value=None) as mock_exec:
        with patch(
            "shared.python.dashboard.widgets.compute_psd",
            return_value=(np.linspace(0, 10, 10), np.random.rand(3, 10)),
        ):
            widget.show_freq_analysis()
            mock_exec.assert_called_once()
            # Verify psd called?
            # widget.show_freq_analysis instantiates FrequencyAnalysisDialog which calls compute_psd


def test_frequency_analysis_dialog(app):
    data = np.random.rand(100, 2)
    fs = 10.0

    # Mock compute_psd
    with patch(
        "shared.python.dashboard.widgets.compute_psd",
        return_value=(np.linspace(0, 10, 10), np.random.rand(2, 10)),
    ):
        dlg = FrequencyAnalysisDialog(None, data, fs, "Test Data")
        assert dlg.ax is not None
