from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.comparative_analysis import AlignedSignals, ComparativeSwingAnalyzer
from shared.python.comparative_plotting import ComparativePlotter
from shared.python.plotting import RecorderInterface


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self, prefix=""):
        self.prefix = prefix
        self.data = {
            "joint_positions": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "joint_velocities": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "club_head_speed": (np.linspace(0, 1, 10), np.random.rand(10)),
            "club_head_position": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "kinetic_energy": (np.linspace(0, 1, 10), np.random.rand(10)),
        }

    def get_time_series(self, field_name: str):
        return self.data.get(field_name, (np.array([]), np.array([])))

    def get_counterfactual_series(self, field_name: str):
        """Return counterfactual data series."""
        return np.array([]), np.array([])

    def get_induced_acceleration_series(self, field_name: str):
        """Return induced acceleration data series."""
        return np.array([]), np.array([])


@pytest.fixture
def mock_analyzer():
    rec_a = MockRecorder("A")
    rec_b = MockRecorder("B")
    analyzer = MagicMock(spec=ComparativeSwingAnalyzer)
    analyzer.recorder_a = rec_a
    analyzer.recorder_b = rec_b
    analyzer.name_a = "Swing A"
    analyzer.name_b = "Swing B"

    # Mock align_signals
    def mock_align(field_name, joint_idx=None):
        if field_name not in rec_a.data:
            return None

        # Get data
        _, val_a = rec_a.data[field_name]
        _, val_b = rec_b.data[field_name]

        # Select joint if needed
        if joint_idx is not None and val_a.ndim > 1:
            val_a = val_a[:, joint_idx]
            val_b = val_b[:, joint_idx]

        times = np.linspace(0, 1, len(val_a))
        return AlignedSignals(
            times=times,
            signal_a=val_a,
            signal_b=val_b,
            error_curve=val_a - val_b,
            rms_error=0.1,
            correlation=0.9,
        )

    analyzer.align_signals.side_effect = mock_align

    # Mock report
    analyzer.generate_comparison_report.return_value = {
        "metrics": [
            MagicMock(name="Metric 1", percent_diff=10.0),
            MagicMock(name="Metric 2", percent_diff=-5.0),
        ]
    }
    # Fix metric mocks to have .name attribute accessible
    analyzer.generate_comparison_report.return_value["metrics"][0].name = "Metric 1"
    analyzer.generate_comparison_report.return_value["metrics"][1].name = "Metric 2"

    return analyzer


@pytest.fixture
def plotter(mock_analyzer):
    return ComparativePlotter(mock_analyzer)


@pytest.fixture
def fig():
    return Figure()


def test_init(plotter, mock_analyzer):
    assert plotter.analyzer == mock_analyzer


def test_plot_comparison(plotter, fig):
    plotter.plot_comparison(fig, "club_head_speed")
    assert len(fig.axes) > 0  # GridSpec creates axes on fig

    # Test unavailable data
    fig.clear()
    plotter.analyzer.align_signals.return_value = None
    plotter.plot_comparison(fig, "nonexistent")
    assert len(fig.axes) > 0  # Should show "Data not available"


def test_plot_phase_comparison(plotter, fig):
    plotter.plot_phase_comparison(fig, joint_idx=0)
    assert len(fig.axes) > 0

    # Test with ax provided
    fig.clear()
    ax = fig.add_subplot(111)
    plotter.plot_phase_comparison(ax=ax, joint_idx=0)
    assert len(fig.axes) == 1


def test_plot_3d_trajectory_comparison(plotter, fig):
    plotter.plot_3d_trajectory_comparison(fig)
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "3d"

    # Test empty data
    plotter.analyzer.recorder_a.data["club_head_position"] = ([], [])
    fig.clear()
    plotter.plot_3d_trajectory_comparison(fig)
    assert len(fig.axes) > 0  # Text axis


def test_plot_dashboard(plotter, fig):
    # This calls plot_comparison with subfigures
    # Note: Figure.add_subfigure is new in Matplotlib 3.4+

    # We need to make sure subfigures are supported or mocked if we are on old matplotlib
    # But pyproject.toml says matplotlib>=3.8.0 so we are good.

    plotter.plot_dashboard(fig)
    # Check if subfigures were added.
    # Figure.subfigs is the list of subfigures
    assert len(fig.subfigs) >= 2 or len(fig.axes) > 0


def test_plot_comparison_with_joint_idx(plotter, fig):
    plotter.plot_comparison(fig, "joint_positions", joint_idx=0)
    assert len(fig.axes) > 0


def test_plot_comparison_no_metrics(plotter, fig):
    plotter.analyzer.generate_comparison_report.return_value = {"metrics": []}
    plotter.plot_dashboard(fig)
    # Check that it didn't crash
