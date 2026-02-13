from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.shared.python.plotting import GolfSwingPlotter, RecorderInterface


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self) -> None:
        self.engine = None
        self.data = {
            "joint_positions": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "joint_velocities": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "joint_accelerations": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "joint_torques": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "actuator_powers": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "kinetic_energy": (np.linspace(0, 1, 10), np.random.rand(10)),
            "potential_energy": (np.linspace(0, 1, 10), np.random.rand(10)),
            "total_energy": (np.linspace(0, 1, 10), np.random.rand(10)),
            "club_head_speed": (np.linspace(0, 1, 10), np.random.rand(10)),
            "club_head_position": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "angular_momentum": (np.linspace(0, 1, 10), np.random.rand(10, 3)),
            "cop_position": (np.linspace(0, 1, 10), np.random.rand(10, 2)),
        }
        self.induced_acc = {"gravity": (np.linspace(0, 1, 10), np.random.rand(10, 3))}
        self.counterfactuals = {"ztcf": (np.linspace(0, 1, 10), np.random.rand(10, 3))}

    def get_time_series(self, field_name: str) -> tuple:
        """Return time series data for the given field."""
        return self.data.get(field_name, (np.array([]), np.array([])))

    def get_induced_acceleration_series(self, source_name: str | int) -> tuple:
        """Return induced acceleration series for the given source."""
        if not isinstance(source_name, str):
            return np.array([]), np.array([])
        return self.induced_acc.get(source_name, (np.array([]), np.array([])))

    def get_counterfactual_series(self, cf_name: str) -> tuple:
        """Return counterfactual series for the given name."""
        return self.counterfactuals.get(cf_name, (np.array([]), np.array([])))

    def set_analysis_config(self, config: dict) -> None:  # noqa: ANN001
        """Set analysis config (no-op for mock)."""


@pytest.fixture
def mock_recorder() -> MockRecorder:
    """Create a MockRecorder instance."""
    return MockRecorder()


@pytest.fixture
def plotter(mock_recorder) -> GolfSwingPlotter:
    """Create a GolfSwingPlotter with mock recorder."""
    return GolfSwingPlotter(mock_recorder, joint_names=["Joint1", "Joint2", "Joint3"])


@pytest.fixture
def fig() -> Figure:
    """Create a matplotlib Figure."""
    return Figure()


def test_init(mock_recorder) -> None:
    """Test GolfSwingPlotter initialization."""
    plotter = GolfSwingPlotter(mock_recorder)
    assert plotter.recorder == mock_recorder
    assert plotter.joint_names == []

    plotter_named = GolfSwingPlotter(mock_recorder, ["J1"])
    assert plotter_named.joint_names == ["J1"]


def test_get_joint_name(plotter) -> None:
    """Test joint name lookup by index."""
    assert plotter.get_joint_name(0) == "Joint1"
    assert plotter.get_joint_name(1) == "Joint2"
    assert plotter.get_joint_name(99) == "Joint 99"


def test_get_aligned_label(plotter) -> None:
    """Test aligned label resolution for floating base offsets."""
    # Perfect match
    assert plotter._get_aligned_label(0, 3) == "Joint1"

    # Mismatch (floating base typical case: 7 positions, 3 joints provided)
    # Assumes joints are at the end
    # offset = 7 - 3 = 4
    # idx 4 -> name_idx 0
    assert plotter._get_aligned_label(4, 7) == "Joint1"
    assert plotter._get_aligned_label(0, 7) == "DoF 0"


def test_plot_joint_angles(plotter, fig) -> None:
    """Test joint angles plotting."""
    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0

    # Test with empty data
    plotter.recorder.data["joint_positions"] = ([], [])
    fig.clear()
    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0  # Should still create axes to show "No data" text


def test_plot_joint_velocities(plotter, fig) -> None:
    """Test joint velocities plotting."""
    plotter.plot_joint_velocities(fig)
    assert len(fig.axes) > 0


def test_plot_joint_torques(plotter, fig) -> None:
    """Test joint torques plotting."""
    plotter.plot_joint_torques(fig)
    assert len(fig.axes) > 0


def test_plot_actuator_powers(plotter, fig) -> None:
    """Test actuator powers plotting."""
    plotter.plot_actuator_powers(fig)
    assert len(fig.axes) > 0


def test_plot_energy_analysis(plotter, fig) -> None:
    """Test energy analysis plotting."""
    plotter.plot_energy_analysis(fig)
    assert len(fig.axes) > 0


def test_plot_club_head_speed(plotter, fig) -> None:
    """Test club head speed plotting."""
    plotter.plot_club_head_speed(fig)
    assert len(fig.axes) > 0


def test_plot_club_head_trajectory(plotter, fig) -> None:
    """Test 3D club head trajectory plotting."""
    plotter.plot_club_head_trajectory(fig)
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "3d"


def test_plot_phase_diagram(plotter, fig) -> None:
    """Test phase diagram plotting."""
    plotter.plot_phase_diagram(fig, joint_idx=0)
    assert len(fig.axes) > 0


def test_plot_torque_comparison(plotter, fig) -> None:
    """Test torque comparison plotting."""
    plotter.plot_torque_comparison(fig)
    assert len(fig.axes) > 0


def test_plot_frequency_analysis(plotter, fig) -> None:
    """Test frequency analysis plotting."""
    with patch("scipy.signal.welch", return_value=(np.array([1, 2]), np.array([1, 2]))):
        plotter.plot_frequency_analysis(fig, joint_idx=0)
        assert len(fig.axes) > 0


def test_plot_spectrogram(plotter, fig) -> None:
    """Test spectrogram plotting."""
    with patch(
        "scipy.signal.spectrogram",
        return_value=(np.array([1]), np.array([1]), np.array([[1]])),
    ):
        plotter.plot_spectrogram(fig, joint_idx=0)
        assert len(fig.axes) > 0


def test_plot_summary_dashboard(plotter, fig) -> None:
    """Test summary dashboard multi-panel plotting."""
    plotter.plot_summary_dashboard(fig)
    # Should have multiple axes
    assert len(fig.axes) >= 6


def test_plot_kinematic_sequence(plotter, fig) -> None:
    """Test kinematic sequence plotting."""
    segments = {"Seg1": 0, "Seg2": 1}
    plotter.plot_kinematic_sequence(fig, segments)
    assert len(fig.axes) > 0


def test_plot_3d_phase_space(plotter, fig) -> None:
    """Test 3D phase space plotting."""
    plotter.plot_3d_phase_space(fig, joint_idx=0)
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "3d"


def test_plot_correlation_matrix(plotter, fig) -> None:
    """Test correlation matrix plotting."""
    plotter.plot_correlation_matrix(fig)
    assert len(fig.axes) > 0


def test_plot_swing_plane(plotter, fig) -> None:
    """Test swing plane fitting and plotting."""
    # Need enough points for fit
    N = 10
    positions = np.random.rand(N, 3)
    # Make them roughly planar
    positions[:, 2] = positions[:, 0] * 0.5 + positions[:, 1] * 0.5
    plotter.recorder.data["club_head_position"] = (np.linspace(0, 1, N), positions)

    plotter.plot_swing_plane(fig)
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "3d"


def test_plot_angular_momentum(plotter, fig) -> None:
    """Test angular momentum plotting."""
    plotter.plot_angular_momentum(fig)
    assert len(fig.axes) > 0


def test_plot_cop_trajectory(plotter, fig) -> None:
    """Test center of pressure trajectory plotting."""
    plotter.plot_cop_trajectory(fig)
    assert len(fig.axes) > 0


def test_plot_cop_vector_field(plotter, fig) -> None:
    """Test center of pressure vector field plotting."""
    plotter.plot_cop_vector_field(fig)
    assert len(fig.axes) > 0


def test_plot_radar_chart(plotter, fig) -> None:
    """Test radar chart plotting."""
    metrics = {"A": 0.5, "B": 0.8, "C": 0.2}
    plotter.plot_radar_chart(fig, metrics)
    assert len(fig.axes) > 0
    assert fig.axes[0].name == "polar"


def test_plot_power_flow(plotter, fig) -> None:
    """Test power flow plotting."""
    plotter.plot_power_flow(fig)
    assert len(fig.axes) > 0


def test_plot_induced_acceleration(plotter, fig) -> None:
    """Test induced acceleration plotting."""
    plotter.plot_induced_acceleration(fig, "gravity")
    assert len(fig.axes) > 0


def test_plot_counterfactual_comparison(plotter, fig) -> None:
    """Test counterfactual comparison plotting."""
    plotter.plot_counterfactual_comparison(fig, "ztcf")
    assert len(fig.axes) > 0
