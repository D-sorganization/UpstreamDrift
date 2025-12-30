from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.plotting import GolfSwingPlotter


@pytest.fixture
def mock_recorder():
    recorder = MagicMock()

    # Default behavior: return some dummy data
    times = np.linspace(0, 1, 100)
    # data = np.random.rand(100, 3)  # Unused

    def get_time_series(field):
        if field == "joint_positions":
            return times, np.random.rand(100, 3)
        elif field == "joint_velocities":
            return times, np.random.rand(100, 3)
        elif field == "joint_torques":
            return times, np.random.rand(100, 3)
        elif field == "actuator_powers":
            return times, np.random.rand(100, 3)
        elif field == "kinetic_energy":
            return times, np.random.rand(100)
        elif field == "potential_energy":
            return times, np.random.rand(100)
        elif field == "total_energy":
            return times, np.random.rand(100)
        elif field == "club_head_speed":
            return times, np.random.rand(100)
        elif field == "club_head_position":
            return times, np.random.rand(100, 3)
        elif field == "angular_momentum":
            return times, np.random.rand(100, 3)
        elif field == "cop_position":
            return times, np.random.rand(100, 2)
        elif field == "joint_accelerations":
            return times, np.random.rand(100, 3)
        return [], []

    recorder.get_time_series.side_effect = get_time_series

    recorder.get_induced_acceleration_series.return_value = (
        times,
        np.random.rand(100, 3),
    )
    recorder.get_counterfactual_series.return_value = (times, np.random.rand(100, 3))

    return recorder


@pytest.fixture
def plotter(mock_recorder):
    return GolfSwingPlotter(
        mock_recorder, joint_names=["Joint 0", "Joint 1", "Joint 2"]
    )


def test_init(plotter):
    assert len(plotter.joint_names) == 3


def test_get_joint_name(plotter):
    assert plotter.get_joint_name(0) == "Joint 0"
    assert plotter.get_joint_name(5) == "Joint 5"


def test_plot_joint_angles(plotter):
    fig = Figure()
    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0

    # Test empty data
    plotter.recorder.get_time_series.side_effect = lambda x: ([], [])
    fig = Figure()
    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0
    assert fig.axes[0].texts[0].get_text() == "No data recorded"


def test_plot_joint_velocities(plotter):
    fig = Figure()
    plotter.plot_joint_velocities(fig)
    assert len(fig.axes) > 0


def test_plot_joint_torques(plotter):
    fig = Figure()
    plotter.plot_joint_torques(fig)
    assert len(fig.axes) > 0


def test_plot_actuator_powers(plotter):
    fig = Figure()
    plotter.plot_actuator_powers(fig)
    assert len(fig.axes) > 0


def test_plot_energy_analysis(plotter):
    fig = Figure()
    plotter.plot_energy_analysis(fig)
    assert len(fig.axes) > 0


def test_plot_club_head_speed(plotter):
    fig = Figure()
    plotter.plot_club_head_speed(fig)
    assert len(fig.axes) > 0


def test_plot_club_head_trajectory(plotter):
    fig = Figure()
    plotter.plot_club_head_trajectory(fig)
    assert len(fig.axes) > 0


def test_plot_phase_diagram(plotter):
    fig = Figure()
    plotter.plot_phase_diagram(fig, joint_idx=0)
    assert len(fig.axes) > 0

    # Test out of bounds
    plotter.recorder.get_time_series.side_effect = lambda x: (
        np.linspace(0, 1, 10),
        np.random.rand(10, 1),
    )
    fig = Figure()
    plotter.plot_phase_diagram(fig, joint_idx=5)
    assert len(fig.axes) > 0
    assert "No data available" in fig.axes[0].texts[0].get_text()


def test_plot_torque_comparison(plotter):
    fig = Figure()
    plotter.plot_torque_comparison(fig)
    assert len(fig.axes) > 0


def test_plot_frequency_analysis(plotter):
    fig = Figure()
    plotter.plot_frequency_analysis(fig, joint_idx=0, signal_type="velocity")
    assert len(fig.axes) > 0

    plotter.plot_frequency_analysis(fig, joint_idx=0, signal_type="position")
    plotter.plot_frequency_analysis(fig, joint_idx=0, signal_type="torque")


def test_plot_spectrogram(plotter):
    fig = Figure()
    plotter.plot_spectrogram(fig, joint_idx=0)
    assert len(fig.axes) > 0


def test_plot_summary_dashboard(plotter):
    fig = Figure()
    plotter.plot_summary_dashboard(fig)
    assert len(fig.axes) == 6


def test_plot_kinematic_sequence(plotter):
    fig = Figure()
    segment_indices = {"Seg1": 0, "Seg2": 1}
    plotter.plot_kinematic_sequence(fig, segment_indices)
    assert len(fig.axes) > 0


def test_plot_3d_phase_space(plotter):
    fig = Figure()
    plotter.plot_3d_phase_space(fig, joint_idx=0)
    assert len(fig.axes) > 0


def test_plot_correlation_matrix(plotter):
    fig = Figure()
    plotter.plot_correlation_matrix(fig)
    assert len(fig.axes) > 0


def test_plot_swing_plane(plotter):
    fig = Figure()
    plotter.plot_swing_plane(fig)
    assert len(fig.axes) > 0

    # Test insufficient data
    plotter.recorder.get_time_series.side_effect = lambda x: (
        np.array([0, 1]),
        np.random.rand(2, 3),
    )
    fig = Figure()
    plotter.plot_swing_plane(fig)
    assert "Insufficient data" in fig.axes[0].texts[0].get_text()


def test_plot_angular_momentum(plotter):
    fig = Figure()
    plotter.plot_angular_momentum(fig)
    assert len(fig.axes) > 0


def test_plot_cop_trajectory(plotter):
    fig = Figure()
    plotter.plot_cop_trajectory(fig)
    assert len(fig.axes) > 0


def test_plot_cop_vector_field(plotter):
    fig = Figure()
    plotter.plot_cop_vector_field(fig)
    assert len(fig.axes) > 0


def test_plot_radar_chart(plotter):
    fig = Figure()
    metrics = {"A": 10, "B": 20, "C": 30}
    plotter.plot_radar_chart(fig, metrics)
    assert len(fig.axes) > 0

    # Less than 3 metrics
    fig = Figure()
    plotter.plot_radar_chart(fig, {"A": 10, "B": 20})
    assert "Need at least 3 metrics" in fig.axes[0].texts[0].get_text()


def test_plot_power_flow(plotter):
    fig = Figure()
    plotter.plot_power_flow(fig)
    assert len(fig.axes) > 0


def test_plot_induced_acceleration(plotter):
    fig = Figure()
    plotter.plot_induced_acceleration(fig, "gravity", joint_idx=0)
    assert len(fig.axes) > 0

    plotter.plot_induced_acceleration(fig, "gravity", joint_idx=None)
    assert len(fig.axes) > 1


def test_plot_counterfactual_comparison(plotter):
    fig = Figure()
    plotter.plot_counterfactual_comparison(fig, "ztcf")
    assert len(fig.axes) > 0
