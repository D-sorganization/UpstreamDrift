from unittest.mock import MagicMock

import numpy as np
import pytest
from matplotlib.figure import Figure

from src.shared.python.plotting import GolfSwingPlotter

# Check if 3D projection is available (broken on some numpy/matplotlib combos)
try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    _HAS_3D = True
except (ImportError, AttributeError):
    _HAS_3D = False

_skip_no_3d = pytest.mark.skipif(not _HAS_3D, reason="mpl_toolkits.mplot3d unavailable")


@pytest.fixture
def mock_recorder():
    recorder = MagicMock()

    # Default behavior: return some dummy data
    times = np.linspace(0, 1, 100)

    def get_time_series(field):
        if (
            field == "joint_positions"
            or field == "joint_velocities"
            or field == "joint_torques"
            or field == "actuator_powers"
        ):
            return times, np.random.rand(100, 3)
        elif (
            field == "kinetic_energy"
            or field == "potential_energy"
            or field == "total_energy"
            or field == "club_head_speed"
        ):
            return times, np.random.rand(100)
        elif field == "club_head_position" or field == "angular_momentum":
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
    plotter.clear_cache()  # Must clear cache after modifying recorder
    fig = Figure()
    plotter.plot_joint_angles(fig)
    assert len(fig.axes) > 0
    assert fig.axes[0].texts[0].get_text() == "No data recorded"


@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_joint_velocities",
        "plot_joint_torques",
        "plot_actuator_powers",
        "plot_energy_analysis",
        "plot_club_head_speed",
    ],
    ids=[
        "joint_velocities",
        "joint_torques",
        "actuator_powers",
        "energy_analysis",
        "club_head_speed",
    ],
)
def test_plot_method_creates_axes(plotter, plot_method):
    fig = Figure()
    getattr(plotter, plot_method)(fig)
    assert len(fig.axes) > 0


@_skip_no_3d
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
    plotter.clear_cache()  # Must clear cache after modifying recorder
    fig = Figure()
    plotter.plot_phase_diagram(fig, joint_idx=5)
    assert len(fig.axes) > 0
    assert "No data available" in fig.axes[0].texts[0].get_text()


@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_torque_comparison",
        "plot_spectrogram",
    ],
    ids=[
        "torque_comparison",
        "spectrogram",
    ],
)
def test_plot_method_creates_axes_extra(plotter, plot_method):
    fig = Figure()
    method = getattr(plotter, plot_method)
    if plot_method == "plot_spectrogram":
        method(fig, joint_idx=0)
    else:
        method(fig)
    assert len(fig.axes) > 0


@pytest.mark.parametrize(
    "signal_type",
    ["velocity", "position", "torque"],
    ids=["freq_velocity", "freq_position", "freq_torque"],
)
def test_plot_frequency_analysis(plotter, signal_type):
    fig = Figure()
    plotter.plot_frequency_analysis(fig, joint_idx=0, signal_type=signal_type)
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


@_skip_no_3d
def test_plot_3d_phase_space(plotter):
    fig = Figure()
    plotter.plot_3d_phase_space(fig, joint_idx=0)
    assert len(fig.axes) > 0


def test_plot_correlation_matrix(plotter):
    fig = Figure()
    plotter.plot_correlation_matrix(fig)
    assert len(fig.axes) > 0


@_skip_no_3d
def test_plot_swing_plane(plotter):
    fig = Figure()
    plotter.plot_swing_plane(fig)
    assert len(fig.axes) > 0

    # Test insufficient data
    plotter.recorder.get_time_series.side_effect = lambda x: (
        np.array([0, 1]),
        np.random.rand(2, 3),
    )
    plotter.clear_cache()  # Must clear cache after modifying recorder
    fig = Figure()
    plotter.plot_swing_plane(fig)
    assert "Insufficient data" in fig.axes[0].texts[0].get_text()


@pytest.mark.parametrize(
    "plot_method",
    [
        "plot_angular_momentum",
        "plot_cop_trajectory",
        "plot_cop_vector_field",
    ],
    ids=[
        "angular_momentum",
        "cop_trajectory",
        "cop_vector_field",
    ],
)
def test_plot_method_spatial(plotter, plot_method):
    fig = Figure()
    getattr(plotter, plot_method)(fig)
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


@pytest.mark.parametrize(
    "plot_method,args",
    [
        ("plot_power_flow", {}),
        ("plot_counterfactual_comparison", {"cf_name": "ztcf"}),
    ],
    ids=["power_flow", "counterfactual_comparison"],
)
def test_plot_method_advanced(plotter, plot_method, args):
    fig = Figure()
    getattr(plotter, plot_method)(fig, **args)
    assert len(fig.axes) > 0


@pytest.mark.parametrize(
    "joint_idx",
    [0, None],
    ids=["specific_joint", "all_joints"],
)
def test_plot_induced_acceleration(plotter, joint_idx):
    fig = Figure()
    plotter.plot_induced_acceleration(fig, "gravity", joint_idx=joint_idx)
    assert len(fig.axes) > 0
