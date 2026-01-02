from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.plotting import GolfSwingPlotter, RecorderInterface

# --- Fixtures ---


class MockRecorder(RecorderInterface):
    """Mock recorder for testing."""

    def __init__(self):
        self.data = {
            "joint_positions": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "joint_velocities": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "joint_accelerations": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "joint_torques": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "kinetic_energy": (np.linspace(0, 1, 10), np.zeros(10)),
            "potential_energy": (np.linspace(0, 1, 10), np.zeros(10)),
            "total_energy": (np.linspace(0, 1, 10), np.zeros(10)),
            "actuator_powers": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "club_head_speed": (np.linspace(0, 1, 10), np.zeros(10)),
            "club_head_position": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "angular_momentum": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "cop_position": (np.linspace(0, 1, 10), np.zeros((10, 2))),
        }
        self.induced_accel = {
            "gravity": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "velocity": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "total": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "control": (np.linspace(0, 1, 10), np.zeros((10, 3))),
        }
        self.counterfactuals = {
            "ztcf_accel": (np.linspace(0, 1, 10), np.zeros((10, 3))),
            "zvcf_torque": (np.linspace(0, 1, 10), np.zeros((10, 3))),
        }

    def get_time_series(self, field_name: str):
        if field_name not in self.data:
            raise KeyError(f"Field {field_name} not found")
        return self.data[field_name]

    def get_induced_acceleration_series(self, source_name: str):
        if source_name not in self.induced_accel:
            raise KeyError(f"Source {source_name} not found")
        return self.induced_accel[source_name]

    def get_counterfactual_series(self, cf_name: str):
        if cf_name not in self.counterfactuals:
            raise KeyError(f"Counterfactual {cf_name} not found")
        return self.counterfactuals[cf_name]


@pytest.fixture
def recorder():
    return MockRecorder()


@pytest.fixture
def plotter(recorder):
    return GolfSwingPlotter(recorder, joint_names=["Joint1", "Joint2", "Joint3"])


@pytest.fixture
def figure():
    return Figure()


# --- Tests ---


def test_plot_joint_angles(plotter, figure):
    plotter.plot_joint_angles(figure)
    assert len(figure.axes) > 0


def test_plot_joint_angles_no_data(plotter, figure):
    plotter.recorder.data["joint_positions"] = ([], [])
    plotter.plot_joint_angles(figure)
    assert len(figure.axes) > 0  # Should still create an axis for "No data" text


def test_plot_angle_angle_diagram(plotter, figure):
    plotter.plot_angle_angle_diagram(figure, 0, 1)
    assert len(figure.axes) > 0


def test_plot_angle_angle_diagram_invalid_index(plotter, figure):
    plotter.plot_angle_angle_diagram(figure, 0, 10)
    assert len(figure.axes) > 0


def test_plot_coupling_angle(plotter, figure):
    coupling_angles = np.zeros(10)
    plotter.plot_coupling_angle(figure, coupling_angles)
    assert len(figure.axes) > 0


def test_plot_coordination_patterns(plotter, figure):
    coupling_angles = np.zeros(10)
    plotter.plot_coordination_patterns(figure, coupling_angles)
    assert len(figure.axes) > 0


def test_plot_stability_metrics(plotter, figure):
    plotter.plot_stability_metrics(figure)
    assert len(figure.axes) > 0


def test_plot_stability_metrics_missing_data(plotter, figure):
    del plotter.recorder.data["cop_position"]
    plotter.plot_stability_metrics(figure)
    assert len(figure.axes) > 0


def test_plot_joint_velocities(plotter, figure):
    plotter.plot_joint_velocities(figure)
    assert len(figure.axes) > 0


def test_plot_joint_torques(plotter, figure):
    plotter.plot_joint_torques(figure)
    assert len(figure.axes) > 0


def test_plot_actuator_powers(plotter, figure):
    plotter.plot_actuator_powers(figure)
    assert len(figure.axes) > 0


def test_plot_energy_analysis(plotter, figure):
    plotter.plot_energy_analysis(figure)
    assert len(figure.axes) > 0


def test_plot_club_head_speed(plotter, figure):
    plotter.plot_club_head_speed(figure)
    assert len(figure.axes) > 0


def test_plot_club_head_trajectory(plotter, figure):
    plotter.plot_club_head_trajectory(figure)
    assert len(figure.axes) > 0


def test_plot_phase_diagram(plotter, figure):
    plotter.plot_phase_diagram(figure, 0)
    assert len(figure.axes) > 0


def test_plot_torque_comparison(plotter, figure):
    plotter.plot_torque_comparison(figure)
    assert len(figure.axes) > 0


@patch("shared.python.signal_processing.compute_psd")
def test_plot_frequency_analysis(mock_psd, plotter, figure):
    mock_psd.return_value = (np.arange(10), np.random.rand(10))
    plotter.plot_frequency_analysis(figure)
    assert len(figure.axes) > 0


@patch("shared.python.signal_processing.compute_spectrogram")
def test_plot_spectrogram(mock_spec, plotter, figure):
    mock_spec.return_value = (np.arange(10), np.arange(10), np.random.rand(10, 10))
    plotter.plot_spectrogram(figure)
    assert len(figure.axes) > 0


def test_plot_summary_dashboard(plotter, figure):
    plotter.plot_summary_dashboard(figure)
    assert len(figure.axes) == 6  # 2x3 grid


def test_plot_kinematic_sequence(plotter, figure):
    segment_indices = {"Pelvis": 0, "Torso": 1, "Arm": 2}
    plotter.plot_kinematic_sequence(figure, segment_indices)
    assert len(figure.axes) > 0


def test_plot_work_loop(plotter, figure):
    plotter.plot_work_loop(figure, 0)
    assert len(figure.axes) > 0


def test_plot_x_factor_cycle(plotter, figure):
    # Mocking compute logic inside plotter indirectly via data
    plotter.recorder.data["joint_positions"] = (
        np.linspace(0, 1, 10),
        np.random.rand(10, 3),
    )
    plotter.plot_x_factor_cycle(figure, 0, 1)
    assert len(figure.axes) > 0


def test_plot_3d_phase_space(plotter, figure):
    plotter.plot_3d_phase_space(figure, 0)
    assert len(figure.axes) > 0


def test_plot_correlation_matrix(plotter, figure):
    plotter.plot_correlation_matrix(figure)
    assert len(figure.axes) > 0


@patch("shared.python.swing_plane_analysis.SwingPlaneAnalyzer")
def test_plot_swing_plane(mock_analyzer_class, plotter, figure):
    mock_analyzer = mock_analyzer_class.return_value
    mock_metrics = MagicMock()
    mock_metrics.steepness_deg = 45.0
    mock_metrics.rmse = 0.1
    mock_metrics.point_on_plane = np.array([0, 0, 0])
    mock_metrics.normal_vector = np.array([0, 0, 1])
    mock_analyzer.analyze.return_value = mock_metrics
    mock_analyzer.calculate_deviation.return_value = np.zeros(10)

    plotter.plot_swing_plane(figure)
    assert len(figure.axes) > 0


def test_plot_angular_momentum(plotter, figure):
    plotter.plot_angular_momentum(figure)
    assert len(figure.axes) > 0


def test_plot_cop_trajectory(plotter, figure):
    plotter.plot_cop_trajectory(figure)
    assert len(figure.axes) > 0


def test_plot_cop_vector_field(plotter, figure):
    plotter.plot_cop_vector_field(figure)
    assert len(figure.axes) > 0


def test_plot_radar_chart(plotter, figure):
    metrics = {"A": 0.5, "B": 0.8, "C": 0.2}
    plotter.plot_radar_chart(figure, metrics)
    assert len(figure.axes) > 0


def test_plot_power_flow(plotter, figure):
    plotter.plot_power_flow(figure)
    assert len(figure.axes) > 0


def test_plot_induced_acceleration(plotter, figure):
    plotter.plot_induced_acceleration(figure, "gravity")
    assert len(figure.axes) > 0


def test_plot_induced_acceleration_breakdown(plotter, figure):
    plotter.plot_induced_acceleration(figure, "breakdown", breakdown_mode=True)
    assert len(figure.axes) > 0


def test_plot_counterfactual_comparison(plotter, figure):
    plotter.plot_counterfactual_comparison(figure, "ztcf_accel")
    assert len(figure.axes) > 0


def test_plot_counterfactual_dual(plotter, figure):
    plotter.plot_counterfactual_comparison(figure, "dual", 0)
    assert len(figure.axes) > 0
