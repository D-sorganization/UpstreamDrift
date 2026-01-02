"""
Unit tests for shared.python.plotting module.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from matplotlib.figure import Figure

# Mock modules that might not be available in all environments
# We need to ensure we can import plotting even if matplotlib backends are missing
# The plotting module itself handles importing MplCanvas with a try-except,
# so we just need to ensure matplotlib.figure.Figure is available or mocked.
# But for unit testing, we want to mock plotting calls anyway.
from shared.python.constants import GRAVITY_M_S2
from shared.python.plotting import GolfSwingPlotter, RecorderInterface


@pytest.fixture
def mock_recorder():
    """Create a mock recorder providing time series data."""
    recorder = MagicMock(spec=RecorderInterface)

    # Setup standard data
    times = np.linspace(0, 1, 100)

    # 2 DOFs for simplicity
    positions = np.stack([np.sin(times), np.cos(times)], axis=1)
    velocities = np.stack([np.cos(times), -np.sin(times)], axis=1)
    accelerations = np.stack([-np.sin(times), -np.cos(times)], axis=1)
    torques = np.stack([np.ones_like(times), -np.ones_like(times)], axis=1)
    powers = torques * velocities

    # Club head data (3D)
    club_pos = np.stack([np.sin(times), np.cos(times), times], axis=1)
    club_speed = np.sqrt(np.sum(velocities**2, axis=1))  # Just dummy scalar

    # Energy
    ke = 0.5 * club_speed**2
    pe = GRAVITY_M_S2 * club_pos[:, 2]
    te = ke + pe

    def get_data(field_name):
        if field_name == "joint_positions":
            return times, positions
        elif field_name == "joint_velocities":
            return times, velocities
        elif field_name == "joint_accelerations":
            return times, accelerations
        elif field_name == "joint_torques":
            return times, torques
        elif field_name == "actuator_powers":
            return times, powers
        elif field_name == "club_head_position":
            return times, club_pos
        elif field_name == "club_head_speed":
            return times, club_speed
        elif field_name == "kinetic_energy":
            return times, ke
        elif field_name == "potential_energy":
            return times, pe
        elif field_name == "total_energy":
            return times, te
        return times, []

    recorder.get_time_series.side_effect = get_data
    return recorder


@pytest.fixture
def plotter(mock_recorder):
    """Create a GolfSwingPlotter instance."""
    return GolfSwingPlotter(mock_recorder, joint_names=["Joint1", "Joint2"])


@pytest.fixture
def mock_figure():
    """Create a mock Matplotlib figure."""
    fig = MagicMock()
    # Mock add_subplot to return a mock axes
    ax = MagicMock()
    fig.add_subplot.return_value = ax

    # Mock projection="3d" behavior
    # add_subplot calls usually look like add_subplot(111, projection='3d')
    # We can just return the same ax mock, or specialize if needed.

    return fig


class TestGolfSwingPlotter:
    """Test suite for GolfSwingPlotter."""

    def test_initialization(self, plotter):
        """Test plotter initialization."""
        assert plotter.recorder is not None
        assert plotter.joint_names == ["Joint1", "Joint2"]
        assert "primary" in plotter.colors

    def test_get_joint_name(self, plotter):
        """Test joint name retrieval."""
        assert plotter.get_joint_name(0) == "Joint1"
        assert plotter.get_joint_name(1) == "Joint2"
        assert plotter.get_joint_name(99) == "Joint 99"

    def test_get_aligned_label(self, plotter):
        """Test aligned label logic."""
        # Exact match
        assert plotter._get_aligned_label(0, 2) == "Joint1"
        assert plotter._get_aligned_label(1, 2) == "Joint2"

        # Mismatch (floating base case: data dim > name dim)
        # e.g., 9 dim data, 2 names. Offset = 7.
        # idx 0-6 should be "DoF X"
        # idx 7 should be "Joint1" (0)
        assert plotter._get_aligned_label(0, 9) == "DoF 0"
        assert plotter._get_aligned_label(7, 9) == "Joint1"

    def test_plot_joint_angles(self, plotter, mock_figure):
        """Test plotting joint angles."""
        plotter.plot_joint_angles(mock_figure)
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        assert ax.plot.call_count == 2  # 2 joints
        ax.set_xlabel.assert_called_with("Time (s)", fontsize=12, fontweight="bold")

    def test_plot_joint_velocities(self, plotter, mock_figure):
        """Test plotting joint velocities."""
        plotter.plot_joint_velocities(mock_figure)
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        assert ax.plot.call_count == 2

    def test_plot_joint_torques(self, plotter, mock_figure):
        """Test plotting joint torques."""
        plotter.plot_joint_torques(mock_figure)
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        assert ax.plot.call_count == 2

    def test_plot_actuator_powers(self, plotter, mock_figure):
        """Test plotting actuator powers."""
        plotter.plot_actuator_powers(mock_figure)
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        assert ax.plot.call_count == 2

    def test_plot_energy_analysis(self, plotter, mock_figure):
        """Test plotting energy analysis."""
        plotter.plot_energy_analysis(mock_figure)
        mock_figure.add_subplot.assert_called()
        ax = mock_figure.add_subplot.return_value
        assert ax.plot.call_count == 3  # KE, PE, TE

    def test_plot_club_head_speed(self, plotter, mock_figure):
        """Test plotting club head speed."""
        plotter.plot_club_head_speed(mock_figure)
        ax = mock_figure.add_subplot.return_value
        ax.plot.assert_called()
        # Should check for mph conversion logic implicitly by success

    def test_plot_club_head_trajectory(self, plotter, mock_figure):
        """Test 3D trajectory plotting."""
        plotter.plot_club_head_trajectory(mock_figure)
        mock_figure.add_subplot.assert_called_with(111, projection="3d")
        ax = mock_figure.add_subplot.return_value
        ax.scatter.assert_called()

    def test_plot_phase_diagram(self, plotter, mock_figure):
        """Test phase diagram plotting."""
        plotter.plot_phase_diagram(mock_figure, joint_idx=0)
        ax = mock_figure.add_subplot.return_value
        ax.scatter.assert_called()

    def test_plot_torque_comparison(self, plotter, mock_figure):
        """Test torque comparison plotting (stackplot)."""
        plotter.plot_torque_comparison(mock_figure)
        ax = mock_figure.add_subplot.return_value
        assert ax.stackplot.call_count == 2  # Positive and negative

    def test_plot_frequency_analysis(self, plotter, mock_figure):
        """Test frequency analysis plotting."""
        # This calls scipy or shared signal processing
        with patch(
            "scipy.signal.welch", return_value=(np.arange(10), np.random.rand(10))
        ):
            plotter.plot_frequency_analysis(mock_figure, joint_idx=0)
            ax = mock_figure.add_subplot.return_value
            ax.semilogy.assert_called()

    def test_plot_spectrogram(self, plotter, mock_figure):
        """Test spectrogram plotting."""
        with patch(
            "scipy.signal.spectrogram",
            return_value=(np.arange(10), np.arange(10), np.random.rand(10, 10)),
        ):
            plotter.plot_spectrogram(mock_figure, joint_idx=0)
            ax = mock_figure.add_subplot.return_value
            ax.pcolormesh.assert_called()

    def test_plot_summary_dashboard(self, plotter, mock_figure):
        """Test dashboard plotting."""
        plotter.plot_summary_dashboard(mock_figure)
        assert mock_figure.add_gridspec.called
        # Should create 6 subplots (2x3 grid)
        assert mock_figure.add_subplot.call_count == 6

    def test_plot_kinematic_sequence(self, plotter, mock_figure):
        """Test kinematic sequence plotting."""
        segments = {"Pelvis": 0, "Torso": 1}
        plotter.plot_kinematic_sequence(mock_figure, segments)
        ax = mock_figure.add_subplot.return_value
        # 2 lines + 2 peak markers = 4 plot calls
        assert ax.plot.call_count == 4

    def test_plot_3d_phase_space(self, plotter, mock_figure):
        """Test 3D phase space plotting."""
        plotter.plot_3d_phase_space(mock_figure, joint_idx=0)
        mock_figure.add_subplot.assert_called_with(111, projection="3d")

    def test_plot_correlation_matrix(self, plotter, mock_figure):
        """Test correlation matrix plotting."""
        plotter.plot_correlation_matrix(mock_figure)
        ax = mock_figure.add_subplot.return_value
        ax.imshow.assert_called()

    def test_plot_swing_plane(self, plotter, mock_figure):
        """Test swing plane plotting."""
        # Need to mock SwingPlaneAnalyzer or ensure data is valid for it
        # The default mocked data is a spiral, which should fit something.

        # Mock SwingPlaneAnalyzer to avoid complex geometry calculations in unit test
        with patch("shared.python.plotting.SwingPlaneAnalyzer") as MockAnalyzer:
            instance = MockAnalyzer.return_value
            instance.analyze.return_value.steepness_deg = 45.0
            instance.analyze.return_value.rmse = 0.01
            instance.analyze.return_value.point_on_plane = np.zeros(3)
            instance.analyze.return_value.normal_vector = np.array([0, 0, 1])
            instance.calculate_deviation.return_value = np.zeros(100)

            plotter.plot_swing_plane(mock_figure)

            mock_figure.add_subplot.assert_called_with(111, projection="3d")
            ax = mock_figure.add_subplot.return_value
            ax.plot_surface.assert_called()

    def test_plot_angular_momentum(self, plotter, mock_figure):
        """Test angular momentum plotting."""
        # Setup mock recorder to return some AM data
        times = np.linspace(0, 1, 100)
        am = np.ones((100, 3))
        plotter.recorder.get_time_series = MagicMock(
            side_effect=lambda name: (
                (times, am) if name == "angular_momentum" else ([], [])
            )
        )

        plotter.plot_angular_momentum(mock_figure)
        ax = mock_figure.add_subplot.return_value
        # Should plot 3 components + magnitude
        assert ax.plot.call_count == 4

    def test_plot_cop_trajectory(self, plotter, mock_figure):
        """Test CoP trajectory plotting."""
        times = np.linspace(0, 1, 100)
        cop = np.random.rand(100, 2)
        plotter.recorder.get_time_series = MagicMock(
            side_effect=lambda name: (
                (times, cop) if name == "cop_position" else ([], [])
            )
        )

        plotter.plot_cop_trajectory(mock_figure)
        ax = mock_figure.add_subplot.return_value
        # Scatter + plot + markers
        assert ax.scatter.call_count >= 1
        assert ax.plot.call_count >= 1

    def test_plot_radar_chart(self, plotter, mock_figure):
        """Test radar chart plotting."""
        metrics = {"A": 0.5, "B": 0.8, "C": 0.2}
        plotter.plot_radar_chart(mock_figure, metrics)
        mock_figure.add_subplot.assert_called_with(111, polar=True)
        ax = mock_figure.add_subplot.return_value
        ax.plot.assert_called()
        ax.fill.assert_called()

    def test_plot_power_flow(self, plotter, mock_figure):
        """Test power flow plotting."""
        times = np.linspace(0, 1, 100)
        powers = np.random.rand(100, 3)
        plotter.recorder.get_time_series = MagicMock(
            side_effect=lambda name: (
                (times, powers) if name == "actuator_powers" else ([], [])
            )
        )

        plotter.plot_power_flow(mock_figure)
        ax = mock_figure.add_subplot.return_value
        # 2 stackplots (pos/neg)
        assert ax.stackplot.call_count == 2

    def test_plot_cop_vector_field(self, plotter, mock_figure):
        """Test CoP vector field plotting."""
        times = np.linspace(0, 1, 100)
        cop = np.random.rand(100, 2)
        plotter.recorder.get_time_series = MagicMock(
            side_effect=lambda name: (
                (times, cop) if name == "cop_position" else ([], [])
            )
        )

        plotter.plot_cop_vector_field(mock_figure)
        ax = mock_figure.add_subplot.return_value
        ax.quiver.assert_called()

    def test_empty_data_handling(self, mock_figure):
        """Test handling of empty data."""
        empty_recorder = MagicMock(spec=RecorderInterface)
        empty_recorder.get_time_series.return_value = ([], [])

        plotter = GolfSwingPlotter(empty_recorder)

        plotter.plot_joint_angles(mock_figure)
        ax = mock_figure.add_subplot.return_value
        ax.text.assert_called_with(
            0.5, 0.5, "No data recorded", ha="center", va="center"
        )



class MockRecorder(RecorderInterface):
    def __init__(self, data):
        self.data = data
        self.counter = 0

    def get_time_series(self, field_name):
        return self.data.get(field_name, (np.array([]), np.array([])))

    def get_induced_acceleration_series(self, source_name):
        return self.data.get(f"induced_{source_name}", (np.array([]), np.array([])))

    def get_counterfactual_series(self, cf_name):
        return self.data.get(f"cf_{cf_name}", (np.array([]), np.array([])))


@pytest.fixture
def sample_recorder():
    times = np.linspace(0, 1, 100)
    torques = np.random.randn(100, 2)
    velocities = np.random.randn(100, 2)

    data = {"joint_torques": (times, torques), "joint_velocities": (times, velocities)}
    return MockRecorder(data)


def test_plot_joint_power_curves(sample_recorder):
    fig = Figure()
    plotter = GolfSwingPlotter(sample_recorder, joint_names=["J0", "J1"])

    plotter.plot_joint_power_curves(fig)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    # Check title/labels
    assert "Joint Power" in ax.get_title()
    # 2 joints + 1 horizontal line = 3 lines
    assert len(ax.lines) == 3


def test_plot_impulse_accumulation(sample_recorder):
    fig = Figure()
    plotter = GolfSwingPlotter(sample_recorder, joint_names=["J0", "J1"])

    plotter.plot_impulse_accumulation(fig)
    assert len(fig.axes) == 1
    ax = fig.axes[0]
    assert "Impulse" in ax.get_title()
    # 2 joints + 1 horizontal line = 3 lines
    assert len(ax.lines) == 3


def test_plot_joint_power_curves_no_data():
    fig = Figure()
    plotter = GolfSwingPlotter(MockRecorder({}))
    plotter.plot_joint_power_curves(fig)
    ax = fig.axes[0]
    assert "No data" in ax.texts[0].get_text()


def test_plot_impulse_accumulation_no_data():
    fig = Figure()
    plotter = GolfSwingPlotter(MockRecorder({}))
    plotter.plot_impulse_accumulation(fig)
    ax = fig.axes[0]
    assert "No data" in ax.texts[0].get_text()
