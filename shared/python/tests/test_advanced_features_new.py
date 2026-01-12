"""Tests for NEW advanced analysis features (Jerk, Lag, MSE)."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.plotting import GolfSwingPlotter, RecorderInterface
from shared.python.signal_processing import compute_jerk, compute_time_shift
from shared.python.statistical_analysis import StatisticalAnalyzer


class MockRecorder(RecorderInterface):
    def __init__(self, times, positions, velocities, accelerations, torques):
        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations
        self.torques = torques

    def get_time_series(self, field_name):
        if field_name == "joint_positions":
            return self.times, self.positions
        elif field_name == "joint_velocities":
            return self.times, self.velocities
        elif field_name == "joint_accelerations":
            return self.times, self.accelerations
        elif field_name == "joint_torques":
            return self.times, self.torques
        return [], []

    def get_induced_acceleration_series(self, source_name: str | int):
        return [], []

    def get_counterfactual_series(self, cf_name: str):
        return [], []


class TestAdvancedSignalProcessing:
    def test_compute_jerk(self):
        """Test jerk computation using cubic polynomial."""
        t = np.linspace(0, 1, 100)
        # Position x(t) = t^3
        # Velocity v(t) = 3t^2
        # Accel a(t) = 6t
        # Jerk j(t) = 6 (constant)

        accel = 6 * t
        fs = 100.0

        jerk = compute_jerk(accel, fs, window_len=7, polyorder=2)

        # Ignore edges where filtering is imperfect
        valid_jerk = jerk[10:-10]
        np.testing.assert_allclose(valid_jerk, 6.0, rtol=0.05)

    def test_compute_time_shift(self):
        """Test time shift detection."""
        fs = 100.0
        t = np.linspace(0, 2, 200)
        x = np.sin(2 * np.pi * 2 * t)  # 2 Hz sine
        # y lags x by 0.1s
        # y(t) = x(t - 0.1)
        # 0.1s = 10 samples
        y = np.sin(2 * np.pi * 2 * (t - 0.1))

        lag = compute_time_shift(x, y, fs)

        # Expected lag is 0.1
        assert lag == pytest.approx(0.1, abs=0.01)

        # y leads x by 0.1s
        y_lead = np.sin(2 * np.pi * 2 * (t + 0.1))
        lag_lead = compute_time_shift(x, y_lead, fs)
        assert lag_lead == pytest.approx(-0.1, abs=0.01)


class TestAdvancedStatisticalAnalysis:
    def test_compute_jerk_metrics(self):
        """Test jerk metrics computation."""
        t = np.linspace(0, 1, 100)
        accel = 6 * t
        vel = 3 * t**2
        pos = t**3

        joint_acc = accel.reshape(-1, 1)
        joint_vel = vel.reshape(-1, 1)
        joint_pos = pos.reshape(-1, 1)

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=joint_pos,
            joint_velocities=joint_vel,
            joint_torques=np.zeros_like(joint_pos),
            joint_accelerations=joint_acc,
        )

        metrics = analyzer.compute_jerk_metrics(0)
        assert metrics is not None
        assert metrics.peak_jerk == pytest.approx(6.0, rel=0.1)
        assert metrics.rms_jerk == pytest.approx(6.0, rel=0.1)

    def test_compute_lag_matrix(self):
        """Test lag matrix computation."""
        t = np.linspace(0, 2, 200)

        x = np.sin(2 * np.pi * 2 * t)
        y = np.sin(2 * np.pi * 2 * (t - 0.1))  # y lags x by 0.1
        z = np.sin(2 * np.pi * 2 * (t - 0.2))  # z lags x by 0.2

        velocities = np.column_stack([x, y, z])

        analyzer = StatisticalAnalyzer(
            times=t,
            joint_positions=np.zeros_like(velocities),
            joint_velocities=velocities,
            joint_torques=np.zeros_like(velocities),
        )

        matrix, labels = analyzer.compute_lag_matrix(data_type="velocity")

        assert matrix.shape == (3, 3)
        # matrix[0, 1] = lag(x, y) = 0.1
        assert matrix[0, 1] == pytest.approx(0.1, abs=0.01)
        # matrix[1, 0] = -0.1
        assert matrix[1, 0] == pytest.approx(-0.1, abs=0.01)
        # matrix[0, 2] = lag(x, z) = 0.2
        assert matrix[0, 2] == pytest.approx(0.2, abs=0.01)

    def test_compute_multiscale_entropy(self):
        """Test MSE computation."""
        # Random noise should have high entropy at scale 1, decay or stay high?
        # White noise: entropy decreases with scale.
        # 1/f noise: entropy constant.
        np.random.seed(42)
        noise = np.random.normal(0, 1, 1000)

        analyzer = StatisticalAnalyzer(
            times=np.arange(1000),
            joint_positions=np.zeros((1000, 1)),
            joint_velocities=np.zeros((1000, 1)),
            joint_torques=np.zeros((1000, 1)),
        )

        scales, mse = analyzer.compute_multiscale_entropy(noise, max_scale=5)

        assert len(scales) == 5
        assert len(mse) == 5
        assert mse[0] > 0
        # Check that entropy values are valid (non-negative)
        assert np.all(mse >= 0)


class TestAdvancedPlotting:
    def test_plots(self):
        """Test that new plots run without error."""
        t = np.linspace(0, 2, 200)
        x = np.sin(2 * np.pi * 2 * t)
        y = np.sin(2 * np.pi * 2 * (t - 0.1))
        vels = np.column_stack([x, y])
        pos = np.cumsum(vels, axis=0) * 0.01
        acc = np.diff(vels, axis=0, prepend=0)

        recorder = MockRecorder(t, pos, vels, acc, np.zeros_like(vels))
        plotter = GolfSwingPlotter(recorder, joint_names=["J1", "J2"])

        fig = Figure()
        plotter.plot_jerk_trajectory(fig)
        assert len(fig.axes) > 0
        fig.clear()

        plotter.plot_lag_matrix(fig)
        assert len(fig.axes) > 0
        fig.clear()

        plotter.plot_multiscale_entropy(fig, max_scale=3)
        assert len(fig.axes) > 0
