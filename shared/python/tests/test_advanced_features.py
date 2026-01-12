"""Tests for advanced features in statistical analysis and plotting."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.comparative_analysis import ComparativeSwingAnalyzer
from shared.python.comparative_plotting import ComparativePlotter
from shared.python.plotting import GolfSwingPlotter, RecorderInterface
from shared.python.signal_processing import compute_jerk, compute_time_shift
from shared.python.statistical_analysis import StatisticalAnalyzer


class MockRecorder:
    def __init__(self, data_dict):
        self.data = data_dict
        # Compatibility with new tests which might pass more args?
        # New tests MockRecorder takes (times, positions, velocities, accelerations, torques)
        # But this one takes dict.
        # I need to unify or rename the class in appended code.
        # I'll rename the new MockRecorder to MockRecorderNew in appended code.

    def get_time_series(self, field_name):
        return self.data.get(field_name, ([], []))

    def get_induced_acceleration_series(self, source_name: str):
        return np.array([]), np.array([])

    def get_counterfactual_series(self, cf_name: str):
        return np.array([]), np.array([])


@pytest.fixture
def sample_data():
    t = np.linspace(0, 1, 100)
    # Simple sine wave
    pos = np.column_stack([np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
    # Derivative roughly
    vel = np.column_stack(
        [2 * np.pi * np.cos(2 * np.pi * t), -2 * np.pi * np.sin(2 * np.pi * t)]
    )
    # Torque proportional to position (spring)
    torque = -10.0 * pos
    return t, pos, vel, torque


@pytest.fixture
def analyzer(sample_data):
    t, pos, vel, torque = sample_data
    return StatisticalAnalyzer(
        times=t,
        joint_positions=pos,
        joint_velocities=vel,
        joint_torques=torque,
    )


def test_joint_stiffness_metrics(analyzer):
    """Test computation of joint stiffness."""
    metrics = analyzer.compute_joint_stiffness(0)
    assert metrics is not None
    # We generated torque = -10 * pos
    # So stiffness should be approx -10
    assert np.isclose(metrics.stiffness, -10.0, atol=0.1)
    assert np.isclose(metrics.r_squared, 1.0, atol=0.01)
    # Area should be small (linear relationship, no hysteresis)
    assert metrics.hysteresis_area < 1.0

    # Test invalid index
    assert analyzer.compute_joint_stiffness(99) is None


def test_dynamic_stiffness(analyzer):
    """Test computation of rolling stiffness."""
    t, k, r2 = analyzer.compute_dynamic_stiffness(0, window_size=20)
    assert len(t) > 0
    assert len(t) == len(k)
    assert len(k) == len(r2)
    # Should be close to -10 everywhere
    assert np.allclose(k, -10.0, atol=1.0)
    assert np.all(r2 > 0.9)


def test_fractal_dimension(analyzer):
    """Test Higuchi Fractal Dimension."""
    # Sine wave is smooth, FD should be close to 1
    # Re-generate clean sine
    t = np.linspace(0, 1, 200)
    data = np.sin(2 * np.pi * t)

    # We call the method on analyzer instance, but need to pass data
    fd = analyzer.compute_fractal_dimension(data, k_max=10)
    assert 1.0 <= fd <= 1.2  # Smooth curve

    # Random noise should be higher
    noise = np.random.normal(0, 1, 200)
    fd_noise = analyzer.compute_fractal_dimension(noise, k_max=10)
    assert fd_noise > 1.5


def test_sample_entropy(analyzer):
    """Test Sample Entropy."""
    # Sine wave is regular -> low entropy
    t = np.linspace(0, 1, 200)
    data = np.sin(2 * np.pi * t)

    samp_en = analyzer.compute_sample_entropy(data, m=2, r=0.2)
    assert samp_en < 0.5  # Regular

    # Random noise -> high entropy
    noise = np.random.normal(0, 1, 200)
    samp_en_noise = analyzer.compute_sample_entropy(noise, m=2, r=0.2)
    assert samp_en_noise > 1.0

def test_permutation_entropy(analyzer):
    """Test Permutation Entropy."""
    # Sine wave is regular -> low entropy
    t = np.linspace(0, 1, 200)
    data = np.sin(2 * np.pi * t)

    # Order 3, Delay 1
    # For a sine wave, the order of 3 points is quite deterministic
    pe = analyzer.compute_permutation_entropy(data, order=3, delay=1)

    # Maximum entropy for order 3 is log2(3!) = log2(6) = 2.58
    # Regular signal should be much lower
    assert pe < 2.0

    # Random noise -> high entropy (close to max)
    noise = np.random.rand(200)
    pe_noise = analyzer.compute_permutation_entropy(noise, order=3, delay=1)
    assert pe_noise > 2.0

def test_plot_joint_stiffness(sample_data):
    """Test plotting of joint stiffness."""
    t, pos, vel, torque = sample_data
    recorder = MockRecorder(
        {
            "joint_positions": (t, pos),
            "joint_torques": (t, torque),
            "joint_velocities": (t, vel),
        }
    )
    plotter = GolfSwingPlotter(recorder)
    fig = Figure()

    # Run plot method
    plotter.plot_joint_stiffness(fig, joint_idx=0)
    assert len(fig.axes) > 0

    # Check if regression line exists
    ax = fig.axes[0]
    lines = ax.get_lines()
    assert (
        len(lines) >= 2
    )  # scatter points (collection) + regression line + maybe trajectory line


def test_plot_dynamic_stiffness(sample_data):
    """Test plotting of dynamic stiffness."""
    t, pos, vel, torque = sample_data
    recorder = MockRecorder(
        {
            "joint_positions": (t, pos),
            "joint_torques": (t, torque),
            "joint_velocities": (t, vel),
        }
    )
    plotter = GolfSwingPlotter(recorder)
    fig = Figure()

    plotter.plot_dynamic_stiffness(fig, joint_idx=0)
    assert len(fig.axes) > 0
    # Should have twin axis
    assert len(fig.axes) == 2


def test_plot_bland_altman(sample_data):
    """Test Bland-Altman plot."""
    t, pos, _, _ = sample_data
    # Create two slightly different signals
    sig_a = pos[:, 0]
    sig_b = sig_a + np.random.normal(0, 0.1, len(t))

    rec_a = MockRecorder({"test_field": (t, sig_a)})
    rec_b = MockRecorder({"test_field": (t, sig_b)})

    analyzer = ComparativeSwingAnalyzer(rec_a, rec_b)
    plotter = ComparativePlotter(analyzer)

    fig = Figure()
    plotter.plot_bland_altman(fig, "test_field")
    assert len(fig.axes) > 0

    ax = fig.axes[0]
    # Check for mean and LoA lines
    lines = ax.get_lines()
    # 3 horizontal lines (mean, upper, lower) + scatter is a collection
    assert len(lines) >= 3


# --- NEW ADVANCED FEATURE TESTS ---

class MockRecorderNew:
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
        )
        # Mock acceleration if analyzer doesn't compute it from velocity
        analyzer.joint_accelerations = joint_acc

        metrics = analyzer.compute_jerk_metrics(0)
        assert metrics is not None
        assert metrics.peak_jerk == pytest.approx(6.0, rel=0.1)
        assert metrics.rms_jerk == pytest.approx(6.0, rel=0.1)

    def test_compute_lag_matrix(self):
        """Test lag matrix computation."""
        t = np.linspace(0, 2, 200)
        fs = 100.0
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

        recorder = MockRecorderNew(t, pos, vels, acc, np.zeros_like(vels))
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
