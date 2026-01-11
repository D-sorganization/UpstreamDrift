"""Tests for advanced features in statistical analysis and plotting."""

import numpy as np
import pytest
from matplotlib.figure import Figure

from shared.python.comparative_analysis import ComparativeSwingAnalyzer
from shared.python.comparative_plotting import ComparativePlotter
from shared.python.plotting import GolfSwingPlotter
from shared.python.statistical_analysis import StatisticalAnalyzer


class MockRecorder:
    def __init__(self, data_dict):
        self.data = data_dict

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
