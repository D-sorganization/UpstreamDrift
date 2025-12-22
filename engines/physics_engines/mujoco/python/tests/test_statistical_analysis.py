import importlib.util
import os
import sys

import numpy as np

# Import StatisticalAnalyzer directly from file to avoid mujoco dependency
# Assuming this test file is in engines/physics_engines/mujoco/python/tests/
# and the source is in:
# engines/physics_engines/mujoco/python/mujoco_humanoid_golf/statistical_analysis.py
file_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__), "../mujoco_humanoid_golf/statistical_analysis.py"
    )
)
spec = importlib.util.spec_from_file_location("statistical_analysis", file_path)
assert spec is not None
assert spec.loader is not None
statistical_analysis = importlib.util.module_from_spec(spec)
sys.modules["statistical_analysis"] = statistical_analysis
spec.loader.exec_module(statistical_analysis)

StatisticalAnalyzer = statistical_analysis.StatisticalAnalyzer
SummaryStatistics = statistical_analysis.SummaryStatistics


def test_compute_summary_stats():
    data = np.array([0, 1, 2, 3, 4, 5, 100], dtype=float)
    times = np.arange(len(data), dtype=float)
    # Mock analyzer with just times
    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=np.zeros((len(data), 1)),
        joint_velocities=np.zeros((len(data), 1)),
        joint_torques=np.zeros((len(data), 1)),
    )

    stats = analyzer.compute_summary_stats(data)

    assert stats.min == 0.0
    assert stats.max == 100.0
    assert stats.range == 100.0
    assert stats.min_time == 0.0
    assert stats.max_time == 6.0
    assert stats.mean == np.mean(data)
    assert stats.median == np.median(data)
    assert stats.std == np.std(data)


def test_generate_comprehensive_report_consistency():
    N = 100
    nq = 2
    times = np.linspace(0, 1.0, N)
    joint_positions = np.zeros((N, nq))
    # Joint 0: 0 to pi radians -> 0 to 180 degrees
    joint_positions[:, 0] = np.linspace(0, np.pi, N)

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=joint_positions,
        joint_velocities=np.zeros((N, nq)),
        joint_torques=np.zeros((N, nq)),
    )

    report = analyzer.generate_comprehensive_report()

    j0 = report["joints"]["joint_0"]
    rom = j0["range_of_motion"]
    pos = j0["position_stats"]

    # Check consistency between ROM and position stats
    assert np.isclose(rom["min_deg"], pos["min"])
    assert np.isclose(rom["max_deg"], pos["max"])
    assert np.isclose(rom["rom_deg"], pos["range"])

    # Check values
    assert np.isclose(pos["min"], 0.0, atol=1e-5)
    assert np.isclose(pos["max"], 180.0, atol=1e-5)


def test_frequency_analysis_and_smoothness():
    N = 1000
    fs = 1000.0
    times = np.arange(N) / fs
    # 50 Hz signal
    signal_50hz = np.sin(2 * np.pi * 50 * times)

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=np.zeros((N, 1)),
        joint_velocities=np.zeros((N, 1)),
        joint_torques=np.zeros((N, 1)),
    )

    # Test frequency analysis
    freqs, psd = analyzer.compute_frequency_analysis(signal_50hz)
    assert len(freqs) > 0
    assert len(psd) > 0

    peak_idx = np.argmax(psd)
    assert abs(freqs[peak_idx] - 50.0) < 5.0

    # Test smoothness (SAL)
    # Use Gaussian bell curve (smooth movement) which is typical for SAL testing
    t_center = 0.5
    width = 0.05
    smooth_signal = np.exp(-((times - t_center) ** 2) / (2 * width**2))

    # Smooth signal
    smoothness = analyzer.compute_smoothness_metric(smooth_signal)
    assert smoothness < 0  # SAL is always negative

    # Add in-band noise (10Hz < 20Hz cutoff)
    noisy_signal = smooth_signal + 0.1 * np.sin(2 * np.pi * 10 * times)
    smoothness_noisy = analyzer.compute_smoothness_metric(noisy_signal)

    # Smoother signal should have higher SAL (closer to 0)
    assert smoothness > smoothness_noisy
