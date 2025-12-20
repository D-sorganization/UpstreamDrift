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
