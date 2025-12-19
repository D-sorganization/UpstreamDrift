import importlib.util
import sys
from pathlib import Path

import numpy as np


# Helper to import module directly from file path, bypassing package __init__
def import_from_source(file_path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# Import the module under test
source_path = Path(
    "engines/physics_engines/mujoco/python/mujoco_golf_pendulum/"
    "statistical_analysis.py"
)
sa_module = import_from_source(
    source_path, "mujoco_golf_pendulum.statistical_analysis"
)

StatisticalAnalyzer = sa_module.StatisticalAnalyzer
SummaryStatistics = sa_module.SummaryStatistics


def test_statistical_analyzer():
    N = 100
    times = np.linspace(0, 1, N)
    joint_positions = np.zeros((N, 2))
    joint_positions[:, 0] = np.linspace(0, np.deg2rad(90), N)  # 0 to 90 degrees
    joint_velocities = np.ones((N, 2))
    joint_torques = np.zeros((N, 2))
    club_head_speed = np.zeros(N)

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        club_head_speed=club_head_speed,
    )

    report = analyzer.generate_comprehensive_report()

    # Check joint 0 stats
    joint0_stats = report["joints"]["joint_0"]
    rom = joint0_stats["range_of_motion"]
    pos_stats = joint0_stats["position_stats"]

    assert np.isclose(rom["min_deg"], 0.0)
    assert np.isclose(rom["max_deg"], 90.0)
    assert np.isclose(rom["rom_deg"], 90.0)

    assert np.isclose(pos_stats["min"], 0.0)
    assert np.isclose(pos_stats["max"], 90.0)
    assert np.isclose(pos_stats["range"], 90.0)


def test_compute_summary_stats():
    N = 100
    times = np.linspace(0, 1, N)
    joint_positions = np.zeros((N, 1))
    joint_velocities = np.zeros((N, 1))
    joint_torques = np.zeros((N, 1))

    analyzer = StatisticalAnalyzer(
        times=times,
        joint_positions=joint_positions,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
    )

    data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = analyzer.compute_summary_stats(data)

    assert stats.min == 1.0
    assert stats.max == 5.0
    assert stats.range == 4.0
    assert stats.mean == 3.0
    assert stats.median == 3.0
    assert np.isclose(stats.std, np.std(data))
