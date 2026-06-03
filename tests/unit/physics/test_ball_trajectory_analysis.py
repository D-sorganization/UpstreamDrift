import math

import numpy as np
import pytest

from src.shared.python.physics.ball_launch_conditions import TrajectoryPoint
from src.shared.python.physics.ball_trajectory_analysis import TrajectoryAnalysisMixin


class _Analyzer(TrajectoryAnalysisMixin):
    pass


def _point(velocity: tuple[float, float, float]) -> TrajectoryPoint:
    return TrajectoryPoint(
        time=0.0,
        position=np.zeros(3),
        velocity=np.array(velocity, dtype=float),
        acceleration=np.zeros(3),
        forces={},
    )


def test_landing_angle_uses_horizontal_velocity_components() -> None:
    trajectory = [_point((0.0, 0.0, 0.0)), _point((3.0, 4.0, -5.0))]

    assert _Analyzer()._calculate_landing_angle(trajectory) == pytest.approx(45.0)


def test_landing_angle_matches_linalg_norm_reference() -> None:
    trajectory = [_point((0.0, 0.0, 0.0)), _point((5.0, 12.0, -13.0))]
    expected = math.degrees(math.atan2(13.0, np.linalg.norm([5.0, 12.0])))

    assert _Analyzer()._calculate_landing_angle(trajectory) == pytest.approx(expected)
