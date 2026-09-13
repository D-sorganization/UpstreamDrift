"""Numerical gimbal margins retain the original angular branch."""

import numpy as np
import pytest
from src.shared.python.motion_matching.gimbal_branch import gimbal_branch_interval

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("angle", [0.0, -1.929, 7.0, -8.0])
def test_interval_contains_initial_angle_and_excludes_singular_endpoints(
    angle: float,
) -> None:
    lo, hi = gimbal_branch_interval(angle, 0.05)
    assert lo <= angle <= hi
    assert hi - lo == pytest.approx(np.pi - 0.1)
    assert abs(np.cos(lo)) == pytest.approx(np.sin(0.05))
    assert abs(np.cos(hi)) == pytest.approx(np.sin(0.05))


def test_preserves_negative_shoulder_branch() -> None:
    lo, hi = gimbal_branch_interval(-1.929, 0.05)
    assert hi == pytest.approx(-np.pi / 2 - 0.05)
    assert lo == pytest.approx(-3 * np.pi / 2 + 0.05)


@pytest.mark.parametrize(
    "angle,margin", [(np.pi / 2, 0.05), (0.0, 0.0), (0.0, 2.0), (float("nan"), 0.05)]
)
def test_rejects_invalid_margin_or_initial_singularity(
    angle: float, margin: float
) -> None:
    with pytest.raises(ValueError):
        gimbal_branch_interval(angle, margin)
