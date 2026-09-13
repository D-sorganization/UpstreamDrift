"""Constrained acceleration allocation must report unreachable directions."""

import numpy as np
import pytest
from src.shared.python.motion_matching.acceleration_effort import (
    allocate_acceleration_effort,
)

pytestmark = pytest.mark.unit


def test_rank_deficient_response_reports_unreachable_acceleration() -> None:
    result = allocate_acceleration_effort(
        np.array([[2.0, 0.0], [0.0, 0.0]]),
        np.array([1.0, 4.0]),
        np.array([5.0, 7.0]),
        acceleration_scales=np.ones(2),
        effort_scales=np.ones(2),
    )
    np.testing.assert_allclose(result.effort, [2.0, 0.0])
    np.testing.assert_allclose(result.achieved_acceleration, [5.0, 4.0])
    np.testing.assert_allclose(result.acceleration_error, [0.0, -3.0])
    assert result.response_rank == 1


def test_redundant_efforts_minimize_scaled_effort_norm() -> None:
    result = allocate_acceleration_effort(
        np.array([[1.0, 1.0]]),
        np.zeros(1),
        np.array([5.0]),
        acceleration_scales=np.ones(1),
        effort_scales=np.array([1.0, 2.0]),
    )
    np.testing.assert_allclose(result.effort, [1.0, 4.0])
    assert not result.effort.flags.writeable


@pytest.mark.parametrize("bad_scale", [0.0, -1.0, float("nan")])
def test_rejects_invalid_scales(bad_scale: float) -> None:
    with pytest.raises(ValueError):
        allocate_acceleration_effort(
            np.eye(1),
            np.zeros(1),
            np.ones(1),
            acceleration_scales=np.array([bad_scale]),
            effort_scales=np.ones(1),
        )
