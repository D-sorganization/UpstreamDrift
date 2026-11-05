"""SHOW-C3c (#14559, D44): which sample of the reference counts as impact.

The bundle records no ball impact, so the harness picks the sample itself: the
first sample where the clubhead centre crosses the ball plane (ball position
and target-normal axis), else the minimum clubhead height in the search range.
The sample is read as a point, never as a max over speed.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.impact_sample import (
    TRANSIENT_NOTE,
    impact_sample,
)

pytestmark = pytest.mark.unit

N = np.array([1.0, 0.0, 0.0])  # target-normal axis


def _head(xs, zs=None) -> np.ndarray:
    xs = np.asarray(xs, dtype=float)
    zs = np.zeros_like(xs) if zs is None else np.asarray(zs, dtype=float)
    return np.stack([xs, np.zeros_like(xs), zs], axis=1)


def test_first_sample_across_the_ball_plane_is_the_impact() -> None:
    head = _head([2.0, 1.0, 0.5, -0.5, -1.0])  # crosses x = 0 between 2 and 3
    idx = impact_sample(head, ball_pos=np.zeros(3), target_normal=N)
    assert idx == 3  # first sample on the far side, not the closest one (2)


def test_a_later_recrossing_is_ignored() -> None:
    head = _head([1.0, -1.0, 1.0, -1.0])
    assert impact_sample(head, ball_pos=np.zeros(3), target_normal=N) == 1


def test_works_for_a_head_approaching_from_the_negative_side() -> None:
    head = _head([-2.0, -1.0, 0.5, 1.0])
    assert impact_sample(head, ball_pos=np.zeros(3), target_normal=N) == 2


def test_the_ball_plane_uses_the_ball_position_along_the_normal() -> None:
    head = _head([5.0, 4.0, 3.0, 2.0, 1.0])
    ball = np.array([2.5, 7.0, -3.0])  # only its component along N matters
    assert impact_sample(head, ball_pos=ball, target_normal=N) == 3


def test_a_trajectory_that_never_reaches_the_ball_is_refused() -> None:
    head = _head([3.0, 2.0, 1.0])
    with pytest.raises(ContractViolationError, match="ball plane"):
        impact_sample(head, ball_pos=np.zeros(3), target_normal=N)


def test_without_a_ball_the_lowest_clubhead_sample_is_used() -> None:
    z = [0.5, 0.2, -0.1, -0.3, -0.2, 0.4]
    head = _head(np.arange(6.0), z)
    assert impact_sample(head) == 3


def test_ties_take_the_first_lowest_sample() -> None:
    head = _head(np.arange(4.0), [0.3, -0.2, -0.2, 0.1])
    assert impact_sample(head) == 1


def test_a_speed_peak_does_not_move_the_sample() -> None:
    # Same geometry, very different spacing (so very different speed): same index.
    slow = _head([2.0, 1.0, 0.5, -0.5, -1.0])
    fast = _head([20.0, 10.0, 5.0, -5.0, -10.0])
    ball = np.zeros(3)
    assert impact_sample(slow, ball, N) == impact_sample(fast, ball, N)


@pytest.mark.parametrize("bad", [np.zeros((1, 3)), np.zeros((4, 2)), np.zeros(5)])
def test_malformed_positions_are_refused(bad) -> None:
    with pytest.raises(ContractViolationError):
        impact_sample(bad)


def test_the_transient_disclosure_names_the_steps() -> None:
    assert "1324" in TRANSIENT_NOTE and "1330" in TRANSIENT_NOTE
