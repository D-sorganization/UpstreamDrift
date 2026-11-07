"""SHOW-C3c (#14559, D44): which sample of the reference counts as impact.

The bundle records no ball impact, so the harness picks the sample itself: the
first sample where the clubhead centre crosses the ball plane (ball position
and target-normal axis), else the minimum clubhead height in the search range.
The sample is read as a point, never as a max over speed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.impact_sample import (
    TRANSIENT_NOTE,
    impact_sample,
    recorded_impact_frame,
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


def test_without_a_ball_the_recorded_impact_event_is_used() -> None:
    head = _head(np.arange(6.0), [0.5, 0.2, -0.1, -0.3, -0.2, 0.4])
    assert impact_sample(head, event_index=4) == 4  # not the lowest sample (3)


def test_a_ball_position_wins_over_the_recorded_event() -> None:
    head = _head([2.0, 1.0, 0.5, -0.5, -1.0])
    assert impact_sample(head, np.zeros(3), N, event_index=1) == 3


def test_neither_a_ball_nor_an_event_is_refused_not_guessed() -> None:
    head = _head(np.arange(4.0), [0.3, -0.2, -0.2, 0.1])  # has a lowest sample
    with pytest.raises(ContractViolationError, match="no ball position"):
        impact_sample(head)


@pytest.mark.parametrize("bad", [-1, 4])
def test_an_event_outside_the_path_is_refused(bad: int) -> None:
    with pytest.raises(ContractViolationError, match="event"):
        impact_sample(_head(np.arange(4.0)), event_index=bad)


def test_the_recorded_impact_frame_is_read_from_the_receipt(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps({"event_frames": {"address": 0, "top": 1103, "impact": 1325}}),
        encoding="utf-8",
    )
    assert recorded_impact_frame(receipt) == 1325


def test_a_missing_receipt_or_event_gives_none(tmp_path) -> None:
    assert recorded_impact_frame(tmp_path / "absent.json") is None
    other = tmp_path / "receipt.json"
    other.write_text(json.dumps({"event_frames": {"top": 3}}), encoding="utf-8")
    assert recorded_impact_frame(other) is None


def test_a_malformed_recorded_frame_is_refused(tmp_path) -> None:
    receipt = tmp_path / "receipt.json"
    receipt.write_text(json.dumps({"event_frames": {"impact": -5}}), encoding="utf-8")
    with pytest.raises(ContractViolationError, match="impact"):
        recorded_impact_frame(receipt)


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
