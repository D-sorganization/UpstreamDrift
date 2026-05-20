"""Tests for src.engines.simscape._output.SimscapeOutput."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.simscape._output import SimscapeOutput


def _valid_output(n: int = 4, n_joints: int = 3) -> SimscapeOutput:
    time = np.linspace(0.0, 0.3, n)
    q = np.zeros((n, n_joints))
    q_club = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (n, 1))
    return SimscapeOutput(
        time=time,
        q=q,
        qd=np.zeros((n, n_joints)),
        qdd=np.zeros((n, n_joints)),
        tau=np.zeros((n, n_joints)),
        omega=np.zeros((n, n_joints)),
        r_butt=np.zeros((n, 3)),
        r_clubhead=np.zeros((n, 3)),
        q_club=q_club,
        v_clubhead=np.zeros((n, 3)),
    )


def test_valid_construction_properties() -> None:
    out = _valid_output(n=5, n_joints=2)
    assert out.n_samples == 5
    assert out.n_joints == 2


def test_rejects_non_ndarray_field() -> None:
    with pytest.raises(TypeError, match="time"):
        SimscapeOutput(
            time=[0.0, 0.1],  # type: ignore[arg-type]
            q=np.zeros((2, 1)),
            qd=np.zeros((2, 1)),
            qdd=np.zeros((2, 1)),
            tau=np.zeros((2, 1)),
            omega=np.zeros((2, 1)),
            r_butt=np.zeros((2, 3)),
            r_clubhead=np.zeros((2, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (2, 1)),
            v_clubhead=np.zeros((2, 3)),
        )


def test_rejects_time_not_1d() -> None:
    with pytest.raises(ValueError, match="time must be 1-D"):
        SimscapeOutput(
            time=np.zeros((2, 1)),
            q=np.zeros((2, 1)),
            qd=np.zeros((2, 1)),
            qdd=np.zeros((2, 1)),
            tau=np.zeros((2, 1)),
            omega=np.zeros((2, 1)),
            r_butt=np.zeros((2, 3)),
            r_clubhead=np.zeros((2, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (2, 1)),
            v_clubhead=np.zeros((2, 3)),
        )


def test_rejects_empty_time() -> None:
    with pytest.raises(ValueError, match="at least one sample"):
        SimscapeOutput(
            time=np.zeros(0),
            q=np.zeros((0, 1)),
            qd=np.zeros((0, 1)),
            qdd=np.zeros((0, 1)),
            tau=np.zeros((0, 1)),
            omega=np.zeros((0, 1)),
            r_butt=np.zeros((0, 3)),
            r_clubhead=np.zeros((0, 3)),
            q_club=np.zeros((0, 4)),
            v_clubhead=np.zeros((0, 3)),
        )


def test_rejects_non_increasing_time() -> None:
    t = np.array([0.0, 0.1, 0.05, 0.2])
    with pytest.raises(ValueError, match="strictly increasing"):
        SimscapeOutput(
            time=t,
            q=np.zeros((4, 1)),
            qd=np.zeros((4, 1)),
            qdd=np.zeros((4, 1)),
            tau=np.zeros((4, 1)),
            omega=np.zeros((4, 1)),
            r_butt=np.zeros((4, 3)),
            r_clubhead=np.zeros((4, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (4, 1)),
            v_clubhead=np.zeros((4, 3)),
        )


def test_rejects_nonzero_start_time() -> None:
    with pytest.raises(ValueError, match=r"time\[0\]"):
        SimscapeOutput(
            time=np.linspace(0.1, 0.4, 4),
            q=np.zeros((4, 1)),
            qd=np.zeros((4, 1)),
            qdd=np.zeros((4, 1)),
            tau=np.zeros((4, 1)),
            omega=np.zeros((4, 1)),
            r_butt=np.zeros((4, 3)),
            r_clubhead=np.zeros((4, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (4, 1)),
            v_clubhead=np.zeros((4, 3)),
        )


def test_rejects_mismatched_joint_dim() -> None:
    with pytest.raises(ValueError, match="qd"):
        SimscapeOutput(
            time=np.linspace(0, 0.3, 4),
            q=np.zeros((4, 3)),
            qd=np.zeros((4, 2)),  # wrong width
            qdd=np.zeros((4, 3)),
            tau=np.zeros((4, 3)),
            omega=np.zeros((4, 3)),
            r_butt=np.zeros((4, 3)),
            r_clubhead=np.zeros((4, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (4, 1)),
            v_clubhead=np.zeros((4, 3)),
        )


def test_rejects_bad_three_vector_shape() -> None:
    with pytest.raises(ValueError, match="r_butt"):
        SimscapeOutput(
            time=np.linspace(0, 0.3, 4),
            q=np.zeros((4, 1)),
            qd=np.zeros((4, 1)),
            qdd=np.zeros((4, 1)),
            tau=np.zeros((4, 1)),
            omega=np.zeros((4, 1)),
            r_butt=np.zeros((4, 2)),  # wrong
            r_clubhead=np.zeros((4, 3)),
            q_club=np.tile([1.0, 0, 0, 0], (4, 1)),
            v_clubhead=np.zeros((4, 3)),
        )


def test_rejects_non_unit_quaternion() -> None:
    bad_q = np.tile([2.0, 0.0, 0.0, 0.0], (4, 1))
    with pytest.raises(ValueError, match="unit-norm"):
        SimscapeOutput(
            time=np.linspace(0, 0.3, 4),
            q=np.zeros((4, 1)),
            qd=np.zeros((4, 1)),
            qdd=np.zeros((4, 1)),
            tau=np.zeros((4, 1)),
            omega=np.zeros((4, 1)),
            r_butt=np.zeros((4, 3)),
            r_clubhead=np.zeros((4, 3)),
            q_club=bad_q,
            v_clubhead=np.zeros((4, 3)),
        )


def test_rejects_quaternion_wrong_shape() -> None:
    with pytest.raises(ValueError, match="q_club"):
        SimscapeOutput(
            time=np.linspace(0, 0.3, 4),
            q=np.zeros((4, 1)),
            qd=np.zeros((4, 1)),
            qdd=np.zeros((4, 1)),
            tau=np.zeros((4, 1)),
            omega=np.zeros((4, 1)),
            r_butt=np.zeros((4, 3)),
            r_clubhead=np.zeros((4, 3)),
            q_club=np.zeros((4, 3)),  # wrong width
            v_clubhead=np.zeros((4, 3)),
        )


def test_single_sample_ok() -> None:
    out = _valid_output(n=1, n_joints=2)
    assert out.n_samples == 1
