"""Sampled actuator feasibility contracts for epic #15493."""

import numpy as np
import pytest

from src.shared.python.optimization.applied_effort_audit import audit_applied_effort

pytestmark = pytest.mark.unit


def audit(times=None, torques=None, lower=None, upper=None, slew=None):
    return audit_applied_effort(
        times_s=[0.0, 0.5, 2.0] if times is None else times,
        applied_torque_nm=[[1.0, -2.0], [3.0, -1.0], [0.0, 2.0]]
        if torques is None
        else torques,
        torque_lower_nm=[-4.0, -3.0] if lower is None else lower,
        torque_upper_nm=[4.0, 3.0] if upper is None else upper,
        slew_limit_nm_s=[5.0, 3.0] if slew is None else slew,
    )


def test_nonuniform_times_and_per_joint_signed_margins():
    result = audit()
    np.testing.assert_allclose(result.torque_margin_nm, [1.0, 1.0])
    np.testing.assert_allclose(result.slew_margin_nm_s, [1.0, 1.0])
    assert result.feasible


def test_feedback_cannot_hide_behind_zero_feedforward_residual():
    # Residual is zero; total applied torque includes 6 Nm of PD feedback.
    result = audit(torques=[[0.0, 0.0], [6.0, 0.0], [6.0, 0.0]])
    np.testing.assert_allclose(result.torque_margin_nm, [-2.0, 3.0])
    np.testing.assert_allclose(result.slew_margin_nm_s, [-7.0, 3.0])
    assert not result.feasible


def test_lower_bound_violation_is_not_treated_as_upper_bound():
    result = audit(torques=[[-5.0, 0.0]] * 3, slew=[0.0, 0.0])
    np.testing.assert_allclose(result.torque_margin_nm, [-1.0, 3.0])
    assert not result.feasible


def test_slew_alone_can_make_bounded_torque_infeasible():
    result = audit(slew=[3.0, 3.0])
    assert all(margin > 0.0 for margin in result.torque_margin_nm)
    assert result.slew_margin_nm_s[0] == -1.0
    assert not result.feasible


def test_exact_limits_and_fixed_actuator_are_feasible():
    result = audit(
        times=[0.0, 1.0],
        torques=[[0.0, -2.0], [0.0, 2.0]],
        lower=[0.0, -2.0],
        upper=[0.0, 2.0],
        slew=[0.0, 4.0],
    )
    assert result.feasible
    np.testing.assert_array_equal(result.torque_margin_nm, [0.0, 0.0])
    np.testing.assert_array_equal(result.slew_margin_nm_s, [0.0, 0.0])


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"times": [0.0]}, "at least two"),
        ({"times": [[0.0, 1.0, 2.0]]}, "one-dimensional"),
        ({"times": [0.0, 0.0, 1.0]}, "strictly increasing"),
        ({"times": [0.0, 2.0, 1.0]}, "strictly increasing"),
        ({"times": [0.0, np.nan, 2.0]}, "finite"),
        ({"torques": [1.0, 2.0, 3.0]}, "shape"),
        ({"torques": [[1.0, 2.0]]}, "shape"),
        ({"torques": np.empty((3, 0))}, "shape"),
        ({"torques": [[np.inf, 0.0]] * 3}, "finite"),
        ({"lower": [-1.0]}, "shape"),
        ({"upper": [[4.0, 3.0]]}, "shape"),
        ({"slew": [-1.0, 3.0]}, "nonnegative"),
        ({"slew": [np.inf, 3.0]}, "finite"),
        ({"upper": [-5.0, 3.0]}, "lower"),
        ({"torques": [[1j, 0.0]] * 3}, "real"),
        ({"times": ["0", "1", "2"]}, "real"),
    ],
)
def test_invalid_histories_fail_explicitly(kwargs, message):
    with pytest.raises(ValueError, match=message):
        audit(**kwargs)


def test_overflow_cannot_produce_a_feasible_receipt():
    with pytest.raises(ValueError, match="finite"):
        audit(times=[0.0, 1e-300, 1.0], torques=[[0.0, 0.0], [1e300, 0.0], [0.0, 0.0]])
