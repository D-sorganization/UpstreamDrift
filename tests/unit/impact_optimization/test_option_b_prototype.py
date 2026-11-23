"""Fixed-gain control primitives; no physical feasibility is implied."""

import numpy as np
import pytest

from src.shared.python.impact_optimization.option_b_prototype import (
    FixedGainController,
    SplineResidual,
)

pytestmark = pytest.mark.unit


def residual(coefficients=None, **changes):
    values = {
        "breakpoints_s": np.array([0.0, 0.5, 1.0]),
        "coefficients_nm": np.zeros((5, 2)) if coefficients is None else coefficients,
        "scale_nm": np.array([1.0, 2.0]),
        "trust_radius": 5.0,
        "min_spacing_s": 0.1,
    }
    values.update(changes)
    return SplineResidual(**values)


def controller(**changes):
    values = {
        "times_s": np.array([0.0, 1.0]),
        "q_ref": np.array([[0.0, 1.0], [1.0, 3.0]]),
        "qd_ref": np.array([[1.0, 2.0], [1.0, 2.0]]),
        "tau_ff_ref": np.array([[4.0, -3.0], [6.0, -5.0]]),
        "kp": np.array([2.0, 3.0]),
        "kd": np.array([0.5, 1.0]),
        "lower_nm": np.array([-5.0, -6.0]),
        "upper_nm": np.array([6.0, 4.0]),
        "residual": residual(),
    }
    values.update(changes)
    return FixedGainController(**values)


def test_controller_saturates_total_including_feedback_not_only_residual():
    decision = controller().evaluate(0.5, np.array([-1.0, 2.0]), np.array([0.0, 0.0]))
    # 5 + 2*(.5 - -1) + .5*(1 - 0) = 8.5, above +6.
    np.testing.assert_allclose(decision.commanded_nm, [8.5, -2.0])
    np.testing.assert_allclose(decision.applied_nm, [6.0, -2.0])
    np.testing.assert_array_equal(decision.saturated, [True, False])


def test_cubic_spline_continuous_and_coefficient_trust_is_enforced():
    coefficients = np.array(
        [[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.6, 0.3], [0.8, 0.4]]
    )
    spline = residual(coefficients)
    assert np.linalg.norm(spline.value(0.5 - 1e-7) - spline.value(0.5 + 1e-7)) < 1e-6
    with pytest.raises(ValueError, match="trust"):
        residual(coefficients * 100)


@pytest.mark.parametrize(
    "knots", [[0.0, 0.05, 1.0], [0.0, 0.5, 0.5], [0.0, np.nan, 1.0]]
)
def test_breakpoints_require_finite_strict_minimum_spacing(knots):
    with pytest.raises(ValueError):
        residual(breakpoints_s=np.array(knots))


def test_no_extrapolation_or_unvalidated_shapes_and_nonfinite_values():
    with pytest.raises(ValueError):
        controller().evaluate(-0.01, np.zeros(2), np.zeros(2))
    with pytest.raises(ValueError):
        controller().evaluate(0.5, np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        controller().evaluate(0.5, np.array([np.nan, 0.0]), np.zeros(2))
    with pytest.raises(ValueError):
        controller(kp=np.array([2.0, -1.0]))
    with pytest.raises(ValueError):
        residual(coefficients=np.zeros((4, 2)))


def test_fixed_gains_and_inputs_are_copied_not_mutated():
    kp = np.array([2.0, 3.0])
    ctrl = controller(kp=kp)
    expected = ctrl.evaluate(0.5, np.zeros(2), np.zeros(2)).commanded_nm.copy()
    kp[:] = 1e6
    np.testing.assert_array_equal(
        ctrl.evaluate(0.5, np.zeros(2), np.zeros(2)).commanded_nm, expected
    )


def test_directional_saturation_records_full_command_and_negative_bound():
    result = controller().evaluate(0.0, np.array([0.0, 10.0]), np.zeros(2))
    # -3 + 3*(1-10) + 1*(2-0) = -28, clipped only after feedback.
    assert result.commanded_nm[1] == pytest.approx(-28.0)
    assert result.applied_nm[1] == pytest.approx(-6.0)
    assert result.saturated[1]


def test_nonzero_residual_is_added_before_total_saturation():
    coeff = np.ones((5, 2)) * 0.5
    result = controller(residual=residual(coeff)).evaluate(
        0.5, np.array([-1.0, 2.0]), np.zeros(2)
    )
    assert result.commanded_nm[0] == pytest.approx(9.0)
    assert result.applied_nm[0] == pytest.approx(6.0)


def test_spline_uses_closed_horizon_and_rejects_extrapolation():
    coeff = np.array([[0.0, 0.0], [0.2, 0.1], [0.4, 0.2], [0.6, 0.3], [0.8, 0.4]])
    spline = residual(coeff)
    np.testing.assert_allclose(spline.value(0.0), coeff[0])
    np.testing.assert_allclose(spline.value(1.0), coeff[-1])
    with pytest.raises(ValueError):
        spline.value(1.001)


def test_rejects_reference_and_spline_contract_mismatch():
    with pytest.raises(ValueError):
        controller(times_s=np.array([0.0, 0.5]))
    with pytest.raises(ValueError):
        controller(q_ref=np.array([[0.0, 1.0], [np.inf, 3.0]]))
    with pytest.raises(ValueError):
        controller(lower_nm=np.array([0.0, -6.0]), upper_nm=np.array([-1.0, 4.0]))


def test_mutating_spline_coefficients_after_construction_does_not_change_output():
    coeff = np.ones((5, 2)) * 0.5
    spline = residual(coeff)
    before = spline.value(0.5).copy()
    coeff[:] = 0.0
    np.testing.assert_array_equal(spline.value(0.5), before)


def test_clipped_total_drives_existing_held_integrator():
    from src.shared.python.motion_matching.same_input.integrator import (
        ROOT_COORDINATES,
        StepPolicy,
        integrate,
    )

    class UnitMassPlant:
        coordinate_order = (*ROOT_COORDINATES, "joint")

        def acceleration(self, q, v, tau):
            return tau.copy()

        def closure_pose_residual(self, q):
            return np.zeros(1)

    times = np.array([0.0, 0.02])
    reference = np.zeros((2, 7))
    reference[:, 6] = 10.0
    gains = np.zeros(7)
    gains[6] = 2.0
    spline = SplineResidual(
        breakpoints_s=times,
        coefficients_nm=np.zeros((4, 7)),
        scale_nm=np.ones(7),
        trust_radius=1.0,
        min_spacing_s=0.01,
    )
    ctrl = FixedGainController(
        times_s=times,
        q_ref=reference,
        qd_ref=np.zeros((2, 7)),
        tau_ff_ref=np.zeros((2, 7)),
        kp=gains,
        kd=np.zeros(7),
        lower_nm=-np.ones(7),
        upper_nm=np.ones(7),
        residual=spline,
    )
    decisions = []

    def source(_index, t, q, v):
        decision = ctrl.evaluate(t, q, v)
        decisions.append(decision)
        return decision.applied_nm

    run = integrate(
        UnitMassPlant(),
        np.zeros(7),
        np.zeros(7),
        source,
        steps=2,
        dt_s=0.01,
        policy=StepPolicy(substeps=2, project=False),
    )
    assert len(decisions) == 2
    assert all(d.commanded_nm[6] > 1.0 and d.saturated[6] for d in decisions)
    np.testing.assert_array_equal(run.efforts[:, 6], [1.0, 1.0])
    assert run.q[-1, 6] == pytest.approx(0.0002)
    assert run.v[-1, 6] == pytest.approx(0.02)
