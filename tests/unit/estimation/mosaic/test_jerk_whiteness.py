"""Tests for kinematic jerk prior, whiteness gate and declared discrepancy (MOSAIC-07, #11539).

Verifies:
1. Whiteness test at gamma = 0 on synthetic data passes on a sufficiently resolved
   basis and fails on an under-resolved basis, with reportable statistics.
2. Declared-discrepancy calculator returns sigma_tau from basis acceleration error
   and mass scale against analytic cases; below-floor declarations raise a precondition error.
3. Jerk prior weight selection avoids acceleration-noise torque blow-up
   (retained negative test reproduced without jerk prior, then closed with jerk prior).
4. Gate outputs include pass/fail/unqualified with underlying numbers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.jerk_whiteness import (
    DeclaredDiscrepancyCalculator,
    DiscrepancyGateResult,
    KinematicJerkPrior,
    WhitenessGateResult,
    compute_basis_acceleration_error,
    compute_declared_discrepancy,
    evaluate_whiteness_gate,
    validate_declared_discrepancy,
)
from src.shared.python.estimation.mosaic.kinematic_basis import BSplineBasis

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. BSplineBasis jerk property and polynomial derivative tests
# ---------------------------------------------------------------------------


def test_basis_evaluates_analytic_jerk() -> None:
    times = np.linspace(0.0, 1.0, 51)
    basis = BSplineBasis.uniform(times, n_coefficients=15, degree=5)
    assert basis.jerk is not None
    assert basis.jerk.shape == (51, 15)

    # Test on cubic polynomial: q(t) = 2 t^3 - 0.5 t^2 + t
    # q'(t) = 6 t^2 - t + 1
    # q''(t) = 12 t - 1
    # q'''(t) = 12 (constant)
    samples = (2.0 * times**3 - 0.5 * times**2 + times)[:, None]
    coeffs = basis.fit_least_squares(samples)
    jerk_eval = basis.jerk @ coeffs
    np.testing.assert_allclose(jerk_eval, 12.0, atol=1e-4)


def test_basis_jerk_raises_or_none_for_low_degree() -> None:
    times = np.linspace(0.0, 1.0, 21)
    basis = BSplineBasis.uniform(times, n_coefficients=6, degree=2)
    assert basis.jerk is None


# ---------------------------------------------------------------------------
# 2. KinematicJerkPrior tests
# ---------------------------------------------------------------------------


def test_jerk_prior_preconditions() -> None:
    times = np.linspace(0.0, 1.0, 31)
    basis_deg5 = BSplineBasis.uniform(times, n_coefficients=10, degree=5)
    basis_deg2 = BSplineBasis.uniform(times, n_coefficients=6, degree=2)

    with pytest.raises(ContractViolationError):
        KinematicJerkPrior(basis=basis_deg5, weight=-1.0)

    with pytest.raises(ContractViolationError):
        KinematicJerkPrior(basis=basis_deg2, weight=1.0)


def test_jerk_prior_residual_and_gradient() -> None:
    times = np.linspace(0.0, 1.0, 41)
    basis = BSplineBasis.uniform(times, n_coefficients=12, degree=5)
    prior = KinematicJerkPrior(basis=basis, weight=2.5)

    coeffs = np.ones((12, 2))
    residual = prior.compute_residual(coeffs)
    assert residual.shape == (41, 2)
    expected_res = 2.5 * (basis.jerk @ coeffs)
    np.testing.assert_allclose(residual, expected_res, atol=1e-12)

    cost = prior.compute_cost(coeffs)
    assert np.isclose(cost, float(np.sum(expected_res**2)))

    normal_matrix = prior.normal_matrix()
    assert normal_matrix.shape == (12, 12)
    np.testing.assert_allclose(
        normal_matrix, (2.5**2) * (basis.jerk.T @ basis.jerk), atol=1e-12
    )

    grad = prior.gradient(coeffs)
    assert grad.shape == (12, 2)
    np.testing.assert_allclose(grad, normal_matrix @ coeffs, atol=1e-12)


# ---------------------------------------------------------------------------
# 3. Whiteness Gate (Gate 1) tests
# ---------------------------------------------------------------------------


def test_whiteness_gate_passes_on_resolved_basis() -> None:
    rng = np.random.default_rng(42)
    times = np.linspace(0.0, 1.0, 200)
    sigma = 0.01

    # Signal: smooth trajectory
    clean_trajectory = np.sin(2.0 * np.pi * times)[:, None]
    noise = rng.normal(scale=sigma, size=clean_trajectory.shape)
    observed = clean_trajectory + noise

    # Sufficiently resolved basis: 40 coefficients for 1 Hz wave over 200 points
    basis_resolved = BSplineBasis.uniform(times, n_coefficients=40, degree=5)
    coeffs = basis_resolved.fit_least_squares(observed)
    predicted = basis_resolved.position @ coeffs

    result = evaluate_whiteness_gate(
        observed=observed,
        predicted=predicted,
        sigma=sigma,
    )

    assert isinstance(result, WhitenessGateResult)
    assert result.status == "pass"
    assert result.passed is True
    # Normalized RMS should be near 1.0 for true Gaussian white noise
    assert 0.7 < result.rms < 1.3
    # Autocorrelation of white noise residuals should be small (|rho| < 0.25)
    assert abs(result.autocorrelation_lag1) < 0.25
    # Durbin-Watson statistic should be close to 2.0 (uncorrelated)
    assert 1.5 < result.durbin_watson < 2.5


def test_whiteness_gate_fails_on_under_resolved_basis() -> None:
    rng = np.random.default_rng(42)
    times = np.linspace(0.0, 1.0, 200)
    sigma = 0.001  # very small noise so under-resolution dominates

    # Fast signal: 5 Hz oscillation
    clean_trajectory = np.sin(2.0 * np.pi * 5.0 * times)[:, None]
    noise = rng.normal(scale=sigma, size=clean_trajectory.shape)
    observed = clean_trajectory + noise

    # Under-resolved basis: only 7 coefficients for a 5 Hz signal over 200 points
    basis_under = BSplineBasis.uniform(times, n_coefficients=7, degree=5)
    coeffs = basis_under.fit_least_squares(observed)
    predicted = basis_under.position @ coeffs

    result = evaluate_whiteness_gate(
        observed=observed,
        predicted=predicted,
        sigma=sigma,
    )

    assert result.status == "fail"
    assert result.passed is False
    # Residuals have severe structured autocorrelation because the spline can't track the wave
    assert abs(result.autocorrelation_lag1) > 0.5 or result.rms > 3.0


def test_whiteness_gate_unqualified_on_missing_or_invalid() -> None:
    result_nan = evaluate_whiteness_gate(
        observed=np.array([[np.nan], [1.0]]),
        predicted=np.array([[0.0], [1.0]]),
        sigma=0.1,
    )
    assert result_nan.status == "unqualified"
    assert "non-finite" in result_nan.reason.lower()


# ---------------------------------------------------------------------------
# 4. Declared Discrepancy Calculator (Gate 2) tests
# ---------------------------------------------------------------------------


def test_declared_discrepancy_calculator_analytic() -> None:
    # Analytic case: basis acceleration error 0.05 m/s^2, mass scale 20.0 kg
    # Minimum floor sigma_tau = 0.05 * 20.0 = 1.0 N (or N*m)
    floor = compute_declared_discrepancy(basis_acceleration_error=0.05, mass_scale=20.0)
    assert np.isclose(floor, 1.0)

    # With safety factor 1.5
    floor_safe = compute_declared_discrepancy(
        basis_acceleration_error=0.05, mass_scale=20.0, safety_factor=1.5
    )
    assert np.isclose(floor_safe, 1.5)


def test_declared_discrepancy_validation_precondition_below_floor() -> None:
    # Floor is 0.02 * 50.0 = 1.0
    # Declaring 0.5 should raise ContractViolationError
    with pytest.raises(ContractViolationError) as exc_info:
        validate_declared_discrepancy(
            declared_sigma_tau=0.5,
            basis_acceleration_error=0.02,
            mass_scale=50.0,
        )
    assert "below floor" in str(exc_info.value).lower()


def test_declared_discrepancy_validation_passes_above_floor() -> None:
    # Floor is 0.02 * 50.0 = 1.0
    res = validate_declared_discrepancy(
        declared_sigma_tau=1.2,
        basis_acceleration_error=0.02,
        mass_scale=50.0,
    )
    assert isinstance(res, DiscrepancyGateResult)
    assert res.status == "pass"
    assert res.passed is True
    assert np.isclose(res.min_discrepancy_floor, 1.0)
    assert np.isclose(res.declared_sigma_tau, 1.2)


def test_compute_basis_acceleration_error_sinusoid() -> None:
    # For a known harmonic trajectory q(t) = sin(2 pi f t)
    # true a(t) = -(2 pi f)^2 sin(2 pi f t)
    times = np.linspace(0.0, 1.0, 100)
    basis = BSplineBasis.uniform(times, n_coefficients=30, degree=5)

    def trajectory(t: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * t)[:, None]

    def acceleration(t: np.ndarray) -> np.ndarray:
        return (-((2.0 * np.pi) ** 2) * np.sin(2.0 * np.pi * t))[:, None]

    acc_error = compute_basis_acceleration_error(
        basis=basis,
        trajectory_func=trajectory,
        acceleration_func=acceleration,
        times=times,
    )
    assert acc_error > 0.0
    assert acc_error < 0.1  # quintic spline with 30 knots on 1 Hz wave is very accurate


# ---------------------------------------------------------------------------
# 5. Retained negative test: acceleration-noise torque blow-up & jerk prior fix
# ---------------------------------------------------------------------------


def test_retained_negative_test_jerk_prior_avoids_acceleration_noise_torque_blowup() -> (
    None
):
    """Reproduce retained negative result: acceleration noise inflates torques without jerk prior.

    Then verify that adding the jerk prior suppresses spurious acceleration fluctuations
    and closes the defect.
    """
    rng = np.random.default_rng(123)
    times = np.linspace(0.0, 1.0, 150)
    dt = times[1] - times[0]

    # True trajectory: 1-DOF pendulum / arm swinging
    freq = 1.2
    q_true = (0.5 * np.cos(2 * np.pi * freq * times))[:, None]
    v_true = (-0.5 * 2 * np.pi * freq * np.sin(2 * np.pi * freq * times))[:, None]
    a_true = (-0.5 * (2 * np.pi * freq) ** 2 * np.cos(2 * np.pi * freq * times))[
        :, None
    ]

    # Dynamics: tau = I * a + m * g * l * sin(q)
    mass, length, inertia = 1.5, 0.8, 0.32
    g = 9.81
    tau_true = inertia * a_true + mass * g * length * np.sin(q_true)

    # Noisy observations (e.g. marker / sensor noise of 2 mm)
    noise_sigma = 0.003
    q_noisy = q_true + rng.normal(scale=noise_sigma, size=q_true.shape)

    # 40-coefficient basis (flexible enough to overfit high-frequency noise if unregularized)
    basis = BSplineBasis.uniform(times, n_coefficients=40, degree=5)

    # Case A (Unregularized / no jerk prior): Least-squares fit to noisy data
    coeffs_unreg = basis.fit_least_squares(q_noisy, ridge=0.0)
    q_unreg, _, a_unreg = basis.evaluate(coeffs_unreg)
    tau_unreg = inertia * a_unreg + mass * g * length * np.sin(q_unreg)

    torque_rms_error_unreg = float(np.sqrt(np.mean((tau_unreg - tau_true) ** 2)))

    # Case B (With kinematic jerk prior): Regularized fit
    # Normal equations: (P^T P + w_jerk^2 J^T J) C = P^T q_noisy
    w_jerk = 5e-5
    jerk_prior = KinematicJerkPrior(basis=basis, weight=w_jerk)
    lhs = basis.position.T @ basis.position + jerk_prior.normal_matrix()
    rhs = basis.position.T @ q_noisy
    coeffs_reg = np.linalg.solve(lhs, rhs)

    q_reg, _, a_reg = basis.evaluate(coeffs_reg)
    tau_reg = inertia * a_reg + mass * g * length * np.sin(q_reg)

    torque_rms_error_reg = float(np.sqrt(np.mean((tau_reg - tau_true) ** 2)))

    # Negative result reproduced: unregularized fit has spurious high torque error due to a_noise (blow-up)
    assert torque_rms_error_unreg > 3.0 * torque_rms_error_reg
    # Jerk prior successfully suppresses torque noise blow-up
    assert torque_rms_error_reg < 0.35 * float(np.sqrt(np.mean(tau_true**2)))
