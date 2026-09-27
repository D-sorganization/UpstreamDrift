"""Unit tests for white-jerk RTS kinematic smoother (Issue #11029)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import block_diag

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.kinematic_smoother import (
    KinematicSmoothingResult,
    NoiseParameters,
    estimate_noise_parameters,
    smooth_kinematic,
    white_jerk_transition,
)

pytestmark = pytest.mark.unit


def test_white_jerk_transition_discretization() -> None:
    """Verify exact continuous-to-discrete white jerk transition matrices."""
    dt = 0.01
    F, Q_unit = white_jerk_transition(dt)

    assert F.shape == (3, 3)
    assert Q_unit.shape == (3, 3)

    expected_F = np.array(
        [
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ]
    )
    expected_Q_unit = np.array(
        [
            [dt**5 / 20.0, dt**4 / 8.0, dt**3 / 6.0],
            [dt**4 / 8.0, dt**3 / 3.0, dt**2 / 2.0],
            [dt**3 / 6.0, dt**2 / 2.0, dt],
        ]
    )

    np.testing.assert_allclose(F, expected_F, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(Q_unit, expected_Q_unit, rtol=1e-14, atol=1e-14)


def test_exactness_dense_batch() -> None:
    """Verify RTS smoother equals dense batch Gaussian posterior to 1e-8 for N=25, nq=1."""
    N = 25
    rate_hz = 10.0
    dt = 1.0 / rate_hz
    qc = 1.0
    r = 0.05

    F, Q_unit = white_jerk_transition(dt)
    Q = qc * Q_unit

    rng = np.random.default_rng(1234)
    t = np.arange(N) * dt
    y_clean = np.sin(t)
    y = y_clean + rng.normal(0.0, np.sqrt(r), size=N)
    q_input = y[:, None]

    P0 = np.eye(3, dtype=np.float64)
    m0 = np.array([y[0], 0.0, 0.0], dtype=np.float64)

    # RTS Smoother
    result = smooth_kinematic(
        q_input,
        rate_hz,
        jerk_psd=qc,
        measurement_var=r,
        initial_mean=m0,
        initial_covariance=P0,
    )

    # Independent dense-batch construction (block F/Q prior propagation)
    Sigma_prior = np.zeros((3 * N, 3 * N), dtype=np.float64)
    mu_prior = np.zeros((N, 3), dtype=np.float64)
    mu_prior[0] = m0
    for k in range(1, N):
        mu_prior[k] = F @ mu_prior[k - 1]

    prior_var = [P0]
    for _k in range(1, N):
        prior_var.append(F @ prior_var[-1] @ F.T + Q)

    for i in range(N):
        Sigma_prior[3 * i : 3 * (i + 1), 3 * i : 3 * (i + 1)] = prior_var[i]
        for j in range(i + 1, N):
            F_pow = np.eye(3, dtype=np.float64)
            for _ in range(j - i):
                F_pow = F_pow @ F
            cov_ij = prior_var[i] @ F_pow.T
            Sigma_prior[3 * i : 3 * (i + 1), 3 * j : 3 * (j + 1)] = cov_ij
            Sigma_prior[3 * j : 3 * (j + 1), 3 * i : 3 * (i + 1)] = cov_ij.T

    H_k = np.array([[1.0, 0.0, 0.0]], dtype=np.float64)
    H_dense = block_diag(*([H_k] * N))
    Sigma_V = r * np.eye(N, dtype=np.float64)

    S_dense = H_dense @ Sigma_prior @ H_dense.T + Sigma_V
    K_dense = np.linalg.solve(S_dense, H_dense @ Sigma_prior).T
    mu_prior_flat = mu_prior.reshape(3 * N)
    mu_post = mu_prior_flat + K_dense @ (y - H_dense @ mu_prior_flat)
    Sigma_post = Sigma_prior - K_dense @ H_dense @ Sigma_prior
    Sigma_post = 0.5 * (Sigma_post + Sigma_post.T)

    dense_m = mu_post.reshape((N, 3))
    rts_m = np.column_stack(
        [result.position[:, 0], result.velocity[:, 0], result.acceleration[:, 0]]
    )

    dense_cov_diag = np.zeros((N, 3), dtype=np.float64)
    for k in range(N):
        dense_cov_diag[k] = np.diag(
            Sigma_post[3 * k : 3 * (k + 1), 3 * k : 3 * (k + 1)]
        )

    rts_cov_diag = np.column_stack(
        [
            result.position_std[:, 0] ** 2,
            result.velocity_std[:, 0] ** 2,
            result.acceleration_std[:, 0] ** 2,
        ]
    )

    np.testing.assert_allclose(rts_m, dense_m, rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(rts_cov_diag, dense_cov_diag, rtol=1e-8, atol=1e-8)


def test_recovery_synthetic_white_jerk() -> None:
    """Recover r within 15% and q_c within factor of 2 on seeded synthetic trajectory."""
    rate_hz = 360.0
    dt = 1.0 / rate_hz
    qc_true = 10.0
    r_true = 1e-4
    N = 3000

    F, Q_unit = white_jerk_transition(dt)
    Q = qc_true * Q_unit

    rng = np.random.default_rng(42)
    x = np.zeros((N, 3), dtype=np.float64)
    L_Q = np.linalg.cholesky(Q)
    for k in range(N - 1):
        x[k + 1] = F @ x[k] + L_Q @ rng.normal(size=3)

    y = x[:, 0] + np.sqrt(r_true) * rng.normal(size=N)
    q = y[:, None]

    params = estimate_noise_parameters(q, rate_hz)

    assert isinstance(params, NoiseParameters)
    assert params.success is not None
    assert all(params.success)

    qc_est = float(params.jerk_psd[0])
    r_est = float(params.measurement_var[0])

    r_error_pct = abs(r_est - r_true) / r_true
    qc_ratio = qc_est / qc_true

    assert r_error_pct < 0.15, (
        f"r error {r_error_pct:.2%} exceeds 15% (r_est={r_est}, r_true={r_true})"
    )
    assert 0.5 < qc_ratio < 2.0, (
        f"qc ratio {qc_ratio:.2f} not within factor of 2 (qc_est={qc_est}, qc_true={qc_true})"
    )


def test_coverage_true_parameters() -> None:
    """With true parameters, fraction of true positions inside +-1.96 sigma is in [0.92, 0.98]."""
    rate_hz = 360.0
    dt = 1.0 / rate_hz
    qc_true = 10.0
    r_true = 1e-4
    N = 3000

    F, Q_unit = white_jerk_transition(dt)
    Q = qc_true * Q_unit

    rng = np.random.default_rng(0)
    x = np.zeros((N, 3), dtype=np.float64)
    L_Q = np.linalg.cholesky(Q)
    for k in range(N - 1):
        x[k + 1] = F @ x[k] + L_Q @ rng.normal(size=3)

    y = x[:, 0] + np.sqrt(r_true) * rng.normal(size=N)
    q = y[:, None]

    result = smooth_kinematic(q, rate_hz, jerk_psd=qc_true, measurement_var=r_true)

    pos_err = np.abs(x[:, 0] - result.position[:, 0])
    sigma = result.position_std[:, 0]
    coverage = float(np.mean(pos_err <= 1.96 * sigma))

    assert 0.92 <= coverage <= 0.98, (
        f"Coverage {coverage:.4f} outside nominal [0.92, 0.98]"
    )


def test_gap_widening_and_finiteness() -> None:
    """Inside a 20-frame NaN gap, posterior sigma is strictly larger and values stay finite."""
    rate_hz = 360.0
    dt = 1.0 / rate_hz
    qc = 10.0
    r = 1e-4
    N = 100

    F, Q_unit = white_jerk_transition(dt)
    Q = qc * Q_unit

    rng = np.random.default_rng(42)
    x = np.zeros((N, 3), dtype=np.float64)
    L_Q = np.linalg.cholesky(Q)
    for k in range(N - 1):
        x[k + 1] = F @ x[k] + L_Q @ rng.normal(size=3)

    y = x[:, 0] + np.sqrt(r) * rng.normal(size=N)

    gap_start = 40
    gap_end = 60
    y[gap_start:gap_end] = np.nan
    q = y[:, None]

    result = smooth_kinematic(q, rate_hz, jerk_psd=qc, measurement_var=r)

    assert np.all(np.isfinite(result.position))
    assert np.all(np.isfinite(result.velocity))
    assert np.all(np.isfinite(result.acceleration))
    assert np.all(np.isfinite(result.position_std))
    assert np.all(np.isfinite(result.velocity_std))
    assert np.all(np.isfinite(result.acceleration_std))

    sigma = result.position_std[:, 0]
    sigma_before = sigma[gap_start - 1]
    sigma_after = sigma[gap_end]
    sigma_inside = sigma[gap_start:gap_end]

    assert np.all(sigma_inside > sigma_before)
    assert np.all(sigma_inside > sigma_after)


def test_derivatives_noiseless_quintic() -> None:
    """On a noiseless quintic, velocity and acceleration match analytic derivatives to within 1e-6."""
    rate_hz = 360.0
    dt = 1.0 / rate_hz
    t = np.arange(0, 1.0, dt)
    N = len(t)

    c = [1.0, 0.5, -0.2, 0.1, -0.05, 0.01]
    p = c[0] + c[1] * t + c[2] * t**2 + c[3] * t**3 + c[4] * t**4 + c[5] * t**5
    v_true = c[1] + 2 * c[2] * t + 3 * c[3] * t**2 + 4 * c[4] * t**3 + 5 * c[5] * t**4
    a_true = 2 * c[2] + 6 * c[3] * t + 12 * c[4] * t**2 + 20 * c[5] * t**3

    signal_scale = float(np.ptp(p))
    r_tiny = 1e-14
    qc = 1e4

    m0 = np.array([p[0], v_true[0], a_true[0]], dtype=np.float64)
    P0 = 1e-6 * np.eye(3, dtype=np.float64)

    q = p[:, None]
    result = smooth_kinematic(
        q,
        rate_hz,
        jerk_psd=qc,
        measurement_var=r_tiny,
        initial_mean=m0,
        initial_covariance=P0,
    )

    err_v = float(np.max(np.abs(result.velocity[:, 0] - v_true))) / signal_scale
    err_a = float(np.max(np.abs(result.acceleration[:, 0] - a_true))) / signal_scale

    assert err_v < 1e-6, f"Velocity relative error {err_v:.2e} >= 1e-6"
    assert err_a < 1e-6, f"Acceleration relative error {err_a:.2e} >= 1e-6"


def test_dbc_contracts() -> None:
    """Preconditions refuse non-2D arrays, <3 frames, non-positive rate, non-positive noise."""
    valid_q = np.zeros((10, 2), dtype=np.float64)

    # 1. Non-2D array
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(np.zeros(10), 360.0, jerk_psd=1.0, measurement_var=1.0)

    # 2. Fewer than 3 frames
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(np.zeros((2, 2)), 360.0, jerk_psd=1.0, measurement_var=1.0)

    # 3. rate_hz <= 0
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(valid_q, 0.0, jerk_psd=1.0, measurement_var=1.0)
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(valid_q, -10.0, jerk_psd=1.0, measurement_var=1.0)

    # 4. Non-positive noise
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(valid_q, 360.0, jerk_psd=0.0, measurement_var=1.0)
    with pytest.raises((ContractViolationError, ValueError)):
        smooth_kinematic(valid_q, 360.0, jerk_psd=1.0, measurement_var=-0.1)

    # white_jerk_transition precondition
    with pytest.raises((ContractViolationError, ValueError)):
        white_jerk_transition(0.0)


def test_vectorized_multichannel() -> None:
    """Process multiple channels simultaneously and verify equality with single-channel runs."""
    rate_hz = 100.0
    N = 50
    rng = np.random.default_rng(999)

    t = np.linspace(0, 1, N)
    q1 = np.sin(2 * np.pi * t) + rng.normal(0, 0.02, N)
    q2 = np.cos(2 * np.pi * t) + rng.normal(0, 0.03, N)
    q_multi = np.column_stack([q1, q2])

    qc = np.array([5.0, 8.0])
    r = np.array([0.001, 0.002])

    res_multi = smooth_kinematic(q_multi, rate_hz, jerk_psd=qc, measurement_var=r)
    res_1 = smooth_kinematic(q1[:, None], rate_hz, jerk_psd=qc[0], measurement_var=r[0])
    res_2 = smooth_kinematic(q2[:, None], rate_hz, jerk_psd=qc[1], measurement_var=r[1])

    assert res_multi.position.shape == (N, 2)
    assert res_multi.velocity.shape == (N, 2)
    assert res_multi.acceleration.shape == (N, 2)
    assert res_multi.position_std.shape == (N, 2)

    np.testing.assert_allclose(
        res_multi.position[:, 0], res_1.position[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        res_multi.position[:, 1], res_2.position[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        res_multi.position_std[:, 0], res_1.position_std[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        res_multi.position_std[:, 1], res_2.position_std[:, 0], rtol=1e-12
    )

    total_lik = res_1.log_marginal_likelihood + res_2.log_marginal_likelihood
    np.testing.assert_allclose(res_multi.log_marginal_likelihood, total_lik, rtol=1e-10)


def test_prior_contract_and_read_only_result() -> None:
    """Half-specified priors, wrong shapes and infinities are refused; results are frozen."""
    q = np.ones((10, 2))
    with pytest.raises(ContractViolationError):
        smooth_kinematic(
            q, 100.0, jerk_psd=1.0, measurement_var=1.0, initial_mean=np.zeros(3)
        )
    with pytest.raises(ContractViolationError):
        smooth_kinematic(
            q,
            100.0,
            jerk_psd=1.0,
            measurement_var=1.0,
            initial_mean=np.zeros((3, 3)),
            initial_covariance=np.eye(3),
        )
    q_inf = q.copy()
    q_inf[3, 1] = np.inf
    with pytest.raises(ContractViolationError):
        smooth_kinematic(q_inf, 100.0, jerk_psd=1.0, measurement_var=1.0)
    result = smooth_kinematic(q, 100.0, jerk_psd=1.0, measurement_var=1.0)
    with pytest.raises(ValueError):
        result.position[0, 0] = 5.0
