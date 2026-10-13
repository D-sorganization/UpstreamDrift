"""Tests for MOSAIC observation factors: camera projection, robust kernels, marker offsets."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.core.contracts import ContractViolationError
from src.shared.python.estimation.mosaic.observation_factors import (
    CameraModel,
    CauchyKernel,
    GemanMcClureKernel,
    HuberKernel,
    MarkerAttachmentFactor,
    OffsetRecoveryResult,
    RobustKernel,
    TukeyKernel,
    recover_marker_offsets,
    rodrigues_from_rotation,
    rotation_from_rodrigues,
)
from src.shared.python.estimation.mosaic.planar_chain import (
    PlanarChain,
    PlanarMarkerSet,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _sample_camera() -> CameraModel:
    fx, fy, cx, cy = 1000.0, 1000.0, 640.0, 480.0
    rvec = np.array([0.1, -0.2, 0.05], dtype=np.float64)
    R = rotation_from_rodrigues(rvec)
    t = np.array([0.2, -0.3, 2.5], dtype=np.float64)
    return CameraModel(fx=fx, fy=fy, cx=cx, cy=cy, R=R, t=t)


# ---------------------------------------------------------------------------
# CameraModel tests
# ---------------------------------------------------------------------------


def test_camera_model_projection_and_shapes() -> None:
    cam = _sample_camera()

    # 1D single point (3,)
    pw_single = np.array([0.5, 0.3, 0.1], dtype=np.float64)
    uv_single = cam.project(pw_single)
    assert uv_single.shape == (2,)

    # Batched points (N, 3)
    pw_batch = np.array(
        [[0.5, 0.3, 0.1], [-0.4, 0.2, 0.0], [0.1, -0.5, 0.2]], dtype=np.float64
    )
    uv_batch = cam.project(pw_batch)
    assert uv_batch.shape == (3, 2)
    np.testing.assert_allclose(uv_batch[0], uv_single, atol=1e-12)

    # Higher dimensional batch (K, N, 3)
    pw_high = np.tile(pw_batch, (2, 1, 1))
    uv_high = cam.project(pw_high)
    assert uv_high.shape == (2, 3, 2)
    np.testing.assert_allclose(uv_high[0], uv_batch, atol=1e-12)

    # 2D planar points (N, 2) embedded with z=0
    pw_planar = pw_batch[:, :2]
    uv_planar = cam.project(pw_planar)
    assert uv_planar.shape == (3, 2)
    pw_planar_3d = np.column_stack([pw_planar, np.zeros(len(pw_planar))])
    np.testing.assert_allclose(uv_planar, cam.project(pw_planar_3d), atol=1e-12)


def test_camera_model_jacobian_wrt_p_world_numerical() -> None:
    cam = _sample_camera()
    pw = np.array(
        [[0.5, 0.3, 0.1], [-0.4, 0.2, -0.1], [0.1, -0.5, 0.3]], dtype=np.float64
    )
    jac_analytic = cam.jacobian_p_world(pw)
    assert jac_analytic.shape == (3, 2, 3)

    eps = 1e-7
    for k in range(3):
        dp = np.zeros(3)
        dp[k] = eps
        uv_pos = cam.project(pw + dp)
        uv_neg = cam.project(pw - dp)
        jac_num = (uv_pos - uv_neg) / (2.0 * eps)
        np.testing.assert_allclose(jac_analytic[:, :, k], jac_num, rtol=1e-5, atol=1e-6)

    # Also test planar (N, 2)
    pw_planar = pw[:, :2]
    jac_planar = cam.jacobian_p_world(pw_planar)
    assert jac_planar.shape == (3, 2, 2)
    for k in range(2):
        dp2 = np.zeros(2)
        dp2[k] = eps
        uv_pos = cam.project(pw_planar + dp2)
        uv_neg = cam.project(pw_planar - dp2)
        jac_num2 = (uv_pos - uv_neg) / (2.0 * eps)
        np.testing.assert_allclose(jac_planar[:, :, k], jac_num2, rtol=1e-5, atol=1e-6)


def test_camera_model_jacobian_wrt_camera_params_numerical() -> None:
    cam = _sample_camera()
    pw = np.array([[0.3, 0.1, 0.2], [-0.2, 0.4, 0.0]], dtype=np.float64)

    # 10 parameters: [fx, fy, cx, cy, rx, ry, rz, tx, ty, tz]
    jac_params = cam.jacobian_camera_params(pw)
    assert jac_params.shape == (2, 2, 10)

    p0 = cam.camera_params
    eps = 1e-7
    for p_idx in range(10):
        dp = np.zeros(10)
        dp[p_idx] = eps
        cam_pos = cam.with_camera_params(p0 + dp)
        cam_neg = cam.with_camera_params(p0 - dp)
        jac_num = (cam_pos.project(pw) - cam_neg.project(pw)) / (2.0 * eps)
        np.testing.assert_allclose(
            jac_params[:, :, p_idx], jac_num, rtol=1e-5, atol=1e-6
        )


def test_camera_model_preconditions() -> None:
    cam = _sample_camera()
    with pytest.raises(ContractViolationError):
        CameraModel(fx=-1.0, fy=1000.0, cx=640.0, cy=480.0, R=cam.R, t=cam.t)
    with pytest.raises(ContractViolationError):
        CameraModel(fx=1000.0, fy=0.0, cx=640.0, cy=480.0, R=cam.R, t=cam.t)
    with pytest.raises(ContractViolationError):
        CameraModel(
            fx=1000.0,
            fy=1000.0,
            cx=640.0,
            cy=480.0,
            R=np.zeros((3, 3)),
            t=cam.t,
        )
    with pytest.raises(ContractViolationError):
        CameraModel(
            fx=1000.0,
            fy=1000.0,
            cx=640.0,
            cy=480.0,
            R=cam.R,
            t=np.zeros(4),
        )
    # Point behind camera (depth Z_c <= 0)
    behind_point = np.array([0.0, 0.0, -10.0])
    with pytest.raises(ContractViolationError):
        cam.project(behind_point)


# ---------------------------------------------------------------------------
# Robust Loss Kernels tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kernel",
    [
        HuberKernel(delta=1.5),
        CauchyKernel(c=2.0),
        GemanMcClureKernel(c=1.8),
        TukeyKernel(c=2.5),
    ],
)
def test_robust_kernels_derivatives_conformance(kernel: RobustKernel) -> None:
    r_vals = np.array([-4.0, -2.5, -1.0, -0.2, 0.0, 0.3, 1.2, 2.7, 5.0])
    s_vals = r_vals**2

    # Weight definition w(r) = psi(r)/r
    psi_vals = kernel.psi(r_vals)
    w_vals = kernel.weight(r_vals)
    np.testing.assert_allclose(psi_vals, w_vals * r_vals, atol=1e-12)

    # Numerical derivative of rho(r^2) w.r.t r equals psi(r)
    eps = 1e-7
    for r in r_vals:
        if abs(r) < 1e-12:
            continue  # origin tested separately
        rho_pos = kernel.rho((r + eps) ** 2)
        rho_neg = kernel.rho((r - eps) ** 2)
        d_rho = (rho_pos - rho_neg) / (2.0 * eps)
        np.testing.assert_allclose(kernel.psi(r), d_rho, rtol=1e-5, atol=1e-6)

    # Numerical derivative of psi(r) w.r.t r equals psi'(r)
    for r in r_vals:
        # Skip points of non-differentiability for Huber (r = +- delta) and Tukey (r = +- c)
        if isinstance(kernel, HuberKernel) and np.isclose(abs(r), kernel.delta):
            continue
        if isinstance(kernel, TukeyKernel) and np.isclose(abs(r), kernel.c):
            continue
        psi_pos = kernel.psi(r + eps)
        psi_neg = kernel.psi(r - eps)
        d_psi = (psi_pos - psi_neg) / (2.0 * eps)
        np.testing.assert_allclose(kernel.psi_deriv(r), d_psi, rtol=1e-5, atol=1e-6)


def test_robust_kernels_asymptotic_and_cutoffs() -> None:
    # Huber: linear growth beyond delta
    huber = HuberKernel(delta=1.0)
    assert np.isclose(huber.weight(0.0), 1.0)
    assert np.isclose(huber.weight(0.5), 1.0)
    assert np.isclose(huber.weight(2.0), 0.5)
    assert np.isclose(huber.psi_deriv(0.5), 1.0)
    assert np.isclose(huber.psi_deriv(2.0), 0.0)

    # Tukey: strictly zero weights and derivatives beyond c
    tukey = TukeyKernel(c=2.0)
    assert np.isclose(tukey.weight(0.0), 1.0)
    assert np.isclose(tukey.weight(3.0), 0.0)
    assert np.isclose(tukey.psi(3.0), 0.0)
    assert np.isclose(tukey.psi_deriv(3.0), 0.0)


def test_robust_kernels_preconditions() -> None:
    with pytest.raises(ContractViolationError):
        HuberKernel(delta=0.0)
    with pytest.raises(ContractViolationError):
        CauchyKernel(c=-1.0)
    with pytest.raises(ContractViolationError):
        GemanMcClureKernel(c=0.0)
    with pytest.raises(ContractViolationError):
        TukeyKernel(c=-0.5)


def test_robust_kernel_outlier_rejection() -> None:
    """Robust kernels maintain graceful degradation under severe injected outliers."""
    rng = np.random.default_rng(42)
    n_points = 100
    true_val = 3.5

    # 80 inliers near 3.5, 20 extreme outliers near 50.0
    inliers = rng.normal(loc=true_val, scale=0.1, size=80)
    outliers = rng.uniform(low=40.0, high=60.0, size=20)
    data = np.concatenate([inliers, outliers])

    # Standard least-squares estimate is simple mean
    l2_estimate = float(np.mean(data))
    l2_error = abs(l2_estimate - true_val)
    assert l2_error > 8.0  # heavily corrupted by outliers

    # Robust M-estimators using IRLS with Huber and Cauchy/Tukey
    kernels: list[RobustKernel] = [
        HuberKernel(delta=1.0),
        CauchyKernel(c=1.0),
        TukeyKernel(c=2.5),
    ]
    for kernel in kernels:
        mu = float(np.median(data))
        for _ in range(30):
            res = data - mu
            w = kernel.weight(res)
            mu = float(np.sum(w * data) / np.sum(w))
        robust_error = abs(mu - true_val)
        # Robust estimator degrades gracefully: error stays < 0.35 compared to L2 error > 8.0
        assert robust_error < 0.35
        # More than 25x better than least-squares
        assert robust_error < l2_error / 25.0
        # Redescending kernels achieve even sharper rejection (< 0.05)
        if isinstance(kernel, (CauchyKernel, TukeyKernel)):
            assert robust_error < 0.05


# ---------------------------------------------------------------------------
# Marker Attachment Offsets Factor tests
# ---------------------------------------------------------------------------


def test_marker_attachment_factor_jacobians_numerical() -> None:
    chain = PlanarChain(
        link_lengths=np.array([0.8, 0.5, 0.3]),
        actuated=np.array([True, True, True]),
    )
    nominal_offsets = np.array(
        [[0.4, 0.05], [0.2, -0.03], [0.15, 0.02]], dtype=np.float64
    )
    link_indices = np.array([0, 1, 2], dtype=np.int64)
    factor = MarkerAttachmentFactor(
        chain=chain,
        link_indices=link_indices,
        nominal_offsets=nominal_offsets,
    )

    q = np.array([[0.3, -0.7, 0.4], [1.2, 0.4, -0.2]], dtype=np.float64)
    delta_p = np.array([[0.01, -0.02], [-0.01, 0.03], [0.02, 0.01]], dtype=np.float64)

    pos = factor.marker_positions(q, delta_p)
    assert pos.shape == (2, 3, 2)

    # 1. Jacobian w.r.t attachment offsets delta_p: shape (N, M, 2, 2) per marker
    j_delta = factor.jacobian_offsets(q)
    assert j_delta.shape == (2, 3, 2, 2)
    eps = 1e-7
    for m in range(3):
        for dim in range(2):
            bump = np.zeros_like(delta_p)
            bump[m, dim] = eps
            fd = (
                factor.marker_positions(q, delta_p + bump)
                - factor.marker_positions(q, delta_p - bump)
            ) / (2.0 * eps)
            np.testing.assert_allclose(
                j_delta[:, m, :, dim], fd[:, m], rtol=1e-5, atol=1e-6
            )

    # 2. Jacobian w.r.t joint angles q: shape (N, M, 2, n_links)
    jq, jl = factor.jacobian_kinematics(q, delta_p)
    assert jq.shape == (2, 3, 2, 3)
    assert jl.shape == (2, 3, 2, 3)
    for joint in range(3):
        dq = np.zeros_like(q)
        dq[:, joint] = eps
        fd_q = (
            factor.marker_positions(q + dq, delta_p)
            - factor.marker_positions(q - dq, delta_p)
        ) / (2.0 * eps)
        np.testing.assert_allclose(jq[..., joint], fd_q, rtol=1e-5, atol=1e-6)

        bump_chain = PlanarChain(
            link_lengths=chain.link_lengths + eps * np.eye(3)[joint],
            actuated=chain.actuated,
        )
        bump_factor = MarkerAttachmentFactor(
            chain=bump_chain,
            link_indices=link_indices,
            nominal_offsets=nominal_offsets,
        )
        fd_l = (bump_factor.marker_positions(q, delta_p) - pos) / eps
        np.testing.assert_allclose(jl[..., joint], fd_l, rtol=1e-5, atol=1e-6)


def test_marker_attachment_offsets_recovery_synthetic() -> None:
    """Recover local marker attachment offsets from kinematics and report observable fraction."""
    chain = PlanarChain(
        link_lengths=np.array([0.9, 0.6, 0.4]),
        actuated=np.array([True, True, True]),
    )
    link_indices = np.array([0, 1, 2], dtype=np.int64)
    nominal_offsets = np.array([[0.45, 0.0], [0.3, 0.0], [0.2, 0.0]], dtype=np.float64)
    true_delta = np.array(
        [[0.015, -0.010], [-0.008, 0.012], [0.005, -0.007]], dtype=np.float64
    )

    factor = MarkerAttachmentFactor(
        chain=chain,
        link_indices=link_indices,
        nominal_offsets=nominal_offsets,
    )

    # Multi-configuration poses ensuring excitation of joint rotations
    rng = np.random.default_rng(123)
    n_nodes = 25
    q = rng.uniform(-np.pi / 2, np.pi / 2, size=(n_nodes, 3))

    # Synthetic observations with true delta
    pos_clean = factor.marker_positions(q, true_delta)

    # Recover with zero noise to strict tolerance
    res: OffsetRecoveryResult = recover_marker_offsets(
        factor=factor,
        q=q,
        observed_positions=pos_clean,
    )
    assert res.observable_fraction == 1.0
    assert res.n_observable == 6
    np.testing.assert_allclose(res.recovered_delta, true_delta, atol=1e-6)

    # Recover with realistic measurement noise (0.5 mm)
    noise = rng.normal(0.0, 0.0005, size=pos_clean.shape)
    res_noisy = recover_marker_offsets(
        factor=factor,
        q=q,
        observed_positions=pos_clean + noise,
    )
    assert res_noisy.observable_fraction == 1.0
    # Stated tolerance: within 1 mm (0.001 m)
    np.testing.assert_allclose(res_noisy.recovered_delta, true_delta, atol=1e-3)


def test_marker_attachment_factor_preconditions() -> None:
    chain = PlanarChain(
        link_lengths=np.array([0.8, 0.5]),
        actuated=np.array([True, True]),
    )
    nominal_offsets = np.array([[0.4, 0.0], [0.2, 0.0]])

    # Negative link index
    with pytest.raises(ContractViolationError):
        MarkerAttachmentFactor(
            chain=chain,
            link_indices=np.array([-1, 0]),
            nominal_offsets=nominal_offsets,
        )

    # Link index >= n_links
    with pytest.raises(ContractViolationError):
        MarkerAttachmentFactor(
            chain=chain,
            link_indices=np.array([0, 5]),
            nominal_offsets=nominal_offsets,
        )

    # Mismatched lengths
    with pytest.raises(ContractViolationError):
        MarkerAttachmentFactor(
            chain=chain,
            link_indices=np.array([0]),
            nominal_offsets=nominal_offsets,
        )
