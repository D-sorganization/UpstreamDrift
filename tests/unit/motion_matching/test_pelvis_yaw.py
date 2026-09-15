"""Unit tests for pelvis yaw orientation metrics, residuals, and analytic Jacobians."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_matching.pelvis_yaw import (
    PelvisYawMetrics,
    compute_pelvis_yaw_metrics,
    compute_pelvis_yaw_residual_and_derivative,
)

pytestmark = [pytest.mark.unit]


def test_pelvis_yaw_metrics_exact_angles() -> None:
    """Verify angle calculation and wrapping."""
    # Target: along +X axis (yaw = 0 deg)
    target = np.zeros((2, 3))
    target[0] = [0.0, 0.0, 0.0]  # WaistLeft
    target[1] = [1.0, 0.0, 0.0]  # WaistRight

    # Pred: along +Y axis (yaw = 90 deg)
    pred = np.zeros((2, 3))
    pred[0] = [0.0, 0.0, 0.0]
    pred[1] = [0.0, 1.0, 0.0]

    metrics = compute_pelvis_yaw_metrics(pred, target, wl_i=0, wr_i=1)
    assert metrics.valid
    assert np.isclose(metrics.yaw_pred_deg, 90.0)
    assert np.isclose(metrics.yaw_target_deg, 0.0)
    assert np.isclose(metrics.yaw_diff_deg, 90.0)

    # Pred: along -X axis (yaw = 180 deg)
    pred[1] = [-1.0, 0.0, 0.0]
    metrics180 = compute_pelvis_yaw_metrics(pred, target, wl_i=0, wr_i=1)
    assert metrics180.valid
    assert np.isclose(abs(metrics180.yaw_diff_deg), 180.0)


def test_pelvis_yaw_no_180_reversal_singularity() -> None:
    """Verify that 180 degree reversal has non-zero residual and maximum penalty."""
    target = np.zeros((2, 3))
    target[0] = [0.0, 0.0, 0.0]
    target[1] = [1.0, 0.0, 0.0]  # u_t = [1, 0]

    pred = np.zeros((2, 3))
    pred[0] = [0.0, 0.0, 0.0]
    pred[1] = [-1.0, 0.0, 0.0]  # u_p = [-1, 0] (180 deg reversed)

    w = 50.0
    res, _, _ = compute_pelvis_yaw_residual_and_derivative(
        pred, target, wl_i=0, wr_i=1, yaw_weight=w
    )
    assert res.shape == (2,)
    # u_p - u_t = [-1, 0] - [1, 0] = [-2, 0]
    # residual = 50 * [-2, 0] = [-100, 0]
    np.testing.assert_allclose(res, [-100.0, 0.0])
    assert np.linalg.norm(res) == pytest.approx(100.0)
    # The old sine residual was 0 at 180 deg. This new residual is strictly non-zero!
    assert np.linalg.norm(res) > 0.0


def test_pelvis_yaw_analytic_jacobian_directional_derivatives() -> None:
    """Verify analytic Jacobian against finite differences along multiple random directions."""
    rng = np.random.default_rng(42)

    n_p = 5
    yaw_weight = 35.0

    target = np.zeros((2, 3))
    target[0] = [0.1, -0.2, 0.0]
    target[1] = [0.4, 0.3, 0.0]

    # Test at multiple non-zero configurations
    for trial in range(10):
        # Generate random base positions and sensitivities
        pred_base = np.zeros((2, 3))
        pred_base[0] = rng.uniform(-0.5, 0.5, size=3)
        # Ensure non-degenerate separation
        offset = rng.uniform(-0.5, 0.5, size=3)
        offset[:2] += np.sign(offset[:2]) * 0.2
        pred_base[1] = pred_base[0] + offset

        # Random Jacobians for WaistLeft and WaistRight w.r.t p in R^n_p
        jac_wl = rng.normal(0, 1.0, size=(3, n_p))
        jac_wr = rng.normal(0, 1.0, size=(3, n_p))

        marker_jac = np.zeros((2, 3, n_p))
        marker_jac[0] = jac_wl
        marker_jac[1] = jac_wr

        # Evaluate analytic residual & Jacobian
        res0, ana_jac, _ = compute_pelvis_yaw_residual_and_derivative(
            pred_base,
            target,
            wl_i=0,
            wr_i=1,
            yaw_weight=yaw_weight,
            marker_jac_term=marker_jac,
        )
        assert ana_jac is not None
        assert ana_jac.shape == (2, n_p)

        # Finite difference check in random directions
        for _ in range(5):
            d = rng.normal(0, 1.0, size=n_p)
            d /= np.linalg.norm(d)
            eps = 1e-7

            # Perturb prediction linearly according to marker_jac @ d
            pred_pert = pred_base.copy()
            pred_pert[0] += eps * (jac_wl @ d)
            pred_pert[1] += eps * (jac_wr @ d)

            res_pert, _, _ = compute_pelvis_yaw_residual_and_derivative(
                pred_pert, target, wl_i=0, wr_i=1, yaw_weight=yaw_weight
            )

            # Numerical directional derivative
            fd_dir = (res_pert - res0) / eps
            ana_dir = ana_jac @ d

            np.testing.assert_allclose(
                ana_dir,
                fd_dir,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Trial {trial}: Directional derivative mismatch",
            )


def test_pelvis_yaw_degenerate_and_missing_markers() -> None:
    """Verify robust behavior with collocated markers, missing markers, and invalid indices."""
    target = np.zeros((2, 3))
    pred = np.zeros((2, 3))

    # Collocated markers (zero separation norm)
    res, jac, m = compute_pelvis_yaw_residual_and_derivative(
        pred,
        target,
        wl_i=0,
        wr_i=1,
        yaw_weight=50.0,
        marker_jac_term=np.zeros((2, 3, 3)),
    )
    assert not m.valid
    assert res.shape == (2,)
    np.testing.assert_allclose(res, 0.0)
    assert jac is not None
    assert jac.shape == (2, 3)
    np.testing.assert_allclose(jac, 0.0)

    # Missing prediction (None)
    res, jac, m = compute_pelvis_yaw_residual_and_derivative(
        None,
        target,
        wl_i=0,
        wr_i=1,
        yaw_weight=50.0,
        marker_jac_term=np.zeros((2, 3, 3)),
    )
    assert not m.valid
    assert res.shape == (2,)
    np.testing.assert_allclose(res, 0.0)
    assert jac is not None
    assert jac.shape == (2, 3)

    # Out of bounds indices
    res, jac, m = compute_pelvis_yaw_residual_and_derivative(
        pred,
        target,
        wl_i=0,
        wr_i=10,
        yaw_weight=50.0,
        marker_jac_term=np.zeros((2, 3, 3)),
    )
    assert not m.valid
    assert res.shape == (2,)
    assert jac is not None
    assert jac.shape == (2, 3)
