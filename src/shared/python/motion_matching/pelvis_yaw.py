"""Pelvis yaw orientation metrics, residuals, and analytic Jacobians.

Provides robust 2-component unit-direction difference formulation without
reversal singularities (eliminating the sin(delta_yaw) zero at 180 degrees),
along with exact analytic Jacobian and metric reporting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PelvisYawMetrics:
    """Metrics describing pelvis yaw orientation alignment."""

    yaw_pred_deg: float
    yaw_target_deg: float
    yaw_diff_deg: float
    pelvis_yaw_error_pct: float
    valid: bool


def _invalid_pelvis_yaw_metrics(
    yaw_pred: float = np.nan,
    yaw_target: float = np.nan,
) -> PelvisYawMetrics:
    return PelvisYawMetrics(
        yaw_pred_deg=yaw_pred,
        yaw_target_deg=yaw_target,
        yaw_diff_deg=np.nan,
        pelvis_yaw_error_pct=np.nan,
        valid=False,
    )


def compute_pelvis_yaw_metrics(
    pred_markers_term: Array | None,
    target_markers_term: Array | None,
    wl_i: int,
    wr_i: int,
    tolerance: float = 1e-6,
    target_valid: npt.NDArray[np.bool_] | None = None,
    pred_valid: npt.NDArray[np.bool_] | None = None,
) -> PelvisYawMetrics:
    """Compute pelvis yaw angles in degrees and percentage error.

    Parameters
    ----------
    pred_markers_term : Array or None
        (K, 3) marker positions predicted at terminal frame.
    target_markers_term : Array or None
        (K, 3) target marker positions at terminal frame.
    wl_i : int
        WaistLeft marker index.
    wr_i : int
        WaistRight marker index.
    tolerance : float
        Minimum norm for planar separation vector.
    target_valid : npt.NDArray[np.bool_] or None
        Optional (K,) boolean validity mask for target markers.
    pred_valid : npt.NDArray[np.bool_] or None
        Optional (K,) boolean validity mask for predicted markers.

    Returns
    -------
    PelvisYawMetrics
        Angles in degrees, wrapped difference in [-180, 180], and error percentage.
    """
    if (
        pred_markers_term is None
        or target_markers_term is None
        or pred_markers_term.ndim < 2
        or target_markers_term.ndim < 2
        or pred_markers_term.shape[0] <= max(wl_i, wr_i)
        or target_markers_term.shape[0] <= max(wl_i, wr_i)
        or wl_i < 0
        or wr_i < 0
        or wl_i == wr_i
    ):
        return _invalid_pelvis_yaw_metrics()

    # Check predicted marker validity
    yaw_pred: float = np.nan
    yaw_pred_valid = False
    if pred_valid is not None:
        if (
            pred_valid.ndim >= 1
            and pred_valid.shape[0] > max(wl_i, wr_i)
            and bool(pred_valid[wl_i])
            and bool(pred_valid[wr_i])
        ):
            yaw_pred_valid = True
    else:
        yaw_pred_valid = True

    if yaw_pred_valid:
        v_p = pred_markers_term[wr_i, :2] - pred_markers_term[wl_i, :2]
        norm_p = float(np.linalg.norm(v_p))
        if np.isfinite(norm_p) and norm_p >= tolerance:
            yaw_pred = float(np.degrees(np.arctan2(v_p[1], v_p[0])))
        else:
            yaw_pred_valid = False

    # Check target marker validity
    yaw_target: float = np.nan
    yaw_target_valid = False
    if target_valid is not None:
        if (
            target_valid.ndim >= 1
            and target_valid.shape[0] > max(wl_i, wr_i)
            and bool(target_valid[wl_i])
            and bool(target_valid[wr_i])
        ):
            yaw_target_valid = True
    else:
        yaw_target_valid = True

    if yaw_target_valid:
        v_t = target_markers_term[wr_i, :2] - target_markers_term[wl_i, :2]
        norm_t = float(np.linalg.norm(v_t))
        if np.isfinite(norm_t) and norm_t >= tolerance:
            yaw_target = float(np.degrees(np.arctan2(v_t[1], v_t[0])))
        else:
            yaw_target_valid = False

    if not yaw_pred_valid or not yaw_target_valid:
        return _invalid_pelvis_yaw_metrics(yaw_pred, yaw_target)

    diff_deg = float((yaw_pred - yaw_target + 180.0) % 360.0 - 180.0)
    error_pct = float(abs(diff_deg) / max(abs(yaw_target), 1.0) * 100.0)

    return PelvisYawMetrics(
        yaw_pred_deg=yaw_pred,
        yaw_target_deg=yaw_target,
        yaw_diff_deg=diff_deg,
        pelvis_yaw_error_pct=error_pct,
        valid=True,
    )


def compute_pelvis_yaw_residual_and_derivative(
    pred_markers_term: Array | None,
    target_markers_term: Array | None,
    wl_i: int,
    wr_i: int,
    yaw_weight: float,
    marker_jac_term: Array | None = None,
    tolerance: float = 1e-6,
    target_valid: npt.NDArray[np.bool_] | None = None,
    pred_valid: npt.NDArray[np.bool_] | None = None,
) -> tuple[Array, Array | None, PelvisYawMetrics]:
    """Compute 2-component unit vector difference residual and analytic Jacobian.

    Residual:
        r_yaw = w_yaw * (u_p - u_t) in R^2
        where u_p = v_p / ||v_p||, u_t = v_t / ||v_t|| in the horizontal xy-plane.

    Properties:
        - Unique minimum at delta_yaw = 0 (r = 0).
        - No reversal singularity at 180 deg: ||u_p - u_t|| = 2 (maximum penalty).
        - Exact derivative:
            du_p / dv_p = (1 / ||v_p||) * (I_2 - u_p u_p^T)
            dr_yaw / dp = w_yaw * (du_p / dv_p) * (J_wr - J_wl) in R^(2 x N_p)

    Parameters
    ----------
    pred_markers_term : Array or None
        (K, 3) predicted marker positions at terminal frame.
    target_markers_term : Array or None
        (K, 3) target marker positions at terminal frame.
    wl_i : int
        WaistLeft index.
    wr_i : int
        WaistRight index.
    yaw_weight : float
        Scalar weight.
    marker_jac_term : Array or None
        (K, 3, N_p) marker Jacobian at terminal frame.
    tolerance : float
        Minimum vector norm before declaring degeneracy.
    target_valid : npt.NDArray[np.bool_] or None
        Optional (K,) boolean validity mask for target markers.
    pred_valid : npt.NDArray[np.bool_] or None
        Optional (K,) boolean validity mask for predicted markers.

    Returns
    -------
    residual : (2,) Array
        2-component planar residual. Always exactly 2 elements.
    jacobian : (2, N_p) Array or None
        Planar Jacobian block if marker_jac_term provided, else None.
    metrics : PelvisYawMetrics
        Descriptive orientation angles and error percentage.
    """
    n_p = marker_jac_term.shape[-1] if marker_jac_term is not None else 0
    zero_jac = (
        np.zeros((2, n_p), dtype=np.float64) if marker_jac_term is not None else None
    )

    metrics = compute_pelvis_yaw_metrics(
        pred_markers_term,
        target_markers_term,
        wl_i,
        wr_i,
        tolerance=tolerance,
        target_valid=target_valid,
        pred_valid=pred_valid,
    )

    if not metrics.valid or pred_markers_term is None or target_markers_term is None:
        return np.zeros(2, dtype=np.float64), zero_jac, metrics

    v_p = pred_markers_term[wr_i, :2] - pred_markers_term[wl_i, :2]
    v_t = target_markers_term[wr_i, :2] - target_markers_term[wl_i, :2]
    norm_p = float(np.linalg.norm(v_p))
    norm_t = float(np.linalg.norm(v_t))

    u_p = v_p / norm_p
    u_t = v_t / norm_t

    residual = yaw_weight * (u_p - u_t)

    jacobian: Array | None = None
    if marker_jac_term is not None:
        if (
            marker_jac_term.ndim != 3
            or marker_jac_term.shape[0] <= max(wl_i, wr_i)
            or marker_jac_term.shape[1] < 2
        ):
            jacobian = zero_jac
        else:
            d_wr = marker_jac_term[wr_i, :2, :]
            d_wl = marker_jac_term[wl_i, :2, :]
            dv_p = d_wr - d_wl
            proj = (np.eye(2) - np.outer(u_p, u_p)) / norm_p
            jacobian = yaw_weight * (proj @ dv_p)

    return residual, jacobian, metrics
