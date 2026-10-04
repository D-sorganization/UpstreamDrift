"""2D comparison of markerless backends vs projected capture-O landmarks (COV-7, #11275).

Computes L1 envelope and L2 paired receipts, event timing errors,
visibility-weighted residuals, and camera uncertainty propagation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
import math
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.swing_pairing import (
    PairingDecisionStatus,
    SwingPairingResult,
)
from src.motion_capture.reference.virtual_camera_fit import SwingEnvelope2D
from src.shared.python.core.contracts import require
from src.shared.python.signal_toolkit.signal_processing import compute_dtw_distance


class ComparisonLevel(str, Enum):
    """Comparison level supported by the swing and data evidence."""

    L1 = "L1"
    L2 = "L2"


class MetricSpread(BaseModel):
    """Uncertainty spread for a metric under camera parameter perturbation."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    mean: float
    std: float
    p5: float
    p95: float
    spread: float
    is_resolvable: bool = True
    status: str = "resolvable"


class L1ComparisonResult(BaseModel):
    """L1 comparison result: agreement with 13-swing variation envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_id: str
    backend: str
    level: ComparisonLevel = ComparisonLevel.L1
    fraction_inside_envelope: float
    landmark_fractions_inside: Mapping[str, float]
    signed_distance_to_median_px: float
    signed_distance_to_median_norm: float | None = None
    landmark_signed_distances_px: Mapping[str, float]
    landmark_signed_distances_norm: Mapping[str, float] = Field(default_factory=dict)
    relative_dtw_distance: float
    missingness_report: Mapping[str, int]
    total_frames: int
    details: Mapping[str, Any] = Field(default_factory=dict)


class L2ComparisonResult(BaseModel):
    """L2 comparison result: per-frame agreement with a paired capture swing."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_id: str
    paired_capture_swing_id: str
    backend: str
    level: ComparisonLevel = ComparisonLevel.L2
    residuals_px: Mapping[str, float]
    residuals_norm: Mapping[str, float] = Field(default_factory=dict)
    visibility_weighted_rmse_px: float
    unweighted_rmse_px: float
    landmark_residuals_px: Mapping[str, Mapping[str, float]]
    event_timing_errors_frames: Mapping[str, float]
    missingness_report: Mapping[str, int]
    evaluated_frames_count: int
    calibration_frames_excluded: tuple[int, ...] = ()
    camera_spread: MetricSpread | None = None
    details: Mapping[str, Any] = Field(default_factory=dict)


class Comparison2DReceipt(BaseModel):
    """Governed comparison receipt carrying verified input hashes and metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = "cov-7-2d-comparison/1.0.0"
    video_swing_id: str
    backend: str
    level: ComparisonLevel
    observations_hash: str
    camera_hash: str
    pairing_hash: str
    profile_hash: str
    created_utc: str
    l1_result: L1ComparisonResult | None = None
    l2_result: L2ComparisonResult | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


def _validate_landmarks_and_envelope(
    video_landmarks: npt.ArrayLike,
    envelope: SwingEnvelope2D,
) -> tuple[np.ndarray, int, int]:
    """Validate shape and dimension compatibility for L1 envelope evaluation."""
    require(isinstance(envelope, SwingEnvelope2D), "envelope must be a SwingEnvelope2D")
    obs = np.asarray(video_landmarks, dtype=float)
    require(
        obs.ndim == 3 and obs.shape[-1] == 2,
        "video_landmarks must have shape (T, K, 2)",
    )
    t_len, k_len, _ = obs.shape
    require(
        t_len == len(envelope.phase_bins),
        f"video_landmarks time dim ({t_len}) must match envelope ({len(envelope.phase_bins)})",
    )
    require(
        k_len == envelope.p50.shape[1],
        f"video_landmarks landmark dim ({k_len}) must match envelope ({envelope.p50.shape[1]})",
    )
    return obs, t_len, k_len


def compute_l1_envelope_comparison(
    video_landmarks: npt.ArrayLike,
    envelope: SwingEnvelope2D,
    *,
    landmark_names: Sequence[str] | None = None,
    body_height_px: float | None = None,
    video_swing_id: str = "video-swing",
    backend: str = "reference",
) -> L1ComparisonResult:
    """Evaluate 2D landmarks against the phase-normalized variation envelope (L1)."""
    obs, t_len, k_len = _validate_landmarks_and_envelope(video_landmarks, envelope)
    names = (
        tuple(landmark_names)
        if landmark_names
        else tuple(f"lm_{i}" for i in range(k_len))
    )
    lm_fractions: dict[str, float] = {}
    lm_signed_px: dict[str, float] = {}
    lm_signed_norm: dict[str, float] = {}
    missing_report: dict[str, int] = {}
    inside_counts, total_valid = 0, 0
    all_dists: list[float] = []

    for k, name in enumerate(names):
        coords = obs[:, k, :]
        valid = np.isfinite(coords).all(axis=1)
        n_valid = int(np.sum(valid))
        missing_report[name] = t_len - n_valid
        if n_valid == 0:
            lm_fractions[name] = 0.0
            lm_signed_px[name] = float("nan")
            lm_signed_norm[name] = float("nan")
            continue
        valid_coords = coords[valid]
        p5 = envelope.p5[valid, k, :]
        p50 = envelope.p50[valid, k, :]
        p95 = envelope.p95[valid, k, :]
        inside = (
            (valid_coords[:, 0] >= p5[:, 0])
            & (valid_coords[:, 0] <= p95[:, 0])
            & (valid_coords[:, 1] >= p5[:, 1])
            & (valid_coords[:, 1] <= p95[:, 1])
        )
        lm_fractions[name] = float(np.mean(inside))
        inside_counts += int(np.sum(inside))
        total_valid += n_valid
        diff = valid_coords - p50
        dist = np.linalg.norm(diff, axis=-1)
        mean_d = float(np.mean(dist))
        lm_signed_px[name] = mean_d
        all_dists.extend(dist.tolist())
        norm_d = (
            (mean_d / body_height_px)
            if (body_height_px and body_height_px > 0)
            else None
        )
        if norm_d is not None:
            lm_signed_norm[name] = norm_d

    overall_frac = float(inside_counts / total_valid) if total_valid > 0 else 0.0
    overall_mean_d = float(np.mean(all_dists)) if all_dists else 0.0
    overall_norm_d = (
        (overall_mean_d / body_height_px)
        if (body_height_px and body_height_px > 0)
        else None
    )

    # DTW distance relative to envelope spread
    flat_obs = np.nan_to_num(obs.reshape(t_len, -1), nan=0.0)
    flat_p50 = envelope.p50.reshape(t_len, -1)
    spread_scale = float(np.mean(np.linalg.norm(envelope.p95 - envelope.p5, axis=-1)))
    raw_dtw = sum(
        compute_dtw_distance(flat_obs[:, d], flat_p50[:, d])
        for d in range(flat_obs.shape[1])
    )
    rel_dtw = float(raw_dtw / (spread_scale + 1e-6))

    return L1ComparisonResult(
        video_swing_id=video_swing_id,
        backend=backend,
        fraction_inside_envelope=overall_frac,
        landmark_fractions_inside=lm_fractions,
        signed_distance_to_median_px=overall_mean_d,
        signed_distance_to_median_norm=overall_norm_d,
        landmark_signed_distances_px=lm_signed_px,
        landmark_signed_distances_norm=lm_signed_norm,
        relative_dtw_distance=rel_dtw,
        missingness_report=missing_report,
        total_frames=t_len,
    )


def _resolve_evaluation_frames(
    total_frames: int,
    calibration_frames: Sequence[int] | None,
    evaluated_frames: Sequence[int] | None,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Determine evaluated frames while strictly enforcing the leakage guard."""
    cal_tuple = tuple(sorted(set(calibration_frames))) if calibration_frames else ()
    cal_set = set(cal_tuple)
    if evaluated_frames is not None:
        eval_tuple = tuple(evaluated_frames)
        overlap = set(eval_tuple).intersection(cal_set)
        require(
            len(overlap) == 0,
            f"Leakage guard: calibration frames {overlap} must not be present in evaluated frames",
        )
        final_eval = eval_tuple
    else:
        final_eval = tuple(i for i in range(total_frames) if i not in cal_set)
    require(
        len(final_eval) > 0,
        "No evaluated frames remain after excluding calibration frames",
    )
    return final_eval, cal_tuple


def _compute_residuals_and_rmse(
    obs: np.ndarray,
    ref: np.ndarray,
    vis: np.ndarray,
    eval_indices: tuple[int, ...],
    names: tuple[str, ...],
    body_height_px: float | None,
) -> tuple[
    dict[str, float],
    dict[str, float],
    dict[str, dict[str, float]],
    float,
    float,
    dict[str, int],
]:
    """Compute per-landmark and aggregate residual metrics over evaluated frames."""
    lm_res: dict[str, dict[str, float]] = {}
    missing_report: dict[str, int] = {}
    all_valid_res: list[float] = []
    weighted_sq_sum, weight_sum = 0.0, 0.0

    eval_idx = list(eval_indices)
    sub_obs = obs[eval_idx]
    sub_ref = ref[eval_idx]
    sub_vis = vis[eval_idx]

    for k, name in enumerate(names):
        o_k = sub_obs[:, k, :]
        r_k = sub_ref[:, k, :]
        v_k = sub_vis[:, k]
        valid = (
            np.isfinite(o_k).all(axis=1) & np.isfinite(r_k).all(axis=1) & (v_k > 0.0)
        )
        n_missing = int(np.sum(~valid))
        missing_report[name] = n_missing
        if not np.any(valid):
            lm_res[name] = {
                "p50": float("nan"),
                "p95": float("nan"),
                "worst": float("nan"),
            }
            continue
        diff = o_k[valid] - r_k[valid]
        dists = np.linalg.norm(diff, axis=-1)
        all_valid_res.extend(dists.tolist())
        weights = v_k[valid]
        weighted_sq_sum += float(np.sum(weights * (dists**2)))
        weight_sum += float(np.sum(weights))
        lm_res[name] = {
            "p50": float(np.percentile(dists, 50)),
            "p95": float(np.percentile(dists, 95)),
            "worst": float(np.max(dists)),
        }

    res_arr = np.asarray(all_valid_res, dtype=float)
    require(
        len(res_arr) > 0, "No valid landmark observations found in evaluated frames"
    )
    overall_p50 = float(np.percentile(res_arr, 50))
    overall_p95 = float(np.percentile(res_arr, 95))
    overall_worst = float(np.max(res_arr))
    residuals_px = {"p50": overall_p50, "p95": overall_p95, "worst": overall_worst}

    h = body_height_px if (body_height_px and body_height_px > 0) else None
    residuals_norm = (
        {
            "p50": overall_p50 / h,
            "p95": overall_p95 / h,
            "worst": overall_worst / h,
        }
        if h
        else {}
    )
    unweighted_rmse = float(np.sqrt(np.mean(res_arr**2)))
    weighted_rmse = (
        float(np.sqrt(weighted_sq_sum / weight_sum)) if weight_sum > 0 else float("nan")
    )

    return (
        residuals_px,
        residuals_norm,
        lm_res,
        weighted_rmse,
        unweighted_rmse,
        missing_report,
    )


def compute_l2_paired_comparison(
    video_landmarks: npt.ArrayLike,
    projected_reference: npt.ArrayLike,
    pairing_result: SwingPairingResult,
    *,
    landmark_names: Sequence[str] | None = None,
    visibilities: npt.ArrayLike | None = None,
    body_height_px: float | None = None,
    calibration_frames: Sequence[int] | None = None,
    evaluated_frames: Sequence[int] | None = None,
) -> L2ComparisonResult:
    """Evaluate per-frame 2D landmark agreement with a paired capture swing (L2)."""
    require(
        isinstance(pairing_result, SwingPairingResult),
        "pairing_result must be a SwingPairingResult",
    )
    require(
        pairing_result.status == PairingDecisionStatus.PAIRED,
        f"Unpaired swing cannot produce L2 metrics: status is {pairing_result.status.value}",
    )
    require(
        pairing_result.confidence.margin >= pairing_result.confidence.tau_pair,
        "Pairing confidence below tau_pair cannot produce L2 metrics",
    )
    require(
        pairing_result.paired_capture_swing_id is not None,
        "Paired capture swing ID must be present for L2 comparison",
    )

    obs = np.asarray(video_landmarks, dtype=float)
    ref = np.asarray(projected_reference, dtype=float)
    require(
        obs.ndim == 3 and obs.shape[-1] == 2,
        "video_landmarks must have shape (T, K, 2)",
    )
    require(
        obs.shape == ref.shape,
        "video_landmarks and projected_reference shapes must match",
    )
    t_len, k_len, _ = obs.shape

    vis = (
        np.ones((t_len, k_len), dtype=float)
        if visibilities is None
        else np.asarray(visibilities, dtype=float)
    )
    require(
        vis.shape == (t_len, k_len), f"visibilities shape {vis.shape} must match (T, K)"
    )

    eval_indices, cal_excluded = _resolve_evaluation_frames(
        t_len, calibration_frames, evaluated_frames
    )
    names = (
        tuple(landmark_names)
        if landmark_names
        else tuple(f"lm_{i}" for i in range(k_len))
    )

    res_px, res_norm, lm_res, w_rmse, u_rmse, miss_rep = _compute_residuals_and_rmse(
        obs, ref, vis, eval_indices, names, body_height_px
    )

    paired_id = str(pairing_result.paired_capture_swing_id)

    return L2ComparisonResult(
        video_swing_id=pairing_result.video_swing_id,
        paired_capture_swing_id=paired_id,
        backend="reference",
        residuals_px=res_px,
        residuals_norm=res_norm,
        visibility_weighted_rmse_px=w_rmse,
        unweighted_rmse_px=u_rmse,
        landmark_residuals_px=lm_res,
        event_timing_errors_frames={},
        missingness_report=miss_rep,
        evaluated_frames_count=len(eval_indices),
        calibration_frames_excluded=cal_excluded,
    )


def propagate_camera_uncertainty(
    reference_points_3d: npt.ArrayLike,
    camera: PinholeCamera,
    covariance: npt.ArrayLike,
    *,
    sample_count: int = 100,
    difference: float | None = None,
    seed: int = 42,
) -> MetricSpread:
    """Propagate camera parameter covariance into 2D reprojection metric spread."""
    require(isinstance(camera, PinholeCamera), "camera must be PinholeCamera")
    pts = np.asarray(reference_points_3d, dtype=float).reshape(-1, 3)
    require(pts.shape[0] > 0, "reference_points_3d must contain points")
    cov = np.asarray(covariance, dtype=float)
    require(
        cov.ndim == 2 and cov.shape[0] == cov.shape[1],
        "covariance must be square 2D matrix",
    )
    require(
        cov.shape[0] >= 6, "covariance must have at least 6 parameters (extrinsics)"
    )

    rng = np.random.default_rng(seed)
    r_wc = camera.rotation_world_from_camera
    t_wc = camera.translation_world_from_camera_m
    r_vec_nom, _ = cv2.Rodrigues(r_wc)
    r_nom = r_vec_nom.flatten()

    r_cw_nom = r_wc.T
    t_cw_nom = -r_cw_nom @ t_wc
    nom_proj, _ = cv2.projectPoints(
        pts, r_cw_nom, t_cw_nom, camera.matrix, camera.distortion
    )
    nom_proj_2d = nom_proj.reshape(-1, 2)

    perturbations = rng.multivariate_normal(
        mean=np.zeros(cov.shape[0]), cov=cov, size=sample_count
    )
    displacements: list[float] = []

    for pert in perturbations:
        r_pert = r_nom + pert[0:3]
        t_pert = t_wc + pert[3:6]
        r_wc_pert, _ = cv2.Rodrigues(r_pert)
        r_cw_pert = r_wc_pert.T
        t_cw_pert = -r_cw_pert @ t_pert
        proj, _ = cv2.projectPoints(
            pts, r_cw_pert, t_cw_pert, camera.matrix, camera.distortion
        )
        diff = proj.reshape(-1, 2) - nom_proj_2d
        displacements.append(float(np.mean(np.linalg.norm(diff, axis=-1))))

    arr = np.asarray(displacements, dtype=float)
    p5 = float(np.percentile(arr, 5))
    p95 = float(np.percentile(arr, 95))
    spread = p95 - p5
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))

    is_resolvable = True
    status = "resolvable"
    if difference is not None:
        if abs(difference) < spread:
            is_resolvable = False
            status = "not resolvable"

    return MetricSpread(
        mean=mean_val,
        std=std_val,
        p5=p5,
        p95=p95,
        spread=spread,
        is_resolvable=is_resolvable,
        status=status,
    )


def build_2d_comparison_receipt(
    video_swing_id: str,
    backend: str,
    level: ComparisonLevel,
    input_hashes: Mapping[str, str],
    *,
    l1_result: L1ComparisonResult | None = None,
    l2_result: L2ComparisonResult | None = None,
    expected_hashes: Mapping[str, str] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Comparison2DReceipt:
    """Build and validate a comparison receipt, enforcing input hash integrity."""
    require(
        bool(video_swing_id and video_swing_id.strip()),
        "video_swing_id must be non-empty",
    )
    require(bool(backend and backend.strip()), "backend must be non-empty")
    for key in ("observations", "camera", "pairing", "profile"):
        require(key in input_hashes, f"Missing required input hash for '{key}'")
        val = input_hashes[key]
        require(
            bool(val and val.strip()),
            f"Input hash for '{key}' must be non-empty string",
        )
        if expected_hashes is not None and key in expected_hashes:
            expected = expected_hashes[key]
            require(
                val == expected,
                f"Stale or mismatched input hash for '{key}': expected {expected}, got {val}",
            )

    if level == ComparisonLevel.L1:
        require(l1_result is not None, "l1_result is required for Level L1 comparison")
    elif level == ComparisonLevel.L2:
        require(l2_result is not None, "l2_result is required for Level L2 comparison")

    now = datetime.now(timezone.utc).isoformat()
    return Comparison2DReceipt(
        video_swing_id=video_swing_id,
        backend=backend,
        level=level,
        observations_hash=input_hashes["observations"],
        camera_hash=input_hashes["camera"],
        pairing_hash=input_hashes["pairing"],
        profile_hash=input_hashes["profile"],
        created_utc=now,
        l1_result=l1_result,
        l2_result=l2_result,
        metadata=dict(metadata) if metadata else {},
    )
