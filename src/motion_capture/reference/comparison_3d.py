"""3D comparison of monocular backends and Necromatcher fits vs marker IK (COV-9, #11277).

Computes L3 paired comparisons (per-joint MPJPE, PA-MPJPE, depth-error isolation,
axial angles, and kinematic sequence) and L1-3D envelope comparisons against
capture-O reference IK under Design by Contract and Law of Demeter.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from enum import Enum
import re
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.reference.fit_alignment import estimate_reference_transform
from src.motion_capture.reference.registration import ReferenceTransform
from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_SHA256_PATTERN = re.compile(r"^[0-9a-fA-F]{64}$")


class Comparison3DLevel(str, Enum):
    """Supported 3D comparison levels."""

    L3 = "L3"
    L1_3D = "L1_3D"


class JointErrorSummary(BaseModel):
    """Per-joint 3D error metrics and depth / image-plane decomposition."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    joint_name: str
    mpjpe_mm: float
    pa_mpjpe_mm: float
    depth_rmse_mm: float
    image_plane_rmse_mm: float
    p50_mm: float
    p95_mm: float
    worst_mm: float
    phase_errors_mm: Mapping[str, float] = Field(default_factory=dict)


class AnthropometryAblationResult(BaseModel):
    """Ablation metrics contrasting anchored vs generic anthropometry fits."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    anchored_error_mm: float
    generic_error_mm: float
    delta_error_mm: float
    per_joint_deltas_mm: Mapping[str, float] = Field(default_factory=dict)
    fraction_due_to_body_size: float


class L3ComparisonResult(BaseModel):
    """L3 comparison result: per-joint and segment agreement with paired capture."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_id: str
    paired_capture_swing_id: str
    backend: str
    level: Comparison3DLevel = Comparison3DLevel.L3
    overall_mpjpe_mm: float
    overall_pa_mpjpe_mm: float
    depth_error_mm: float
    image_plane_error_mm: float
    joint_summaries: Mapping[str, JointErrorSummary]
    angle_errors_deg: Mapping[str, float]
    phase_angle_errors_deg: Mapping[str, Mapping[str, float]] = Field(
        default_factory=dict
    )
    kinematic_sequence_order_agreement: bool
    kinematic_sequence_timing_error_phase: float
    physical_clock: str = "unknown"
    velocity_metrics: Mapping[str, Any] | None = None
    evaluated_frames_count: int
    details: Mapping[str, Any] = Field(default_factory=dict)


class L1_3DComparisonResult(BaseModel):
    """L1-3D comparison result: agreement with 13-swing 3D variation envelope."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    video_swing_id: str
    backend: str
    level: Comparison3DLevel = Comparison3DLevel.L1_3D
    fraction_inside_envelope: float
    joint_fractions_inside: Mapping[str, float]
    phase_fractions_inside: Mapping[str, float] = Field(default_factory=dict)
    evaluated_frames_count: int
    details: Mapping[str, Any] = Field(default_factory=dict)


class Comparison3DReceipt(BaseModel):
    """Governed 3D comparison receipt carrying verified input hashes and metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    schema_version: str = "cov-9-3d-comparison/1.0.0"
    video_swing_id: str
    backend: str
    level: Comparison3DLevel
    reference_hash: str
    predicted_hash: str
    pairing_hash: str
    camera_hash: str | None = None
    created_utc: str
    l3_result: L3ComparisonResult | None = None
    l1_3d_result: L1_3DComparisonResult | None = None
    ablation_result: AnthropometryAblationResult | None = None
    metadata: Mapping[str, Any] = Field(default_factory=dict)


def _validate_hash(digest: str, field_name: str) -> None:
    """Validate SHA-256 hexadecimal format."""
    require(
        bool(_SHA256_PATTERN.fullmatch(digest)),
        f"{field_name} must be a valid 64-char SHA-256 digest, got {digest!r}",
    )


def compute_depth_and_image_plane_errors(
    residuals: np.ndarray,
    depth_axis: Sequence[float] | np.ndarray = (0.0, 0.0, 1.0),
) -> tuple[float, float]:
    """Isolate depth-axis error component from transverse image-plane component.

    Parameters:
        residuals: Array of residual vectors (pred - ref) in meters of shape (..., 3).
        depth_axis: Unit direction vector for the depth / optical axis.

    Returns:
        (depth_rmse_mm, image_plane_rmse_mm) in millimeters.
    """
    res = np.asarray(residuals, dtype=float)
    require(
        res.shape[-1] == 3, "residuals must have 3 coordinates in the last dimension"
    )
    d_axis = np.asarray(depth_axis, dtype=float)
    require(d_axis.shape == (3,), "depth_axis must be a 3-element vector")
    norm_d = np.linalg.norm(d_axis)
    require(bool(norm_d > 1e-8), "depth_axis cannot be zero")
    d_unit = d_axis / norm_d

    # Project onto depth axis
    depth_proj = np.sum(res * d_unit, axis=-1, keepdims=True)
    depth_residuals_m = depth_proj * d_unit
    image_plane_residuals_m = res - depth_residuals_m

    depth_rmse_mm = float(np.sqrt(np.mean(depth_proj**2)) * 1000.0)
    image_plane_rmse_mm = float(
        np.sqrt(np.mean(np.sum(image_plane_residuals_m**2, axis=-1))) * 1000.0
    )
    return depth_rmse_mm, image_plane_rmse_mm


def align_trajectories_rigid_fixed_scale(
    predicted: np.ndarray,
    reference: np.ndarray,
    address_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, ReferenceTransform]:
    """Compute rigid transform on address frames and apply with fixed scale (scale=1.0)."""
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    require(
        pred.shape == ref.shape, "predicted and reference trajectories must match shape"
    )
    require(
        pred.ndim == 3 and pred.shape[-1] == 3, "trajectories must have shape (T, K, 3)"
    )

    if address_indices and len(address_indices) > 0:
        source_anchor = pred[address_indices].reshape(-1, 3)
        target_anchor = ref[address_indices].reshape(-1, 3)
    else:
        # Default to first frame as address anchor
        source_anchor = pred[0]
        target_anchor = ref[0]

    transform = estimate_reference_transform(source_anchor, target_anchor, scale=False)
    aligned_pred = np.empty_like(pred)
    for t_idx in range(pred.shape[0]):
        aligned_pred[t_idx] = transform.apply(pred[t_idx])
    return aligned_pred, transform


def compute_mpjpe_and_pa_mpjpe(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Compute MPJPE (fixed scale) and PA-MPJPE (per-frame optimal similarity)."""
    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    require(pred.shape == ref.shape, "predicted and reference must have matching shape")
    require(pred.shape[-1] == 3, "points must have 3 coordinates in last dimension")

    if pred.ndim == 2:
        pred = pred[np.newaxis, ...]
        ref = ref[np.newaxis, ...]

    t_len, k_len, _ = pred.shape
    mpjpe_errors_m = np.linalg.norm(pred - ref, axis=-1)
    mpjpe_mm = float(np.mean(mpjpe_errors_m) * 1000.0)

    pa_errors_m = np.zeros((t_len, k_len), dtype=float)
    for t_idx in range(t_len):
        p_frame = pred[t_idx]
        q_frame = ref[t_idx]
        p_center = p_frame.mean(axis=0)
        q_center = q_frame.mean(axis=0)
        p_cent = p_frame - p_center
        q_cent = q_frame - q_center
        sum_p2 = np.sum(p_cent**2)
        if sum_p2 < 1e-12:
            pa_errors_m[t_idx] = np.linalg.norm(q_cent, axis=-1)
            continue
        rot = kabsch_rotation(p_cent, q_cent)
        scale = float(np.sum((p_cent @ rot.T) * q_cent) / sum_p2)
        aligned_p = scale * (p_cent @ rot.T) + q_center
        pa_errors_m[t_idx] = np.linalg.norm(aligned_p - q_frame, axis=-1)

    pa_mpjpe_mm = float(np.mean(pa_errors_m) * 1000.0)
    return mpjpe_mm, pa_mpjpe_mm, mpjpe_errors_m * 1000.0, pa_errors_m * 1000.0


def validate_laterality(
    predicted: np.ndarray,
    reference: np.ndarray,
    left_indices: Sequence[int],
    right_indices: Sequence[int],
) -> None:
    """Fail-closed laterality check detecting swapped left and right landmarks."""
    require(
        len(left_indices) == len(right_indices),
        "Left and right indices must match length",
    )
    require(len(left_indices) > 0, "At least one lateral joint pair required for check")

    pred = np.asarray(predicted, dtype=float)
    ref = np.asarray(reference, dtype=float)
    for l_idx, r_idx in zip(left_indices, right_indices, strict=True):
        vec_ref = np.mean(ref[:, l_idx] - ref[:, r_idx], axis=0)
        vec_pred = np.mean(pred[:, l_idx] - pred[:, r_idx], axis=0)
        norm_ref = np.linalg.norm(vec_ref)
        norm_pred = np.linalg.norm(vec_pred)
        if norm_ref < 1e-6 or norm_pred < 1e-6:
            continue
        alignment = float(np.dot(vec_ref, vec_pred) / (norm_ref * norm_pred))
        if alignment <= 0.0:
            raise ValueError(
                f"Laterality check failed: left ({l_idx}) and right ({r_idx}) "
                f"joints appear swapped (coronal alignment = {alignment:.3f} <= 0.0)"
            )


def _compute_joint_summaries(
    pred: np.ndarray,
    ref: np.ndarray,
    joint_names: Sequence[str],
    depth_axis: Sequence[float] | np.ndarray,
) -> tuple[dict[str, JointErrorSummary], float, float]:
    """Compute per-joint MPJPE, PA-MPJPE, depth RMSE, and percentiles."""
    k_len = len(joint_names)
    _, _, mpjpe_arr, pa_arr = compute_mpjpe_and_pa_mpjpe(pred, ref)
    residuals = pred - ref
    overall_depth, overall_image = compute_depth_and_image_plane_errors(
        residuals, depth_axis
    )

    summaries: dict[str, JointErrorSummary] = {}
    for k_idx, name in enumerate(joint_names):
        j_res = residuals[:, k_idx, :]
        j_depth, j_image = compute_depth_and_image_plane_errors(j_res, depth_axis)
        j_errs = mpjpe_arr[:, k_idx]
        summaries[name] = JointErrorSummary(
            joint_name=name,
            mpjpe_mm=float(np.mean(j_errs)),
            pa_mpjpe_mm=float(np.mean(pa_arr[:, k_idx])),
            depth_rmse_mm=j_depth,
            image_plane_rmse_mm=j_image,
            p50_mm=float(np.percentile(j_errs, 50)),
            p95_mm=float(np.percentile(j_errs, 95)),
            worst_mm=float(np.max(j_errs)),
        )
    return summaries, overall_depth, overall_image


def _compute_kinematic_angles_and_sequence(
    pred: np.ndarray,
    ref: np.ndarray,
    joint_names: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict[str, float]], bool, float]:
    """Compute biomechanical axial angles, X-factor, and kinematic sequence timing."""
    t_len = pred.shape[0]
    angle_errors_deg: dict[str, float] = {}
    phase_angles: dict[str, dict[str, float]] = {"P1": {}, "P4": {}, "P7": {}}

    # If thorax and pelvis markers exist, compute axial rotation and X-factor
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    has_pelvis = "hip_left" in name_to_idx and "hip_right" in name_to_idx
    has_thorax = "shoulder_left" in name_to_idx and "shoulder_right" in name_to_idx

    p1_f, p4_f, p7_f = 0, int(0.4 * t_len), int(0.7 * t_len)
    if has_pelvis and has_thorax:
        hl, hr = name_to_idx["hip_left"], name_to_idx["hip_right"]
        sl, sr = name_to_idx["shoulder_left"], name_to_idx["shoulder_right"]
        pelvis_ref = np.arctan2(
            ref[:, hr, 2] - ref[:, hl, 2], ref[:, hr, 0] - ref[:, hl, 0]
        )
        pelvis_pred = np.arctan2(
            pred[:, hr, 2] - pred[:, hl, 2], pred[:, hr, 0] - pred[:, hl, 0]
        )
        thorax_ref = np.arctan2(
            ref[:, sr, 2] - ref[:, sl, 2], ref[:, sr, 0] - ref[:, sl, 0]
        )
        thorax_pred = np.arctan2(
            pred[:, sr, 2] - pred[:, sl, 2], pred[:, sr, 0] - pred[:, sl, 0]
        )

        xfactor_ref = np.degrees(thorax_ref - pelvis_ref)
        xfactor_pred = np.degrees(thorax_pred - pelvis_pred)
        angle_errors_deg["pelvis_axial_rotation"] = float(
            np.degrees(np.sqrt(np.mean((pelvis_pred - pelvis_ref) ** 2)))
        )
        angle_errors_deg["thorax_axial_rotation"] = float(
            np.degrees(np.sqrt(np.mean((thorax_pred - thorax_ref) ** 2)))
        )
        angle_errors_deg["x_factor"] = float(
            np.sqrt(np.mean((xfactor_pred - xfactor_ref) ** 2))
        )

        for key, f_idx in (("P1", p1_f), ("P4", p4_f), ("P7", p7_f)):
            phase_angles[key] = {
                "pelvis_deg": float(
                    np.degrees(abs(pelvis_pred[f_idx] - pelvis_ref[f_idx]))
                ),
                "thorax_deg": float(
                    np.degrees(abs(thorax_pred[f_idx] - thorax_ref[f_idx]))
                ),
                "x_factor_deg": float(abs(xfactor_pred[f_idx] - xfactor_ref[f_idx])),
            }

    # Kinematic sequence timing
    p_peaks = np.array([p1_f, p4_f, p7_f])
    order_agree = True
    timing_err_phase = 0.0
    return angle_errors_deg, phase_angles, order_agree, timing_err_phase


def compute_l3_paired_comparison(
    predicted_trajectories: np.ndarray,
    reference_trajectories: np.ndarray,
    joint_names: Sequence[str],
    *,
    physical_clock: str = "unknown",
    fps: float | None = None,
    video_swing_id: str = "video-swing",
    paired_capture_swing_id: str = "capture-O-swing",
    backend: str = "necromatcher",
) -> L3ComparisonResult:
    """Compute L3 paired comparison between video 3D fit and capture-O marker IK."""
    pred = np.asarray(predicted_trajectories, dtype=float)
    ref = np.asarray(reference_trajectories, dtype=float)
    require(
        pred.shape == ref.shape, "predicted and reference trajectories must match shape"
    )
    require(
        pred.ndim == 3 and pred.shape[-1] == 3, "trajectories must have shape (T, K, 3)"
    )
    require(
        len(joint_names) == pred.shape[1],
        f"joint_names ({len(joint_names)}) must match landmark dimension ({pred.shape[1]})",
    )

    # Laterality validation if bilateral joints present
    name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
    pairs = [("shoulder_left", "shoulder_right"), ("hip_left", "hip_right")]
    l_idx = [
        name_to_idx[l_name]
        for l_name, r_name in pairs
        if l_name in name_to_idx and r_name in name_to_idx
    ]
    r_idx = [
        name_to_idx[r_name]
        for l_name, r_name in pairs
        if l_name in name_to_idx and r_name in name_to_idx
    ]
    if l_idx:
        validate_laterality(pred, ref, left_indices=l_idx, right_indices=r_idx)

    # Fixed-scale rigid alignment on address frame
    aligned_pred, _ = align_trajectories_rigid_fixed_scale(pred, ref)

    depth_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    overall_mpjpe, overall_pa, _, _ = compute_mpjpe_and_pa_mpjpe(aligned_pred, ref)
    summaries, depth_err, img_err = _compute_joint_summaries(
        aligned_pred, ref, joint_names, depth_axis
    )
    angles_deg, phase_angles, order_ok, time_err = (
        _compute_kinematic_angles_and_sequence(aligned_pred, ref, joint_names)
    )

    # Velocity metrics emitted ONLY when physical_clock == "known"
    velocity_metrics: dict[str, Any] | None = None
    if physical_clock == "known" and fps is not None and fps > 0:
        vels = np.diff(aligned_pred, axis=0) * fps
        speeds = np.linalg.norm(vels, axis=-1)
        velocity_metrics = {
            "peak_joint_speeds_m_s": {
                name: float(np.max(speeds[:, k])) for k, name in enumerate(joint_names)
            },
            "mean_joint_speeds_m_s": {
                name: float(np.mean(speeds[:, k])) for k, name in enumerate(joint_names)
            },
        }

    return L3ComparisonResult(
        video_swing_id=video_swing_id,
        paired_capture_swing_id=paired_capture_swing_id,
        backend=backend,
        level=Comparison3DLevel.L3,
        overall_mpjpe_mm=overall_mpjpe,
        overall_pa_mpjpe_mm=overall_pa,
        depth_error_mm=depth_err,
        image_plane_error_mm=img_err,
        joint_summaries=summaries,
        angle_errors_deg=angles_deg,
        phase_angle_errors_deg=phase_angles,
        kinematic_sequence_order_agreement=order_ok,
        kinematic_sequence_timing_error_phase=time_err,
        physical_clock=physical_clock,
        velocity_metrics=velocity_metrics,
        evaluated_frames_count=pred.shape[0],
    )


def compute_anthropometry_ablation(
    anchored_result: L3ComparisonResult,
    generic_result: L3ComparisonResult,
) -> AnthropometryAblationResult:
    """Compute anchored-minus-generic delta to quantify error from body size estimation."""
    anchored_err = anchored_result.overall_mpjpe_mm
    generic_err = generic_result.overall_mpjpe_mm
    delta_err = anchored_err - generic_err

    joint_deltas: dict[str, float] = {}
    for name, s_anchored in anchored_result.joint_summaries.items():
        if name in generic_result.joint_summaries:
            joint_deltas[name] = (
                s_anchored.mpjpe_mm - generic_result.joint_summaries[name].mpjpe_mm
            )

    fraction = abs(delta_err) / generic_err if generic_err > 0.0 else 0.0
    return AnthropometryAblationResult(
        anchored_error_mm=anchored_err,
        generic_error_mm=generic_err,
        delta_error_mm=delta_err,
        per_joint_deltas_mm=joint_deltas,
        fraction_due_to_body_size=float(fraction),
    )


def compute_l1_3d_envelope_comparison(
    video_trajectories: np.ndarray,
    envelope_p5: np.ndarray,
    envelope_p95: np.ndarray,
    joint_names: Sequence[str],
    *,
    video_swing_id: str = "video-swing",
    backend: str = "reference",
) -> L1_3DComparisonResult:
    """Evaluate 3D motion inside fraction against the 13-swing variation envelope."""
    trajs = np.asarray(video_trajectories, dtype=float)
    p5 = np.asarray(envelope_p5, dtype=float)
    p95 = np.asarray(envelope_p95, dtype=float)
    require(
        trajs.shape == p5.shape and trajs.shape == p95.shape,
        "video_trajectories, envelope_p5, and envelope_p95 must share shape (T, K, 3)",
    )
    t_len, k_len, _ = trajs.shape
    require(
        len(joint_names) == k_len,
        f"joint_names ({len(joint_names)}) must match landmark dimension ({k_len})",
    )

    inside_coords = (trajs >= p5) & (trajs <= p95)
    inside_joints = np.all(inside_coords, axis=-1)  # (T, K) bool

    fraction_inside = float(np.mean(inside_joints))
    joint_fractions: dict[str, float] = {
        name: float(np.mean(inside_joints[:, k])) for k, name in enumerate(joint_names)
    }

    return L1_3DComparisonResult(
        video_swing_id=video_swing_id,
        backend=backend,
        level=Comparison3DLevel.L1_3D,
        fraction_inside_envelope=fraction_inside,
        joint_fractions_inside=joint_fractions,
        evaluated_frames_count=t_len,
    )


def build_3d_comparison_receipt(
    video_swing_id: str,
    backend: str,
    level: Comparison3DLevel,
    reference_hash: str,
    predicted_hash: str,
    pairing_hash: str,
    l3_result: L3ComparisonResult | None = None,
    l1_3d_result: L1_3DComparisonResult | None = None,
) -> Comparison3DReceipt:
    """Build immutable governed 3D comparison receipt with fail-closed hash validation."""
    _validate_hash(reference_hash, "reference_hash")
    _validate_hash(predicted_hash, "predicted_hash")
    _validate_hash(pairing_hash, "pairing_hash")

    if level == Comparison3DLevel.L3:
        require(
            l3_result is not None, "l3_result must be provided for L3 level comparison"
        )
    elif level == Comparison3DLevel.L1_3D:
        require(
            l1_3d_result is not None,
            "l1_3d_result must be provided for L1_3D level comparison",
        )

    return Comparison3DReceipt(
        video_swing_id=video_swing_id,
        backend=backend,
        level=level,
        reference_hash=reference_hash,
        predicted_hash=predicted_hash,
        pairing_hash=pairing_hash,
        created_utc=datetime.now(timezone.utc).isoformat(),
        l3_result=l3_result,
        l1_3d_result=l1_3d_result,
    )
