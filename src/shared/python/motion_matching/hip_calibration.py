"""Functional hip-joint calibration of a full-body specification from the capture.

The Rajagopal lower limbs are attached to the native pelvis (the ``Hip``
frame on the LowerTorso body) by a rigid pelvis alignment. When that
alignment is wrong the legs hang from the wrong points and no marker fit can
recover. This module estimates each hip joint centre functionally: the knee
marker of a leg moves on a sphere about the hip centre when expressed in the
pelvis frame (the pelvis pose per frame comes from the qualified waist marker
offsets), so a linear sphere fit gives the centre and the radius. The
anatomical pelvis axes follow from the two centres (right axis) and the pelvis
frame's superior direction (up), and the hip joints of a specification are rewritten so
that they sit at the functional centres with those axes. Everything else in
the document is untouched; the result is a new document with its own hash.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.marker_calibration import (
    rigid_pose_from_markers,
)

Array: TypeAlias = NDArray[np.float64]


@dataclass(frozen=True)
class SphereFit:
    """Least-squares sphere through a point cloud."""

    centre: tuple[float, float, float]
    radius_m: float
    residual_sd_m: float
    points: int


def fit_sphere(points: Array) -> SphereFit:
    """Algebraic least-squares sphere fit. Precondition: >= 4 finite points."""
    x = np.asarray(points, dtype=float)
    if x.ndim != 2 or x.shape[1] != 3 or x.shape[0] < 4 or not np.isfinite(x).all():
        raise ValueError("Sphere fit needs at least four finite 3-D points")
    design = np.column_stack([2.0 * x, np.ones(x.shape[0])])
    solution = np.linalg.lstsq(design, np.sum(x * x, axis=1), rcond=None)[0]
    centre = solution[:3]
    radius = float(np.sqrt(max(solution[3] + centre @ centre, 0.0)))
    distances = np.linalg.norm(x - centre, axis=1)
    return SphereFit(
        (float(centre[0]), float(centre[1]), float(centre[2])),
        radius,
        float(np.std(distances)),
        int(x.shape[0]),
    )


@dataclass(frozen=True)
class HipCalibration:
    """Functional hip centres in the pelvis (``Hip``) frame and the pelvis axes."""

    centre_r: tuple[float, float, float]
    centre_l: tuple[float, float, float]
    radius_r_m: float
    radius_l_m: float
    residual_sd_r_m: float
    residual_sd_l_m: float
    frames: int
    pelvis_axes: tuple[tuple[float, float, float], ...]  # columns: forward, up, right
    waist_fit_max_residual_m: float


@dataclass(frozen=True)
class HipRotationZero:
    """Zero-twist rotation angle offsets in degrees for right and left hips."""

    offset_r_deg: float
    offset_l_deg: float

    @property
    def r(self) -> float:
        return self.offset_r_deg

    @property
    def l(self) -> float:  # noqa: E743
        return self.offset_l_deg

    def __getitem__(self, key: str) -> float:
        if key == "r":
            return self.offset_r_deg
        if key == "l":
            return self.offset_l_deg
        raise KeyError(f"Invalid side {key!r}; expected 'r' or 'l'")

    def __iter__(self):
        yield self.offset_r_deg
        yield self.offset_l_deg

    def to_dict(self) -> dict[str, float]:
        return {"r": self.offset_r_deg, "l": self.offset_l_deg}


def functional_hip_calibration(
    points: Array,
    valid: NDArray[Any],
    labels: Sequence[str],
    waist_offsets: Mapping[str, Sequence[float]],
    *,
    knee_labels: tuple[str, str] = ("RKneeOut", "LKneeOut"),
    superior_axis: Sequence[float] = (0.0, 0.0, 1.0),
) -> HipCalibration:
    """Estimate both hip centres and the anatomical pelvis axes in the pelvis frame.

    ``waist_offsets`` are the qualified offsets of the waist markers in the
    pelvis frame; the pelvis pose per frame is the rigid fit of those offsets
    to the captured waist markers. ``superior_axis`` is the pelvis frame's own
    up direction (the native Hip frame carries the spine along +z); the
    anatomical up axis is that direction made orthogonal to the hip-to-hip
    line, and forward completes the right-handed (forward, up, right) triad.
    Preconditions: points (frames, markers, 3) with validity, at least three
    waist labels present, both knee labels present, at least four frames with
    a complete pelvis and knee, a finite nonzero superior axis not parallel to
    the hip line. Postcondition: ``pelvis_axes`` is a proper rotation (columns
    forward, up, right in the pelvis frame) and the right axis points from
    the left to the right hip centre.
    """
    pts = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    labels = tuple(labels)
    if pts.ndim != 3 or pts.shape[2] != 3 or mask.shape != pts.shape[:2]:
        raise ValueError("Points must be (frames, markers, 3) with matching validity")
    waist = [label for label in waist_offsets if label in labels]
    if len(waist) < 3:
        raise ValueError("At least three waist markers with offsets are required")
    if any(label not in labels for label in knee_labels):
        raise ValueError("Both knee labels must be capture labels")
    waist_cols = [labels.index(label) for label in waist]
    offsets = np.asarray([waist_offsets[label] for label in waist], dtype=float)
    knee_cols = [labels.index(label) for label in knee_labels]
    local: list[list[Array]] = [[], []]
    residual = 0.0
    for f in range(pts.shape[0]):
        if not mask[f, waist_cols].all():
            continue
        rotation, translation = rigid_pose_from_markers(offsets, pts[f, waist_cols])
        residual = max(
            residual,
            float(
                np.abs(offsets @ rotation.T + translation - pts[f, waist_cols]).max()
            ),
        )
        for side, col in enumerate(knee_cols):
            if mask[f, col]:
                local[side].append(rotation.T @ (pts[f, col] - translation))
    if min(len(local[0]), len(local[1])) < 4:
        raise ValueError("Too few frames with a complete pelvis and knee marker")
    fit_r, fit_l = fit_sphere(np.array(local[0])), fit_sphere(np.array(local[1]))
    c_r, c_l = np.array(fit_r.centre), np.array(fit_l.centre)
    right = c_r - c_l
    if np.linalg.norm(right) < 1e-6:
        raise ValueError("Hip centres coincide")
    right /= np.linalg.norm(right)
    up_raw = np.asarray(superior_axis, dtype=float)
    if up_raw.shape != (3,) or not np.isfinite(up_raw).all():
        raise ValueError("Superior axis must be a finite 3-vector")
    up = up_raw - (up_raw @ right) * right
    if np.linalg.norm(up) < 1e-6:
        raise ValueError("Superior axis must not be parallel to the hip line")
    up /= np.linalg.norm(up)
    forward = np.cross(up, right)
    axes = np.column_stack([forward, up, right])
    return HipCalibration(
        centre_r=fit_r.centre,
        centre_l=fit_l.centre,
        radius_r_m=fit_r.radius_m,
        radius_l_m=fit_l.radius_m,
        residual_sd_r_m=fit_r.residual_sd_m,
        residual_sd_l_m=fit_l.residual_sd_m,
        frames=min(fit_r.points, fit_l.points),
        pelvis_axes=(
            (float(axes[0, 0]), float(axes[1, 0]), float(axes[2, 0])),
            (float(axes[0, 1]), float(axes[1, 1]), float(axes[2, 1])),
            (float(axes[0, 2]), float(axes[1, 2]), float(axes[2, 2])),
        ),
        waist_fit_max_residual_m=residual,
    )


def hip_rotation_zero(
    points: Array,
    valid: NDArray[Any],
    labels: Sequence[str],
    waist_offsets: Mapping[str, Sequence[float]],
    *,
    knee_out_labels: tuple[str, str] = ("RKneeOut", "LKneeOut"),
    knee_in_labels: tuple[str, str] = ("RKneeIn", "LKneeIn"),
    ankle_out_labels: tuple[str, str] = ("RAnkleOut", "LAnkleOut"),
    calibration: HipCalibration | None = None,
    superior_axis: Sequence[float] = (0.0, 0.0, 1.0),
    max_frames: int = 24,
) -> HipRotationZero:
    """Estimate the zero-twist rotation angle offsets (deg) for right and left hips.

    The angle measures the rotation of the thigh around its longitudinal axis in
    the anatomical pelvis frame (X forward, Y up, Z right). For each leg:
    - If a medial knee marker is available, the mediolateral axis is formed
      directly from the knee-in to knee-out vector.
    - If only the lateral knee marker is available, the mediolateral axis is
      estimated as the normal to the knee flexion plane formed by the functional
      hip centre, the lateral knee, and the lateral ankle marker.

    Preconditions:
    - points must be (frames, markers, 3) with matching boolean validity.
    - at least three waist markers with offsets are required.
    - both knee_out_labels must be present in labels.
    - either medial knee markers or lateral ankle markers must be present in labels.
    - at least one frame must have valid waist and knee markers.
    """
    pts = np.asarray(points, dtype=float)
    mask = np.asarray(valid, dtype=bool)
    labels_tuple = tuple(labels)
    if pts.ndim != 3 or pts.shape[2] != 3 or mask.shape != pts.shape[:2]:
        raise ValueError("Points must be (frames, markers, 3) with matching validity")
    waist = [label for label in waist_offsets if label in labels_tuple]
    if len(waist) < 3:
        raise ValueError("At least three waist markers with offsets are required")
    if any(label not in labels_tuple for label in knee_out_labels):
        raise ValueError("Both knee_out_labels must be capture labels")

    has_knee_in = all(label in labels_tuple for label in knee_in_labels)
    has_ankle_out = all(label in labels_tuple for label in ankle_out_labels)
    if not has_knee_in and not has_ankle_out:
        raise ValueError(
            "Either medial knee markers or lateral ankle markers must be present"
        )

    if calibration is None:
        calibration = functional_hip_calibration(
            points,
            valid,
            labels_tuple,
            waist_offsets,
            knee_labels=knee_out_labels,
            superior_axis=superior_axis,
        )

    axes = np.array(calibration.pelvis_axes).T
    waist_cols = [labels_tuple.index(label) for label in waist]
    offsets = np.asarray([waist_offsets[label] for label in waist], dtype=float)
    centres = (np.array(calibration.centre_r), np.array(calibration.centre_l))

    k_out_cols = [labels_tuple.index(label) for label in knee_out_labels]
    k_in_cols = (
        [labels_tuple.index(label) for label in knee_in_labels] if has_knee_in else None
    )
    a_out_cols = (
        [labels_tuple.index(label) for label in ankle_out_labels]
        if has_ankle_out
        else None
    )

    angles_r: list[float] = []
    angles_l: list[float] = []
    n_frames = min(pts.shape[0], max_frames) if max_frames > 0 else pts.shape[0]

    for f in range(n_frames):
        if not mask[f, waist_cols].all():
            continue
        rotation, translation = rigid_pose_from_markers(offsets, pts[f, waist_cols])

        for side, angles_list in enumerate((angles_r, angles_l)):
            k_out_idx = k_out_cols[side]
            if not mask[f, k_out_idx]:
                continue
            p_k_out = rotation.T @ (pts[f, k_out_idx] - translation)

            if k_in_cols is not None and mask[f, k_in_cols[side]]:
                p_k_in = rotation.T @ (pts[f, k_in_cols[side]] - translation)
                v_lat = axes.T @ (p_k_out - p_k_in)
                if side == 0:
                    theta = float(np.degrees(np.arctan2(v_lat[0], v_lat[2])))
                else:
                    theta = float(np.degrees(np.arctan2(-v_lat[0], -v_lat[2])))
                angles_list.append(theta)
            elif a_out_cols is not None and mask[f, a_out_cols[side]]:
                p_a_out = rotation.T @ (pts[f, a_out_cols[side]] - translation)
                c_hip = centres[side]
                v_thigh = axes.T @ (p_k_out - c_hip)
                v_shank = axes.T @ (p_a_out - p_k_out)
                if side == 0:
                    v_lat = np.cross(v_shank, v_thigh)
                    theta = float(np.degrees(np.arctan2(v_lat[0], v_lat[2])))
                else:
                    v_lat = np.cross(v_thigh, v_shank)
                    theta = float(np.degrees(np.arctan2(-v_lat[0], -v_lat[2])))
                angles_list.append(theta)

    if not angles_r or not angles_l:
        raise ValueError("Insufficient valid frames to determine hip zero twist")

    return HipRotationZero(
        offset_r_deg=float(np.mean(angles_r)),
        offset_l_deg=float(np.mean(angles_l)),
    )


def _matrix(value: Any) -> Array:
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("Expected a finite 4x4 transform")
    return m


def apply_hip_calibration(
    document: Mapping[str, Any],
    calibration: HipCalibration,
    hip_from_pelvis_old: Any,
    *,
    hip_frame: str = "Hip",
    zero_twist_deg: (
        HipRotationZero | Mapping[str, float] | tuple[float, float] | None
    ) = None,
) -> dict[str, Any]:
    """Return a copy of ``document`` with both hip joints at the functional centres.

    ``hip_from_pelvis_old`` is the 4x4 pelvis alignment the document was built
    with (OpenSim pelvis frame -> pelvis ``Hip`` frame). Each hip joint's
    parent transform ``P = H A_old X`` (``H`` the Hip frame placement on the
    pelvis body, ``X`` the OpenSim-side joint frame) is rewritten as
    ``H A_new X`` where ``A_new`` carries the calibrated pelvis axes and places
    the joint centre at the functional centre. When ``zero_twist_deg`` is
    provided, each joint's ``parent_to_base`` is post-multiplied by a rotation
    about the joint's longitudinal twist axis ($R_z(\\theta)$) to zero the
    measured anatomical thigh rotation. Postcondition: the new parent
    transforms have their translations at ``H @ centre`` and the rest of the
    document is unchanged.
    """
    frames = {f["name"]: f for f in document["frames"]}
    if hip_frame not in frames:
        raise ValueError(f"Document has no frame named {hip_frame}")
    h = _matrix(frames[hip_frame]["placement"])
    a_old = _matrix(hip_from_pelvis_old)
    axes = np.array(calibration.pelvis_axes).T
    if abs(np.linalg.det(axes) - 1.0) > 1e-9:
        raise ValueError("Pelvis axes must form a proper rotation")

    twist_r = 0.0
    twist_l = 0.0
    if zero_twist_deg is not None:
        if isinstance(zero_twist_deg, HipRotationZero):
            twist_r = zero_twist_deg.offset_r_deg
            twist_l = zero_twist_deg.offset_l_deg
        elif isinstance(zero_twist_deg, Mapping):
            twist_r = float(zero_twist_deg["r"])
            twist_l = float(zero_twist_deg["l"])
        elif isinstance(zero_twist_deg, (tuple, list)) and len(zero_twist_deg) == 2:
            twist_r = float(zero_twist_deg[0])
            twist_l = float(zero_twist_deg[1])
        else:
            raise ValueError(f"Unsupported zero_twist_deg type: {type(zero_twist_deg)}")

    joints = []
    replaced = 0
    for joint in document["joints"]:
        entry = dict(joint)
        if joint["name"] in ("hip_r", "hip_l"):
            is_r = joint["name"] == "hip_r"
            centre = np.array(calibration.centre_r if is_r else calibration.centre_l)
            old = _matrix(joint["parent_to_base"])
            x = np.linalg.inv(a_old) @ np.linalg.inv(h) @ old
            a_new = np.eye(4)
            a_new[:3, :3] = axes
            a_new[:3, 3] = centre - axes @ x[:3, 3]
            new = h @ a_new @ x
            twist = twist_r if is_r else twist_l
            if twist != 0.0:
                rad = np.radians(twist)
                c, s = np.cos(rad), np.sin(rad)
                rz = np.array(
                    [
                        [c, -s, 0.0, 0.0],
                        [s, c, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                )
                new = new @ rz
            entry["parent_to_base"] = new.tolist()
            replaced += 1
        joints.append(entry)
    if replaced != 2:
        raise ValueError("Document must contain hip_r and hip_l joints")
    out = dict(document)
    out["joints"] = joints
    if zero_twist_deg is not None:
        subject = dict(out.get("subject", {}))
        subject["hip_zero_twist_deg"] = {
            "r": round(twist_r, 2),
            "l": round(twist_l, 2),
        }
        out["subject"] = subject
    provenance = (
        str(document.get("provenance", ""))
        + " | hip joints relocated to functional centres from the capture "
        f"(radii {calibration.radius_r_m:.4f}/{calibration.radius_l_m:.4f} m, "
        f"sphere sd {calibration.residual_sd_r_m * 1e3:.1f}/"
        f"{calibration.residual_sd_l_m * 1e3:.1f} mm over {calibration.frames} frames)"
    )
    if zero_twist_deg is not None:
        provenance += (
            f" with zero twist calibrated (R: {twist_r:.2f} deg, L: {twist_l:.2f} deg)"
        )
    out["provenance"] = provenance
    return out
