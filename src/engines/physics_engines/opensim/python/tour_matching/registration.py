"""Capture-to-world rigid registration and canonical golf camera views (OG-04, #10398).

Provides:
1. Capture-to-World Rigid Registration:
   - Evaluates unit conversion and vertical-axis alignment.
   - Computes proper 3D rigid transforms (R, t) with det(R) == +1 (no reflections).
   - Inversion operations that satisfy round-trip identity to <= 1e-8 m.
   - Ground plane registration (placing lowest stance markers at Y = 0).
   - Target line registration (aligning target stance along +X target direction).
2. Golf Camera Views:
   - Canonical orthogonal / perspective viewpoints: FRONT_VIEW, SIDE_VIEW, DOWN_THE_LINE, OVERHEAD.
   - Distinct viewer camera setups that decouple viewing projection from model states and kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from src.shared.python.body_part_viz.fitters._kabsch import kabsch_rotation
from src.shared.python.contracts import ensure, require
from src.shared.python.motion_matching.tour_capture_contract import TourCapture


class CameraPreset(str, Enum):
    """Canonical golf camera perspective presets."""

    FRONT_VIEW = "front_view"  # Face-on (standing in front of golfer looking at chest)
    SIDE_VIEW = "side_view"  # Side / profile view
    DOWN_THE_LINE = "down_the_line"  # Behind golfer looking down the target line (+X)
    OVERHEAD = "overhead"  # Top-down looking down (-Y)


@dataclass(frozen=True)
class GolfCameraView:
    """Camera specification for rendering or visualization."""

    preset: CameraPreset
    position: NDArray[np.float64]
    target: NDArray[np.float64]
    up: NDArray[np.float64]
    fov_deg: float = 45.0

    def __post_init__(self) -> None:
        pos = np.asarray(self.position, dtype=np.float64)
        tgt = np.asarray(self.target, dtype=np.float64)
        up = np.asarray(self.up, dtype=np.float64)
        require(pos.shape == (3,), "position must be 3D vector")
        require(tgt.shape == (3,), "target must be 3D vector")
        require(up.shape == (3,), "up must be 3D vector")
        require(bool(np.isfinite(pos).all()), "position coordinates must be finite")
        require(bool(np.isfinite(tgt).all()), "target coordinates must be finite")
        require(bool(np.isfinite(up).all()), "up coordinates must be finite")
        norm_up = float(math.sqrt(np.vdot(up, up)))  # Bolt optimization
        require(norm_up > 1e-6, "up vector cannot be degenerate zero")
        object.__setattr__(self, "position", pos)
        object.__setattr__(self, "target", tgt)
        object.__setattr__(self, "up", up / norm_up)


@dataclass(frozen=True)
class CaptureRegistration:
    """Rigid 3D transformation (R, t) mapping source points into target frame.

    Formula: target_point = source_point @ rotation.T + translation
    """

    rotation: NDArray[np.float64]
    translation: NDArray[np.float64]
    source_frame: str = "capture"
    target_frame: str = "world"

    def __post_init__(self) -> None:
        R = np.asarray(self.rotation, dtype=np.float64)
        t = np.asarray(self.translation, dtype=np.float64)
        require(R.shape == (3, 3), f"rotation must have shape (3, 3), got {R.shape}")
        require(t.shape == (3,), f"translation must have shape (3,), got {t.shape}")
        require(bool(np.isfinite(R).all()), "rotation matrix elements must be finite")
        require(
            bool(np.isfinite(t).all()), "translation vector elements must be finite"
        )

        det = float(np.linalg.det(R))
        require(
            math.isclose(det, 1.0, rel_tol=1e-5),
            f"Rotation matrix must be a proper rotation with det(R) == +1; got det = {det:.6f} (reflections rejected)",
        )
        is_ortho = np.allclose(R @ R.T, np.eye(3), atol=1e-6)
        require(is_ortho, "Rotation matrix must be orthogonal (R @ R.T == I)")

        object.__setattr__(self, "rotation", R)
        object.__setattr__(self, "translation", t)

    def inverse(self) -> CaptureRegistration:
        """Compute the exact inverse transform: source = (target - t) @ R."""
        R_inv = self.rotation.T
        t_inv = -R_inv @ self.translation
        return CaptureRegistration(
            rotation=R_inv,
            translation=t_inv,
            source_frame=self.target_frame,
            target_frame=self.source_frame,
        )


def register_points(
    points: NDArray[np.float64], registration: CaptureRegistration
) -> NDArray[np.float64]:
    """Transform points using the given rigid registration."""
    pts = np.asarray(points, dtype=np.float64)
    require(
        pts.ndim in (1, 2, 3) and pts.shape[-1] == 3,
        f"Points must have last dimension 3; got {pts.shape}",
    )
    transformed = pts @ registration.rotation.T + registration.translation
    return transformed


def compute_capture_registration(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
    *,
    source_frame: str = "capture",
    target_frame: str = "world",
) -> CaptureRegistration:
    """Compute optimal proper 3D rigid transform (R, t) mapping source to target via Kabsch.

    Preconditions:
    - Matching shapes (N, 3) with N >= 3.
    - Points must not be collinear or degenerate.
    """
    p = np.asarray(source_points, dtype=np.float64)
    q = np.asarray(target_points, dtype=np.float64)
    require(p.shape == q.shape, f"Shape mismatch: {p.shape} vs {q.shape}")
    require(p.ndim == 2 and p.shape[1] == 3, f"Points must be (N, 3); got {p.shape}")
    require(
        p.shape[0] >= 3, f"At least 3 point correspondences required; got {p.shape[0]}"
    )
    require(bool(np.isfinite(p).all()), "source points must be finite")
    require(bool(np.isfinite(q).all()), "target points must be finite")

    pc = np.mean(p, axis=0)
    qc = np.mean(q, axis=0)
    p_centered = p - pc
    q_centered = q - qc

    # Degeneracy check: rank of covariance / points
    _, s_p, _ = np.linalg.svd(p_centered)
    if s_p[1] < 1e-6:
        raise ValueError(
            "Source points are collinear or degenerate; cannot compute 3D rotation"
        )

    R = kabsch_rotation(p_centered, q_centered)
    t = qc - R @ pc

    reg = CaptureRegistration(
        rotation=R,
        translation=t,
        source_frame=source_frame,
        target_frame=target_frame,
    )
    ensure(
        math.isclose(float(np.linalg.det(reg.rotation)), 1.0, rel_tol=1e-5),
        "Rotation must be proper",
    )
    return reg


def align_tour_capture_to_golf_world(
    capture: TourCapture,
    *,
    ground_labels: Sequence[str] = (
        "LToeIn",
        "RToeIn",
        "LToeOut",
        "RToeOut",
        "LAnkleOut",
        "RAnkleOut",
    ),
    stance_left_labels: Sequence[str] = ("WaistLeft", "LAnkleOut"),
    stance_right_labels: Sequence[str] = ("WaistRight", "RAnkleOut"),
) -> tuple[TourCapture, CaptureRegistration]:
    """Align raw tour capture into canonical golf world coordinates.

    World coordinate conventions:
    - +X: Forward along target line (toward flag / target direction).
    - +Y: Vertical Up.
    - +Z: Lateral (toward golfer's trail / rear side for right-handed golfer).

    Alignment steps:
    1. Ground registration: Shift coordinates so that the ground support plane
       (lowest vertical coordinate of foot contact markers at address frame 0) rests at Y = 0.
    2. Stance yaw orientation: Rotate around vertical axis Y so that the golfer stands
       with the target line pointing along +X and chest facing -Z.
    """
    pts0 = capture.points_m[0]
    valid0 = capture.valid[0]

    # Ground alignment: find min Y across valid ground support markers at frame 0
    ground_indices = [
        capture.index(lbl)
        for lbl in ground_labels
        if lbl in capture.labels and valid0[capture.index(lbl)]
    ]
    require(
        len(ground_indices) > 0, "No valid ground support markers at address frame 0"
    )
    ground_y = float(np.min(pts0[ground_indices, 1]))

    # Midpoint of waist at address frame 0
    w_l = (
        pts0[capture.index("WaistLeft")]
        if "WaistLeft" in capture.labels and valid0[capture.index("WaistLeft")]
        else None
    )
    w_r = (
        pts0[capture.index("WaistRight")]
        if "WaistRight" in capture.labels and valid0[capture.index("WaistRight")]
        else None
    )
    if w_l is not None and w_r is not None:
        pelvis_origin = 0.5 * (w_l + w_r)
    else:
        valid_pts = pts0[valid0]
        pelvis_origin = np.mean(valid_pts, axis=0)

    # Stance lateral axis: vector from Right to Left hip/waist
    if w_l is not None and w_r is not None:
        lateral_vec = w_l - w_r
        lateral_vec[1] = 0.0  # project onto horizontal plane
        norm_lat = float(
            math.sqrt(np.vdot(lateral_vec, lateral_vec))
        )  # Bolt optimization
        if norm_lat > 1e-4:
            lateral_dir = lateral_vec / norm_lat
            # For a right-handed golfer at address facing the ball:
            # Stance vector (WaistRight -> WaistLeft) points approximately along target direction (+X)
            yaw_angle = math.atan2(lateral_dir[2], lateral_dir[0])
            # Desired: lateral_dir aligned with +X (yaw = 0)
            theta_y = -yaw_angle
        else:
            theta_y = 0.0
    else:
        theta_y = 0.0

    # Construct rigid rotation around Y
    cos_t = math.cos(theta_y)
    sin_t = math.sin(theta_y)
    R_align = np.array(
        [
            [cos_t, 0.0, sin_t],
            [0.0, 1.0, 0.0],
            [-sin_t, 0.0, cos_t],
        ],
        dtype=np.float64,
    )

    # Translation to center pelvis over origin in X and Z, and place feet on Y=0
    # First apply rotation to center
    rot_pelvis = pelvis_origin @ R_align.T
    t_align = np.array([-rot_pelvis[0], -ground_y, -rot_pelvis[2]], dtype=np.float64)

    registration = CaptureRegistration(
        rotation=R_align,
        translation=t_align,
        source_frame="capture",
        target_frame="world_aligned",
    )

    # Apply registration to all frames of TourCapture
    aligned_points = np.zeros_like(capture.points_m)
    for f in range(capture.frames):
        v = capture.valid[f]
        if np.any(v):
            aligned_points[f, v] = register_points(capture.points_m[f, v], registration)
        aligned_points[f, ~v] = np.nan

    aligned_capture = TourCapture(
        time_s=capture.time_s,
        labels=capture.labels,
        points_m=aligned_points,
        valid=capture.valid,
        source_sha256=capture.source_sha256,
    )

    return aligned_capture, registration


def get_golf_camera_view(
    preset: CameraPreset,
    *,
    target_pos: Sequence[float] | None = None,
    distance_m: float = 3.0,
) -> GolfCameraView:
    """Return canonical GolfCameraView for a given preset.

    Conventions (with golfer centered at target):
    - FRONT_VIEW (Face-on): Camera at -Z looking toward +Z at golfer's chest.
    - SIDE_VIEW: Camera looking laterally along -X at golfer.
    - DOWN_THE_LINE: Camera at -X looking toward +X along the target line.
    - OVERHEAD: Camera at +Y looking straight down (-Y).
    """
    tgt = np.array(
        target_pos if target_pos is not None else [0.0, 1.0, 0.0], dtype=np.float64
    )

    if preset == CameraPreset.FRONT_VIEW:
        pos = tgt + np.array([0.0, 0.1, -distance_m], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif preset == CameraPreset.SIDE_VIEW:
        pos = tgt + np.array([distance_m, 0.1, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif preset == CameraPreset.DOWN_THE_LINE:
        pos = tgt + np.array([-distance_m, 0.2, 0.0], dtype=np.float64)
        up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    elif preset == CameraPreset.OVERHEAD:
        pos = tgt + np.array([0.0, distance_m, 0.0], dtype=np.float64)
        up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
        raise ValueError(f"Unknown camera preset: {preset}")

    return GolfCameraView(
        preset=preset,
        position=pos,
        target=tgt,
        up=up,
        fov_deg=45.0,
    )
