"""Articulated golfer silhouette rendering bound to subject morphology and kinematics (ST-05, #10232).

This module implements:
- `ArticulatedSilhouetteRenderer`: Fulfills `SilhouetteRenderer` protocol by binding
  subject model visual envelope, camera geometry, and canonical articulated state vectors
  evaluated via forward kinematics across full body (spine, torso, arms, hands, thighs, shins, feet).
- `CANONICAL_ARTICULATED_STATE_FIELDS`: 37-element ordered fields for full canonical state layout.
- `state_vector_from_joint_dict`: Helper mapping joint angle dictionary to state tuple.
- `state_vector_to_joint_dict`: Helper mapping state tuple to joint angle dictionary.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import numpy as np

from src.shared.python.motion_matching.diagnostics.forward_kinematics import (
    SegmentLengths,
    SkeletonPose,
    forward_kinematics,
)
from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)

from .contracts import (
    CANONICAL_ARTICULATED_CONVENTION,
    RenderRequest,
    RenderResult,
    SilhouetteRenderer,
    SubjectModelBinding,
)
from .projection import (
    PinholeCameraModel,
    _rasterize_ellipse,
    project_point_to_pixel,
    resolve_camera_and_dimensions,
)

# 37 canonical articulated state fields: 3 translation + 24 reference golfer fields + 10 lower-body fields
CANONICAL_ARTICULATED_STATE_FIELDS: tuple[str, ...] = (
    (
        "TranslationStartPositionX",
        "TranslationStartPositionY",
        "TranslationStartPositionZ",
    )
    + REFERENCE_GOLFER_FIELDS
    + (
        "LHipStartPositionX",
        "LHipStartPositionY",
        "LHipStartPositionZ",
        "RHipStartPositionX",
        "RHipStartPositionY",
        "RHipStartPositionZ",
        "LKneeStartPosition",
        "RKneeStartPosition",
        "LAnkleStartPosition",
        "RAnkleStartPosition",
    )
)

_CANONICAL_FIELD_COUNT = len(CANONICAL_ARTICULATED_STATE_FIELDS)

# Canonical body segments: (p1_name, p2_name, nominal_radius_m)
_BODY_SEGMENTS: tuple[tuple[str, str, float], ...] = (
    ("pelvis", "spine_top", 0.14),
    ("spine_top", "torso_top", 0.16),
    ("torso_top", "l_shoulder", 0.07),
    ("torso_top", "r_shoulder", 0.07),
    ("l_shoulder", "l_elbow", 0.06),
    ("r_shoulder", "r_elbow", 0.06),
    ("l_elbow", "l_wrist", 0.05),
    ("r_elbow", "r_wrist", 0.05),
    ("l_wrist", "l_hand", 0.04),
    ("r_wrist", "r_hand", 0.04),
    # Lower limb segments: thighs, shins, feet
    ("pelvis", "l_hip", 0.09),
    ("pelvis", "r_hip", 0.09),
    ("l_hip", "l_knee", 0.08),
    ("r_hip", "r_knee", 0.08),
    ("l_knee", "l_ankle", 0.06),
    ("r_knee", "r_ankle", 0.06),
    ("l_ankle", "l_foot", 0.05),
    ("r_ankle", "r_foot", 0.05),
)


def state_vector_from_joint_dict(
    angles: Mapping[str, float],
    *,
    translation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, ...]:
    """Construct a canonical state tuple from a joint angle mapping."""
    vals = [
        float(angles.get("TranslationStartPositionX", translation[0])),
        float(angles.get("TranslationStartPositionY", translation[1])),
        float(angles.get("TranslationStartPositionZ", translation[2])),
    ]
    for field in CANONICAL_ARTICULATED_STATE_FIELDS[3:]:
        vals.append(float(angles.get(field, 0.0)))
    return tuple(vals)


def state_vector_to_joint_dict(state: tuple[float, ...]) -> dict[str, float]:
    """Extract joint angles and translation mapping from a canonical state tuple."""
    if len(state) == _CANONICAL_FIELD_COUNT:
        return {
            field: float(val)
            for field, val in zip(
                CANONICAL_ARTICULATED_STATE_FIELDS, state, strict=True
            )
        }
    if len(state) == 27:
        res = {
            field: float(val)
            for field, val in zip(
                CANONICAL_ARTICULATED_STATE_FIELDS[:27], state, strict=True
            )
        }
        for field in CANONICAL_ARTICULATED_STATE_FIELDS[27:]:
            res[field] = 0.0
        return res
    raise ValueError(
        f"Expected {_CANONICAL_FIELD_COUNT} (or 27) elements for canonical articulated state, got {len(state)}"
    )


def _rasterize_3d_segment(
    mask: list[int],
    p1: np.ndarray,
    p2: np.ndarray,
    radius_m: float,
    camera: PinholeCameraModel,
    width: int,
    height: int,
    depth_epsilon: float = 1e-3,
) -> None:
    """Rasterize a 3D swept sphere segment into a 2D mask with bounded viewport work."""
    r_cam = camera.rotation_world_to_camera
    t_cam = camera.translation_world_to_camera

    # Transform to camera coordinates
    z1 = float(r_cam[6] * p1[0] + r_cam[7] * p1[1] + r_cam[8] * p1[2] + t_cam[2])
    z2 = float(r_cam[6] * p2[0] + r_cam[7] * p2[1] + r_cam[8] * p2[2] + t_cam[2])

    if z1 <= depth_epsilon and z2 <= depth_epsilon:
        return

    # Clip segment if one point is behind camera plane
    p1_eff = p1.copy().astype(float)
    p2_eff = p2.copy().astype(float)
    if z1 <= depth_epsilon:
        denom = max(1e-9, (z2 - z1))
        t = min(max((depth_epsilon - z1) / denom, 0.0), 1.0)
        p1_eff = p1_eff + t * (p2_eff - p1_eff)
        z1 = depth_epsilon
    elif z2 <= depth_epsilon:
        denom = max(1e-9, (z1 - z2))
        t = min(max((depth_epsilon - z2) / denom, 0.0), 1.0)
        p2_eff = p2_eff + t * (p1_eff - p2_eff)
        z2 = depth_epsilon

    u1, v1, _ = project_point_to_pixel(
        (float(p1_eff[0]), float(p1_eff[1]), float(p1_eff[2])), camera
    )
    u2, v2, _ = project_point_to_pixel(
        (float(p2_eff[0]), float(p2_eff[1]), float(p2_eff[2])), camera
    )

    rx1 = camera.fx * radius_m / z1
    ry1 = camera.fy * radius_m / z1
    rx2 = camera.fx * radius_m / z2
    ry2 = camera.fy * radius_m / z2

    max_r = max(rx1, ry1, rx2, ry2)
    min_u = min(u1, u2) - max_r
    max_u = max(u1, u2) + max_r
    min_v = min(v1, v2) - max_r
    max_v = max(v1, v2) + max_r
    if max_u < 0 or min_u >= width or max_v < 0 or min_v >= height:
        return

    dist_px = math.hypot(u2 - u1, v2 - v1)
    min_r = max(0.5, min(rx1, ry1, rx2, ry2))
    # Bounded sampling work prevents infinite/excessive loops on near-plane crossings
    max_screen_extent = math.hypot(width, height) + 2.0 * max_r
    effective_dist_px = min(dist_px, max_screen_extent * 2.0)
    max_allowed_steps = max(64, int(4 * max(width, height)))
    num_steps = min(
        max(1, int(math.ceil(effective_dist_px / max(1.0, min_r * 0.75)))),
        max_allowed_steps,
    )

    for step in range(num_steps + 1):
        alpha = float(step) / float(num_steps)
        p_curr = (1.0 - alpha) * p1_eff + alpha * p2_eff
        zc = (1.0 - alpha) * z1 + alpha * z2
        if zc <= depth_epsilon:
            continue
        uc, vc, _ = project_point_to_pixel(
            (float(p_curr[0]), float(p_curr[1]), float(p_curr[2])), camera
        )
        rx = camera.fx * radius_m / zc
        ry = camera.fy * radius_m / zc
        if uc + rx < 0 or uc - rx >= width or vc + ry < 0 or vc - ry >= height:
            continue
        _rasterize_ellipse(mask, uc, vc, rx, ry, width, height)


class ArticulatedSilhouetteRenderer:
    """Slotted renderer binding articulated golfer states to silhouette masks (ST-05, #10232).

    Fulfills `SilhouetteRenderer` protocol:
    - Binds `SubjectModelBinding`, `PinholeCameraModel`, and kinematic lengths.
    - Evaluates forward kinematics via `motion_matching.diagnostics.forward_kinematics`.
    - Rasterizes articulated body segments (pelvis, spine, torso, shoulders, arms, hands) into `body_mask`.
    - Rasterizes club shaft and clubhead into `club_mask`.
    - Accurately clips geometry extending beyond image borders without dropping partially visible limbs.
    - Accurately projects anamorphic cameras ($f_x \\neq f_y$) as ellipses.
    - Enforces camera effective dimension parity against request image sizes.
    - Strictly rejects unsupported or unrecognised state conventions.
    """

    __slots__ = (
        "_cameras",
        "_subject_binding",
        "_segment_lengths",
        "_club_radius_m",
        "_body_scale",
    )

    def __init__(
        self,
        cameras: Mapping[str, PinholeCameraModel],
        subject_binding: SubjectModelBinding,
        *,
        segment_lengths: SegmentLengths | None = None,
        club_radius_m: float = 0.05,
    ) -> None:
        self._cameras = dict(cameras)
        if not isinstance(subject_binding, SubjectModelBinding):
            raise TypeError(
                f"Expected SubjectModelBinding, got {type(subject_binding).__name__}"
            )
        self._subject_binding = subject_binding
        env = subject_binding.visual_envelope
        # Reference golfer height is 1.78 m (SI units)
        nominal_height_m = float(env.get("height_m", 1.78))
        self._body_scale = nominal_height_m / 1.78
        if segment_lengths is not None:
            self._segment_lengths = segment_lengths
        else:
            scale = self._body_scale
            self._segment_lengths = SegmentLengths(
                pelvis_to_spine=0.20 * scale,
                spine_to_torso=0.20 * scale,
                torso_to_shoulder=0.18 * scale,
                upper_arm=0.30 * scale,
                forearm=0.27 * scale,
                hand=0.10 * scale,
                club_shaft=1.10,
                pelvis_to_hip=0.10 * scale,
                thigh=0.44 * scale,
                shin=0.42 * scale,
                foot=0.18 * scale,
            )
        if club_radius_m <= 0.0 or not math.isfinite(club_radius_m):
            raise ValueError(f"club_radius_m must be positive, got {club_radius_m}")
        self._club_radius_m = float(club_radius_m)

    @property
    def subject_binding(self) -> SubjectModelBinding:
        return self._subject_binding

    @property
    def club_radius_m(self) -> float:
        return self._club_radius_m

    def _validate_request(
        self, request: RenderRequest
    ) -> tuple[PinholeCameraModel, int, int]:
        camera, width, height = resolve_camera_and_dimensions(self._cameras, request)

        if request.state_convention != CANONICAL_ARTICULATED_CONVENTION:
            raise ValueError(
                f"ArticulatedSilhouetteRenderer requires state_convention={CANONICAL_ARTICULATED_CONVENTION!r}, "
                f"got {request.state_convention!r}"
            )

        if len(request.state) not in (_CANONICAL_FIELD_COUNT, 27):
            raise ValueError(
                f"ArticulatedSilhouetteRenderer requires {_CANONICAL_FIELD_COUNT} (or 27) state elements for "
                f"{CANONICAL_ARTICULATED_CONVENTION!r}, got {len(request.state)}"
            )

        return camera, width, height

    def _rasterize_body(
        self,
        body_mask: list[int],
        pts: Mapping[str, np.ndarray],
        camera: PinholeCameraModel,
        width: int,
        height: int,
    ) -> None:
        for p1_name, p2_name, base_r in _BODY_SEGMENTS:
            r = base_r * self._body_scale
            p1 = pts[p1_name]
            p2 = pts[p2_name]
            _rasterize_3d_segment(body_mask, p1, p2, r, camera, width, height)

        torso_top = pts["torso_top"]
        spine_top = pts["spine_top"]
        head_vec = torso_top - spine_top
        head_norm = float(np.linalg.norm(head_vec))
        if head_norm > 1e-6:
            head_dir = head_vec / head_norm
        else:
            head_dir = np.array([0.0, 0.0, 1.0])
        head_center = torso_top + 0.15 * self._body_scale * head_dir
        _rasterize_3d_segment(
            body_mask,
            torso_top,
            head_center,
            0.10 * self._body_scale,
            camera,
            width,
            height,
        )

    def _rasterize_club(
        self,
        club_mask: list[int],
        pts: Mapping[str, np.ndarray],
        camera: PinholeCameraModel,
        width: int,
        height: int,
    ) -> None:
        butt = pts["butt"]
        clubhead = pts["clubhead"]
        _rasterize_3d_segment(club_mask, butt, clubhead, 0.015, camera, width, height)

        r_cam = camera.rotation_world_to_camera
        t_cam = camera.translation_world_to_camera
        zc_head = float(
            r_cam[6] * clubhead[0]
            + r_cam[7] * clubhead[1]
            + r_cam[8] * clubhead[2]
            + t_cam[2]
        )
        if zc_head > 1e-6:
            cu, cv, _ = project_point_to_pixel(
                (float(clubhead[0]), float(clubhead[1]), float(clubhead[2])),
                camera,
            )
            rx_head = camera.fx * self._club_radius_m / zc_head
            ry_head = camera.fy * self._club_radius_m / zc_head
            _rasterize_ellipse(club_mask, cu, cv, rx_head, ry_head, width, height)

    def render(self, request: RenderRequest) -> RenderResult:
        """Render articulated body and club silhouettes from canonical state vector."""
        camera, width, height = self._validate_request(request)

        angles = state_vector_to_joint_dict(request.state)
        pose = forward_kinematics(
            angles,
            lengths=self._segment_lengths,
            include_lower_body=True,
        )
        pts = pose.points

        total_px = width * height
        body_mask = [0] * total_px
        club_mask = [0] * total_px
        vis_mask = [1] * total_px

        self._rasterize_body(body_mask, pts, camera, width, height)
        self._rasterize_club(club_mask, pts, camera, width, height)

        return RenderResult(
            body_mask=tuple(body_mask),
            club_mask=tuple(club_mask),
            visibility_mask=tuple(vis_mask),
        )
