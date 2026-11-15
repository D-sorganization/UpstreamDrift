"""Wonik Allegro 4-finger visual hand backend (BSD-2-Clause Menagerie meshes, #15152)."""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.grip_pose import Hand
from src.shared.python.model_appearance.hand_backends.allegro_settings import (
    _DIGIT_TAG,
    AllegroHandVisualSettings,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.allegro_solver import (
    _baseline_qpos,
    _canonical_part_key,
    _compute_mount_matrix,
    _forward_model,
    _hand_from_side,
    _iter_visible_geoms,
    _load_model,
    _radial_xy,
    _require_mujoco,
    _scale_vector,
    _select_part_vertices,
    _transform_to_grip_frame,
    _validate_grip_radius,
    apply_joint_angles,
    solve_allegro_hand_qpos,
)
from src.shared.python.model_appearance.hand_backends.types import (
    Array,
    HandSide,
    HandVisualResult,
    JointAngles,
)


def _resolve_call(
    grip_spec: GripSpec,
    side: HandSide,
    grip_radius_m: float | None,
) -> tuple[AllegroHandVisualSettings, Hand, HandGripSpec, float]:
    cfg = visual_settings()
    hand = _hand_from_side(side)
    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    radius = _validate_grip_radius(
        cfg.grip_radius_nominal_m if grip_radius_m is None else grip_radius_m, cfg
    )
    return cfg, hand, spec_hand, radius


def _pose_and_mount(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: JointAngles,
    *,
    grip_radius_m: float | None = None,
    zero_qpos: bool = False,
) -> tuple[AllegroHandVisualSettings, Hand, Array, Array]:
    cfg, hand, spec_hand, radius = _resolve_call(grip_spec, side, grip_radius_m)
    handedness = grip_spec.handedness
    if zero_qpos:
        model = _load_model(hand == Hand.TRAIL, handedness)
        qpos = _baseline_qpos(model)
        scale = _scale_vector(grip_spec.hand_size, cfg)
        mount = np.eye(4)
        mount[:3, :3] = np.diag(scale)
        return cfg, hand, qpos, mount
    if joint_angles is None:
        solved = solve_allegro_hand_qpos(
            hand, spec_hand, grip_spec=grip_spec, grip_radius_m=radius, settings=cfg
        )
        return cfg, hand, solved.qpos, solved.mount_matrix

    mujoco = _require_mujoco()
    model = _load_model(hand == Hand.TRAIL, handedness)
    data = mujoco.MjData(model)
    qpos = apply_joint_angles(_baseline_qpos(model), model, joint_angles)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    base_mount = _compute_mount_matrix(
        model,
        data,
        hand=hand,
        spec_hand=spec_hand,
        grip_spec=grip_spec,
        grip_radius_m=radius,
        cfg=cfg,
    )
    r_mat = np.asarray(
        apply_shaft_roll(
            np.eye(3),
            APPEARANCE_GRIP_SHAFT_AXIS,
            resolved_shaft_roll_deg(grip_spec, side),
        ),
        dtype=np.float64,
    )
    mount = base_mount.copy()
    mount[:3, :3] = r_mat @ base_mount[:3, :3]
    mount[:3, 3] = r_mat @ base_mount[:3, 3]
    return cfg, hand, qpos, mount


def allegro_mount_determinant(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
) -> float:
    """Return linear mount determinant (must stay positive for proper orientation)."""
    _cfg, _hand, _qpos, mount = _pose_and_mount(
        grip_spec, side, None, grip_radius_m=grip_radius_m
    )
    return float(np.linalg.det(mount[:3, :3]))


def extract_allegro_hand_parts(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
    zero_qpos: bool = False,
    joint_angles: JointAngles = None,
) -> dict[str, Mesh]:
    """Extract per-link Allegro visual meshes in grip_hand_frame."""
    cfg, hand, qpos, mount = _pose_and_mount(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=grip_radius_m,
        zero_qpos=zero_qpos,
    )
    model, data = _forward_model(hand, qpos, grip_spec.handedness)
    buckets: dict[str, list[Mesh]] = {}
    for bname, mname, gverts, faces in _iter_visible_geoms(model, data, cfg):
        key = _canonical_part_key(bname, mname)
        buckets.setdefault(key, []).append(
            Mesh(_transform_to_grip_frame(gverts, mount), faces)
        )
    return {
        key: (meshes[0] if len(meshes) == 1 else concat_meshes(*meshes))
        for key, meshes in buckets.items()
    }


def measure_allegro_anthropometry(
    grip_spec: GripSpec,
    side: HandSide,
) -> dict[str, float]:
    """Measure uncurled scaled Allegro hand length, palm breadth, and proximal widths."""
    parts = extract_allegro_hand_parts(grip_spec, side, zero_qpos=True)
    palm_pts = parts["palm"].vertices
    palm_plus_fingers = np.vstack([palm_pts, parts["middle_tip"].vertices])
    dims: dict[str, float] = {
        "hand_length_m": float(palm_plus_fingers[:, 0].max() - palm_pts[:, 0].min()),
        "palm_breadth_m": float(np.ptp(palm_pts[:, 1])),
    }
    for finger in ("index", "middle", "ring"):
        dims[f"{finger}_proximal_width_m"] = float(
            np.ptp(parts[f"{finger}_proximal"].vertices[:, 1])
        )
    return dims


def finger_pad_contact_gaps(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
    joint_angles: JointAngles = None,
) -> dict[str, float]:
    """Compute pad-to-grip radial clearance across each digit's fingertip pad."""
    cfg, hand, qpos, mount = _pose_and_mount(
        grip_spec, side, joint_angles, grip_radius_m=grip_radius_m
    )
    radius = (
        cfg.grip_radius_nominal_m if grip_radius_m is None else float(grip_radius_m)
    )
    model, data = _forward_model(hand, qpos, grip_spec.handedness)
    gaps: dict[str, float] = {}
    for finger in _DIGIT_TAG:
        pts = _select_part_vertices(model, data, cfg, finger=finger, segments=("tip",))
        mounted = _transform_to_grip_frame(pts, mount)
        gaps[finger] = float(np.min(_radial_xy(mounted) - radius))
    return gaps


def penetration_metrics(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
    joint_angles: JointAngles = None,
) -> dict[str, float]:
    """Compute club radial penetration and minimum inter-finger clearance (m)."""
    cfg, hand, qpos, mount = _pose_and_mount(
        grip_spec, side, joint_angles, grip_radius_m=grip_radius_m
    )
    radius = (
        cfg.grip_radius_nominal_m if grip_radius_m is None else float(grip_radius_m)
    )
    model, data = _forward_model(hand, qpos, grip_spec.handedness)
    all_pts = _transform_to_grip_frame(
        _select_part_vertices(
            model,
            data,
            cfg,
            segments=("palm", "base", "proximal", "medial", "distal", "tip"),
        ),
        mount,
    )
    club_pen = float(max(0.0, radius - float(_radial_xy(all_pts).min())))

    clouds: dict[str, list[Array]] = {}
    for bname, _mname, verts, _faces in _iter_visible_geoms(model, data, cfg):
        for fname, tag in _DIGIT_TAG.items():
            if bname.startswith(f"{tag}_") and not bname.endswith("_base"):
                clouds.setdefault(fname, []).append(
                    _transform_to_grip_frame(verts, mount)
                )
                break
    merged = {k: np.vstack(v) for k, v in clouds.items() if v}
    keys = sorted(merged)
    ff_gap = math.inf
    for i, a in enumerate(keys):
        tree_a = cKDTree(merged[a])
        for b in keys[i + 1 :]:
            ff_gap = min(ff_gap, float(tree_a.query(merged[b], k=1)[0].min()))
    return {
        "club_penetration_m": club_pen,
        "finger_finger_gap_m": float(ff_gap),
    }


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: JointAngles = None,
    *,
    grip_radius_m: float | None = None,
) -> HandVisualResult:
    """Build scaled, wrapped Wonik Allegro 4-finger visual mesh in grip_hand_frame."""
    cfg, hand, qpos, mount = _pose_and_mount(
        grip_spec, side, joint_angles, grip_radius_m=grip_radius_m
    )
    model, data = _forward_model(hand, qpos, grip_spec.handedness)
    all_verts: list[Array] = []
    all_faces: list[NDArray[np.int64]] = []
    base = 0
    for _bname, _mname, gverts, faces in _iter_visible_geoms(model, data, cfg):
        all_verts.append(_transform_to_grip_frame(gverts, mount))
        all_faces.append(faces + base)
        base += len(gverts)
    if not all_verts:
        raise ValueError("no Wonik Allegro visual geoms found")
    mesh = Mesh(np.vstack(all_verts), np.vstack(all_faces))
    return HandVisualResult(mesh=mesh, material_rgba=cfg.skin_tone_rgba)


__all__ = [
    "allegro_mount_determinant",
    "build_hand_visual",
    "extract_allegro_hand_parts",
    "finger_pad_contact_gaps",
    "measure_allegro_anthropometry",
    "penetration_metrics",
]
