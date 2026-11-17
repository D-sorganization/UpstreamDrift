"""Shadow Dexterous Hand visual backend (Apache-2.0 Menagerie meshes, #15122)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import cKDTree

from src.shared.python.core.handedness import Handedness
from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import Mesh, concat_meshes
from src.shared.python.model_appearance.grip_pose import GripPose, Hand
from src.shared.python.model_appearance.hand_backends.shadow_settings import (
    _CURL_STAGE_FINGERS,
    _FINGER_BODY_TAG,
    _FINGER_CHAINS,
    _TANG_REF_X_MODEL,
    Array,
    ShadowHandPose,
    ShadowHandVisualSettings,
    visual_settings,
)
from src.shared.python.model_appearance.hand_backends.shadow_skin import (
    build_shadow_skin_mesh,
    measure_shadow_skin_envelope,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    HandVisualBackend,
    register_hand_visual_backend,
)
from src.shared.python.model_appearance.hand_backends.shadow_solver import (
    JointAngles,
    _apply_joint_angles,
    _baseline_qpos,
    _canonical_part_key,
    _forward_and_mount,
    _forward_model,
    _hand_from_side,
    _iter_visible_geoms,
    _joint_qpos_index,
    _load_model,
    _radial_xy,
    _require_mujoco,
    _scale_vector,
    _select_part_vertices,
    _transform_to_grip_frame,
    _validate_grip_radius,
    apply_wrist_offsets,
    map_hand_grip_spec_to_wrist,
    solve_shadow_hand_qpos,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
)

BACKEND_NAME = "shadow"


def shadow_distal_pad_centroid(
    hand: Hand,
    pose: ShadowHandPose,
    finger: str,
    *,
    settings: ShadowHandVisualSettings | None = None,
) -> Array:
    """Distal-pad centroid in the hand-grip frame (retarget reference point)."""
    cfg = visual_settings() if settings is None else settings
    model, data = _forward_model(hand, pose.qpos, pose.handedness)
    pts = _select_part_vertices(model, data, cfg, finger=finger, segments=("distal",))
    return _transform_to_grip_frame(pts, pose.mount_matrix).mean(axis=0)


def finger_pad_contact_gaps(
    hand: Hand,
    pose: ShadowHandPose,
    *,
    grip_radius_m: float,
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    """Radial gap to the grip cylinder for each finger's distal/middle pads (m)."""
    cfg = visual_settings() if settings is None else settings
    model, data = _forward_model(hand, pose.qpos, pose.handedness)
    target = grip_radius_m + cfg.pad_clearance_m
    gaps: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little", "thumb"):
        pts = _select_part_vertices(
            model, data, cfg, finger=finger, segments=("middle", "distal")
        )
        mounted = _transform_to_grip_frame(pts, pose.mount_matrix)
        gaps[finger] = float(np.min(np.abs(_radial_xy(mounted) - target)))
    return gaps


def _uv_ellipsoid(
    center: Array, radii: tuple[float, float, float], n_lat: int = 8, n_lon: int = 12
) -> tuple[Array, NDArray[np.int64]]:
    verts: list[list[float]] = [[0.0, 0.0, 1.0]]
    for i in range(1, n_lat):
        th = math.pi * i / n_lat
        st, ct = math.sin(th), math.cos(th)
        for j in range(n_lon):
            ph = 2.0 * math.pi * j / n_lon
            verts.append([st * math.cos(ph), st * math.sin(ph), ct])
    verts.append([0.0, 0.0, -1.0])
    south = len(verts) - 1
    faces: list[list[int]] = []
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + jn])
    for i in range(n_lat - 2):
        r0, r1 = 1 + i * n_lon, 1 + (i + 1) * n_lon
        for j in range(n_lon):
            jn = (j + 1) % n_lon
            faces.append([r0 + j, r1 + j, r1 + jn])
            faces.append([r0 + j, r1 + jn, r0 + jn])
    r_last = 1 + (n_lat - 2) * n_lon
    for j in range(n_lon):
        jn = (j + 1) % n_lon
        faces.append([r_last + j, south, r_last + jn])
    scaled = np.asarray(verts, float) * np.asarray(radii, float) + center
    return scaled, np.asarray(faces, dtype=np.int64)


def _mcp_knuckle_bridges(
    model: Any,
    data: Any,
    cfg: ShadowHandVisualSettings,
    mount: Array,
) -> list[tuple[Array, NDArray[np.int64]]]:
    mujoco = _require_mujoco()
    r = cfg.mcp_bridge_radius_m
    radii = (r * 1.15, r * 0.95, r * 0.90)
    bridges: list[tuple[Array, NDArray[np.int64]]] = []
    for bid in range(model.nbody):
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        if not any(bname.endswith(f"{t}proximal") for t in ("ff", "mf", "rf", "lf")):
            continue
        center = np.asarray(data.xpos[bid], float)
        bv, bf = _uv_ellipsoid(center, radii)
        bridges.append((_transform_to_grip_frame(bv, mount), bf))
    return bridges


def extract_shadow_hand_mesh(
    hand: Hand,
    qpos: Array,
    *,
    mount_matrix: Array | None = None,
    settings: ShadowHandVisualSettings | None = None,
    handedness: Handedness = "right",
) -> Mesh:
    """Humanized skin mesh (no forearm / wrist / knuckle hardware) in grip frame."""
    cfg = visual_settings() if settings is None else settings
    model, data = _forward_model(hand, qpos, handedness)
    mount = cfg.mount_matrix if mount_matrix is None else mount_matrix
    all_verts: list[Array] = []
    all_faces: list[NDArray[np.int64]] = []
    base = 0
    for _bname, _mname, gverts, faces in _iter_visible_geoms(model, data, cfg):
        all_verts.append(_transform_to_grip_frame(gverts, mount))
        all_faces.append(faces + base)
        base += len(gverts)
    for bv, bf in _mcp_knuckle_bridges(model, data, cfg, mount):
        all_verts.append(bv)
        all_faces.append(bf + base)
        base += len(bv)
    if not all_verts:
        raise ValueError("no humanized Shadow hand geoms after filtering")
    return Mesh(np.vstack(all_verts), np.vstack(all_faces))


def _resolve_visual_call(
    grip_spec: GripSpec,
    side: HandSide,
    grip_radius_m: float | None,
    settings: ShadowHandVisualSettings | None = None,
) -> tuple[ShadowHandVisualSettings, Hand, HandGripSpec, float]:
    cfg = visual_settings() if settings is None else settings
    hand = _hand_from_side(side)
    spec_hand = grip_spec.trail if hand == Hand.TRAIL else grip_spec.lead
    radius = _validate_grip_radius(
        cfg.grip_radius_nominal_m if grip_radius_m is None else grip_radius_m, cfg
    )
    return cfg, hand, spec_hand, radius


def extract_shadow_hand_parts(
    grip_spec: GripSpec,
    side: HandSide,
    *,
    grip_radius_m: float | None = None,
    zero_qpos: bool = False,
    joint_angles: JointAngles = None,
) -> dict[str, Mesh]:
    """Per-segment visual meshes in ``grip_hand_frame`` (or scaled world if zero)."""
    cfg, hand, spec_hand, radius = _resolve_visual_call(grip_spec, side, grip_radius_m)
    if zero_qpos:
        mujoco = _require_mujoco()
        model = _load_model(hand == Hand.TRAIL, grip_spec.handedness)
        data = mujoco.MjData(model)
        data.qpos[:] = np.zeros(model.nq, dtype=float)
        mujoco.mj_forward(model, data)
        scale = _scale_vector(grip_spec.hand_size, cfg)
        mount = np.eye(4)
        mount[:3, :3] = np.diag(scale)
    else:
        solved = solve_shadow_hand_qpos(
            hand, spec_hand, grip_spec=grip_spec, grip_radius_m=radius, settings=cfg
        )
        qpos = _apply_joint_angles(
            solved.qpos,
            _load_model(hand == Hand.TRAIL, grip_spec.handedness),
            hand,
            joint_angles,
            handedness=grip_spec.handedness,
        )
        model, data = _forward_model(hand, qpos, grip_spec.handedness)
        mount = solved.mount_matrix

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


def measure_shadow_anthropometry(
    grip_spec: GripSpec,
    side: HandSide,
) -> dict[str, float]:
    """Measure uncurled anatomical hand length, palm breadth, and proximal widths."""
    parts = extract_shadow_hand_parts(grip_spec, side, zero_qpos=True)
    palm_plus_fingers = np.vstack(
        [parts["palm"].vertices, parts["middle_distal"].vertices]
    )
    palm_pts = parts["palm"].vertices
    dims: dict[str, float] = {
        "hand_length_m": float(palm_plus_fingers[:, 0].max() - palm_pts[:, 0].min()),
        "palm_breadth_m": float(np.ptp(palm_pts[:, 1])),
    }
    for finger in ("index", "middle", "ring", "little"):
        dims[f"{finger}_proximal_width_m"] = float(
            np.ptp(parts[f"{finger}_proximal"].vertices[:, 1])
        )
    return dims


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: JointAngles = None,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
    settings: ShadowHandVisualSettings | None = None,
) -> HandVisualResult:
    """Humanized Shadow skin mesh in ``grip_hand_frame`` (visual only, ADR-0057 A2).

    The hand is posed unrolled; the per-hand shaft roll is then applied as a
    rigid rotation about the shaft axis.
    """
    cfg, hand, spec_hand, radius = _resolve_visual_call(
        grip_spec, side, grip_radius_m, settings=settings
    )
    if cfg.skin_mode not in ("lofted_envelope", "robot_shell"):
        raise ValueError(f"unknown Shadow skin_mode {cfg.skin_mode!r}")
    handedness = grip_spec.handedness
    if joint_angles is None:
        solved = solve_shadow_hand_qpos(
            hand, spec_hand, grip_spec=grip_spec, grip_radius_m=radius, settings=cfg
        )
        qpos, mount = solved.qpos, solved.mount_matrix
    else:
        mujoco = _require_mujoco()
        model = _load_model(hand == Hand.TRAIL, handedness)
        data = mujoco.MjData(model)
        base_q = _baseline_qpos(model, hand, handedness)
        if not isinstance(joint_angles, np.ndarray):
            apply_wrist_offsets(
                base_q, model, hand, spec_hand, handedness=handedness, settings=cfg
            )
        qpos = _apply_joint_angles(
            base_q, model, hand, joint_angles, handedness=handedness
        )
        mount = _forward_and_mount(model, data, qpos, hand, grip_spec, radius, cfg)
    if cfg.skin_mode == "lofted_envelope":
        mesh = build_shadow_skin_mesh(
            hand,
            qpos,
            mount_matrix=mount,
            grip_spec=grip_spec,
            grip_radius_m=radius,
            settings=cfg,
        )
    else:
        mesh = extract_shadow_hand_mesh(
            hand, qpos, mount_matrix=mount, settings=cfg, handedness=handedness
        )
    roll = apply_shaft_roll(
        np.eye(3), APPEARANCE_GRIP_SHAFT_AXIS, resolved_shaft_roll_deg(grip_spec, side)
    )[:3, :3]
    rgb = cfg.skin_tone_rgba[:3] if skin_rgb is None else skin_rgb
    return HandVisualResult(
        backend=BACKEND_NAME,
        side=side,
        mesh=Mesh(mesh.vertices @ roll.T, mesh.faces),
        skin_rgb=(float(rgb[0]), float(rgb[1]), float(rgb[2])),
        grip_radius_m=radius,
    )


def register() -> None:
    """Add the Shadow backend to the shared registry (called lazily)."""
    register_hand_visual_backend(
        HandVisualBackend(
            name=BACKEND_NAME,
            builder=build_hand_visual,
            description="Humanized Shadow Dexterous Hand skin (Menagerie, #15122)",
            capabilities=frozenset({"self_posing", "explicit_joint_angles"}),
        )
    )


def build_shadow_grip_hand_mesh(
    hand: Hand,
    *,
    grip_radius_m: float,
    pose: GripPose | None = None,
    grip_spec: GripSpec | None = None,
    settings: ShadowHandVisualSettings | None = None,
) -> Mesh:
    """Compatibility wrapper used by placement helpers."""
    _ = pose
    spec = GripSpec.legacy() if grip_spec is None else grip_spec
    side: HandSide = "trail" if hand == Hand.TRAIL else "lead"
    return build_hand_visual(
        spec, side, None, grip_radius_m=grip_radius_m, settings=settings
    ).mesh


def penetration_metrics(
    hand: Hand,
    qpos: Array,
    *,
    grip_radius_m: float,
    mount_matrix: Array | None = None,
    settings: ShadowHandVisualSettings | None = None,
    handedness: Handedness = "right",
) -> dict[str, float]:
    """Compute club radial penetration and inter-finger minimum clearance (m)."""
    cfg = visual_settings() if settings is None else settings
    model, data = _forward_model(hand, qpos, handedness)
    mount = cfg.mount_matrix if mount_matrix is None else mount_matrix
    fingers = _transform_to_grip_frame(
        _select_part_vertices(
            model,
            data,
            cfg,
            segments=("palm", "metacarpal", "proximal", "middle", "distal"),
        ),
        mount,
    )
    club_pen = float(max(0.0, grip_radius_m - float(_radial_xy(fingers).min())))

    partition: dict[str, list[Array]] = {}
    for bname, _mname, verts, _faces in _iter_visible_geoms(model, data, cfg):
        for tag in ("ff", "mf", "rf", "lf", "th"):
            if f"_{tag}" in bname or tag in bname:
                partition.setdefault(tag, []).append(
                    _transform_to_grip_frame(verts, mount)
                )
                break
    clouds = {k: np.vstack(v) for k, v in partition.items() if v}
    keys = sorted(clouds)
    ff_gap = math.inf
    for i, a in enumerate(keys):
        tree_a = cKDTree(clouds[a])
        for b in keys[i + 1 :]:
            ff_gap = min(
                ff_gap, float(np.asarray(tree_a.query(clouds[b], k=1)[0]).min())
            )
    return {
        "club_penetration_m": club_pen,
        "finger_finger_gap_m": float(ff_gap),
    }


__all__ = [
    "BACKEND_NAME",
    "_CURL_STAGE_FINGERS",
    "_FINGER_BODY_TAG",
    "_FINGER_CHAINS",
    "_TANG_REF_X_MODEL",
    "Any",
    "ShadowHandPose",
    "ShadowHandVisualSettings",
    "apply_wrist_offsets",
    "build_hand_visual",
    "build_shadow_grip_hand_mesh",
    "build_shadow_skin_mesh",
    "register",
    "extract_shadow_hand_mesh",
    "extract_shadow_hand_parts",
    "finger_pad_contact_gaps",
    "map_hand_grip_spec_to_wrist",
    "measure_shadow_anthropometry",
    "measure_shadow_skin_envelope",
    "penetration_metrics",
    "shadow_distal_pad_centroid",
    "solve_shadow_hand_qpos",
    "visual_settings",
]
