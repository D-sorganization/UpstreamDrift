"""Shadow Dexterous Hand visual backend (Apache-2.0 Menagerie meshes, #15122)."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation as SciRot

from src.shared.python.contracts import require
from src.shared.python.grip_spec.types import GripSpec, HandGripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_pose import GripPose, Hand
from src.shared.python.model_appearance.grip_shadow_hand_assets import (
    shadow_hand_mjcf_path,
)
from src.shared.python.model_appearance.hand_backends.types import (
    HandSide,
    HandVisualResult,
    JointAngles,
)

Array = NDArray[np.float64]
_CONFIG_PATH = (
    Path(__file__).resolve().parents[5] / "scripts/config/grip_shadow_hand_visual.json"
)

_GRASP_SOFT_QPOS: tuple[float, ...] = (
    -0.01386,
    -0.5706,
    -0.3126,
    1.013,
    0.5084,
    0.002212,
    -0.139,
    0.6854,
    0.9554,
    0.01895,
    0.1659,
    1.13,
    0.04252,
    0.02969,
    0.3183,
    -0.2968,
    0.8024,
    0.8261,
    0.01958,
    -0.04934,
    0.92,
    0.1932,
    0.5457,
    0.1011,
)

_CURL_STAGE_FINGERS: dict[str, tuple[str, ...]] = {
    "lead_fingers_index_to_pinky": ("index", "middle", "ring", "little"),
    "trail_fingers": ("index", "middle", "ring", "little"),
}

_FINGER_CHAINS: dict[str, tuple[str, ...]] = {
    "index": ("FFJ4", "FFJ3", "FFJ2", "FFJ1"),
    "middle": ("MFJ4", "MFJ3", "MFJ2", "MFJ1"),
    "ring": ("RFJ4", "RFJ3", "RFJ2", "RFJ1"),
    "little": ("LFJ5", "LFJ4", "LFJ3", "LFJ2", "LFJ1"),
    "thumb": ("THJ5", "THJ4", "THJ3", "THJ2", "THJ1"),
}

_FINGER_BODY_TAG: dict[str, str] = {
    "index": "ff",
    "middle": "mf",
    "ring": "rf",
    "little": "lf",
    "thumb": "th",
}


@dataclass(frozen=True)
class ShadowHandPose:
    """MuJoCo qpos and grip-frame mount used together for one solved hand."""

    qpos: Array
    mount_matrix: Array


@dataclass(frozen=True)
class ShadowHandVisualSettings:
    club_penetration_tolerance_m: float
    club_assembly_axial_overhang_m: float
    finger_finger_penetration_tolerance_m: float
    finger_contact_tolerance_m: float
    pad_clearance_m: float
    curl_bisect_iterations: int
    curl_joint_step_rad: float
    grip_radius_nominal_m: float
    mount_matrix: Array
    wrist_gain: dict[str, float]
    palm_cup_gain: dict[str, float]
    skin_tone_rgba: tuple[float, float, float, float]
    hidden_body_substrings: tuple[str, ...]
    hidden_mesh_substrings: tuple[str, ...]
    length_scale: float
    breadth_scale: float


def _load_settings() -> ShadowHandVisualSettings:
    raw = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    rot_cfg = raw["mount_rotation_deg"]
    rot = SciRot.from_euler(
        rot_cfg["axis1"], rot_cfg["angle1"], degrees=True
    ) * SciRot.from_euler(rot_cfg["axis2"], rot_cfg["angle2"], degrees=True)
    mount = np.eye(4)
    mount[:3, :3] = rot.as_matrix()
    trans = raw.get("mount_translation_m", (0.0, 0.0, 0.0))
    mount[:3, 3] = np.asarray(trans, float)
    anthro = raw.get("anthropometry", {})
    return ShadowHandVisualSettings(
        club_penetration_tolerance_m=float(raw["club_penetration_tolerance_m"]),
        club_assembly_axial_overhang_m=float(
            raw.get("club_assembly_axial_overhang_m", 0.22)
        ),
        finger_finger_penetration_tolerance_m=float(
            raw["finger_finger_penetration_tolerance_m"]
        ),
        finger_contact_tolerance_m=float(raw.get("finger_contact_tolerance_m", 0.004)),
        pad_clearance_m=float(raw["pad_clearance_m"]),
        curl_bisect_iterations=int(raw["curl_bisect_iterations"]),
        curl_joint_step_rad=float(raw["curl_joint_step_rad"]),
        grip_radius_nominal_m=float(raw["grip_radius_nominal_m"]),
        mount_matrix=mount,
        wrist_gain=dict(raw["wrist_from_spec"]),
        palm_cup_gain=dict(raw.get("palm_cup_from_spec", {})),
        skin_tone_rgba=tuple(
            float(x) for x in raw.get("skin_tone_rgba", (0.78, 0.62, 0.54, 1.0))
        ),  # type: ignore[return-value]
        hidden_body_substrings=tuple(
            raw.get("hidden_body_substrings", ("forearm", "_wrist"))
        ),
        hidden_mesh_substrings=tuple(
            raw.get("hidden_mesh_substrings", ("forearm", "wrist"))
        ),
        length_scale=float(anthro.get("length_scale", 1.0)),
        breadth_scale=float(anthro.get("breadth_scale", 1.0)),
    )


@lru_cache(maxsize=1)
def visual_settings() -> ShadowHandVisualSettings:
    return _load_settings()


def _require_mujoco():
    import os

    os.environ.setdefault("MUJOCO_GL", "disable")
    try:
        import mujoco
    except ImportError as exc:  # pragma: no cover
        raise ImportError("mujoco is required for Shadow hand grip visuals") from exc
    return mujoco


def _hand_from_side(side: HandSide) -> Hand:
    return Hand.TRAIL if side == "trail" else Hand.LEAD


def _prefix(hand: Hand) -> str:
    return "rh_" if hand == Hand.TRAIL else "lh_"


def _joint_qpos_index(model: Any, joint_name: str) -> int:
    mujoco = _require_mujoco()
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"joint {joint_name!r} not in model")
    return int(model.jnt_qposadr[jid])


def _baseline_qpos(model: Any) -> Array:
    q = np.zeros(model.nq, dtype=float)
    base = np.asarray(_GRASP_SOFT_QPOS, dtype=float)
    q[: min(len(base), model.nq)] = base[: model.nq]
    return q


def _apply_joint_angles(
    qpos: Array,
    model: Any,
    hand: Hand,
    joint_angles: JointAngles,
) -> Array:
    if joint_angles is None:
        return qpos
    if isinstance(joint_angles, np.ndarray):
        require(len(joint_angles) == model.nq, "joint_angles qpos length mismatch")
        return np.asarray(joint_angles, float)
    out = qpos.copy()
    prefix = _prefix(hand)
    for key, value in joint_angles.items():
        jname = key if key.startswith(prefix) else f"{prefix}{key}"
        out[_joint_qpos_index(model, jname)] = float(value)
    return out


def map_hand_grip_spec_to_wrist(
    spec_hand: HandGripSpec,
    *,
    handedness: Literal["left", "right"],
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    """Map spec to palm cup / wrist / thumb-base joints before finger curl."""
    cfg = visual_settings() if settings is None else settings
    g = cfg.wrist_gain
    cup = cfg.palm_cup_gain
    sign = -1.0 if handedness == "left" else 1.0
    palm_delta = spec_hand.palm_finger_depth - 0.5
    cross_delta = math.radians(spec_hand.club_cross_angle_deg - 40.0)
    return {
        "WRJ1": float(g["club_cross_to_wrj1_gain"]) * cross_delta,
        "WRJ2": float(g["palm_depth_to_wrj2_gain"]) * palm_delta,
        "LFJ5": float(cup.get("palm_depth_to_lfj5_gain", 0.0)) * palm_delta
        + float(cup.get("club_cross_to_lfj5_gain", 0.0)) * cross_delta,
        "FFJ4": sign * float(g["v_azimuth_to_ffj4_gain"]) * spec_hand.v_azimuth_deg,
        "THJ5": sign
        * float(g["thumb_azimuth_to_thj5_gain"])
        * spec_hand.thumb_azimuth_deg
        + float(cup.get("palm_depth_to_thj5_gain", 0.0)) * palm_delta,
        "THJ4": float(g["thumb_straight_to_thj4_gain"])
        * (1.0 - spec_hand.thumb_straightness),
    }


def apply_wrist_offsets(
    qpos: Array,
    model: Any,
    hand: Hand,
    spec_hand: HandGripSpec,
    *,
    handedness: Literal["left", "right"],
) -> None:
    offsets = map_hand_grip_spec_to_wrist(spec_hand, handedness=handedness)
    prefix = _prefix(hand)
    for short, delta in offsets.items():
        jname = f"{prefix}{short}"
        try:
            idx = _joint_qpos_index(model, jname)
        except ValueError:
            continue
        qpos[idx] = qpos[idx] + delta


def _grip_axis() -> Array:
    return np.array([0.0, 0.0, 1.0])


def _cylinder_gap(
    points: Array, radius: float, clearance: float, axis: Array | None = None
) -> float:
    ax = _grip_axis() if axis is None else axis
    ax = ax / np.linalg.norm(ax)
    perp = points - np.outer(points @ ax, ax)
    radial = np.linalg.norm(perp, axis=1)
    target = radius + clearance
    return float(np.maximum(0.0, target - radial).max())


def _radial_xy(points: Array) -> Array:
    return np.linalg.norm(points[:, :2], axis=1)


def _radial_surface_deficit(points: Array, radius: float, clearance: float) -> float:
    """Positive when the nearest pad is outside the clearance cylinder."""
    target = radius + clearance
    return float(np.min(_radial_xy(points)) - target)


def _palm_vertices_model(model: Any, data: Any) -> Array:
    mujoco = _require_mujoco()
    chunks: list[Array] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if "palm" not in bname:
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + count], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        chunks.append(local @ rot.T + np.asarray(data.geom_xpos[gid], float))
    if not chunks:
        raise ValueError("palm geom missing from Shadow hand model")
    return np.vstack(chunks)


def _compute_mount_matrix(
    model: Any,
    data: Any,
    *,
    cfg: ShadowHandVisualSettings,
    grip_radius_m: float,
) -> Array:
    """Place the palm on the grip cylinder in the hand-grip frame."""
    rot = cfg.mount_matrix[:3, :3].copy()
    base_t = cfg.mount_matrix[:3, 3].copy()
    palm = _palm_vertices_model(model, data)
    mounted = _apply_anthropometry(palm @ rot.T, cfg)
    centre = mounted.mean(axis=0)
    target = grip_radius_m + cfg.pad_clearance_m
    radial = float(np.linalg.norm(centre[:2]))
    trans = base_t.copy()
    if radial > 1e-6:
        outward = centre[:2] / radial
        delta = target - radial
        trans[0] += outward[0] * delta
        trans[1] += outward[1] * delta
    trans[2] -= float(centre[2])
    mount = np.eye(4)
    mount[:3, :3] = rot
    mount[:3, 3] = trans
    return mount


def _transform_to_grip_frame(
    points: Array,
    mount: Array,
    *,
    settings: ShadowHandVisualSettings | None = None,
) -> Array:
    cfg = visual_settings() if settings is None else settings
    mounted = points @ mount[:3, :3].T + mount[:3, 3]
    return _apply_anthropometry(mounted, cfg)


def _finger_chain_vertices(
    model: Any,
    data: Any,
    hand: Hand,
    finger: str,
    *,
    cfg: ShadowHandVisualSettings,
) -> Array:
    """Distal / middle vertices for one digit in model frame."""
    mujoco = _require_mujoco()
    tag = _FINGER_BODY_TAG[finger]
    prefix = _prefix(hand)
    chunks: list[Array] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        if _skip_robotic_geom(mujoco, model, gid, cfg):
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not bname.startswith(prefix):
            continue
        if tag not in bname or not ("distal" in bname or "middle" in bname):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + count], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        chunks.append(local @ rot.T + np.asarray(data.geom_xpos[gid], float))
    if not chunks:
        raise ValueError(f"no distal/middle geoms for finger {finger!r}")
    return np.vstack(chunks)


def _skip_robotic_geom(
    mujoco: Any, model: Any, gid: int, cfg: ShadowHandVisualSettings
) -> bool:
    body_id = int(model.geom_bodyid[gid])
    bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
    if any(tag in bname for tag in cfg.hidden_body_substrings):
        return True
    mesh_id = int(model.geom_dataid[gid])
    if mesh_id >= 0:
        mname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, mesh_id) or ""
        if any(tag in mname for tag in cfg.hidden_mesh_substrings):
            return True
    return False


def _exterior_contact_vertices(model: Any, data: Any) -> Array:
    mujoco = _require_mujoco()
    chunks: list[Array] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not any(tag in bname for tag in ("distal", "middle", "thhub", "palm")):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + count], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        chunks.append(local @ rot.T + np.asarray(data.geom_xpos[gid], float))
    if not chunks:
        raise ValueError("no contact vertices for penetration metrics")
    return np.vstack(chunks)


def _finger_partition(model: Any, data: Any) -> dict[str, Array]:
    mujoco = _require_mujoco()
    cfg = visual_settings()
    out: dict[str, list[Array]] = {}
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        if _skip_robotic_geom(mujoco, model, gid, cfg):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        key = "other"
        for finger in ("ff", "mf", "rf", "lf", "th"):
            if f"_{finger}" in bname or finger in bname:
                key = finger
                break
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + count], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        pts = local @ rot.T + np.asarray(data.geom_xpos[gid], float)
        out.setdefault(key, []).append(pts)
    return {k: np.vstack(v) for k, v in out.items() if v}


def _max_point_gap(a: Array, b: Array) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
    return float(np.sqrt(d2.min()))


def _curl_finger_chain(
    qpos: Array,
    model: Any,
    data: Any,
    hand: Hand,
    finger: str,
    radius: float,
    settings: ShadowHandVisualSettings,
    *,
    mount: Array,
) -> None:
    mujoco = _require_mujoco()
    prefix = _prefix(hand)
    chain = tuple(reversed(_FINGER_CHAINS[finger]))
    contact_tol = settings.finger_contact_tolerance_m
    for joint_short in chain:
        jname = f"{prefix}{joint_short}"
        try:
            qidx = _joint_qpos_index(model, jname)
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        except ValueError:
            continue
        lo, hi = float(model.jnt_range[jid, 0]), float(model.jnt_range[jid, 1])
        lo = max(lo, qpos[qidx])
        for _ in range(settings.curl_bisect_iterations):
            mid = 0.5 * (lo + hi)
            trial = qpos.copy()
            trial[qidx] = mid
            data.qpos[:] = trial
            mujoco.mj_forward(model, data)
            pts = _finger_chain_vertices(model, data, hand, finger, cfg=settings)
            pts = _transform_to_grip_frame(pts, mount)
            deficit = _radial_surface_deficit(pts, radius, settings.pad_clearance_m)
            if deficit > contact_tol:
                lo = mid
            elif deficit < -settings.club_penetration_tolerance_m:
                hi = mid
            else:
                lo = mid
                break
        qpos[qidx] = lo
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)


def _finger_curl_sequence(hand: Hand, grip_spec: GripSpec) -> tuple[str, ...]:
    prefix = "lead_" if hand == Hand.LEAD else "trail_"
    digits: list[str] = []
    for stage in grip_spec.curl_order:
        if not stage.startswith(prefix):
            continue
        digits.extend(_CURL_STAGE_FINGERS.get(stage, ()))
    if not digits:
        digits = ["index", "middle", "ring", "little"]
    return tuple(digits)


def solve_shadow_hand_qpos(
    hand: Hand,
    spec_hand: HandGripSpec,
    *,
    grip_spec: GripSpec,
    grip_radius_m: float,
    settings: ShadowHandVisualSettings | None = None,
) -> ShadowHandPose:
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    qpos = _baseline_qpos(model)
    apply_wrist_offsets(qpos, model, hand, spec_hand, handedness=grip_spec.handedness)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mount = _compute_mount_matrix(model, data, cfg=cfg, grip_radius_m=grip_radius_m)
    for finger in _finger_curl_sequence(hand, grip_spec):
        _curl_finger_chain(
            qpos, model, data, hand, finger, grip_radius_m, cfg, mount=mount
        )
    _curl_finger_chain(
        qpos, model, data, hand, "thumb", grip_radius_m, cfg, mount=mount
    )
    qpos = _relax_club_penetration(
        qpos, model, data, hand, grip_radius_m, cfg.club_penetration_tolerance_m, mount
    )
    return ShadowHandPose(qpos=qpos, mount_matrix=mount)


def _relax_club_penetration(
    qpos: Array,
    model: Any,
    data: Any,
    hand: Hand,
    radius: float,
    tolerance_m: float,
    mount: Array,
) -> Array:
    mujoco = _require_mujoco()
    cfg = visual_settings()
    prefix = _prefix(hand)
    for _ in range(24):
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pts = _exterior_contact_vertices(model, data)
        pts = _transform_to_grip_frame(pts, mount)
        if _cylinder_gap(pts, radius, 0.0) <= tolerance_m:
            return qpos
        stepped = False
        for joint_short in (
            "FFJ3",
            "FFJ2",
            "FFJ1",
            "MFJ3",
            "MFJ2",
            "MFJ1",
            "RFJ3",
            "RFJ2",
            "RFJ1",
            "LFJ3",
            "LFJ2",
            "LFJ1",
            "THJ4",
            "THJ2",
            "THJ1",
        ):
            jname = f"{prefix}{joint_short}"
            try:
                qidx = _joint_qpos_index(model, jname)
            except ValueError:
                continue
            if qpos[qidx] <= 0.0:
                continue
            qpos[qidx] = max(0.0, qpos[qidx] - cfg.curl_joint_step_rad)
            stepped = True
        if not stepped:
            return qpos
    return qpos


def _apply_anthropometry(verts: Array, cfg: ShadowHandVisualSettings) -> Array:
    out = verts.copy()
    out[:, 0] *= cfg.breadth_scale
    out[:, 1] *= cfg.breadth_scale
    out[:, 2] *= cfg.length_scale
    return out


def _finger_distal_vertices(
    model: Any,
    data: Any,
    hand: Hand,
    finger: str,
    *,
    cfg: ShadowHandVisualSettings,
) -> Array:
    mujoco = _require_mujoco()
    tag = _FINGER_BODY_TAG[finger]
    prefix = _prefix(hand)
    chunks: list[Array] = []
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        if _skip_robotic_geom(mujoco, model, gid, cfg):
            continue
        body_id = int(model.geom_bodyid[gid])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if not bname.startswith(prefix) or tag not in bname or "distal" not in bname:
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        adr = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr : adr + count], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        chunks.append(local @ rot.T + np.asarray(data.geom_xpos[gid], float))
    if not chunks:
        return _finger_chain_vertices(model, data, hand, finger, cfg=cfg)
    return np.vstack(chunks)


def shadow_distal_pad_centroid(
    hand: Hand,
    pose: ShadowHandPose,
    finger: str,
    *,
    settings: ShadowHandVisualSettings | None = None,
) -> Array:
    """Distal-pad centroid in the hand-grip frame (retarget reference point)."""
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = pose.qpos
    mujoco.mj_forward(model, data)
    pts = _finger_distal_vertices(model, data, hand, finger, cfg=cfg)
    mounted = _transform_to_grip_frame(pts, pose.mount_matrix, settings=cfg)
    return mounted.mean(axis=0)


def finger_pad_contact_gaps(
    hand: Hand,
    pose: ShadowHandPose,
    *,
    grip_radius_m: float,
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    """Radial gap to the grip cylinder for each finger's distal/middle pads (m)."""
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = pose.qpos
    mujoco.mj_forward(model, data)
    target = grip_radius_m + cfg.pad_clearance_m
    gaps: dict[str, float] = {}
    for finger in ("index", "middle", "ring", "little", "thumb"):
        pts = _finger_chain_vertices(model, data, hand, finger, cfg=cfg)
        pts = _transform_to_grip_frame(pts, pose.mount_matrix)
        gaps[finger] = float(np.min(np.abs(_radial_xy(pts) - target)))
    return gaps


def extract_shadow_hand_mesh(
    hand: Hand,
    qpos: Array,
    *,
    mount_matrix: Array | None = None,
    settings: ShadowHandVisualSettings | None = None,
) -> Mesh:
    """Humanized skin mesh (no forearm / wrist hardware) in the hand-grip frame."""
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    require(len(qpos) == model.nq, "qpos length mismatch")
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    mount = cfg.mount_matrix if mount_matrix is None else mount_matrix
    all_verts: list[Array] = []
    all_faces: list[NDArray[np.int64]] = []
    base = 0
    for gid in range(model.ngeom):
        if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        if int(model.geom_group[gid]) != 2:
            continue
        if _skip_robotic_geom(mujoco, model, gid, cfg):
            continue
        mesh_id = int(model.geom_dataid[gid])
        if mesh_id < 0:
            continue
        adr_v = int(model.mesh_vertadr[mesh_id])
        count_v = int(model.mesh_vertnum[mesh_id])
        local = np.asarray(model.mesh_vert[adr_v : adr_v + count_v], float)
        rot = np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
        gverts = local @ rot.T + np.asarray(data.geom_xpos[gid], float)
        gverts = _transform_to_grip_frame(gverts, mount, settings=cfg)
        adr_f = int(model.mesh_faceadr[mesh_id])
        count_f = int(model.mesh_facenum[mesh_id])
        f = np.asarray(model.mesh_face[adr_f : adr_f + count_f], np.int64).reshape(
            -1, 3
        )
        all_verts.append(gverts)
        all_faces.append(f + base)
        base += len(gverts)
    if not all_verts:
        raise ValueError("no humanized Shadow hand geoms after filtering")
    return Mesh(np.vstack(all_verts), np.vstack(all_faces))


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandSide,
    joint_angles: JointAngles,
    *,
    grip_radius_m: float | None = None,
) -> HandVisualResult:
    """Shadow skin mesh in the hand-grip frame (visual only)."""
    hand = _hand_from_side(side)
    spec_hand = grip_spec.lead if hand == Hand.LEAD else grip_spec.trail
    radius = (
        visual_settings().grip_radius_nominal_m
        if grip_radius_m is None
        else float(grip_radius_m)
    )
    mujoco = _require_mujoco()
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    if joint_angles is None:
        solved = solve_shadow_hand_qpos(
            hand, spec_hand, grip_spec=grip_spec, grip_radius_m=radius
        )
        mesh = extract_shadow_hand_mesh(
            hand, solved.qpos, mount_matrix=solved.mount_matrix
        )
    elif isinstance(joint_angles, np.ndarray):
        qpos = _apply_joint_angles(_baseline_qpos(model), model, hand, joint_angles)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        mount = _compute_mount_matrix(
            model, data, cfg=visual_settings(), grip_radius_m=radius
        )
        mesh = extract_shadow_hand_mesh(hand, qpos, mount_matrix=mount)
    else:
        qpos = _baseline_qpos(model)
        apply_wrist_offsets(
            qpos, model, hand, spec_hand, handedness=grip_spec.handedness
        )
        qpos = _apply_joint_angles(qpos, model, hand, joint_angles)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        mount = _compute_mount_matrix(
            model, data, cfg=visual_settings(), grip_radius_m=radius
        )
        mesh = extract_shadow_hand_mesh(hand, qpos, mount_matrix=mount)
    cfg = visual_settings()
    return HandVisualResult(mesh=mesh, material_rgba=cfg.skin_tone_rgba)


def build_shadow_grip_hand_mesh(
    hand: Hand,
    *,
    grip_radius_m: float,
    pose: GripPose | None = None,
    grip_spec: GripSpec | None = None,
) -> Mesh:
    """Compatibility wrapper used by placement helpers."""
    _ = pose
    spec = GripSpec.legacy() if grip_spec is None else grip_spec
    side: HandSide = "trail" if hand == Hand.TRAIL else "lead"
    return build_hand_visual(spec, side, None, grip_radius_m=grip_radius_m).mesh


def penetration_metrics(
    hand: Hand,
    qpos: Array,
    *,
    grip_radius_m: float,
    mount_matrix: Array | None = None,
    settings: ShadowHandVisualSettings | None = None,
) -> dict[str, float]:
    mujoco = _require_mujoco()
    cfg = visual_settings() if settings is None else settings
    path = shadow_hand_mjcf_path(trail=hand == Hand.TRAIL)
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    if mount_matrix is None:
        mount = _compute_mount_matrix(model, data, cfg=cfg, grip_radius_m=grip_radius_m)
    else:
        mount = mount_matrix
    fingers = _exterior_contact_vertices(model, data)
    fingers = _transform_to_grip_frame(fingers, mount, settings=cfg)
    club_pen = _cylinder_gap(fingers, grip_radius_m, 0.0)
    parts = _finger_partition(model, data)
    mounted = {
        k: _transform_to_grip_frame(v, mount, settings=cfg) for k, v in parts.items()
    }
    keys = sorted(mounted)
    ff_gap = math.inf
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            ff_gap = min(ff_gap, _max_point_gap(mounted[a], mounted[b]))
    return {
        "club_penetration_m": club_pen,
        "finger_finger_gap_m": float(ff_gap),
    }


__all__ = [
    "ShadowHandPose",
    "ShadowHandVisualSettings",
    "apply_wrist_offsets",
    "build_hand_visual",
    "build_shadow_grip_hand_mesh",
    "extract_shadow_hand_mesh",
    "finger_pad_contact_gaps",
    "shadow_distal_pad_centroid",
    "map_hand_grip_spec_to_wrist",
    "penetration_metrics",
    "solve_shadow_hand_qpos",
    "visual_settings",
]
