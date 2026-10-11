"""Anatomical segment frames that tie the spec skeleton to MyoFullBody (#11644).

The spec (a Simscape golf model plus a calibrated Rajagopal lower limb) and
MyoFullBody do not share body-frame conventions or a zero pose: the spec's
zero pose holds the arms level in front of the body and its hip zero twist is
calibrated to the capture, while MyoFullBody's zero pose is the OpenSim
standing neutral.  Naming joints one-to-one is therefore wrong.

Instead each model gets an *anatomical frame* per segment, built from the
model's own zero-pose geometry with the same rule:

* ``y`` is the long axis (distal landmark to proximal landmark, or the heel to
  toe direction for the foot);
* ``x`` is the segment's distal hinge axis (knee, ankle, elbow, wrist
  deviation, metatarsophalangeal), signed to point to the subject's right;
* ``z = x * y``.

The frame is body-fixed, ``F(q) = R_body(q) K`` with a constant ``K`` read at
the zero pose.  Pelvis and thorax use the hip line and the vertical.  A hand
(or toe segment) takes the frame of its parent at the zero pose, i.e. a neutral
wrist (metatarsophalangeal) angle.  ``G`` is the constant rotation between the
two world frames, taken from the pelvis frames.  A segment target for MyoFullBody
is then ``G R_spec K_spec K_myo^T``: no joint-name or sign convention enters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.contracts import require

Array = np.ndarray
SIDES = ("l", "r")
UP = np.array([0.0, 0.0, 1.0])

# Segments in solve order (parents before children).
SEGMENTS: tuple[str, ...] = (
    "pelvis",
    "thorax",
    *(f"{name}_{s}" for s in SIDES for name in ("humerus", "forearm", "hand")),
    *(f"{name}_{s}" for s in SIDES for name in ("femur", "tibia", "foot", "toes")),
)


@dataclass(frozen=True)
class FrameDef:
    """How one segment's anatomical frame is read from a model at its zero pose.

    ``y_from`` / ``y_to`` name bodies whose origins give the long axis
    (``y = pos(y_to) - pos(y_from)``); ``None`` means world up.  ``x_joint`` names
    the hinge whose axis is ``x``; ``x_pair`` names the (left, right) bodies whose
    origins give a horizontal right-pointing line (hips for the pelvis,
    shoulders for the thorax).  ``sign_at_reference`` reads the right-pointing sign
    of the hinge axis at a reference pose instead of the zero pose (needed for the
    spec legs, whose zero pose is a calibrated, twisted posture).  ``aligned_to``
    makes the segment take the frame of another segment at the zero pose.
    """

    body: str
    y_from: str | None = None
    y_to: str | None = None
    x_joint: str | None = None
    aligned_to: str | None = None
    x_pair: tuple[str, str] | None = None
    sign_at_reference: bool = False


def _suffix_defs(side: str) -> dict[str, FrameDef]:
    s = side
    return {
        f"femur_{s}": FrameDef(
            f"femur_{s}",
            f"tibia_{s}",
            f"femur_{s}",
            f"knee_angle_{s}",
            sign_at_reference=True,
        ),
        f"tibia_{s}": FrameDef(
            f"tibia_{s}",
            f"talus_{s}",
            f"tibia_{s}",
            f"ankle_angle_{s}",
            sign_at_reference=True,
        ),
        f"foot_{s}": FrameDef(
            f"calcn_{s}",
            f"calcn_{s}",
            f"toes_{s}",
            f"ankle_angle_{s}",
            sign_at_reference=True,
        ),
        f"toes_{s}": FrameDef(f"toes_{s}", aligned_to=f"foot_{s}"),
    }


def spec_frame_defs() -> dict[str, FrameDef]:
    """Frame definitions on the spec skeleton (body names matched by suffix)."""
    defs: dict[str, FrameDef] = {
        "pelvis": FrameDef("LowerTorso", x_pair=("femur_l", "femur_r")),
        "thorax": FrameDef("COMRod", x_pair=("LUpperArm", "RUpperArm")),
    }
    arm = {
        "l": (
            "LUpperArm",
            "Left Elbow Joint/Spherical Solid",
            "LLowerForearm",
            "Clubface Vector",
            "LEInput",
            "LWInputX",
        ),
        "r": (
            "RUpperArm",
            "Right Elbow Joint/Spherical Solid1",
            "RLowerForearm",
            "RHandStandoff",
            "REInput",
            "RWInputX",
        ),
    }
    for s, (upper, elbow, lower, hand, elbow_joint, wrist_joint) in arm.items():
        defs[f"humerus_{s}"] = FrameDef(upper, elbow, upper, elbow_joint)
        defs[f"forearm_{s}"] = FrameDef(lower, hand, elbow, wrist_joint)
        defs[f"hand_{s}"] = FrameDef(hand, aligned_to=f"forearm_{s}")
        defs.update(_suffix_defs(s))
    return defs


def myo_frame_defs() -> dict[str, FrameDef]:
    """Frame definitions on MyoFullBody."""
    defs: dict[str, FrameDef] = {
        "pelvis": FrameDef("pelvis", x_pair=("femur_l", "femur_r")),
        "thorax": FrameDef("torso", x_pair=("humerus_l", "humerus_r")),
    }
    for s in SIDES:
        defs[f"humerus_{s}"] = FrameDef(
            f"humerus_{s}", f"ulna_{s}", f"humerus_{s}", f"elbow_flexion_{s}"
        )
        defs[f"forearm_{s}"] = FrameDef(
            f"radius_{s}", f"lunate_{s}", f"ulna_{s}", f"deviation_{s}"
        )
        defs[f"hand_{s}"] = FrameDef(f"lunate_{s}", aligned_to=f"forearm_{s}")
        defs.update(_suffix_defs(s))
    return defs


def frame_from_axes(y_dir: Array, x_dir: Array) -> Array:
    """Right-handed frame with columns ``(x, y, z)``; ``x`` is made orthogonal to ``y``.

    Raises:
        ValueError: if either vector is ~zero or the two are (anti)parallel.
    """
    y = np.asarray(y_dir, dtype=float)
    x = np.asarray(x_dir, dtype=float)
    require(
        bool(np.linalg.norm(y) > 1e-9 and np.linalg.norm(x) > 1e-9), "zero-length axis"
    )
    y = y / np.linalg.norm(y)
    x = x - float(x @ y) * y
    require(bool(np.linalg.norm(x) > 1e-6), "axes are parallel")
    x = x / np.linalg.norm(x)
    return np.column_stack([x, y, np.cross(x, y)])


def _resolve(model: Any, kind: Any, name: str) -> int:
    """Index of the object whose name equals ``name`` or ends with ``/name``."""
    import mujoco

    for i in range(_count(model, kind)):
        full = mujoco.mj_id2name(model, kind, i)
        if full and (full == name or full.endswith(name)):
            return i
    raise KeyError(f"{name!r} not found in model")


def _count(model: Any, kind: Any) -> int:
    import mujoco

    return int(
        {
            mujoco.mjtObj.mjOBJ_BODY: model.nbody,
            mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        }[kind]
    )


@dataclass(frozen=True)
class SegmentFrames:
    """Constants of one model: ``body`` index and ``K`` per segment, plus world data."""

    body: dict[str, int]
    constant: dict[str, Array]
    right: Array
    thorax_frame: Array


def segment_frames(
    model: Any,
    defs: dict[str, FrameDef],
    q_zero: Array,
    q_reference: Array | None = None,
) -> SegmentFrames:
    """Read the segment constants ``K`` from ``model`` at the zero pose ``q_zero``.

    Args:
        q_reference: pose at which hinge axes flagged ``sign_at_reference`` are
            signed to point right (defaults to ``q_zero``).  It must keep those
            hinge axes within 60 degrees of the hip line.

    Raises:
        KeyError: if a named body or joint is missing.
        ValueError: if landmarks are degenerate or a reference axis is ambiguous.
    """
    import mujoco

    data = mujoco.MjData(model)
    ref = mujoco.MjData(model)
    data.qpos[:] = q_zero
    ref.qpos[:] = q_zero if q_reference is None else q_reference
    mujoco.mj_kinematics(model, data)
    mujoco.mj_kinematics(model, ref)
    body_kind, joint_kind = mujoco.mjtObj.mjOBJ_BODY, mujoco.mjtObj.mjOBJ_JOINT

    def origin(name: str, state: Any = data) -> Array:
        return np.asarray(state.xpos[_resolve(model, body_kind, name)]).copy()

    def hip_line(state: Any) -> Array:
        line = origin("femur_r", state) - origin("femur_l", state)
        line[2] = 0.0
        return line / np.linalg.norm(line)

    def axis(joint: str, at_reference: bool) -> Array:
        j = _resolve(model, joint_kind, joint)
        state = ref if at_reference else data
        side = np.asarray(state.xaxis[j]).copy()
        dot = float(side @ hip_line(state))
        require(abs(dot) > 0.5 or not at_reference, f"axis of {joint} is ambiguous")
        return np.asarray(data.xaxis[j]).copy() * (1.0 if dot >= 0.0 else -1.0)

    world: dict[str, Array] = {}
    bodies: dict[str, int] = {}
    for name in SEGMENTS:
        d = defs[name]
        bodies[name] = _resolve(model, body_kind, d.body)
        if d.aligned_to is not None:
            continue
        if d.y_from is None:
            assert d.x_pair is not None
            line = origin(d.x_pair[1]) - origin(d.x_pair[0])
            line[2] = 0.0
            world[name] = frame_from_axes(UP, line)
        else:
            assert d.y_to is not None and d.x_joint is not None
            world[name] = frame_from_axes(
                origin(d.y_to) - origin(d.y_from), axis(d.x_joint, d.sign_at_reference)
            )
    constant: dict[str, Array] = {}
    for name in SEGMENTS:
        d = defs[name]
        frame = world[d.aligned_to] if d.aligned_to is not None else world[name]
        rot = np.asarray(data.xmat[bodies[name]]).reshape(3, 3)
        constant[name] = rot.T @ frame
    return SegmentFrames(bodies, constant, hip_line(data), world["thorax"])


def world_alignment(spec: SegmentFrames, myo: SegmentFrames) -> Array:
    """Rotation ``G`` taking spec world vectors to MyoFullBody world vectors.

    Taken from the thorax frames (shoulder line and vertical), which are exactly
    axis-aligned in both zero poses; the spec's relocated hip centres are a
    1.5 degree skewed line and are not used for the world alignment.
    """
    return myo.thorax_frame @ spec.thorax_frame.T
