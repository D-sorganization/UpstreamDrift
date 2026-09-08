"""The articulated golfer, after the MATLAB 3-D golf model (#9710, #9714).

Version 2.0 places every body frame exactly where ``GolfSwing3D_Kinetic``
places its sensor frames, using the joint conventions identified from the
model's own dataset-generator logs (``simscape.py``; evidence in
``docs/motion_capture/evidence/simscape_axes.md``, residual <= 2.7e-4 rad
on 620 logged frames). Each of the 27 shared DOFs is therefore the Simscape
angle itself, up to the sign recorded in :data:`SIMSCAPE_NAMES`; no axis
conversion happens at export time.

Chain, as the logs prove it is built (the guide's prose lists the spine
universal and torso revolute the other way round):

    World -[Hip: translation + 3 rotations]- pelvis (Simscape "Hip" body)
    pelvis -[Torso revolute, about the trunk axis]- torso ("TorsoLogs" sensor)
    torso -[0.061 m up]-[Spine universal]- spine ("SpineLogs" sensor)
    spine -[0.249 m: (0, -0.0508, -0.2438) m]- hub (rigid; the scapula pivot)
    hub -[Scapula universal, at the hub]- scapula
    scapula -[0.254 m strut]-[Shoulder gimbal]- shoulder = upper arm
    shoulder -[0.3047 m along +x]-[Elbow hinge]- elbow
    elbow -[Forearm revolute, pro/supination]- forearm ("LFLogs" sensor)
    forearm -[forearm length along +z]-[Wrist universal]- wrist

Legs and the head are not in the MATLAB model and are added for the
detectors (hips, knees, ankles, nose); the wrist universal has no logged
angles and keeps a plausible convention. The Simscape world is taken as
z up, x toward the target and y to the golfer's right (right-handed, ADR-0041
with y and z swapped); that assumption only moves the root's hip angles and
translation, which are per-session offsets in any case.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from .kinematics import ArticulatedModel, Joint, ModelSpec
from .session import LandmarkMap

PI = float(np.pi)
HALF_PI = PI / 2
CYCLIC = 2 * PI / 3 / np.sqrt(3.0)  # 120 degrees about a body diagonal
SWAP = PI / np.sqrt(2.0)  # 180 degrees about a face diagonal
ZERO = (0.0, 0.0, 0.0)

#: Simscape world (x toward target, y to the golfer's right, z up) from ours
#: (x toward target, y up, z to the golfer's right): x_s = x, y_s = -z, z_s = y.
WORLD_SIMSCAPE_FROM_OURS = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
#: Constant frame ahead of the hip primitives, in the Simscape world.
HIP_PRE_SIMSCAPE = (0.410384, 0.410352, -1.531556)
Vec = tuple[float, float, float]


def _vec(values: np.ndarray) -> Vec:
    return (float(values[0]), float(values[1]), float(values[2]))


ROOT_PRE: Vec = _vec(
    Rotation.from_matrix(
        WORLD_SIMSCAPE_FROM_OURS.T @ Rotation.from_rotvec(HIP_PRE_SIMSCAPE).as_matrix()
    ).as_rotvec()
)

SCAPULA_RANGE = ((-1.6, 1.6), (-1.6, 1.6))
SHOULDER_RANGE = ((-PI, PI), (-PI, PI), (-PI, PI))
ELBOW_RANGE = ((-0.1, 2.8),)
FOREARM_RANGE = ((-PI, PI),)
WRIST_RANGE = ((-1.3, 1.3), (-0.7, 0.7))
SPINE_RANGE = ((-0.9, 0.9), (-0.9, 0.9))
TORSO_RANGE = ((-2.5, 2.5),)
HIP_ROT_RANGE = ((-1.6, 1.6), (-1.6, 1.6), (-1.0, 1.0))
KNEE_RANGE = ((0.0, 2.6),)

HUB_OFFSET_M = (0.0, -0.0508, -0.2438)
HUB_LENGTH_M = float(np.linalg.norm(HUB_OFFSET_M))
HUB_DIRECTION: Vec = _vec(np.asarray(HUB_OFFSET_M) / HUB_LENGTH_M)

#: Default segment lengths, metres. Simscape values where the logs give them
#: (``simscape_axes.md`` offsets); the rest are anthropometric defaults.
DEFAULT_LENGTHS_M: dict[str, float] = {
    "lower_torso": 0.061,  # Torso origin -> spine universal
    "upper_torso": HUB_LENGTH_M,  # spine universal -> hub, |(0, -0.0508, -0.2438)|
    "hub_to_shoulder": 0.254,  # scapula strut (HubtoSLength)
    "head": 0.18,  # hub -> nose (not in the MATLAB model)
    "upper_arm": 0.3047,  # shoulder -> elbow along the upper arm's +x
    "forearm": 0.26,  # elbow -> wrist centre (Simscape forearm+hand is 0.3556)
    "hip_half": 0.10,  # pelvis -> hip centre (legs: not in the MATLAB model)
    "thigh": 0.44,
    "shank": 0.42,
}


def _arm(side: str) -> tuple[Joint, ...]:
    """Identified conventions differ in label between sides; both are exact."""
    left = side == "left"
    scap_axes, scap_pre, scap_post = ("xz", (CYCLIC, -CYCLIC, CYCLIC), (-HALF_PI, 0, 0))
    sh_axes, sh_pre, sh_post = ("xyz", (SWAP, 0.0, -SWAP), ZERO)
    el_axes, el_pre = ("y", (-PI, 0.0, 0.0))
    fa_axes, fa_post = ("x", (-CYCLIC, CYCLIC, -CYCLIC))
    if not left:
        scap_axes, scap_pre, scap_post = ("yz", (0.0, -SWAP, -SWAP), (CYCLIC,) * 3)
        sh_axes, sh_pre, sh_post = ("yzx", (0.0, -SWAP, -SWAP), (CYCLIC,) * 3)
        el_axes, el_pre = ("x", (0.0, 0.0, -HALF_PI))
        fa_axes, fa_post = ("y", (0.0, SWAP, SWAP))
    strut: Vec = (0.0, 0.0, -1.0 if left else 1.0)
    return (
        Joint(
            f"{side}_scapula",
            "hub",
            ZERO,
            None,
            scap_axes,
            SCAPULA_RANGE,
            False,
            scap_pre,
            scap_post,
        ),
        Joint(
            f"{side}_shoulder",
            f"{side}_scapula",
            strut,
            "hub_to_shoulder",
            sh_axes,
            SHOULDER_RANGE,
            True,
            sh_pre,
            sh_post,
        ),
        Joint(
            f"{side}_elbow",
            f"{side}_shoulder",
            (1.0, 0.0, 0.0),
            "upper_arm",
            el_axes,
            ELBOW_RANGE,
            True,
            el_pre,
        ),
        Joint(
            f"{side}_forearm",
            f"{side}_elbow",
            ZERO,
            None,
            fa_axes,
            FOREARM_RANGE,
            False,
            ZERO,
            fa_post,
        ),
        Joint(
            f"{side}_wrist",
            f"{side}_forearm",
            (0.0, 0.0, 1.0),
            "forearm",
            "xz",
            WRIST_RANGE,
        ),
    )


def _leg(side: str, sign: float) -> tuple[Joint, ...]:
    """Pelvis frame: x to the golfer's left, z up the trunk."""
    return (
        Joint(
            f"{side}_hip", "pelvis", (sign, 0.0, 0.0), "hip_half", "xyz", HIP_ROT_RANGE
        ),
        Joint(
            f"{side}_knee", f"{side}_hip", (0.0, 0.0, -1.0), "thigh", "x", KNEE_RANGE
        ),
        Joint(f"{side}_ankle", f"{side}_knee", (0.0, 0.0, -1.0), "shank", ""),
    )


GOLFER_SPEC = ModelSpec(
    name="golfer-scapula/2.0",
    joints=(
        # Hip: translation + the three hip primitives (y, -x, z after the
        # identified constant frame); pelvis.rx carries -HipY.
        Joint("pelvis", None, axes="yxz", pre_rotvec=ROOT_PRE),
        Joint(
            "torso",
            "pelvis",
            ZERO,
            None,
            "z",
            TORSO_RANGE,
            False,
            ZERO,
            (0.0, 0.0, HALF_PI),
        ),
        Joint(
            "spine",
            "torso",
            (0.0, 0.0, 1.0),
            "lower_torso",
            "zx",
            SPINE_RANGE,
            False,
            (-CYCLIC, CYCLIC, -CYCLIC),
            (-CYCLIC, -CYCLIC, -CYCLIC),
        ),
        Joint("hub", "spine", HUB_DIRECTION, "upper_torso", ""),
        Joint("nose", "hub", HUB_DIRECTION, "head", "", landmark=True),
        *_arm("left"),
        *_arm("right"),
        *_leg("left", 1.0),
        *_leg("right", -1.0),
    ),
    lengths_m=DEFAULT_LENGTHS_M,
)

#: Which reconstruct joint observes which model landmark.
GOLFER_LANDMARK_MAP = LandmarkMap(
    to_reconstruct={
        "pelvis": "mid_hip",
        "hub": "neck",
        "nose": "nose",
        "left_shoulder": "left_shoulder",
        "left_elbow": "left_elbow",
        "left_wrist": "left_wrist",
        "right_shoulder": "right_shoulder",
        "right_elbow": "right_elbow",
        "right_wrist": "right_wrist",
        "left_hip": "left_hip",
        "left_knee": "left_knee",
        "left_ankle": "left_ankle",
        "right_hip": "right_hip",
        "right_knee": "right_knee",
        "right_ankle": "right_ankle",
    },
    # model length <- tape-measured reconstruct segment (child joint name)
    length_from_segment={
        "upper_arm": "left_elbow",
        "forearm": "left_wrist",
        "thigh": "left_knee",
        "shank": "left_ankle",
        "hip_half": "left_hip",
        "hub_to_shoulder": "left_shoulder",
    },
)

#: Our DOF -> (Simscape start-position variable, sign). Validated: our angle
#: times the sign is the logged Simscape angle (``simscape_axes.md``). The
#: wrist rows are named only; Simscape logs no wrist angles.
SIMSCAPE_NAMES: dict[str, tuple[str, float]] = {
    "pelvis.tx": ("TranslationStartPositionX", 1.0),
    "pelvis.ty": ("TranslationStartPositionY", 1.0),
    "pelvis.tz": ("TranslationStartPositionZ", 1.0),
    "pelvis.ry": ("HipStartPositionX", 1.0),
    "pelvis.rx": ("HipStartPositionY", -1.0),
    "pelvis.rz": ("HipStartPositionZ", 1.0),
    "torso.rz": ("TorsoStartPosition", 1.0),
    "spine.rz": ("SpineStartPositionX", 1.0),
    "spine.rx": ("SpineStartPositionY", 1.0),
    "left_scapula.rx": ("LScapStartPositionX", 1.0),
    "left_scapula.rz": ("LScapStartPositionY", -1.0),
    "right_scapula.ry": ("RScapStartPositionX", 1.0),
    "right_scapula.rz": ("RScapStartPositionY", 1.0),
    "left_shoulder.rx": ("LSStartPositionX", 1.0),
    "left_shoulder.ry": ("LSStartPositionY", 1.0),
    "left_shoulder.rz": ("LSStartPositionZ", 1.0),
    "right_shoulder.ry": ("RSStartPositionX", 1.0),
    "right_shoulder.rz": ("RSStartPositionY", 1.0),
    "right_shoulder.rx": ("RSStartPositionZ", 1.0),
    "left_elbow.ry": ("LEStartPosition", 1.0),
    "right_elbow.rx": ("REStartPosition", 1.0),
    "left_forearm.rx": ("LFStartPosition", 1.0),
    "right_forearm.ry": ("RFStartPosition", 1.0),
    "left_wrist.rx": ("LWStartPositionX", 1.0),
    "left_wrist.rz": ("LWStartPositionY", 1.0),
    "right_wrist.rx": ("RWStartPositionX", 1.0),
    "right_wrist.rz": ("RWStartPositionY", 1.0),
}

#: Simscape log column (``*Logs_AngularPosition*``) per Simscape variable.
SIMSCAPE_LOG_COLUMNS: dict[str, str] = {
    "HipStartPositionX": "HipLogs_HipAngularPositionX",
    "HipStartPositionY": "HipLogs_HipAngularPositionY",
    "HipStartPositionZ": "HipLogs_HipAngularPositionZ",
    "TorsoStartPosition": "TorsoLogs_AngularPosition",
    "SpineStartPositionX": "SpineLogs_AngularPositionX",
    "SpineStartPositionY": "SpineLogs_AngularPositionY",
    "LScapStartPositionX": "LScapLogs_AngularPositionX",
    "LScapStartPositionY": "LScapLogs_AngularPositionY",
    "RScapStartPositionX": "RScapLogs_AngularPositionX",
    "RScapStartPositionY": "RScapLogs_AngularPositionY",
    "LSStartPositionX": "LSLogs_AngularPositionX",
    "LSStartPositionY": "LSLogs_AngularPositionY",
    "LSStartPositionZ": "LSLogs_AngularPosition_Z",
    "RSStartPositionX": "RSLogs_AngularPositionX",
    "RSStartPositionY": "RSLogs_AngularPositionY",
    "RSStartPositionZ": "RSLogs_AngularPosition_Z",
    "LEStartPosition": "AngularKinematicsLogs_LEAngularPosition",
    "REStartPosition": "AngularKinematicsLogs_REAngularPosition",
    "LFStartPosition": "LFLogs_AngularPosition",
    "RFStartPosition": "RFLogs_AngularPosition",
}

#: Our body -> Simscape sensor whose logged rotation it reproduces.
SIMSCAPE_SENSORS: dict[str, str] = {
    "torso": "Torso",
    "spine": "Spine",
    "left_scapula": "LScap",
    "right_scapula": "RScap",
    "left_shoulder": "LS",
    "right_shoulder": "RS",
    "left_forearm": "LF",
    "right_forearm": "RF",
}

CONVENTION_NOTE = (
    "upstreamdrift golfer-scapula/2.0: body frames are the GolfSwing3D_Kinetic "
    "sensor frames; angles are the Simscape start-position variables (degrees, "
    "translations metres in the Simscape world: z up, x toward the target, y to "
    "the golfer's right). Conventions validated against the model's logs, "
    "residual <= 2.7e-4 rad (#9714, docs/motion_capture/evidence/simscape_axes.md); "
    "wrist rows are named only."
)


def golfer_model() -> ArticulatedModel:
    return ArticulatedModel(GOLFER_SPEC)


def simscape_variable_names() -> dict[str, str]:
    """Our DOF -> Simscape variable name (signs dropped), for reports."""
    return {ours: theirs for ours, (theirs, _) in SIMSCAPE_NAMES.items()}


def simscape_angles_to_q(
    model: ArticulatedModel, angles_deg: dict[str, np.ndarray]
) -> np.ndarray:
    """State ``(T, n_dof)`` from Simscape variables (degrees); others zero.

    Precondition: ``model`` is the golfer and every array has one length.
    """
    n = len(next(iter(angles_deg.values())))
    q = np.zeros((n, model.n_dof))
    col = {name: i for i, name in enumerate(model.dof_names)}
    for ours, (theirs, sign) in SIMSCAPE_NAMES.items():
        if theirs in angles_deg:
            values = np.asarray(angles_deg[theirs], dtype=float)
            q[:, col[ours]] = sign * (
                values if ours.split(".")[1].startswith("t") else np.radians(values)
            )
    return q


def simscape_rows(
    dof_names: list[str], q: np.ndarray, fps: float
) -> list[dict[str, Any]]:
    """Per-frame rows of the 27 Simscape variables, degrees (translations metres).

    Precondition: ``q`` is ``(T, len(dof_names))``. Postcondition: every row
    carries ``time_s`` and every Simscape name in :data:`SIMSCAPE_NAMES`;
    translations are re-axed into the Simscape world.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(dof_names):
        raise ValueError(f"q must be (T, {len(dof_names)}), got {q.shape}")
    col = {name: i for i, name in enumerate(dof_names)}
    translation = q[:, [col["pelvis.tx"], col["pelvis.ty"], col["pelvis.tz"]]]
    translation = translation @ WORLD_SIMSCAPE_FROM_OURS.T
    rows = []
    for t in range(q.shape[0]):
        row: dict[str, Any] = {"time_s": t / fps}
        for ours, (theirs, sign) in SIMSCAPE_NAMES.items():
            kind = ours.split(".")[1]
            if kind.startswith("t"):
                row[theirs] = float(translation[t, "xyz".index(kind[1])])
            else:
                row[theirs] = float(np.degrees(sign * q[t, col[ours]]))
        rows.append(row)
    return rows


def write_simscape_csv(joint_angles_json: Path, out: Path | None = None) -> Path:
    """``model/joint_angles_simscape.csv`` from a fit's ``joint_angles.json``.

    Precondition: the JSON was written by ``fit_session_model`` for the
    golfer spec (every Simscape DOF present). The first line is a comment
    carrying :data:`CONVENTION_NOTE`.
    """
    payload = json.loads(joint_angles_json.read_text(encoding="utf-8"))
    names = list(payload["dof_names"])
    missing = [n for n in SIMSCAPE_NAMES if n not in names]
    if missing:
        raise ValueError(f"joint angles lack Simscape DOFs: {missing}")
    rows = simscape_rows(
        names, np.asarray(payload["q"], dtype=float), float(payload["fps"])
    )
    target = out or joint_angles_json.with_name("joint_angles_simscape.csv")
    with target.open("w", encoding="utf-8", newline="") as handle:
        handle.write(f"# {CONVENTION_NOTE}\n")
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return target
