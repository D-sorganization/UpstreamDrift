"""The articulated golfer, after the MATLAB 3-D golf model (#9710).

Topology from ``src/engines/Simscape_Multibody_Models/3D_Golf_Model``
(``matlab/MATLAB_GOLF_MODEL_GUIDE.md`` lines 52-66,
``matlab/src/functions/model/readJointStateTargets_GolfSwing3D.m`` lines
12-50, ``matlab/motion_matching/shared/compute_skeleton_fk.m`` lines 36-47
and 159-165):

    World -[Hip 6-DOF]- Pelvis -[Spine universal Rx Ry]- Spine
    Spine -[Torso revolute Rz, axial]- Torso top = Hub
    Hub -[Scapula universal Rx Ry, centred at the hub]- scapula strut
         (``HubtoSLength``, 0.254 m) -[Shoulder gimbal Rx Ry Rz]- Upper arm
    Upper arm -[Elbow revolute]- Forearm -[Forearm revolute, pro/supination]-
    -[Wrist universal]- Hand

That is the scapula the user means: a strut of fixed length from an upper
pivot (the hub, base of the neck) to the glenohumeral joint, with two
degrees of freedom at the pivot. The MATLAB model has no legs and no head;
the detectors observe hips, knees, ankles and the nose, so those are added
here and marked as such. Joint-angle names follow the Simscape
``*StartPosition*`` / ``*AngularPosition*`` vocabulary for the 27 shared DOFs
so the fit can drive the MATLAB model later; axis conventions are recorded,
not yet validated against Simscape, and the export says so.

World frame (ADR-0041): x toward the target, y up, z to the golfer's right.
Rest pose: standing, arms hanging (-y), scapula struts along ±z.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .kinematics import ArticulatedModel, Joint, ModelSpec
from .session import LandmarkMap

PI = float(np.pi)
SCAPULA_RANGE = ((-0.7, 0.7), (-0.7, 0.7))  # about 40 degrees of scapular travel
SHOULDER_RANGE = ((-PI, PI), (-PI, PI), (-PI, PI))
ELBOW_RANGE = ((0.0, 2.6),)
FOREARM_RANGE = ((-1.6, 1.6),)
WRIST_RANGE = ((-1.3, 1.3), (-0.7, 0.7))
SPINE_RANGE = ((-0.9, 0.9), (-0.9, 0.9))
TORSO_RANGE = ((-2.0, 2.0),)
HIP_ROT_RANGE = ((-1.6, 1.6), (-1.6, 1.6), (-1.0, 1.0))
KNEE_RANGE = ((0.0, 2.6),)

#: Default segment lengths, metres. MATLAB values where the model has them
#: (``compute_skeleton_fk.m`` 251-260, model workspace in inches x 0.0254).
DEFAULT_LENGTHS_M: dict[str, float] = {
    "lower_torso": 0.15,  # pelvis -> spine joint (LowerTorsoLength)
    "upper_torso": 0.3048,  # spine -> hub (UpperTorsoLength)
    "hub_to_shoulder": 0.254,  # scapula strut (HubtoSLength)
    "head": 0.18,  # hub -> nose (not in the MATLAB model)
    "upper_arm": 0.3048,  # UpperArmLength
    "forearm": 0.26,  # elbow -> wrist centre (LowerArmLength incl. hand is 0.3556)
    "hip_half": 0.10,  # pelvis -> hip centre (legs: not in the MATLAB model)
    "thigh": 0.44,
    "shank": 0.42,
}


def _arm(side: str, sign: float) -> tuple[Joint, ...]:
    return (
        Joint(
            f"{side}_scapula", "hub", (0.0, 0.0, 0.0), None, "xy", SCAPULA_RANGE, False
        ),
        Joint(
            f"{side}_shoulder",
            f"{side}_scapula",
            (0.0, 0.0, sign),
            "hub_to_shoulder",
            "xyz",
            SHOULDER_RANGE,
        ),
        Joint(
            f"{side}_elbow",
            f"{side}_shoulder",
            (0.0, -1.0, 0.0),
            "upper_arm",
            "x",
            ELBOW_RANGE,
        ),
        Joint(
            f"{side}_forearm",
            f"{side}_elbow",
            (0.0, 0.0, 0.0),
            None,
            "y",
            FOREARM_RANGE,
            False,
        ),
        Joint(
            f"{side}_wrist",
            f"{side}_forearm",
            (0.0, -1.0, 0.0),
            "forearm",
            "xz",
            WRIST_RANGE,
        ),
    )


def _leg(side: str, sign: float) -> tuple[Joint, ...]:
    return (
        Joint(
            f"{side}_hip", "pelvis", (0.0, 0.0, sign), "hip_half", "xyz", HIP_ROT_RANGE
        ),
        Joint(
            f"{side}_knee", f"{side}_hip", (0.0, -1.0, 0.0), "thigh", "x", KNEE_RANGE
        ),
        Joint(f"{side}_ankle", f"{side}_knee", (0.0, -1.0, 0.0), "shank", ""),
    )


GOLFER_SPEC = ModelSpec(
    name="golfer-scapula/1.0",
    joints=(
        Joint("pelvis", None, axes="xyz"),  # Hip 6-DOF: translation + rotation
        # Simscape has z up: its Spine Rx Ry (forward bend, side bend) are our
        # x and z; its Torso Rz (axial rotation about the spine) is our y.
        Joint(
            "spine", "pelvis", (0.0, 1.0, 0.0), "lower_torso", "xz", SPINE_RANGE, False
        ),
        Joint("hub", "spine", (0.0, 1.0, 0.0), "upper_torso", "y", TORSO_RANGE),
        Joint("nose", "hub", (0.0, 1.0, 0.0), "head", "", landmark=True),
        *_arm("left", -1.0),
        *_arm("right", 1.0),
        *_leg("left", -1.0),
        *_leg("right", 1.0),
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

#: Our DOF -> Simscape start-position variable (27 shared DOFs). Legs and the
#: head have no MATLAB counterpart. Axis conventions are named, not verified.
SIMSCAPE_NAMES: dict[str, str] = {
    "pelvis.tx": "TranslationStartPositionX",
    "pelvis.ty": "TranslationStartPositionY",
    "pelvis.tz": "TranslationStartPositionZ",
    "pelvis.rx": "HipStartPositionX",
    "pelvis.ry": "HipStartPositionY",
    "pelvis.rz": "HipStartPositionZ",
    "spine.rx": "SpineStartPositionX",
    "spine.rz": "SpineStartPositionY",
    "hub.ry": "TorsoStartPosition",
    "left_scapula.rx": "LScapStartPositionX",
    "left_scapula.ry": "LScapStartPositionY",
    "right_scapula.rx": "RScapStartPositionX",
    "right_scapula.ry": "RScapStartPositionY",
    "left_shoulder.rx": "LSStartPositionX",
    "left_shoulder.ry": "LSStartPositionY",
    "left_shoulder.rz": "LSStartPositionZ",
    "right_shoulder.rx": "RSStartPositionX",
    "right_shoulder.ry": "RSStartPositionY",
    "right_shoulder.rz": "RSStartPositionZ",
    "left_elbow.rx": "LEStartPosition",
    "right_elbow.rx": "REStartPosition",
    "left_forearm.ry": "LFStartPosition",
    "right_forearm.ry": "RFStartPosition",
    "left_wrist.rx": "LWStartPositionX",
    "left_wrist.rz": "LWStartPositionY",
    "right_wrist.rx": "RWStartPositionX",
    "right_wrist.rz": "RWStartPositionY",
}

CONVENTION_NOTE = (
    "upstreamdrift golfer-scapula/1.0: intrinsic xyz, radians internally, world "
    "x toward target / y up / z to the golfer's right. Names follow the Simscape "
    "start-position vocabulary; axis signs and orders are NOT yet validated "
    "against GolfSwing3D_Kinetic (#9714)."
)


def golfer_model() -> ArticulatedModel:
    return ArticulatedModel(GOLFER_SPEC)


def simscape_rows(
    dof_names: list[str], q: np.ndarray, fps: float
) -> list[dict[str, Any]]:
    """Per-frame rows of the 27 Simscape variables, degrees (translations metres).

    Precondition: ``q`` is ``(T, len(dof_names))``. Postcondition: every row
    carries ``time_s`` and every Simscape name in :data:`SIMSCAPE_NAMES`.
    """
    q = np.asarray(q, dtype=float)
    if q.ndim != 2 or q.shape[1] != len(dof_names):
        raise ValueError(f"q must be (T, {len(dof_names)}), got {q.shape}")
    col = {name: i for i, name in enumerate(dof_names)}
    rows = []
    for t in range(q.shape[0]):
        row: dict[str, Any] = {"time_s": t / fps}
        for ours, theirs in SIMSCAPE_NAMES.items():
            value = float(q[t, col[ours]])
            row[theirs] = (
                value if ours.split(".")[1].startswith("t") else np.degrees(value)
            )
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
