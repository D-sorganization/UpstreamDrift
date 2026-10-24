"""Anatomical marker priors for the tour marker set on golf_humanoid (#13434).

Every prior is anchored to the model's own joint geometry, read from the
(scaled) model, so it follows the subject scaling: the knee, ankle and
metatarsophalangeal joint centres, the elbow and wrist joints, the acromial
(glenohumeral) joints, and the pelvis origin at the mid-ASIS. The offset from
the anchor is a declared surface-landmark estimate; ``radius_m`` bounds how
far calibration may move the marker from it. Waist-belt spacing uses the
subject's own capture (fronts +-0.17 m, backs +-0.085 m, 0.134 m apart) because
a belt is not a bony landmark.

Frames (OpenSim body frames): x anterior, y superior, z right; metres. ``s`` is
+1 on the right and -1 on the left, so a lateral offset is ``s * d``.

These priors replace the free OS-3 calibration, which allowed 0.5 m offsets
and converged on a mirrored solution (see #13434). They are declared
estimates, not measurements; the calibration reports each marker's final
deviation from its prior.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import ensure, require
from src.shared.python.engine_core.opensim_runtime import import_opensim

Vec3 = tuple[float, float, float]
Offsets = Mapping[str, tuple[str, Vec3]]

BONY_RADIUS_M = 0.04
SOFT_RADIUS_M = 0.06
HEAD_RADIUS_M = 0.10


@dataclass(frozen=True)
class AnatomicalPrior:
    """A marker's anatomical prior placement and its admissible radius."""

    body: str
    offset_m: Vec3
    radius_m: float
    landmark: str


def _parent_location(joint: Any) -> np.ndarray:
    """Joint centre expressed in the parent body frame."""
    frame = joint.getParentFrame()
    return np.asarray(frame.findTransformInBaseFrame().p().to_numpy(), float)


def _joint_anchors(model: Any, names: tuple[str, ...]) -> dict[str, np.ndarray]:
    joints = model.getJointSet()
    anchors = {}
    for name in names:
        require(joints.contains(name), f"model has no joint {name!r}")
        anchors[name] = _parent_location(joints.get(name))
    return anchors


def _vec(base: np.ndarray, dx: float, dy: float, dz: float) -> Vec3:
    return (float(base[0] + dx), float(base[1] + dy), float(base[2] + dz))


def _leg_priors(side: str, a: Mapping[str, np.ndarray]) -> dict[str, AnatomicalPrior]:
    s = 1.0 if side == "r" else -1.0
    p = "R" if side == "r" else "L"
    knee, ankle, mtp = a[f"walker_knee_{side}"], a[f"ankle_{side}"], a[f"mtp_{side}"]
    return {
        f"{p}KneeOut": AnatomicalPrior(
            f"femur_{side}",
            _vec(knee, 0.0, 0.0, s * 0.05),
            BONY_RADIUS_M,
            "lateral femoral epicondyle",
        ),
        f"{p}AnkleOut": AnatomicalPrior(
            f"tibia_{side}",
            _vec(ankle, -0.005, -0.005, s * 0.035),
            BONY_RADIUS_M,
            "lateral malleolus",
        ),
        f"{p}ToeIn": AnatomicalPrior(
            f"calcn_{side}",
            _vec(mtp, 0.0, 0.005, -s * 0.025),
            BONY_RADIUS_M,
            "first metatarsal head",
        ),
        f"{p}ToeOut": AnatomicalPrior(
            f"calcn_{side}",
            _vec(mtp, -0.03, 0.005, s * 0.035),
            BONY_RADIUS_M,
            "fifth metatarsal head",
        ),
    }


def _arm_priors(side: str, a: Mapping[str, np.ndarray]) -> dict[str, AnatomicalPrior]:
    s = 1.0 if side == "r" else -1.0
    p = "R" if side == "r" else "L"
    elbow, wrist = a[f"elbow_{side}"], a[f"radius_hand_{side}"]
    return {
        f"{p}UArmHigh": AnatomicalPrior(
            f"humerus_{side}",
            (0.0, float(elbow[1] + 0.16), s * 0.045),
            SOFT_RADIUS_M,
            "lateral upper arm, 0.16 m proximal of the elbow",
        ),
        f"{p}ElbowOut": AnatomicalPrior(
            f"humerus_{side}",
            _vec(elbow, 0.0, 0.0, s * 0.04),
            BONY_RADIUS_M,
            "lateral humeral epicondyle",
        ),
        f"{p}WristTop": AnatomicalPrior(
            f"radius_{side}",
            _vec(wrist, -0.025, 0.01, 0.0),
            SOFT_RADIUS_M,
            "dorsal wrist",
        ),
    }


def _pelvis_priors() -> dict[str, AnatomicalPrior]:
    out = {}
    for side, s in (("Right", 1.0), ("Left", -1.0)):
        out[f"Waist{side}"] = AnatomicalPrior(
            "pelvis",
            (-0.02, 0.04, s * 0.17),
            SOFT_RADIUS_M,
            "waist belt, front, over the iliac crest",
        )
        out[f"Waist{side[0]}Back"] = AnatomicalPrior(
            "pelvis",
            (-0.155, 0.09, s * 0.085),
            SOFT_RADIUS_M,
            "waist belt, back",
        )
    return out


def _torso_priors(a: Mapping[str, np.ndarray]) -> dict[str, AnatomicalPrior]:
    gh = a["acromial_r"]
    x0, y0, half = float(gh[0]), float(gh[1]), abs(float(gh[2]))
    out = {
        "BackTop": AnatomicalPrior(
            "torso", (x0 - 0.055, y0 + 0.06, 0.0), SOFT_RADIUS_M, "C7"
        ),
        "HeadTop": AnatomicalPrior(
            "torso", (x0 - 0.02, y0 + 0.34, 0.0), HEAD_RADIUS_M, "vertex"
        ),
        "HeadFront": AnatomicalPrior(
            "torso", (x0 + 0.08, y0 + 0.26, 0.0), HEAD_RADIUS_M, "forehead"
        ),
        "HeadSide": AnatomicalPrior(
            "torso", (x0, y0 + 0.26, -0.08), HEAD_RADIUS_M, "left temple"
        ),
    }
    for side, s in (("Right", 1.0), ("Left", -1.0)):
        out[f"Back{side}"] = AnatomicalPrior(
            "torso",
            (x0 - 0.09, y0 - 0.08, s * 0.09),
            SOFT_RADIUS_M,
            f"{side.lower()} scapula",
        )
    for p, s in (("R", 1.0), ("L", -1.0)):
        out[f"{p}ShoulderTop"] = AnatomicalPrior(
            "torso",
            (x0, y0 + 0.035, s * 0.96 * half),
            BONY_RADIUS_M,
            "acromion",
        )
        out[f"{p}ShoulderBack"] = AnatomicalPrior(
            "torso",
            (x0 - 0.07, y0 + 0.01, s * 0.72 * half),
            SOFT_RADIUS_M,
            "posterior shoulder",
        )
    return out


_ANCHOR_JOINTS = tuple(
    f"{j}_{side}"
    for side in ("r", "l")
    for j in ("walker_knee", "ankle", "mtp", "elbow", "radius_hand", "acromial")
)


def anatomical_marker_priors(model_path: Path | str) -> dict[str, AnatomicalPrior]:
    """Anatomical priors for the 28 body markers of the tour marker set.

    Precondition: the model file exists and has the golf_humanoid joints.
    Postcondition: 28 priors, every lateral marker on its labelled side
    (left z < 0, right z > 0) and every radius positive. Club markers have
    no anatomical prior and are not returned.
    """
    path = Path(model_path)
    require(path.is_file(), f"model not found: {path}")
    opensim = import_opensim()
    anchors = _joint_anchors(opensim.Model(str(path)), _ANCHOR_JOINTS)
    priors = {**_pelvis_priors(), **_torso_priors(anchors)}
    for side in ("r", "l"):
        priors.update(_leg_priors(side, anchors))
        priors.update(_arm_priors(side, anchors))
    ensure(len(priors) == 28, f"expected 28 priors, got {len(priors)}")
    ensure(
        all(lateral_side_ok(name, p.offset_m) for name, p in priors.items()),
        "a prior lies on the wrong side",
    )
    return priors


def marker_side(name: str) -> int:
    """+1 for a right-side label, -1 for a left-side label, 0 for midline."""
    if (
        "Right" in name
        or name == "WaistRBack"
        or (name[:1] == "R" and name[1:2].isupper())
    ):
        return 1
    if (
        "Left" in name
        or name == "WaistLBack"
        or (name[:1] == "L" and name[1:2].isupper())
    ):
        return -1
    return 0


def lateral_side_ok(name: str, offset: Vec3, tol_m: float = 0.0) -> bool:
    """Whether a lateral marker's z lies on its labelled side (midline: always).

    Medial markers (``ToeIn``) sit near the foot axis, so only their lateral
    counterparts are side-checked strictly; a medial marker passes when it is
    within ``max(tol_m, 0.05)`` m of the midline of its own body.
    """
    side = marker_side(name)
    if side == 0:
        return True
    z = float(offset[2])
    if name.endswith("ToeIn"):
        return side * z >= -max(tol_m, 0.05)
    return side * z > -tol_m


def clamp_to_priors(
    offsets: Offsets, priors: Mapping[str, AnatomicalPrior]
) -> dict[str, tuple[str, Vec3]]:
    """Project every prior-bearing offset into its prior's ball.

    Postcondition: each returned offset of a prior-bearing marker lies within
    ``radius_m`` of the prior and keeps its body; other markers are unchanged.
    """
    out: dict[str, tuple[str, Vec3]] = {}
    for name, (body, offset) in offsets.items():
        prior = priors.get(name)
        if prior is None:
            out[name] = (body, tuple(float(v) for v in offset))  # type: ignore[assignment]
            continue
        require(body == prior.body, f"{name} is on {body}, prior on {prior.body}")
        centre = np.asarray(prior.offset_m, float)
        delta = np.asarray(offset, float) - centre
        dist = float(np.linalg.norm(delta))
        if dist > prior.radius_m:
            delta *= prior.radius_m / dist
        point = centre + delta
        out[name] = (body, (float(point[0]), float(point[1]), float(point[2])))
    return out
