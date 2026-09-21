"""Anthropometric candidate of a full-body document (unqualified by design).

Given a full-body document whose upper body is the Simscape-qualified native
geometry, produce a candidate with the arm and shoulder-link lengths and every
segment's mass, centre of mass and inertia set from de Leva's table for a
stated stature and body mass. Trunk lengths are left alone: the native
topology carries its only trunk joint at the base of the neck, which cannot
be corrected by scaling and is the subject of the anthropometry review.

The candidate changes the upper-body slice, so ``validate_full_body_spec``
rejects it against the qualified base on purpose; engines load it directly
and every receipt must say "anthropometric candidate, unqualified".
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.anthropometry import (
    DE_LEVA_MALE,
    inertia_about_axis,
    segment_parameters,
)
from src.shared.python.motion_matching.segment_scaling import scale_segments

Array: TypeAlias = NDArray[np.float64]


# Native body name suffixes and the de Leva segment each stands for.
UPPER_ARM = {"LUpperArm": "upper_arm", "RUpperArm": "upper_arm"}
FOREARM_PROXIMAL = {"Spherical Solid": "forearm", "Spherical Solid1": "forearm"}
FOREARM_DISTAL = {"LLowerForearm": "forearm", "RLowerForearm": "forearm"}
HUB_LINKS = ("HubtoLS", "HubtoRS")
LEGS = {
    "femur_r": "thigh",
    "femur_l": "thigh",
    "tibia_r": "shank",
    "tibia_l": "shank",
    "calcn_r": "foot",
    "calcn_l": "foot",
}
FOOT_SPLIT = {"calcn": 0.80, "talus": 0.05, "toes": 0.15}  # of the foot mass
BIACROMIAL_FRACTION_OF_STATURE = 0.228  # about 0.39 m at 1.71 m (male mean)


def _suffix(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _transform(value: Any) -> Array:
    m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("Expected a finite 4x4 transform")
    return m


def zero_pose_joint_positions(document: Mapping[str, Any]) -> dict[str, Array]:
    """World position of every joint at the zero pose from the document alone.

    A body's frame is its joint's follower frame (``pose = parent_pose @
    parent_to_base @ inv(child_to_follower)``), matching the engine exporters.
    Postcondition: one entry per joint, in document order.
    """
    poses: dict[str, Array] = {"world": np.eye(4)}
    anchors: dict[str, Array] = {}
    pending = list(document["joints"])
    while pending:
        progressed = False
        for joint in list(pending):
            if joint["parent"] in poses:
                base = poses[joint["parent"]] @ _transform(joint["parent_to_base"])
                anchors[joint["name"]] = base[:3, 3].copy()
                poses[joint["child"]] = base @ np.linalg.inv(
                    _transform(joint["child_to_follower"])
                )
                pending.remove(joint)
                progressed = True
        if not progressed:
            raise ValueError("Joint tree is not rooted at world")
    return anchors


def _joint_of_child(document: Mapping[str, Any], body: str) -> dict:
    return next(j for j in document["joints"] if j["child"] == body)


def _children(document: Mapping[str, Any], body: str) -> list[dict]:
    return [j for j in document["joints"] if j["parent"] == body]


def _segment_axis_in_body(
    document: Mapping[str, Any], body: str
) -> tuple[Array, Array, float]:
    """Unit vector from the body's own joint toward its (first) child joint, body frame."""
    own = _transform(_joint_of_child(document, body)["child_to_follower"])[:3, 3]
    kids = _children(document, body)
    if kids:
        tip = _transform(kids[0]["parent_to_base"])[:3, 3]
    else:
        solids = [s for s in document["bodies"] if s["name"] == body][0]["solids"]
        tip = max(
            (
                _transform(s["placement"])[:3, :3] @ np.asarray(s["com_m"], float)
                + _transform(s["placement"])[:3, 3]
                for s in solids
            ),
            key=lambda p: float(np.linalg.norm(p - own)),
        )
    axis = tip - own
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError(f"Body {body} has no segment axis")
    return axis / norm, own, norm


def _set_segment_solid(
    document: dict,
    body: str,
    segment: str,
    stature_m: float,
    mass_kg: float,
    mass_override: float | None = None,
) -> None:
    params = segment_parameters(stature_m, mass_kg, segment)
    axis, own, length = _segment_axis_in_body(document, body)
    mass = params.mass_kg if mass_override is None else mass_override
    com = own + axis * (DE_LEVA_MALE[segment].com_fraction * length)
    inertia = inertia_about_axis(mass, length, DE_LEVA_MALE[segment].radii, axis)
    entry = next(b for b in document["bodies"] if b["name"] == body)
    entry["solids"] = [
        {
            "name": f"{body}/anthropometric",
            "mass_kg": float(mass),
            "com_m": com.tolist(),
            "inertia_com_kg_m2": inertia.tolist(),
            "placement": np.eye(4).tolist(),
        }
    ]


def _scale_for_length(document: Mapping[str, Any], body: str, target_m: float) -> float:
    """Scale factor for ``scale_segments`` so own-joint to child-joint spans ``target_m``.

    A body's own joint offset (``child_to_follower``) is not scaled by
    ``scale_segments``; only its child joint offset is, so the factor solves
    ``own + s * child = target``. Precondition: positive target, one child.
    """
    if target_m <= 0:
        raise ValueError("Target length must be positive")
    own = float(
        np.linalg.norm(
            _transform(_joint_of_child(document, body)["child_to_follower"])[:3, 3]
        )
    )
    kids = _children(document, body)
    if len(kids) != 1:
        raise ValueError(f"Body {body} must carry exactly one child joint to be scaled")
    child = float(np.linalg.norm(_transform(kids[0]["parent_to_base"])[:3, 3]))
    if child < 1e-9:
        raise ValueError(f"Body {body} has no child offset to scale")
    factor = (target_m - own) / child
    if factor <= 0:
        raise ValueError(
            f"Target {target_m} m is shorter than the fixed offset of {body}"
        )
    return factor


def anthropometric_candidate(
    document: Mapping[str, Any], *, stature_m: float, mass_kg: float
) -> dict[str, Any]:
    """Return the candidate document (deep-copied, own provenance and subject block).

    Lengths: upper arms and the two forearm bodies scaled so shoulder to elbow
    and elbow to wrist match de Leva; hub links scaled to half the biacromial
    breadth. Masses and inertias: every arm, hub-link, trunk and leg body set
    from de Leva; the club and hands are kept. Postcondition: the total mass
    of the candidate is within 10 % of ``mass_kg``.
    """
    if stature_m <= 0 or mass_kg <= 0:
        raise ValueError("Stature and mass must be positive")
    doc = json_copy(document)
    names = {_suffix(b["name"]): b["name"] for b in doc["bodies"]}
    anchors = zero_pose_joint_positions(doc)

    def length(joint_a: str, joint_b: str) -> float:
        return float(np.linalg.norm(anchors[joint_a] - anchors[joint_b]))

    upper_arm = segment_parameters(stature_m, mass_kg, "upper_arm").length_m
    forearm = segment_parameters(stature_m, mass_kg, "forearm").length_m
    scales: dict[str, float] = {}
    for side in ("L", "R"):
        word = "Left" if side == "L" else "Right"

        def joint_named(part: str, word: str = word) -> str:
            return next(
                j["name"] for j in doc["joints"] if f"{word} {part}" in j["name"]
            )

        shoulder = joint_named("Shoulder Joint")
        elbow = joint_named("Elbow Joint")
        forearm_joint = joint_named("Forearm")
        wrist = joint_named("Wrist")
        scap = joint_named("Scapula")
        scales[names[f"{side}UpperArm"]] = _scale_for_length(
            doc, names[f"{side}UpperArm"], upper_arm
        )
        proximal = length(elbow, forearm_joint)
        distal = length(forearm_joint, wrist)
        total = proximal + distal
        proximal_body = names["Spherical Solid" if side == "L" else "Spherical Solid1"]
        scales[proximal_body] = _scale_for_length(
            doc, proximal_body, forearm * proximal / total
        )
        distal_body = names[f"{side}LowerForearm"]
        scales[distal_body] = _scale_for_length(
            doc, distal_body, forearm * distal / total
        )
        scales[names[f"Hubto{side}S"]] = _scale_for_length(
            doc, names[f"Hubto{side}S"], BIACROMIAL_FRACTION_OF_STATURE * stature_m / 2
        )
    doc = scale_segments(doc, scales)

    trunk = segment_parameters(stature_m, mass_kg, "trunk")
    head = segment_parameters(stature_m, mass_kg, "head")
    _set_segment_solid(
        doc, names["LowerTorso"], "trunk", stature_m, mass_kg, trunk.mass_kg
    )
    _set_segment_solid(doc, names["COMRod"], "head", stature_m, mass_kg, head.mass_kg)
    _set_segment_solid(
        doc, names["UpperTorsoBase"], "lower_trunk", stature_m, mass_kg, 0.5
    )
    for link in HUB_LINKS:
        _set_segment_solid(doc, names[link], "upper_arm", stature_m, mass_kg, 1.0)
    body_segment_shares = [
        (body, segment, 0.5 if segment == "forearm" else 1.0)
        for body, segment in {**UPPER_ARM, **FOREARM_PROXIMAL, **FOREARM_DISTAL}.items()
    ] + [
        (body, segment, FOOT_SPLIT["calcn"] if segment == "foot" else 1.0)
        for body, segment in LEGS.items()
    ]
    for body, segment, share in body_segment_shares:
        params = segment_parameters(stature_m, mass_kg, segment)
        _set_segment_solid(
            doc,
            names[body],
            segment,
            stature_m,
            mass_kg,
            params.mass_kg * share,
        )
    foot = segment_parameters(stature_m, mass_kg, "foot").mass_kg
    for side in ("r", "l"):
        for part in ("talus", "toes"):
            entry = next(
                b for b in doc["bodies"] if b["name"] == names[f"{part}_{side}"]
            )
            for solid in entry["solids"]:
                if solid["mass_kg"] > 0:
                    ratio = FOOT_SPLIT[part] * foot / solid["mass_kg"]
                    solid["mass_kg"] = FOOT_SPLIT[part] * foot
                    solid["inertia_com_kg_m2"] = (
                        ratio * np.asarray(solid["inertia_com_kg_m2"])
                    ).tolist()
    total = sum(s["mass_kg"] for b in doc["bodies"] for s in b["solids"])
    if abs(total - mass_kg) > 0.10 * mass_kg:
        raise ValueError(
            f"Candidate mass {total:.1f} kg is not within 10 % of {mass_kg} kg"
        )
    doc["subject"] = {
        "stature_m": stature_m,
        "mass_kg": mass_kg,
        "source": "de Leva 1996 male table; stature from segment proportions",
    }
    doc["upper_body_qualification"] = (
        "anthropometric candidate, unqualified (arm and shoulder-link lengths, masses and inertias changed)"
    )
    doc["provenance"] = (
        str(doc.get("provenance", ""))
        + f" | anthropometric candidate at {stature_m:.3f} m, {mass_kg:.1f} kg"
    )
    return doc


def json_copy(document: Mapping[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(document))
