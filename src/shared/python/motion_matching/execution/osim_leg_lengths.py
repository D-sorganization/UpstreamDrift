"""Thigh and shank lengths of the OpenSim legs as builder settings (#15296).

The spec builder reads its legs from the OpenSim model file, so their lengths
are that file's stored values: they do not follow subject stature and differ
left to right. A stated length scales one segment of both legs about its own
frame origin to that joint centre distance. Kinematic settings: the mass is
kept and the inertia follows the length only.
"""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

import numpy as np

#: Setting -> (OpenSim body, proximal joint, distal joint) per side suffix.
SEGMENTS = {
    "thigh_length_m": ("femur", "hip", "walker_knee"),
    "shank_length_m": ("tibia", "walker_knee", "ankle"),
}
SIDES = ("r", "l")


def positive_length_m(name: str, value: Any) -> float:
    """``value`` as a positive finite length in m.

    Raises ``TypeError`` for a non-number and ``ValueError`` otherwise.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number in m, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive (m), got {value}")
    return float(value)


def _length(joints: Mapping[str, Any], proximal: str, distal: str) -> float:
    """Distance between two joint centres of one body, in its own frame."""
    span = np.asarray(joints[distal]["parent_translation"], dtype=float) - np.asarray(
        joints[proximal]["child_translation"], dtype=float
    )
    return float(np.linalg.norm(span))


def leg_lengths_m(joints: Mapping[str, Any]) -> dict[str, float]:
    """Joint centre distances of ``read_osim`` joints, m.

    Keys ``thigh_<side>`` (hip to knee) and ``shank_<side>`` (knee to ankle).
    """
    return {
        f"{setting.split('_')[0]}_{side}": _length(
            joints, f"{proximal}_{side}", f"{distal}_{side}"
        )
        for setting, (_, proximal, distal) in SEGMENTS.items()
        for side in SIDES
    }


def _scaled_body(body: Mapping[str, Any], factor: float) -> dict[str, Any]:
    return {
        **body,
        "com": np.asarray(body["com"], dtype=float) * factor,
        "inertia": np.asarray(body["inertia"], dtype=float) * factor**2,
    }


def _scaled_joint(joint: Mapping[str, Any], body: str, factor: float) -> dict[str, Any]:
    out = dict(joint)
    for end in ("parent", "child"):
        if joint[f"{end}_body"] == body:
            out[f"{end}_translation"] = (
                np.asarray(joint[f"{end}_translation"], dtype=float) * factor
            )
    return out


def scale_leg_lengths(
    bodies: Mapping[str, Any],
    joints: Mapping[str, Any],
    thigh_length_m: float | None = None,
    shank_length_m: float | None = None,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """``read_osim`` bodies and joints with both legs at the stated lengths.

    ``thigh_length_m`` is hip joint centre to knee joint centre and
    ``shank_length_m`` knee joint centre to ankle joint centre, m. ``None``
    keeps a segment as stored; with neither the inputs are returned. Every
    frame on a scaled body, its mass centre and (by the factor squared) its
    inertia follow the length. The inputs are never modified.
    Raises ``TypeError`` / ``ValueError`` as :func:`positive_length_m`.
    """
    targets = {"thigh_length_m": thigh_length_m, "shank_length_m": shank_length_m}
    out_bodies, out_joints = dict(bodies), dict(joints)
    for setting, target in targets.items():
        if target is None:
            continue
        length = positive_length_m(setting, target)
        body, proximal, distal = SEGMENTS[setting]
        for side in SIDES:
            name = f"{body}_{side}"
            factor = length / _length(joints, f"{proximal}_{side}", f"{distal}_{side}")
            out_bodies[name] = _scaled_body(bodies[name], factor)
            for key, joint in out_joints.items():
                out_joints[key] = _scaled_joint(joint, name, factor)
    if all(target is None for target in targets.values()):
        return bodies, joints
    return out_bodies, out_joints
