"""Estimate torso and limb inertias from a bounded frame prior on newly sized models.

The original segment moments are retained at zero. An affine transverse scale
changes the mass-normalized second moment while keeping mass and COM fixed.
This is a physical estimate, not a measurement or the visual skin integral.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.pelvis_frame import _check_rigid
from src.shared.python.motion_matching.full_body_spec import (
    canonical_sha256,
    upper_body_slice,
)

TORSO_SOLIDS = {
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso": ("pelvis",),
    "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/COMRod": (
        "middle_trunk",
        "upper_trunk",
    ),
}


_ARM = "solid_reference:GolfSwing3D_Kinetic/"
LIMB_SOLIDS = {
    _ARM + "LUpperArm": (
        "upper_arm",
        _ARM + "HubtoLS",
        _ARM + "Left Elbow Joint/Spherical Solid",
    ),
    _ARM + "Left Elbow Joint/Spherical Solid": (
        "forearm_proximal",
        _ARM + "LUpperArm",
        _ARM + "Left Forearm/LLowerForearm",
    ),
    _ARM + "Left Forearm/LLowerForearm": (
        "forearm_distal",
        _ARM + "Left Elbow Joint/Spherical Solid",
        _ARM + "Club/Clubface Vector",
    ),
    _ARM + "RUpperArm": (
        "upper_arm",
        _ARM + "HubtoRS",
        _ARM + "Right Elbow Joint/Spherical Solid1",
    ),
    _ARM + "Right Elbow Joint/Spherical Solid1": (
        "forearm_proximal",
        _ARM + "RUpperArm",
        _ARM + "Right Forearm/RLowerForearm",
    ),
    _ARM + "Right Forearm/RLowerForearm": (
        "forearm_distal",
        _ARM + "Right Elbow Joint/Spherical Solid1",
        _ARM + "Grip/RHandStandoff",
    ),
    "femur_l": ("femur_l", _ARM + "Hips and Torso Inputs/LowerTorso", "tibia_l"),
    "femur_r": ("femur_r", _ARM + "Hips and Torso Inputs/LowerTorso", "tibia_r"),
    "tibia_l": ("tibia_l", "femur_l", "talus_l"),
    "tibia_r": ("tibia_r", "femur_r", "talus_r"),
}


def _transform_inertia(solid: dict[str, Any], affine: np.ndarray, name: str) -> None:
    """Transform a realizable COM second moment while retaining mass and COM."""
    inertia = np.asarray(solid["inertia_com_kg_m2"], dtype=float)
    if (
        inertia.shape != (3, 3)
        or not np.isfinite(inertia).all()
        or not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0)
    ):
        raise ValueError(f"invalid inertia for solid {name}")
    second = 0.5 * np.trace(inertia) * np.eye(3) - inertia
    if np.linalg.eigvalsh(second).min() <= 0:
        raise ValueError(f"unrealizable inertia for solid {name}")
    moment = affine @ second @ affine.T
    solid["inertia_com_kg_m2"] = (np.trace(moment) * np.eye(3) - moment).tolist()


def _apply_limb_inertias(
    spec: dict[str, Any], bodies: dict[str, Any], scale: float
) -> None:
    """Apply transverse estimates using validated joint endpoints and solid frames."""
    for body_name, (solid_name, parent_name, child_name) in LIMB_SOLIDS.items():
        body = bodies.get(body_name)
        if body is None:
            raise ValueError(f"missing limb body {body_name}")
        incoming = [j for j in spec["joints"] if j["child"] == body_name]
        outgoing = [j for j in spec["joints"] if j["parent"] == body_name]
        if (
            len(incoming) != 1
            or len(outgoing) != 1
            or incoming[0]["parent"] != parent_name
            or outgoing[0]["child"] != child_name
        ):
            raise ValueError(f"ambiguous limb joint mapping for {body_name}")
        start = _check_rigid(incoming[0]["child_to_follower"], "incoming limb joint")[
            :3, 3
        ]
        end = _check_rigid(outgoing[0]["parent_to_base"], "outgoing limb joint")[:3, 3]
        axis = end - start
        length = float(np.linalg.norm(axis))
        if not 0.05 < length < 1.0:
            raise ValueError(f"unsupported limb joint separation for {body_name}")
        solids = [
            v for v in body["solids"] if v["name"].rsplit("/", 1)[-1] == solid_name
        ]
        if len(solids) != 1:
            raise ValueError(f"expected exactly one limb solid {solid_name}")
        solid = solids[0]
        placement = _check_rigid(solid["placement"], "limb solid placement")
        direction = placement[:3, :3].T @ (axis / length)
        affine = scale * np.eye(3) + (1.0 - scale) * np.outer(direction, direction)
        _transform_inertia(solid, affine, solid_name)


def validate_frame_build(frame_build: float) -> None:
    """Reject nonfinite, nonnumeric or unsupported frame estimates."""
    if (
        isinstance(frame_build, bool)
        or not isinstance(frame_build, (int, float))
        or not math.isfinite(frame_build)
        or abs(frame_build) > 1
    ):
        raise ValueError("frame_build must be finite and in [-1, 1]")


def apply_frame_inertia(spec: Mapping[str, Any], frame_build: float) -> dict[str, Any]:
    """Return a copied candidate with estimated segment inertias; reject unsupported frames."""
    validate_frame_build(frame_build)
    result = copy.deepcopy(dict(spec))
    if frame_build == 0:
        return result
    poses = spec_body_poses(result)
    bodies = {body["name"]: body for body in result["bodies"]}
    scale = 1.0 + 0.18 * frame_build
    affine = np.diag([scale, scale, 1.0])
    for body_name, names in TORSO_SOLIDS.items():
        if body_name not in bodies or body_name not in poses:
            raise ValueError(f"missing expected torso body {body_name}")
        if not np.allclose(poses[body_name][:3, 2], [0, 0, 1], atol=1e-10, rtol=0):
            raise ValueError(f"torso axis is not vertical for {body_name}")
        solids = bodies[body_name]["solids"]
        for name in names:
            found = [
                solid for solid in solids if solid["name"] == f"{body_name}/{name}"
            ]
            if len(found) != 1:
                raise ValueError(f"expected exactly one torso solid {name}")
            solid = found[0]
            if not np.allclose(solid["placement"], np.eye(4), atol=1e-10, rtol=0):
                raise ValueError(f"unsupported placement for torso solid {name}")
            _transform_inertia(solid, affine, name)
    _apply_limb_inertias(result, bodies, scale)
    result["upper_body_sha256"] = canonical_sha256(upper_body_slice(result))
    return result
