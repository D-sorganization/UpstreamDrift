"""Estimate torso inertias from a bounded frame prior on newly sized models.

The de Leva segment moments are retained at zero. An affine transverse scale
changes the mass-normalized second moment while keeping mass and COM fixed.
This is a physical estimate, not a measurement or the visual skin integral.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Mapping

import numpy as np

from src.shared.python.model_appearance.skinned_body import spec_body_poses
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
    """Return a copied candidate with estimated torso inertias; reject unsupported frames."""
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
            inertia = np.asarray(solid["inertia_com_kg_m2"], dtype=float)
            if (
                inertia.shape != (3, 3)
                or not np.isfinite(inertia).all()
                or not np.allclose(inertia, inertia.T, atol=1e-12, rtol=0)
            ):
                raise ValueError(f"invalid inertia for torso solid {name}")
            second = 0.5 * np.trace(inertia) * np.eye(3) - inertia
            if np.linalg.eigvalsh(second).min() <= 0:
                raise ValueError(f"unrealizable inertia for torso solid {name}")
            moment = affine @ second @ affine.T
            solid["inertia_com_kg_m2"] = (
                np.trace(moment) * np.eye(3) - moment
            ).tolist()
    result["upper_body_sha256"] = canonical_sha256(upper_body_slice(result))
    return result
