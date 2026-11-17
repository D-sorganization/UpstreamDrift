"""Shared-frame offsets when drawing lead and trail together."""

from __future__ import annotations

import numpy as np

from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_pose import Hand


def axial_offset_vs_lead_m(grip_spec: GripSpec, role: str) -> float:
    """Shift along hand-grip +z for trail relative to lead in club axial order."""
    pose = grip_spec.to_grip_pose()
    lead_z = pose.placement(Hand.LEAD).distance_below_butt_m
    if role == "lead":
        return 0.0
    trail_z = pose.placement(Hand.TRAIL).distance_below_butt_m
    return float(trail_z - lead_z)


def translate_mesh(mesh: Mesh, delta: np.ndarray) -> Mesh:
    return Mesh(mesh.vertices + np.asarray(delta, float), mesh.faces.copy())


def mesh_in_shared_visual_frame(mesh: Mesh, grip_spec: GripSpec, role: str) -> Mesh:
    dz = axial_offset_vs_lead_m(grip_spec, role)
    return translate_mesh(mesh, np.array([0.0, 0.0, dz], dtype=float))


__all__ = ["axial_offset_vs_lead_m", "mesh_in_shared_visual_frame", "translate_mesh"]
