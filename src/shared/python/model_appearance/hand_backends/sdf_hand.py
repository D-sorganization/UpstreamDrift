"""Implicit SDF hand visual backend (epic #15150, issue #15155).

``build_hand_visual`` poses a skeleton, meshes the smooth-min field, then
applies shaft roll to the vertices. The analytic SDF stays on the unrolled
skeleton; :meth:`HandVisualResult.sdf` inverse-rotates query points.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_spec.shaft_roll import (
    APPEARANCE_GRIP_SHAFT_AXIS,
    apply_shaft_roll,
    resolved_shaft_roll_deg,
)
from src.shared.python.grip_spec.types import GripSpec
from src.shared.python.model_appearance.geometry import Mesh
from src.shared.python.model_appearance.grip_hand_mesh import GRIP_RADIUS_RANGE_M
from src.shared.python.model_appearance.hand_backends.contract import (
    HAND_ROLES,
    HandJointAngles,
    HandRole,
)
from src.shared.python.model_appearance.hand_backends.sdf_field import evaluate_hand_sdf
from src.shared.python.model_appearance.hand_backends.sdf_mesh import (
    marching_cubes_mesh,
)
from src.shared.python.model_appearance.hand_backends.sdf_pose import (
    HandSkeleton,
    pose_hand_skeleton,
)
from src.shared.python.model_appearance.hand_backends.sdf_settings import (
    SdfHandSettings,
    load_sdf_hand_settings,
)


@dataclass(eq=False)
class HandVisualResult:
    """One posed SDF hand. Array fields are excluded from equality."""

    backend_id: str
    role: HandRole
    mesh: Mesh
    grip_radius_m: float
    material: str
    skeleton: HandSkeleton
    settings: SdfHandSettings
    roll: np.ndarray

    def sdf(self, points: np.ndarray) -> np.ndarray:
        """Hand SDF for points in the shaft-rolled grip frame.

        Shaft roll is ``vertices @ roll.T``, so the inverse is ``points @ roll``.
        """
        local = np.asarray(points, dtype=float) @ np.asarray(self.roll, dtype=float)
        return evaluate_hand_sdf(
            self.skeleton.primitives, local, dict(self.settings.blend_m)
        )


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles,
    *,
    grip_radius_m: float | None = None,
    settings: SdfHandSettings | None = None,
    resolution: int | None = None,
) -> HandVisualResult:
    """Mesh one SDF hand in the hand-grip frame, then apply shaft roll.

    Preconditions: ``side`` is lead or trail, ``grip_spec`` is a
    :class:`GripSpec`, ``joint_angles`` is a :class:`HandJointAngles`, the
    grip radius lies in ``GRIP_RADIUS_RANGE_M``, and ``resolution`` is at
    least the settings minimum. Validation runs before meshing.
    Postcondition: finite vertices, outward winding, positive volume.
    """
    if side not in HAND_ROLES:
        raise ValueError(f"side must be one of {HAND_ROLES}, got {side!r}")
    if not isinstance(grip_spec, GripSpec):
        raise TypeError(f"grip_spec must be a GripSpec, got {type(grip_spec).__name__}")
    if not isinstance(joint_angles, HandJointAngles):
        raise TypeError(
            f"joint_angles must be HandJointAngles, got {type(joint_angles).__name__}"
        )
    loaded = load_sdf_hand_settings() if settings is None else settings
    if not isinstance(loaded, SdfHandSettings):
        raise TypeError("settings must be SdfHandSettings")
    radius = (
        loaded.default_grip_radius_m if grip_radius_m is None else float(grip_radius_m)
    )
    lo, hi = GRIP_RADIUS_RANGE_M
    if not (lo <= radius <= hi):
        raise ValueError(f"grip_radius_m must be in [{lo}, {hi}] m, got {radius}")
    samples = loaded.resolution if resolution is None else int(resolution)
    if samples < loaded.min_resolution:
        raise ValueError(
            f"resolution must be >= {loaded.min_resolution}, got {samples}"
        )
    skeleton = pose_hand_skeleton(
        grip_spec,
        side,
        joint_angles,
        grip_radius_m=radius,
        settings=loaded,
    )
    mesh = marching_cubes_mesh(
        skeleton.primitives,
        dict(loaded.blend_m),
        resolution=samples,
        bounds_pad_m=loaded.bounds_pad_m,
        taubin_iterations=loaded.taubin_iterations,
        taubin_lambda=loaded.taubin_lambda,
        taubin_mu=loaded.taubin_mu,
    )
    roll = apply_shaft_roll(
        np.eye(3),
        APPEARANCE_GRIP_SHAFT_AXIS,
        resolved_shaft_roll_deg(grip_spec, side),
    )[:3, :3]
    rolled = Mesh(mesh.vertices @ roll.T, mesh.faces)
    if not np.isfinite(rolled.vertices).all():
        raise ValueError("SDF hand mesh has non-finite vertices")
    if rolled.volume() <= 0.0:
        raise ValueError("SDF hand mesh volume must be positive after shaft roll")
    return HandVisualResult(
        backend_id="sdf_hand",
        role=side,
        mesh=rolled,
        grip_radius_m=radius,
        material=loaded.skin_material,
        skeleton=skeleton,
        settings=loaded,
        roll=np.asarray(roll, dtype=float),
    )


__all__ = ["HandVisualResult", "build_hand_visual"]
