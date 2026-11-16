"""Implicit SDF hand visual backend (epic #15150, issue #15155).

``build_hand_visual`` is the ADR-0057 contract builder registered as
``sdf_hand``; ``build_sdf_hand`` poses a skeleton, meshes the smooth-min field, then
applies shaft roll to the vertices. The analytic SDF stays on the unrolled
skeleton; :meth:`SdfHandVisual.sdf` inverse-rotates query points.
"""

from __future__ import annotations

import importlib.util
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
    GripArticulation,
    HandRole,
)
from src.shared.python.model_appearance.hand_backends.joint_angles import (
    HandJointAngles,
)
from src.shared.python.model_appearance.hand_backends.registry import (
    HandVisualBackend,
    register_hand_visual_backend,
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
from src.shared.python.model_appearance.hand_backends.types import HandVisualResult
from src.shared.python.model_appearance.library import MATERIALS

BACKEND_NAME = "sdf_hand"


@dataclass(eq=False)
class SdfHandVisual:
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


def build_sdf_hand(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: GripArticulation,
    *,
    grip_radius_m: float | None = None,
    settings: SdfHandSettings | None = None,
    resolution: int | None = None,
) -> SdfHandVisual:
    """Mesh one SDF hand in the hand-grip frame, then apply shaft roll.

    Preconditions: ``side`` is lead or trail, ``grip_spec`` is a
    :class:`GripSpec`, ``joint_angles`` is a :class:`GripArticulation`, the
    grip radius lies in ``GRIP_RADIUS_RANGE_M``, and ``resolution`` is at
    least the settings minimum. Validation runs before meshing.
    Postcondition: finite vertices, outward winding, positive volume.
    """
    if side not in HAND_ROLES:
        raise ValueError(f"side must be one of {HAND_ROLES}, got {side!r}")
    if not isinstance(grip_spec, GripSpec):
        raise TypeError(f"grip_spec must be a GripSpec, got {type(grip_spec).__name__}")
    if not isinstance(joint_angles, GripArticulation):
        raise TypeError(
            f"joint_angles must be GripArticulation, got {type(joint_angles).__name__}"
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
    return SdfHandVisual(
        backend_id=BACKEND_NAME,
        role=side,
        mesh=rolled,
        grip_radius_m=radius,
        material=loaded.skin_material,
        skeleton=skeleton,
        settings=loaded,
        roll=np.asarray(roll, dtype=float),
    )


def build_hand_visual(
    grip_spec: GripSpec,
    side: HandRole,
    joint_angles: HandJointAngles | None = None,
    *,
    grip_radius_m: float,
    skin_rgb: tuple[float, float, float] | None = None,
    settings: SdfHandSettings | None = None,
) -> HandVisualResult:
    """The SDF hand as an ADR-0057 :class:`HandVisualResult`.

    The SDF hand poses itself from ``grip_spec``; explicit joint angles are
    not supported. ``skin_rgb`` defaults to the settings' skin material.
    """
    if joint_angles is not None:
        raise ValueError("sdf_hand poses itself; pass joint_angles=None")
    built = build_sdf_hand(
        grip_spec,
        side,
        GripArticulation.from_grip_spec(grip_spec, side),
        grip_radius_m=grip_radius_m,
        settings=settings,
    )
    if skin_rgb is None:
        r, g, b, _ = MATERIALS[built.material].base_color
        skin_rgb = (float(r), float(g), float(b))
    return HandVisualResult(
        backend=BACKEND_NAME,
        side=side,
        mesh=built.mesh,
        skin_rgb=skin_rgb,
        grip_radius_m=built.grip_radius_m,
    )


def register() -> None:
    """Add the SDF hand to the shared registry (called lazily).

    Registers only when scikit-image (the ``sdf-hand`` extra) is installed,
    so an install without marching cubes lists no backend it cannot build.
    """
    if importlib.util.find_spec("skimage") is None:
        return
    register_hand_visual_backend(
        HandVisualBackend(
            name=BACKEND_NAME,
            builder=build_hand_visual,
            description="Implicit SDF sculpted hand (smooth-min bones, marching cubes)",
            capabilities=frozenset({"self_posing"}),
        )
    )


__all__ = [
    "BACKEND_NAME",
    "SdfHandVisual",
    "build_hand_visual",
    "build_sdf_hand",
    "register",
]
