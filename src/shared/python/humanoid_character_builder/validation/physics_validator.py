"""
Physics validation for humanoid models.

This module provides validation checks for:
- Inertia tensor physical validity
- Static stability
- Collision detection
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from humanoid_character_builder.core.model import (
    GeneratedLink,
    HumanoidModel,
)


@dataclass
class ValidationResult:
    """Result of a validation check."""

    is_valid: bool
    messages: list[str] = field(default_factory=list)

    @classmethod
    def ok(cls) -> ValidationResult:
        """Create a passing validation result with no messages."""
        return cls(True, [])

    @classmethod
    def error(cls, msg: str) -> ValidationResult:
        """Create a failing validation result with a single error *msg*."""
        return cls(False, [f"ERROR: {msg}"])

    @classmethod
    def warning(cls, msg: str) -> ValidationResult:
        """Create a passing validation result carrying a single warning *msg*."""
        return cls(True, [f"WARNING: {msg}"])

    def add_error(self, msg: str) -> None:
        """Append an error message and mark the result as invalid."""
        if not (msg is not None):
            raise ValueError("msg must be provided")
        self.is_valid = False
        self.messages.append(f"ERROR: {msg}")

    def add_warning(self, msg: str) -> None:
        """Append a warning message without changing validity status."""
        self.messages.append(f"WARNING: {msg}")


@dataclass
class StabilityResult:
    """Result of static stability analysis."""

    is_stable: bool
    margin: float
    tipping_angle: float  # radians


class PhysicsValidator:
    """Validator for physical properties of the humanoid model."""

    def validate_inertia(self, link: GeneratedLink) -> ValidationResult:
        """
        Check inertia tensor is physically valid.

        Checks:
        - Symmetry
        - Positive definiteness
        - Triangle inequality
        """
        if not (link is not None):
            raise ValueError("link must be provided")
        result = ValidationResult.ok()

        try:
            inertia_mat = link.inertia.as_matrix()
        except (ValueError, TypeError, AttributeError) as e:
            return ValidationResult.error(f"Failed to get inertia matrix: {e}")

        # Symmetric check
        if not np.allclose(inertia_mat, inertia_mat.T, atol=1e-6):
            result.add_error("Inertia must be symmetric")

        # Positive definite check
        # Eigenvalues must be positive
        try:
            eigvals = np.linalg.eigvals(inertia_mat)
            if not np.all(eigvals > 0):
                result.add_error("Inertia must be positive definite")
        except np.linalg.LinAlgError:
            result.add_error("Inertia matrix singular or invalid")

        # Triangle inequality
        # For principal moments I1, I2, I3 (eigenvalues):
        # I1 + I2 >= I3, etc.
        # The trace-based inequality Ixx + Iyy >= Izz is true for ANY frame.

        ixx, iyy, izz = inertia_mat[0, 0], inertia_mat[1, 1], inertia_mat[2, 2]
        if not (ixx + iyy >= izz and iyy + izz >= ixx and izz + ixx >= iyy):
            result.add_warning("Inertia violates triangle inequality (diagonal check)")

        return result

    def check_static_stability(self, model: HumanoidModel) -> StabilityResult:
        """
        Analyze static balance of the model.

        Assumes the model is in its default configuration (usually T-pose or A-pose).
        Checks if the global Center of Mass (COM) projects into the support polygon.
        """
        if not (model is not None):
            raise ValueError("model must be provided")
        com = model.compute_center_of_mass()
        support = model.compute_support_polygon()

        # Project COM to XY plane
        com_xy = (com[0], com[1])
        com_z = com[2]

        is_stable = support.contains(com_xy)
        margin = support.distance_to_edge(com_xy)

        tipping_angle = 0.0
        if is_stable and com_z > 0 and margin > 0:
            # Angle to tip over: atan(margin / com_height)
            tipping_angle = np.arctan(margin / com_z)
        elif not is_stable:
            # If unstable, tipping angle is undefined or 0?
            # Or negative angle indicating it's already tipped?
            tipping_angle = 0.0

        return StabilityResult(
            is_stable=is_stable, margin=margin, tipping_angle=tipping_angle
        )

    def check_self_collisions(self, model: HumanoidModel) -> list[str]:
        """
        Check for self-collisions (intersections) between links.

        Uses Axis-Aligned Bounding Box (AABB) checks in the global frame.
        Skips adjacent links (parent-child).

        Returns:
            List of messages describing detected collisions.
        """
        if not (model is not None):
            raise ValueError("model must be provided")
        messages = []

        transforms = model.get_global_transforms()
        links = list(model.links.keys())

        # Precompute AABBs
        aabbs = {}
        for name in links:
            if name not in transforms:
                continue

            link = model.links[name]
            geom = link.collision_geometry or link.visual_geometry

            # Define local bounds based on geometry type
            min_bound = np.array([-0.05, -0.05, -0.05])
            max_bound = np.array([0.05, 0.05, 0.05])

            if geom:
                gtype = geom.get("type")
                if gtype == "box":
                    s = geom.get("size", (0.1, 0.1, 0.1))
                    min_bound = np.array([-s[0] / 2, -s[1] / 2, -s[2] / 2])
                    max_bound = np.array([s[0] / 2, s[1] / 2, s[2] / 2])
                elif gtype in ("cylinder", "capsule"):
                    r = geom.get("radius", 0.05)
                    cyl_len = geom.get("length", 0.1)
                    # Cylinder along Z usually (in URDF primitive)
                    min_bound = np.array([-r, -r, -cyl_len / 2])
                    max_bound = np.array([r, r, cyl_len / 2])
                elif gtype == "sphere":
                    r = geom.get("radius", 0.05)
                    min_bound = np.array([-r, -r, -r])
                    max_bound = np.array([r, r, r])

            # Transform 8 corners to global
            corners = []
            for x in [min_bound[0], max_bound[0]]:
                for y in [min_bound[1], max_bound[1]]:
                    for z in [min_bound[2], max_bound[2]]:
                        corners.append([x, y, z])

            T = transforms[name]
            global_corners = []
            for c in corners:
                gc = T[:3, :3] @ np.array(c) + T[:3, 3]
                global_corners.append(gc)

            global_corners_array = np.array(global_corners)
            aabb_min = np.min(global_corners_array, axis=0)
            aabb_max = np.max(global_corners_array, axis=0)
            aabbs[name] = (aabb_min, aabb_max)

        # Check intersections
        for i, name1 in enumerate(links):
            for j in range(i + 1, len(links)):
                name2 = links[j]

                # Skip if adjacent
                # Check if name1 is parent of name2 or vice versa
                # Helper: is_connected(name1, name2)
                if self._are_connected(model, name1, name2):
                    continue

                if name1 not in aabbs or name2 not in aabbs:
                    continue

                min1, max1 = aabbs[name1]
                min2, max2 = aabbs[name2]

                # Check AABB overlap
                if (
                    min1[0] <= max2[0]
                    and max1[0] >= min2[0]
                    and min1[1] <= max2[1]
                    and max1[1] >= min2[1]
                    and min1[2] <= max2[2]
                    and max1[2] >= min2[2]
                ):
                    messages.append(f"Potential collision between {name1} and {name2}")

        return messages

    def _are_connected(self, model: HumanoidModel, name1: str, name2: str) -> bool:
        """Check if two links are directly connected by a joint."""
        # Check child map
        if not (model is not None):
            raise ValueError("model must be provided")
        for joint in model.children_map.get(name1, []):
            if joint.child == name2:
                return True
        return any(joint.child == name1 for joint in model.children_map.get(name2, []))
