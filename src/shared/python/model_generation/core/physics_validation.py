"""Physics Validation for URDF Models.

This module provides comprehensive physics validation to catch issues
before simulation:
- Inertia tensor validation (SPD, triangle inequality)
- Static stability analysis
- Collision self-intersection detection
- Dynamic feasibility checks

These validations extend the base Validator class with physics-specific
checks that ensure models will behave correctly in physics simulators.

Example:
    validator = PhysicsValidator()
    result = validator.validate_physics(model)
    if not result.is_stable:
        logger.info(f"Stability margin: {result.stability_margin}")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from model_generation.core.validation import ValidationResult, Validator

if TYPE_CHECKING:
    from model_generation.core.types import Inertia, Joint, Link

logger = logging.getLogger(__name__)


@dataclass
class InertiaValidationResult:
    """Detailed result of inertia tensor validation."""

    is_valid: bool
    is_symmetric: bool
    is_positive_definite: bool
    satisfies_triangle_inequality: bool
    eigenvalues: tuple[float, float, float] | None = None
    condition_number: float | None = None
    principal_axes: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class StabilityResult:
    """Result of static stability analysis."""

    is_stable: bool
    center_of_mass: tuple[float, float, float]
    support_polygon: list[tuple[float, float]] | None = None
    margin_to_edge: float = 0.0
    tipping_angle_deg: float = 90.0
    most_unstable_direction: tuple[float, float] | None = None
    gravity_torque: float = 0.0


@dataclass
class CollisionCheckResult:
    """Result of collision geometry validation."""

    has_self_intersection: bool
    penetration_pairs: list[tuple[str, str]] = field(default_factory=list)
    min_separation: float = float("inf")
    contact_points: list[tuple[float, float, float]] = field(default_factory=list)


@dataclass
class PhysicsValidationResult:
    """Complete physics validation result."""

    is_valid: bool
    inertia_results: dict[str, InertiaValidationResult] = field(default_factory=dict)
    stability: StabilityResult | None = None
    collision: CollisionCheckResult | None = None
    validation_result: ValidationResult | None = None
    total_mass: float = 0.0
    center_of_mass: tuple[float, float, float] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class PhysicsValidator:
    """Comprehensive physics validation for URDF models.

    Extends the base Validator with physics-specific checks:
    - Detailed inertia tensor analysis
    - Static stability computation
    - Collision geometry validation
    - Dynamic feasibility analysis
    """

    def __init__(self, gravity: np.ndarray | None = None) -> None:
        """Initialize physics validator.

        Args:
            gravity: Gravity vector [m/s²] (default: [0, 0, -9.81])
        """
        self.gravity = gravity if gravity is not None else np.array([0.0, 0.0, -9.81])
        self._validator = Validator()

    def validate_inertia_tensor(
        self,
        inertia: Inertia,
        component: str | None = None,
    ) -> InertiaValidationResult:
        """Validate an inertia tensor comprehensively.

        Checks:
        - Symmetry of the 3x3 matrix
        - Positive definiteness via Cholesky
        - Triangle inequality: |Ia - Ib| <= Ic <= Ia + Ib
        - Condition number for numerical stability
        - Eigenvalue analysis

        Args:
            inertia: Inertia object to validate
            component: Optional component name for error messages

        Returns:
            Detailed InertiaValidationResult
        """
        if not (inertia is not None):
            raise ValueError("inertia must be provided")
        if not (inertia is not None):
            raise ValueError("inertia must be provided")
        result = InertiaValidationResult(
            is_valid=True,
            is_symmetric=True,
            is_positive_definite=True,
            satisfies_triangle_inequality=True,
        )

        # Build the 3x3 inertia matrix
        tensor = np.array(
            [
                [inertia.ixx, inertia.ixy, inertia.ixz],
                [inertia.ixy, inertia.iyy, inertia.iyz],
                [inertia.ixz, inertia.iyz, inertia.izz],
            ]
        )

        # Check symmetry
        if not np.allclose(tensor, tensor.T, rtol=1e-10):
            result.is_symmetric = False
            result.is_valid = False
            result.errors.append(f"Inertia tensor is not symmetric for {component or 'unknown'}")

        # Compute eigenvalues
        try:
            eigenvalues = np.linalg.eigvalsh(tensor)
            result.eigenvalues = tuple(eigenvalues.tolist())

            # Check positive definiteness
            if not np.all(eigenvalues > 0):
                result.is_positive_definite = False
                result.is_valid = False
                result.errors.append(
                    f"Inertia tensor is not positive definite. " f"Eigenvalues: {eigenvalues}"
                )

            # Compute condition number
            if eigenvalues.min() > 0:
                result.condition_number = float(eigenvalues.max() / eigenvalues.min())
                if result.condition_number > 1e5:
                    result.warnings.append(
                        f"High condition number ({result.condition_number:.2e}) "
                        "may cause numerical instability"
                    )

            # Compute principal axes
            _, eigenvectors = np.linalg.eigh(tensor)
            result.principal_axes = eigenvectors

        except np.linalg.LinAlgError as e:
            result.is_valid = False
            result.errors.append(f"Failed to compute eigenvalues: {e}")
            return result

        # Check triangle inequality
        Ixx, Iyy, Izz = inertia.ixx, inertia.iyy, inertia.izz
        inequalities = [
            (Ixx + Iyy >= Izz, f"Ixx + Iyy >= Izz: {Ixx} + {Iyy} >= {Izz}"),
            (Iyy + Izz >= Ixx, f"Iyy + Izz >= Ixx: {Iyy} + {Izz} >= {Ixx}"),
            (Izz + Ixx >= Iyy, f"Izz + Ixx >= Iyy: {Izz} + {Ixx} >= {Iyy}"),
        ]

        for valid, desc in inequalities:
            if not valid:
                result.satisfies_triangle_inequality = False
                result.warnings.append(f"Triangle inequality violated: {desc}")

        # Check for very small inertia values
        min_inertia = min(Ixx, Iyy, Izz)
        if min_inertia < 1e-9:
            result.warnings.append(
                f"Very small inertia value ({min_inertia:.2e}) may cause " "numerical instability"
            )

        return result

    @staticmethod
    def _compute_center_of_mass(
        links: list[Link],
    ) -> tuple[np.ndarray, float] | None:
        """Compute overall center of mass from links with inertial data.

        Returns:
            Tuple of (com_array, total_mass) or None if no mass found.
        """
        total_mass = 0.0
        weighted_position = np.zeros(3)

        for link in links:
            if hasattr(link, "inertial") and link.inertial:
                mass = link.inertial.mass
                origin = link.inertial.origin
                pos = np.array([origin.x, origin.y, origin.z])
                weighted_position += mass * pos
                total_mass += mass

        if total_mass <= 0:
            return None
        return weighted_position / total_mass, total_mass

    @staticmethod
    def _find_support_polygon(
        links: list[Link],
        support_link_names: list[str] | None,
    ) -> tuple[list[tuple[float, float]], list[str] | None]:
        """Identify support links and extract their 2D positions.

        Returns:
            Tuple of (support_points, resolved_support_link_names).
        """
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        if support_link_names is None:
            link_z_positions = []
            for link in links:
                if hasattr(link, "inertial") and link.inertial:
                    z = link.inertial.origin.z
                    link_z_positions.append((link.name, z))

            if link_z_positions:
                min_z = min(z for _, z in link_z_positions)
                support_link_names = [name for name, z in link_z_positions if abs(z - min_z) < 0.1]

        support_points: list[tuple[float, float]] = []
        for link in links:
            if (
                link.name in (support_link_names or [])
                and hasattr(link, "inertial")
                and link.inertial
            ):
                origin = link.inertial.origin
                support_points.append((origin.x, origin.y))

        return support_points, support_link_names

    def _evaluate_stability(
        self,
        com: np.ndarray,
        total_mass: float,
        support_points: list[tuple[float, float]],
    ) -> tuple[bool, float, list[tuple[float, float]] | None, float]:
        """Evaluate stability metrics from COM and support polygon.

        Returns:
            Tuple of (is_stable, margin, support_polygon, tipping_angle_deg).
        """
        if not (com is not None):
            raise ValueError("com must be provided")
        if not (com is not None):
            raise ValueError("com must be provided")
        support_polygon: list[tuple[float, float]] | None = None

        if len(support_points) < 3:
            is_stable = len(support_points) > 0
            margin = 0.0 if is_stable else float("inf")
            support_polygon = list(support_points) if support_points else None
        else:
            try:
                from scipy.spatial import ConvexHull

                points = np.array(support_points)
                hull = ConvexHull(points)
                support_polygon = [tuple(points[i].tolist()) for i in hull.vertices]

                com_2d = com[:2]
                is_stable = self._point_in_polygon(com_2d, support_polygon)
                margin = self._distance_to_polygon_edge(com_2d, support_polygon)

            except ImportError:
                is_stable = True
                support_polygon = list(support_points)
                margin = 0.0

        if margin > 0 and total_mass > 0:
            height = com[2] if com[2] > 0 else 1.0
            tipping_angle = float(np.degrees(np.arctan(margin / height)))
        else:
            tipping_angle = 0.0

        return is_stable, margin, support_polygon, tipping_angle

    def check_static_stability(
        self,
        links: list[Link],
        support_link_names: list[str] | None = None,
    ) -> StabilityResult:
        """Analyze static stability of the model.

        Computes:
        - Center of mass position
        - Support polygon from ground contact links
        - Stability margin (distance from COM to polygon edge)
        - Tipping angle

        Args:
            links: List of Link objects
            support_link_names: Names of links in ground contact (default: lowest links)

        Returns:
            StabilityResult with stability metrics
        """
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        if not links:
            return StabilityResult(
                is_stable=False,
                center_of_mass=(0.0, 0.0, 0.0),
            )

        com_result = self._compute_center_of_mass(links)
        if com_result is None:
            return StabilityResult(
                is_stable=False,
                center_of_mass=(0.0, 0.0, 0.0),
            )

        com, total_mass = com_result
        support_points, support_link_names = self._find_support_polygon(links, support_link_names)
        is_stable, margin, support_polygon, tipping_angle = self._evaluate_stability(
            com, total_mass, support_points
        )

        return StabilityResult(
            is_stable=is_stable,
            center_of_mass=tuple(com.tolist()),
            support_polygon=support_polygon if support_points else None,
            margin_to_edge=margin,
            tipping_angle_deg=tipping_angle,
        )

    def check_collision_geometry(
        self,
        links: list[Link],
    ) -> CollisionCheckResult:
        """Check for self-intersection in collision geometries.

        This is a simplified check that detects obvious overlaps.
        Full mesh intersection would require a collision library.

        Args:
            links: List of Link objects with collision geometry

        Returns:
            CollisionCheckResult with intersection data
        """
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        result = CollisionCheckResult(has_self_intersection=False)
        collision_spheres = []

        # Extract bounding spheres for each collision geometry
        for link in links:
            if hasattr(link, "collision") and link.collision:
                # Simplified: use bounding sphere approximation
                geom = link.collision.geometry
                origin = link.collision.origin

                # Estimate bounding sphere based on geometry type
                if hasattr(geom, "radius"):
                    radius = geom.radius
                elif hasattr(geom, "size"):
                    radius = max(geom.size) / 2
                else:
                    radius = 0.1  # Default

                center = np.array([origin.x, origin.y, origin.z])
                collision_spheres.append((link.name, center, radius))

        # Check pairwise distances
        for i, (name1, center1, radius1) in enumerate(collision_spheres):
            for _j, (name2, center2, radius2) in enumerate(collision_spheres[i + 1 :]):
                distance = np.linalg.norm(center1 - center2)
                min_separation = distance - radius1 - radius2

                result.min_separation = min(result.min_separation, min_separation)

                if min_separation < 0:
                    result.has_self_intersection = True
                    result.penetration_pairs.append((name1, name2))
                    # Approximate contact point
                    contact = (center1 + center2) / 2
                    result.contact_points.append(tuple(contact.tolist()))

        return result

    def validate_physics(
        self,
        links: list[Link],
        joints: list[Joint] | None = None,
        check_stability: bool = True,
        check_collisions: bool = True,
    ) -> PhysicsValidationResult:
        """Comprehensive physics validation.

        Args:
            links: List of Link objects
            joints: Optional list of Joint objects
            check_stability: Whether to perform stability analysis
            check_collisions: Whether to check for self-intersections

        Returns:
            Complete PhysicsValidationResult
        """
        if not (links is not None):
            raise ValueError("links must be provided")
        if not (links is not None):
            raise ValueError("links must be provided")
        result = PhysicsValidationResult(is_valid=True)

        # Validate each link's inertia
        total_mass = 0.0
        weighted_position = np.zeros(3)

        for link in links:
            if hasattr(link, "inertial") and link.inertial:
                inertial = link.inertial

                # Check if inertial has full tensor attributes
                has_inertia_tensor = hasattr(inertial, "ixx") and hasattr(inertial, "iyy")

                if has_inertia_tensor:
                    inertia_result = self.validate_inertia_tensor(inertial, component=link.name)
                    result.inertia_results[link.name] = inertia_result

                    if not inertia_result.is_valid:
                        result.is_valid = False
                        result.errors.extend(inertia_result.errors)
                    result.warnings.extend(inertia_result.warnings)

                # Accumulate for COM
                mass = inertial.mass
                origin = inertial.origin
                pos = np.array([origin.x, origin.y, origin.z])
                weighted_position += mass * pos
                total_mass += mass

        result.total_mass = total_mass
        if total_mass > 0:
            com = weighted_position / total_mass
            result.center_of_mass = tuple(com.tolist())

        # Stability analysis
        if check_stability:
            result.stability = self.check_static_stability(links)
            if not result.stability.is_stable:
                result.warnings.append("Model may be statically unstable")

        # Collision check
        if check_collisions:
            result.collision = self.check_collision_geometry(links)
            if result.collision.has_self_intersection:
                result.warnings.append(
                    f"Self-intersection detected in collision geometry: "
                    f"{result.collision.penetration_pairs}"
                )

        # Base validation
        if joints:
            # link_names = {link.name for link in links}
            base_result = Validator.validate_model(links, joints)
            result.validation_result = base_result
            if not base_result.is_valid:
                result.is_valid = False
                result.errors.extend(base_result.get_error_messages())
            result.warnings.extend(base_result.get_warning_messages())

        return result

    @staticmethod
    def _point_in_polygon(
        point: np.ndarray,
        polygon: list[tuple[float, float]],
    ) -> bool:
        """Check if a 2D point is inside a polygon using ray casting."""
        if not (point is not None):
            raise ValueError("point must be provided")
        if not (point is not None):
            raise ValueError("point must be provided")
        x, y = point
        n = len(polygon)
        inside = False

        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]

            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i

        return inside

    @staticmethod
    def _distance_to_polygon_edge(
        point: np.ndarray,
        polygon: list[tuple[float, float]],
    ) -> float:
        """Compute minimum distance from point to polygon edge."""
        if not (point is not None):
            raise ValueError("point must be provided")
        if not (point is not None):
            raise ValueError("point must be provided")
        min_dist = float("inf")
        n = len(polygon)

        for i in range(n):
            p1 = np.array(polygon[i])
            p2 = np.array(polygon[(i + 1) % n])

            # Distance to line segment
            v = p2 - p1
            w = point - p1
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)

            if c2 == 0 or c1 <= 0:
                dist = np.linalg.norm(point - p1)
            elif c2 <= c1:
                dist = np.linalg.norm(point - p2)
            else:
                b = c1 / c2
                pb = p1 + b * v
                dist = np.linalg.norm(point - pb)

            min_dist = min(min_dist, float(dist))

        return min_dist


__all__ = [
    "PhysicsValidator",
    "PhysicsValidationResult",
    "InertiaValidationResult",
    "StabilityResult",
    "CollisionCheckResult",
]
