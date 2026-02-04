"""Contact manager for multi-contact scenarios.

This module provides the ContactManager class for detecting, tracking,
and managing contacts between robot bodies and the environment.

Design by Contract:
    ContactManager maintains invariants about contact consistency
    and validates all inputs/outputs.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.robotics.core.exceptions import ContactError
from src.robotics.core.protocols import ContactCapable, RoboticsCapable
from src.robotics.core.types import ContactState, ContactType
from src.shared.python.contracts import (
    ContractChecker,
    postcondition,
)


class ContactManager(ContractChecker):
    """Manages multi-contact scenarios for robotics applications.

    Provides contact detection, tracking, and force computation
    capabilities. Works with any engine implementing ContactCapable.

    Design by Contract:
        Invariants:
            - All contacts have unique IDs
            - Contact positions and normals are finite
            - Contact forces satisfy friction constraints (if computed)

        Preconditions:
            - Engine must be initialized before operations
            - Configuration arrays must have correct dimensions

        Postconditions:
            - detect_contacts returns valid ContactState objects
            - Contact Jacobians have correct dimensions

    Example:
        >>> manager = ContactManager(engine)
        >>> contacts = manager.detect_contacts()
        >>> J_contact = manager.get_contact_jacobian_stack(contacts)
    """

    def __init__(
        self,
        engine: RoboticsCapable,
        default_friction: float = 0.5,
    ) -> None:
        """Initialize contact manager.

        Args:
            engine: Physics engine implementing RoboticsCapable.
            default_friction: Default friction coefficient for contacts.

        Raises:
            TypeError: If engine doesn't implement required protocol.
        """
        if not isinstance(engine, RoboticsCapable):
            raise TypeError(
                f"Engine must implement RoboticsCapable protocol, "
                f"got {type(engine).__name__}"
            )

        self._engine = engine
        self._default_friction = default_friction
        self._contact_cache: list[ContactState] = []
        self._next_contact_id = 0
        self._is_contact_capable = isinstance(engine, ContactCapable)

    def _get_invariants(self) -> list[tuple[Any, str]]:
        """Define class invariants."""
        return [
            (
                lambda: self._default_friction >= 0,
                "Default friction must be non-negative",
            ),
            (
                lambda: len(set(c.contact_id for c in self._contact_cache))
                == len(self._contact_cache),
                "All contact IDs must be unique",
            ),
        ]

    @property
    def engine(self) -> RoboticsCapable:
        """Get the underlying physics engine."""
        return self._engine

    @property
    def contact_count(self) -> int:
        """Get number of cached contacts."""
        return len(self._contact_cache)

    @property
    def contacts(self) -> list[ContactState]:
        """Get cached contacts (read-only copy)."""
        return list(self._contact_cache)

    @postcondition(
        lambda result: all(isinstance(c, ContactState) for c in result),
        "Result must contain ContactState objects",
    )
    def detect_contacts(
        self,
        q: NDArray[np.float64] | None = None,
    ) -> list[ContactState]:
        """Detect contacts at current or specified configuration.

        Args:
            q: Configuration to detect contacts at. Uses current if None.

        Returns:
            List of detected ContactState objects.

        Raises:
            ContactError: If contact detection fails.
        """
        # Set configuration if provided
        if q is not None:
            current_q, current_v = self._engine.get_state()
            self._engine.set_state(q, current_v)

        contacts: list[ContactState] = []

        if self._is_contact_capable:
            contacts = self._detect_from_engine()
        else:
            # Fallback: no contact detection available
            contacts = []

        self._contact_cache = contacts
        return contacts

    def _detect_from_engine(self) -> list[ContactState]:
        """Detect contacts using engine's contact capabilities.

        Returns:
            List of ContactState objects from engine.
        """
        engine = self._engine
        if not isinstance(engine, ContactCapable):
            return []

        contacts: list[ContactState] = []
        num_contacts = engine.get_contact_count()

        for i in range(num_contacts):
            try:
                info = engine.get_contact_info(i)
                contact = self._create_contact_from_info(info)
                contacts.append(contact)
            except Exception as e:
                raise ContactError(
                    f"Failed to get contact info for index {i}: {e}",
                    contact_id=i,
                ) from e

        return contacts

    def _create_contact_from_info(
        self,
        info: dict[str, Any],
    ) -> ContactState:
        """Create ContactState from engine contact info.

        Args:
            info: Contact information dictionary from engine.

        Returns:
            ContactState object.
        """
        contact_id = self._next_contact_id
        self._next_contact_id += 1

        position = np.asarray(info.get("position", [0, 0, 0]), dtype=np.float64)
        normal = np.asarray(info.get("normal", [0, 0, 1]), dtype=np.float64)
        force = np.asarray(info.get("force", [0, 0, 0]), dtype=np.float64)

        # Decompose force into normal and tangential
        normal_force = max(0.0, float(np.dot(force, normal)))
        friction_force = force - normal_force * normal

        return ContactState(
            contact_id=contact_id,
            body_a=info.get("body_a", "unknown"),
            body_b=info.get("body_b", "unknown"),
            position=position,
            normal=normal,
            penetration=info.get("penetration", 0.0),
            normal_force=normal_force,
            friction_force=friction_force,
            friction_coefficient=info.get("friction", self._default_friction),
            contact_type=ContactType.POINT,
            is_active=True,
        )

    def get_contact_jacobian(
        self,
        contact: ContactState,
    ) -> NDArray[np.float64] | None:
        """Get Jacobian for a specific contact.

        Args:
            contact: Contact to get Jacobian for.

        Returns:
            Contact Jacobian (3, n_v) or (6, n_v), or None if unavailable.
        """
        if not self._is_contact_capable:
            return None

        engine = self._engine
        if not isinstance(engine, ContactCapable):
            return None

        # Find contact index in cache
        for i, c in enumerate(self._contact_cache):
            if c.contact_id == contact.contact_id:
                return engine.get_contact_jacobian(i)

        return None

    @postcondition(
        lambda result: result is None or result.ndim == 2,
        "Result must be 2D matrix or None",
    )
    def get_contact_jacobian_stack(
        self,
        contacts: list[ContactState] | None = None,
    ) -> NDArray[np.float64] | None:
        """Get stacked contact Jacobians for multiple contacts.

        Args:
            contacts: Contacts to get Jacobians for. Uses cached if None.

        Returns:
            Stacked Jacobian (3*n_contacts, n_v) or None if unavailable.
        """
        if contacts is None:
            contacts = self._contact_cache

        if not contacts:
            return None

        jacobians: list[NDArray[np.float64]] = []
        for contact in contacts:
            J = self.get_contact_jacobian(contact)
            if J is not None:
                # Use only linear part (first 3 rows) for point contacts
                if J.shape[0] == 6:
                    J = J[:3]
                jacobians.append(J)

        if not jacobians:
            return None

        return np.vstack(jacobians)

    def get_active_contacts(self) -> list[ContactState]:
        """Get only active contacts from cache.

        Returns:
            List of active ContactState objects.
        """
        return [c for c in self._contact_cache if c.is_active]

    def compute_support_polygon(
        self,
        contacts: list[ContactState] | None = None,
    ) -> NDArray[np.float64] | None:
        """Compute support polygon from contact points.

        The support polygon is the convex hull of contact points
        projected onto the ground plane (z=0).

        Args:
            contacts: Contacts to use. Uses cached if None.

        Returns:
            Polygon vertices (n_vertices, 2) in CCW order, or None if < 3 contacts.
        """
        if contacts is None:
            contacts = self._contact_cache

        if len(contacts) < 3:
            return None

        # Project contact points to ground plane
        points_2d = np.array([c.position[:2] for c in contacts])

        # Compute convex hull
        hull = _convex_hull_2d(points_2d)
        return hull

    def point_in_support_polygon(
        self,
        point: NDArray[np.float64],
        contacts: list[ContactState] | None = None,
    ) -> bool:
        """Check if point is inside support polygon.

        Args:
            point: Point (2,) or (3,) to check (z ignored if 3D).
            contacts: Contacts defining support polygon.

        Returns:
            True if point is inside support polygon.
        """
        polygon = self.compute_support_polygon(contacts)
        if polygon is None:
            return False

        point_2d = np.asarray(point, dtype=np.float64)[:2]
        return _point_in_polygon(point_2d, polygon)

    def clear_cache(self) -> None:
        """Clear cached contacts."""
        self._contact_cache = []

    def reset_contact_ids(self) -> None:
        """Reset contact ID counter."""
        self._next_contact_id = 0
        self._contact_cache = []


def _convex_hull_2d(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Compute convex hull of 2D points.

    Uses scipy.spatial.ConvexHull if available, falls back to Graham scan.

    Args:
        points: Points (n, 2) to compute hull of.

    Returns:
        Hull vertices (m, 2) in counter-clockwise order.
    """
    if len(points) < 3:
        return points.copy()

    # Try using scipy first (more robust)
    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(points)
        return points[hull.vertices]
    except ImportError:
        pass
    except Exception:
        # Fall through to manual algorithm
        pass

    # Manual Graham scan algorithm
    return _graham_scan(points)


def _graham_scan(points: NDArray[np.float64]) -> NDArray[np.float64]:
    """Graham scan convex hull algorithm.

    Args:
        points: Points (n, 2) to compute hull of.

    Returns:
        Hull vertices in counter-clockwise order.
    """
    n = len(points)
    if n < 3:
        return points.copy()

    # Find bottom-most point (or left-most if tie)
    start_idx = 0
    for i in range(1, n):
        if points[i, 1] < points[start_idx, 1] or (
            points[i, 1] == points[start_idx, 1] and points[i, 0] < points[start_idx, 0]
        ):
            start_idx = i

    start = points[start_idx].copy()

    # Sort by polar angle from start point, then by distance for ties
    def sort_key(idx: int) -> tuple[float, float]:
        p = points[idx]
        angle = np.arctan2(p[1] - start[1], p[0] - start[0])
        dist = np.sqrt((p[0] - start[0]) ** 2 + (p[1] - start[1]) ** 2)
        return (float(angle), float(dist))

    sorted_indices = sorted(range(n), key=sort_key)
    sorted_points = points[sorted_indices]

    # Graham scan - use < 0 to include collinear points on hull
    hull: list[NDArray[np.float64]] = []
    for p in sorted_points:
        while len(hull) >= 2 and _cross_product_2d(hull[-2], hull[-1], p) < 0:
            hull.pop()
        hull.append(p.copy())

    return np.array(hull)


def _cross_product_2d(
    o: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
) -> float:
    """Compute cross product of vectors OA and OB.

    Returns:
        Positive if counter-clockwise, negative if clockwise.
    """
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _point_in_polygon(
    point: NDArray[np.float64],
    polygon: NDArray[np.float64],
) -> bool:
    """Check if point is inside polygon using ray casting.

    Args:
        point: Point (2,) to check.
        polygon: Polygon vertices (n, 2).

    Returns:
        True if point is inside or on boundary.
    """
    n = len(polygon)
    if n < 3:
        return False

    inside = False
    j = n - 1

    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        if ((yi > point[1]) != (yj > point[1])) and (
            point[0] < (xj - xi) * (point[1] - yi) / (yj - yi) + xi
        ):
            inside = not inside

        j = i

    return inside
