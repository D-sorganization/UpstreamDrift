"""Inertia Ellipse Visualization Module.

Provides visualization of inertia ellipsoids for body segments and collections
of segments across all physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite).

The inertia ellipsoid represents the rotational inertia properties of a rigid body
or collection of bodies. The ellipsoid's principal axes correspond to the principal
moments of inertia, and its shape indicates how mass is distributed relative to rotation.

Features:
- Individual segment inertia ellipsoids
- Composite inertia for body segment groups (full body, upper body, etc.)
- Support for club inclusion in composite calculations
- Cross-engine compatibility

References:
    Featherstone, R. (2008). "Rigid Body Dynamics Algorithms"
    Springer, Section 2.13: Spatial Inertia.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    pass

LOGGER = logging.getLogger(__name__)


class SegmentGroup(Enum):
    """Predefined body segment groups for inertia visualization.

    These groups define collections of body segments for composite
    inertia ellipse computation.
    """

    FULL_BODY = "full_body"
    FULL_BODY_WITH_CLUB = "full_body_with_club"
    UPPER_BODY = "upper_body"
    UPPER_BODY_WITH_CLUB = "upper_body_with_club"
    LOWER_BODY = "lower_body"
    LEFT_ARM = "left_arm"
    RIGHT_ARM = "right_arm"
    TORSO = "torso"
    CLUB_ONLY = "club_only"
    CUSTOM = "custom"


# Default segment definitions for humanoid models
# These map segment group to body names (engine-specific names may vary)
DEFAULT_SEGMENT_MAPPINGS: dict[SegmentGroup, list[str]] = {
    SegmentGroup.FULL_BODY: [
        "pelvis",
        "torso",
        "head",
        "l_upper_arm",
        "l_lower_arm",
        "l_hand",
        "r_upper_arm",
        "r_lower_arm",
        "r_hand",
        "l_thigh",
        "l_shin",
        "l_foot",
        "r_thigh",
        "r_shin",
        "r_foot",
    ],
    SegmentGroup.FULL_BODY_WITH_CLUB: [
        "pelvis",
        "torso",
        "head",
        "l_upper_arm",
        "l_lower_arm",
        "l_hand",
        "r_upper_arm",
        "r_lower_arm",
        "r_hand",
        "l_thigh",
        "l_shin",
        "l_foot",
        "r_thigh",
        "r_shin",
        "r_foot",
        "club_shaft",
        "club_head",
    ],
    SegmentGroup.UPPER_BODY: [
        "torso",
        "head",
        "l_upper_arm",
        "l_lower_arm",
        "l_hand",
        "r_upper_arm",
        "r_lower_arm",
        "r_hand",
    ],
    SegmentGroup.UPPER_BODY_WITH_CLUB: [
        "torso",
        "head",
        "l_upper_arm",
        "l_lower_arm",
        "l_hand",
        "r_upper_arm",
        "r_lower_arm",
        "r_hand",
        "club_shaft",
        "club_head",
    ],
    SegmentGroup.LOWER_BODY: [
        "pelvis",
        "l_thigh",
        "l_shin",
        "l_foot",
        "r_thigh",
        "r_shin",
        "r_foot",
    ],
    SegmentGroup.LEFT_ARM: [
        "l_upper_arm",
        "l_lower_arm",
        "l_hand",
    ],
    SegmentGroup.RIGHT_ARM: [
        "r_upper_arm",
        "r_lower_arm",
        "r_hand",
    ],
    SegmentGroup.TORSO: [
        "torso",
    ],
    SegmentGroup.CLUB_ONLY: [
        "club_shaft",
        "club_head",
    ],
    SegmentGroup.CUSTOM: [],
}

# Alternative naming conventions for different engines
BODY_NAME_ALIASES: dict[str, list[str]] = {
    # Standard name -> possible alternatives
    "pelvis": ["pelvis", "Pelvis", "root", "hips", "hip"],
    "torso": ["torso", "Torso", "trunk", "Trunk", "chest", "abdomen"],
    "head": ["head", "Head", "skull"],
    "l_upper_arm": ["l_upper_arm", "left_upper_arm", "LeftUpperArm", "l_humerus"],
    "l_lower_arm": ["l_lower_arm", "left_lower_arm", "LeftLowerArm", "l_radius"],
    "l_hand": ["l_hand", "left_hand", "LeftHand", "l_wrist"],
    "r_upper_arm": ["r_upper_arm", "right_upper_arm", "RightUpperArm", "r_humerus"],
    "r_lower_arm": ["r_lower_arm", "right_lower_arm", "RightLowerArm", "r_radius"],
    "r_hand": ["r_hand", "right_hand", "RightHand", "r_wrist"],
    "l_thigh": ["l_thigh", "left_thigh", "LeftThigh", "l_femur"],
    "l_shin": ["l_shin", "left_shin", "LeftShin", "l_tibia", "l_lower_leg"],
    "l_foot": ["l_foot", "left_foot", "LeftFoot", "l_ankle"],
    "r_thigh": ["r_thigh", "right_thigh", "RightThigh", "r_femur"],
    "r_shin": ["r_shin", "right_shin", "RightShin", "r_tibia", "r_lower_leg"],
    "r_foot": ["r_foot", "right_foot", "RightFoot", "r_ankle"],
    "club_shaft": ["club_shaft", "shaft", "club_body", "golf_shaft"],
    "club_head": ["club_head", "clubhead", "club_face", "golf_head"],
}


@dataclass
class InertiaEllipseConfig:
    """Configuration for inertia ellipse visualization.

    Attributes:
        enabled: Whether inertia ellipse visualization is enabled
        segment_group: Which segment group to visualize
        custom_segments: Custom list of body names (when segment_group is CUSTOM)
        show_individual: Show individual segment ellipsoids
        show_composite: Show composite ellipsoid for entire group
        opacity: Transparency of ellipsoid visualization (0-1)
        color_individual: RGBA color for individual ellipsoids [0-1]
        color_composite: RGBA color for composite ellipsoid [0-1]
        scale_factor: Scaling factor for ellipsoid visualization (for visibility)
        reference_frame: Reference frame for composite ('world' or 'com')
    """

    enabled: bool = False
    segment_group: SegmentGroup = SegmentGroup.FULL_BODY
    custom_segments: list[str] = field(default_factory=list)
    show_individual: bool = False
    show_composite: bool = True
    opacity: float = 0.5
    color_individual: list[float] = field(
        default_factory=lambda: [0.2, 0.6, 1.0, 0.5]  # Light blue
    )
    color_composite: list[float] = field(
        default_factory=lambda: [1.0, 0.4, 0.2, 0.6]  # Orange
    )
    scale_factor: float = 1.0
    reference_frame: str = "com"  # 'world' or 'com'


@dataclass
class BodyInertiaData:
    """Inertia data for a single body segment.

    Attributes:
        name: Body name
        mass: Body mass [kg]
        com_world: Center of mass in world frame [m]
        inertia_local: 3x3 inertia tensor in local frame [kg*m^2]
        rotation: 3x3 rotation matrix from local to world frame
    """

    name: str
    mass: float
    com_world: np.ndarray  # (3,)
    inertia_local: np.ndarray  # (3, 3)
    rotation: np.ndarray  # (3, 3)

    @property
    def inertia_world(self) -> np.ndarray:
        """Compute inertia tensor in world frame.

        Returns:
            (3, 3) inertia tensor rotated to world frame [kg*m^2]
        """
        # I_world = R @ I_local @ R^T
        return self.rotation @ self.inertia_local @ self.rotation.T


@dataclass
class InertiaEllipseData:
    """Computed inertia ellipsoid data.

    The inertia ellipsoid visualizes the principal moments of inertia.
    Radii are proportional to 1/sqrt(I_principal) so that smaller
    moments (easier rotation) correspond to larger ellipsoid extent.

    Attributes:
        center: Center position in world frame [m]
        radii: Principal radii (visualization scale) [m]
        axes: Principal axes as 3x3 rotation matrix (columns are unit vectors)
        principal_moments: Principal moments of inertia [kg*m^2]
        total_mass: Total mass of the body/bodies [kg]
        body_names: Names of bodies included in this ellipsoid
        segment_group: Segment group this ellipsoid represents
    """

    center: np.ndarray  # (3,) COM position
    radii: np.ndarray  # (3,) visualization radii
    axes: np.ndarray  # (3, 3) rotation matrix
    principal_moments: np.ndarray  # (3,) I1, I2, I3
    total_mass: float
    body_names: list[str]
    segment_group: SegmentGroup | None = None


def compute_body_inertia_ellipse(
    body_data: BodyInertiaData,
    scale_factor: float = 1.0,
) -> InertiaEllipseData:
    """Compute inertia ellipsoid for a single body.

    Args:
        body_data: Inertia data for the body
        scale_factor: Scale factor for visualization radii

    Returns:
        InertiaEllipseData with ellipsoid parameters
    """
    # Get inertia in world frame
    I_world = body_data.inertia_world

    # Eigendecomposition for principal moments and axes
    eigenvalues, eigenvectors = np.linalg.eigh(I_world)

    # Sort by eigenvalue (smallest first for consistency)
    idx = np.argsort(eigenvalues)
    principal_moments = eigenvalues[idx]
    axes = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]

    # Compute visualization radii: r_i = scale / sqrt(I_i)
    # Add small epsilon to avoid division by zero for degenerate cases
    eps = 1e-10
    radii = scale_factor / np.sqrt(principal_moments + eps)

    return InertiaEllipseData(
        center=body_data.com_world.copy(),
        radii=radii,
        axes=axes,
        principal_moments=principal_moments,
        total_mass=body_data.mass,
        body_names=[body_data.name],
        segment_group=None,
    )


def compute_composite_inertia(
    bodies: list[BodyInertiaData],
    reference_point: np.ndarray | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Compute composite inertia tensor for multiple bodies.

    Uses the parallel axis theorem to combine inertias about a common point.

    Args:
        bodies: List of body inertia data
        reference_point: Point about which to compute inertia.
                        If None, uses combined center of mass.

    Returns:
        Tuple of (total_mass, com_position, inertia_tensor):
            - total_mass: Sum of body masses [kg]
            - com_position: Combined center of mass [m]
            - inertia_tensor: 3x3 combined inertia tensor [kg*m^2]
    """
    if not bodies:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    # Compute total mass and combined COM
    total_mass = sum(b.mass for b in bodies)
    if total_mass < 1e-10:
        return 0.0, np.zeros(3), np.zeros((3, 3))

    com = sum(b.mass * b.com_world for b in bodies) / total_mass

    # Use COM as reference point if not specified
    if reference_point is None:
        reference_point = com

    # Combine inertias using parallel axis theorem
    # I_total = sum_i [I_i + m_i * (d_i^T d_i I - d_i d_i^T)]
    # where d_i is the offset from reference point to body i's COM

    I_total = np.zeros((3, 3))

    for body in bodies:
        # Body's inertia in world frame about its own COM
        I_body_world = body.inertia_world

        # Offset from reference point to body's COM
        d = body.com_world - reference_point

        # Parallel axis theorem: I = I_com + m * (|d|^2 * I - d * d^T)
        d_sq = np.dot(d, d)
        I_parallel = body.mass * (d_sq * np.eye(3) - np.outer(d, d))

        I_total += I_body_world + I_parallel

    return total_mass, com, I_total


def compute_composite_inertia_ellipse(
    bodies: list[BodyInertiaData],
    segment_group: SegmentGroup | None = None,
    scale_factor: float = 1.0,
    reference_frame: str = "com",
) -> InertiaEllipseData | None:
    """Compute inertia ellipsoid for a collection of bodies.

    Args:
        bodies: List of body inertia data
        segment_group: Segment group being computed (for metadata)
        scale_factor: Scale factor for visualization radii
        reference_frame: 'com' to compute about COM, 'world' about origin

    Returns:
        InertiaEllipseData with composite ellipsoid parameters, or None if no bodies
    """
    if not bodies:
        return None

    # Determine reference point
    reference_point = None if reference_frame == "com" else np.zeros(3)

    # Compute composite inertia
    total_mass, com, I_total = compute_composite_inertia(bodies, reference_point)

    if total_mass < 1e-10:
        return None

    # Eigendecomposition for principal moments and axes
    eigenvalues, eigenvectors = np.linalg.eigh(I_total)

    # Sort by eigenvalue (smallest first)
    idx = np.argsort(eigenvalues)
    principal_moments = eigenvalues[idx]
    axes = eigenvectors[:, idx]

    # Ensure right-handed coordinate system
    if np.linalg.det(axes) < 0:
        axes[:, 2] = -axes[:, 2]

    # Compute visualization radii
    eps = 1e-10
    radii = scale_factor / np.sqrt(np.maximum(principal_moments, eps))

    # Determine ellipsoid center
    center = com if reference_frame == "com" else np.zeros(3)

    return InertiaEllipseData(
        center=center,
        radii=radii,
        axes=axes,
        principal_moments=principal_moments,
        total_mass=total_mass,
        body_names=[b.name for b in bodies],
        segment_group=segment_group,
    )


def resolve_body_name(
    requested_name: str,
    available_names: list[str],
) -> str | None:
    """Resolve a body name using aliases.

    Args:
        requested_name: The standard body name requested
        available_names: List of body names available in the model

    Returns:
        Matching body name from available_names, or None if not found
    """
    # Direct match
    if requested_name in available_names:
        return requested_name

    # Check aliases
    aliases = BODY_NAME_ALIASES.get(requested_name, [requested_name])
    for alias in aliases:
        if alias in available_names:
            return alias

    # Case-insensitive search as fallback
    lower_requested = requested_name.lower()
    for name in available_names:
        if name.lower() == lower_requested:
            return name

    return None


def get_segment_bodies(
    segment_group: SegmentGroup,
    available_bodies: list[str],
    custom_segments: list[str] | None = None,
) -> list[str]:
    """Get list of body names for a segment group.

    Args:
        segment_group: The segment group to get bodies for
        available_bodies: Bodies available in the model
        custom_segments: Custom body list (for CUSTOM segment group)

    Returns:
        List of resolved body names that exist in the model
    """
    if segment_group == SegmentGroup.CUSTOM:
        requested = custom_segments or []
    else:
        requested = DEFAULT_SEGMENT_MAPPINGS.get(segment_group, [])

    # Resolve each requested body name
    resolved = []
    for name in requested:
        resolved_name = resolve_body_name(name, available_bodies)
        if resolved_name is not None:
            resolved.append(resolved_name)
        else:
            LOGGER.debug(f"Body '{name}' not found in model, skipping")

    return resolved


@runtime_checkable
class InertiaEllipseProvider(Protocol):
    """Protocol for physics engines that support inertia ellipse computation.

    Engines implementing this protocol can provide body inertia data
    for inertia ellipse visualization.
    """

    def get_body_names(self) -> list[str]:
        """Get list of all body names in the model.

        Returns:
            List of body name strings
        """
        ...

    def get_body_inertia_data(self, body_name: str) -> BodyInertiaData | None:
        """Get inertia data for a specific body.

        Args:
            body_name: Name of the body

        Returns:
            BodyInertiaData for the body, or None if not found
        """
        ...

    def get_all_body_inertia_data(self) -> list[BodyInertiaData]:
        """Get inertia data for all bodies in the model.

        Returns:
            List of BodyInertiaData for all bodies
        """
        ...


class InertiaEllipseVisualizer:
    """High-level interface for inertia ellipse visualization.

    Works with any physics engine that implements InertiaEllipseProvider.
    """

    def __init__(
        self,
        engine: Any,
        config: InertiaEllipseConfig | None = None,
    ) -> None:
        """Initialize the visualizer.

        Args:
            engine: Physics engine (should implement InertiaEllipseProvider methods)
            config: Visualization configuration
        """
        self.engine = engine
        self.config = config or InertiaEllipseConfig()
        self._ellipse_cache: dict[str, InertiaEllipseData] = {}

    def set_config(self, config: InertiaEllipseConfig) -> None:
        """Update visualization configuration.

        Args:
            config: New configuration
        """
        self.config = config
        self._ellipse_cache.clear()

    def set_segment_group(self, segment_group: SegmentGroup) -> None:
        """Change the segment group being visualized.

        Args:
            segment_group: New segment group
        """
        self.config.segment_group = segment_group
        self._ellipse_cache.clear()

    def compute_ellipses(self) -> dict[str, InertiaEllipseData]:
        """Compute inertia ellipsoids based on current configuration.

        Returns:
            Dictionary mapping names to InertiaEllipseData:
            - 'composite': Composite ellipsoid for the segment group
            - Individual body names: Individual body ellipsoids (if show_individual)
        """
        if not self.config.enabled:
            return {}

        results: dict[str, InertiaEllipseData] = {}

        # Get available bodies from engine
        try:
            available_bodies = self.engine.get_body_names()
        except AttributeError:
            LOGGER.warning("Engine does not support get_body_names()")
            return {}

        # Get bodies for the configured segment group
        segment_bodies = get_segment_bodies(
            self.config.segment_group,
            available_bodies,
            self.config.custom_segments,
        )

        if not segment_bodies:
            LOGGER.warning(
                f"No bodies found for segment group {self.config.segment_group}"
            )
            return {}

        # Collect inertia data for all bodies in the group
        body_data_list: list[BodyInertiaData] = []
        for body_name in segment_bodies:
            try:
                body_data = self.engine.get_body_inertia_data(body_name)
                if body_data is not None:
                    body_data_list.append(body_data)

                    # Compute individual ellipsoid if requested
                    if self.config.show_individual:
                        ellipse = compute_body_inertia_ellipse(
                            body_data, self.config.scale_factor
                        )
                        results[body_name] = ellipse
            except AttributeError:
                LOGGER.warning(f"Engine does not support get_body_inertia_data()")
                return {}
            except Exception as e:
                LOGGER.warning(f"Failed to get inertia for '{body_name}': {e}")

        # Compute composite ellipsoid if requested
        if self.config.show_composite and body_data_list:
            composite = compute_composite_inertia_ellipse(
                body_data_list,
                self.config.segment_group,
                self.config.scale_factor,
                self.config.reference_frame,
            )
            if composite is not None:
                results["composite"] = composite

        self._ellipse_cache = results
        return results

    def get_cached_ellipses(self) -> dict[str, InertiaEllipseData]:
        """Get previously computed ellipsoids.

        Returns:
            Cached ellipsoid dictionary
        """
        return self._ellipse_cache

    def get_inertia_summary(self) -> dict[str, Any]:
        """Get summary of current inertia visualization.

        Returns:
            Dictionary with summary statistics
        """
        if "composite" not in self._ellipse_cache:
            self.compute_ellipses()

        composite = self._ellipse_cache.get("composite")
        if composite is None:
            return {}

        return {
            "segment_group": self.config.segment_group.value,
            "total_mass": composite.total_mass,
            "center_of_mass": composite.center.tolist(),
            "principal_moments": composite.principal_moments.tolist(),
            "principal_axes": composite.axes.tolist(),
            "radii": composite.radii.tolist(),
            "body_count": len(composite.body_names),
            "bodies": composite.body_names,
        }


def generate_inertia_ellipse_mesh(
    ellipse: InertiaEllipseData,
    n_meridians: int = 20,
    n_parallels: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate triangle mesh for an inertia ellipsoid.

    Args:
        ellipse: InertiaEllipseData defining the ellipsoid
        n_meridians: Number of longitudinal slices
        n_parallels: Number of latitudinal slices

    Returns:
        Tuple of (vertices, faces):
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of triangle face indices
    """
    # Generate unit sphere
    phi = np.linspace(0, np.pi, n_parallels + 1)
    theta = np.linspace(0, 2 * np.pi, n_meridians + 1)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Unit sphere coordinates
    x = np.sin(phi_grid) * np.cos(theta_grid)
    y = np.sin(phi_grid) * np.sin(theta_grid)
    z = np.cos(phi_grid)

    # Stack into (3, n_vertices)
    sphere_points = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=0)

    # Scale by radii
    scaled = sphere_points * ellipse.radii[:, np.newaxis]

    # Rotate by principal axes
    rotated = ellipse.axes @ scaled

    # Translate to center
    vertices = rotated.T + ellipse.center

    # Generate faces (same as ellipsoid_visualization.py)
    n_verts_per_row = n_parallels + 1
    n_rows = n_meridians + 1

    i_indices = np.arange(n_rows - 1)
    j_indices = np.arange(n_verts_per_row - 1)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="ij")

    v0 = (i_grid * n_verts_per_row + j_grid).ravel()
    v1 = v0 + 1
    v2 = v0 + n_verts_per_row
    v3 = v2 + 1

    n_quads = len(v0)
    faces = np.empty((n_quads * 2, 3), dtype=int)
    faces[0::2] = np.column_stack([v0, v2, v1])
    faces[1::2] = np.column_stack([v1, v2, v3])

    return vertices, faces
