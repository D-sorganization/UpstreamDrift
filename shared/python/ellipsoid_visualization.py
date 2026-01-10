"""3D Ellipsoid Visualization Module.

Guideline I (Mobility and Force Ellipsoids) Implementation.

Provides 3D visualization of manipulability and force ellipsoids computed
from Jacobian matrices, using meshcat for WebGL-based rendering.

Features:
- Velocity manipulability ellipsoid (J @ J^T)
- Force transmission ellipsoid (J^T @ J)
- Time-varying ellipsoid evolution through swing phases
- Export to JSON/OBJ formats

References:
    Yoshikawa, T. (1985). "Manipulability of Robotic Mechanisms"
    The International Journal of Robotics Research, 4(2), 3-9.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from shared.python.interfaces import PhysicsEngine

LOGGER = logging.getLogger(__name__)

# Default colors (RGBA as 0-1 floats converted to hex for meshcat)
VELOCITY_ELLIPSOID_COLOR = 0x00FF88  # Green for velocity/mobility
FORCE_ELLIPSOID_COLOR = 0xFF8800  # Orange for force transmission
SINGULAR_ELLIPSOID_COLOR = 0xFF0000  # Red for near-singular

# Numerical tolerances
SINGULAR_VALUE_TOLERANCE = 1e-15  # [unitless] Threshold for near-zero singular values
# Maximum force capability for singular directions (1000x max non-singular radius)
FORCE_ELLIPSOID_SINGULAR_SCALE = 1000.0


@dataclass
class EllipsoidData:
    """Data structure for a 3D ellipsoid.

    Attributes:
        center: Center position [x, y, z] in world frame [m]
        radii: Principal radii [r1, r2, r3] (singular values) [m or rad]
        axes: Principal axes as 3x3 rotation matrix (columns are unit vectors)
        body_name: Name of the body this ellipsoid is attached to
        ellipsoid_type: 'velocity' or 'force'
        condition_number: Condition number κ = r_max / r_min
        timestep: Time at which this ellipsoid was computed [s]
    """

    center: np.ndarray
    radii: np.ndarray
    axes: np.ndarray
    body_name: str
    ellipsoid_type: str = "velocity"
    condition_number: float = 1.0
    timestep: float = 0.0


@dataclass
class EllipsoidSequence:
    """Time-varying sequence of ellipsoids for animation.

    Attributes:
        ellipsoids: List of EllipsoidData at each timestep
        timesteps: Array of timestamps [s]
        body_name: Name of the body being tracked
    """

    ellipsoids: list[EllipsoidData] = field(default_factory=list)
    timesteps: np.ndarray = field(default_factory=lambda: np.array([]))
    body_name: str = ""


def compute_velocity_ellipsoid(
    engine: PhysicsEngine,
    body_name: str,
    center_position: np.ndarray | None = None,
) -> EllipsoidData | None:
    """Compute velocity manipulability ellipsoid for a body.

    The velocity ellipsoid visualizes the set of achievable end-effector
    velocities for unit joint velocity norm: ||q̇|| ≤ 1.

    Shape derived from J @ J^T (for task-space) or J^T @ J (for joint-space).

    Args:
        engine: Physics engine with loaded model
        body_name: Name of body (e.g., "clubhead", "right_hand")
        center_position: Optional center position override [x, y, z]

    Returns:
        EllipsoidData with velocity ellipsoid parameters, or None if failed

    Example:
        >>> ellipsoid = compute_velocity_ellipsoid(engine, "clubhead")
        >>> print(f"Max velocity reach: {ellipsoid.radii.max():.3f} m/s")
        >>> print(f"Min velocity reach: {ellipsoid.radii.min():.3f} m/s")
    """
    jac_dict = engine.compute_jacobian(body_name)
    if jac_dict is None:
        LOGGER.warning(f"Could not compute Jacobian for '{body_name}'")
        return None

    # Use linear Jacobian (3×n) for position-based ellipsoid
    J = jac_dict.get("linear")
    if J is None or J.size == 0:
        LOGGER.warning(f"Empty Jacobian for '{body_name}'")
        return None

    # SVD: J = U @ S @ V^T
    # Velocity ellipsoid radii = singular values
    # Velocity ellipsoid axes = left singular vectors (U)
    try:
        U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError as e:
        LOGGER.error(f"SVD failed for '{body_name}': {e}")
        return None

    # Compute condition number
    if sigma.min() > SINGULAR_VALUE_TOLERANCE:
        kappa = sigma.max() / sigma.min()
    else:
        kappa = float("inf")

    # Default center to origin if not provided
    if center_position is None:
        center_position = np.zeros(3)

    return EllipsoidData(
        center=center_position,
        radii=sigma,
        axes=U,  # Left singular vectors as column axes
        body_name=body_name,
        ellipsoid_type="velocity",
        condition_number=kappa,
        timestep=engine.get_time(),
    )


def compute_force_ellipsoid(
    engine: PhysicsEngine,
    body_name: str,
    center_position: np.ndarray | None = None,
) -> EllipsoidData | None:
    """Compute force transmission ellipsoid for a body.

    The force ellipsoid visualizes the set of achievable end-effector forces
    for unit joint torque norm: ||τ|| ≤ 1.

    Force ellipsoid is the dual of velocity ellipsoid with inverted radii:
        Force radii = 1 / velocity radii (σ⁻¹)

    Args:
        engine: Physics engine with loaded model
        body_name: Name of body (e.g., "clubhead", "right_hand")
        center_position: Optional center position override [x, y, z]

    Returns:
        EllipsoidData with force ellipsoid parameters, or None if failed

    Example:
        >>> ellipsoid = compute_force_ellipsoid(engine, "clubhead")
        >>> print(f"Max force capability: {ellipsoid.radii.max():.3f} N")
    """
    jac_dict = engine.compute_jacobian(body_name)
    if jac_dict is None:
        LOGGER.warning(f"Could not compute Jacobian for '{body_name}'")
        return None

    J = jac_dict.get("linear")
    if J is None or J.size == 0:
        LOGGER.warning(f"Empty Jacobian for '{body_name}'")
        return None

    try:
        U, sigma, Vt = np.linalg.svd(J, full_matrices=False)
    except np.linalg.LinAlgError as e:
        LOGGER.error(f"SVD failed for '{body_name}': {e}")
        return None

    # Force ellipsoid radii are INVERSE of velocity radii
    # This is because f = J^(-T) @ τ, so force capability is inversely related
    # For near-zero singular values, use FORCE_ELLIPSOID_SINGULAR_SCALE * max_radius
    force_radii = np.where(
        sigma > SINGULAR_VALUE_TOLERANCE,
        1.0 / sigma,
        sigma.max() * FORCE_ELLIPSOID_SINGULAR_SCALE,
    )

    if sigma.min() > SINGULAR_VALUE_TOLERANCE:
        kappa = sigma.max() / sigma.min()
    else:
        kappa = float("inf")

    if center_position is None:
        center_position = np.zeros(3)

    return EllipsoidData(
        center=center_position,
        radii=force_radii,
        axes=U,
        body_name=body_name,
        ellipsoid_type="force",
        condition_number=kappa,
        timestep=engine.get_time(),
    )


def ellipsoid_to_json(ellipsoid: EllipsoidData) -> dict:
    """Convert EllipsoidData to JSON-serializable dictionary.

    Args:
        ellipsoid: EllipsoidData to convert

    Returns:
        Dictionary suitable for JSON serialization
    """
    return {
        "center": ellipsoid.center.tolist(),
        "radii": ellipsoid.radii.tolist(),
        "axes": ellipsoid.axes.tolist(),
        "body_name": ellipsoid.body_name,
        "ellipsoid_type": ellipsoid.ellipsoid_type,
        "condition_number": ellipsoid.condition_number,
        "timestep": ellipsoid.timestep,
    }


def export_ellipsoid_sequence_json(
    sequence: EllipsoidSequence,
    output_path: Path | str,
) -> None:
    """Export ellipsoid sequence to JSON file for external visualization.

    Args:
        sequence: EllipsoidSequence to export
        output_path: Path to output JSON file
    """
    data = {
        "body_name": sequence.body_name,
        "timesteps": sequence.timesteps.tolist(),
        "ellipsoids": [ellipsoid_to_json(e) for e in sequence.ellipsoids],
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    LOGGER.info(f"Exported {len(sequence.ellipsoids)} ellipsoids to {output_path}")


def generate_ellipsoid_mesh(
    ellipsoid: EllipsoidData,
    n_meridians: int = 16,
    n_parallels: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate triangle mesh vertices and faces for an ellipsoid.

    Creates a UV-sphere mesh transformed by the ellipsoid parameters.

    Args:
        ellipsoid: EllipsoidData defining the ellipsoid shape
        n_meridians: Number of longitudinal slices
        n_parallels: Number of latitudinal slices

    Returns:
        Tuple of (vertices, faces):
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of triangle face indices
    """
    # Generate unit sphere vertices
    phi = np.linspace(0, np.pi, n_parallels + 1)
    theta = np.linspace(0, 2 * np.pi, n_meridians + 1)
    phi_grid, theta_grid = np.meshgrid(phi, theta)

    # Unit sphere coordinates
    x_sphere = np.sin(phi_grid) * np.cos(theta_grid)
    y_sphere = np.sin(phi_grid) * np.sin(theta_grid)
    z_sphere = np.cos(phi_grid)

    # Stack into (3, n_vertices) array
    sphere_points = np.stack(
        [x_sphere.ravel(), y_sphere.ravel(), z_sphere.ravel()], axis=0
    )

    # Scale by radii (in principal axis frame)
    # Handle case where we have fewer than 3 radii (2D Jacobian)
    radii = ellipsoid.radii
    if len(radii) < 3:
        # Pad with zeros for missing dimensions
        radii = np.concatenate([radii, np.zeros(3 - len(radii))])

    scaled = sphere_points * radii[:3, np.newaxis]

    # Rotate by principal axes
    axes = ellipsoid.axes
    if axes.shape[0] < 3:
        # Pad axes matrix for 2D case
        axes_full = np.eye(3)
        axes_full[: axes.shape[0], : axes.shape[1]] = axes
        axes = axes_full

    rotated = axes @ scaled

    # Translate to center
    vertices = rotated.T + ellipsoid.center

    # Generate triangle faces (simple grid triangulation, vectorized)
    # Grid is (n_meridians + 1) x (n_parallels + 1), so vertices per row = n_parallels + 1
    n_verts_per_row = n_parallels + 1
    n_rows = n_meridians + 1

    # Create grid indices using meshgrid for vectorized face generation
    i_indices = np.arange(n_rows - 1)
    j_indices = np.arange(n_verts_per_row - 1)
    i_grid, j_grid = np.meshgrid(i_indices, j_indices, indexing="ij")

    # Flatten and compute vertex indices
    v0 = (i_grid * n_verts_per_row + j_grid).ravel()
    v1 = v0 + 1
    v2 = v0 + n_verts_per_row
    v3 = v2 + 1

    # Create faces array (2 triangles per quad)
    n_quads = len(v0)
    faces = np.empty((n_quads * 2, 3), dtype=int)
    faces[0::2] = np.column_stack([v0, v2, v1])  # First triangle of each quad
    faces[1::2] = np.column_stack([v1, v2, v3])  # Second triangle of each quad

    return vertices, faces


def export_ellipsoid_obj(
    ellipsoid: EllipsoidData,
    output_path: Path | str,
) -> None:
    """Export ellipsoid as OBJ mesh file.

    Args:
        ellipsoid: EllipsoidData to export
        output_path: Path to output OBJ file
    """
    vertices, faces = generate_ellipsoid_mesh(ellipsoid)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"# Ellipsoid for {ellipsoid.body_name}\n")
        f.write(f"# Type: {ellipsoid.ellipsoid_type}\n")
        f.write(f"# Condition number: {ellipsoid.condition_number:.2e}\n\n")

        # Write vertices (batched for performance)
        vertex_lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in vertices]
        f.writelines(vertex_lines)

        f.write("\n")

        # Write faces (OBJ uses 1-indexed vertices, batched for performance)
        face_lines = [f"f {face[0]+1} {face[1]+1} {face[2]+1}\n" for face in faces]
        f.writelines(face_lines)

    LOGGER.info(f"Exported ellipsoid mesh to {output_path}")


def export_ellipsoid_stl(
    ellipsoid: EllipsoidData,
    output_path: Path | str,
    binary: bool = True,
) -> None:
    """Export ellipsoid as STL mesh file.

    Args:
        ellipsoid: EllipsoidData to export
        output_path: Path to output STL file
        binary: If True, write binary STL (smaller, faster); else ASCII

    Example:
        >>> ellipsoid = compute_velocity_ellipsoid(engine, "clubhead")
        >>> export_ellipsoid_stl(ellipsoid, "clubhead_velocity.stl")
    """
    vertices, faces = generate_ellipsoid_mesh(ellipsoid)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if binary:
        _write_stl_binary(vertices, faces, output_path, ellipsoid)
    else:
        _write_stl_ascii(vertices, faces, output_path, ellipsoid)

    LOGGER.info(f"Exported ellipsoid STL to {output_path}")


def _write_stl_binary(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    ellipsoid: EllipsoidData,
) -> None:
    """Write binary STL file."""
    import struct

    n_triangles = len(faces)

    with open(output_path, "wb") as f:
        # 80-byte header
        header = (
            f"Ellipsoid: {ellipsoid.body_name} ({ellipsoid.ellipsoid_type})".encode(
                "ascii"
            )[:80].ljust(80, b"\0")
        )
        f.write(header)

        # Number of triangles (uint32)
        f.write(struct.pack("<I", n_triangles))

        # Write each triangle
        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 1.0])

            # Normal (3 floats)
            f.write(struct.pack("<3f", *normal))
            # Vertex 1 (3 floats)
            f.write(struct.pack("<3f", *v0))
            # Vertex 2 (3 floats)
            f.write(struct.pack("<3f", *v1))
            # Vertex 3 (3 floats)
            f.write(struct.pack("<3f", *v2))
            # Attribute byte count (uint16, usually 0)
            f.write(struct.pack("<H", 0))


def _write_stl_ascii(
    vertices: np.ndarray,
    faces: np.ndarray,
    output_path: Path,
    ellipsoid: EllipsoidData,
) -> None:
    """Write ASCII STL file."""
    with open(output_path, "w") as f:
        f.write(f"solid {ellipsoid.body_name}_{ellipsoid.ellipsoid_type}_ellipsoid\n")

        for face in faces:
            v0, v1, v2 = vertices[face[0]], vertices[face[1]], vertices[face[2]]

            # Compute face normal
            edge1 = v1 - v0
            edge2 = v2 - v0
            normal = np.cross(edge1, edge2)
            norm_len = np.linalg.norm(normal)
            if norm_len > 1e-10:
                normal = normal / norm_len
            else:
                normal = np.array([0.0, 0.0, 1.0])

            f.write(f"  facet normal {normal[0]:.6e} {normal[1]:.6e} {normal[2]:.6e}\n")
            f.write("    outer loop\n")
            f.write(f"      vertex {v0[0]:.6e} {v0[1]:.6e} {v0[2]:.6e}\n")
            f.write(f"      vertex {v1[0]:.6e} {v1[1]:.6e} {v1[2]:.6e}\n")
            f.write(f"      vertex {v2[0]:.6e} {v2[1]:.6e} {v2[2]:.6e}\n")
            f.write("    endloop\n")
            f.write("  endfacet\n")

        f.write(
            f"endsolid {ellipsoid.body_name}_{ellipsoid.ellipsoid_type}_ellipsoid\n"
        )


class EllipsoidVisualizer:
    """High-level interface for ellipsoid visualization.

    Provides methods for computing and displaying ellipsoids across
    multiple bodies and time steps.
    """

    def __init__(self, engine: PhysicsEngine) -> None:
        """Initialize visualizer with a physics engine.

        Args:
            engine: Physics engine with loaded model
        """
        self.engine = engine
        self.ellipsoid_cache: dict[str, EllipsoidData] = {}
        self.sequences: dict[str, EllipsoidSequence] = {}

    def update_ellipsoids(self, body_names: list[str]) -> dict[str, EllipsoidData]:
        """Compute ellipsoids for specified bodies at current state.

        Args:
            body_names: List of body names to compute ellipsoids for

        Returns:
            Dictionary mapping body names to their EllipsoidData
        """
        results = {}
        for name in body_names:
            ellipsoid = compute_velocity_ellipsoid(self.engine, name)
            if ellipsoid is not None:
                self.ellipsoid_cache[name] = ellipsoid
                results[name] = ellipsoid
        return results

    def record_frame(self, body_names: list[str]) -> None:
        """Record current ellipsoids to sequences for animation.

        Performance: Uses list append instead of np.append() for 100× speedup.
        Call finalize_sequences() after recording to convert lists to arrays.

        Args:
            body_names: List of body names to record
        """
        t = self.engine.get_time()

        for name in body_names:
            ellipsoid = compute_velocity_ellipsoid(self.engine, name)
            if ellipsoid is None:
                continue

            if name not in self.sequences:
                sequence = EllipsoidSequence(body_name=name)
                # Use list for efficient appending during recording
                sequence._timesteps_list = []  # type: ignore[attr-defined]
                self.sequences[name] = sequence

            self.sequences[name].ellipsoids.append(ellipsoid)
            # Append to list instead of np.append (100× faster)
            self.sequences[name]._timesteps_list.append(t)  # type: ignore[attr-defined]

    def finalize_sequences(self) -> None:
        """Convert internal timestep lists to NumPy arrays after recording.

        Call this method after all record_frame() calls are complete.
        Performance optimization: Avoids O(n²) np.append() overhead.
        """
        for sequence in self.sequences.values():
            if hasattr(sequence, "_timesteps_list"):
                sequence.timesteps = np.array(sequence._timesteps_list)
                # Clean up temporary list to save memory
                delattr(sequence, "_timesteps_list")

    def export_all_json(self, output_dir: Path | str) -> None:
        """Export all recorded sequences to JSON files.

        Automatically finalizes sequences before export.

        Args:
            output_dir: Directory to save JSON files
        """
        # Ensure timesteps are converted to arrays
        self.finalize_sequences()

        output_dir = Path(output_dir)
        for name, sequence in self.sequences.items():
            filename = f"{name}_ellipsoids.json"
            export_ellipsoid_sequence_json(sequence, output_dir / filename)

    def get_manipulability_summary(self, body_name: str) -> dict[str, float] | None:
        """Get summary statistics for current ellipsoid.

        Args:
            body_name: Name of body

        Returns:
            Dictionary with manipulability metrics, or None if not computed
        """
        if body_name not in self.ellipsoid_cache:
            return None

        ellipsoid = self.ellipsoid_cache[body_name]
        radii = ellipsoid.radii

        return {
            "max_radius": float(radii.max()),
            "min_radius": float(radii.min()),
            "manipulability_index": float(np.prod(radii)),
            "condition_number": ellipsoid.condition_number,
            "isotropy": float(radii.min() / radii.max()) if radii.max() > 0 else 0,
        }
