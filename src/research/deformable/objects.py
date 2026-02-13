"""Deformable object simulation classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.constants import GRAVITY

if TYPE_CHECKING:
    from numpy.typing import NDArray


@dataclass
class MaterialProperties:
    """Material properties for deformable objects.

    Attributes:
        youngs_modulus: Young's modulus (Pa).
        poisson_ratio: Poisson's ratio (dimensionless).
        density: Material density (kg/m³).
        damping: Damping coefficient.
        bending_stiffness: Bending stiffness for shells/cloth.
        shear_stiffness: Shear stiffness for cloth.
    """

    youngs_modulus: float = 1e6
    poisson_ratio: float = 0.3
    density: float = 1000.0
    damping: float = 0.01
    bending_stiffness: float | None = None
    shear_stiffness: float | None = None

    @property
    def shear_modulus(self) -> float:
        """Compute shear modulus from Young's modulus."""
        return self.youngs_modulus / (2 * (1 + self.poisson_ratio))

    @property
    def bulk_modulus(self) -> float:
        """Compute bulk modulus from Young's modulus."""
        return self.youngs_modulus / (3 * (1 - 2 * self.poisson_ratio))


class DeformableObject(ABC):
    """Base class for deformable objects.

    Provides interface for simulating deformable bodies including
    mesh representation, material properties, and force application.

    Attributes:
        mesh: Mesh node positions (N, 3).
        material: Material properties.
    """

    def __init__(
        self,
        mesh: NDArray[np.floating],
        material: MaterialProperties,
    ) -> None:
        """Initialize deformable object.

        Args:
            mesh: Initial mesh node positions (N, 3).
            material: Material properties.
        """
        self._mesh = mesh.copy()
        self._rest_mesh = mesh.copy()
        self._velocities = np.zeros_like(mesh)
        self._material = material
        self._external_forces = np.zeros_like(mesh)
        self._fixed_nodes: set[int] = set()

    @property
    def mesh(self) -> NDArray[np.floating]:
        """Current mesh node positions."""
        return self._mesh

    @property
    def material(self) -> MaterialProperties:
        """Material properties."""
        return self._material

    @property
    def n_nodes(self) -> int:
        """Number of mesh nodes."""
        return len(self._mesh)

    def get_node_positions(self) -> NDArray[np.floating]:
        """Get current node positions.

        Returns:
            Node positions (N, 3).
        """
        return self._mesh.copy()

    def get_node_velocities(self) -> NDArray[np.floating]:
        """Get current node velocities.

        Returns:
            Node velocities (N, 3).
        """
        return self._velocities.copy()

    def set_node_positions(self, positions: NDArray[np.floating]) -> None:
        """Set node positions directly.

        Args:
            positions: New node positions (N, 3).
        """
        self._mesh = positions.copy()

    def apply_external_force(
        self,
        node_indices: NDArray[np.intp] | list[int],
        forces: NDArray[np.floating],
    ) -> None:
        """Apply external forces to specific nodes.

        Args:
            node_indices: Indices of nodes to apply force to.
            forces: Force vectors (len(node_indices), 3) or (3,) for all.
        """
        if forces.ndim == 1:
            forces = np.tile(forces, (len(node_indices), 1))

        for i, idx in enumerate(node_indices):
            self._external_forces[idx] += forces[i]

    def clear_external_forces(self) -> None:
        """Clear all external forces."""
        self._external_forces = np.zeros_like(self._mesh)

    def fix_nodes(self, node_indices: list[int]) -> None:
        """Fix nodes in place (Dirichlet boundary condition).

        Args:
            node_indices: Indices of nodes to fix.
        """
        self._fixed_nodes.update(node_indices)

    def unfix_nodes(self, node_indices: list[int]) -> None:
        """Release fixed nodes.

        Args:
            node_indices: Indices of nodes to release.
        """
        self._fixed_nodes -= set(node_indices)

    @abstractmethod
    def step(self, dt: float) -> None:
        """Advance simulation by one timestep.

        Args:
            dt: Timestep in seconds.
        """

    @abstractmethod
    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute internal elastic forces.

        Returns:
            Internal force vectors (N, 3).
        """

    def reset(self) -> None:
        """Reset to rest configuration."""
        self._mesh = self._rest_mesh.copy()
        self._velocities = np.zeros_like(self._mesh)
        self._external_forces = np.zeros_like(self._mesh)


class SoftBody(DeformableObject):
    """Volumetric soft body simulation using FEM.

    Uses tetrahedral finite elements with neo-Hookean material model.

    Attributes:
        tetrahedra: Tetrahedral element connectivity.
    """

    def __init__(
        self,
        mesh: NDArray[np.floating],
        tetrahedra: NDArray[np.intp],
        material: MaterialProperties,
    ) -> None:
        """Initialize soft body.

        Args:
            mesh: Node positions (N, 3).
            tetrahedra: Tetrahedral connectivity (M, 4).
            material: Material properties.
        """
        super().__init__(mesh, material)
        self._tetrahedra = tetrahedra
        self._rest_volumes = self._compute_volumes(self._rest_mesh)
        self._B_matrices = self._compute_shape_matrices()

    def _compute_volumes(self, positions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Compute volumes of tetrahedra.

        Args:
            positions: Node positions.

        Returns:
            Volumes for each tetrahedron.
        """
        volumes = np.zeros(len(self._tetrahedra))

        for i, tet in enumerate(self._tetrahedra):
            v0 = positions[tet[0]]
            v1 = positions[tet[1]]
            v2 = positions[tet[2]]
            v3 = positions[tet[3]]

            # Volume = |det([v1-v0, v2-v0, v3-v0])| / 6
            mat = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            volumes[i] = abs(np.linalg.det(mat)) / 6

        return volumes

    def _compute_shape_matrices(self) -> list[NDArray[np.floating]]:
        """Compute shape function matrices for each element.

        Returns:
            List of B matrices.
        """
        B_matrices = []

        for tet in self._tetrahedra:
            v0 = self._rest_mesh[tet[0]]
            v1 = self._rest_mesh[tet[1]]
            v2 = self._rest_mesh[tet[2]]
            v3 = self._rest_mesh[tet[3]]

            # Shape function derivatives (constant strain tetrahedron)
            D = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            try:
                B = np.linalg.inv(D)
            except np.linalg.LinAlgError:
                B = np.zeros((3, 3))

            B_matrices.append(B)

        return B_matrices

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute internal elastic forces using FEM.

        Returns:
            Internal forces (N, 3).
        """
        forces = np.zeros_like(self._mesh)

        mu = self._material.shear_modulus
        lam = (
            self._material.youngs_modulus
            * self._material.poisson_ratio
            / (
                (1 + self._material.poisson_ratio)
                * (1 - 2 * self._material.poisson_ratio)
            )
        )

        for i, tet in enumerate(self._tetrahedra):
            # Compute deformation gradient F
            v0 = self._mesh[tet[0]]
            v1 = self._mesh[tet[1]]
            v2 = self._mesh[tet[2]]
            v3 = self._mesh[tet[3]]

            D = np.column_stack([v1 - v0, v2 - v0, v3 - v0])
            F = D @ self._B_matrices[i]

            # Neo-Hookean stress (simplified)
            J = np.linalg.det(F)
            if J <= 0:
                J = 0.01  # Prevent inversion

            # First Piola-Kirchhoff stress
            P = mu * (F - np.linalg.inv(F).T) + lam * np.log(J) * np.linalg.inv(F).T

            # Nodal forces
            H = -self._rest_volumes[i] * P @ self._B_matrices[i].T

            forces[tet[1]] += H[:, 0]
            forces[tet[2]] += H[:, 1]
            forces[tet[3]] += H[:, 2]
            forces[tet[0]] -= H[:, 0] + H[:, 1] + H[:, 2]

        return forces

    def step(self, dt: float) -> None:
        """Advance simulation using explicit Euler.

        Args:
            dt: Timestep.
        """
        # Compute forces
        internal_forces = self.compute_internal_forces()
        total_forces = internal_forces + self._external_forces

        # Apply damping
        total_forces -= self._material.damping * self._velocities

        # Compute acceleration (mass assumed uniform)
        node_mass = self._material.density * np.sum(self._rest_volumes) / self.n_nodes
        accelerations = total_forces / node_mass

        # Update velocities and positions
        self._velocities += accelerations * dt
        self._mesh += self._velocities * dt

        # Fix boundary nodes
        for idx in self._fixed_nodes:
            self._mesh[idx] = self._rest_mesh[idx]
            self._velocities[idx] = 0

        # Clear external forces
        self.clear_external_forces()


class Cable(DeformableObject):
    """1D deformable cable/rope simulation.

    Uses mass-spring model with bending resistance.

    Attributes:
        rest_length: Rest length of the cable.
    """

    def __init__(
        self,
        mesh: NDArray[np.floating],
        material: MaterialProperties,
        rest_lengths: NDArray[np.floating] | None = None,
    ) -> None:
        """Initialize cable.

        Args:
            mesh: Node positions along cable (N, 3).
            material: Material properties.
            rest_lengths: Rest lengths between nodes (optional).
        """
        super().__init__(mesh, material)

        if rest_lengths is None:
            # Compute from initial mesh
            self._rest_lengths = np.linalg.norm(np.diff(mesh, axis=0), axis=1)
        else:
            self._rest_lengths = rest_lengths

        self._total_rest_length = float(np.sum(self._rest_lengths))

    @property
    def rest_length(self) -> float:
        """Total rest length of cable."""
        return self._total_rest_length

    def get_length(self) -> float:
        """Get current cable length.

        Returns:
            Current total length.
        """
        segments = np.diff(self._mesh, axis=0)
        return float(np.sum(np.linalg.norm(segments, axis=1)))

    def get_tension(self) -> float:
        """Get average cable tension.

        Returns:
            Average tension in N.
        """
        forces = self.compute_internal_forces()
        # Average force magnitude
        return float(np.mean(np.linalg.norm(forces, axis=1)))

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute spring and bending forces.

        Returns:
            Internal forces (N, 3).
        """
        forces = np.zeros_like(self._mesh)
        k_stretch = self._material.youngs_modulus
        k_bend = self._material.bending_stiffness or k_stretch * 0.1

        # Spring forces
        for i in range(len(self._mesh) - 1):
            delta = self._mesh[i + 1] - self._mesh[i]
            length = np.linalg.norm(delta)

            if length > 1e-10:
                direction = delta / length
                strain = (length - self._rest_lengths[i]) / self._rest_lengths[i]
                force_mag = k_stretch * strain

                force = force_mag * direction
                forces[i] += force
                forces[i + 1] -= force

        # Bending forces
        for i in range(1, len(self._mesh) - 1):
            v1 = self._mesh[i] - self._mesh[i - 1]
            v2 = self._mesh[i + 1] - self._mesh[i]

            # Angle between segments
            l1 = np.linalg.norm(v1)
            l2 = np.linalg.norm(v2)

            if l1 > 1e-10 and l2 > 1e-10:
                cos_angle = np.dot(v1, v2) / (l1 * l2)
                cos_angle = np.clip(cos_angle, -1, 1)

                # Bending force (simplified)
                bend_force = k_bend * (1 - cos_angle)
                direction = v2 / l2 - v1 / l1
                direction_norm = np.linalg.norm(direction)

                if direction_norm > 1e-10:
                    forces[i] -= bend_force * direction / direction_norm

        return forces

    def step(self, dt: float) -> None:
        """Advance cable simulation.

        Args:
            dt: Timestep.
        """
        internal_forces = self.compute_internal_forces()
        total_forces = internal_forces + self._external_forces

        # Gravity
        gravity = np.array([0.0, 0.0, -GRAVITY])
        node_mass = self._material.density * self._total_rest_length / self.n_nodes
        total_forces += node_mass * gravity

        # Damping
        total_forces -= self._material.damping * self._velocities

        # Integration
        accelerations = total_forces / node_mass
        self._velocities += accelerations * dt
        self._mesh += self._velocities * dt

        # Fixed nodes
        for idx in self._fixed_nodes:
            self._mesh[idx] = self._rest_mesh[idx]
            self._velocities[idx] = 0

        self.clear_external_forces()


class Cloth(DeformableObject):
    """2D deformable cloth/fabric simulation.

    Uses mass-spring model with stretch, shear, and bend springs.

    Attributes:
        width: Number of nodes in width direction.
        height: Number of nodes in height direction.
    """

    def __init__(
        self,
        mesh: NDArray[np.floating],
        width: int,
        height: int,
        material: MaterialProperties,
    ) -> None:
        """Initialize cloth.

        Args:
            mesh: Node positions (width*height, 3).
            width: Grid width.
            height: Grid height.
            material: Material properties.
        """
        super().__init__(mesh, material)
        self._width = width
        self._height = height

        # Build spring connectivity
        self._springs = self._build_springs()

    @property
    def width(self) -> int:
        """Grid width."""
        return self._width

    @property
    def height(self) -> int:
        """Grid height."""
        return self._height

    def _build_springs(self) -> list[tuple[int, int, float, str]]:
        """Build spring connectivity.

        Returns:
            List of (i, j, rest_length, type) tuples.
        """
        springs = []

        def node_idx(x: int, y: int) -> int:
            """Convert 2-D grid coordinates to a flat node index."""
            return y * self._width + x

        # Structural springs (horizontal and vertical)
        for y in range(self._height):
            for x in range(self._width):
                idx = node_idx(x, y)

                # Horizontal
                if x < self._width - 1:
                    idx2 = node_idx(x + 1, y)
                    rest = np.linalg.norm(self._rest_mesh[idx] - self._rest_mesh[idx2])
                    springs.append((idx, idx2, rest, "stretch"))

                # Vertical
                if y < self._height - 1:
                    idx2 = node_idx(x, y + 1)
                    rest = np.linalg.norm(self._rest_mesh[idx] - self._rest_mesh[idx2])
                    springs.append((idx, idx2, rest, "stretch"))

        # Shear springs (diagonal)
        for y in range(self._height - 1):
            for x in range(self._width - 1):
                idx = node_idx(x, y)

                # Diagonal 1
                idx2 = node_idx(x + 1, y + 1)
                rest = np.linalg.norm(self._rest_mesh[idx] - self._rest_mesh[idx2])
                springs.append((idx, idx2, rest, "shear"))

                # Diagonal 2
                idx1 = node_idx(x + 1, y)
                idx2 = node_idx(x, y + 1)
                rest = np.linalg.norm(self._rest_mesh[idx1] - self._rest_mesh[idx2])
                springs.append((idx1, idx2, rest, "shear"))

        # Bend springs (skip one node)
        for y in range(self._height):
            for x in range(self._width):
                idx = node_idx(x, y)

                # Horizontal bend
                if x < self._width - 2:
                    idx2 = node_idx(x + 2, y)
                    rest = np.linalg.norm(self._rest_mesh[idx] - self._rest_mesh[idx2])
                    springs.append((idx, idx2, rest, "bend"))

                # Vertical bend
                if y < self._height - 2:
                    idx2 = node_idx(x, y + 2)
                    rest = np.linalg.norm(self._rest_mesh[idx] - self._rest_mesh[idx2])
                    springs.append((idx, idx2, rest, "bend"))

        return springs  # type: ignore[return-value]

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute spring forces for cloth.

        Returns:
            Internal forces (N, 3).
        """
        forces = np.zeros_like(self._mesh)

        k_stretch = self._material.youngs_modulus
        k_shear = self._material.shear_stiffness or k_stretch * 0.5
        k_bend = self._material.bending_stiffness or k_stretch * 0.1

        for i, j, rest_length, spring_type in self._springs:
            delta = self._mesh[j] - self._mesh[i]
            length = np.linalg.norm(delta)

            if length < 1e-10:
                continue

            # Spring stiffness based on type
            if spring_type == "stretch":
                k = k_stretch
            elif spring_type == "shear":
                k = k_shear
            else:  # bend
                k = k_bend

            direction = delta / length
            strain = length - rest_length
            force = k * strain * direction

            forces[i] += force
            forces[j] -= force

        return forces

    def step(self, dt: float) -> None:
        """Advance cloth simulation.

        Args:
            dt: Timestep.
        """
        internal_forces = self.compute_internal_forces()
        total_forces = internal_forces + self._external_forces

        # Gravity
        gravity = np.array([0.0, 0.0, -GRAVITY])
        node_mass = self._material.density * 0.01  # Assume thin cloth
        total_forces += node_mass * gravity

        # Damping
        total_forces -= self._material.damping * self._velocities

        # Integration
        accelerations = total_forces / node_mass
        self._velocities += accelerations * dt
        self._mesh += self._velocities * dt

        # Fixed nodes
        for idx in self._fixed_nodes:
            self._mesh[idx] = self._rest_mesh[idx]
            self._velocities[idx] = 0

        self.clear_external_forces()

    def attach_to_body(
        self,
        body_id: str,
        attachment_nodes: list[int],
        body_positions: NDArray[np.floating],
    ) -> None:
        """Attach cloth nodes to a rigid body.

        Args:
            body_id: Rigid body identifier.
            attachment_nodes: Node indices to attach.
            body_positions: Positions of attachment points on body.
        """
        for node_idx, body_pos in zip(attachment_nodes, body_positions, strict=True):
            self._mesh[node_idx] = body_pos
            self._velocities[node_idx] = 0
            self._fixed_nodes.add(node_idx)
