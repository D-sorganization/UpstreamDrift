# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Deformable object simulation classes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import math
import numpy as np

from src.shared.python.core.constants import GRAVITY

if TYPE_CHECKING:
    from numpy.typing import NDArray

_MIN_SPRING_LENGTH = 1e-10


def _validate_node_mesh(mesh: NDArray[np.floating]) -> None:
    if mesh.ndim != 2 or mesh.shape[1] != 3:
        raise ValueError("mesh must have shape (N, 3)")
    if not np.all(np.isfinite(mesh)):
        raise ValueError("mesh must contain only finite coordinates")


def _validate_spring_arrays(
    mesh: NDArray[np.floating],
    i_idx: NDArray[np.intp],
    j_idx: NDArray[np.intp],
    rest_lengths: NDArray[np.floating],
    stiffness: NDArray[np.floating],
) -> None:
    if not (
        i_idx.shape == j_idx.shape == rest_lengths.shape == stiffness.shape
        and i_idx.ndim == 1
    ):
        raise ValueError("spring arrays must be one-dimensional arrays of equal length")
    if i_idx.size == 0:
        return
    if np.min(i_idx) < 0 or np.min(j_idx) < 0:
        raise ValueError("spring indices must be non-negative")
    if np.max(i_idx) >= len(mesh) or np.max(j_idx) >= len(mesh):
        raise ValueError("spring indices must be within mesh bounds")
    if not (np.all(np.isfinite(rest_lengths)) and np.all(np.isfinite(stiffness))):
        raise ValueError("spring rest lengths and stiffnesses must be finite")


def _accumulate_spring_forces(
    mesh: NDArray[np.floating],
    i_idx: NDArray[np.intp],
    j_idx: NDArray[np.intp],
    rest_lengths: NDArray[np.floating],
    stiffness: NDArray[np.floating],
    *,
    rest_normalized: bool,
) -> NDArray[np.floating]:
    """Vectorized spring force scatter for cable and cloth internals."""
    _validate_node_mesh(mesh)
    _validate_spring_arrays(mesh, i_idx, j_idx, rest_lengths, stiffness)
    forces = np.zeros_like(mesh)
    if i_idx.size == 0:
        return forces

    delta = mesh[j_idx] - mesh[i_idx]
    lengths = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    active = lengths > _MIN_SPRING_LENGTH
    if not np.any(active):
        return forces

    safe_lengths = lengths[active]
    active_delta = delta[active]
    if rest_normalized:
        active_rest = rest_lengths[active]
        if np.any(active_rest <= _MIN_SPRING_LENGTH):
            raise ValueError("cable spring rest lengths must be positive")
        extension = (safe_lengths - active_rest) / active_rest
    else:
        extension = safe_lengths - rest_lengths[active]

    spring_forces = (stiffness[active] * extension / safe_lengths)[
        :, np.newaxis
    ] * active_delta
    np.add.at(forces, i_idx[active], spring_forces)
    np.add.at(forces, j_idx[active], -spring_forces)
    if not np.all(np.isfinite(forces)):
        raise ValueError("computed spring forces must be finite")
    return forces


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
        if mesh is None:
            raise ValueError("mesh must be provided")
        _validate_node_mesh(mesh)
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
        if positions is None:
            raise ValueError("positions must be provided")
        if positions.shape != self._mesh.shape:
            raise ValueError("positions must match current mesh shape")
        _validate_node_mesh(positions)
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
        if node_indices is None:
            raise ValueError("node_indices must be provided")
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
        if mesh is None:
            raise ValueError("mesh must be provided")
        super().__init__(mesh, material)
        tetrahedra = np.asarray(tetrahedra, dtype=np.intp)
        if tetrahedra.ndim != 2 or tetrahedra.shape[1] != 4:
            raise ValueError("tetrahedra must have shape (M, 4)")
        if tetrahedra.size and (
            np.min(tetrahedra) < 0 or np.max(tetrahedra) >= self.n_nodes
        ):
            raise ValueError("tetrahedra indices must be within mesh bounds")
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
        if positions is None:
            raise ValueError("positions must be provided")
        _validate_node_mesh(positions)
        if len(self._tetrahedra) == 0:
            return np.zeros(0, dtype=positions.dtype)

        vertices = positions[self._tetrahedra]
        mats = np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
        return np.abs(np.linalg.det(mats)) / 6

    def _compute_shape_matrices(self) -> NDArray[np.floating]:
        """Compute shape function matrices for each element.

        Returns:
            B matrices with shape (M, 3, 3).
        """
        if len(self._tetrahedra) == 0:
            return np.zeros((0, 3, 3), dtype=self._rest_mesh.dtype)

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

        return np.asarray(B_matrices, dtype=self._rest_mesh.dtype)

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute internal elastic forces using FEM.

        Returns:
            Internal forces (N, 3).
        """
        forces = np.zeros_like(self._mesh)
        if len(self._tetrahedra) == 0:
            return forces

        mu = self._material.shear_modulus
        lam = (
            self._material.youngs_modulus
            * self._material.poisson_ratio
            / (
                (1 + self._material.poisson_ratio)
                * (1 - 2 * self._material.poisson_ratio)
            )
        )

        vertices = self._mesh[self._tetrahedra]
        D = np.stack(
            (
                vertices[:, 1] - vertices[:, 0],
                vertices[:, 2] - vertices[:, 0],
                vertices[:, 3] - vertices[:, 0],
            ),
            axis=2,
        )
        F = D @ self._B_matrices

        # Neo-Hookean stress (simplified)
        J = np.linalg.det(F)
        safe_J = np.where(J <= 0, 0.01, J)

        # First Piola-Kirchhoff stress. Invert each deformation gradient once.
        F_inv_t = np.linalg.inv(F).transpose(0, 2, 1)
        P = (
            mu * (F - F_inv_t)
            + lam * np.log(safe_J)[:, np.newaxis, np.newaxis] * F_inv_t
        )

        # Nodal forces
        H = -self._rest_volumes[:, np.newaxis, np.newaxis] * (
            P @ self._B_matrices.transpose(0, 2, 1)
        )

        np.add.at(forces, self._tetrahedra[:, 1], H[:, :, 0])
        np.add.at(forces, self._tetrahedra[:, 2], H[:, :, 1])
        np.add.at(forces, self._tetrahedra[:, 3], H[:, :, 2])
        np.add.at(forces, self._tetrahedra[:, 0], -np.einsum("ijk->ij", H))

        if not np.all(np.isfinite(forces)):
            raise ValueError("computed FEM internal forces must be finite")

        return forces

    def step(self, dt: float) -> None:
        """Advance simulation using explicit Euler.

        Args:
            dt: Timestep.
        """
        # Compute forces
        if dt is None:
            raise ValueError("dt must be provided")
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
        if mesh is None:
            raise ValueError("mesh must be provided")
        super().__init__(mesh, material)

        if rest_lengths is None:
            # Compute from initial mesh
            # ⚡ Bolt: np.einsum is much faster than np.sum(np.square(...), axis=-1)
            # and avoids temporary array allocations
            diffs = np.diff(mesh, axis=0)
            self._rest_lengths = np.sqrt(np.einsum("ij,ij->i", diffs, diffs))
        else:
            self._rest_lengths = np.asarray(rest_lengths, dtype=mesh.dtype).copy()

        self._total_rest_length = float(np.sum(self._rest_lengths))
        self._spring_i = np.arange(max(len(self._mesh) - 1, 0), dtype=np.intp)
        self._spring_j = self._spring_i + 1
        self._spring_rest_lengths = self._rest_lengths.copy()
        self._spring_stiffness = np.full(
            self._spring_i.shape,
            self._material.youngs_modulus,
            dtype=self._mesh.dtype,
        )

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
        # ⚡ Bolt: np.einsum is much faster than np.sum(np.square(...), axis=-1)
        # and avoids temporary array allocations
        return float(np.sum(np.sqrt(np.einsum("ij,ij->i", segments, segments))))

    def get_tension(self) -> float:
        """Get average cable tension.

        Returns:
            Average tension in N.
        """
        forces = self.compute_internal_forces()
        # Average force magnitude
        # ⚡ Bolt: np.einsum is much faster than np.sum(np.square(...), axis=-1)
        # and avoids temporary array allocations
        return float(np.mean(np.sqrt(np.einsum("ij,ij->i", forces, forces))))

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute spring and bending forces.

        Returns:
            Internal forces (N, 3).
        """
        forces = np.zeros_like(self._mesh)
        k_stretch = self._material.youngs_modulus
        k_bend = self._material.bending_stiffness or k_stretch * 0.1

        # Spring forces
        self._spring_stiffness.fill(k_stretch)
        forces += _accumulate_spring_forces(
            self._mesh,
            self._spring_i,
            self._spring_j,
            self._spring_rest_lengths,
            self._spring_stiffness,
            rest_normalized=True,
        )

        # Bending forces
        if len(self._mesh) > 2:
            v1 = self._mesh[1:-1] - self._mesh[:-2]
            v2 = self._mesh[2:] - self._mesh[1:-1]
            l1 = np.sqrt(np.einsum("ij,ij->i", v1, v1))
            l2 = np.sqrt(np.einsum("ij,ij->i", v2, v2))
            active = (l1 > _MIN_SPRING_LENGTH) & (l2 > _MIN_SPRING_LENGTH)
            if np.any(active):
                cos_angle = np.einsum("ij,ij->i", v1[active], v2[active]) / (
                    l1[active] * l2[active]
                )
                cos_angle = np.clip(cos_angle, -1, 1)
                bend_force = k_bend * (1 - cos_angle)
                direction = (
                    v2[active] / l2[active, np.newaxis]
                    - v1[active] / l1[active, np.newaxis]
                )
                direction_norm = np.sqrt(np.einsum("ij,ij->i", direction, direction))
                bending_active = direction_norm > _MIN_SPRING_LENGTH
                if np.any(bending_active):
                    middle_indices = np.arange(1, len(self._mesh) - 1, dtype=np.intp)[
                        active
                    ][bending_active]
                    forces[middle_indices] -= (
                        bend_force[bending_active, np.newaxis]
                        * direction[bending_active]
                        / direction_norm[bending_active, np.newaxis]
                    )

        return forces

    def step(self, dt: float) -> None:
        """Advance cable simulation.

        Args:
            dt: Timestep.
        """
        if dt is None:
            raise ValueError("dt must be provided")
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
        if mesh is None:
            raise ValueError("mesh must be provided")
        super().__init__(mesh, material)
        self._width = width
        self._height = height

        # Build spring connectivity
        self._springs = self._build_springs()
        self._build_spring_arrays()

    @property
    def width(self) -> int:
        """Grid width."""
        return self._width

    @property
    def height(self) -> int:
        """Grid height."""
        return self._height

    def _build_springs(self) -> list[tuple[int, int, float, str]]:  # noqa: C901
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
                    diff = self._rest_mesh[idx] - self._rest_mesh[idx2]
                    rest = math.hypot(diff[0], diff[1], diff[2])
                    springs.append((idx, idx2, rest, "stretch"))

                # Vertical
                if y < self._height - 1:
                    idx2 = node_idx(x, y + 1)
                    diff = self._rest_mesh[idx] - self._rest_mesh[idx2]
                    rest = math.hypot(diff[0], diff[1], diff[2])
                    springs.append((idx, idx2, rest, "stretch"))

        # Shear springs (diagonal)
        for y in range(self._height - 1):
            for x in range(self._width - 1):
                idx = node_idx(x, y)

                # Diagonal 1
                idx2 = node_idx(x + 1, y + 1)
                diff = self._rest_mesh[idx] - self._rest_mesh[idx2]
                rest = math.hypot(diff[0], diff[1], diff[2])
                springs.append((idx, idx2, rest, "shear"))

                # Diagonal 2
                idx1 = node_idx(x + 1, y)
                idx2 = node_idx(x, y + 1)
                diff = self._rest_mesh[idx1] - self._rest_mesh[idx2]
                rest = math.hypot(diff[0], diff[1], diff[2])
                springs.append((idx1, idx2, rest, "shear"))

        # Bend springs (skip one node)
        for y in range(self._height):
            for x in range(self._width):
                idx = node_idx(x, y)

                # Horizontal bend
                if x < self._width - 2:
                    idx2 = node_idx(x + 2, y)
                    diff = self._rest_mesh[idx] - self._rest_mesh[idx2]
                    rest = math.hypot(diff[0], diff[1], diff[2])
                    springs.append((idx, idx2, rest, "bend"))

                # Vertical bend
                if y < self._height - 2:
                    idx2 = node_idx(x, y + 2)
                    diff = self._rest_mesh[idx] - self._rest_mesh[idx2]
                    rest = math.hypot(diff[0], diff[1], diff[2])
                    springs.append((idx, idx2, rest, "bend"))

        return springs  # type: ignore[return-value]

    def _build_spring_arrays(self) -> None:
        """Cache cloth spring connectivity in vectorized arrays."""
        if not self._springs:
            self._spring_i = np.zeros(0, dtype=np.intp)
            self._spring_j = np.zeros(0, dtype=np.intp)
            self._spring_rest_lengths = np.zeros(0, dtype=self._mesh.dtype)
            self._spring_stiffness = np.zeros(0, dtype=self._mesh.dtype)
            return

        k_stretch = self._material.youngs_modulus
        k_shear = self._material.shear_stiffness or k_stretch * 0.5
        k_bend = self._material.bending_stiffness or k_stretch * 0.1
        stiffness_by_type = {
            "stretch": k_stretch,
            "shear": k_shear,
            "bend": k_bend,
        }
        self._spring_i = np.fromiter(
            (spring[0] for spring in self._springs),
            dtype=np.intp,
            count=len(self._springs),
        )
        self._spring_j = np.fromiter(
            (spring[1] for spring in self._springs),
            dtype=np.intp,
            count=len(self._springs),
        )
        self._spring_rest_lengths = np.fromiter(
            (spring[2] for spring in self._springs),
            dtype=self._mesh.dtype,
            count=len(self._springs),
        )
        self._spring_stiffness = np.fromiter(
            (stiffness_by_type[spring[3]] for spring in self._springs),
            dtype=self._mesh.dtype,
            count=len(self._springs),
        )

    def compute_internal_forces(self) -> NDArray[np.floating]:
        """Compute spring forces for cloth.

        Returns:
            Internal forces (N, 3).
        """
        return _accumulate_spring_forces(
            self._mesh,
            self._spring_i,
            self._spring_j,
            self._spring_rest_lengths,
            self._spring_stiffness,
            rest_normalized=False,
        )

    def step(self, dt: float) -> None:
        """Advance cloth simulation.

        Args:
            dt: Timestep.
        """
        if dt is None:
            raise ValueError("dt must be provided")
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
