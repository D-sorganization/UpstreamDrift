"""The F1 background grid and its particle-grid transfers.

Transfer scheme: APIC, and why not the other two
------------------------------------------------

Material-point methods differ mainly in what they carry back and forth
between particles and the grid, and the classical choices both fail here:

* **PIC** (Harlow 1964) resets each particle's velocity to the
  interpolated grid velocity, discarding every mode the grid cannot
  represent.  The loss shows up as strong, resolution-dependent numerical
  damping.  In this tier that is fatal in a specific way: the shear bands
  ADR-0033 asks F1 to *show* are exactly the fine-scale velocity
  structure PIC erases, so a PIC solve would render a smooth picture of a
  localisation that the model damped away.
* **FLIP** (Brackbill & Ruppel 1986) carries the grid velocity
  *increment* instead, which preserves the energy PIC loses but leaves
  the particle velocity field with null-space noise the grid never sees.
  In a plastic material the noise feeds the return mapping and shows up
  as ringing and spurious yielding.

**APIC** (Jiang, Schroeder, Selle, Teran & Stomakhin 2015, *"The affine
particle-in-cell method"*, ACM Trans. Graph. **34**(4):51) carries a
per-particle affine velocity field ``C_p`` alongside the velocity.  It is
as stable as PIC and, unlike either predecessor, conserves **both linear
and angular momentum** exactly across the transfer.  That last property
is the reason it is chosen and not merely accepted: the F1 verification
suite tests angular-momentum conservation of a P2G/G2P round trip to
round-off, and under PIC or FLIP that test could not pass, so the
transfer scheme is falsifiable here rather than asserted.

MLS-MPM (Hu et al. 2018) would be a defensible alternative -- it is APIC
with the stress term folded into the same affine machinery -- but it
couples the transfer and the force discretisation, and keeping them
separate is what lets the two be verified independently below.

Basis functions
---------------

Quadratic B-splines, following Steffen, Kirby & Berzins (2008),
*"Analysis and reduction of quadrature errors in the material point
method"*, Int. J. Numer. Meth. Engng **76**:922-948.  They are ``C^1``,
so a particle crossing a cell boundary sees no gradient discontinuity --
the cell-crossing noise that afflicts linear-hat MPM simply does not
arise.  Each particle touches a fixed ``3 x 3`` node stencil.

Two identities of this basis are used as verification anchors, both to
round-off:

* ``sum_i w_ip = 1`` -- the partition of unity, which makes the P2G mass
  transfer conserve total mass exactly.
* ``sum_i grad w_ip = 0`` -- which makes the internal forces sum to zero,
  so total linear momentum changes only by gravity and the contact
  impulse and by nothing else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import SolverInputError

__all__ = [
    "NODES_PER_PARTICLE",
    "STENCIL_WIDTH",
    "GridInterpolation",
    "PlaneStrainGrid",
    "affine_from_grid_velocity",
    "apic_angular_momentum",
    "cross_2d",
    "gather_velocity",
    "scatter_mass",
    "scatter_momentum",
    "scatter_stress_force",
    "velocity_gradient",
]

STENCIL_WIDTH = 3
"""Nodes per axis in a quadratic B-spline stencil."""

NODES_PER_PARTICLE = STENCIL_WIDTH * STENCIL_WIDTH
"""Nodes one plane-strain particle writes to: ``3 x 3``."""

_DIMENSION = 2
_APIC_INERTIA_FACTOR = 4.0
"""``D_p^{-1} dx^2`` for quadratic B-splines: ``D_p = (dx^2 / 4) I``.

Constant because the quadratic B-spline stencil is symmetric about the
particle, which is precisely why APIC is cheap with this basis (Jiang et
al. 2015, section 5.2)."""


@dataclass(frozen=True, slots=True)
class GridInterpolation:
    """One step's particle-to-node stencil, computed once and reused.

    Every transfer in a step -- mass, momentum, force, and both gathers --
    reads the same weights, so they are formed once.  That is not only a
    speed decision: recomputing them would let the scatter and the gather
    drift apart under a future edit, and the conservation identities this
    solver verifies hold only if they are literally the same numbers.

    Attributes:
        node_index: ``(n, 9)`` flat node indices, row-major in ``(x, z)``.
        weight: ``(n, 9)`` B-spline weights, each row summing to 1.
        weight_gradient: ``(n, 9, 2)`` weight gradients in 1/m, each row
            summing to the zero vector.
        offset_m: ``(n, 9, 2)`` node position minus particle position.
    """

    node_index: NDArray[np.int64]
    weight: NDArray[np.float64]
    weight_gradient: NDArray[np.float64]
    offset_m: NDArray[np.float64]

    @property
    def n_particles(self) -> int:
        """Number of particles this stencil was formed for."""
        return int(self.node_index.shape[0])


@dataclass(frozen=True)
class PlaneStrainGrid:
    """A uniform Eulerian grid over the in-plane ``(x, z)`` section.

    The plane is the swing plane: ``x`` runs along the club's path and
    ``z`` is up, matching the world frame the F0 solver and
    :class:`~bunkershot3d.solvers.protocol.IntrusionState` already use, so
    a caller never has to remember a second axis convention.  ``y`` --
    heel to toe -- is the out-of-plane direction that does not exist in
    this model.

    Attributes:
        origin_m: ``(2,)`` position of node ``(0, 0)``.
        cell_size_m: ``dx``, uniform in both axes.
        node_counts: ``(nx, nz)`` node counts, at least 3 on each axis.
    """

    origin_m: NDArray[np.float64]
    cell_size_m: float
    node_counts: tuple[int, int]

    def __init__(
        self,
        origin_m: ArrayLike,
        cell_size_m: float,
        node_counts: tuple[int, int],
    ) -> None:
        origin = np.array(origin_m, dtype=np.float64, copy=True).reshape(-1)
        if origin.shape != (_DIMENSION,):
            raise SolverInputError(
                f"origin_m must be a 2-vector, got shape {np.shape(origin_m)!r}"
            )
        if not np.all(np.isfinite(origin)):
            raise SolverInputError(f"origin_m must be finite, got {origin!r}")
        size = float(cell_size_m)
        if not math.isfinite(size) or size <= 0.0:
            raise SolverInputError(f"cell_size_m must be positive, got {size!r}")
        counts = tuple(int(value) for value in node_counts)
        if len(counts) != _DIMENSION:
            raise SolverInputError(f"node_counts must be (nx, nz), got {node_counts!r}")
        if any(value < STENCIL_WIDTH for value in counts):
            raise SolverInputError(
                f"each axis needs at least {STENCIL_WIDTH} nodes to carry one "
                f"quadratic B-spline stencil, got {counts!r}"
            )
        origin.flags.writeable = False
        object.__setattr__(self, "origin_m", origin)
        object.__setattr__(self, "cell_size_m", size)
        object.__setattr__(self, "node_counts", counts)

    # -------------------------------------------------------------- shape

    @property
    def n_nodes(self) -> int:
        """Total node count."""
        return self.node_counts[0] * self.node_counts[1]

    @property
    def cell_volume_m2(self) -> float:
        """Area of one cell -- the plane-strain analogue of cell volume."""
        return self.cell_size_m * self.cell_size_m

    @property
    def extent_m(self) -> NDArray[np.float64]:
        """``(2,)`` physical span from node 0 to the last node on each axis."""
        return (np.array(self.node_counts, dtype=np.float64) - 1.0) * self.cell_size_m

    def node_positions_m(self) -> NDArray[np.float64]:
        """``(n_nodes, 2)`` node positions, in flat index order."""
        n_x, n_z = self.node_counts
        index_x = np.arange(n_x, dtype=np.float64)
        index_z = np.arange(n_z, dtype=np.float64)
        grid_x, grid_z = np.meshgrid(index_x, index_z, indexing="ij")
        offsets = np.stack([grid_x.ravel(), grid_z.ravel()], axis=1)
        return self.origin_m + offsets * self.cell_size_m

    @classmethod
    def covering(
        cls,
        lower_m: ArrayLike,
        upper_m: ArrayLike,
        cell_size_m: float,
        *,
        pad_cells: int = 3,
    ) -> PlaneStrainGrid:
        """Build the smallest grid covering a box, with a stencil margin.

        Args:
            lower_m: ``(2,)`` lower corner of the region to cover.
            upper_m: ``(2,)`` upper corner.
            cell_size_m: ``dx``.
            pad_cells: Cells of margin added on every side. Two is the
                minimum a quadratic stencil needs; the default of three
                leaves a cell of travel before a particle can reach the
                edge, which is what turns a stray particle into a raised
                error rather than a silently clipped one.

        Returns:
            The grid.

        Raises:
            SolverInputError: If the box is malformed or ``pad_cells`` is
                below the stencil minimum.
        """
        lower = np.array(lower_m, dtype=np.float64).reshape(-1)
        upper = np.array(upper_m, dtype=np.float64).reshape(-1)
        if lower.shape != (_DIMENSION,) or upper.shape != (_DIMENSION,):
            raise SolverInputError("lower_m and upper_m must both be 2-vectors")
        if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(upper)):
            raise SolverInputError("the covered box must be finite")
        if np.any(upper < lower):
            raise SolverInputError(
                f"upper_m {upper!r} must not be below lower_m {lower!r}"
            )
        if int(pad_cells) < STENCIL_WIDTH - 1:
            raise SolverInputError(
                f"pad_cells must be at least {STENCIL_WIDTH - 1} so a particle "
                f"at the edge still has a full stencil, got {pad_cells!r}"
            )
        size = float(cell_size_m)
        if not math.isfinite(size) or size <= 0.0:
            raise SolverInputError(f"cell_size_m must be positive, got {size!r}")
        pad = int(pad_cells)
        origin = lower - pad * size
        span_cells = np.ceil((upper - lower) / size).astype(np.int64) + 2 * pad
        counts = (int(span_cells[0]) + 1, int(span_cells[1]) + 1)
        return cls(origin, size, counts)

    # ------------------------------------------------------- interpolation

    def interpolate(self, positions_m: NDArray[np.float64]) -> GridInterpolation:
        """Form the ``3 x 3`` quadratic B-spline stencil for each particle.

        Args:
            positions_m: ``(n, 2)`` particle positions.

        Returns:
            The stencil.

        Raises:
            SolverInputError: If a particle is non-finite or has left the
                grid's interior. Leaving is raised rather than clamped:
                a clamped particle is a silent mass sink, and a solver
                that quietly loses sand is exactly the failure mode
                ADR-0033 was written about.
        """
        positions = np.asarray(positions_m, dtype=np.float64)
        if positions.ndim != _DIMENSION or positions.shape[1] != _DIMENSION:
            raise SolverInputError(
                f"positions_m must have shape (n, 2), got {positions.shape!r}"
            )
        if not np.all(np.isfinite(positions)):
            raise SolverInputError("particle positions contain non-finite values")

        local = (positions - self.origin_m) / self.cell_size_m
        base = np.floor(local - 0.5).astype(np.int64)
        counts = np.array(self.node_counts, dtype=np.int64)
        if np.any(base < 0) or np.any(base + STENCIL_WIDTH > counts):
            escaped = np.flatnonzero(
                np.any(base < 0, axis=1) | np.any(base + STENCIL_WIDTH > counts, axis=1)
            )
            worst = positions[escaped[0]]
            raise SolverInputError(
                f"{escaped.size} particle(s) left the grid interior; the first is "
                f"at {worst!r} m on a grid spanning "
                f"{self.origin_m!r} to {self.origin_m + self.extent_m!r} m. "
                "Enlarge the domain rather than clamping: a clamped particle is "
                "a silent mass sink."
            )

        fraction = local - base
        weights_1d = np.stack(
            [
                0.5 * (1.5 - fraction) ** 2,
                0.75 - (fraction - 1.0) ** 2,
                0.5 * (fraction - 0.5) ** 2,
            ],
            axis=0,
        )
        gradients_1d = (
            np.stack(
                [
                    fraction - 1.5,
                    -2.0 * (fraction - 1.0),
                    fraction - 0.5,
                ],
                axis=0,
            )
            / self.cell_size_m
        )

        weight_x = weights_1d[:, :, 0]
        weight_z = weights_1d[:, :, 1]
        gradient_x = gradients_1d[:, :, 0]
        gradient_z = gradients_1d[:, :, 1]
        n_particles = positions.shape[0]

        weight = np.einsum("an,bn->nab", weight_x, weight_z).reshape(
            n_particles, NODES_PER_PARTICLE
        )
        weight_gradient = np.stack(
            [
                np.einsum("an,bn->nab", gradient_x, weight_z).reshape(
                    n_particles, NODES_PER_PARTICLE
                ),
                np.einsum("an,bn->nab", weight_x, gradient_z).reshape(
                    n_particles, NODES_PER_PARTICLE
                ),
            ],
            axis=2,
        )

        stencil = np.arange(STENCIL_WIDTH, dtype=np.int64)
        index_x = base[:, 0][:, None] + stencil[None, :]
        index_z = base[:, 1][:, None] + stencil[None, :]
        node_index = (
            index_x[:, :, None] * self.node_counts[1] + index_z[:, None, :]
        ).reshape(n_particles, NODES_PER_PARTICLE)

        node_x = self.origin_m[0] + index_x * self.cell_size_m
        node_z = self.origin_m[1] + index_z * self.cell_size_m
        offset_x = np.repeat(node_x - positions[:, 0:1], STENCIL_WIDTH, axis=1)
        offset_z = np.tile(node_z - positions[:, 1:2], (1, STENCIL_WIDTH))
        offset = np.stack([offset_x, offset_z], axis=2)

        return GridInterpolation(
            node_index=node_index,
            weight=weight,
            weight_gradient=weight_gradient,
            offset_m=offset,
        )


def _scatter(
    grid: PlaneStrainGrid,
    node_index: NDArray[np.int64],
    values: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Sum ``values`` onto nodes by flat index."""
    totals = np.bincount(
        node_index.ravel(), weights=values.ravel(), minlength=grid.n_nodes
    )
    return totals.astype(np.float64, copy=False)


def _scatter_planar(
    grid: PlaneStrainGrid,
    node_index: NDArray[np.int64],
    weighted: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Scatter both in-plane components of ``weighted`` onto the nodes.

    Plane strain carries exactly two components, so momentum and internal
    force scatter through the same shape: ``(n_particles, n_stencil, 2)`` in,
    ``(n_nodes, 2)`` out. Keeping the pair in one place means the component
    order is fixed once rather than repeated per caller, where a transposed
    axis would be a silent sign error rather than a failure.
    """
    return np.stack(
        [
            _scatter(grid, node_index, weighted[:, :, 0]),
            _scatter(grid, node_index, weighted[:, :, 1]),
        ],
        axis=1,
    )


def scatter_mass(
    grid: PlaneStrainGrid,
    stencil: GridInterpolation,
    particle_mass_kg: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer particle mass to the nodes.

    Exact to round-off because the B-spline weights are a partition of
    unity: ``sum_i m_i = sum_p m_p sum_i w_ip = sum_p m_p``.

    Args:
        grid: The background grid.
        stencil: This step's interpolation.
        particle_mass_kg: ``(n,)`` particle masses, per unit width.

    Returns:
        ``(n_nodes,)`` nodal masses in kg/m.
    """
    return _scatter(
        grid, stencil.node_index, stencil.weight * particle_mass_kg[:, None]
    )


def scatter_momentum(
    grid: PlaneStrainGrid,
    stencil: GridInterpolation,
    particle_mass_kg: NDArray[np.float64],
    particle_velocity_m_s: NDArray[np.float64],
    affine_velocity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer particle momentum to the nodes, APIC-style.

    ``(m v)_i = sum_p w_ip m_p (v_p + C_p (x_i - x_p))``.  The affine term
    is what makes the transfer conserve angular momentum: the antisymmetric
    part of ``C_p`` carries the particle's local spin, which PIC discards
    and FLIP fails to reconstruct.

    Args:
        grid: The background grid.
        stencil: This step's interpolation.
        particle_mass_kg: ``(n,)`` particle masses.
        particle_velocity_m_s: ``(n, 2)`` particle velocities.
        affine_velocity: ``(n, 2, 2)`` APIC affine matrices ``C_p``.

    Returns:
        ``(n_nodes, 2)`` nodal momentum.
    """
    affine_term = np.einsum("nij,naj->nai", affine_velocity, stencil.offset_m)
    carried = particle_velocity_m_s[:, None, :] + affine_term
    weighted = stencil.weight[:, :, None] * particle_mass_kg[:, None, None] * carried
    return _scatter_planar(grid, stencil.node_index, weighted)


def scatter_stress_force(
    grid: PlaneStrainGrid,
    stencil: GridInterpolation,
    particle_volume_m2: NDArray[np.float64],
    cauchy_stress_pa: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Transfer the internal stress divergence to the nodes.

    ``f_i = -sum_p V_p sigma_p grad w_ip``, the standard MPM internal
    force in the current configuration.  Because ``sum_i grad w_ip = 0``
    for a B-spline stencil, ``sum_i f_i = 0`` to round-off for **any**
    stress field: internal forces cannot move the centre of mass, and the
    conservation suite tests exactly that.

    Args:
        grid: The background grid.
        stencil: This step's interpolation.
        particle_volume_m2: ``(n,)`` current particle areas.
        cauchy_stress_pa: ``(n, 2, 2)`` symmetric Cauchy stresses.

    Returns:
        ``(n_nodes, 2)`` nodal internal forces in N/m.
    """
    traction = np.einsum("nkl,nal->nak", cauchy_stress_pa, stencil.weight_gradient)
    weighted = -particle_volume_m2[:, None, None] * traction
    return _scatter_planar(grid, stencil.node_index, weighted)


def gather_velocity(
    stencil: GridInterpolation, node_velocity_m_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Interpolate the nodal velocity back to the particles.

    Args:
        stencil: This step's interpolation.
        node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities.

    Returns:
        ``(n, 2)`` particle velocities.
    """
    sampled = node_velocity_m_s[stencil.node_index]
    return np.einsum("na,nak->nk", stencil.weight, sampled)


def affine_from_grid_velocity(
    grid: PlaneStrainGrid,
    stencil: GridInterpolation,
    node_velocity_m_s: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Rebuild the APIC affine matrix ``C_p`` from the grid.

    ``C_p = D_p^{-1} sum_i w_ip v_i (x_i - x_p)^T`` with
    ``D_p = (dx^2 / 4) I`` for quadratic B-splines.

    Args:
        grid: The background grid.
        stencil: This step's interpolation.
        node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities.

    Returns:
        ``(n, 2, 2)`` affine velocity matrices.
    """
    sampled = node_velocity_m_s[stencil.node_index]
    weighted = stencil.weight[:, :, None] * sampled
    moment = np.einsum("nak,nal->nkl", weighted, stencil.offset_m)
    return moment * (_APIC_INERTIA_FACTOR / (grid.cell_size_m * grid.cell_size_m))


def cross_2d(
    left: NDArray[np.float64], right: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Scalar cross product ``a_x b_z - a_z b_x`` of stacked 2-vectors.

    In plane strain the only surviving component of a cross product is the
    out-of-plane one, so a torque or an angular momentum is a scalar here.

    Args:
        left: ``(..., 2)`` left operand.
        right: ``(..., 2)`` right operand.

    Returns:
        ``(...)`` the out-of-plane component.
    """
    return left[..., 0] * right[..., 1] - left[..., 1] * right[..., 0]


def apic_angular_momentum(
    grid: PlaneStrainGrid,
    positions_m: NDArray[np.float64],
    velocity_m_s: NDArray[np.float64],
    affine_velocity: NDArray[np.float64],
    mass_kg: NDArray[np.float64],
) -> float:
    """Total APIC angular momentum of a particle set, about the origin.

    The conserved quantity is **not** ``sum m (x x v)``.  Carrying the
    affine matrix means each particle also holds the angular momentum of
    its own local velocity field, and for quadratic B-splines
    ``D_p = (dx^2 / 4) I`` gives

        ``L = sum_p m_p [ x_p x v_p + (dx^2 / 4)(C_zx - C_xz) ]``

    which is exactly what a P2G/G2P round trip preserves (Jiang et al.
    2015, section 5.3).  Omitting the second term is the commonest way to
    "measure" APIC's angular-momentum conservation and conclude it does
    not have it.

    Args:
        grid: The background grid, for ``dx``.
        positions_m: ``(n, 2)`` particle positions.
        velocity_m_s: ``(n, 2)`` particle velocities.
        affine_velocity: ``(n, 2, 2)`` affine matrices ``C_p``.
        mass_kg: ``(n,)`` particle masses.

    Returns:
        The total angular momentum in kg m/s (per unit width).
    """
    orbital = cross_2d(positions_m, velocity_m_s)
    spin = (grid.cell_size_m * grid.cell_size_m / _APIC_INERTIA_FACTOR) * (
        affine_velocity[:, 1, 0] - affine_velocity[:, 0, 1]
    )
    return float((mass_kg * (orbital + spin)).sum())


def velocity_gradient(
    stencil: GridInterpolation, node_velocity_m_s: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Particle velocity gradient ``sum_i v_i (grad w_ip)^T``.

    Args:
        stencil: This step's interpolation.
        node_velocity_m_s: ``(n_nodes, 2)`` nodal velocities.

    Returns:
        ``(n, 2, 2)`` velocity gradients in 1/s, index ``[k, l] =
        dv_k / dx_l``.
    """
    sampled = node_velocity_m_s[stencil.node_index]
    return np.einsum("nak,nal->nkl", sampled, stencil.weight_gradient)
