"""The F1 grid, its B-spline basis and the APIC transfers.

Everything here is an **identity of the scheme**, not an approximation,
so every tolerance is a round-off tolerance and none of them was widened
to make a test pass.  Three identities carry the conservation guarantees
the solver later relies on:

* ``sum_i w_ip = 1``          -> P2G conserves mass exactly
* ``sum_i grad w_ip = 0``     -> internal forces sum to zero exactly
* the APIC round trip         -> linear *and* angular momentum survive

The last one is the load-bearing justification for choosing APIC over PIC
or FLIP, so it is tested directly rather than cited.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.grid import (
    NODES_PER_PARTICLE,
    PlaneStrainGrid,
    affine_from_grid_velocity,
    apic_angular_momentum,
    cross_2d,
    gather_velocity,
    scatter_mass,
    scatter_momentum,
    scatter_stress_force,
    velocity_gradient,
)

pytestmark = pytest.mark.unit

_ROUND_OFF = 1e-12


@pytest.fixture
def grid() -> PlaneStrainGrid:
    return PlaneStrainGrid(
        origin_m=(-0.02, -0.03), cell_size_m=0.002, node_counts=(24, 30)
    )


@pytest.fixture
def particles(grid: PlaneStrainGrid) -> np.ndarray:
    rng = np.random.default_rng(20260816)
    lower = grid.origin_m + 2.0 * grid.cell_size_m
    upper = grid.origin_m + grid.extent_m - 2.0 * grid.cell_size_m
    return lower + rng.random((300, 2)) * (upper - lower)


class TestGridConstruction:
    """A grid that cannot carry a stencil is refused, not silently used."""

    def test_node_positions_are_uniform(self, grid: PlaneStrainGrid) -> None:
        positions = grid.node_positions_m()
        assert positions.shape == (grid.n_nodes, 2)
        np.testing.assert_allclose(positions[0], grid.origin_m)
        np.testing.assert_allclose(
            positions[-1], grid.origin_m + grid.extent_m, atol=1e-15
        )

    def test_refuses_a_grid_too_small_for_a_stencil(self) -> None:
        with pytest.raises(SolverInputError, match="at least 3 nodes"):
            PlaneStrainGrid((0.0, 0.0), 0.01, (2, 8))

    @pytest.mark.parametrize("size", [0.0, -0.01, float("inf")])
    def test_refuses_an_unusable_cell_size(self, size: float) -> None:
        with pytest.raises(SolverInputError, match="cell_size_m"):
            PlaneStrainGrid((0.0, 0.0), size, (8, 8))

    def test_covering_leaves_a_stencil_margin(self) -> None:
        covering = PlaneStrainGrid.covering((0.0, 0.0), (0.05, 0.03), 0.005)
        assert np.all(covering.origin_m <= -0.005)
        upper = covering.origin_m + covering.extent_m
        assert upper[0] >= 0.05 and upper[1] >= 0.03

    def test_covering_refuses_too_thin_a_margin(self) -> None:
        with pytest.raises(SolverInputError, match="pad_cells"):
            PlaneStrainGrid.covering((0.0, 0.0), (0.05, 0.03), 0.005, pad_cells=1)


class TestBasisIdentities:
    """The two sums the conservation proofs stand on."""

    def test_weights_are_a_partition_of_unity(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        stencil = grid.interpolate(particles)
        assert stencil.weight.shape == (particles.shape[0], NODES_PER_PARTICLE)
        np.testing.assert_allclose(
            stencil.weight.sum(axis=1), 1.0, rtol=0.0, atol=_ROUND_OFF
        )

    def test_weights_are_non_negative(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        stencil = grid.interpolate(particles)
        assert float(stencil.weight.min()) >= 0.0

    def test_weight_gradients_sum_to_zero(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        stencil = grid.interpolate(particles)
        residual = np.abs(stencil.weight_gradient.sum(axis=1)).max()
        assert residual <= _ROUND_OFF / grid.cell_size_m

    def test_basis_reproduces_a_linear_field(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        """``sum_i w_ip x_i = x_p``: the reason APIC's affine term vanishes."""
        stencil = grid.interpolate(particles)
        nodes = grid.node_positions_m()[stencil.node_index]
        interpolated = np.einsum("na,nak->nk", stencil.weight, nodes)
        np.testing.assert_allclose(interpolated, particles, rtol=0.0, atol=1e-14)

    def test_gradient_matches_a_finite_difference(self, grid: PlaneStrainGrid) -> None:
        point = np.array([[0.0031, -0.0042]])
        step = 1e-7
        analytic = grid.interpolate(point).weight_gradient[0]
        for axis in (0, 1):
            shift = np.zeros((1, 2))
            shift[0, axis] = step
            forward = grid.interpolate(point + shift).weight
            backward = grid.interpolate(point - shift).weight
            numeric = (forward - backward)[0] / (2.0 * step)
            np.testing.assert_allclose(
                analytic[:, axis], numeric, rtol=1e-6, atol=1e-6 / grid.cell_size_m
            )

    def test_a_particle_off_the_grid_raises(self, grid: PlaneStrainGrid) -> None:
        outside = grid.origin_m + grid.extent_m + grid.cell_size_m
        with pytest.raises(SolverInputError, match="left the grid interior"):
            grid.interpolate(outside[None, :])

    def test_a_non_finite_particle_raises(self, grid: PlaneStrainGrid) -> None:
        with pytest.raises(SolverInputError, match="non-finite"):
            grid.interpolate(np.array([[np.nan, 0.0]]))


class TestTransfers:
    """P2G and G2P, and what each of them conserves."""

    def test_mass_transfer_is_exact(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        mass = np.full(particles.shape[0], 3.7e-4)
        stencil = grid.interpolate(particles)
        nodal = scatter_mass(grid, stencil, mass)
        assert nodal.shape == (grid.n_nodes,)
        assert abs(nodal.sum() - mass.sum()) <= _ROUND_OFF * mass.sum()

    def test_linear_momentum_transfer_is_exact(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        rng = np.random.default_rng(3)
        count = particles.shape[0]
        mass = rng.uniform(1e-4, 5e-4, size=count)
        velocity = rng.normal(scale=2.0, size=(count, 2))
        affine = rng.normal(scale=50.0, size=(count, 2, 2))
        stencil = grid.interpolate(particles)
        nodal = scatter_momentum(grid, stencil, mass, velocity, affine)
        expected = (mass[:, None] * velocity).sum(axis=0)
        scale = float(np.abs(expected).max())
        np.testing.assert_allclose(
            nodal.sum(axis=0), expected, rtol=0.0, atol=_ROUND_OFF * scale
        )

    def test_internal_forces_sum_to_zero(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        """Any stress field at all: this is a basis identity, not physics."""
        rng = np.random.default_rng(5)
        count = particles.shape[0]
        volume = rng.uniform(1e-7, 2e-7, size=count)
        stress = rng.normal(scale=1e5, size=(count, 2, 2))
        stress = 0.5 * (stress + np.transpose(stress, (0, 2, 1)))
        stencil = grid.interpolate(particles)
        force = scatter_stress_force(grid, stencil, volume, stress)
        scale = float(np.abs(force).max())
        np.testing.assert_allclose(force.sum(axis=0), 0.0, rtol=0.0, atol=1e-10 * scale)

    def test_gather_reproduces_a_uniform_velocity(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        node_velocity = np.tile(np.array([1.5, -0.25]), (grid.n_nodes, 1))
        stencil = grid.interpolate(particles)
        gathered = gather_velocity(stencil, node_velocity)
        expected = np.broadcast_to(np.array([1.5, -0.25]), gathered.shape)
        np.testing.assert_allclose(gathered, expected, rtol=0.0, atol=1e-14)

    def test_velocity_gradient_of_a_uniform_field_is_zero(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        node_velocity = np.tile(np.array([1.5, -0.25]), (grid.n_nodes, 1))
        stencil = grid.interpolate(particles)
        gradient = velocity_gradient(stencil, node_velocity)
        assert float(np.abs(gradient).max()) <= 1e-9

    def test_velocity_gradient_recovers_a_linear_field(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        target = np.array([[2.0, -0.5], [0.75, 1.25]])
        nodes = grid.node_positions_m()
        node_velocity = nodes @ target.T
        stencil = grid.interpolate(particles)
        gradient = velocity_gradient(stencil, node_velocity)
        np.testing.assert_allclose(
            gradient, np.broadcast_to(target, gradient.shape), rtol=1e-9, atol=1e-9
        )


class TestApicRoundTrip:
    """Why APIC and not PIC or FLIP: both momenta survive the round trip."""

    def _round_trip(self, grid: PlaneStrainGrid, particles: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        count = particles.shape[0]
        mass = rng.uniform(1e-4, 5e-4, size=count)
        velocity = rng.normal(scale=2.0, size=(count, 2))
        affine = rng.normal(scale=80.0, size=(count, 2, 2))
        stencil = grid.interpolate(particles)
        nodal_mass = scatter_mass(grid, stencil, mass)
        nodal_momentum = scatter_momentum(grid, stencil, mass, velocity, affine)
        live = nodal_mass > 0.0
        node_velocity = np.zeros_like(nodal_momentum)
        node_velocity[live] = nodal_momentum[live] / nodal_mass[live, None]
        new_velocity = gather_velocity(stencil, node_velocity)
        new_affine = affine_from_grid_velocity(grid, stencil, node_velocity)
        return mass, velocity, affine, new_velocity, new_affine

    def test_linear_momentum_survives(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        mass, velocity, _, new_velocity, _ = self._round_trip(grid, particles, 17)
        before = (mass[:, None] * velocity).sum(axis=0)
        after = (mass[:, None] * new_velocity).sum(axis=0)
        scale = float(np.abs(before).max())
        np.testing.assert_allclose(after, before, rtol=0.0, atol=_ROUND_OFF * scale)

    def test_angular_momentum_survives(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        mass, velocity, affine, new_velocity, new_affine = self._round_trip(
            grid, particles, 19
        )
        before = apic_angular_momentum(grid, particles, velocity, affine, mass)
        after = apic_angular_momentum(grid, particles, new_velocity, new_affine, mass)
        scale = abs(before) + float(
            (mass * np.abs(cross_2d(particles, velocity))).sum()
        )
        assert abs(after - before) <= _ROUND_OFF * scale

    def test_pic_would_lose_the_angular_momentum_apic_keeps(
        self, grid: PlaneStrainGrid, particles: np.ndarray
    ) -> None:
        """The scheme choice is falsifiable, not a matter of taste.

        Dropping the affine term is exactly PIC.  If discarding it made no
        difference, APIC would be pointless machinery and this test would
        be the one to say so.
        """
        mass, velocity, affine, new_velocity, new_affine = self._round_trip(
            grid, particles, 23
        )
        zero_affine = np.zeros_like(new_affine)
        before = apic_angular_momentum(grid, particles, velocity, affine, mass)
        apic = apic_angular_momentum(grid, particles, new_velocity, new_affine, mass)
        pic = apic_angular_momentum(grid, particles, new_velocity, zero_affine, mass)
        assert abs(apic - before) < abs(pic - before)


class TestCrossProduct:
    """Plane strain leaves one component of a cross product alive."""

    def test_matches_the_embedded_three_dimensional_cross(self) -> None:
        """NumPy 2 deprecated the 2-D cross, so the check embeds in 3-D."""
        rng = np.random.default_rng(29)
        left = rng.normal(size=(50, 2))
        right = rng.normal(size=(50, 2))
        zeros = np.zeros(50)
        left_3d = np.stack([left[:, 0], left[:, 1], zeros], axis=1)
        right_3d = np.stack([right[:, 0], right[:, 1], zeros], axis=1)
        expected = np.cross(left_3d, right_3d)[:, 2]
        np.testing.assert_allclose(cross_2d(left, right), expected, atol=1e-15)
