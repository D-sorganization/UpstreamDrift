"""Several bodies contacting the sand within one step (#8733 §2).

Two things are pinned here, and they are different claims.

**The ordering is chosen, not inherited.**  The grid projection writes a
velocity-level constraint straight onto the node, so when two bodies
share a node the last projection wins.  ``contact_order`` fixes that
order from the bodies themselves -- slowest first, fastest last -- which
means (a) the fastest body's non-penetration is the one that holds
exactly, and (b) rearranging the caller's argument list cannot change the
answer.  The second half of that is only worth testing because the first
half proves the order matters at all, so both are tested.

**The momentum ledger stays exact with any number of bodies.**  Each
body's ledger is the nodal momentum change *at its own stage*, so the
stages telescope and the impulses sum to what the contact projections
actually moved.  The conservation case marches a real bed against two
bodies in a domain with no walls acting, where the only momentum sources
are gravity and the bodies, and closes the budget to round-off.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.body import RigidSection
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.contact import (
    apply_body_contacts,
    contact_order,
    ledger_from_impulses,
    push_out_bodies,
)
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.state import DomainWalls, WallCondition, settled_bed

pytestmark = pytest.mark.unit

_NODES = np.array([[0.0, 0.0]], dtype=np.float64)
_MASSES = np.array([1.0], dtype=np.float64)


def _slab_x(face_m: float, speed_m_s: float, *, friction: float = 1.0) -> RigidSection:
    """A slab filling ``x <= face_m``, driving along ``+x``.

    Wide enough that the only face near the origin is the vertical one, so
    the contact normal there is exactly ``(1, 0)`` and the arithmetic in
    these tests can be done by hand.
    """
    return RigidSection(
        [[-1.0, -1.0], [face_m, -1.0], [face_m, 1.0], [-1.0, 1.0]],
        velocity_m_s=(speed_m_s, 0.0),
        friction=friction,
    )


def _slab_z(face_m: float, speed_m_s: float, *, friction: float = 1.0) -> RigidSection:
    """A slab filling ``z <= face_m``, driving along ``+z``.

    Its contact normal at the origin is ``(0, 1)``, so a node it shares
    with :func:`_slab_x` is constrained in two different directions -- the
    configuration in which the projection order actually changes the
    answer.
    """
    return RigidSection(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, face_m], [-1.0, face_m]],
        velocity_m_s=(0.0, speed_m_s),
        friction=friction,
    )


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


class TestContactOrder:
    """Slowest first, fastest last, ties in the caller's order."""

    def test_the_fastest_body_goes_last(self) -> None:
        assert contact_order([_slab_x(0.02, 5.0), _slab_z(0.02, 1.0)]) == (1, 0)

    def test_an_already_ordered_sequence_is_left_alone(self) -> None:
        assert contact_order([_slab_x(0.02, 1.0), _slab_z(0.02, 5.0)]) == (0, 1)

    def test_ties_keep_the_callers_order(self) -> None:
        bodies = [_slab_x(0.02, 3.0), _slab_z(0.05, 3.0), _slab_x(0.08, 3.0)]
        assert contact_order(bodies) == (0, 1, 2)

    def test_rotation_counts_toward_the_speed(self) -> None:
        spinning = RigidSection(
            [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]],
            velocity_m_s=(0.0, 0.0),
            angular_velocity_rad_s=200.0,
        )
        assert contact_order([spinning, _slab_x(0.02, 1.0)]) == (1, 0)

    def test_an_empty_sequence_orders_to_nothing(self) -> None:
        assert contact_order([]) == ()

    def test_a_non_body_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="RigidSection"):
            contact_order([_slab_x(0.02, 1.0), "not a body"])  # type: ignore[list-item]


class TestOrderMatters:
    """The pin: the order is load-bearing, and it is the chosen one.

    A node inside a fast body driving along ``+x`` and a slow body driving
    along ``+z``, both sticking. Applying the fast one last leaves the node
    exactly on the fast body's velocity; applying it first leaves the node
    somewhere else. Nothing about the choice is testable unless the two
    answers genuinely differ, so that is the first case here.
    """

    @staticmethod
    def _pair() -> tuple[RigidSection, RigidSection]:
        return _slab_x(0.02, 5.0), _slab_z(0.02, 1.0)

    def test_projecting_the_fast_body_first_gives_a_different_answer(self) -> None:
        fast, slow = self._pair()
        velocity = np.zeros((1, 2), dtype=np.float64)
        after_fast, _ = fast.project_grid_velocity(
            _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        after_slow_last, _ = slow.project_grid_velocity(
            _NODES, after_fast, _MASSES, time_step_s=1e-5
        )
        assert after_slow_last[0] == pytest.approx([4.0, 1.0])

    def test_the_fastest_bodys_constraint_holds_exactly(self) -> None:
        fast, slow = self._pair()
        velocity = np.zeros((1, 2), dtype=np.float64)
        projected, _ = apply_body_contacts(
            [fast, slow], _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        assert projected[0] == pytest.approx([5.0, 0.0])
        relative = projected[0] - fast.velocity_at(_NODES)[0]
        assert float(relative @ np.array([1.0, 0.0])) == pytest.approx(0.0)

    def test_the_slower_bodys_constraint_may_be_left_violated(self) -> None:
        # Stated rather than hidden: this is what the choice costs. The sand
        # ends the step moving into the slow body's face, and the slow body
        # under-collects, rather than sand passing through the fast one.
        fast, slow = self._pair()
        velocity = np.zeros((1, 2), dtype=np.float64)
        projected, _ = apply_body_contacts(
            [fast, slow], _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        relative = projected[0] - slow.velocity_at(_NODES)[0]
        assert float(relative @ np.array([0.0, 1.0])) < 0.0

    def test_swapping_the_arguments_changes_nothing(self) -> None:
        fast, slow = self._pair()
        velocity = np.zeros((1, 2), dtype=np.float64)
        forward, forward_ledgers = apply_body_contacts(
            [fast, slow], _NODES, velocity.copy(), _MASSES, time_step_s=1e-5
        )
        backward, backward_ledgers = apply_body_contacts(
            [slow, fast], _NODES, velocity.copy(), _MASSES, time_step_s=1e-5
        )
        assert forward == pytest.approx(backward)
        assert forward_ledgers[0].impulse_n_s == pytest.approx(
            backward_ledgers[1].impulse_n_s
        )
        assert forward_ledgers[1].impulse_n_s == pytest.approx(
            backward_ledgers[0].impulse_n_s
        )

    def test_the_ledgers_come_back_in_the_callers_order(self) -> None:
        fast, slow = self._pair()
        velocity = np.zeros((1, 2), dtype=np.float64)
        _, ledgers = apply_body_contacts(
            [fast, slow], _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        assert ledgers[0].impulse_n_s[0] == pytest.approx([5.0, -1.0])
        assert ledgers[1].impulse_n_s[0] == pytest.approx([0.0, 1.0])

    def test_no_bodies_leaves_the_grid_alone(self) -> None:
        velocity = np.array([[3.0, -2.0]], dtype=np.float64)
        projected, ledgers = apply_body_contacts(
            [], _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        assert projected is velocity
        assert ledgers == ()


class TestStagesTelescope:
    """The per-body ledgers add up to what the projections moved."""

    def test_the_impulses_sum_to_the_nodal_momentum_change(self) -> None:
        bodies = [_slab_x(0.02, 5.0), _slab_z(0.02, 1.0)]
        velocity = np.zeros((1, 2), dtype=np.float64)
        projected, ledgers = apply_body_contacts(
            bodies, _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        moved = (_MASSES[:, None] * (projected - velocity)).sum(axis=0)
        summed = sum(ledger.impulse_n_s.sum(axis=0) for ledger in ledgers)
        assert summed == pytest.approx(moved, abs=1e-15)

    def test_the_sum_is_the_same_in_either_argument_order(self) -> None:
        fast = _slab_x(0.02, 5.0)
        slow = _slab_z(0.02, 1.0)
        velocity = np.zeros((1, 2), dtype=np.float64)
        totals = []
        for bodies in ([fast, slow], [slow, fast]):
            _, ledgers = apply_body_contacts(
                bodies, _NODES, velocity.copy(), _MASSES, time_step_s=1e-5
            )
            totals.append(sum(ledger.impulse_n_s.sum(axis=0) for ledger in ledgers))
        assert totals[0] == pytest.approx(totals[1])


class TestLedgerAssembly:
    """``BodyContact`` is the same number read from the two ends."""

    def test_force_is_minus_the_impulse_over_the_step(self) -> None:
        bodies = [_slab_x(0.02, 5.0), _slab_z(0.02, 1.0)]
        velocity = np.zeros((1, 2), dtype=np.float64)
        _, impulses = apply_body_contacts(
            bodies, _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        for ledger in ledger_from_impulses(bodies, impulses, (0, 0), 1e-5):
            assert ledger.force_n_per_m == pytest.approx(
                -ledger.impulse_on_sand_n_s / 1e-5
            )

    def test_misaligned_sequences_are_refused(self) -> None:
        bodies = [_slab_x(0.02, 5.0)]
        velocity = np.zeros((1, 2), dtype=np.float64)
        _, impulses = apply_body_contacts(
            bodies, _NODES, velocity, _MASSES, time_step_s=1e-5
        )
        with pytest.raises(SolverInputError, match="align"):
            ledger_from_impulses(bodies, impulses, (0, 0), 1e-5)


class TestPushOutOrder:
    """The backstop follows the same order as the contact projection."""

    def test_the_last_repair_stands(self) -> None:
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((1, 2), dtype=np.float64)
        # The fast body is the deeper one, so its repair is both last and
        # the one that has anything left to do.
        bodies = [_slab_x(0.05, 5.0), _slab_x(0.02, 1.0)]
        repaired, _, counts = push_out_bodies(bodies, positions, velocities)
        assert repaired[0, 0] == pytest.approx(0.05)
        assert counts == (1, 1)

    def test_a_particle_the_last_body_does_not_hold_is_left_where_it_was_put(
        self,
    ) -> None:
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((1, 2), dtype=np.float64)
        bodies = [_slab_x(0.02, 5.0), _slab_x(0.05, 1.0)]
        repaired, _, counts = push_out_bodies(bodies, positions, velocities)
        assert repaired[0, 0] == pytest.approx(0.05)
        assert counts == (0, 1)

    def test_no_bodies_leaves_the_particles_alone(self) -> None:
        positions = np.array([[0.0, 0.0]], dtype=np.float64)
        velocities = np.zeros((1, 2), dtype=np.float64)
        repaired, moved, counts = push_out_bodies([], positions, velocities)
        assert repaired is positions
        assert moved is velocities
        assert counts == ()


class TestMultiBodyMarch:
    """A real bed marched against two bodies at once."""

    @staticmethod
    def _open_solver(material: SandContinuum) -> PlaneStrainMPMSolver:
        """A solver whose walls do nothing, so the budget has no wall term."""
        return PlaneStrainMPMSolver(
            material=material,
            cell_size_m=0.004,
            effective_width_m=0.03,
            bed_depth_m=0.02,
            walls=DomainWalls(
                lower_x=WallCondition.FREE,
                upper_x=WallCondition.FREE,
                lower_z=WallCondition.FREE,
                upper_z=WallCondition.FREE,
            ),
        )

    @staticmethod
    def _bed(solver: PlaneStrainMPMSolver):
        from bunkershot3d.solvers.mpm.grid import PlaneStrainGrid

        bounds = (-0.06, 0.06)
        grid = PlaneStrainGrid.covering(
            (bounds[0] - 0.01, -0.05), (bounds[1] + 0.01, 0.06), solver.cell_size_m
        )
        particles = settled_bed(
            solver.material,
            x_bounds_m=bounds,
            free_surface_height_m=0.0,
            depth_m=solver.bed_depth_m,
            cell_size_m=solver.cell_size_m,
            particles_per_cell_axis=solver.particles_per_cell_axis,
            gravity_m_s2=solver.gravity_m_s2,
        )
        return grid, particles, bounds

    def test_the_ledger_closes_against_the_sands_momentum_change(
        self, material: SandContinuum
    ) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        club = RigidSection(
            [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
            velocity_m_s=(2.0, -1.0),
            friction=0.3,
        )
        ball = RigidSection(
            [[0.02, 0.0], [0.036, 0.0], [0.036, 0.016], [0.02, 0.016]],
            velocity_m_s=(0.0, -0.2),
            friction=0.3,
        )
        total_mass = particles.total_mass_kg
        start = particles.linear_momentum_kg_m_s().copy()

        step_s = 2.0e-6
        run = solver.march_bodies(
            particles,
            (club, ball),
            grid,
            n_steps=4,
            time_step_s=step_s,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )

        # A step whose pushout fired has moved particles geometrically, and
        # a geometric repair is outside the momentum budget by construction.
        assert run.max_pushed_out() == 0
        # Both bodies must actually have delivered something, or the
        # identity below would close for the trivial reason.
        for index in (0, 1):
            delivered = sum(
                step.body_contacts[index].impulse_on_sand_n_s for step in run.steps
            )
            assert float(np.hypot(delivered[0], delivered[1])) > 0.0
        contact = sum(step.total_impulse_on_sand_n_s() for step in run.steps)
        gravity = np.array(
            [0.0, -total_mass * solver.gravity_m_s2 * step_s * run.n_steps]
        )
        change = run.steps[-1].linear_momentum_kg_m_s - start
        assert change == pytest.approx(contact + gravity, abs=1e-12)

    def test_every_body_gets_its_own_ledger(self, material: SandContinuum) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        club = RigidSection(
            [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
            velocity_m_s=(2.0, -1.0),
        )
        ball = RigidSection(
            [[0.02, 0.0], [0.036, 0.0], [0.036, 0.016], [0.02, 0.016]],
            velocity_m_s=(0.0, -0.2),
        )
        run = solver.march_bodies(
            particles,
            (club, ball),
            grid,
            n_steps=2,
            time_step_s=2.0e-6,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        assert all(step.n_bodies == 2 for step in run.steps)
        assert run.section is not None
        assert len(run.extra_sections) == 1

    def test_the_primary_body_is_the_one_the_scalars_report(
        self, material: SandContinuum
    ) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        club = RigidSection(
            [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
            velocity_m_s=(2.0, -1.0),
        )
        ball = RigidSection(
            [[0.02, 0.0], [0.036, 0.0], [0.036, 0.016], [0.02, 0.016]],
            velocity_m_s=(0.0, -0.2),
        )
        run = solver.march_bodies(
            particles,
            (club, ball),
            grid,
            n_steps=2,
            time_step_s=2.0e-6,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        for step in run.steps:
            assert step.contact_force_n_per_m == pytest.approx(
                step.body_contacts[0].force_n_per_m
            )
            assert step.n_contacts == step.body_contacts[0].n_contacts

    def test_the_single_body_march_still_works(self, material: SandContinuum) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        club = RigidSection(
            [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
            velocity_m_s=(2.0, -1.0),
        )
        run = solver.march(
            particles,
            club,
            grid,
            n_steps=2,
            time_step_s=2.0e-6,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        assert run.n_steps == 2
        assert run.extra_sections == ()
        assert all(step.n_bodies == 1 for step in run.steps)

    def test_a_bed_with_no_intruder_reports_no_bodies(
        self, material: SandContinuum
    ) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        run = solver.march(
            particles,
            None,
            grid,
            n_steps=2,
            time_step_s=2.0e-6,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        assert run.section is None
        assert all(step.n_bodies == 0 for step in run.steps)
        assert all(
            step.total_impulse_on_sand_n_s() == pytest.approx([0.0, 0.0])
            for step in run.steps
        )

    def test_the_courant_check_sees_a_fast_secondary_body(
        self, material: SandContinuum
    ) -> None:
        solver = self._open_solver(material)
        grid, particles, bounds = self._bed(solver)
        slow = RigidSection(
            [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
            velocity_m_s=(1.0, 0.0),
        )
        rocket = RigidSection(
            [[0.02, 0.0], [0.036, 0.0], [0.036, 0.016], [0.02, 0.016]],
            velocity_m_s=(5000.0, 0.0),
        )
        with pytest.raises(SolverInputError, match="body 1"):
            solver.march_bodies(
                particles,
                (slow, rocket),
                grid,
                n_steps=1,
                time_step_s=1.0e-5,
                free_surface_height_m=0.0,
                bed_x_bounds_m=bounds,
            )
