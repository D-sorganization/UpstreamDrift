"""What actually reaches the ball, resolved over its surface (#8712).

The claim under test is not "a number comes out". It is that the number
that comes out is *about the sand arriving at the ball* and carries every
qualification #8733 attached to it:

* the traction is **per unit out-of-plane width**, on an **infinite
  cylinder rather than a sphere**, and the API says so in its names
  rather than in a caption;
* the below-equator / face-side split and the sector resolution are
  **in-plane and qualitative**, and both refuse any heel-toe or lateral
  reading;
* **ball launch is still F0's**, so nothing here returns a launch speed,
  angle or spin;
* the comparison against the club is a **ratio and a timing**, because
  absolute club force is refused at this tier and multiplying a
  per-unit-width flux on an infinite cylinder by a width would invent the
  third dimension the model does not have.

The last group is the point of the whole exercise: a number about the
ball is not more trustworthy than the tier that produced it, so every
result carries ``BEYOND_VALIDATION``, the 1.44 m/s published-speed
ceiling and NASA-STD-7009B validation 0 of 4.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers import IntrusionState, RefusalPolicy, SurfaceElements
from bunkershot3d.solvers.envelope import MAX_VALIDATED_SPEED_M_S, EnvelopeStatus
from bunkershot3d.solvers.exceptions import OutOfEnvelopeError, SolverInputError
from bunkershot3d.solvers.mpm.ball import BallSection
from bunkershot3d.solvers.mpm.ballreach import (
    DEFAULT_BALL_SECTORS,
    MIN_BALL_SECTORS,
    BallReachHistory,
    BallSurfaceSectors,
    SandVersusClub,
    ball_reach_history,
    compare_sand_and_club,
    resolve_ball_traction,
    resolve_sectors,
)
from bunkershot3d.solvers.mpm.body import ContactImpulse, RigidSection
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.grid import PlaneStrainGrid
from bunkershot3d.solvers.mpm.solver import MPMRun, PlaneStrainMPMSolver
from bunkershot3d.solvers.mpm.state import DomainWalls, WallCondition, settled_bed
from bunkershot3d.solvers.mpm.wholeshot import F1ShotSettings, simulate_f1_shot
from bunkershot3d.solvers.protocol import FidelityTier

pytestmark = pytest.mark.unit

_BALL_RADIUS_M = 0.008
_CENTRE_M = (0.0, 0.0)


def _impulse(
    positions: list[list[float]], vectors: list[list[float]]
) -> ContactImpulse:
    """A hand-built ledger: what the *body* applied to the sand.

    The sign matters and is the easiest thing to get backwards, so the
    tests below build the body-on-sand impulse the solver produces and
    check that the module reports the *sand on ball* reaction.
    """
    return ContactImpulse(
        node_index=np.arange(len(positions), dtype=np.int64),
        impulse_n_s=np.asarray(vectors, dtype=np.float64),
        position_m=np.asarray(positions, dtype=np.float64),
        stress_force_n=np.zeros(2, dtype=np.float64),
        n_swept=0,
    )


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


@pytest.fixture(scope="module")
def open_solver(material: SandContinuum) -> PlaneStrainMPMSolver:
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


def _bed(
    solver: PlaneStrainMPMSolver,
) -> tuple[PlaneStrainGrid, object, tuple[float, float]]:
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


def _club() -> RigidSection:
    """A blade driving along ``+x`` and down, to the left of the ball."""
    return RigidSection(
        [[-0.05, 0.0], [-0.03, 0.0], [-0.03, 0.016], [-0.05, 0.016]],
        velocity_m_s=(2.0, -1.0),
        friction=0.3,
    )


def _ball() -> BallSection:
    """A ball sitting on the sand to the right of the club, at rest.

    Tangent rather than plugged, because a ball that starts *inside* the
    bed fires the particle pushout backstop on the first step, and a
    geometric repair is outside the momentum budget by construction --
    which would make the conservation case below untestable for a reason
    that has nothing to do with the ball.
    """
    return BallSection.resting_on(
        x_m=0.026, radius_m=_BALL_RADIUS_M, n_facets=12, friction=0.3
    )


@pytest.fixture(scope="module")
def run(open_solver: PlaneStrainMPMSolver) -> MPMRun:
    """Four steps of a club and a ball sharing one bed."""
    grid, particles, bounds = _bed(open_solver)
    return open_solver.march_bodies(
        particles,
        (_club(), _ball().section),
        grid,
        n_steps=4,
        time_step_s=2.0e-6,
        free_surface_height_m=0.0,
        bed_x_bounds_m=bounds,
    )


@pytest.fixture(scope="module")
def history(run: MPMRun, open_solver: PlaneStrainMPMSolver) -> BallReachHistory:
    return ball_reach_history(
        run, _ball(), verdict=_verdict(open_solver), approach_direction=(1.0, -0.5)
    )


def _verdict(solver: PlaneStrainMPMSolver):
    from bunkershot3d.solvers.mpm.envelope import evaluate_f1_envelope

    return evaluate_f1_envelope(
        speed_m_s=2.0,
        feature_lengths_m={"ball_diameter": 2.0 * _BALL_RADIUS_M},
        grain_diameter_m=solver.material.grain_diameter_m,
        cell_size_m=solver.cell_size_m,
        effective_width_m=solver.effective_width_m,
    )


class TestSectorResolution:
    """The impulse is resolved over the ball's in-plane surface."""

    def test_the_sectors_partition_the_total_exactly(self) -> None:
        nodes = _impulse(
            [[0.008, 0.0], [0.0, -0.008], [-0.006, 0.006]],
            [[-1.0, 0.5], [0.0, -2.0], [3.0, 1.0]],
        )
        sectors = resolve_sectors(
            nodes, centre_m=_CENTRE_M, approach_direction=(1.0, 0.0)
        )
        # The sand's reaction is minus what the body put into the sand.
        expected = -np.asarray(nodes.impulse_n_s).sum(axis=0)
        assert sectors.impulse_n_s_per_m.sum(axis=0) == pytest.approx(expected)
        assert sectors.total_n_s_per_m == pytest.approx(expected)
        assert int(sectors.n_contacts.sum()) == 3

    def test_a_contact_below_the_centre_lands_in_a_lower_sector(self) -> None:
        nodes = _impulse([[0.0, -0.008]], [[0.0, -1.0]])
        sectors = resolve_sectors(
            nodes, centre_m=_CENTRE_M, approach_direction=(1.0, 0.0), n_sectors=4
        )
        # Sector 3 spans [270, 360) degrees; straight down is its lower edge.
        assert int(np.argmax(sectors.n_contacts)) == 3
        assert sectors.edges_rad[3] == pytest.approx(1.5 * math.pi)

    def test_the_equator_is_always_a_sector_boundary(self) -> None:
        sectors = resolve_sectors(
            _impulse([[0.0, -0.008]], [[0.0, -1.0]]),
            centre_m=_CENTRE_M,
            approach_direction=(1.0, 0.0),
            n_sectors=DEFAULT_BALL_SECTORS,
        )
        assert bool(np.isclose(sectors.edges_rad, math.pi).any())

    def test_a_purely_radial_push_has_no_tangential_part(self) -> None:
        # The body pushes the sand outward at the bottom, so the sand
        # pushes the ball straight up: purely compressive, no shear.
        nodes = _impulse([[0.0, -0.008]], [[0.0, -1.0]])
        sectors = resolve_sectors(
            nodes, centre_m=_CENTRE_M, approach_direction=(1.0, 0.0), n_sectors=4
        )
        assert sectors.radial_n_s_per_m[3] == pytest.approx(1.0)
        assert sectors.tangential_n_s_per_m[3] == pytest.approx(0.0, abs=1e-12)

    def test_a_purely_tangential_drag_has_no_radial_part(self) -> None:
        nodes = _impulse([[0.0, -0.008]], [[-1.0, 0.0]])
        sectors = resolve_sectors(
            nodes, centre_m=_CENTRE_M, approach_direction=(1.0, 0.0), n_sectors=4
        )
        assert sectors.radial_n_s_per_m[3] == pytest.approx(0.0, abs=1e-12)
        assert abs(sectors.tangential_n_s_per_m[3]) == pytest.approx(1.0)

    def test_radial_and_tangential_recover_the_sector_magnitude(self) -> None:
        nodes = _impulse([[0.006, -0.005]], [[-1.3, 0.7]])
        sectors = resolve_sectors(
            nodes, centre_m=_CENTRE_M, approach_direction=(1.0, 0.0), n_sectors=4
        )
        recovered = np.hypot(sectors.radial_n_s_per_m, sectors.tangential_n_s_per_m)
        assert recovered == pytest.approx(sectors.magnitude_n_s_per_m)

    def test_an_empty_ledger_resolves_to_zero_everywhere(self) -> None:
        sectors = resolve_sectors(
            _impulse([], []), centre_m=_CENTRE_M, approach_direction=(1.0, 0.0)
        )
        assert sectors.n_sectors == DEFAULT_BALL_SECTORS
        assert int(sectors.n_contacts.sum()) == 0
        assert sectors.total_n_s_per_m == pytest.approx([0.0, 0.0])
        assert sectors.fractions == pytest.approx(np.zeros(DEFAULT_BALL_SECTORS))

    def test_an_odd_sector_count_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="even"):
            resolve_sectors(
                _impulse([], []),
                centre_m=_CENTRE_M,
                approach_direction=(1.0, 0.0),
                n_sectors=7,
            )

    def test_too_few_sectors_are_refused(self) -> None:
        with pytest.raises(SolverInputError, match=str(MIN_BALL_SECTORS)):
            resolve_sectors(
                _impulse([], []),
                centre_m=_CENTRE_M,
                approach_direction=(1.0, 0.0),
                n_sectors=2,
            )

    def test_an_approach_direction_of_no_length_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="no length"):
            resolve_sectors(
                _impulse([], []), centre_m=_CENTRE_M, approach_direction=(0.0, 0.0)
            )


class TestSectorsRefuseTheThirdDimension:
    """A direction plane strain does not have is refused, not approximated."""

    @staticmethod
    def _sectors() -> BallSurfaceSectors:
        return resolve_sectors(
            _impulse([[0.0, -0.008]], [[0.0, -1.0]]),
            centre_m=_CENTRE_M,
            approach_direction=(1.0, 0.0),
        )

    def test_the_resolution_flags_itself_qualitative(self) -> None:
        assert self._sectors().is_qualitative is True

    def test_heel_toe_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            self._sectors().heel_toe_fraction()

    def test_lateral_distribution_raises(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            self._sectors().lateral_distribution()

    def test_an_absolute_force_raises_because_the_ball_has_no_width(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            self._sectors().total_force_n()

    def test_the_summary_carries_the_cylinder_note(self) -> None:
        summary = self._sectors().summary()
        assert "infinite cylinder" in summary
        assert "per unit width" in summary or "per metre of width" in summary


class TestOneStepSample:
    """One step's traction on the ball, from that step's own ledger."""

    def test_the_traction_is_the_sand_pushing_the_ball(self, run: MPMRun) -> None:
        contact = run.steps[0].body_contacts[1]
        sample = resolve_ball_traction(
            contact,
            _ball(),
            time_s=run.steps[0].time_s,
            time_step_s=run.time_step_s,
            approach_direction=(1.0, -0.5),
        )
        # force_n_per_m on the ledger is already the force on the body.
        assert sample.traction_n_per_m == pytest.approx(contact.force_n_per_m)
        assert sample.impulse_n_s_per_m == pytest.approx(
            contact.force_n_per_m * run.time_step_s
        )

    def test_the_split_and_the_sectors_describe_the_same_impulse(
        self, run: MPMRun
    ) -> None:
        contact = run.steps[-1].body_contacts[1]
        sample = resolve_ball_traction(
            contact,
            _ball(),
            time_s=run.steps[-1].time_s,
            time_step_s=run.time_step_s,
            approach_direction=(1.0, -0.5),
        )
        assert sample.n_contacts == contact.n_contacts
        # The split is signed the solver's way (body on sand); the sectors
        # report the reaction, so the two differ by exactly a sign.
        assert sample.sectors.total_n_s_per_m == pytest.approx(
            -np.asarray(sample.split.total_n_s)
        )


class TestReachHistory:
    """When the sand arrives, how hard, and how much in total."""

    def test_the_history_has_one_sample_per_step(
        self, history: BallReachHistory, run: MPMRun
    ) -> None:
        assert len(history.samples) == run.n_steps
        assert history.time_history_s() == pytest.approx(run.time_history_s())

    def test_the_sand_reaches_the_ball_and_the_arrival_is_timed(
        self, history: BallReachHistory
    ) -> None:
        arrival = history.first_arrival_s
        assert arrival is not None
        assert arrival >= 0.0

    def test_a_ball_the_sand_never_reaches_reports_no_arrival(
        self, open_solver: PlaneStrainMPMSolver
    ) -> None:
        grid, particles, bounds = _bed(open_solver)
        airborne = BallSection.at((0.026, 0.05), radius_m=_BALL_RADIUS_M, n_facets=12)
        aloft = open_solver.march_bodies(
            particles,
            (_club(), airborne.section),
            grid,
            n_steps=2,
            time_step_s=2.0e-6,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        untouched = ball_reach_history(
            aloft,
            airborne,
            verdict=_verdict(open_solver),
            approach_direction=(1.0, 0.0),
        )
        assert untouched.first_arrival_s is None
        assert untouched.total_impulse_magnitude_n_s_per_m == pytest.approx(0.0)

    def test_the_peak_is_reported_with_its_time(
        self, history: BallReachHistory
    ) -> None:
        magnitudes = np.hypot(
            history.traction_history_n_per_m()[:, 0],
            history.traction_history_n_per_m()[:, 1],
        )
        assert history.peak_traction_n_per_m == pytest.approx(magnitudes.max())
        assert history.peak_traction_time_s in set(history.time_history_s().tolist())

    def test_the_total_impulse_is_the_ledger_summed(
        self, history: BallReachHistory, run: MPMRun
    ) -> None:
        from_ledger = -sum(
            step.body_contacts[1].impulse_on_sand_n_s for step in run.steps
        )
        assert history.total_impulse_n_s_per_m == pytest.approx(from_ledger)

    def test_the_sector_totals_add_to_the_total_impulse(
        self, history: BallReachHistory
    ) -> None:
        assert history.sector_impulse_n_s_per_m().sum(axis=0) == pytest.approx(
            history.total_impulse_n_s_per_m
        )

    def test_the_onset_threshold_is_the_callers_to_choose(
        self, history: BallReachHistory
    ) -> None:
        # A ball lying in a bunker touches sand before the swing starts,
        # so "arrival" and "loading" are different questions.
        assert history.loading_onset_s(fraction_of_peak=1.0) == pytest.approx(
            history.peak_traction_time_s
        )
        early = history.loading_onset_s(fraction_of_peak=0.1)
        assert early is not None
        assert early <= history.peak_traction_time_s

    @pytest.mark.parametrize("fraction", [0.0, -0.5, 1.5, math.nan])
    def test_an_onset_threshold_outside_the_unit_interval_is_refused(
        self, history: BallReachHistory, fraction: float
    ) -> None:
        with pytest.raises(SolverInputError, match="fraction_of_peak"):
            history.loading_onset_s(fraction_of_peak=fraction)

    def test_the_two_in_plane_splits_are_fractions_that_close(
        self, history: BallReachHistory
    ) -> None:
        assert 0.0 <= history.below_equator_fraction <= 1.0
        assert 0.0 <= history.face_side_fraction <= 1.0
        assert history.is_qualitative is True


class TestHistoryCarriesItsTier:
    """A number about the ball is not better than the tier behind it."""

    def test_the_tier_is_f1(self, history: BallReachHistory) -> None:
        assert history.fidelity_tier is FidelityTier.F1

    def test_the_status_can_be_no_better_than_beyond_validation(
        self, history: BallReachHistory
    ) -> None:
        assert history.verdict.status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_summary_states_the_tier_the_ceiling_and_the_validation_level(
        self, history: BallReachHistory
    ) -> None:
        summary = history.summary()
        assert "BEYOND_VALIDATION" in summary
        assert str(MAX_VALIDATED_SPEED_M_S) in summary
        assert "0 of 4" in summary
        assert "infinite cylinder" in summary

    def test_launch_is_still_f0s(self, history: BallReachHistory) -> None:
        with pytest.raises(OutOfEnvelopeError, match="ball_launch"):
            history.launch_velocity_m_s()

    def test_a_heel_toe_history_raises(self, history: BallReachHistory) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            history.heel_toe_history()

    def test_an_absolute_force_on_the_ball_raises(
        self, history: BallReachHistory
    ) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            history.total_force_on_ball_n()

    def test_a_body_index_outside_the_step_is_refused(
        self, run: MPMRun, open_solver: PlaneStrainMPMSolver
    ) -> None:
        with pytest.raises(SolverInputError, match="body_index"):
            ball_reach_history(
                run,
                _ball(),
                verdict=_verdict(open_solver),
                body_index=9,
                approach_direction=(1.0, 0.0),
            )


class TestSandVersusClub:
    """The comparison the epic was built for, kept to what is quotable."""

    @pytest.fixture(scope="class")
    def comparison(self, run: MPMRun, history: BallReachHistory) -> SandVersusClub:
        return compare_sand_and_club(run, history)

    def test_the_club_impulse_is_the_clubs_own_ledger(
        self, comparison: SandVersusClub, run: MPMRun
    ) -> None:
        expected = sum(step.body_contacts[0].impulse_on_sand_n_s for step in run.steps)
        assert comparison.club_impulse_on_sand_n_s_per_m == pytest.approx(expected)

    def test_the_transmitted_fraction_is_dimensionless_and_small(
        self, comparison: SandVersusClub
    ) -> None:
        fraction = comparison.transmitted_fraction
        assert 0.0 <= fraction < 1.0

    def test_a_club_that_delivered_nothing_transmits_nothing(
        self, history: BallReachHistory
    ) -> None:
        assert (
            SandVersusClub(
                club_impulse_on_sand_n_s_per_m=np.zeros(2),
                sand_impulse_on_ball_n_s_per_m=np.zeros(2),
                club_peak_time_s=0.0,
                ball_peak_time_s=0.0,
                club_first_contact_s=None,
                ball_first_arrival_s=None,
                verdict=history.verdict,
            ).transmitted_fraction
            == 0.0
        )

    def test_absolute_club_force_is_refused(self, comparison: SandVersusClub) -> None:
        with pytest.raises(OutOfEnvelopeError, match="club_force"):
            comparison.club_force_n()

    def test_absolute_ball_force_is_refused(self, comparison: SandVersusClub) -> None:
        with pytest.raises(OutOfEnvelopeError, match="out_of_plane"):
            comparison.ball_force_n()

    def test_the_summary_states_the_ratio_and_the_tier(
        self, comparison: SandVersusClub
    ) -> None:
        summary = comparison.summary()
        assert "BEYOND_VALIDATION" in summary
        assert "per metre of width" in summary

    def test_a_club_index_outside_the_step_is_refused(
        self, run: MPMRun, history: BallReachHistory
    ) -> None:
        with pytest.raises(SolverInputError, match="club_index"):
            compare_sand_and_club(run, history, club_index=9)


class TestTheLedgerStillCloses:
    """The ball's impulse is in the ledger and the total still closes.

    The identity is the one #8733 §2 pinned: with every wall ``FREE`` the
    only momentum sources are gravity and the bodies, so the sand's
    momentum change equals the summed contact impulses plus ``M g dt n``.
    Adding a *circular* ball and reading its traction off the same ledger
    must not disturb that -- the resolution is a re-reading of the ledger,
    not a second force model.
    """

    def test_the_budget_closes_to_round_off_with_the_ball_in_it(
        self, open_solver: PlaneStrainMPMSolver
    ) -> None:
        grid, particles, bounds = _bed(open_solver)
        total_mass = particles.total_mass_kg
        start = particles.linear_momentum_kg_m_s().copy()
        step_s = 2.0e-6
        run = open_solver.march_bodies(
            particles,
            (_club(), _ball().section),
            grid,
            n_steps=4,
            time_step_s=step_s,
            free_surface_height_m=0.0,
            bed_x_bounds_m=bounds,
        )
        assert run.max_pushed_out() == 0

        contact = sum(step.total_impulse_on_sand_n_s() for step in run.steps)
        gravity = np.array(
            [0.0, -total_mass * open_solver.gravity_m_s2 * step_s * run.n_steps]
        )
        change = run.steps[-1].linear_momentum_kg_m_s - start
        residual = float(np.hypot(*(change - contact - gravity)))
        scale = float(np.hypot(*change))
        assert scale > 0.0
        assert residual / scale <= 5.0e-15

    def test_the_balls_own_impulse_is_a_named_term_in_that_budget(
        self, open_solver: PlaneStrainMPMSolver, run: MPMRun, history: BallReachHistory
    ) -> None:
        ball_term = sum(step.body_contacts[1].impulse_on_sand_n_s for step in run.steps)
        assert float(np.hypot(*ball_term)) > 0.0
        # What the history reports the sand delivered *to the ball* is
        # exactly the negative of the ball's term in the sand's budget.
        assert history.total_impulse_n_s_per_m == pytest.approx(-ball_term)

    def test_the_node_ledger_survives_onto_the_body_contact(self, run: MPMRun) -> None:
        for step in run.steps:
            for contact in step.body_contacts:
                assert contact.nodes.n_contacts == contact.n_contacts


class TestThroughAWholeShotMarch:
    """The history read off a real marched shot, not a prescribed one.

    The pose replay is the thing being pinned here. ``ball_reach_history``
    does not store a pose per step; it re-applies the same ``advanced``
    the march applied, which is exact only because extra bodies in a
    march are prescribed rather than integrated. A *moving* ball is
    therefore the case that would catch a drift between the two.
    """

    @staticmethod
    def _delivery(speed_m_s: float, attack_deg: float) -> IntrusionState:
        corners = np.array(
            [
                [-0.012, 0.0, -0.005],
                [0.012, 0.0, -0.005],
                [0.012, 0.0, 0.005],
                [-0.012, 0.0, 0.005],
            ]
        )
        normals = np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1))
        areas = np.full(corners.shape[0], 2.0e-4)
        angle = math.radians(attack_deg)
        return IntrusionState(
            SurfaceElements(corners, normals, areas),
            (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
            free_surface_height_m=0.0,
        )

    @pytest.fixture(scope="class")
    def marched(self, material: SandContinuum) -> tuple[object, BallSection]:
        solver = PlaneStrainMPMSolver(
            material=material,
            cell_size_m=0.006,
            effective_width_m=0.03,
            bed_depth_m=0.018,
            run_in_lengths=0.5,
            refusal_policy=RefusalPolicy.REPORT,
        )
        ball = BallSection.at(
            (0.03, 0.012), radius_m=0.008, n_facets=8, velocity_m_s=(0.4, 0.0)
        )
        shot = simulate_f1_shot(
            solver,
            self._delivery(2.0, 25.0),
            settings=F1ShotSettings(
                head_mass_kg=0.05,
                max_time_s=0.0015,
                free_flight_lead_m=0.002,
                travel_allowance_m=0.05,
                ejecta_headroom_m=0.04,
                require_exit=False,
            ),
            extra_bodies=(ball.section,),
        )
        return shot, ball

    def test_the_history_replays_the_balls_own_march(
        self, marched: tuple, open_solver: PlaneStrainMPMSolver
    ) -> None:
        shot, ball = marched
        history = ball_reach_history(
            shot.run, ball, verdict=shot.shot.verdict, approach_direction=(1.0, -0.5)
        )
        assert len(history.samples) == shot.run.n_steps
        # The moving ball ends where the march left it, so the replayed
        # final pose is the run's own recorded one.
        replayed = ball.centre_m + (shot.run.n_steps - 1) * shot.run.time_step_s * (
            ball.velocity_m_s
        )
        assert shot.run.extra_sections[0].reference_point_m == pytest.approx(
            replayed + shot.run.time_step_s * ball.velocity_m_s
        )

    def test_the_comparison_reads_both_sides_off_one_solve(
        self, marched: tuple
    ) -> None:
        shot, ball = marched
        history = ball_reach_history(
            shot.run, ball, verdict=shot.shot.verdict, approach_direction=(1.0, -0.5)
        )
        comparison = compare_sand_and_club(shot.run, history)
        assert comparison.transmitted_fraction >= 0.0
        assert comparison.verdict is history.verdict
        assert "%" in comparison.summary()
