"""Characterization tests for ``putting_dynamics`` (epic #14141, PR 1).

**These are characterization (drift-detection) tests, not physical
validation.** They pin *invariants and properties* the package already
relies on (finiteness, non-negativity, monotonicity, symmetry,
determinism, terminal state) plus closed forms derived independently
from first principles or published definitions (cited per test). No
expected value is a snapshot of the model's own output: a failure here
means a contract broke, not that "the number moved."
"""

from __future__ import annotations

import math

import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_RADIUS_M,
    GRAVITY_M_S2,
)
from src.shared.python.putting_dynamics import (
    BallState,
    FrictionParams,
    HeightField,
    Mode,
    PutterState,
    SurfaceSpec,
    ball_kinetic_energy_j,
    bumpy_friction_field,
    bumpy_height_field,
    simulate_ball,
    simulate_strike,
    stimp_to_rolling_mu,
)

pytestmark = pytest.mark.unit

_G = float(GRAVITY_M_S2)
_R = float(GOLF_BALL_RADIUS_M)

#: Meters per foot — the Stimpmeter reading's native unit (module
#: credit: USGA Stimpmeter Instruction Booklet).
_FOOT_M = 0.3048


def _pure_roll_ball(v0_mps: float, vy_mps: float = 0.0) -> BallState:
    """A ball already at the no-slip rolling speed (skips the skid phase)."""
    speed = math.hypot(v0_mps, vy_mps)
    return BallState(
        x_m=0.0, y_m=0.0, vx_mps=v0_mps, vy_mps=vy_mps, spin_rad_s=speed / _R
    )


class TestSpeedsAndDistancesAreNonNegativeAndFinite:
    """Property: no sample, nor any summary field, is NaN/Inf or negative."""

    @pytest.mark.parametrize("speed_mps", [1.2, 1.8, 2.5, 3.5])
    @pytest.mark.parametrize("loft_deg", [-2.0, 0.0, 4.0, 8.0])
    def test_struck_putt_stays_finite_and_non_negative(
        self, speed_mps: float, loft_deg: float
    ) -> None:
        putter = PutterState(
            head_mass_kg=0.34, moi_kg_m2=4.2e-4, loft_deg=loft_deg, speed_mps=speed_mps
        )
        result = simulate_strike(putter, SurfaceSpec.flat_uniform(stimp_ft=10.0))
        for sample in result.samples:
            assert math.isfinite(sample.speed_mps) and sample.speed_mps >= 0.0
            assert math.isfinite(sample.height_m) and sample.height_m >= 0.0
            assert math.isfinite(sample.x_m)
            assert math.isfinite(sample.y_m)
        for field in (
            result.rest_x_m,
            result.rest_y_m,
            result.total_distance_m,
            result.time_s,
            result.skid_distance_m,
        ):
            assert math.isfinite(field)
        assert result.total_distance_m >= 0.0
        assert result.time_s >= 0.0
        assert 0.0 <= result.skid_distance_m <= result.total_distance_m + 1e-9


class TestTerminalState:
    """Property: a putt ends captured, at rest, or at the time cap."""

    def test_un_holed_putt_comes_to_rest(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        result = simulate_ball(_pure_roll_ball(1.4), surface)
        assert not result.holed
        assert result.final_mode is Mode.REST
        assert result.samples[-1].speed_mps == 0.0

    def test_dying_putt_over_the_hole_is_captured(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        mu_r = surface.friction.mu_roll0
        v0 = math.sqrt(2.0 * mu_r * _G * 1.5)  # rolls ~1.5 m on a straight line
        result = simulate_ball(_pure_roll_ball(v0), surface, hole_x_m=1.3, hole_y_m=0.0)
        assert result.holed
        assert result.speed_at_hole_mps is not None

    def test_capture_never_happens_if_the_ball_never_reaches_the_hole(self) -> None:
        """A hole far beyond the roll-out distance is never crossed."""
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        result = simulate_ball(
            _pure_roll_ball(1.0), surface, hole_x_m=50.0, hole_y_m=0.0
        )
        assert result.total_distance_m < 50.0
        assert not result.holed
        assert result.speed_at_hole_mps is None

    def test_slope_steeper_than_rolling_resistance_runs_to_the_time_cap(self) -> None:
        """A slope steeper than the rolling and static-hold bounds never
        settles (``friction`` module docstring), so the ``_MAX_TIME_S``
        cap ends the loop; hence "timed out" is a third terminal state.
        """
        surface = SurfaceSpec(
            height=HeightField.planar(10.0, 0.0, extent_m=30.0),
            friction_field=SurfaceSpec.flat_uniform().friction_field,
            friction=FrictionParams(mu_roll0=stimp_to_rolling_mu(10.0)),
        )
        result = simulate_ball(BallState(x_m=0.0, y_m=0.0), surface)
        assert not result.holed
        assert result.final_mode is Mode.ROLL
        assert result.time_s > 55.0  # _MAX_TIME_S is 60 s


class TestMechanicalEnergyNeverIncreasesOnAFlatGreen:
    """Property: friction only dissipates — ground-mode KE never rises.

    On a flat green height is constant, so potential energy is
    constant and "mechanical energy" reduces to the ball's
    translational + rotational kinetic energy. Both the rolling and
    sliding friction laws are purely dissipative (module docstrings),
    so this must hold regardless of the exact friction coefficients.
    """

    @pytest.mark.parametrize("v0_mps,spin_scale", [(1.6, 1.0), (1.6, 1.3), (2.0, 1.5)])
    def test_ground_phase_energy_is_monotone_non_increasing(
        self, v0_mps: float, spin_scale: float
    ) -> None:
        """Pure-roll and over-spun starts never hit the slide->roll pin."""
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        ball = BallState(
            x_m=0.0, y_m=0.0, vx_mps=v0_mps, spin_rad_s=spin_scale * v0_mps / _R
        )
        result = simulate_ball(ball, surface)
        ground_samples = [s for s in result.samples if s.mode is not Mode.AIRBORNE]
        energies = [
            ball_kinetic_energy_j(s.speed_mps, s.spin_rad_s) for s in ground_samples
        ]
        # Tiny tolerance for RK4 discretization noise, not a physical
        # energy source.
        assert all(
            later <= earlier + 1e-9
            for earlier, later in zip(energies, energies[1:], strict=False)
        )

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "FINDING (#14141 PR1): the slide->roll 'within_one_step'"
            " early-pin in simulate_ball snaps spin up to rolling speed"
            " for an under-spun start, injecting ~2e-5 J of KE. Physics"
            " fix deferred to the roll/friction migration PR (#14141)."
        ),
    )
    def test_under_spun_start_has_a_tiny_energy_bump_at_the_roll_transition(
        self,
    ) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        v0_mps, spin_scale = 2.2, 0.8
        ball = BallState(
            x_m=0.0, y_m=0.0, vx_mps=v0_mps, spin_rad_s=spin_scale * v0_mps / _R
        )
        result = simulate_ball(ball, surface)
        ground_samples = [s for s in result.samples if s.mode is not Mode.AIRBORNE]
        energies = [
            ball_kinetic_energy_j(s.speed_mps, s.spin_rad_s) for s in ground_samples
        ]
        assert all(
            later <= earlier + 1e-9
            for earlier, later in zip(energies, energies[1:], strict=False)
        )


class TestLateralMirrorSymmetry:
    """Property: mirroring lateral velocity mirrors the lateral result.

    A flat, uniform green has no preferred left/right direction, so
    negating the initial lateral velocity must negate every sample's
    ``y`` while leaving every sample's ``x`` (and all timing) alone.
    """

    def test_mirrored_lateral_velocity_mirrors_the_whole_trajectory(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        left = simulate_ball(_pure_roll_ball(1.3, 0.4), surface)
        right = simulate_ball(_pure_roll_ball(1.3, -0.4), surface)
        assert len(left.samples) == len(right.samples)
        for a, b in zip(left.samples, right.samples, strict=True):
            assert a.t_s == pytest.approx(b.t_s, abs=1e-12)
            assert a.x_m == pytest.approx(b.x_m, abs=1e-9)
            assert a.y_m == pytest.approx(-b.y_m, abs=1e-9)
        assert left.rest_y_m > 0.0
        assert left.rest_y_m == pytest.approx(-right.rest_y_m, rel=1e-9)
        assert left.rest_x_m == pytest.approx(right.rest_x_m, rel=1e-9)


class TestMonotonicity:
    """Property: more speed, or a faster green, both mean a longer roll."""

    def test_faster_strike_on_a_flat_green_rolls_out_farther(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        distances = [
            simulate_strike(
                PutterState(
                    head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=2.0, speed_mps=v
                ),
                surface,
            ).total_distance_m
            for v in (1.5, 2.0, 2.5, 3.0)
        ]
        assert all(a < b for a, b in zip(distances, distances[1:], strict=False))

    def test_higher_stimp_rolls_out_farther_for_the_same_launch_speed(self) -> None:
        # Stay below the default static-hold bound (0.08) so every
        # sample stimp produces a valid FrictionParams.
        distances = [
            simulate_ball(
                _pure_roll_ball(1.4), SurfaceSpec.flat_uniform(stimp_ft=stimp)
            ).total_distance_m
            for stimp in (9.0, 11.0, 13.0, 15.0)
        ]
        assert all(a < b for a, b in zip(distances, distances[1:], strict=False))


class TestDeterminism:
    """Property: a fixed seed reproduces the identical putt, bit-for-bit."""

    def test_same_seed_reproduces_the_full_trajectory(self) -> None:
        def build() -> SurfaceSpec:
            flat = SurfaceSpec.flat_uniform(stimp_ft=11.0)
            return SurfaceSpec(
                height=bumpy_height_field(99, 0.004, 1.0, flat.height),
                friction_field=bumpy_friction_field(99, 0.15, 1.0, flat.friction_field),
                friction=flat.friction,
            )

        putter = PutterState(
            head_mass_kg=0.33, moi_kg_m2=4.0e-4, loft_deg=4.0, speed_mps=2.2
        )
        a = simulate_strike(putter, build(), hole_x_m=2.5, hole_y_m=0.0)
        b = simulate_strike(putter, build(), hole_x_m=2.5, hole_y_m=0.0)
        assert a.samples == b.samples
        assert a.holed == b.holed
        assert a.rest_x_m == b.rest_x_m
        assert a.rest_y_m == b.rest_y_m
        assert a.time_s == b.time_s


class TestIndependentClosedForms:
    """Closed forms derived from first principles, not from the model.

    Each formula below is restated from a definition independent of
    the module under test, so a drift in the module's constants or
    physics shows up here even though the module's own analytic-limit
    tests (``test_solver.py``) still pass.
    """

    def test_flat_green_rolling_distance_matches_elementary_kinematics(self) -> None:
        """``d = v^2 / (2 a)`` for constant deceleration ``a = mu_r g``.

        Elementary kinematics (``v_f^2 = v_0^2 - 2 a d``, ``v_f = 0``);
        independent of the solver's RK4 implementation. A pure-roll
        start (spin already matched to speed) skips the skid phase, so
        the whole path is governed by this one formula.
        """
        mu_r = 0.06  # chosen directly, not derived via stimp_to_rolling_mu
        surface = SurfaceSpec.flat_uniform(friction=FrictionParams(mu_roll0=mu_r))
        v0 = 1.4
        result = simulate_ball(_pure_roll_ball(v0), surface)
        a = mu_r * _G
        expected_d = v0**2 / (2.0 * a)
        # RK4 step + the v_stop_mps cutoff (friction.py) truncate the
        # last few millimeters versus the idealized v_f = 0 endpoint.
        assert result.total_distance_m == pytest.approx(expected_d, rel=0.01)

    def test_usga_stimpmeter_definition_recovers_rolling_mu(self) -> None:
        """USGA Stimpmeter definition, restated independently of friction.py.

        The USGA Stimpmeter Instruction Booklet releases the ball from
        a fixed V-groove notch 30 inches up a ramp inclined ~20 degrees,
        giving a release speed of ``v_release ~= 1.83 m/s``; the stimp
        reading ``S`` [ft] is the roll distance under the green's
        constant rolling deceleration ``mu_r g``. Solving
        ``v_release^2 = 2 (mu_r g) (S * 0.3048)`` for ``mu_r`` gives the
        formula below — written here from the definition, not imported
        from :mod:`~.friction`, so the test still catches drift in that
        module's restated constant.
        """
        v_release_mps = 1.83  # USGA Stimpmeter Instruction Booklet
        for stimp_ft in (8.0, 10.0, 13.0):
            expected_mu_r = v_release_mps**2 / (2.0 * _G * stimp_ft * _FOOT_M)
            assert stimp_to_rolling_mu(stimp_ft) == pytest.approx(
                expected_mu_r, rel=1e-9
            )


class TestContractPreconditions:
    """Invalid inputs to the public façade raise ValueError (DbC)."""

    def test_putter_state_rejects_non_positive_speed(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            PutterState(
                head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=3.0, speed_mps=0.0
            )

    def test_putter_state_rejects_implausible_loft(self) -> None:
        with pytest.raises(ValueError, match="loft"):
            PutterState(
                head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=45.0, speed_mps=2.0
            )

    def test_ball_state_rejects_excessive_speed(self) -> None:
        with pytest.raises(ValueError, match="speed"):
            BallState(x_m=0.0, y_m=0.0, vx_mps=25.0)

    def test_simulate_ball_rejects_a_half_specified_hole(self) -> None:
        surface = SurfaceSpec.flat_uniform(stimp_ft=10.0)
        with pytest.raises(ValueError, match="hole"):
            simulate_ball(BallState(x_m=0.0, y_m=0.0), surface, hole_x_m=1.0)

    def test_height_field_rejects_a_non_array_grid(self) -> None:
        with pytest.raises(ValueError):
            HeightField(heights_m=[[0.0, 0.0], [0.0, 0.0]])  # type: ignore[arg-type]

    def test_stimp_to_rolling_mu_rejects_out_of_band_reading(self) -> None:
        with pytest.raises(ValueError, match="stimp"):
            stimp_to_rolling_mu(100.0)
