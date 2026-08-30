"""Divot geometry and the dig/skid discriminator, against hand arithmetic.

Issue #8614 (W7). Every expected value below is worked out in the test itself
from the synthetic trace's definition, so a passing test means the arithmetic is
right rather than that the code agrees with its own previous output.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    F1_ENTRAINMENT_FACTOR_BOUNDS,
    AcceleratedSandMass,
    DigSkidVerdict,
    dig_vs_skid,
    divot_metrics,
    lateral_spread_factor,
    sole_depth_profile,
    submerged_interval,
)

from .conftest import (
    DIVOT_WIDTH_M,
    SAND_BULK_DENSITY_KG_M3,
    SAND_FRICTION_ANGLE_DEG,
    VEE_SPEED_MPS,
    build_piecewise_trace,
    build_vee_trace,
)

pytestmark = pytest.mark.unit


class TestSubmergedInterval:
    """Entry and exit crossings of the undisturbed sand surface."""

    def test_entry_and_exit_land_on_the_defined_breakpoints(
        self, vee_trace, head, scene
    ) -> None:
        """The vee trace is defined to cross zero depth at -0.120 and +0.060 m."""
        interval = submerged_interval(vee_trace, head, scene)

        assert interval.entry_travel_m == pytest.approx(-0.120, abs=1e-12)
        assert interval.exit_travel_m == pytest.approx(0.060, abs=1e-12)
        # Travel / speed: 0.180 m at 20 m/s = 9.0 ms.
        assert interval.duration_s == pytest.approx(0.180 / VEE_SPEED_MPS, rel=1e-12)

    def test_a_trace_that_never_enters_the_sand_is_refused(self, head, scene) -> None:
        """A sole held above the surface has no divot, and saying so is the answer."""
        airborne = build_piecewise_trace(
            [(-0.200, -0.030), (0.000, -0.010), (0.100, -0.030)]
        )

        with pytest.raises(ValueError, match="never goes below"):
            submerged_interval(airborne, head, scene)

    def test_a_trace_that_ends_submerged_is_refused(self, head, scene) -> None:
        """Without an exit in the record, divot length cannot be measured."""
        unfinished = build_piecewise_trace(
            [(-0.200, -0.020), (-0.120, 0.000), (0.100, 0.040)]
        )

        with pytest.raises(ValueError, match="still below the surface"):
            submerged_interval(unfinished, head, scene)


class TestDivotMetrics:
    """The divot the vee trace cuts, computed by hand."""

    @pytest.fixture
    def metrics(self, vee_trace, head, scene):
        """Divot metrics for the vee trace at a 20 mm width in 1550 kg/m^3 sand."""
        return divot_metrics(
            vee_trace,
            head,
            scene,
            width_m=DIVOT_WIDTH_M,
            bulk_density_kg_m3=SAND_BULK_DENSITY_KG_M3,
            friction_angle_deg=SAND_FRICTION_ANGLE_DEG,
        )

    def test_entry_distance_behind_the_ball(self, metrics) -> None:
        """Entry is at station -0.120 m, i.e. 120 mm behind the ball."""
        assert metrics.entry_distance_behind_ball_m == pytest.approx(0.120, abs=1e-12)
        assert metrics.entry_point_m[2] == pytest.approx(0.0, abs=1e-12)

    def test_maximum_depth_and_where_it_happens(self, metrics) -> None:
        """The apex is defined at station -0.040 m, 0.25 * 0.080 = 0.020 m deep."""
        assert metrics.max_depth_m == pytest.approx(0.020, rel=1e-12)
        assert metrics.max_depth_behind_ball_m == pytest.approx(0.040, abs=1e-12)

    def test_exit_and_length(self, metrics) -> None:
        """Exit is 0.020 / 0.20 = 0.100 m past the apex; length = 0.060 + 0.120."""
        assert metrics.exit_distance_past_ball_m == pytest.approx(0.060, abs=1e-12)
        assert metrics.length_m == pytest.approx(0.180, rel=1e-12)

    def test_section_area_is_the_triangle_area(self, metrics) -> None:
        """A triangular depth profile: area = 0.5 * base * height.

        0.5 * 0.180 m * 0.020 m = 1.8e-3 m^2. Both breakpoints sit on samples,
        so the trapezoidal integration is exact rather than approximate.
        """
        assert metrics.section_area_m2 == pytest.approx(0.5 * 0.180 * 0.020, rel=1e-9)

    def test_volume_and_mass(self, metrics) -> None:
        """1.8e-3 m^2 * 0.020 m = 3.6e-5 m^3; * 1550 kg/m^3 = 0.0558 kg."""
        assert metrics.volume_m3 == pytest.approx(3.6e-5, rel=1e-9)
        assert metrics.mass_kg == pytest.approx(3.6e-5 * 1550.0, rel=1e-9)
        assert metrics.mass_kg == pytest.approx(0.0558, rel=1e-9)

    def test_width_and_density_must_be_positive(self, vee_trace, head, scene) -> None:
        """A zero width would silently report a zero-mass divot."""
        with pytest.raises(ValueError, match="width_m must be positive"):
            divot_metrics(
                vee_trace,
                head,
                scene,
                width_m=0.0,
                bulk_density_kg_m3=1550.0,
                friction_angle_deg=SAND_FRICTION_ANGLE_DEG,
            )
        with pytest.raises(ValueError, match="bulk_density_kg_m3 must be positive"):
            divot_metrics(
                vee_trace,
                head,
                scene,
                width_m=0.02,
                bulk_density_kg_m3=-1.0,
                friction_angle_deg=SAND_FRICTION_ANGLE_DEG,
            )

    def test_the_bed_friction_angle_has_no_default(
        self, vee_trace, head, scene
    ) -> None:
        """The angle the divot walls lie back at is the bed's, not this module's.

        Defaulting it would be an invented divot shape, and the sand state
        already carries the number (issue #8659).
        """
        with pytest.raises(TypeError, match="friction_angle_deg"):
            divot_metrics(
                vee_trace,
                head,
                scene,
                width_m=0.02,
                bulk_density_kg_m3=1550.0,
            )
        with pytest.raises(ValueError, match="friction_angle_deg must lie"):
            divot_metrics(
                vee_trace,
                head,
                scene,
                width_m=0.02,
                bulk_density_kg_m3=1550.0,
                friction_angle_deg=0.0,
            )

    def test_the_depth_squared_integral_is_the_triangle_second_moment(
        self, metrics
    ) -> None:
        """Two straight limbs, so ``integral d^2 ds`` is exact by hand.

        Descending limb: depth rises 0 -> 0.020 m over 0.080 m of travel, so
        ``integral d^2 ds = 0.080 * 0.020^2 / 3``. Ascending limb: 0.020 -> 0
        over 0.100 m, so ``0.100 * 0.020^2 / 3``. Together
        ``0.180 * 4e-4 / 3 = 2.4e-5 m^3``.

        The tolerance is quadrature, not slop. The trapezoid rule is exact for
        the vee's *linear* depth -- which is why the section area is checked at
        1e-9 -- but not for its square, so this integral carries an
        ``O(dx^2)`` error at the trace's 2 mm sampling. It comes out
        2.4006e-5, 0.025 % high, and a tighter bound would only be pinning the
        sample spacing.
        """
        assert metrics.depth_squared_integral_m3 == pytest.approx(2.4e-5, rel=1e-3)


class TestAcceleratedSandMass:
    """The mass the strike moved, which is not the mass under the sole.

    Issue #8659: dividing the delivered impulse by the swept prism implied
    sand leaving faster than the head that threw it, so the prism is no longer
    the denominator. Every expected value below is worked out from the vee
    trace's own definition and the two stated factors.
    """

    @pytest.fixture
    def metrics(self, vee_trace, head, scene):
        """Divot metrics for the vee trace, in a 34 deg bed at 20 mm."""
        return divot_metrics(
            vee_trace,
            head,
            scene,
            width_m=DIVOT_WIDTH_M,
            bulk_density_kg_m3=SAND_BULK_DENSITY_KG_M3,
            friction_angle_deg=SAND_FRICTION_ANGLE_DEG,
        )

    def test_the_prism_is_carried_through_unchanged(self, metrics) -> None:
        """The reported divot mass is still the prism, with its provenance."""
        assert metrics.accelerated_mass.prismatic_kg == pytest.approx(
            metrics.mass_kg, rel=1e-12
        )

    def test_the_lateral_factor_is_the_hand_computed_trapezoid(self, metrics) -> None:
        """``1 + cot(34 deg) * (integral d^2 ds) / (w * integral d ds)``.

        cot(34 deg) = 1.482561 and the vee's two integrals are 2.4e-5 m^3 over
        1.8e-3 m^2, so a 20 mm sole is widened by 1 + 1.482561 * 0.666667 =
        1.98837. The realised value is 1.98862, 0.013 % high, because the
        trapezoid rule is exact for the vee's linear depth but not for its
        square -- the same ``O(dx^2)`` quadrature error the second-moment test
        above records. Both the hand value and the exact composition of the
        two integrals are checked, so neither a wrong formula nor a wrong
        integral could pass.
        """
        assert metrics.accelerated_mass.lateral_factor == pytest.approx(
            1.0
            + (1.0 / np.tan(np.radians(SAND_FRICTION_ANGLE_DEG)))
            * (2.4e-5 / (DIVOT_WIDTH_M * 1.8e-3)),
            rel=1e-3,
        )
        assert metrics.accelerated_mass.lateral_factor == pytest.approx(
            lateral_spread_factor(
                metrics.section_area_m2,
                metrics.depth_squared_integral_m3,
                width_m=DIVOT_WIDTH_M,
                wall_angle_deg=SAND_FRICTION_ANGLE_DEG,
            ),
            rel=1e-12,
        )
        assert metrics.accelerated_mass.lateral_factor > 1.0

    def test_the_interval_edges_are_the_two_stated_models(self, metrics) -> None:
        """Lower is F1 alone; upper is F1's widest reading times the walls."""
        accelerated = metrics.accelerated_mass
        lower_factor, upper_factor = F1_ENTRAINMENT_FACTOR_BOUNDS

        assert accelerated.lower_kg == pytest.approx(
            metrics.mass_kg * lower_factor, rel=1e-12
        )
        assert accelerated.upper_kg == pytest.approx(
            metrics.mass_kg * upper_factor * accelerated.lateral_factor, rel=1e-12
        )
        assert accelerated.bounds_kg == (accelerated.lower_kg, accelerated.upper_kg)

    def test_the_central_value_is_the_geometric_mean_of_the_edges(
        self, metrics
    ) -> None:
        accelerated = metrics.accelerated_mass
        assert accelerated.central_kg == pytest.approx(
            np.sqrt(accelerated.lower_kg * accelerated.upper_kg), rel=1e-12
        )
        assert accelerated.lower_kg < accelerated.central_kg < accelerated.upper_kg

    def test_the_accelerated_mass_never_falls_below_the_prism(self, metrics) -> None:
        """Sand outside the swept prism can only add mass, never remove it."""
        assert metrics.accelerated_mass.lower_kg > metrics.mass_kg

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("prismatic_kg", 0.0),
            ("prismatic_kg", -1.0),
            ("entrainment_lower", 0.5),
            ("entrainment_upper", float("nan")),
            ("lateral_factor", 0.9),
            ("wall_angle_deg", 0.0),
            ("wall_angle_deg", 90.0),
        ],
    )
    def test_an_impossible_interval_is_refused(self, field: str, value: float) -> None:
        """A ``raise`` and not a contract: ball launch divides by this."""
        fields = {
            "prismatic_kg": 0.05,
            "entrainment_lower": 2.0,
            "entrainment_upper": 3.0,
            "lateral_factor": 1.5,
            "wall_angle_deg": 34.0,
        }
        fields[field] = value

        with pytest.raises(ValueError):
            AcceleratedSandMass(**fields)

    def test_bounds_out_of_order_are_refused(self) -> None:
        with pytest.raises(ValueError, match="out of order"):
            AcceleratedSandMass(
                prismatic_kg=0.05,
                entrainment_lower=3.0,
                entrainment_upper=2.0,
                lateral_factor=1.5,
                wall_angle_deg=34.0,
            )

    def test_a_flat_divot_has_no_walls_to_widen(self) -> None:
        """A zero section is not widened; it is returned as it is."""
        assert lateral_spread_factor(0.0, 0.0, width_m=0.02, wall_angle_deg=34.0) == 1.0

    def test_the_widening_grows_with_the_wall_lying_flatter(self) -> None:
        """A looser bed lays its walls back further and moves more sand."""
        steep = lateral_spread_factor(1.8e-3, 2.4e-5, width_m=0.02, wall_angle_deg=45.0)
        shallow = lateral_spread_factor(
            1.8e-3, 2.4e-5, width_m=0.02, wall_angle_deg=25.0
        )
        assert shallow > steep > 1.0


class TestDepthProfile:
    """The depth-versus-travel trace itself, which is a requested output."""

    def test_profile_matches_the_defining_slopes(self, vee_trace, head, scene) -> None:
        """Sampling the profile 10 mm and 40 mm past entry recovers the slopes."""
        profile = sole_depth_profile(vee_trace, head, scene)

        # Descending limb: 0.25 * 0.010 m and 0.25 * 0.040 m.
        assert profile.depth_at_travel_m(-0.110) == pytest.approx(0.0025, rel=1e-9)
        assert profile.depth_at_travel_m(-0.080) == pytest.approx(0.010, rel=1e-9)
        # Ascending limb: 0.020 - 0.20 * 0.040 m.
        assert profile.depth_at_travel_m(0.000) == pytest.approx(0.012, rel=1e-9)


class TestDigVersusSkid:
    """The discriminator, and the vertical impulse balance beside it.

    Every trace below shares one ascending limb of slope 0.20 at 20 m/s, so the
    sole leaves the sand climbing at exactly 4 m/s and only the descent it
    arrived with changes. The descent-return ratio is then a ratio of two
    exactly known speeds.
    """

    def test_a_sole_that_arrives_steep_and_leaves_slowly_is_a_dig(
        self, head, scene
    ) -> None:
        """Down at 0.50, out at 0.20: 4 / 10 = 0.40, below the 0.50 threshold.

        The sole enters at 120 mm behind the ball descending on a 0.50 slope, so
        at 20 m/s its downward speed is 10 m/s; it bottoms 20 mm deep 40 mm
        later and climbs out on the 0.20 limb at 4 m/s. The sand kept 60 % of
        the descent it was handed. arctan(0.50) = 26.5651 deg, reported negative
        for a descending blow.
        """
        trace = build_piecewise_trace(
            [
                (-0.200, -0.040),
                (-0.120, 0.000),
                (-0.080, 0.020),
                (0.020, 0.000),
                (0.100, -0.016),
            ]
        )

        result = dig_vs_skid(trace, head, scene)

        assert result.entry_descent_speed_mps == pytest.approx(10.0, rel=1e-9)
        assert result.exit_climb_speed_mps == pytest.approx(4.0, rel=1e-9)
        assert result.descent_return_ratio == pytest.approx(0.40, rel=1e-9)
        assert result.verdict is DigSkidVerdict.DIG
        assert np.degrees(result.entry_attack_angle_rad) == pytest.approx(
            -26.5650511771, rel=1e-9
        )

    def test_the_vee_trace_sits_on_the_skid_threshold(
        self, vee_trace, head, scene
    ) -> None:
        """Down at 0.25 (5 m/s), out at 0.20 (4 m/s): exactly 0.80.

        The threshold is inclusive, so the boundary case is a skid rather than
        marginal, and that is pinned here rather than left to a comparison.
        """
        result = dig_vs_skid(vee_trace, head, scene)

        assert result.entry_descent_speed_mps == pytest.approx(5.0, rel=1e-9)
        assert result.exit_climb_speed_mps == pytest.approx(4.0, rel=1e-9)
        assert result.descent_return_ratio == pytest.approx(0.80, rel=1e-9)
        assert result.verdict is DigSkidVerdict.SKID

    def test_a_sole_handed_all_its_descent_back_is_a_skid(self, head, scene) -> None:
        """Down at 0.25 (5 m/s), out at 0.25 (5 m/s): the sand returned it all."""
        trace = build_piecewise_trace(
            [
                (-0.200, -0.020),
                (-0.120, 0.000),
                (-0.040, 0.020),
                (0.040, 0.000),
                (0.100, -0.015),
            ]
        )

        result = dig_vs_skid(trace, head, scene)

        assert result.descent_return_ratio == pytest.approx(1.0, rel=1e-9)
        assert result.verdict is DigSkidVerdict.SKID

    def test_the_marginal_band_is_reported_rather_than_forced(
        self, head, scene
    ) -> None:
        """Down at 0.3125 (6.25 m/s), out at 4 m/s: 0.64, between the bands."""
        trace = build_piecewise_trace(
            [
                (-0.200, -0.025),
                (-0.120, 0.000),
                (-0.056, 0.020),
                (0.044, 0.000),
                (0.100, -0.0112),
            ]
        )

        result = dig_vs_skid(trace, head, scene)

        assert result.descent_return_ratio == pytest.approx(0.64, rel=1e-9)
        assert result.verdict is DigSkidVerdict.MARGINAL

    def test_thresholds_must_leave_a_band(self, vee_trace, head, scene) -> None:
        """An inverted pair of thresholds would make the verdict meaningless."""
        with pytest.raises(ValueError):
            dig_vs_skid(
                vee_trace,
                head,
                scene,
                dig_descent_return=0.9,
                skid_descent_return=0.4,
            )

    def test_a_sole_that_is_not_descending_at_entry_is_refused(
        self, head, scene
    ) -> None:
        """Without a descent to return, the ratio has no denominator.

        A 0.1 mm scuff one sample wide, 78 mm before the real entry, is the
        first sample the interval finder calls submerged. The centred velocity
        there straddles two samples at the same height, so the measured descent
        is exactly zero and the ratio would be a division by nothing.
        """
        trace = build_piecewise_trace(
            [
                (-0.200, -0.020),
                (-0.122, -0.001),
                (-0.120, 0.0001),
                (-0.118, -0.001),
                (-0.040, 0.020),
                (0.060, 0.000),
                (0.100, -0.008),
            ],
        )

        with pytest.raises(ValueError, match="descending"):
            dig_vs_skid(trace, head, scene)

    def test_vertical_impulse_balance_closes_on_the_reported_terms(
        self, head, scene
    ) -> None:
        """The four impulse terms are an identity, by construction.

        With a constant 500 N upward sand force the sampled window runs from the
        first submerged sample (station -0.118 m) to the last (station +0.058 m),
        which is 88 intervals of 0.1 ms = 8.8 ms:

        * sand impulse    = 500 N * 8.8e-3 s = 4.40 N.s
        * gravity impulse = -0.300 kg * 9.80665 m/s^2 * 8.8e-3 s = -0.0258896 N.s
        * momentum change = 0.300 kg * (4 - (-5)) m/s = 2.70 N.s, because the sole
          descends at 0.25 * 20 = 5 m/s and leaves rising at 0.20 * 20 = 4 m/s
        * constraint      = 2.70 - 4.40 + 0.0258896 = -1.6741104 N.s
        """
        trace = build_vee_trace(force_N=np.array([0.0, 0.0, 500.0]))

        result = dig_vs_skid(trace, head, scene)

        sampled_duration_s = 88 * 1.0e-4
        assert result.vertical_sand_impulse_Ns == pytest.approx(
            500.0 * sampled_duration_s, rel=1e-9
        )
        assert result.gravity_impulse_Ns == pytest.approx(
            -0.300 * 9.80665 * sampled_duration_s, rel=1e-9
        )
        assert result.measured_vertical_momentum_change_Ns == pytest.approx(
            0.300 * 9.0, rel=1e-9
        )
        assert result.constraint_vertical_impulse_Ns == pytest.approx(
            result.measured_vertical_momentum_change_Ns
            - result.vertical_sand_impulse_Ns
            - result.gravity_impulse_Ns,
            rel=1e-12,
        )
        assert result.constraint_vertical_impulse_Ns == pytest.approx(
            -1.6741104, rel=1e-6
        )
