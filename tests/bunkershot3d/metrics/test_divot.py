"""Divot geometry and the dig/skid discriminator, against hand arithmetic.

Issue #8614 (W7). Every expected value below is worked out in the test itself
from the synthetic trace's definition, so a passing test means the arithmetic is
right rather than that the code agrees with its own previous output.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    DigSkidVerdict,
    dig_vs_skid,
    divot_metrics,
    sole_depth_profile,
    submerged_interval,
)

from .conftest import (
    DIVOT_WIDTH_M,
    SAND_BULK_DENSITY_KG_M3,
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
                vee_trace, head, scene, width_m=0.0, bulk_density_kg_m3=1550.0
            )
        with pytest.raises(ValueError, match="bulk_density_kg_m3 must be positive"):
            divot_metrics(vee_trace, head, scene, width_m=0.02, bulk_density_kg_m3=-1.0)


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
