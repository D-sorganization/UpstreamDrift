"""Forgiveness sensitivities against the published baselines (issue #8614, W7).

Wivou et al. (2016) measured carry vs entry distance at r = -0.98 and carry vs
divot depth at r = -0.91. Those are the numbers to beat, and they are
measurements from the literature rather than outputs of this model.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    SWEEP_RANGES,
    WIVOU_2016_CARRY_CORRELATION,
    forgiveness_report,
    forgiveness_sensitivity,
)

pytestmark = pytest.mark.unit

TARGET_CARRY_M = 30.0


class TestSweepRanges:
    """The registry of measured sweep ranges, each carrying its source."""

    def test_entry_distance_and_divot_depth_match_the_research_table(self) -> None:
        """25-150 mm and 20-60 mm, as measured."""
        entry = SWEEP_RANGES["entry_distance_behind_ball_m"]
        divot = SWEEP_RANGES["divot_depth_m"]

        assert (entry.low, entry.high) == (0.025, 0.150)
        assert (divot.low, divot.high) == (0.020, 0.060)
        assert "Wivou" in entry.source

    def test_angles_are_stored_in_radians(self) -> None:
        """Attack angle -2 to -12 deg, face open 0-30 deg, shaft lean 4-14 deg."""
        attack = SWEEP_RANGES["attack_angle_rad"]
        face = SWEEP_RANGES["face_open_angle_rad"]
        lean = SWEEP_RANGES["shaft_lean_rad"]

        assert np.degrees(attack.low) == pytest.approx(-12.0)
        assert np.degrees(attack.high) == pytest.approx(-2.0)
        assert np.degrees(face.high) == pytest.approx(30.0)
        assert np.degrees(lean.low) == pytest.approx(4.0)
        assert np.degrees(lean.high) == pytest.approx(14.0)

    def test_symmetric_ranges_are_symmetric(self) -> None:
        """Strike location +/-15 mm and lie +/-5 deg."""
        strike = SWEEP_RANGES["strike_location_heel_toe_m"]
        lie = SWEEP_RANGES["lie_deviation_rad"]

        assert (strike.low, strike.high) == (-0.015, 0.015)
        assert np.degrees(lie.low) == pytest.approx(-5.0)
        assert np.degrees(lie.high) == pytest.approx(5.0)

    def test_penetrometer_readings_are_converted_to_kilopascals(self) -> None:
        """1.6 kg/cm^2 = 156.9 kPa and 2.8 kg/cm^2 = 274.6 kPa."""
        firmness = SWEEP_RANGES["sand_firmness_kPa"]

        assert firmness.low == pytest.approx(156.9, rel=1e-3)
        assert firmness.high == pytest.approx(274.6, rel=1e-3)

    def test_the_registry_is_read_only(self) -> None:
        """A measured range is not something a caller edits in place."""
        with pytest.raises(TypeError):
            SWEEP_RANGES["entry_distance_behind_ball_m"] = None  # type: ignore[index]


class TestFactorSensitivity:
    """One factor at a time, in the linear form the baselines are stated in."""

    def test_a_perfectly_linear_response_correlates_at_minus_one(self) -> None:
        """carry = 30 - 40 * (d - 0.025) over the full 25-150 mm range.

        The slope is exactly -40 m per metre of entry distance, the span is
        0.125 m, so carry falls 5 m across the sweep: a fractional change of
        -5 / 30 = -1/6. The correlation is exactly -1, which is *worse* than the
        -0.98 baseline, so the design is not more forgiving than the measurement.
        """
        entry_m = np.linspace(0.025, 0.150, 6)
        carry_m = 30.0 - 40.0 * (entry_m - 0.025)

        sensitivity = forgiveness_sensitivity(
            entry_m,
            carry_m,
            factor="entry_distance_behind_ball_m",
            target_carry_m=TARGET_CARRY_M,
        )

        assert sensitivity.correlation_r == pytest.approx(-1.0, rel=1e-12)
        assert sensitivity.slope_m_per_unit == pytest.approx(-40.0, rel=1e-9)
        assert sensitivity.carry_change_over_span_m == pytest.approx(-5.0, rel=1e-9)
        assert sensitivity.fractional_carry_change == pytest.approx(
            -1.0 / 6.0, rel=1e-9
        )
        assert sensitivity.unit == "m"
        assert sensitivity.baseline_r == -0.98
        assert sensitivity.more_forgiving_than_baseline is False

    def test_a_weakly_tracking_design_beats_the_baseline(self) -> None:
        """A response dominated by scatter correlates below |r| = 0.91."""
        depth_m = np.linspace(0.020, 0.060, 9)
        carry_m = np.array([30.0, 28.0, 31.0, 29.0, 30.5, 28.5, 30.0, 29.5, 30.2])

        sensitivity = forgiveness_sensitivity(
            depth_m, carry_m, factor="divot_depth_m", target_carry_m=TARGET_CARRY_M
        )

        assert abs(sensitivity.correlation_r) < 0.91
        assert sensitivity.baseline_r == -0.91
        assert sensitivity.more_forgiving_than_baseline is True

    def test_range_coverage_is_reported(self) -> None:
        """A sweep that stops short of the registered range says so."""
        partial_m = np.linspace(0.025, 0.100, 5)
        carry_m = 30.0 - 10.0 * partial_m

        partial = forgiveness_sensitivity(
            partial_m,
            carry_m,
            factor="entry_distance_behind_ball_m",
            target_carry_m=TARGET_CARRY_M,
        )
        full_m = np.linspace(0.025, 0.150, 5)
        full = forgiveness_sensitivity(
            full_m,
            30.0 - 10.0 * full_m,
            factor="entry_distance_behind_ball_m",
            target_carry_m=TARGET_CARRY_M,
        )

        assert partial.covers_declared_range is False
        assert full.covers_declared_range is True

    def test_an_unregistered_factor_has_no_baseline_or_coverage(self) -> None:
        """Nothing is invented for a factor the registry does not know."""
        values = np.linspace(0.0, 1.0, 5)

        sensitivity = forgiveness_sensitivity(
            values, 30.0 - values, factor="grind_relief", target_carry_m=TARGET_CARRY_M
        )

        assert sensitivity.baseline_r is None
        assert sensitivity.more_forgiving_than_baseline is None
        assert sensitivity.covers_declared_range is None

    def test_two_samples_are_refused(self) -> None:
        """Two points are always perfectly correlated, which says nothing."""
        with pytest.raises(ValueError, match="at least 3 samples"):
            forgiveness_sensitivity(
                np.array([0.025, 0.150]),
                np.array([30.0, 25.0]),
                factor="entry_distance_behind_ball_m",
                target_carry_m=TARGET_CARRY_M,
            )

    def test_a_constant_factor_is_refused(self) -> None:
        """No variation in the factor means no sensitivity to measure."""
        with pytest.raises(ValueError, match="all equal"):
            forgiveness_sensitivity(
                np.full(5, 0.05),
                np.linspace(30.0, 25.0, 5),
                factor="entry_distance_behind_ball_m",
                target_carry_m=TARGET_CARRY_M,
            )

    def test_a_constant_carry_is_refused_rather_than_scored_zero(self) -> None:
        """A perfectly flat response has an undefined correlation; say so."""
        with pytest.raises(ValueError, match="carry samples are all equal"):
            forgiveness_sensitivity(
                np.linspace(0.025, 0.150, 5),
                np.full(5, 30.0),
                factor="entry_distance_behind_ball_m",
                target_carry_m=TARGET_CARRY_M,
            )


class TestForgivenessReport:
    """Several factors, ranked by how much carry they actually move."""

    @pytest.fixture
    def report(self):
        """Entry distance moving carry 5 m, divot depth moving it 1 m."""
        entry_m = np.linspace(0.025, 0.150, 6)
        depth_m = np.linspace(0.020, 0.060, 6)
        return forgiveness_report(
            {
                "divot_depth_m": (depth_m, 30.0 - 25.0 * (depth_m - 0.020)),
                "entry_distance_behind_ball_m": (
                    entry_m,
                    30.0 - 40.0 * (entry_m - 0.025),
                ),
            },
            target_carry_m=TARGET_CARRY_M,
        )

    def test_the_ranking_is_by_fractional_carry_change(self, report) -> None:
        """Entry distance moves carry 5 m; divot depth moves it 1 m."""
        ranked = report.ranked()

        assert ranked[0].factor == "entry_distance_behind_ball_m"
        assert ranked[0].carry_change_over_span_m == pytest.approx(-5.0, rel=1e-9)
        assert ranked[1].factor == "divot_depth_m"
        assert ranked[1].carry_change_over_span_m == pytest.approx(-1.0, rel=1e-9)

    def test_both_perfect_correlations_fail_their_baselines(self, report) -> None:
        """A noise-free linear model tracks harder than the measured shots do."""
        assert len(report.worse_than_baseline()) == 2

    def test_an_empty_report_is_refused(self) -> None:
        """A report over no factors is a mistake, not an empty answer."""
        with pytest.raises(ValueError, match="at least one factor"):
            forgiveness_report({}, target_carry_m=TARGET_CARRY_M)


def test_the_published_baselines_are_recorded_verbatim() -> None:
    """r = -0.98 for entry distance and r = -0.91 for divot depth."""
    assert WIVOU_2016_CARRY_CORRELATION["entry_distance_behind_ball_m"] == -0.98
    assert WIVOU_2016_CARRY_CORRELATION["divot_depth_m"] == -0.91
