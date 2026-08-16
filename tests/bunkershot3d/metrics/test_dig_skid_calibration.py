"""The dig-versus-skid verdict may not be quoted as a finding (issue #8703).

The discriminator was measured across the demo's full 77-point sweep --
geometric bounce 14-26 deg, sole width 16-24 mm, attack angle -2 to -14 deg,
four sand conditions -- and returned ``MARGINAL`` at every single point, with
slope ratios spanning only 0.9987 to 1.0000.

The head is not held on rails: ``simulate_shot`` integrates translation under
the sand wrench. The fault is one of **scale**. The shipped ``entry_window_m``
of 10 mm is about 0.4 ms of travel at a 25 m/s delivery, and a 0.3 kg head
under an order-5 N.s impulse cannot bend measurably in 0.4 ms.

Resizing the window was measured before being rejected. Over 48 design points
the ratio span grows from 0.0015 at 10 mm to 0.28 at half the divot length --
but its correlation with maximum sole depth is **negative at every informative
window** (-0.50 to -0.68), so the deepest-cutting designs come out nearest the
skid threshold. A resized window ships an inverted verdict, not a calibrated
one.

Underneath both observations is one structural fact, pinned by
:class:`TestTheRatioIsPinnedAtBothEndsOfTheWindow`: the entry chord starts at
the entry crossing and the depth returns to zero at the exit crossing, so the
ratio is 1 for a vanishing window and 0 for a window spanning the divot, **for
every design**. Any threshold on it is a statement about window placement.

So the verdict is marked rather than retuned, and this file pins the marking.
"""

from __future__ import annotations

import pytest

from bunkershot3d.metrics import (
    DEFAULT_DIG_SLOPE_RATIO,
    DEFAULT_SKID_SLOPE_RATIO,
    DIG_SKID_UNCALIBRATED_REASON,
    DIG_SKID_UNDEFLECTED_ENTRY_REASON,
    MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES,
    DigSkidCalibration,
    dig_vs_skid,
)

from .conftest import VEE_DX_M, build_piecewise_trace, build_vee_trace

pytestmark = pytest.mark.unit

#: Entry and exit crossings of the vee fixture, so the divot is 180 mm long.
VEE_DIVOT_LENGTH_M = 0.180


class TestTheRatioIsPinnedAtBothEndsOfTheWindow:
    """Why no window size is a free parameter, and both ends are refused."""

    def test_a_window_shorter_than_one_sample_of_travel_is_refused(
        self, vee_trace, head, scene
    ) -> None:
        """Below one sample the numerator is interpolated inside one step.

        The vee trace samples every ``VEE_DX_M`` = 2 mm of travel, so a 1 mm
        window is resolved by a single linear interpolation between the entry
        sample and the next -- the same straight segment the delivered slope
        was measured on. The answer would be the input divided by itself.
        """
        with pytest.raises(ValueError, match="shorter than one sample of travel"):
            dig_vs_skid(vee_trace, head, scene, entry_window_m=0.5 * VEE_DX_M)

    def test_a_window_spanning_the_whole_divot_is_refused(
        self, vee_trace, head, scene
    ) -> None:
        """The chord through the exit crossing is identically zero.

        Depth is zero at entry and zero again at exit, so a chord across the
        divot has slope 0 whatever the sole did in between, and the verdict
        would be SKID for every design.
        """
        with pytest.raises(ValueError, match="reaches the whole"):
            dig_vs_skid(vee_trace, head, scene, entry_window_m=VEE_DIVOT_LENGTH_M)

    @pytest.mark.parametrize("window_m", [0.5 * VEE_DX_M, VEE_DIVOT_LENGTH_M])
    def test_both_refusals_name_the_issue(
        self, vee_trace, head, scene, window_m: float
    ) -> None:
        with pytest.raises(ValueError, match="8703"):
            dig_vs_skid(vee_trace, head, scene, entry_window_m=window_m)

    def test_the_shipped_default_is_still_answered(
        self, vee_trace, head, scene
    ) -> None:
        """The refusals are narrow: they do not swallow the shipped window."""
        result = dig_vs_skid(vee_trace, head, scene)

        assert result.calibration.entry_window_samples == pytest.approx(
            0.010 / VEE_DX_M, rel=1e-9
        )
        assert result.calibration.entry_window_divot_fraction == pytest.approx(
            0.010 / VEE_DIVOT_LENGTH_M, rel=1e-9
        )


class TestEveryVerdictDeclaresItselfUncalibrated:
    """Marked at the API level, not only in report prose."""

    def test_the_result_carries_a_calibration_record(
        self, vee_trace, head, scene
    ) -> None:
        result = dig_vs_skid(vee_trace, head, scene)

        assert isinstance(result.calibration, DigSkidCalibration)

    def test_the_calibration_is_never_calibrated(self, vee_trace, head, scene) -> None:
        """No threshold on this ratio has been established (issue #8703)."""
        assert dig_vs_skid(vee_trace, head, scene).calibration.calibrated is False

    def test_requiring_a_calibrated_verdict_refuses(
        self, vee_trace, head, scene
    ) -> None:
        """A caller that asks whether it may quote the verdict is told no."""
        calibration = dig_vs_skid(vee_trace, head, scene).calibration

        with pytest.raises(ValueError, match="not calibrated"):
            calibration.require_calibrated()

    def test_the_reason_states_the_saturation_and_the_inversion(
        self, vee_trace, head, scene
    ) -> None:
        reasons = dig_vs_skid(vee_trace, head, scene).calibration.reasons

        assert DIG_SKID_UNCALIBRATED_REASON in reasons
        assert "8703" in DIG_SKID_UNCALIBRATED_REASON
        assert "0.9987" in DIG_SKID_UNCALIBRATED_REASON
        assert "negativ" in DIG_SKID_UNCALIBRATED_REASON

    def test_the_calibration_names_the_thresholds_the_verdict_used(
        self, vee_trace, head, scene
    ) -> None:
        """The bands are conventions, so the verdict has to carry them."""
        calibration = dig_vs_skid(vee_trace, head, scene).calibration

        assert calibration.dig_slope_ratio == DEFAULT_DIG_SLOPE_RATIO
        assert calibration.skid_slope_ratio == DEFAULT_SKID_SLOPE_RATIO

    def test_no_threshold_of_the_discriminator_is_measured(
        self, vee_trace, head, scene
    ) -> None:
        """Mirrors ``MaterialResponse.measured_constants`` and the ball model."""
        assert (
            dig_vs_skid(vee_trace, head, scene).calibration.measured_constants() == ()
        )

    def test_the_summary_is_human_readable(self, vee_trace, head, scene) -> None:
        summary = dig_vs_skid(vee_trace, head, scene).calibration.summary()

        assert "UNCALIBRATED" in summary
        assert "samples" in summary


class TestTheDegeneraciesThatSurviveTheWindowAreReported:
    """Answerable windows can still carry no information, and say so."""

    def test_an_entry_still_on_the_delivered_line_is_flagged(
        self, vee_trace, head, scene
    ) -> None:
        """The vee trace enters on exactly the slope it was delivered on.

        Its ratio is 1.0, and that is arithmetic rather than physics: over
        this window the sand has not bent the path at all.
        """
        result = dig_vs_skid(vee_trace, head, scene)

        assert result.slope_ratio == pytest.approx(1.0, rel=1e-12)
        assert result.calibration.undeflected_entry is True
        assert DIG_SKID_UNDEFLECTED_ENTRY_REASON in result.calibration.reasons

    def test_a_deflected_entry_is_not_flagged(self, head, scene) -> None:
        """Delivered at 0.20, penetrating at 0.30: the sole path did bend."""
        trace = build_piecewise_trace(
            [
                (-0.200, -0.016),
                (-0.120, 0.000),
                (-0.060, 0.018),
                (0.030, 0.000),
                (0.100, -0.014),
            ]
        )

        result = dig_vs_skid(trace, head, scene)

        assert result.calibration.undeflected_entry is False
        assert DIG_SKID_UNDEFLECTED_ENTRY_REASON not in result.calibration.reasons

    def test_a_window_of_one_or_two_samples_is_reported(self, head, scene) -> None:
        """The regime the demo ran in: 10 mm on a 6.24 mm sample spacing.

        It is answerable, so it is answered -- with the reason why the answer
        carries almost no information stated beside it.
        """
        trace = build_vee_trace()
        window_m = 1.5 * VEE_DX_M

        result = dig_vs_skid(trace, head, scene, entry_window_m=window_m)

        assert result.calibration.entry_window_samples == pytest.approx(1.5, rel=1e-9)
        assert result.calibration.entry_window_samples < (
            MIN_INFORMATIVE_ENTRY_WINDOW_SAMPLES
        )
        assert any("samples of along-track" in r for r in result.calibration.reasons)

    def test_a_wide_chord_is_reported_as_a_placement_on_the_pinned_arc(
        self, vee_trace, head, scene
    ) -> None:
        """Half the divot spreads the ratio, and inverts it; say so."""
        result = dig_vs_skid(
            vee_trace, head, scene, entry_window_m=0.5 * VEE_DIVOT_LENGTH_M
        )

        assert result.calibration.entry_window_divot_fraction == pytest.approx(0.5)
        assert any("fixed arc" in r for r in result.calibration.reasons)
