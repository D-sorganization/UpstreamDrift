"""A separating dig/skid verdict is still not a measured one (issue #8703).

The verdict now rests on the **descent-return ratio** -- the sole's upward
speed leaving the sand over its downward speed entering it -- which separates
the design space where the entry slope ratio it replaced did not. Separation is
pinned by :mod:`tests.bunkershot3d.metrics.test_dig_skid_separation`; this file
pins the honesty that has to survive it.

Two things stay true after the fix:

* the DIG and SKID thresholds on the ratio are **conventions**. Nobody has
  published a vertical restitution for a wedge sole leaving bunker sand, so
  ``calibrated`` is ``False`` and ``require_calibrated`` refuses; and
* the F0 model's response to *marketed bounce* disagrees with fitting
  practice in the shallow, non-burying regime -- over the workbench's 384-point
  sweep the ratio falls from 0.424 to 0.339 as marketed bounce rises from 8 to
  26 deg at -14 deg of attack, and maximum sole depth rises from 18.0 to
  21.5 mm over the same span, so the *model* puts more bounce deeper there.
  The verdict reports what the model did; it must not be read as a bounce
  recommendation, and the calibration record says so.

The history the fix rests on is kept in :mod:`bunkershot3d.metrics.divot`'s
module docstring rather than repeated here.
"""

from __future__ import annotations

import pytest

from bunkershot3d.metrics import (
    DEFAULT_DIG_DESCENT_RETURN,
    DEFAULT_SKID_DESCENT_RETURN,
    DIG_SKID_BOUNCE_ORDERING_REASON,
    DIG_SKID_COARSE_WINDOW_REASON,
    DIG_SKID_UNCALIBRATED_REASON,
    MIN_RESOLVED_SUBMERGED_SAMPLES,
    DigSkidCalibration,
    dig_vs_skid,
)

from .conftest import build_piecewise_trace

pytestmark = pytest.mark.unit

#: The fixed head of the coarse-window diagnostic, before its first placeholder.
_COARSE_PREFIX = "the entry and exit speeds are centred di"


class TestEveryVerdictDeclaresItselfUncalibrated:
    """Marked at the API level, not only in report prose."""

    def test_the_result_carries_a_calibration_record(
        self, vee_trace, head, scene
    ) -> None:
        result = dig_vs_skid(vee_trace, head, scene)

        assert isinstance(result.calibration, DigSkidCalibration)

    def test_the_calibration_is_never_calibrated(self, vee_trace, head, scene) -> None:
        """No threshold on this ratio has been measured (issue #8703)."""
        assert dig_vs_skid(vee_trace, head, scene).calibration.calibrated is False

    def test_requiring_a_calibrated_verdict_refuses(
        self, vee_trace, head, scene
    ) -> None:
        """A caller that asks whether it may quote the verdict is told no."""
        calibration = dig_vs_skid(vee_trace, head, scene).calibration

        with pytest.raises(ValueError, match="not calibrated"):
            calibration.require_calibrated()

    def test_the_reason_states_what_is_unmeasured(self, vee_trace, head, scene) -> None:
        reasons = dig_vs_skid(vee_trace, head, scene).calibration.reasons

        assert DIG_SKID_UNCALIBRATED_REASON in reasons
        assert "8703" in DIG_SKID_UNCALIBRATED_REASON
        assert "restitution" in DIG_SKID_UNCALIBRATED_REASON

    def test_the_bounce_ordering_caveat_travels_with_every_verdict(
        self, vee_trace, head, scene
    ) -> None:
        """A designer must not read the verdict as a bounce recommendation."""
        reasons = dig_vs_skid(vee_trace, head, scene).calibration.reasons

        assert DIG_SKID_BOUNCE_ORDERING_REASON in reasons
        assert "bounce" in DIG_SKID_BOUNCE_ORDERING_REASON

    def test_the_calibration_names_the_thresholds_the_verdict_used(
        self, vee_trace, head, scene
    ) -> None:
        """The bands are conventions, so the verdict has to carry them."""
        calibration = dig_vs_skid(vee_trace, head, scene).calibration

        assert calibration.dig_descent_return == DEFAULT_DIG_DESCENT_RETURN
        assert calibration.skid_descent_return == DEFAULT_SKID_DESCENT_RETURN

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


class TestTheResolutionOfTheTwoSpeedsIsReported:
    """Both speeds are centred differences, so the sample count matters."""

    def test_the_submerged_sample_count_is_carried(
        self, vee_trace, head, scene
    ) -> None:
        """The vee trace is submerged from station -0.118 m to +0.058 m.

        That is 89 samples at the 2 mm spacing, inclusive of both ends.
        """
        calibration = dig_vs_skid(vee_trace, head, scene).calibration

        assert calibration.submerged_samples == 89

    def test_a_well_resolved_window_is_not_flagged(
        self, vee_trace, head, scene
    ) -> None:
        assert not any(
            reason.startswith(_COARSE_PREFIX)
            for reason in dig_vs_skid(vee_trace, head, scene).calibration.reasons
        )

    def test_a_window_of_a_few_samples_is_reported(self, head, scene) -> None:
        """A 6 mm dip is three samples of travel: answerable, barely.

        It is answered, with the reason why the two speeds are resolution
        limited stated beside the answer rather than left to the reader.
        """
        trace = build_piecewise_trace(
            [
                (-0.200, -0.020),
                (-0.120, 0.000),
                (-0.116, 0.001),
                (-0.112, 0.000),
                (0.100, -0.053),
            ]
        )

        calibration = dig_vs_skid(trace, head, scene).calibration

        assert calibration.submerged_samples < MIN_RESOLVED_SUBMERGED_SAMPLES
        assert any(reason.startswith(_COARSE_PREFIX) for reason in calibration.reasons)

    def test_a_window_too_short_to_hold_two_distinct_speeds_is_refused(
        self, head, scene
    ) -> None:
        """One or two submerged samples share their centred differences.

        The entry and exit speeds would then be the same measurement, and the
        ratio would be an arithmetic identity rather than a strike.
        """
        trace = build_piecewise_trace(
            [
                (-0.200, -0.020),
                (-0.120, 0.000),
                (-0.118, 0.0005),
                (-0.116, 0.000),
                (0.100, -0.054),
            ]
        )

        with pytest.raises(ValueError, match="submerged samples"):
            dig_vs_skid(trace, head, scene)
