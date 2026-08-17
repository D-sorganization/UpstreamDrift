"""The scalar traces and the validity band (issue #8708, epic #8699).

The 3-D view localises; the traces quantify. Separately each is half an
answer, which is why #8708 asks for them on one time cursor: a designer
seeing a force peak at 6.2 ms wants to know instantly which part of the sole
was loaded then.

Everything here is headless. The drawing is tested in
``test_trace_panel_render``.

The property this module exists to protect is the **band**. A single validity
badge on a panel says "this shot was BEYOND_VALIDATION" and leaves the reader
to assume that applies evenly. It does not: a shot can sit inside the stated
envelope during the free-flight lead-in and leave it the moment the sole
loads, and that transition is exactly when the numbers stop meaning what they
appear to mean. A band over time says *when*.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.solvers import EnvelopeStatus
from src.tools.bunker_shot_gui.traces import (
    ScalarTrace,
    ShotTraces,
    TraceGroup,
    ValidityBand,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session")
def nominal_traces(nominal_shot) -> ShotTraces:  # type: ignore[no-untyped-def]
    """The scalar traces of the nominal shot."""
    traces = nominal_shot.traces
    assert traces is not None, nominal_shot.unavailable
    return traces


class TestTheTracesAreTheOnesTheIssueNames:
    """#8708 lists five quantities by name."""

    def test_the_wrench_components_are_all_present(
        self, nominal_traces: ShotTraces
    ) -> None:
        forces = nominal_traces.group(TraceGroup.SAND_FORCE)
        torques = nominal_traces.group(TraceGroup.SAND_TORQUE)
        assert len(forces) == 3
        assert len(torques) == 3

    def test_the_sole_depth_is_present(self, nominal_traces: ShotTraces) -> None:
        assert nominal_traces.trace("sole depth") is not None

    def test_the_speed_lost_is_present(self, nominal_traces: ShotTraces) -> None:
        assert nominal_traces.trace("speed lost") is not None

    def test_the_contact_patch_area_is_present(
        self, nominal_traces: ShotTraces
    ) -> None:
        assert nominal_traces.trace("contact patch area") is not None

    def test_every_trace_has_one_value_per_sample(
        self, nominal_traces: ShotTraces
    ) -> None:
        for trace in nominal_traces.traces:
            assert trace.values.shape == (nominal_traces.n_frames,)

    def test_an_unknown_trace_names_the_ones_there_are(
        self, nominal_traces: ShotTraces
    ) -> None:
        with pytest.raises(KeyError, match="sole depth"):
            nominal_traces.require("torque about the moon")


class TestEveryTraceStatesItsUnit:
    """The demo report's standard, and #8708 restates it."""

    def test_no_trace_is_unitless(self, nominal_traces: ShotTraces) -> None:
        for trace in nominal_traces.traces:
            assert trace.unit

    def test_the_units_are_the_ones_the_values_are_in(
        self, nominal_traces: ShotTraces
    ) -> None:
        """No downstream rescaling: what is plotted is what is stored."""
        depth = nominal_traces.require("sole depth")
        assert depth.unit == "mm"
        assert float(np.abs(depth.values).max()) > 1.0

    def test_the_axis_label_carries_the_unit(self, nominal_traces: ShotTraces) -> None:
        for trace in nominal_traces.traces:
            assert f"[{trace.unit}]" in trace.axis_label

    def test_time_is_reported_in_milliseconds_with_the_unit_stated(
        self, nominal_traces: ShotTraces
    ) -> None:
        assert nominal_traces.time_unit == "ms"
        assert np.allclose(nominal_traces.time_display, nominal_traces.time_s * 1e3)


class TestTheSoleDepthIsTheSoleDepth:
    """#8708 depends on the #8701 fix; this pins that it consumed it."""

    def test_the_depth_is_the_geometric_sole_depth_not_the_engaged_depth(
        self, nominal_traces: ShotTraces, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        depth = nominal_traces.require("sole depth")
        assert np.allclose(depth.values, nominal_shot.scene.sole_depth_m * 1e3)

    def test_the_depth_is_negative_while_the_head_is_still_in_the_air(
        self, nominal_traces: ShotTraces
    ) -> None:
        """The record opens with free flight, so it must start above zero."""
        assert float(nominal_traces.require("sole depth").values[0]) < 0.0

    def test_the_depth_does_not_read_zero_while_the_sole_is_buried(
        self, nominal_traces: ShotTraces
    ) -> None:
        """The exact defect #8701 describes: buried, but reported as zero."""
        values = nominal_traces.require("sole depth").values
        buried = values > 1.0
        assert buried.any()
        assert not np.any(values[buried] == 0.0)


class TestSpeedLostIsCumulative:
    """ "Where does the head lose speed" is a running quantity."""

    def test_no_speed_is_lost_at_the_first_sample(
        self, nominal_traces: ShotTraces
    ) -> None:
        assert float(nominal_traces.require("speed lost").values[0]) == pytest.approx(
            0.0, abs=1e-12
        )

    def test_the_head_loses_speed_over_the_strike(
        self, nominal_traces: ShotTraces
    ) -> None:
        assert float(nominal_traces.require("speed lost").values[-1]) > 0.0


class TestTheValidityIsABandNotABadge:
    """The load-bearing requirement of #8708."""

    def test_the_band_has_one_status_per_sample(
        self, nominal_traces: ShotTraces
    ) -> None:
        band = nominal_traces.band
        assert isinstance(band, ValidityBand)
        assert len(band.statuses) == nominal_traces.n_frames

    def test_the_band_reduces_to_contiguous_spans(
        self, nominal_traces: ShotTraces
    ) -> None:
        spans = nominal_traces.band.spans()
        assert spans
        assert spans[0].start_s == pytest.approx(nominal_traces.time_s[0])
        assert spans[-1].end_s == pytest.approx(nominal_traces.time_s[-1])

    def test_the_spans_tile_the_record_without_gaps(
        self, nominal_traces: ShotTraces
    ) -> None:
        spans = nominal_traces.band.spans()
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier.end_s == pytest.approx(later.start_s)

    def test_neighbouring_spans_never_share_a_status(
        self, nominal_traces: ShotTraces
    ) -> None:
        spans = nominal_traces.band.spans()
        for earlier, later in zip(spans, spans[1:], strict=False):
            assert earlier.status is not later.status

    def test_the_worst_span_is_the_verdict_the_whole_shot_carries(
        self, nominal_traces: ShotTraces, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        """The band is a reconstruction; this pins that it reconstructs.

        ``ShotResult.verdict`` is ``worst_of`` the per-step verdicts the march
        recorded and then discarded. If the band's own worst status disagrees,
        the band is judging the shot by different rules from the solver.
        """
        assert nominal_traces.band.worst is nominal_shot.verdict.status

    def test_the_band_can_say_that_the_shot_left_the_envelope_partway(
        self, nominal_traces: ShotTraces
    ) -> None:
        band = nominal_traces.band
        assert band.changes is (len(band.spans()) > 1)

    def test_every_span_states_its_status_in_words(
        self, nominal_traces: ShotTraces
    ) -> None:
        for span in nominal_traces.band.spans():
            assert span.label
            assert span.duration_s >= 0.0

    def test_the_status_at_a_frame_is_the_status_of_its_span(
        self, nominal_traces: ShotTraces
    ) -> None:
        band = nominal_traces.band
        for frame in (0, band.n_frames // 2, band.n_frames - 1):
            moment = float(band.time_s[frame])
            covering = [
                span for span in band.spans() if span.start_s <= moment <= span.end_s
            ]
            assert band.status_at(frame) in {span.status for span in covering}

    def test_a_frame_outside_the_record_is_refused(
        self, nominal_traces: ShotTraces
    ) -> None:
        with pytest.raises(ValueError, match="outside the recorded shot"):
            nominal_traces.band.status_at(nominal_traces.n_frames)


class TestTheBandCatchesARealMidShotTransition:
    """The case a badge cannot express, on a shot that actually does it.

    On the nominal greenside delivery the band is uniform, and that is not a
    bug: ``MAX_VALIDATED_SPEED_M_S`` is 1.44 m/s, so a 25 m/s head is past
    the published corpus from the first free-flight sample and never comes
    back. Drop the delivery to 1.5 m/s and the sand slows the head through
    that ceiling *during the strike*, which is the transition #8708 is about
    -- so the machinery is exercised against a shot that changes regime
    rather than only against one that cannot.
    """

    def test_a_slow_delivery_changes_regime_partway_through(
        self, decelerating_traces: ShotTraces
    ) -> None:
        assert decelerating_traces.band.changes is True

    def test_the_transition_happens_inside_the_record(
        self, decelerating_traces: ShotTraces
    ) -> None:
        spans = decelerating_traces.band.spans()
        assert len(spans) >= 2
        crossing = spans[0].end_s
        assert float(decelerating_traces.time_s[0]) < crossing
        assert crossing < float(decelerating_traces.time_s[-1])

    def test_the_shot_improves_rather_than_degrades_as_it_slows(
        self, decelerating_traces: ShotTraces
    ) -> None:
        """Losing speed moves the head back toward the published corpus."""
        spans = decelerating_traces.band.spans()
        assert spans[0].status is EnvelopeStatus.BEYOND_VALIDATION
        assert spans[-1].status is EnvelopeStatus.EXTRAPOLATED

    def test_the_worst_span_is_still_the_verdict_the_shot_carries(
        self, decelerating_shot
    ) -> None:  # type: ignore[no-untyped-def]
        """The reconstruction has to hold on a changing shot too."""
        assert decelerating_shot.traces.band.worst is decelerating_shot.verdict.status

    def test_a_badge_would_have_mislabelled_most_of_the_record(
        self, decelerating_traces: ShotTraces
    ) -> None:
        """Why the band exists, stated as a measurement.

        The whole-shot verdict is the worst one anywhere in it. Here that
        verdict is wrong for the majority of the record, which is exactly
        what a single badge in the corner of a panel would assert.
        """
        band = decelerating_traces.band
        worst = band.worst
        mislabelled = sum(1 for status in band.statuses if status is not worst)
        assert mislabelled > band.n_frames // 2


class TestTheBandDefendsItself:
    """``raise``, never ``assert``."""

    def test_a_band_with_the_wrong_number_of_statuses_is_refused(self) -> None:
        with pytest.raises(ValueError, match="one status per sample"):
            ValidityBand(
                time_s=np.array([0.0, 1.0, 2.0]),
                statuses=(EnvelopeStatus.WITHIN, EnvelopeStatus.WITHIN),
            )

    def test_a_band_holding_something_that_is_not_a_status_is_refused(self) -> None:
        with pytest.raises(ValueError, match="EnvelopeStatus"):
            ValidityBand(time_s=np.array([0.0, 1.0]), statuses=("within", "within"))

    def test_a_band_needs_at_least_two_samples(self) -> None:
        with pytest.raises(ValueError, match="at least 2 samples"):
            ValidityBand(time_s=np.array([0.0]), statuses=(EnvelopeStatus.WITHIN,))

    def test_a_band_with_non_increasing_time_is_refused(self) -> None:
        with pytest.raises(ValueError, match="strictly increasing"):
            ValidityBand(
                time_s=np.array([0.0, 0.0]),
                statuses=(EnvelopeStatus.WITHIN, EnvelopeStatus.WITHIN),
            )


class TestTheTraceSetDefendsItself:
    """A panel that plots mismatched arrays is worse than an empty one."""

    def test_a_trace_of_the_wrong_length_is_refused(self) -> None:
        band = ValidityBand(
            time_s=np.array([0.0, 1.0]),
            statuses=(EnvelopeStatus.WITHIN, EnvelopeStatus.WITHIN),
        )
        with pytest.raises(ValueError, match="one value per sample"):
            ShotTraces(
                time_s=np.array([0.0, 1.0]),
                traces=(
                    ScalarTrace(
                        name="bad",
                        unit="N",
                        values=np.array([1.0, 2.0, 3.0]),
                        group=TraceGroup.SAND_FORCE,
                    ),
                ),
                band=band,
            )

    def test_a_band_from_a_different_shot_is_refused(self) -> None:
        band = ValidityBand(
            time_s=np.array([0.0, 1.0, 2.0]),
            statuses=(EnvelopeStatus.WITHIN,) * 3,
        )
        with pytest.raises(ValueError, match="same shot"):
            ShotTraces(time_s=np.array([0.0, 1.0]), traces=(), band=band)

    def test_a_trace_with_a_non_finite_value_is_refused(self) -> None:
        with pytest.raises(ValueError, match="finite"):
            ScalarTrace(
                name="bad",
                unit="N",
                values=np.array([1.0, np.nan]),
                group=TraceGroup.SAND_FORCE,
            )

    def test_a_trace_without_a_unit_is_refused(self) -> None:
        with pytest.raises(ValueError, match="unit"):
            ScalarTrace(
                name="bad",
                unit="",
                values=np.array([1.0, 2.0]),
                group=TraceGroup.SAND_FORCE,
            )

    def test_two_traces_cannot_share_a_name(self) -> None:
        band = ValidityBand(
            time_s=np.array([0.0, 1.0]),
            statuses=(EnvelopeStatus.WITHIN, EnvelopeStatus.WITHIN),
        )
        duplicate = ScalarTrace(
            name="same",
            unit="N",
            values=np.array([1.0, 2.0]),
            group=TraceGroup.SAND_FORCE,
        )
        with pytest.raises(ValueError, match="unique"):
            ShotTraces(
                time_s=np.array([0.0, 1.0]),
                traces=(duplicate, duplicate),
                band=band,
            )


class TestTheTracesAreGroupedForPlotting:
    """Six wrench components on one axis in newtons is not a panel."""

    def test_the_groups_present_are_reported_in_a_stable_order(
        self, nominal_traces: ShotTraces
    ) -> None:
        groups = nominal_traces.groups()
        assert groups == tuple(
            group for group in TraceGroup if nominal_traces.group(group)
        )

    def test_every_group_states_its_shared_unit(
        self, nominal_traces: ShotTraces
    ) -> None:
        for group in nominal_traces.groups():
            units = {trace.unit for trace in nominal_traces.group(group)}
            assert len(units) == 1

    def test_every_group_has_a_heading(self, nominal_traces: ShotTraces) -> None:
        for group in TraceGroup:
            assert group.label
