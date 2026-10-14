"""Playing a 3-D particle trajectory back (issue #9545, folded-in #9693).

Every artifact here is the SYNTHETIC TEST FIXTURE from
``_synthetic_particle_artifact``: no production solver emits the
trajectory artifact yet. These tests pin the viewer's contract, not
physics -- that a scrubbed grain keeps its identity, that display
decimation never changes a sampled value, and that club and ball are
placed by simulation time and never extrapolated across missing data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")

from bunkershot3d.fields.schema import ParticleKind  # noqa: E402
from bunkershot3d.fields.trajectory_reader import ParticleTrajectoryReader  # noqa: E402
from src.tools.bunker_shot_gui.particle_playback import (  # noqa: E402
    BodySource,
    BodyTrack,
    DisplayDecimation,
    FrameRateReport,
    ParticlePlayback,
    measure_frame_rate,
)
from tests.unit.tools.bunker_shot_gui._synthetic_particle_artifact import (  # noqa: E402
    FRAME_TIMES_S,
    GRAIN_IDS,
    SYNTHETIC_SOLVER_NAME,
    grain_position,
    write_synthetic_artifact,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture()
def reader(tmp_path: Path):  # type: ignore[no-untyped-def]
    """The synthetic fixture, written, verified and open."""
    path = write_synthetic_artifact(tmp_path / "synthetic.h5")
    with ParticleTrajectoryReader(path) as opened:
        yield opened


def _club() -> BodyTrack:
    """A club sampled on its own, finer, unequal cadence."""
    times = np.array([0.0, 0.5e-4, 2.0e-4, 4.0e-4, 6.0e-4])
    positions = np.column_stack([times * 30.0, np.zeros(5), np.full(5, 0.01)])
    return BodyTrack("club", times, positions, BodySource.SIMULATED)


def _ball() -> BodyTrack:
    """A ball with no data before release and a dropped sample mid-flight."""
    times = np.array([2.0e-4, 2.6e-4, 3.4e-4, 5.0e-4])
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.001, 0.0, 0.002], [np.nan] * 3, [0.004, 0.0, 0.008]]
    )
    return BodyTrack("ball", times, positions, BodySource.SIMULATED)


class TestIdentitySurvivesScrubbing:
    """Grain 31 and grain 2 cross; scrubbing must not swap them."""

    def test_every_frame_matches_the_reader_by_id(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        for order in ([0, 1, 2, 3, 4], [4, 2, 0, 3, 1]):
            for index in order:
                playback.seek_frame(index)
                view = playback.view()
                for grain_id, position in zip(
                    view.grain_ids, view.positions_m, strict=True
                ):
                    expected = reader.positions_of(int(grain_id))[index]
                    np.testing.assert_array_equal(position, expected)

    def test_crossing_grains_keep_their_own_tracks(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        playback.seek_frame(4)
        view = playback.view()
        row = int(np.flatnonzero(view.grain_ids == 31)[0])
        np.testing.assert_allclose(view.positions_m[row], grain_position(0, 5.0e-4))

    def test_inactive_and_lost_grains_are_not_drawn(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        playback.seek_frame(0)
        assert 47 not in set(playback.view().visible_ids().tolist())
        playback.seek_frame(4)
        assert 23 not in set(playback.view().visible_ids().tolist())


class TestTransport:
    """Scrub and step move by retained frame; time is the frame's own."""

    def test_step_clamps_at_both_ends(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        assert playback.step(-1) == 0
        assert playback.step(10) == len(FRAME_TIMES_S) - 1

    def test_seek_time_picks_the_nearest_retained_frame(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        assert playback.seek_time(2.7e-4) == 2
        assert playback.seek_time(2.8e-4) == 3
        assert playback.time_s == pytest.approx(FRAME_TIMES_S[3])

    def test_scrub_maps_a_fraction_onto_time_not_frame_index(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader)
        # Half the duration is 2.5e-4 s, which is frame 2 -- not frame 2 of
        # 5 by coincidence of index arithmetic: frame 2 is at 2.5e-4 s.
        assert playback.scrub(0.5) == 2
        assert playback.scrub(1.0) == len(FRAME_TIMES_S) - 1

    @pytest.mark.parametrize("bad", [float("nan"), -0.1, 1.1])
    def test_scrub_refuses_a_fraction_outside_the_unit_interval(
        self, reader, bad: float
    ) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="fraction"):
            ParticlePlayback(reader).scrub(bad)

    def test_seek_frame_refuses_an_index_outside_the_trajectory(self, reader) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="frame"):
            ParticlePlayback(reader).seek_frame(len(FRAME_TIMES_S))

    def test_seek_time_refuses_a_non_finite_time(self, reader) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="finite"):
            ParticlePlayback(reader).seek_time(float("inf"))


class TestDecimationIsDisplayOnly:
    """Fewer glyphs, the same numbers -- and the picture says so."""

    def test_decimation_is_deterministic_in_roster_order(self) -> None:
        decimation = DisplayDecimation.for_count(10, max_glyphs=4)
        np.testing.assert_array_equal(decimation.rows, [0, 3, 6, 9])
        assert decimation == DisplayDecimation.for_count(10, max_glyphs=4)

    def test_no_decimation_carries_no_decimation_label(self) -> None:
        decimation = DisplayDecimation.for_count(6, max_glyphs=6)
        assert not decimation.active
        assert "decimation" not in decimation.label()
        assert "6 of 6" in decimation.label()

    def test_active_decimation_is_labelled_with_counts(self) -> None:
        label = DisplayDecimation.for_count(10, max_glyphs=4).label()
        assert "display decimation" in label
        assert "4 of 10" in label

    @pytest.mark.parametrize("bad", [0, -3])
    def test_a_non_positive_glyph_budget_is_refused(self, bad: int) -> None:
        with pytest.raises(ValueError, match="max_glyphs"):
            DisplayDecimation.for_count(10, max_glyphs=bad)

    def test_decimated_values_are_the_undecimated_values(self, reader) -> None:  # type: ignore[no-untyped-def]
        full = ParticlePlayback(reader)
        thin = ParticlePlayback(reader, max_glyphs=2)
        for index in range(len(FRAME_TIMES_S)):
            full.seek_frame(index)
            thin.seek_frame(index)
            whole, sparse = full.view(), thin.view()
            assert sparse.grain_ids.size == 2
            for grain_id, position in zip(
                sparse.grain_ids, sparse.positions_m, strict=True
            ):
                row = int(np.flatnonzero(whole.grain_ids == grain_id)[0])
                np.testing.assert_array_equal(position, whole.positions_m[row])

    def test_the_stamp_carries_the_decimation_label(self, reader) -> None:  # type: ignore[no-untyped-def]
        stamp = "\n".join(ParticlePlayback(reader, max_glyphs=2).view().stamp_lines)
        assert "display decimation" in stamp
        assert "of 6" in stamp


class TestBodiesAreSynchronisedByTime:
    """Club and ball are placed at the particle frame's simulation time."""

    def test_the_club_is_interpolated_at_the_frame_time(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader, bodies=(_club(),))
        playback.seek_frame(2)
        club = playback.view().body("club")
        assert club.position_m is not None
        np.testing.assert_allclose(club.position_m, [2.5e-4 * 30.0, 0.0, 0.01])

    def test_no_ball_before_release(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader, bodies=(_ball(),))
        playback.seek_frame(1)
        ball = playback.view().body("ball")
        assert ball.position_m is None
        assert "no data" in ball.status

    def test_a_missing_sample_is_not_bridged(self, reader) -> None:  # type: ignore[no-untyped-def]
        playback = ParticlePlayback(reader, bodies=(_ball(),))
        playback.seek_frame(3)  # 3.0e-4 s, between 2.6e-4 and the NaN at 3.4e-4
        assert playback.view().body("ball").position_m is None

    def test_an_exact_sample_is_returned_unchanged(self) -> None:
        track = _ball()
        np.testing.assert_array_equal(track.position_at(5.0e-4), [0.004, 0.0, 0.008])

    def test_the_body_source_is_stamped(self, reader) -> None:  # type: ignore[no-untyped-def]
        held = BodyTrack(
            "ball",
            np.array([0.0, 1.0e-3]),
            np.zeros((2, 3)),
            BodySource.HELD,
        )
        view = ParticlePlayback(reader, bodies=(held,)).view()
        assert "held" in view.body("ball").status
        assert any("ball" in line for line in view.stamp_lines)

    def test_body_names_must_be_unique(self, reader) -> None:  # type: ignore[no-untyped-def]
        with pytest.raises(ValueError, match="unique"):
            ParticlePlayback(reader, bodies=(_club(), _club()))

    def test_body_times_must_increase(self) -> None:
        with pytest.raises(ValueError, match="increasing"):
            BodyTrack(
                "club", np.array([0.0, 0.0]), np.zeros((2, 3)), BodySource.SIMULATED
            )

    def test_body_positions_must_match_the_times(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            BodyTrack(
                "club", np.array([0.0, 1.0]), np.zeros((3, 3)), BodySource.SIMULATED
            )


class TestTheStampTellsTheTruth:
    """Tier, verdict, source and particle kind are in every frame."""

    def test_the_stamp_names_the_fixture_and_the_verdict(self, reader) -> None:  # type: ignore[no-untyped-def]
        stamp = "\n".join(ParticlePlayback(reader).view().stamp_lines)
        assert SYNTHETIC_SOLVER_NAME in stamp
        assert "BEYOND VALIDATION" in stamp
        assert "F3" in stamp

    def test_material_points_are_not_called_grains(self, tmp_path: Path) -> None:
        path = write_synthetic_artifact(
            tmp_path / "mp.h5", kind=ParticleKind.MATERIAL_POINT
        )
        with ParticleTrajectoryReader(path) as opened:
            stamp = "\n".join(ParticlePlayback(opened).view().stamp_lines)
        assert "material points" in stamp
        assert "not solved grains" in stamp

    def test_the_view_arrays_are_read_only(self, reader) -> None:  # type: ignore[no-untyped-def]
        view = ParticlePlayback(reader).view()
        with pytest.raises(ValueError):
            view.positions_m[0, 0] = 1.0
        assert view.grain_ids.size == GRAIN_IDS.size


class TestFrameRateMeasurement:
    """The fps hook times real calls and reports median and tail."""

    def test_the_report_is_computed_from_measured_intervals(self) -> None:
        ticks = iter([0.0, 0.010, 0.010, 0.030, 0.030, 0.040, 0.040, 0.090])
        rendered: list[int] = []
        report = measure_frame_rate(
            rendered.append, [0, 1, 2, 3], clock=lambda: next(ticks)
        )
        assert rendered == [0, 1, 2, 3]
        assert isinstance(report, FrameRateReport)
        assert report.n_frames == 4
        assert report.median_frame_s == pytest.approx(0.015)
        assert report.max_frame_s == pytest.approx(0.050)
        assert report.p95_frame_s == pytest.approx(0.050)
        assert report.median_fps == pytest.approx(1.0 / 0.015)

    def test_an_empty_measurement_is_refused(self) -> None:
        with pytest.raises(ValueError, match="frame"):
            measure_frame_rate(lambda _: None, [])
