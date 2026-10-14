"""Versioned 3-D particle-trajectory artifact (issue #9545, folded-in #9692).

The artifact is only worth having if a reader can trust three things
without looking at a number: that grain 17 in frame 40 is the same grain
as grain 17 in frame 0, that the file it is reading was finished, and
that nothing in it was changed after it was written.  Each test class
below attacks one of those claims and asks the reader what it thinks.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py", reason="the trajectory artifact is HDF5")

from bunkershot3d.exceptions import BunkerShot3DValueError  # noqa: E402
from bunkershot3d.fields.schema import GrainRoster, ParticleKind  # noqa: E402
from bunkershot3d.fields.standing import (  # noqa: E402
    FieldIntegrityError,
    RetentionPolicy,
    RetentionRecord,
)
from bunkershot3d.fields.trajectory import (  # noqa: E402
    TRAJECTORY_SCHEMA_VERSION,
    ContactEvent,
    ParticleTrajectoryWriter,
    TrajectoryIncompleteError,
    TrajectoryVersionError,
    final_marker_path,
)
from bunkershot3d.fields.trajectory_reader import (  # noqa: E402
    ParticleTrajectoryReader,
)
from tests.bunkershot3d.fields.test_field_schema import provenance  # noqa: E402

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

#: Deliberately unsorted, so an index-for-id mix-up cannot pass by luck.
_IDS = np.array([17, 4, 9], dtype=np.int64)
_RADII = np.array([2.3e-4, 2.0e-4, 2.6e-4])
#: Irregular retained cadence: the gaps are 100, 200 and 50 microseconds.
_TIMES = (0.0, 1.0e-4, 3.0e-4, 3.5e-4)


def roster(**overrides: object) -> GrainRoster:
    """The three-grain roster used throughout."""
    values: dict[str, object] = {
        "grain_ids": _IDS,
        "radius_m": _RADII,
        "kind": ParticleKind.DISCRETE_GRAIN,
    }
    values.update(overrides)
    return GrainRoster(**values)  # type: ignore[arg-type]


def _frame_state(
    index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Positions, velocities and masks of the crossing fixture at one frame.

    Grain 17 runs +x and grain 4 runs -x along the same line, so they
    cross between frames 1 and 2: a reader that tracked identity by
    position (nearest neighbour) would swap them.  Grain 9 is inactive
    in frame 0 and appears in frame 1; grain 4 escapes the domain in
    frame 3 and is lost thereafter.
    """
    s = _TIMES[index] / _TIMES[-1]
    positions = np.array(
        [
            [-0.01 + 0.02 * s, 0.0, 0.005],
            [0.01 - 0.02 * s, 0.0, 0.005],
            [0.0, 0.003, 0.004],
        ]
    )
    velocity = np.array([[2.0, 0.0, 0.1], [-2.0, 0.0, 0.1], [0.0, 0.5, 0.0]])
    active = np.array([True, index < 3, index >= 1])
    lost = np.array([False, index >= 3, False])
    return positions, velocity, active, lost


def write_fixture(path: Path, *, rows_per_chunk: int = 4096) -> Path:
    """Write the crossing fixture, with two contact events, and finalise it."""
    with ParticleTrajectoryWriter(
        path, roster(), provenance(), rows_per_chunk=rows_per_chunk
    ) as writer:
        for index, time_s in enumerate(_TIMES):
            writer.append_frame(time_s, *_frame_state(index))
        writer.append_contact(
            ContactEvent(
                time_s=1.0e-4,
                grain_id=17,
                body_id=1,
                point_m=np.array([0.0, 0.0, 0.005]),
                force_n=np.array([0.2, 0.0, 0.0]),
            )
        )
        writer.append_contact(
            ContactEvent(
                time_s=3.0e-4,
                grain_id=9,
                body_id=1,
                point_m=np.array([0.0, 0.003, 0.004]),
                force_n=np.full(3, np.nan),
            )
        )
    return path


def _marker(path: Path) -> dict:
    return json.loads(final_marker_path(path).read_text(encoding="utf-8"))


def _rewrite_marker(path: Path, marker: dict) -> None:
    final_marker_path(path).write_text(json.dumps(marker), encoding="utf-8")


def _redigest(path: Path, dataset: str) -> None:
    """Re-sign one array in the marker, as a careless tool would."""
    with h5py.File(path, "r") as handle:
        data = np.ascontiguousarray(handle[dataset][...])
    marker = _marker(path)
    marker["array_sha256"][dataset] = hashlib.sha256(data.tobytes()).hexdigest()
    _rewrite_marker(path, marker)


@pytest.fixture
def stored(tmp_path: Path) -> Path:
    return write_fixture(tmp_path / "shot.traj.h5")


class TestGrainRoster:
    """The roster is the identity contract; it refuses ambiguous identity."""

    def test_a_valid_roster_is_immutable(self) -> None:
        made = roster()
        assert made.n_grains == 3
        with pytest.raises(ValueError):
            made.grain_ids[0] = 99

    def test_duplicate_ids_are_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="duplicate"):
            roster(grain_ids=np.array([17, 4, 17]))

    @pytest.mark.parametrize("bad", [0.0, -1.0e-4, np.nan, np.inf])
    def test_nonpositive_or_nonfinite_radii_are_refused(self, bad: float) -> None:
        with pytest.raises(BunkerShot3DValueError, match="radius"):
            roster(radius_m=np.array([2.0e-4, bad, 2.0e-4]))

    def test_mismatched_lengths_are_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="one radius per grain"):
            roster(radius_m=np.array([2.0e-4, 2.0e-4]))

    def test_an_empty_roster_is_refused(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="at least one"):
            roster(grain_ids=np.array([], dtype=np.int64), radius_m=np.array([]))


class TestRoundTrip:
    """What goes in comes back, attached to the same grain."""

    def test_identity_survives_the_crossing(self, stored: Path) -> None:
        with ParticleTrajectoryReader(stored) as reader:
            track_17 = reader.positions_of(17)
            track_4 = reader.positions_of(4)
        for index in range(3):
            expected = _frame_state(index)[0]
            np.testing.assert_array_equal(track_17[index], expected[0])
            np.testing.assert_array_equal(track_4[index], expected[1])
        # Grain 17 ends on the +x side it was heading for.
        assert track_17[-1, 0] > 0.0

    def test_irregular_times_and_metadata_round_trip(self, stored: Path) -> None:
        with ParticleTrajectoryReader(stored) as reader:
            np.testing.assert_array_equal(reader.time_s, np.asarray(_TIMES))
            assert reader.n_frames == len(_TIMES)
            assert reader.roster.kind is ParticleKind.DISCRETE_GRAIN
            np.testing.assert_array_equal(reader.roster.grain_ids, _IDS)
            np.testing.assert_array_equal(reader.roster.radius_m, _RADII)
            assert reader.provenance == provenance()
            assert reader.schema_version == TRAJECTORY_SCHEMA_VERSION

    def test_inactive_and_lost_grains_are_unavailable_not_zero(
        self, stored: Path
    ) -> None:
        with ParticleTrajectoryReader(stored) as reader:
            first = reader.frame(0)
            last = reader.frame(3)
        assert not first.active[2]
        assert np.all(np.isnan(first.positions_m[2]))
        assert np.all(np.isnan(first.velocity_m_s[2]))
        assert last.lost[1] and not last.active[1]
        assert np.all(np.isnan(last.positions_m[1]))

    def test_contact_events_round_trip(self, stored: Path) -> None:
        with ParticleTrajectoryReader(stored) as reader:
            events = reader.contacts()
        assert [event.grain_id for event in events] == [17, 9]
        np.testing.assert_array_equal(events[0].force_n, [0.2, 0.0, 0.0])
        assert np.all(np.isnan(events[1].force_n))

    def test_the_marker_is_written_last_and_names_the_artifact(
        self, stored: Path
    ) -> None:
        marker = _marker(stored)
        assert marker["artifact"] == stored.name
        assert marker["trajectory_schema_version"] == TRAJECTORY_SCHEMA_VERSION
        assert marker["n_frames"] == len(_TIMES)
        assert marker["n_grains"] == 3
        assert marker["n_contacts"] == 2
        assert final_marker_path(stored).stat().st_mtime_ns >= (
            stored.stat().st_mtime_ns
        )


class TestWriterPreconditions:
    """The writer refuses states that would read back ambiguously."""

    @pytest.fixture
    def writer(self, tmp_path: Path):
        made = ParticleTrajectoryWriter(tmp_path / "w.h5", roster(), provenance())
        made.append_frame(0.0, *_frame_state(0))
        yield made
        made.abort()

    def test_backward_or_repeated_time_is_refused(self, writer) -> None:
        with pytest.raises(BunkerShot3DValueError, match="increasing"):
            writer.append_frame(0.0, *_frame_state(1))

    def test_bad_dimensions_are_refused(self, writer) -> None:
        positions, velocity, active, lost = _frame_state(1)
        with pytest.raises(BunkerShot3DValueError, match="shape"):
            writer.append_frame(1.0e-4, positions[:, :2], velocity, active, lost)

    def test_nonfinite_active_state_is_refused(self, writer) -> None:
        positions, velocity, active, lost = _frame_state(1)
        positions[0, 1] = np.nan
        with pytest.raises(BunkerShot3DValueError, match="finite"):
            writer.append_frame(1.0e-4, positions, velocity, active, lost)

    def test_a_grain_cannot_be_both_active_and_lost(self, writer) -> None:
        positions, velocity, active, lost = _frame_state(1)
        lost[0] = True
        with pytest.raises(BunkerShot3DValueError, match="active and lost"):
            writer.append_frame(1.0e-4, positions, velocity, active, lost)

    def test_a_lost_grain_cannot_return(self, tmp_path: Path) -> None:
        writer = ParticleTrajectoryWriter(tmp_path / "r.h5", roster(), provenance())
        try:
            for index in range(4):
                writer.append_frame(_TIMES[index], *_frame_state(index))
            positions, velocity, active, lost = _frame_state(1)
            with pytest.raises(BunkerShot3DValueError, match="lost"):
                writer.append_frame(1.0, positions, velocity, active, lost)
        finally:
            writer.abort()

    def test_contact_with_an_unknown_grain_is_refused(self, writer) -> None:
        event = ContactEvent(
            time_s=0.0,
            grain_id=12345,
            body_id=1,
            point_m=np.zeros(3),
            force_n=np.zeros(3),
        )
        with pytest.raises(BunkerShot3DValueError, match="roster"):
            writer.append_contact(event)

    def test_finalising_with_no_frames_is_refused(self, tmp_path: Path) -> None:
        writer = ParticleTrajectoryWriter(tmp_path / "e.h5", roster(), provenance())
        with pytest.raises(BunkerShot3DValueError, match="no frames"):
            writer.finalize()
        writer.abort()

    def test_a_retention_record_must_match_what_was_kept(self, writer) -> None:
        record = RetentionRecord(
            policy=RetentionPolicy(target_frames=4),
            steps_marched=40,
            time_stride=10,
            frames_kept=4,
            time_step_s=1.0e-5,
            samples_in_domain=3,
            samples_kept=3,
        )
        with pytest.raises(BunkerShot3DValueError, match="frames_kept"):
            writer.finalize(retention=record)

    def test_a_contact_event_refuses_a_zero_filled_missing_point(self) -> None:
        with pytest.raises(BunkerShot3DValueError, match="point_m"):
            ContactEvent(
                time_s=0.0,
                grain_id=17,
                body_id=1,
                point_m=np.array([np.nan, 0.0, 0.0]),
                force_n=np.zeros(3),
            )


class TestAtomicFinalisation:
    """An interrupted or cancelled write is never a completed shot."""

    def test_a_missing_marker_is_refused(self, stored: Path) -> None:
        final_marker_path(stored).unlink()
        with pytest.raises(TrajectoryIncompleteError, match="marker"):
            ParticleTrajectoryReader(stored).open()

    def test_a_truncated_marker_is_refused(self, stored: Path) -> None:
        text = final_marker_path(stored).read_text(encoding="utf-8")
        final_marker_path(stored).write_text(text[: len(text) // 2], "utf-8")
        with pytest.raises(TrajectoryIncompleteError, match="partial"):
            ParticleTrajectoryReader(stored).open()

    def test_a_marker_missing_a_field_is_refused(self, stored: Path) -> None:
        marker = _marker(stored)
        del marker["array_sha256"]
        _rewrite_marker(stored, marker)
        with pytest.raises(TrajectoryIncompleteError, match="array_sha256"):
            ParticleTrajectoryReader(stored).open()

    def test_a_marker_for_another_artifact_is_refused(self, stored: Path) -> None:
        marker = _marker(stored)
        marker["artifact"] = "other.h5"
        _rewrite_marker(stored, marker)
        with pytest.raises(TrajectoryIncompleteError, match="other.h5"):
            ParticleTrajectoryReader(stored).open()

    def test_an_exception_mid_write_leaves_nothing_readable(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "cancelled.h5"
        with pytest.raises(RuntimeError, match="cancelled"):
            with ParticleTrajectoryWriter(path, roster(), provenance()) as writer:
                writer.append_frame(0.0, *_frame_state(0))
                raise RuntimeError("cancelled by the user")
        assert not path.exists()
        assert not final_marker_path(path).exists()
        assert list(tmp_path.iterdir()) == []

    def test_an_unfinished_write_does_not_reach_the_destination(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "running.h5"
        writer = ParticleTrajectoryWriter(path, roster(), provenance())
        writer.append_frame(0.0, *_frame_state(0))
        try:
            assert not path.exists()
            with pytest.raises(TrajectoryIncompleteError):
                ParticleTrajectoryReader(path).open()
        finally:
            writer.abort()

    def test_rewriting_over_a_complete_artifact_revokes_its_marker(
        self, stored: Path
    ) -> None:
        writer = ParticleTrajectoryWriter(stored, roster(), provenance())
        try:
            # The old file is still there, but it is no longer vouched for:
            # a crash now must not leave an old marker beside a new file.
            assert stored.exists()
            with pytest.raises(TrajectoryIncompleteError):
                ParticleTrajectoryReader(stored).open()
        finally:
            writer.abort()


class TestIntegrity:
    """A changed coordinate, label or version is detected, not trusted."""

    def test_a_corrupted_coordinate_is_refused(self, stored: Path) -> None:
        with h5py.File(stored, "r+") as handle:
            handle["frames/positions"][0, 0] += 1.0e-3
        with pytest.raises(FieldIntegrityError, match="frames/positions"):
            ParticleTrajectoryReader(stored).open()

    def test_a_relabelled_verdict_is_refused(self, stored: Path) -> None:
        with h5py.File(stored, "r+") as handle:
            metadata = json.loads(handle.attrs["metadata"])
            metadata["provenance"]["envelope_status"] = "within"
            handle.attrs["metadata"] = json.dumps(metadata)
        with pytest.raises(FieldIntegrityError, match="metadata"):
            ParticleTrajectoryReader(stored).open()

    def test_an_incompatible_marker_version_is_actionable(self, stored: Path) -> None:
        marker = _marker(stored)
        marker["trajectory_schema_version"] = TRAJECTORY_SCHEMA_VERSION + 1
        _rewrite_marker(stored, marker)
        with pytest.raises(TrajectoryVersionError, match="regenerat"):
            ParticleTrajectoryReader(stored).open()

    def test_an_incompatible_file_version_is_refused(self, stored: Path) -> None:
        with h5py.File(stored, "r+") as handle:
            handle.attrs["trajectory_schema_version"] = np.int64(99)
        with pytest.raises(TrajectoryVersionError):
            ParticleTrajectoryReader(stored).open()

    def test_duplicated_grain_ids_are_refused_even_when_re_signed(
        self, stored: Path
    ) -> None:
        with h5py.File(stored, "r+") as handle:
            handle["roster/grain_id"][2] = 17
        _redigest(stored, "roster/grain_id")
        with pytest.raises(FieldIntegrityError, match="duplicate"):
            ParticleTrajectoryReader(stored).open()

    def test_inconsistent_masks_are_refused_even_when_re_signed(
        self, stored: Path
    ) -> None:
        with h5py.File(stored, "r+") as handle:
            handle["frames/lost"][0] = True  # grain 17, frame 0, also active
        _redigest(stored, "frames/lost")
        with pytest.raises(FieldIntegrityError, match="active and lost"):
            ParticleTrajectoryReader(stored).open()

    def test_a_resurrected_grain_is_refused_even_when_re_signed(
        self, stored: Path
    ) -> None:
        n_grains = len(_IDS)
        with h5py.File(stored, "r+") as handle:
            # Grain 4 is lost in the last frame; mark it lost in frame 1 too,
            # so it is lost, then back, then lost again.
            handle["frames/lost"][1 * n_grains + 1] = True
            handle["frames/active"][1 * n_grains + 1] = False
        _redigest(stored, "frames/lost")
        _redigest(stored, "frames/active")
        with pytest.raises(FieldIntegrityError, match="lost"):
            ParticleTrajectoryReader(stored).open()


class TestBoundedStreaming:
    """The writer's memory is fixed by its chunk, not by the run length."""

    def test_an_artifact_larger_than_the_buffer_streams_through_it(
        self, tmp_path: Path
    ) -> None:
        n_grains, n_frames, chunk = 50, 200, 64
        ids = np.arange(1000, 1000 + n_grains, dtype=np.int64)
        big = GrainRoster(
            grain_ids=ids,
            radius_m=np.full(n_grains, 2.0e-4),
            kind=ParticleKind.COARSE_PARTICLE,
        )
        rng = np.random.default_rng(9545)
        path = tmp_path / "big.h5"
        active = np.ones(n_grains, dtype=bool)
        lost = np.zeros(n_grains, dtype=bool)
        with ParticleTrajectoryWriter(
            path, big, provenance(), rows_per_chunk=chunk
        ) as writer:
            for frame in range(n_frames):
                writer.append_frame(
                    frame * 1.0e-5,
                    rng.normal(size=(n_grains, 3)),
                    rng.normal(size=(n_grains, 3)),
                    active,
                    lost,
                )
            peak = writer.peak_buffered_rows
        total_rows = n_grains * n_frames
        assert total_rows > 100 * chunk
        assert peak <= chunk + n_grains
        with ParticleTrajectoryReader(path, rows_per_block=chunk) as reader:
            assert reader.n_frames == n_frames
            assert reader.roster.kind is ParticleKind.COARSE_PARTICLE
            assert reader.frame(n_frames - 1).positions_m.shape == (n_grains, 3)
