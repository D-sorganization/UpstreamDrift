"""BunkerShot3D result schema v2 (issue #8617, findings B17/B18).

v1 wrote one HDF5 *group per timestep* holding 3-element datasets, and read it
back with ``sorted(grp.keys())`` over ``f"t_{time:.6f}"`` strings -- so the
ordering broke lexicographically at t >= 10 s ("t_9.900000" > "t_10.000000").

v2 stores contiguous chunked arrays, an explicit integer ``schema_version``
written first and read first, and a run manifest. The reader accepts v1 and v2
and migrates v1 *on read* -- never in place.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import h5py
import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bunkershot3d.io.schema import (
    SCHEMA_VERSION,
    SCHEMA_VERSION_ATTR,
    BunkerShotResultReader,
    BunkerShotResultWriter,
)
from bunkershot3d.provenance import (
    PROVENANCE_SUFFIX,
    RunManifest,
    Validity,
    root_seed_sequence,
    seed_record,
)

pytestmark = pytest.mark.unit

_COUNTER = itertools.count()


def _manifest() -> RunManifest:
    return RunManifest(
        config_hash="a" * 64,
        physics_hash="b" * 64,
        seeds=(seed_record(root_seed_sequence(entropy=5), "grains"),),
        solver="drft",
        fidelity_tier="F0",
        validity=Validity.VALID,
    )


def _write_v1_file(path: Path, times: np.ndarray) -> dict[str, np.ndarray]:
    """Write a legacy (v1) group-per-timestep file the way the old writer did."""
    rng = np.random.default_rng(0)
    positions = rng.standard_normal((times.size, 3))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (times.size, 1))
    forces = rng.standard_normal((times.size, 3))
    torques = rng.standard_normal((times.size, 3))
    grains = [rng.standard_normal((2, 3)) for _ in times]

    with h5py.File(path, "w") as handle:
        club = handle.create_group("clubhead")
        wrench = handle.create_group("wrench")
        grain = handle.create_group("grains")
        for i, time in enumerate(times):
            key = f"t_{time:.6f}"
            sub = club.create_group(key)
            sub.attrs["time"] = time
            sub.create_dataset("position", data=positions[i])
            sub.create_dataset("orientation", data=quats[i])

            wsub = wrench.create_group(key)
            wsub.attrs["time"] = time
            wsub.create_dataset("force", data=forces[i])
            wsub.create_dataset("torque", data=torques[i])

            gsub = grain.create_group(key)
            gsub.attrs["time"] = time
            gsub.create_dataset("positions", data=grains[i])
            gsub.create_dataset("velocities", data=grains[i] * 0.5)
    return {
        "times": times,
        "positions": positions,
        "quats": quats,
        "forces": forces,
        "torques": torques,
    }


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------


def test_writer_stamps_integer_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "v2.h5"
    with BunkerShotResultWriter(path):
        pass
    with h5py.File(path, "r") as handle:
        version = handle.attrs[SCHEMA_VERSION_ATTR]
    # v3 added the sand-field payload of issue #8710; the streams below are
    # unchanged, which is why every other case in this module still stands.
    assert int(version) == SCHEMA_VERSION == 3
    assert not isinstance(version, (str, bytes)), "version must be an integer"


def test_schema_version_is_the_first_root_attribute(tmp_path: Path) -> None:
    """Written first so a reader can dispatch before touching anything else."""
    path = tmp_path / "first.h5"
    writer = BunkerShotResultWriter(path, manifest=_manifest())
    writer.write_clubhead_state(0.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    writer.close()
    with h5py.File(path, "r") as handle:
        assert next(iter(handle.attrs)) == SCHEMA_VERSION_ATTR


def test_reader_reports_file_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "v2.h5"
    with BunkerShotResultWriter(path):
        pass
    with BunkerShotResultReader(path) as reader:
        assert reader.schema_version == SCHEMA_VERSION


def test_reader_rejects_future_schema_version(tmp_path: Path) -> None:
    path = tmp_path / "future.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs[SCHEMA_VERSION_ATTR] = SCHEMA_VERSION + 99
    with pytest.raises(ValueError, match="schema version"):
        BunkerShotResultReader(path)


def test_reader_treats_a_versionless_file_as_v1(tmp_path: Path) -> None:
    path = tmp_path / "legacy.h5"
    _write_v1_file(path, np.array([0.0, 0.001]))
    with BunkerShotResultReader(path) as reader:
        assert reader.schema_version == 1


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def test_v2_stores_contiguous_chunked_arrays_not_groups(tmp_path: Path) -> None:
    path = tmp_path / "layout.h5"
    writer = BunkerShotResultWriter(path, time_chunk=4)
    for i in range(10):
        writer.write_clubhead_state(
            i * 1e-3, np.full(3, float(i)), np.array([1.0, 0.0, 0.0, 0.0])
        )
    writer.close()

    with h5py.File(path, "r") as handle:
        club = handle["clubhead"]
        assert set(club.keys()) == {"t", "position", "orientation"}
        assert isinstance(club["position"], h5py.Dataset)
        assert club["t"].shape == (10,)
        assert club["position"].shape == (10, 3)
        assert club["orientation"].shape == (10, 4)
        assert club["position"].chunks is not None
        assert club["position"].chunks[0] == 4, "chunked along time"


def test_empty_result_reads_back_as_empty_arrays(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    with BunkerShotResultWriter(path):
        pass
    with BunkerShotResultReader(path) as reader:
        times, positions, quats = reader.read_clubhead_states()
        assert times.shape == (0,)
        assert positions.shape == (0, 3)
        assert quats.shape == (0, 4)
        wtimes, forces, torques = reader.read_contact_wrenches()
        assert wtimes.shape == (0,)
        assert forces.shape == (0, 3)
        assert torques.shape == (0, 3)
        gtimes, gpos, gvel = reader.read_grain_states()
        assert gtimes.shape == (0,)
        assert gpos == [] and gvel == []


def test_writer_rejects_time_going_backwards(tmp_path: Path) -> None:
    writer = BunkerShotResultWriter(tmp_path / "back.h5")
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    writer.write_clubhead_state(1.0, np.zeros(3), quat)
    with pytest.raises(ValueError, match="non-decreasing"):
        writer.write_clubhead_state(0.5, np.zeros(3), quat)
    writer.close()


def test_writer_snapshots_mutable_solver_buffers(tmp_path: Path) -> None:
    """Drivers pass live views (``mjData.xpos[body]``) that they then overwrite."""
    path = tmp_path / "aliased.h5"
    live_position = np.zeros(3)
    live_force = np.zeros(3)
    live_grains = np.zeros((2, 3))
    writer = BunkerShotResultWriter(path, time_chunk=64)
    for step in range(4):
        live_position[:] = float(step)
        live_force[:] = float(step)
        live_grains[:] = float(step)
        writer.write_clubhead_state(
            step * 1e-3, live_position, np.array([1.0, 0.0, 0.0, 0.0])
        )
        writer.write_contact_wrench(step * 1e-3, live_force, live_force)
        writer.write_grain_state(step * 1e-3, live_grains, live_grains)
    writer.close()

    with BunkerShotResultReader(path) as reader:
        _, positions, _ = reader.read_clubhead_states()
        _, forces, _ = reader.read_contact_wrenches()
        _, grains, _ = reader.read_grain_states()

    np.testing.assert_allclose(positions[:, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose(forces[:, 0], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_allclose([frame[0, 0] for frame in grains], [0.0, 1.0, 2.0, 3.0])


def test_writer_rejects_wrong_shapes(tmp_path: Path) -> None:
    writer = BunkerShotResultWriter(tmp_path / "shape.h5")
    with pytest.raises(ValueError, match="shape"):
        writer.write_clubhead_state(0.0, np.zeros(2), np.zeros(4))
    with pytest.raises(ValueError, match="shape"):
        writer.write_contact_wrench(0.0, np.zeros(3), np.zeros(4))
    writer.close()


# ---------------------------------------------------------------------------
# Time ordering -- the exact case the v1 string sort broke on
# ---------------------------------------------------------------------------

_LONG_RUN_TIMES = np.array([9.5, 9.999, 10.0, 10.5, 11.0, 100.0, 101.25])


def test_v2_time_order_is_correct_past_ten_seconds(tmp_path: Path) -> None:
    path = tmp_path / "long_v2.h5"
    writer = BunkerShotResultWriter(path)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    for i, time in enumerate(_LONG_RUN_TIMES):
        writer.write_clubhead_state(float(time), np.full(3, float(i)), quat)
        writer.write_contact_wrench(float(time), np.full(3, float(i)), np.zeros(3))
    writer.close()

    with BunkerShotResultReader(path) as reader:
        times, positions, _ = reader.read_clubhead_states()
        wtimes, forces, _ = reader.read_contact_wrenches()

    np.testing.assert_allclose(times, _LONG_RUN_TIMES)
    assert np.all(np.diff(times) > 0)
    np.testing.assert_allclose(positions[:, 0], np.arange(_LONG_RUN_TIMES.size))
    np.testing.assert_allclose(wtimes, _LONG_RUN_TIMES)
    np.testing.assert_allclose(forces[:, 0], np.arange(_LONG_RUN_TIMES.size))


def test_v1_time_order_is_corrected_on_read_past_ten_seconds(tmp_path: Path) -> None:
    """Lexicographic key order puts "t_10.000000" before "t_9.500000"."""
    path = tmp_path / "long_v1.h5"
    ref = _write_v1_file(path, _LONG_RUN_TIMES)

    with h5py.File(path, "r") as handle:
        lexicographic = sorted(handle["clubhead"].keys())
    assert lexicographic[0] == "t_10.000000", "precondition: string sort is wrong"

    with BunkerShotResultReader(path) as reader:
        times, positions, _ = reader.read_clubhead_states()
        wtimes, forces, _ = reader.read_contact_wrenches()
        gtimes, _, _ = reader.read_grain_states()

    np.testing.assert_allclose(times, ref["times"])
    np.testing.assert_allclose(positions, ref["positions"])
    np.testing.assert_allclose(wtimes, ref["times"])
    np.testing.assert_allclose(forces, ref["forces"])
    np.testing.assert_allclose(gtimes, ref["times"])


def test_v1_file_reads_correctly(tmp_path: Path) -> None:
    path = tmp_path / "v1.h5"
    ref = _write_v1_file(path, np.array([0.0, 0.0005, 0.001]))
    with BunkerShotResultReader(path) as reader:
        times, positions, quats = reader.read_clubhead_states()
        _, forces, torques = reader.read_contact_wrenches()
        gtimes, gpos, gvel = reader.read_grain_states()

    np.testing.assert_allclose(times, ref["times"])
    np.testing.assert_allclose(positions, ref["positions"])
    np.testing.assert_allclose(quats, ref["quats"])
    np.testing.assert_allclose(forces, ref["forces"])
    np.testing.assert_allclose(torques, ref["torques"])
    assert gtimes.shape == (3,)
    assert len(gpos) == 3 and gpos[0].shape == (2, 3)
    assert len(gvel) == 3


def test_v1_migration_does_not_rewrite_the_source_file(tmp_path: Path) -> None:
    path = tmp_path / "readonly.h5"
    _write_v1_file(path, np.array([0.0, 0.001]))
    before = path.read_bytes()
    with BunkerShotResultReader(path) as reader:
        reader.read_clubhead_states()
    assert path.read_bytes() == before, "migration must happen on read, not in place"


# ---------------------------------------------------------------------------
# Grain state (ragged: particle count may vary between frames)
# ---------------------------------------------------------------------------


def test_grain_states_round_trip_with_varying_particle_counts(tmp_path: Path) -> None:
    path = tmp_path / "grains.h5"
    rng = np.random.default_rng(3)
    frames = [rng.standard_normal((n, 3)) for n in (5, 2, 7)]
    writer = BunkerShotResultWriter(path)
    for i, positions in enumerate(frames):
        writer.write_grain_state(i * 1e-3, positions, positions * 2.0)
    writer.close()

    with BunkerShotResultReader(path) as reader:
        times, positions_out, velocities_out = reader.read_grain_states()

    np.testing.assert_allclose(times, [0.0, 1e-3, 2e-3])
    assert [p.shape for p in positions_out] == [(5, 3), (2, 3), (7, 3)]
    for expected, got in zip(frames, positions_out, strict=True):
        np.testing.assert_allclose(got, expected)
    for expected, got in zip(frames, velocities_out, strict=True):
        np.testing.assert_allclose(got, expected * 2.0)


# ---------------------------------------------------------------------------
# Round-trip property
# ---------------------------------------------------------------------------

_ELEMENTS = st.floats(
    min_value=-1e4, max_value=1e4, allow_nan=False, allow_infinity=False, width=64
)


@st.composite
def _traces(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = draw(st.integers(min_value=1, max_value=40))
    start = draw(st.floats(min_value=0.0, max_value=50.0, allow_nan=False))
    step = draw(st.floats(min_value=1e-4, max_value=1.0, allow_nan=False))
    times = start + step * np.arange(n, dtype=float)
    flat = draw(st.lists(_ELEMENTS, min_size=n * 10, max_size=n * 10))
    block = np.asarray(flat, dtype=float).reshape(n, 10)
    return times, block[:, 0:3], block[:, 3:7], block[:, 7:10]


@given(payload=_traces())
@settings(
    deadline=None,
    max_examples=25,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_round_trip_preserves_arrays_exactly(
    tmp_path_factory: pytest.TempPathFactory,
    payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    times, positions, quats, forces = payload
    path = tmp_path_factory.mktemp("rt") / f"case_{next(_COUNTER)}.h5"

    writer = BunkerShotResultWriter(path, time_chunk=8)
    for i, time in enumerate(times):
        writer.write_clubhead_state(float(time), positions[i], quats[i])
        writer.write_contact_wrench(float(time), forces[i], positions[i])
        writer.write_grain_state(float(time), positions[: i + 1], forces[: i + 1])
    writer.close()

    with BunkerShotResultReader(path) as reader:
        t_out, p_out, q_out = reader.read_clubhead_states()
        wt_out, f_out, tq_out = reader.read_contact_wrenches()
        gt_out, gp_out, gv_out = reader.read_grain_states()

    np.testing.assert_array_equal(t_out, times)
    np.testing.assert_array_equal(p_out, positions)
    np.testing.assert_array_equal(q_out, quats)
    np.testing.assert_array_equal(wt_out, times)
    np.testing.assert_array_equal(f_out, forces)
    np.testing.assert_array_equal(tq_out, positions)
    np.testing.assert_array_equal(gt_out, times)
    assert len(gp_out) == times.size
    for i, frame in enumerate(gp_out):
        np.testing.assert_array_equal(frame, positions[: i + 1])
    for i, frame in enumerate(gv_out):
        np.testing.assert_array_equal(frame, forces[: i + 1])


# ---------------------------------------------------------------------------
# Manifest integration
# ---------------------------------------------------------------------------


def test_writer_persists_the_manifest_as_attrs_and_sidecar(tmp_path: Path) -> None:
    path = tmp_path / "manifested.h5"
    writer = BunkerShotResultWriter(path, manifest=_manifest())
    writer.write_clubhead_state(0.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    writer.close()

    with BunkerShotResultReader(path) as reader:
        manifest = reader.manifest
    assert manifest is not None
    assert manifest.config_hash == "a" * 64
    assert manifest.seeds[0].entropy == 5
    assert manifest.wall_clock_s > 0.0, "writer records its own wall clock"

    sidecar = path.parent / f"{path.name}{PROVENANCE_SUFFIX}"
    assert sidecar.is_file()


def test_reader_manifest_is_none_when_not_supplied(tmp_path: Path) -> None:
    path = tmp_path / "bare.h5"
    with BunkerShotResultWriter(path):
        pass
    with BunkerShotResultReader(path) as reader:
        assert reader.manifest is None
    assert not (tmp_path / f"bare.h5{PROVENANCE_SUFFIX}").exists()
