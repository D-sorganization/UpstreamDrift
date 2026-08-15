"""``read_bunkershot3d_result`` across BunkerShot3D schema v1 and v2 (#8617).

The unified-trace importer is the live downstream consumer of the BunkerShot3D
result file. It must keep reading legacy v1 (group-per-timestep) files *and*
read the new v2 contiguous-array layout, with correct time ordering past the
t >= 10 s point where the v1 lexicographic key sort broke.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from bunkershot3d.io.schema import BunkerShotResultWriter
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.trace_io import read_bunkershot3d_result

pytestmark = pytest.mark.unit

_TIMES = np.array([9.5, 9.999, 10.0, 10.5, 11.0])


def _payload() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(11)
    n = _TIMES.size
    return {
        "times": _TIMES,
        "positions": rng.standard_normal((n, 3)),
        "quats": np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        "forces": rng.standard_normal((n, 3)),
        "torques": rng.standard_normal((n, 3)),
    }


def _write_v1(path: Path, data: dict[str, np.ndarray]) -> None:
    with h5py.File(path, "w") as handle:
        club = handle.create_group("clubhead")
        wrench = handle.create_group("wrench")
        for i, time in enumerate(data["times"]):
            key = f"t_{time:.6f}"
            sub = club.create_group(key)
            sub.attrs["time"] = time
            sub.create_dataset("position", data=data["positions"][i])
            sub.create_dataset("orientation", data=data["quats"][i])
            wsub = wrench.create_group(key)
            wsub.attrs["time"] = time
            wsub.create_dataset("force", data=data["forces"][i])
            wsub.create_dataset("torque", data=data["torques"][i])


def _write_v2(path: Path, data: dict[str, np.ndarray]) -> None:
    writer = BunkerShotResultWriter(path)
    for i, time in enumerate(data["times"]):
        writer.write_clubhead_state(float(time), data["positions"][i], data["quats"][i])
        writer.write_contact_wrench(float(time), data["forces"][i], data["torques"][i])
    writer.close()


@pytest.mark.parametrize("writer", [_write_v1, _write_v2], ids=["v1", "v2"])
def test_reads_both_schema_versions_into_a_trace(tmp_path: Path, writer) -> None:
    data = _payload()
    path = tmp_path / "result.h5"
    writer(path, data)

    trace = read_bunkershot3d_result(path)

    assert isinstance(trace, Trace)
    assert trace.backend == "bunkershot3d"
    np.testing.assert_allclose(trace.t, data["times"])
    assert trace.markers is not None
    assert trace.markers.shape == (data["times"].size, 1, 3)
    np.testing.assert_allclose(trace.markers[:, 0, :], data["positions"])
    assert trace.wrench is not None
    assert trace.wrench.shape == (data["times"].size, 6)
    np.testing.assert_allclose(trace.wrench[:, :3], data["forces"])
    np.testing.assert_allclose(trace.wrench[:, 3:], data["torques"])
    assert trace.q.shape == (data["times"].size, 0)
    assert trace.v.shape == (data["times"].size, 0)


@pytest.mark.parametrize("writer", [_write_v1, _write_v2], ids=["v1", "v2"])
def test_time_is_increasing_past_ten_seconds(tmp_path: Path, writer) -> None:
    data = _payload()
    path = tmp_path / "long.h5"
    writer(path, data)

    trace = read_bunkershot3d_result(path)

    assert np.all(np.diff(trace.t) > 0.0)
    np.testing.assert_allclose(trace.t, np.sort(data["times"]))


def test_missing_clubhead_group_still_raises(tmp_path: Path) -> None:
    path = tmp_path / "not_bunkershot.h5"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("something_else", data=[1.0])
    with pytest.raises(ValueError, match="clubhead"):
        read_bunkershot3d_result(path)


def test_result_without_wrench_yields_none(tmp_path: Path) -> None:
    path = tmp_path / "no_wrench.h5"
    writer = BunkerShotResultWriter(path)
    writer.write_clubhead_state(0.0, np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))
    writer.close()

    trace = read_bunkershot3d_result(path)

    assert trace.wrench is None
    assert trace.markers is not None and trace.markers.shape == (1, 1, 3)


def test_mismatched_wrench_length_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "ragged.h5"
    writer = BunkerShotResultWriter(path)
    quat = np.array([1.0, 0.0, 0.0, 0.0])
    writer.write_clubhead_state(0.0, np.zeros(3), quat)
    writer.write_clubhead_state(0.001, np.zeros(3), quat)
    writer.write_contact_wrench(0.0, np.zeros(3), np.zeros(3))
    writer.close()

    with pytest.raises(ValueError, match="wrench"):
        read_bunkershot3d_result(path)


def test_dt_is_the_mean_sample_interval(tmp_path: Path) -> None:
    data = _payload()
    path = tmp_path / "dt.h5"
    _write_v2(path, data)

    trace = read_bunkershot3d_result(path)

    assert trace.dt == pytest.approx(float(np.mean(np.diff(data["times"]))))
