"""Tests for the v2 Trace schema extension (CC-4).

Covers:
* Trace v2 optional groups: torques, wrench, markers, contacts.
* Round-trip identity for all optional groups.
* Backward compatibility: v1 HDF5 files auto-migrate through read_trace.
* Migration helper: migrate_from_v1.
* BunkerShot3D import helper: read_bunkershot3d_result.
* Shape validation for the new optional arrays.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest

from src.shared.python.simulation_backends.protocol import (
    SCHEMA_VERSION,
    Trace,
)
from src.shared.python.simulation_backends.trace_io import (
    migrate_from_v1,
    read_bunkershot3d_result,
    read_trace,
    write_trace,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(42)
_T = 11
_NQ = 2
_NU = 2
_N_MARKERS = 4
_N_CONTACTS = 3
_N_MUSCLES = 2


def _base_trace(**kwargs) -> Trace:
    t = np.arange(_T, dtype=float) * 0.01
    q = _RNG.standard_normal((_T, _NQ))
    v = _RNG.standard_normal((_T, _NQ))
    return Trace(t=t, q=q, v=v, dt=0.01, backend="ode", **kwargs)


# ---------------------------------------------------------------------------
# SCHEMA_VERSION
# ---------------------------------------------------------------------------


def test_schema_version_is_2() -> None:
    """SCHEMA_VERSION must be 2.x after the CC-4 bump."""
    major = SCHEMA_VERSION.split(".", 1)[0]
    assert major == "2", f"Expected major 2, got {SCHEMA_VERSION!r}"


# ---------------------------------------------------------------------------
# Optional group write/read round-trips
# ---------------------------------------------------------------------------


def test_trace_v2_round_trip_with_torques(tmp_path) -> None:
    """A Trace with torques round-trips losslessly."""
    torques = _RNG.standard_normal((_T, _NU))
    trace = _base_trace(torques=torques)
    path = tmp_path / "torques.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert isinstance(loaded, Trace)
    assert loaded.torques is not None
    np.testing.assert_allclose(loaded.torques, torques)


def test_trace_v2_round_trip_with_wrench(tmp_path) -> None:
    """A Trace with wrench (T, 6) round-trips losslessly."""
    wrench = _RNG.standard_normal((_T, 6))
    trace = _base_trace(wrench=wrench)
    path = tmp_path / "wrench.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert loaded.wrench is not None
    np.testing.assert_allclose(loaded.wrench, wrench)


def test_trace_v2_round_trip_with_markers(tmp_path) -> None:
    """A Trace with markers (T, n_markers, 3) round-trips losslessly."""
    markers = _RNG.standard_normal((_T, _N_MARKERS, 3))
    trace = _base_trace(markers=markers)
    path = tmp_path / "markers.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert loaded.markers is not None
    np.testing.assert_allclose(loaded.markers, markers)


def test_trace_v2_round_trip_with_contacts(tmp_path) -> None:
    """A Trace with contacts (T, n_contacts, 3) round-trips losslessly."""
    contacts = _RNG.standard_normal((_T, _N_CONTACTS, 3))
    trace = _base_trace(contacts=contacts)
    path = tmp_path / "contacts.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert loaded.contacts is not None
    np.testing.assert_allclose(loaded.contacts, contacts)


def test_trace_v2_round_trip_all_optional_groups(tmp_path) -> None:
    """A Trace with all optional groups set round-trips completely."""
    torques = _RNG.standard_normal((_T, _NU))
    wrench = _RNG.standard_normal((_T, 6))
    markers = _RNG.standard_normal((_T, _N_MARKERS, 3))
    contacts = _RNG.standard_normal((_T, _N_CONTACTS, 3))

    trace = _base_trace(
        torques=torques, wrench=wrench, markers=markers, contacts=contacts
    )
    path = tmp_path / "all_groups.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    np.testing.assert_allclose(loaded.torques, torques)
    np.testing.assert_allclose(loaded.wrench, wrench)
    np.testing.assert_allclose(loaded.markers, markers)
    np.testing.assert_allclose(loaded.contacts, contacts)


def test_trace_v2_round_trip_with_muscle_outputs(tmp_path) -> None:
    """A Trace with MyoSuite muscle-output groups round-trips losslessly."""
    muscle_names = ("biceps", "triceps")
    activations = _RNG.random((_T, _N_MUSCLES))
    forces = _RNG.standard_normal((_T, _N_MUSCLES))
    lengths = _RNG.random((_T, _N_MUSCLES))
    velocities = _RNG.standard_normal((_T, _N_MUSCLES))
    trace = _base_trace(
        muscle_names=muscle_names,
        muscle_activations=activations,
        muscle_forces=forces,
        muscle_lengths=lengths,
        muscle_velocities=velocities,
    )
    path = tmp_path / "muscles.h5"

    write_trace(trace, path)
    loaded = read_trace(path)

    assert isinstance(loaded, Trace)
    assert loaded.muscle_names == muscle_names
    np.testing.assert_allclose(loaded.muscle_activations, activations)
    np.testing.assert_allclose(loaded.muscle_forces, forces)
    np.testing.assert_allclose(loaded.muscle_lengths, lengths)
    np.testing.assert_allclose(loaded.muscle_velocities, velocities)


def test_trace_v2_optional_groups_absent_when_none(tmp_path) -> None:
    """When optional arrays are None the HDF5 file has no corresponding dataset."""
    trace = _base_trace()
    path = tmp_path / "no_optional.h5"

    write_trace(trace, path)

    with h5py.File(path, "r") as f:
        assert "torques" not in f
        assert "wrench" not in f
        assert "markers" not in f
        assert "contacts" not in f
        assert "muscle_names" not in f
        assert "muscle_activations" not in f
        assert "muscle_forces" not in f
        assert "muscle_lengths" not in f
        assert "muscle_velocities" not in f

    loaded = read_trace(path)
    assert loaded.torques is None
    assert loaded.wrench is None
    assert loaded.markers is None
    assert loaded.contacts is None
    assert loaded.muscle_names == ()
    assert loaded.muscle_activations is None
    assert loaded.muscle_forces is None
    assert loaded.muscle_lengths is None
    assert loaded.muscle_velocities is None


def test_trace_v2_schema_version_stamped(tmp_path) -> None:
    """Written file carries the current SCHEMA_VERSION."""
    trace = _base_trace()
    path = tmp_path / "version.h5"
    write_trace(trace, path)

    with h5py.File(path, "r") as f:
        sv = f.attrs["schema_version"]
        if isinstance(sv, bytes):
            sv = sv.decode()
    assert sv == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Backward compatibility: v1 file auto-migrates
# ---------------------------------------------------------------------------


def _write_v1_file(path, *, with_controls: bool = False) -> None:
    """Write a minimal v1.0.0 trace file directly via h5py."""
    t = np.arange(_T, dtype=float) * 0.01
    q = _RNG.standard_normal((_T, _NQ))
    v = _RNG.standard_normal((_T, _NQ))
    with h5py.File(path, "w") as f:
        f.attrs["schema_version"] = "1.0.0"
        f.attrs["backend"] = "ode"
        f.attrs["dt"] = 0.01
        f.attrs["kind"] = "single"
        f.create_dataset("t", data=t)
        f.create_dataset("q", data=q)
        f.create_dataset("v", data=v)
        if with_controls:
            f.create_dataset("u", data=_RNG.standard_normal((_T, _NU)))


def test_read_trace_auto_migrates_v1_file(tmp_path) -> None:
    """read_trace accepts a v1 file and returns a valid Trace."""
    path = tmp_path / "v1.h5"
    _write_v1_file(path)

    loaded = read_trace(path)
    assert isinstance(loaded, Trace)
    assert loaded.t.shape == (_T,)
    assert loaded.torques is None
    assert loaded.wrench is None


def test_read_trace_v1_with_controls_auto_migrates(tmp_path) -> None:
    """v1 file with controls round-trips controls through auto-migration."""
    path = tmp_path / "v1_u.h5"
    _write_v1_file(path, with_controls=True)

    loaded = read_trace(path)
    assert loaded.u is not None
    assert loaded.u.shape == (_T, _NU)


def test_migrate_from_v1_returns_trace(tmp_path) -> None:
    """migrate_from_v1 reads a v1 file and returns a Trace."""
    path = tmp_path / "v1_mig.h5"
    _write_v1_file(path)

    trace = migrate_from_v1(path)
    assert isinstance(trace, Trace)
    assert trace.schema_version == "1.0.0"
    assert trace.torques is None
    assert trace.wrench is None
    assert trace.markers is None
    assert trace.contacts is None
    assert trace.muscle_names == ()
    assert trace.muscle_activations is None
    assert trace.muscle_forces is None
    assert trace.muscle_lengths is None
    assert trace.muscle_velocities is None


def test_migrate_from_v1_rejects_non_v1_file(tmp_path) -> None:
    """migrate_from_v1 raises ValueError when the file is not major version 1."""
    trace = _base_trace()
    path = tmp_path / "v2.h5"
    write_trace(trace, path)

    with pytest.raises(ValueError, match="version 1"):
        migrate_from_v1(path)


# ---------------------------------------------------------------------------
# BunkerShot3D import
# ---------------------------------------------------------------------------


def _write_bunkershot_file(path) -> dict:
    """Write a minimal BunkerShot3D HDF5 file; return the data for assertion."""
    n = 5
    times = np.arange(n, dtype=float) * 0.01
    positions = _RNG.standard_normal((n, 3))
    quats = np.tile([1.0, 0.0, 0.0, 0.0], (n, 1))
    forces = _RNG.standard_normal((n, 3))
    torques = _RNG.standard_normal((n, 3))

    with h5py.File(path, "w") as f:
        cg = f.create_group("clubhead")
        wg = f.create_group("wrench")
        f.create_group("grains")

        for i, t in enumerate(times):
            key = f"t_{t:.6f}"
            sg = cg.create_group(key)
            sg.attrs["time"] = t
            sg.create_dataset("position", data=positions[i])
            sg.create_dataset("orientation", data=quats[i])

            wsg = wg.create_group(key)
            wsg.attrs["time"] = t
            wsg.create_dataset("force", data=forces[i])
            wsg.create_dataset("torque", data=torques[i])

    return {
        "n": n,
        "times": times,
        "positions": positions,
        "forces": forces,
        "torques": torques,
    }


def test_read_bunkershot3d_result_returns_trace(tmp_path) -> None:
    """read_bunkershot3d_result converts a BunkerShot3D HDF5 file into a Trace."""
    path = tmp_path / "bunker.h5"
    ref = _write_bunkershot_file(path)

    trace = read_bunkershot3d_result(path)

    assert isinstance(trace, Trace)
    assert trace.t.shape == (ref["n"],)
    np.testing.assert_allclose(trace.t, ref["times"])


def test_read_bunkershot3d_result_populates_markers(tmp_path) -> None:
    """Clubhead positions become the markers array."""
    path = tmp_path / "bunker_markers.h5"
    ref = _write_bunkershot_file(path)

    trace = read_bunkershot3d_result(path)

    assert trace.markers is not None
    assert trace.markers.shape == (ref["n"], 1, 3)
    np.testing.assert_allclose(trace.markers[:, 0, :], ref["positions"])


def test_read_bunkershot3d_result_populates_wrench(tmp_path) -> None:
    """Contact wrenches become the wrench (T, 6) array."""
    path = tmp_path / "bunker_wrench.h5"
    ref = _write_bunkershot_file(path)

    trace = read_bunkershot3d_result(path)

    assert trace.wrench is not None
    assert trace.wrench.shape == (ref["n"], 6)
    np.testing.assert_allclose(trace.wrench[:, :3], ref["forces"])
    np.testing.assert_allclose(trace.wrench[:, 3:], ref["torques"])


# ---------------------------------------------------------------------------
# Shape validation
# ---------------------------------------------------------------------------


def test_trace_v2_torques_wrong_first_dim_raises() -> None:
    """torques with wrong time dimension raises ValueError."""
    torques = _RNG.standard_normal((_T + 1, _NU))
    with pytest.raises(ValueError, match="torques"):
        _base_trace(torques=torques)


def test_trace_v2_wrench_wrong_second_dim_raises() -> None:
    """wrench with shape other than (T, 6) raises ValueError."""
    wrench = _RNG.standard_normal((_T, 3))
    with pytest.raises(ValueError, match="wrench"):
        _base_trace(wrench=wrench)


def test_trace_v2_markers_wrong_last_dim_raises() -> None:
    """markers with shape other than (T, n, 3) raises ValueError."""
    markers = _RNG.standard_normal((_T, _N_MARKERS, 2))
    with pytest.raises(ValueError, match="markers"):
        _base_trace(markers=markers)


def test_trace_v2_contacts_wrong_first_dim_raises() -> None:
    """contacts with wrong time dimension raises ValueError."""
    contacts = _RNG.standard_normal((_T + 2, _N_CONTACTS, 3))
    with pytest.raises(ValueError, match="contacts"):
        _base_trace(contacts=contacts)


def test_trace_v2_muscle_outputs_wrong_first_dim_raises() -> None:
    """muscle histories with wrong time dimension raise ValueError."""
    activations = _RNG.random((_T + 1, _N_MUSCLES))
    with pytest.raises(ValueError, match="muscle_activations"):
        _base_trace(muscle_activations=activations)


def test_trace_v2_muscle_names_wrong_width_raises() -> None:
    """muscle_names must match muscle-output columns when provided."""
    activations = _RNG.random((_T, _N_MUSCLES))
    with pytest.raises(ValueError, match="muscle_names"):
        _base_trace(
            muscle_names=("one_name",),
            muscle_activations=activations,
        )
