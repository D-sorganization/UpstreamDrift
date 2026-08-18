"""Versioned HDF5 (de)serialisation for :class:`Trace` and :class:`BatchTrace`.

This is the *shared schema*: a single on-disk format every backend writes and
the analysis layer reads, so a rollout produced on the GPU can be replayed,
diffed, or cross-validated on CPU. The format is self-describing and versioned
via :data:`~simulation_backends.protocol.SCHEMA_VERSION`.

On-disk layout (HDF5)
---------------------
Root group attributes:

* ``schema_version`` -- the ``MAJOR.MINOR.PATCH`` schema string at write time.
* ``backend`` -- name of the backend that produced the trace.
* ``dt`` -- integration step [s] as a float.
* ``kind`` -- ``"single"`` for a :class:`Trace`, ``"batch"`` for a
  :class:`BatchTrace`. The reader dispatches on this attribute.
* ``meta_<key>`` -- one attribute per scalar metadata entry. Only ``str`` /
  ``int`` / ``float`` / ``bool`` values are persisted; richer objects are
  skipped silently (the attribute namespace only round-trips scalars).

Datasets (all single traces):

* ``t`` -- sample times, shape ``(T,)``.
* ``q`` -- positions, shape ``(T, nq)`` (single) or ``(N, T, nq)`` (batch).
* ``v`` -- velocities, matching ``q``.
* ``u`` -- controls, written **only** when present (``None`` is omitted).
* ``torques`` -- joint torques ``(T, nu)``; written only when present (v2+).
* ``wrench`` -- contact wrench ``(T, 6)``; written only when present (v2+).
* ``markers`` -- marker positions ``(T, n_markers, 3)``; written only when
  present (v2+).
* ``contacts`` -- contact points ``(T, n_contacts, 3)``; written only when
  present (v2+).
* ``muscle_names`` -- UTF-8 muscle output column labels; written only when
  present (v2.1+).
* ``muscle_activations`` / ``muscle_forces`` / ``muscle_lengths`` /
  ``muscle_velocities`` -- MyoSuite muscle histories ``(T, n_muscles)``;
  written only when present (v2.1+).

Versioning policy
-----------------
The reader accepts schema **major 1** (v1.x legacy) and **major 2** (current).
A v1 file is auto-migrated: new optional datasets default to ``None``. Any
other major raises :class:`ValueError`. Minor/patch differences within an
accepted major are always accepted.

Migration helpers
-----------------
:func:`migrate_from_v1` reads a v1.x file explicitly and returns a :class:`Trace`.
:func:`read_bunkershot3d_result` converts a BunkerShot3D HDF5 file into a
:class:`Trace` (clubhead positions → ``markers``; wrenches → ``wrench``).
See ``docs/simulation_backends/results_schema_v2.md`` for the full schema.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .protocol import SCHEMA_VERSION, BatchTrace, Trace

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["migrate_from_v1", "read_bunkershot3d_result", "read_trace", "write_trace"]

#: Prefix applied to every metadata entry stored as a root HDF5 attribute.
_META_PREFIX = "meta_"

#: Scalar attribute value types that HDF5 round-trips losslessly. Note ``bool``
#: precedes ``int`` only for clarity; ``isinstance`` order is irrelevant here.
_SCALAR_ATTR_TYPES = (str, bool, int, float)

#: Discriminator stored in the root ``kind`` attribute.
_KIND_SINGLE = "single"
_KIND_BATCH = "batch"


def _validate_path(path: str | os.PathLike[str]) -> None:
    """Validate the serialisation ``path`` precondition.

    Args:
        path: Destination/source filesystem path.

    Raises:
        TypeError: If ``path`` is not a ``str`` or :class:`os.PathLike`.
        ValueError: If ``path`` is empty / whitespace.
    """
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be str or os.PathLike, got {type(path).__name__}")
    if not str(os.fspath(path)).strip():
        raise ValueError("path must be a non-empty filesystem path")


def _meta_scalars(meta: Mapping[str, object]) -> dict[str, str | bool | int | float]:
    """Return the subset of ``meta`` whose values are HDF5-storable scalars.

    Non-scalar values (arrays, nested mappings, ``None``, ...) are dropped so
    the attribute namespace only ever holds round-trippable primitives.

    Args:
        meta: Free-form provenance metadata.

    Returns:
        Mapping of key to scalar value, keys order-preserved.
    """
    # ``bool`` is a subclass of ``int``; both are acceptable scalars, so a
    # single isinstance check against the tuple is sufficient and correct.
    return {
        key: value
        for key, value in meta.items()
        if isinstance(value, _SCALAR_ATTR_TYPES)
    }


def _write_optional_dataset(
    root: h5py.Group, name: str, data: np.ndarray | None
) -> None:
    """Write ``data`` as dataset ``name`` under ``root`` iff it is not None."""
    if data is not None:
        root.create_dataset(name, data=np.asarray(data, dtype=float))


def _read_optional_dataset(root: h5py.Group, name: str) -> np.ndarray | None:
    """Return dataset ``name`` from ``root`` as a float array, or None if absent."""
    if name not in root:
        return None
    return np.asarray(root[name][()], dtype=float)


def _write_string_dataset(root: h5py.Group, name: str, data: tuple[str, ...]) -> None:
    """Write UTF-8 string tuple ``data`` as dataset ``name`` iff non-empty."""
    if data:
        root.create_dataset(
            name,
            data=np.asarray(data, dtype=h5py.string_dtype(encoding="utf-8")),
        )


def _read_string_dataset(root: h5py.Group, name: str) -> tuple[str, ...]:
    """Return UTF-8 strings from dataset ``name``, or an empty tuple if absent."""
    if name not in root:
        return ()
    raw = root[name][()]
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in raw
    )


def _write_common(
    root: h5py.Group,
    *,
    kind: str,
    backend: str,
    dt: float,
    schema_version: str,
    meta: Mapping[str, object],
    t: np.ndarray,
    q: np.ndarray,
    v: np.ndarray,
    u: np.ndarray | None,
    torques: np.ndarray | None = None,
    wrench: np.ndarray | None = None,
    markers: np.ndarray | None = None,
    contacts: np.ndarray | None = None,
    muscle_names: tuple[str, ...] = (),
    muscle_activations: np.ndarray | None = None,
    muscle_forces: np.ndarray | None = None,
    muscle_lengths: np.ndarray | None = None,
    muscle_velocities: np.ndarray | None = None,
) -> None:
    """Write the shared attribute/dataset payload for either trace kind.

    Centralises the layout so :class:`Trace` and :class:`BatchTrace` cannot
    drift (DRY); only the array ranks differ between callers.

    Args:
        root: Open HDF5 group to populate (typically the file root).
        kind: ``"single"`` or ``"batch"`` discriminator.
        backend: Producing backend name.
        dt: Integration step [s].
        schema_version: Schema string to stamp.
        meta: Provenance metadata (scalars persisted, others skipped).
        t: Sample-time array.
        q: Position array.
        v: Velocity array.
        u: Control array, or ``None`` to omit the dataset.
        torques: Optional joint-torque array ``(T, nu)``.
        wrench: Optional contact wrench ``(T, 6)``.
        markers: Optional marker-position array ``(T, n_markers, 3)``.
        contacts: Optional contact-point array ``(T, n_contacts, 3)``.
        muscle_names: Optional muscle output column labels.
        muscle_activations: Optional activation history ``(T, n_muscles)``.
        muscle_forces: Optional force history ``(T, n_muscles)``.
        muscle_lengths: Optional length history ``(T, n_muscles)``.
        muscle_velocities: Optional velocity history ``(T, n_muscles)``.

    Postconditions:
        ``root`` has ``schema_version``/``backend``/``dt``/``kind`` attrs, the
        ``t``/``q``/``v`` datasets, optional datasets written iff not None.
    """
    root.attrs["schema_version"] = schema_version
    root.attrs["backend"] = backend
    root.attrs["dt"] = float(dt)
    root.attrs["kind"] = kind
    for key, value in _meta_scalars(meta).items():
        root.attrs[f"{_META_PREFIX}{key}"] = value

    root.create_dataset("t", data=np.asarray(t, dtype=float))
    root.create_dataset("q", data=np.asarray(q, dtype=float))
    root.create_dataset("v", data=np.asarray(v, dtype=float))
    _write_optional_dataset(root, "u", u)
    _write_optional_dataset(root, "torques", torques)
    _write_optional_dataset(root, "wrench", wrench)
    _write_optional_dataset(root, "markers", markers)
    _write_optional_dataset(root, "contacts", contacts)
    _write_string_dataset(root, "muscle_names", muscle_names)
    _write_optional_dataset(root, "muscle_activations", muscle_activations)
    _write_optional_dataset(root, "muscle_forces", muscle_forces)
    _write_optional_dataset(root, "muscle_lengths", muscle_lengths)
    _write_optional_dataset(root, "muscle_velocities", muscle_velocities)


def write_trace(trace: Trace | BatchTrace, path: str | os.PathLike[str]) -> None:
    """Serialise a trace to a versioned HDF5 file.

    Args:
        trace: A :class:`Trace` (single rollout) or :class:`BatchTrace`
            (batched rollout) to persist.
        path: Destination filesystem path. Any existing file is overwritten.

    Raises:
        TypeError: If ``trace`` is neither a :class:`Trace` nor a
            :class:`BatchTrace`, or if ``path`` is the wrong type.
        ValueError: If ``path`` is empty.

    Postconditions:
        A self-describing HDF5 file exists at ``path`` whose ``kind`` attribute
        round-trips through :func:`read_trace` to an equal dataclass.
    """
    _validate_path(path)
    if isinstance(trace, BatchTrace):
        kind = _KIND_BATCH
    elif isinstance(trace, Trace):
        kind = _KIND_SINGLE
    else:
        raise TypeError(
            f"trace must be a Trace or BatchTrace, got {type(trace).__name__}"
        )

    with h5py.File(os.fspath(path), "w") as handle:
        torques = getattr(trace, "torques", None)
        wrench = getattr(trace, "wrench", None)
        markers = getattr(trace, "markers", None)
        contacts = getattr(trace, "contacts", None)
        muscle_names = getattr(trace, "muscle_names", ())
        muscle_activations = getattr(trace, "muscle_activations", None)
        muscle_forces = getattr(trace, "muscle_forces", None)
        muscle_lengths = getattr(trace, "muscle_lengths", None)
        muscle_velocities = getattr(trace, "muscle_velocities", None)
        _write_common(
            handle,
            kind=kind,
            backend=trace.backend,
            dt=trace.dt,
            schema_version=trace.schema_version,
            meta=trace.meta,
            t=trace.t,
            q=trace.q,
            v=trace.v,
            u=trace.u,
            torques=torques,
            wrench=wrench,
            markers=markers,
            contacts=contacts,
            muscle_names=muscle_names,
            muscle_activations=muscle_activations,
            muscle_forces=muscle_forces,
            muscle_lengths=muscle_lengths,
            muscle_velocities=muscle_velocities,
        )


def _check_schema_compatible(file_version: str, path: str) -> None:
    """Validate that the file's schema major is accepted by the reader.

    Accepted majors: ``"1"`` (v1.x legacy, auto-migrated) and ``"2"``
    (current). Any other major raises :class:`ValueError`.

    Args:
        file_version: ``schema_version`` attribute read from the file.
        path: Source path, included in error messages for diagnostics.

    Raises:
        ValueError: If the major is not ``"1"`` or ``"2"``.
    """
    file_major = _major(file_version)
    current_major = _major(SCHEMA_VERSION)
    accepted = {current_major, "1"}
    if file_major not in accepted:
        raise ValueError(
            f"incompatible trace schema in {path!r}: file major version "
            f"{file_major} (schema {file_version!r}) is not in the "
            f"supported set {sorted(accepted)}"
        )


def _major(version: str) -> str:
    """Return the MAJOR component of a ``MAJOR.MINOR.PATCH`` version string.

    Args:
        version: A semantic-version-style string.

    Returns:
        The substring before the first ``"."`` (the whole string if none).
    """
    return str(version).split(".", 1)[0]


def _read_attr(root: h5py.Group, name: str) -> object:
    """Read a root attribute, decoding HDF5 byte strings to ``str``.

    h5py may return ``bytes`` for string attributes depending on how they were
    written; normalise to ``str`` so the rebuilt dataclass matches the source.

    Args:
        root: Open HDF5 group.
        name: Attribute name.

    Returns:
        The attribute value, with ``bytes`` decoded as UTF-8.
    """
    value = root.attrs[name]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return value


def _read_meta(root: h5py.Group) -> dict[str, object]:
    """Reconstruct the ``meta`` mapping from ``meta_*`` root attributes.

    Args:
        root: Open HDF5 group.

    Returns:
        Metadata mapping with the ``meta_`` prefix stripped from each key and
        numpy scalar types coerced to native Python scalars.
    """
    meta: dict[str, object] = {}
    for name in root.attrs:
        if not name.startswith(_META_PREFIX):
            continue
        key = name[len(_META_PREFIX) :]
        meta[key] = _coerce_scalar(_read_attr(root, name))
    return meta


def _coerce_scalar(value: object) -> object:
    """Coerce numpy scalar attribute values to native Python scalars.

    h5py returns numpy types (e.g. ``numpy.int64``, ``numpy.bool_``) for scalar
    attributes; convert them so round-tripped metadata equals the original
    Python primitives.

    Args:
        value: A raw attribute value.

    Returns:
        ``value.item()`` for numpy scalars, otherwise ``value`` unchanged.
    """
    if isinstance(value, np.generic):
        return value.item()
    return value


def read_trace(path: str | os.PathLike[str]) -> Trace | BatchTrace:
    """Deserialise a versioned HDF5 file into a trace dataclass.

    The concrete return type is chosen by the file's ``kind`` attribute:
    ``"single"`` rebuilds a :class:`Trace`, ``"batch"`` a :class:`BatchTrace`.

    Args:
        path: Source filesystem path written by :func:`write_trace`.

    Returns:
        The reconstructed :class:`Trace` or :class:`BatchTrace`.

    Raises:
        TypeError: If ``path`` is the wrong type.
        ValueError: If ``path`` is empty, the file's schema major version does
            not match :data:`SCHEMA_VERSION`, or ``kind`` is unrecognised.

    Postconditions:
        For any ``x`` written by :func:`write_trace`, the returned object equals
        ``x`` on ``t``/``q``/``v``/``u``, ``dt``, ``backend``, scalar ``meta``,
        and ``schema_version``.
    """
    _validate_path(path)
    spath = os.fspath(path)
    with h5py.File(spath, "r") as handle:
        schema_version = str(_read_attr(handle, "schema_version"))
        _check_schema_compatible(schema_version, str(spath))

        kind = str(_read_attr(handle, "kind"))
        backend = str(_read_attr(handle, "backend"))
        dt = float(_read_attr(handle, "dt"))  # type: ignore[arg-type]
        meta = _read_meta(handle)

        t = np.asarray(handle["t"][()], dtype=float)
        q = np.asarray(handle["q"][()], dtype=float)
        v = np.asarray(handle["v"][()], dtype=float)
        u = _read_optional_dataset(handle, "u")

        torques = _read_optional_dataset(handle, "torques")
        wrench = _read_optional_dataset(handle, "wrench")
        markers = _read_optional_dataset(handle, "markers")
        contacts = _read_optional_dataset(handle, "contacts")
        muscle_names = _read_string_dataset(handle, "muscle_names")
        muscle_activations = _read_optional_dataset(handle, "muscle_activations")
        muscle_forces = _read_optional_dataset(handle, "muscle_forces")
        muscle_lengths = _read_optional_dataset(handle, "muscle_lengths")
        muscle_velocities = _read_optional_dataset(handle, "muscle_velocities")

    if kind == _KIND_SINGLE:
        return Trace(
            t=t,
            q=q,
            v=v,
            u=u,
            dt=dt,
            backend=backend,
            meta=meta,
            schema_version=schema_version,
            torques=torques,
            wrench=wrench,
            markers=markers,
            contacts=contacts,
            muscle_names=muscle_names,
            muscle_activations=muscle_activations,
            muscle_forces=muscle_forces,
            muscle_lengths=muscle_lengths,
            muscle_velocities=muscle_velocities,
        )
    if kind == _KIND_BATCH:
        return BatchTrace(
            t=t,
            q=q,
            v=v,
            u=u,
            dt=dt,
            backend=backend,
            meta=meta,
            schema_version=schema_version,
        )
    raise ValueError(
        f"unrecognised trace kind {kind!r} in {spath!r}; "
        f"expected {_KIND_SINGLE!r} or {_KIND_BATCH!r}"
    )


def migrate_from_v1(path: str | os.PathLike[str]) -> Trace:
    """Read a v1.x :class:`Trace` file and return a :class:`Trace`.

    Unlike :func:`read_trace`, this function explicitly requires the file to
    be major version 1 and raises if it is not. New optional fields
    (``torques``, ``wrench``, ``markers``, ``contacts``) default to ``None``.

    Args:
        path: Source filesystem path of a v1.x trace file.

    Returns:
        A :class:`Trace` with ``schema_version`` preserved from the file.

    Raises:
        TypeError: If ``path`` is not a str or :class:`os.PathLike`.
        ValueError: If ``path`` is empty or the file is not major version 1.
    """
    _validate_path(path)
    spath = os.fspath(path)
    with h5py.File(spath, "r") as handle:
        schema_version = str(_read_attr(handle, "schema_version"))
        file_major = _major(schema_version)
        if file_major != "1":
            raise ValueError(
                f"migrate_from_v1 requires a version 1 file; "
                f"got schema {schema_version!r} in {spath!r}"
            )
        kind = str(_read_attr(handle, "kind"))
        backend = str(_read_attr(handle, "backend"))
        dt = float(_read_attr(handle, "dt"))  # type: ignore[arg-type]
        meta = _read_meta(handle)
        t = np.asarray(handle["t"][()], dtype=float)
        q = np.asarray(handle["q"][()], dtype=float)
        v = np.asarray(handle["v"][()], dtype=float)
        u = _read_optional_dataset(handle, "u")

    if kind != _KIND_SINGLE:
        raise ValueError(
            f"migrate_from_v1 only supports single-trace files; "
            f"got kind={kind!r} in {spath!r}"
        )
    return Trace(
        t=t,
        q=q,
        v=v,
        u=u,
        dt=dt,
        backend=backend,
        meta=meta,
        schema_version=schema_version,
    )


def read_bunkershot3d_result(path: str | os.PathLike[str]) -> Trace:
    """Convert a BunkerShot3D HDF5 result file into a :class:`Trace`.

    BunkerShot3D stores clubhead states and contact wrenches in per-timestep
    sub-groups (``/clubhead/t_<t>/`` and ``/wrench/t_<t>/``). This function
    reads those groups and maps them to the unified v2 schema:

    * Clubhead positions → ``Trace.markers`` shape ``(T, 1, 3)`` [m].
    * Contact wrenches → ``Trace.wrench`` shape ``(T, 6)`` [N, N, N, N·m, …].
    * ``Trace.q`` / ``Trace.v`` are empty ``(T, 0)`` arrays (no joint states).
    * ``Trace.backend`` is set to ``"bunkershot3d"``.

    Args:
        path: Source filesystem path of a BunkerShot3D HDF5 result file.

    Returns:
        A :class:`Trace` with schema_version equal to :data:`SCHEMA_VERSION`.

    Raises:
        TypeError: If ``path`` is not a str or :class:`os.PathLike`.
        ValueError: If ``path`` is empty or the file lacks a ``/clubhead``
            group.
    """
    _validate_path(path)
    spath = os.fspath(path)
    with h5py.File(spath, "r") as f:
        if "clubhead" not in f:
            raise ValueError(
                f"not a BunkerShot3D file: missing /clubhead group in {spath!r}"
            )
        clubhead_grp = f["clubhead"]
        keys = sorted(clubhead_grp.keys())
        times = np.array([clubhead_grp[k].attrs["time"] for k in keys], dtype=float)
        positions = np.array(
            [clubhead_grp[k]["position"][:] for k in keys], dtype=float
        )

        if "wrench" in f:
            wrench_grp = f["wrench"]
            wkeys = sorted(wrench_grp.keys())
            forces = np.array([wrench_grp[k]["force"][:] for k in wkeys], dtype=float)
            torques = np.array([wrench_grp[k]["torque"][:] for k in wkeys], dtype=float)
            wrench = np.concatenate([forces, torques], axis=1)
        else:
            wrench = None

    n = len(times)
    markers = positions.reshape(n, 1, 3)
    empty = np.empty((n, 0), dtype=float)
    return Trace(
        t=times,
        q=empty,
        v=empty,
        dt=float(np.mean(np.diff(times))) if n > 1 else 0.0,
        backend="bunkershot3d",
        markers=markers,
        wrench=wrench,
    )
