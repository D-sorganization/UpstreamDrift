"""Versioned HDF5 (de)serialisation for :class:`Trace` and :class:`BatchTrace`.

This is the M3.3 *shared schema*: a single on-disk format every backend writes
and the analysis layer reads, so a rollout produced on the GPU can be replayed,
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

Datasets:

* ``t`` -- sample times, shape ``(T,)``.
* ``q`` -- positions, shape ``(T, nq)`` (single) or ``(N, T, nq)`` (batch).
* ``v`` -- velocities, matching ``q``.
* ``u`` -- controls, written **only** when present (``None`` is omitted).

Versioning policy
-----------------
The reader requires the file's schema *MAJOR* component to equal the running
:data:`SCHEMA_VERSION` major; a mismatch raises :class:`ValueError` rather than
silently misinterpreting an incompatible layout. Minor/patch differences are
accepted (additive, backward-compatible changes).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import h5py
import numpy as np

from .protocol import SCHEMA_VERSION, BatchTrace, Trace

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = ["read_trace", "write_trace"]

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

    Postconditions:
        ``root`` has ``schema_version``/``backend``/``dt``/``kind`` attrs, the
        ``t``/``q``/``v`` datasets, and a ``u`` dataset iff ``u is not None``.
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
    if u is not None:
        root.create_dataset("u", data=np.asarray(u, dtype=float))


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
        )


def _check_schema_compatible(file_version: str, path: str) -> None:
    """Validate that the file's schema major matches the running schema.

    Args:
        file_version: ``schema_version`` attribute read from the file.
        path: Source path, included in error messages for diagnostics.

    Raises:
        ValueError: If the major components differ (incompatible layout).
    """
    file_major = _major(file_version)
    current_major = _major(SCHEMA_VERSION)
    if file_major != current_major:
        raise ValueError(
            f"incompatible trace schema in {path!r}: file major version "
            f"{file_major} (schema {file_version!r}) does not match the "
            f"supported major version {current_major} (schema "
            f"{SCHEMA_VERSION!r})"
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
        u = np.asarray(handle["u"][()], dtype=float) if "u" in handle else None

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
