"""Versioned HDF5 result schema for BunkerShot3D (issue #8617, B17/B18).

Layout v3 (current)
-------------------
Root attributes, ``schema_version`` written **first** so a reader can dispatch
before touching anything else::

    schema_version : int64   -- 3
    format         : str     -- "bunkershot3d-result"
    manifest_*               -- run manifest, see bunkershot3d.provenance

Contiguous, chunked-along-time datasets (no per-timestep groups)::

    /clubhead/t            (T,)      float64  sample times [s]
    /clubhead/position     (T, 3)    float64  clubhead origin [m]
    /clubhead/orientation  (T, 4)    float64  unit quaternion, scalar-first
    /wrench/t              (T,)      float64
    /wrench/force          (T, 3)    float64  [N]
    /wrench/torque         (T, 3)    float64  [N.m]
    /grains/t              (T,)      float64
    /grains/counts         (T,)      int64    grains recorded per frame
    /grains/positions      (M, 3)    float64  frames concatenated, M = sum(counts)
    /grains/velocities     (M, 3)    float64

Grain frames are stored ragged (concatenated points plus a per-frame count)
because a downsampled grain population need not have a constant size.

v3 adds the sand-field payload of issue #8710, written whole rather than
appended because a field is produced by one march and has no streaming
consumer::

    /sand_field/t              (T,)         sample times [s]
    /sand_field/velocity       (T, N, D)    [m/s]
    /sand_field/density        (T, N)       [kg/m^3]
    /sand_field/shear_rate     (T, N)       [1/s], optional
    /sand_field/positions      (T, N, D)    [m], PARTICLE layout only
    /sand_field/body_outline   (T, V, D)    [m], intruder section, optional
    /sand_field.attrs["metadata"]      : str  -- canonical JSON, tier included
    /sand_field.attrs["content_sha256"]: str  -- covers metadata *and* arrays

The digest is why the tier is data rather than a filename convention: editing
the stored tier and re-copying the file breaks it, and
:mod:`bunkershot3d.fields.store` refuses to load a field whose digest does not
match its contents.

Layout v1 (legacy, read-only)
-----------------------------
One group per timestep -- ``/clubhead/t_0.000500/{position,orientation}`` --
with the sample time in a ``time`` attribute. 2 kHz x 0.5 s produced 1000 groups
per stream, and the read path sorted ``f"t_{time:.6f}"`` *as strings*, so frame
order broke at t >= 10 s ("t_10.000000" sorts before "t_9.500000").

:class:`BunkerShotResultReader` accepts v1, v2 and v3. v1 files are migrated
**on read** -- ordered by the numeric ``time`` attribute, never by key string,
and never rewritten in place. A v2 file simply has no ``/sand_field`` group,
which reads back as "no field was recorded" rather than as an error.
"""

from __future__ import annotations

import time as _time
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np

from src.shared.python.core.contracts import require

from ..provenance import RunManifest

__all__ = [
    "DEFAULT_TIME_CHUNK",
    "FIELD_DIGEST_ATTR",
    "FIELD_GROUP",
    "FIELD_METADATA_ATTR",
    "FORMAT_ATTR",
    "FORMAT_NAME",
    "SCHEMA_VERSION",
    "SCHEMA_VERSION_ATTR",
    "SUPPORTED_SCHEMA_VERSIONS",
    "BunkerShotResultReader",
    "BunkerShotResultWriter",
    "SandFieldPayload",
]

#: Schema version emitted by :class:`BunkerShotResultWriter`.
SCHEMA_VERSION = 3

#: Schema versions :class:`BunkerShotResultReader` can read.
SUPPORTED_SCHEMA_VERSIONS = (1, 2, 3)

#: Root attribute holding the integer schema version.
SCHEMA_VERSION_ATTR = "schema_version"

#: Root attribute naming the on-disk format.
FORMAT_ATTR = "format"

#: Value of :data:`FORMAT_ATTR`.
FORMAT_NAME = "bunkershot3d-result"

#: Frames per HDF5 chunk along the time axis.
DEFAULT_TIME_CHUNK = 1024

#: Group holding the sand-field payload (v3+).
FIELD_GROUP = "sand_field"

#: Attribute on :data:`FIELD_GROUP` carrying the field metadata as canonical JSON.
FIELD_METADATA_ATTR = "metadata"

#: Attribute on :data:`FIELD_GROUP` carrying the digest of metadata plus arrays.
FIELD_DIGEST_ATTR = "content_sha256"

_CLUBHEAD = "clubhead"
_WRENCH = "wrench"
_GRAINS = "grains"
_TIME = "t"
_COUNTS = "counts"
_FIELD_ARRAYS = (
    "velocity",
    "density",
    "shear_rate",
    "positions",
    "body_outline",
)


class SandFieldPayload(NamedTuple):
    """A sand field as it sits on disk, before any schema is imposed on it.

    The io layer deliberately does not know what a
    :class:`~bunkershot3d.fields.schema.SandFieldSeries` is: it moves
    named arrays, a metadata string and a digest, and
    :mod:`bunkershot3d.fields.store` is the only place that knows how
    those become a typed field.  That keeps a change to the field schema
    from being a change to the file format, which is the whole reason
    the two carry separate version numbers.

    Attributes:
        time_s: ``(T,)`` frame times.
        arrays: Name to array, for the arrays that were written. Names
            outside :data:`_FIELD_ARRAYS` are rejected on write.
        metadata: Canonical-JSON metadata, including the fidelity tier.
        digest: SHA-256 over the metadata and the arrays together.
    """

    time_s: np.ndarray
    arrays: dict[str, np.ndarray]
    metadata: str
    digest: str


class _Appender:
    """Buffered append to a resizable HDF5 dataset chunked along time.

    Rows are accumulated in memory and flushed in blocks so a 2 kHz run costs
    one dataset resize per chunk rather than one HDF5 object per sample.
    """

    def __init__(
        self,
        group: h5py.Group,
        name: str,
        width: int | None,
        chunk: int,
        dtype: Any = np.float64,
    ) -> None:
        """Initialise the appender.

        Args:
            group: Destination HDF5 group.
            name: Dataset name within ``group``.
            width: Row width, or ``None`` for a 1-D dataset.
            chunk: Rows per HDF5 chunk (and per flush).
            dtype: Dataset element type.
        """
        self._group = group
        self._name = name
        self._width = width
        self._chunk = chunk
        self._dtype = dtype
        self._blocks: list[np.ndarray] = []
        self._buffered = 0

    @property
    def _tail(self) -> tuple[int, ...]:
        """Return the trailing (non-time) dataset shape."""
        return () if self._width is None else (self._width,)

    def append(self, row: Any) -> None:
        """Buffer a single row."""
        self.extend(np.asarray(row, dtype=self._dtype).reshape((1, *self._tail)))

    def extend(self, block: np.ndarray) -> None:
        """Buffer a block of rows shaped ``(k, *tail)``.

        The block is copied: callers routinely pass a live view into a solver's
        state (``mjData.xpos[body]``), which the solver overwrites on the next
        step. Buffering the view instead of its value would write the final
        state into every frame.
        """
        snapshot = np.array(block, dtype=self._dtype, copy=True)
        self._blocks.append(snapshot)
        self._buffered += int(snapshot.shape[0])
        if self._buffered >= self._chunk:
            self.flush()

    def flush(self) -> None:
        """Write buffered rows to the dataset, creating it on first use."""
        if not self._blocks:
            return
        block = np.concatenate(self._blocks, axis=0)
        self._blocks = []
        self._buffered = 0
        dataset = self._group.get(self._name)
        if dataset is None:
            dataset = self._group.create_dataset(
                self._name,
                shape=(0, *self._tail),
                maxshape=(None, *self._tail),
                chunks=(self._chunk, *self._tail),
                dtype=self._dtype,
            )
        start = int(dataset.shape[0])
        dataset.resize(start + int(block.shape[0]), axis=0)
        dataset[start:] = block


class _Stream:
    """One time series (clubhead / wrench) of fixed-width rows."""

    def __init__(self, group: h5py.Group, widths: dict[str, int], chunk: int) -> None:
        """Initialise the stream.

        Args:
            group: Destination HDF5 group.
            widths: Dataset name to row width, e.g. ``{"position": 3}``.
            chunk: Frames per chunk.
        """
        self.group = group
        self._widths = dict(widths)
        self._time = _Appender(group, _TIME, None, chunk)
        self._columns = {
            name: _Appender(group, name, width, chunk) for name, width in widths.items()
        }
        self._last_time: float | None = None

    def append(self, time: float, values: dict[str, np.ndarray]) -> None:
        """Append one frame.

        Args:
            time: Sample time [s]; must not go backwards.
            values: One array per column, each of the declared width.

        Raises:
            ValueError: If ``time`` is not finite, goes backwards, or a value
                has the wrong shape.
        """
        stamp = _validated_time(time, self._last_time)
        # Validate every column before writing any of it, so a rejected frame
        # cannot leave the streams at different lengths.
        checked = {}
        for name, width in self._widths.items():
            array = np.asarray(values[name], dtype=np.float64)
            if array.shape != (width,):
                raise ValueError(
                    f"{name} must have shape {(width,)}, got {array.shape}"
                )
            checked[name] = array
        self._time.append(stamp)
        for name, array in checked.items():
            self._columns[name].append(array)
        self._last_time = stamp

    def flush(self) -> None:
        """Flush the time axis and every column."""
        self._time.flush()
        for appender in self._columns.values():
            appender.flush()


class _RaggedStream:
    """Grain frames whose particle count may vary between samples."""

    def __init__(self, group: h5py.Group, chunk: int) -> None:
        """Initialise the stream.

        Args:
            group: Destination HDF5 group.
            chunk: Frames per chunk on the time axis.
        """
        self.group = group
        self._time = _Appender(group, _TIME, None, chunk)
        self._counts = _Appender(group, _COUNTS, None, chunk, dtype=np.int64)
        self._positions = _Appender(group, "positions", 3, chunk)
        self._velocities = _Appender(group, "velocities", 3, chunk)
        self._last_time: float | None = None

    def append(
        self, time: float, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """Append one grain frame.

        Args:
            time: Sample time [s]; must not go backwards.
            positions: ``(N, 3)`` grain positions [m].
            velocities: ``(N, 3)`` grain velocities [m/s].

        Raises:
            ValueError: If the arrays are not ``(N, 3)`` or disagree in length.
        """
        stamp = _validated_time(time, self._last_time)
        pos = np.asarray(positions, dtype=np.float64)
        vel = np.asarray(velocities, dtype=np.float64)
        if pos.ndim != 2 or pos.shape[1] != 3:
            raise ValueError(f"positions must have shape (N, 3), got {pos.shape}")
        if vel.shape != pos.shape:
            raise ValueError(f"velocities must have shape {pos.shape}, got {vel.shape}")
        self._time.append(stamp)
        self._counts.append(np.int64(pos.shape[0]))
        self._positions.extend(pos)
        self._velocities.extend(vel)
        self._last_time = stamp

    def flush(self) -> None:
        """Flush the time axis, counts and both point arrays."""
        for appender in (self._time, self._counts, self._positions, self._velocities):
            appender.flush()


def _validated_time(time: float, last_time: float | None) -> float:
    """Return ``time`` as a float after checking it is usable as a sample time.

    Args:
        time: Proposed sample time [s].
        last_time: Previously written sample time, or ``None``.

    Returns:
        ``time`` as a Python float.

    Raises:
        ValueError: If ``time`` is not finite or is earlier than ``last_time``.
    """
    stamp = float(time)
    if not np.isfinite(stamp):
        raise ValueError(f"time must be finite, got {time!r}")
    if last_time is not None and stamp < last_time:
        raise ValueError(
            "sample times must be non-decreasing (the v2 layout preserves write "
            f"order); got {stamp} after {last_time}"
        )
    return stamp


class BunkerShotResultWriter:
    """Writes BunkerShot3D results in schema v2.

    Example:
        >>> writer = BunkerShotResultWriter(path, manifest=manifest)
        >>> writer.write_clubhead_state(0.0, position, quaternion)
        >>> writer.close()
    """

    def __init__(
        self,
        filepath: Path | str,
        *,
        manifest: RunManifest | None = None,
        time_chunk: int = DEFAULT_TIME_CHUNK,
    ) -> None:
        """Create the result file and stamp its schema version.

        Args:
            filepath: Destination path; an existing file is overwritten.
            manifest: Run provenance, written to the root attributes and a
                sibling ``<file>.provenance.json`` when the writer is closed.
            time_chunk: Frames per HDF5 chunk along the time axis.

        Raises:
            ValueError: If ``time_chunk`` is not positive.
        """
        require(time_chunk > 0, "time_chunk must be positive", value=time_chunk)
        self.filepath = Path(filepath)
        self._manifest = manifest
        self._started = _time.perf_counter()
        self._closed = False
        # track_order keeps schema_version first among the root attributes.
        self.file = h5py.File(self.filepath, "w", track_order=True)
        self.file.attrs[SCHEMA_VERSION_ATTR] = np.int64(SCHEMA_VERSION)
        self.file.attrs[FORMAT_ATTR] = FORMAT_NAME

        self.clubhead_group = self.file.create_group(_CLUBHEAD)
        self.wrench_group = self.file.create_group(_WRENCH)
        self.grains_group = self.file.create_group(_GRAINS)
        self._clubhead = _Stream(
            self.clubhead_group, {"position": 3, "orientation": 4}, time_chunk
        )
        self._wrench = _Stream(self.wrench_group, {"force": 3, "torque": 3}, time_chunk)
        self._grains = _RaggedStream(self.grains_group, time_chunk)

    def __enter__(self) -> BunkerShotResultWriter:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file on context exit."""
        self.close()

    def write_clubhead_state(
        self, time: float, position: np.ndarray, orientation_quat: np.ndarray
    ) -> None:
        """Append one clubhead state.

        Args:
            time: Sample time [s]; must not go backwards.
            position: ``(3,)`` clubhead origin [m].
            orientation_quat: ``(4,)`` unit quaternion, scalar-first.

        Raises:
            ValueError: If a shape is wrong or ``time`` goes backwards.
        """
        self._clubhead.append(
            time, {"position": position, "orientation": orientation_quat}
        )

    def write_contact_wrench(
        self, time: float, force: np.ndarray, torque: np.ndarray
    ) -> None:
        """Append one contact wrench.

        Args:
            time: Sample time [s]; must not go backwards.
            force: ``(3,)`` world-frame force [N].
            torque: ``(3,)`` world-frame torque [N.m].

        Raises:
            ValueError: If a shape is wrong or ``time`` goes backwards.
        """
        self._wrench.append(time, {"force": force, "torque": torque})

    def write_grain_state(
        self, time: float, positions: np.ndarray, velocities: np.ndarray
    ) -> None:
        """Append one (possibly downsampled) grain frame.

        Args:
            time: Sample time [s]; must not go backwards.
            positions: ``(N, 3)`` grain positions [m].
            velocities: ``(N, 3)`` grain velocities [m/s].

        Raises:
            ValueError: If a shape is wrong or ``time`` goes backwards.
        """
        self._grains.append(time, positions, velocities)

    def write_sand_field(
        self,
        payload: SandFieldPayload,
        *,
        compression: str = "",
        compression_level: int = 4,
    ) -> None:
        """Write the sand-field payload, once, as whole datasets.

        Unlike the streams, a field is produced by one march and consumed
        whole, so it is written in one call rather than appended: an
        appender would buy nothing and would let a half-written field --
        one with a digest covering frames that are not there -- reach
        disk.

        Args:
            payload: The arrays, metadata and digest.
            compression: HDF5 filter name, or ``""`` for none. Fields are
                the only datasets in this format large enough for a
                filter to be worth its CPU.
            compression_level: Filter level, where the filter takes one.

        Raises:
            ValueError: If a field group already exists, if an array is
                not named in the schema, if an array's leading axis does
                not match the time axis, or if the metadata or digest is
                empty.
        """
        if FIELD_GROUP in self.file:
            raise ValueError(
                f"/{FIELD_GROUP} is already written; a result carries one field, "
                "and a second write would leave the digest covering neither"
            )
        if not payload.metadata.strip():
            raise ValueError(
                "a sand field must carry its metadata: the fidelity tier and "
                "validity status live there, not in the filename"
            )
        if not payload.digest.strip():
            raise ValueError(
                "a sand field must carry its content digest, or its declared "
                "tier could be edited without trace"
            )
        times = np.asarray(payload.time_s, dtype=np.float64).reshape(-1)
        if times.size == 0:
            raise ValueError("a sand field must have at least one frame")
        if np.any(np.diff(times) < 0.0):
            raise ValueError(
                "sand field times must be non-decreasing; an out-of-order field "
                "animates backwards through the impact"
            )
        for name, array in payload.arrays.items():
            if name not in _FIELD_ARRAYS:
                raise ValueError(
                    f"unknown sand field array {name!r}; this schema stores "
                    f"{list(_FIELD_ARRAYS)}"
                )
            block = np.asarray(array)
            if block.ndim < 2 or block.shape[0] != times.size:
                raise ValueError(
                    f"{name} must have shape (T, ...) with T = {times.size}, got "
                    f"{block.shape}"
                )

        group = self.file.create_group(FIELD_GROUP)
        group.attrs[FIELD_METADATA_ATTR] = payload.metadata
        group.attrs[FIELD_DIGEST_ATTR] = payload.digest
        group.create_dataset(_TIME, data=times)
        options: dict[str, Any] = {}
        if compression:
            options["compression"] = compression
            if compression == "gzip":
                options["compression_opts"] = int(compression_level)
        for name in _FIELD_ARRAYS:
            if name not in payload.arrays:
                continue
            # Copied, because a caller may hand a live view of solver state.
            block = np.array(payload.arrays[name], copy=True)
            group.create_dataset(
                name,
                data=block,
                chunks=(1, *block.shape[1:]) if compression else None,
                **options,
            )

    def set_manifest(self, manifest: RunManifest) -> None:
        """Attach (or replace) the run manifest before closing.

        Args:
            manifest: Run provenance to persist.
        """
        self._manifest = manifest

    def close(self) -> None:
        """Flush buffers, persist the manifest, and close the file.

        Postconditions:
            The file holds contiguous datasets for every stream written; when a
            manifest was supplied, the root carries ``manifest_*`` attributes
            and a sibling ``<file>.provenance.json`` exists.
        """
        if self._closed:
            return
        self._clubhead.flush()
        self._wrench.flush()
        self._grains.flush()
        manifest = self._manifest
        if manifest is not None:
            elapsed = _time.perf_counter() - self._started
            manifest = manifest.with_wall_clock(max(elapsed, 1e-9))
            manifest.write_attrs(self.file)
        self.file.close()
        self._closed = True
        if manifest is not None:
            manifest.write_sidecar(
                self.filepath, artifact_extra={"schema_version": SCHEMA_VERSION}
            )


def _as_text(value: Any) -> str:
    """Return an HDF5 string attribute as ``str``.

    h5py hands back ``bytes`` for fixed-length string attributes and
    ``str`` for variable-length ones; ``str(b"...")`` would silently
    produce ``"b'...'"`` and turn a digest comparison into a permanent
    mismatch, so the two cases are separated here rather than assumed.

    Args:
        value: The raw attribute value.

    Returns:
        The value as text.
    """
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_schema_version(handle: h5py.File, path: Path) -> int:
    """Return the schema version of an open result file.

    A file without the attribute is legacy v1: the group-per-timestep layout
    predates versioning.

    Args:
        handle: Open HDF5 file.
        path: Source path, for diagnostics.

    Returns:
        The file's schema version.

    Raises:
        ValueError: If the recorded version is not supported.
    """
    raw = handle.attrs.get(SCHEMA_VERSION_ATTR)
    if raw is None:
        return 1
    version = int(raw)
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"unsupported BunkerShot3D result schema version {version} in {path}; "
            f"this reader accepts {list(SUPPORTED_SCHEMA_VERSIONS)}"
        )
    return version


def _require_group(handle: h5py.File, name: str, path: Path) -> h5py.Group:
    """Return group ``name``, or raise if the file is not a result file.

    Args:
        handle: Open HDF5 file.
        name: Group name.
        path: Source path, for diagnostics.

    Returns:
        The requested group.

    Raises:
        ValueError: If the group is absent.
    """
    group = handle.get(name)
    if group is None:
        raise ValueError(f"not a BunkerShot3D result: missing /{name} group in {path}")
    return group


def _read_v2_column(group: h5py.Group, name: str, width: int | None) -> np.ndarray:
    """Return dataset ``name`` from ``group``, or an empty array if absent.

    Args:
        group: Source group.
        name: Dataset name.
        width: Row width, or ``None`` for a 1-D dataset.

    Returns:
        The dataset contents as a float64 array.
    """
    dataset = group.get(name)
    if dataset is None:
        shape = (0,) if width is None else (0, width)
        return np.empty(shape, dtype=float)
    return np.asarray(dataset[()], dtype=float)


def _v1_frames(group: h5py.Group, path: Path) -> list[h5py.Group]:
    """Return v1 per-timestep subgroups ordered by their numeric ``time`` attr.

    Sorting on the ``time`` attribute -- not on the ``t_<time>`` key string --
    is the fix for the ordering break at t >= 10 s.

    Args:
        group: A v1 stream group.
        path: Source path, for diagnostics.

    Returns:
        The subgroups in ascending time order.

    Raises:
        ValueError: If a subgroup has no ``time`` attribute.
    """
    frames = []
    for key in group:
        subgroup = group[key]
        if "time" not in subgroup.attrs:
            raise ValueError(
                f"legacy frame /{group.name}/{key} in {path} has no 'time' "
                "attribute, so its position in the sequence is unknowable"
            )
        frames.append((float(subgroup.attrs["time"]), key, subgroup))
    frames.sort(key=lambda item: (item[0], item[1]))
    return [subgroup for _, _, subgroup in frames]


def _read_v1_stream(
    group: h5py.Group, names: tuple[str, str], path: Path
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Read a v1 stream as ordered times plus the two per-frame arrays.

    Args:
        group: A v1 stream group.
        names: The two dataset names inside each frame.
        path: Source path, for diagnostics.

    Returns:
        ``(times, first_arrays, second_arrays)``.
    """
    frames = _v1_frames(group, path)
    times = np.array([float(frame.attrs["time"]) for frame in frames], dtype=float)
    first = [np.asarray(frame[names[0]][()], dtype=float) for frame in frames]
    second = [np.asarray(frame[names[1]][()], dtype=float) for frame in frames]
    return times, first, second


def _stack(arrays: list[np.ndarray], width: int) -> np.ndarray:
    """Stack per-frame arrays into a ``(T, width)`` block.

    Args:
        arrays: One ``(width,)`` array per frame.
        width: Expected row width.

    Returns:
        The stacked array, empty-shaped when there are no frames.
    """
    if not arrays:
        return np.empty((0, width), dtype=float)
    return np.stack(arrays, axis=0)


class BunkerShotResultReader:
    """Reads BunkerShot3D results, accepting schema v1 and v2.

    v1 files are migrated on read: frames are ordered by their numeric time and
    returned as contiguous arrays. The source file is never modified.
    """

    def __init__(self, filepath: Path | str) -> None:
        """Open a result file and read its schema version first.

        Args:
            filepath: Source path.

        Raises:
            ValueError: If the file records an unsupported schema version.
        """
        self.filepath = Path(filepath)
        self.file = h5py.File(self.filepath, "r")
        try:
            self._schema_version = _read_schema_version(self.file, self.filepath)
        except Exception:
            self.file.close()
            raise
        self._manifest = RunManifest.read_attrs(self.file)

    def __enter__(self) -> BunkerShotResultReader:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the file on context exit."""
        self.close()

    @property
    def schema_version(self) -> int:
        """Schema version of the file as stored on disk (1 or 2)."""
        return self._schema_version

    @property
    def manifest(self) -> RunManifest | None:
        """Run manifest recorded in the file, if any (v2 only)."""
        return self._manifest

    def read_clubhead_states(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every clubhead state in time order.

        Returns:
            ``(times (T,), positions (T, 3), orientations (T, 4))``.

        Raises:
            ValueError: If the file has no ``/clubhead`` group.
        """
        group = _require_group(self.file, _CLUBHEAD, self.filepath)
        if self._schema_version == 1:
            times, positions, quats = _read_v1_stream(
                group, ("position", "orientation"), self.filepath
            )
            return times, _stack(positions, 3), _stack(quats, 4)
        return (
            _read_v2_column(group, _TIME, None),
            _read_v2_column(group, "position", 3),
            _read_v2_column(group, "orientation", 4),
        )

    def read_contact_wrenches(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Read every contact wrench in time order.

        Returns:
            ``(times (T,), forces (T, 3), torques (T, 3))``.

        Raises:
            ValueError: If the file has no ``/wrench`` group.
        """
        group = _require_group(self.file, _WRENCH, self.filepath)
        if self._schema_version == 1:
            times, forces, torques = _read_v1_stream(
                group, ("force", "torque"), self.filepath
            )
            return times, _stack(forces, 3), _stack(torques, 3)
        return (
            _read_v2_column(group, _TIME, None),
            _read_v2_column(group, "force", 3),
            _read_v2_column(group, "torque", 3),
        )

    def read_grain_states(
        self,
    ) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
        """Read every grain frame in time order.

        Positions and velocities are returned as lists because the recorded
        particle count may vary between frames.

        Returns:
            ``(times (T,), positions [(N_i, 3)], velocities [(N_i, 3)])``.

        Raises:
            ValueError: If the file has no ``/grains`` group.
        """
        group = _require_group(self.file, _GRAINS, self.filepath)
        if self._schema_version == 1:
            times, positions, velocities = _read_v1_stream(
                group, ("positions", "velocities"), self.filepath
            )
            return times, positions, velocities

        times = _read_v2_column(group, _TIME, None)
        counts_dataset = group.get(_COUNTS)
        if counts_dataset is None:
            return times, [], []
        counts = np.asarray(counts_dataset[()], dtype=np.int64)
        edges = np.cumsum(counts)[:-1]
        positions = np.split(_read_v2_column(group, "positions", 3), edges)
        velocities = np.split(_read_v2_column(group, "velocities", 3), edges)
        return times, list(positions), list(velocities)

    def read_sand_field(self) -> SandFieldPayload | None:
        """Read the sand-field payload, or ``None`` if none was recorded.

        A v1 or v2 file, and a v3 file whose run recorded no field, both
        return ``None``: "no field here" is an answer, not a failure.

        Returns:
            The payload, or ``None``.

        Raises:
            ValueError: If the group exists but has lost its metadata or
                its digest, which would leave the arrays with no
                recorded tier at all.
        """
        group = self.file.get(FIELD_GROUP)
        if group is None:
            return None
        metadata = group.attrs.get(FIELD_METADATA_ATTR)
        digest = group.attrs.get(FIELD_DIGEST_ATTR)
        if metadata is None or digest is None:
            raise ValueError(
                f"/{FIELD_GROUP} in {self.filepath} has no "
                f"{FIELD_METADATA_ATTR}/{FIELD_DIGEST_ATTR} attribute, so the "
                "tier and validity of these arrays are unknowable"
            )
        arrays = {
            name: np.asarray(group[name][()]) for name in _FIELD_ARRAYS if name in group
        }
        return SandFieldPayload(
            time_s=np.asarray(group[_TIME][()], dtype=float),
            arrays=arrays,
            metadata=_as_text(metadata),
            digest=_as_text(digest),
        )

    def close(self) -> None:
        """Close the HDF5 file."""
        self.file.close()
