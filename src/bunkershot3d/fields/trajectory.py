"""Versioned 3-D particle trajectories and contact events (issue #9545).

Folded in from #9692.  :class:`~bunkershot3d.fields.schema.FieldLayout`
already permits ``PARTICLE`` samples, but a sand-field series has no
notion of *which* grain a sample is.  This module is the persistent
contract that does: a fixed :class:`~bunkershot3d.fields.schema.GrainRoster`
of stable ids and radii, per-frame positions and velocities in roster
order, ``active``/``lost`` masks, and a table of grain-body contact
events -- streamed to disk in bounded chunks and finalised atomically.

On-disk layout, trajectory schema v1
------------------------------------

One HDF5 file plus a JSON finalisation marker beside it::

    <name>                     HDF5
      .attrs[format]           "bunkershot3d-particle-trajectory"
      .attrs[trajectory_schema_version]  1
      .attrs[metadata]         canonical JSON: kind, provenance, units,
                               retention, omitted channels
      /roster/grain_id         (G,)       int64    stable ids
      /roster/radius_m         (G,)       float64  [m]
      /frames/t                (T,)       float64  [s], strictly increasing
      /frames/positions        (T*G, 3)   float64  [m], NaN when inactive
      /frames/velocity         (T*G, 3)   float64  [m/s], NaN when inactive
      /frames/active           (T*G,)     bool
      /frames/lost             (T*G,)     bool     monotone: lost stays lost
      /contacts/t              (E,)       float64  [s], non-decreasing
      /contacts/grain_id       (E,)       int64    a roster id
      /contacts/body_id        (E,)       int64
      /contacts/point          (E, 3)     float64  [m], world frame
      /contacts/force          (E, 3)     float64  [N], NaN = not recorded
    <name>.final.json          the finalisation marker, written last

Frame ``f``'s grain ``g`` is row ``f * G + g``: the per-frame arrays are
stored flat so they stream through the same
:class:`~bunkershot3d.io.schema.ChunkedAppender` the result writer uses,
at a memory cost fixed by the chunk rather than by the run length.

What a reader may trust, and why
--------------------------------

* **Completion.**  The file is written to ``.<name>.partial``, renamed
  into place only when complete, and the marker is written after the
  rename (atomically, via the shared ``atomic_write_json``).  Starting a
  new write first revokes any old marker, so a cancelled or crashed
  write can never pair a new file with an old promise.
* **Integrity.**  The marker carries a SHA-256 per array and over the
  metadata text.  A changed coordinate, provenance verdict or grain id
  is detected; there is no "load anyway".
* **Identity.**  The roster and the masks are re-validated on read with
  the same checks the writer applied, so a file that was edited and then
  re-signed still cannot swap, duplicate or resurrect a grain.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import h5py
import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json

from ..exceptions import BunkerShot3DStateError, BunkerShot3DValueError
from ..io.schema import ChunkedAppender
from ..provenance.hashing import canonical_json
from .schema import GrainRoster
from .standing import (
    FIELD_SCHEMA_VERSION,
    FieldIntegrityError,
    FieldProvenance,
    RetentionRecord,
)
from .units import TIME_UNIT, VELOCITY_UNIT

__all__ = [
    "ARRAY_DATASETS",
    "DEFAULT_ROWS_PER_CHUNK",
    "FINAL_MARKER_SUFFIX",
    "OMITTED_CHANNELS",
    "TRAJECTORY_FORMAT",
    "TRAJECTORY_SCHEMA_VERSION",
    "ContactEvent",
    "FrameState",
    "ParticleTrajectoryWriter",
    "StreamDigest",
    "TrajectoryIncompleteError",
    "TrajectoryIntegrityError",
    "TrajectoryVersionError",
    "checked_frame",
    "final_marker_path",
]

TRAJECTORY_SCHEMA_VERSION = 1
"""Version of the trajectory artifact. Independent of the field and
container versions: a trajectory is a different artifact, not a field."""

TRAJECTORY_FORMAT = "bunkershot3d-particle-trajectory"
FORMAT_ATTR = "format"
VERSION_ATTR = "trajectory_schema_version"
METADATA_ATTR = "metadata"
FINAL_MARKER_SUFFIX = ".final.json"
DEFAULT_ROWS_PER_CHUNK = 4096

OMITTED_CHANNELS = ("orientation", "angular_velocity", "material_size_bin")
"""Channels v1 does not record. Named in the metadata so a consumer reads
"not recorded" rather than inferring "zero"."""

_DIM = 3


class _Spec(NamedTuple):
    """One stored array: HDF5 group, leaf name, row width, element type."""

    group: str
    leaf: str
    width: int | None
    dtype: Any

    @property
    def key(self) -> str:
        return f"{self.group}/{self.leaf}"


ROSTER_SPECS = (
    _Spec("roster", "grain_id", None, np.int64),
    _Spec("roster", "radius_m", None, np.float64),
)
FRAME_SPECS = (
    _Spec("frames", "t", None, np.float64),
    _Spec("frames", "positions", _DIM, np.float64),
    _Spec("frames", "velocity", _DIM, np.float64),
    _Spec("frames", "active", None, np.bool_),
    _Spec("frames", "lost", None, np.bool_),
)
CONTACT_SPECS = (
    _Spec("contacts", "t", None, np.float64),
    _Spec("contacts", "grain_id", None, np.int64),
    _Spec("contacts", "body_id", None, np.int64),
    _Spec("contacts", "point", _DIM, np.float64),
    _Spec("contacts", "force", _DIM, np.float64),
)
ALL_SPECS = (*ROSTER_SPECS, *FRAME_SPECS, *CONTACT_SPECS)
ARRAY_DATASETS = tuple(spec.key for spec in ALL_SPECS)
"""Every array the marker digests, in a fixed order."""


class TrajectoryIntegrityError(FieldIntegrityError):
    """A stored trajectory's contents do not match what it declares."""


class TrajectoryIncompleteError(TrajectoryIntegrityError):
    """The trajectory was never finalised: interrupted, cancelled, or the
    marker is missing, partial or names another artifact."""


class TrajectoryVersionError(TrajectoryIntegrityError):
    """The trajectory was written by an incompatible schema version."""


def final_marker_path(path: Path | str) -> Path:
    """Where the finalisation marker of ``path`` lives."""
    target = Path(path)
    return target.with_name(f"{target.name}{FINAL_MARKER_SUFFIX}")


def _partial_path(path: Path) -> Path:
    """The staging file a write goes to before it is renamed into place."""
    return path.with_name(f".{path.name}.partial")


class StreamDigest:
    """SHA-256 over an array delivered in successive blocks.

    Blocks are hashed as C-order bytes of the stored element type, so the
    digest of a streamed write equals the digest of the whole array read
    back in any block size.
    """

    def __init__(self, dtype: Any) -> None:
        self._dtype = np.dtype(dtype)
        self._hash = hashlib.sha256()

    def update(self, block: NDArray[Any]) -> None:
        """Fold the next block in."""
        data = np.ascontiguousarray(block, dtype=self._dtype)
        self._hash.update(data.tobytes(order="C"))

    def hexdigest(self) -> str:
        """The digest so far."""
        return self._hash.hexdigest()


@dataclass(frozen=True)
class ContactEvent:
    """One grain-body contact, with an explicit reference point.

    Attributes:
        time_s: Simulation time of the event.
        grain_id: A roster id.
        body_id: The contacting body (club, ball, ...), non-negative.
        point_m: ``(3,)`` world-frame contact point; always known.
        force_n: ``(3,)`` force on the grain, or all-NaN when the tier did
            not resolve it -- never zeros standing in for "unknown".
    """

    time_s: float
    grain_id: int
    body_id: int
    point_m: NDArray[np.float64]
    force_n: NDArray[np.float64]

    def __post_init__(self) -> None:
        stamp = float(self.time_s)
        point = np.array(self.point_m, dtype=np.float64, copy=True).reshape(-1)
        force = np.array(self.force_n, dtype=np.float64, copy=True).reshape(-1)
        if not np.isfinite(stamp):
            raise BunkerShot3DValueError(f"contact time must be finite, got {stamp}")
        if int(self.grain_id) < 0 or int(self.body_id) < 0:
            raise BunkerShot3DValueError("contact grain and body ids must be >= 0")
        if point.shape != (_DIM,) or not bool(np.all(np.isfinite(point))):
            raise BunkerShot3DValueError(
                "point_m must be a finite (3,) world point; a contact with no "
                "reference point has no meaningful wrench"
            )
        finite = np.isfinite(force)
        if force.shape != (_DIM,) or not (bool(finite.all()) or not finite.any()):
            raise BunkerShot3DValueError(
                "force_n must be a finite (3,) vector, or all NaN when unresolved"
            )
        point.flags.writeable = False
        force.flags.writeable = False
        object.__setattr__(self, "time_s", stamp)
        object.__setattr__(self, "grain_id", int(self.grain_id))
        object.__setattr__(self, "body_id", int(self.body_id))
        object.__setattr__(self, "point_m", point)
        object.__setattr__(self, "force_n", force)

    @property
    def force_available(self) -> bool:
        """Whether the tier resolved the contact force."""
        return bool(np.all(np.isfinite(self.force_n)))


class FrameState(NamedTuple):
    """One retained instant, in roster order, after validation."""

    positions_m: NDArray[np.float64]
    velocity_m_s: NDArray[np.float64]
    active: NDArray[np.bool_]
    lost: NDArray[np.bool_]


def checked_frame(
    n_grains: int,
    positions_m: NDArray[np.float64],
    velocity_m_s: NDArray[np.float64],
    active: NDArray[np.bool_],
    lost: NDArray[np.bool_],
    previous_lost: NDArray[np.bool_] | None,
) -> FrameState:
    """Validate one frame and return it with inactive state set to NaN.

    Shared by the writer and the reader, so a file is held on read to
    exactly the contract it was written under.

    Raises:
        BunkerShot3DValueError: On a wrong shape, a non-boolean mask, a
            grain both active and lost, a lost grain that returns, or a
            non-finite active position or velocity.
    """
    positions = np.array(positions_m, dtype=np.float64, copy=True)
    velocity = np.array(velocity_m_s, dtype=np.float64, copy=True)
    active_mask = np.asarray(active)
    lost_mask = np.asarray(lost)
    for name, array in (("positions_m", positions), ("velocity_m_s", velocity)):
        if array.shape != (n_grains, _DIM):
            raise BunkerShot3DValueError(
                f"{name} must have shape ({n_grains}, 3), got {array.shape}"
            )
    for name, mask in (("active", active_mask), ("lost", lost_mask)):
        if mask.shape != (n_grains,) or mask.dtype != np.bool_:
            raise BunkerShot3DValueError(
                f"{name} must be a boolean mask of shape ({n_grains},), got "
                f"{mask.dtype} {mask.shape}"
            )
    if bool(np.any(active_mask & lost_mask)):
        raise BunkerShot3DValueError("a grain cannot be both active and lost")
    if previous_lost is not None and bool(np.any(previous_lost & ~lost_mask)):
        raise BunkerShot3DValueError(
            "a lost grain cannot return; lost is permanent so an escaped grain "
            "is never re-identified as a new one"
        )
    live = np.concatenate([positions[active_mask], velocity[active_mask]])
    if not bool(np.all(np.isfinite(live))):
        raise BunkerShot3DValueError(
            "active grains must have finite positions and velocities"
        )
    positions[~active_mask] = np.nan
    velocity[~active_mask] = np.nan
    return FrameState(positions, velocity, active_mask.copy(), lost_mask.copy())


def trajectory_metadata(
    roster: GrainRoster,
    provenance: FieldProvenance,
    retention: RetentionRecord | None,
    counts: tuple[int, int],
) -> dict[str, Any]:
    """The JSON-safe metadata a trajectory carries and digests.

    Args:
        roster: The grains.
        provenance: Tier, verdict, solver and settings -- reused from the
            field schema so a trajectory and a field state standing alike.
        retention: What was dropped to get here, when the producer knows.
        counts: ``(n_frames, n_contacts)``.
    """
    n_frames, n_contacts = counts
    return {
        "format": TRAJECTORY_FORMAT,
        "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
        "field_schema_version": FIELD_SCHEMA_VERSION,
        "particle_kind": roster.kind.value,
        "n_grains": roster.n_grains,
        "n_frames": int(n_frames),
        "n_contacts": int(n_contacts),
        "dimension": _DIM,
        "row_order": "frame-major: frame f, grain g at row f * n_grains + g",
        "provenance": provenance.to_dict(),
        "retention": None if retention is None else retention.to_dict(),
        "omitted_channels": list(OMITTED_CHANNELS),
        "units": {
            "time": TIME_UNIT,
            "length": "m",
            "velocity": VELOCITY_UNIT,
            "force": "N",
        },
    }


class ParticleTrajectoryWriter:
    """Stream a particle trajectory to disk and finalise it atomically.

    Use as a context manager: a clean exit finalises, an exception aborts
    and leaves nothing a reader would accept.

    Example:
        >>> with ParticleTrajectoryWriter(path, roster, provenance) as w:
        ...     w.append_frame(t, positions, velocity, active, lost)
    """

    def __init__(
        self,
        path: Path | str,
        roster: GrainRoster,
        provenance: FieldProvenance,
        *,
        rows_per_chunk: int = DEFAULT_ROWS_PER_CHUNK,
    ) -> None:
        """Revoke any old marker and open the staging file.

        Raises:
            BunkerShot3DValueError: If ``rows_per_chunk`` is not positive
                or the destination directory does not exist.
        """
        if int(rows_per_chunk) < 1:
            raise BunkerShot3DValueError(
                f"rows_per_chunk must be positive, got {rows_per_chunk!r}"
            )
        self._path = Path(path)
        parent_dir = self._path.parent
        if not parent_dir.is_dir():
            raise BunkerShot3DValueError(f"{parent_dir} is not a directory")
        self._roster = roster
        self._provenance = provenance
        self._chunk = int(rows_per_chunk)
        final_marker_path(self._path).unlink(missing_ok=True)
        self._staging = _partial_path(self._path)
        self._file: h5py.File | None = h5py.File(self._staging, "w", track_order=True)
        self._file.attrs[FORMAT_ATTR] = TRAJECTORY_FORMAT
        self._file.attrs[VERSION_ATTR] = np.int64(TRAJECTORY_SCHEMA_VERSION)
        self._digests = {spec.key: StreamDigest(spec.dtype) for spec in ALL_SPECS}
        self._appenders = {
            spec.key: ChunkedAppender(
                self._file.require_group(spec.group),
                spec.leaf,
                spec.width,
                self._chunk,
                dtype=spec.dtype,
            )
            for spec in (*FRAME_SPECS, *CONTACT_SPECS)
        }
        self._write_roster(self._file)
        self._n_frames = 0
        self._n_contacts = 0
        self._last_time: float | None = None
        self._last_contact_time: float | None = None
        self._previous_lost: NDArray[np.bool_] | None = None
        self._peak_buffered = 0

    def __enter__(self) -> ParticleTrajectoryWriter:
        return self

    def __exit__(self, exc_type: object, *_: object) -> None:
        if exc_type is not None:
            self.abort()
        elif self._file is not None:
            self.finalize()

    @property
    def peak_buffered_rows(self) -> int:
        """Most rows any one stream has held in memory at once."""
        return self._peak_buffered

    def _write_roster(self, handle: h5py.File) -> None:
        """Store the roster whole; it is small and fixed for the run."""
        arrays = (self._roster.grain_ids, self._roster.radius_m)
        for spec, array in zip(ROSTER_SPECS, arrays, strict=True):
            handle.require_group(spec.group).create_dataset(spec.leaf, data=array)
            self._digests[spec.key].update(array)

    def _require_open(self) -> h5py.File:
        if self._file is None:
            raise BunkerShot3DStateError(
                "this trajectory writer is already finalised or aborted"
            )
        return self._file

    def _extend(self, key: str, block: NDArray[Any]) -> None:
        """Digest and buffer one block, tracking the memory high-water mark."""
        appender = self._appenders[key]
        rows = int(np.shape(block)[0])
        self._peak_buffered = max(self._peak_buffered, appender.buffered_rows + rows)
        self._digests[key].update(block)
        appender.extend(np.asarray(block))

    def append_frame(
        self,
        time_s: float,
        positions_m: NDArray[np.float64],
        velocity_m_s: NDArray[np.float64],
        active: NDArray[np.bool_],
        lost: NDArray[np.bool_],
    ) -> None:
        """Append one retained frame in roster order.

        Preconditions (checked unconditionally): ``time_s`` is finite and
        strictly after the previous frame; arrays are ``(G, 3)``; masks
        are boolean ``(G,)``; see :func:`checked_frame` for the rest.

        Postcondition: inactive grains are stored as NaN.
        """
        self._require_open()
        stamp = float(time_s)
        if not np.isfinite(stamp) or (
            self._last_time is not None and stamp <= self._last_time
        ):
            raise BunkerShot3DValueError(
                f"frame times must be finite and strictly increasing; got {stamp} "
                f"after {self._last_time}"
            )
        state = checked_frame(
            self._roster.n_grains,
            positions_m,
            velocity_m_s,
            active,
            lost,
            self._previous_lost,
        )
        self._extend("frames/t", np.array([stamp]))
        self._extend("frames/positions", state.positions_m)
        self._extend("frames/velocity", state.velocity_m_s)
        self._extend("frames/active", state.active)
        self._extend("frames/lost", state.lost)
        self._last_time = stamp
        self._previous_lost = state.lost
        self._n_frames += 1

    def append_contact(self, event: ContactEvent) -> None:
        """Append one contact event.

        Raises:
            BunkerShot3DValueError: If the grain is not on the roster or
                the event is earlier than the previous one.
        """
        self._require_open()
        if not self._roster.contains(event.grain_id):
            raise BunkerShot3DValueError(
                f"contact names grain {event.grain_id}, which is not on the roster"
            )
        if self._last_contact_time is not None and (
            event.time_s < self._last_contact_time
        ):
            raise BunkerShot3DValueError("contact events must be non-decreasing")
        self._extend("contacts/t", np.array([event.time_s]))
        self._extend("contacts/grain_id", np.array([event.grain_id], dtype=np.int64))
        self._extend("contacts/body_id", np.array([event.body_id], dtype=np.int64))
        self._extend("contacts/point", event.point_m.reshape(1, _DIM))
        self._extend("contacts/force", event.force_n.reshape(1, _DIM))
        self._last_contact_time = event.time_s
        self._n_contacts += 1

    def _check_retention(self, retention: RetentionRecord | None) -> None:
        if retention is None:
            return
        if retention.frames_kept != self._n_frames:
            raise BunkerShot3DValueError(
                f"retention frames_kept={retention.frames_kept} but "
                f"{self._n_frames} frames were written"
            )
        if retention.samples_kept != self._roster.n_grains:
            raise BunkerShot3DValueError(
                f"retention samples_kept={retention.samples_kept} but the roster "
                f"has {self._roster.n_grains} grains"
            )

    def finalize(self, *, retention: RetentionRecord | None = None) -> Path:
        """Close, rename into place, then write the marker -- in that order.

        Postconditions: the artifact exists at its destination; its marker
        exists, was written after it, and digests every array and the
        metadata; the staging file is gone.

        Raises:
            BunkerShot3DValueError: If no frame was written, or
                ``retention`` disagrees with what was written. The writer
                stays open, so the caller can still abort.
        """
        handle = self._require_open()
        if self._n_frames == 0:
            raise BunkerShot3DValueError(
                "a trajectory with no frames records no motion; abort instead"
            )
        self._check_retention(retention)
        for appender in self._appenders.values():
            appender.flush()
        metadata = canonical_json(
            trajectory_metadata(
                self._roster,
                self._provenance,
                retention,
                (self._n_frames, self._n_contacts),
            )
        )
        handle.attrs[METADATA_ATTR] = metadata
        handle.close()
        self._file = None
        self._staging.replace(self._path)
        atomic_write_json(final_marker_path(self._path), self._marker(metadata))
        return self._path

    def _marker(self, metadata: str) -> Mapping[str, Any]:
        return {
            "format": TRAJECTORY_FORMAT,
            "trajectory_schema_version": TRAJECTORY_SCHEMA_VERSION,
            "artifact": self._path.name,
            "n_frames": self._n_frames,
            "n_grains": self._roster.n_grains,
            "n_contacts": self._n_contacts,
            "metadata_sha256": hashlib.sha256(metadata.encode("utf-8")).hexdigest(),
            "array_sha256": {key: d.hexdigest() for key, d in self._digests.items()},
        }

    def abort(self) -> None:
        """Discard the write; idempotent and a no-op after finalising.

        Postcondition: no staging file and no marker remain for this path.
        """
        if self._file is None:
            return
        self._file.close()
        self._file = None
        self._staging.unlink(missing_ok=True)
