"""Reading a particle trajectory back, and refusing it unless it is whole.

Issue #9545 (folded-in #9692).  The writer half lives in
:mod:`bunkershot3d.fields.trajectory`; this module is the half a renderer
or an analysis touches, so it is where the trust decisions are made.

Opening a :class:`ParticleTrajectoryReader` verifies, in order:

1. the finalisation marker exists, parses, carries every field, is the
   current schema version and vouches for *this* file
   (:class:`~bunkershot3d.fields.trajectory.TrajectoryIncompleteError`,
   :class:`~bunkershot3d.fields.trajectory.TrajectoryVersionError`);
2. the file's own version and format attributes agree;
3. the metadata text and every array match their recorded SHA-256,
   streamed in bounded blocks;
4. the roster, the frame masks and the contact table satisfy the same
   contract the writer enforced -- so an edited-and-re-signed file still
   cannot duplicate, swap or resurrect a grain.

Only then are frames served, lazily, one block at a time.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DStateError, BunkerShot3DValueError
from .schema import GrainRoster, ParticleKind
from .standing import FieldProvenance
from .trajectory import (
    ALL_SPECS,
    ARRAY_DATASETS,
    DEFAULT_ROWS_PER_CHUNK,
    FORMAT_ATTR,
    METADATA_ATTR,
    TRAJECTORY_FORMAT,
    TRAJECTORY_SCHEMA_VERSION,
    VERSION_ATTR,
    ContactEvent,
    FrameState,
    StreamDigest,
    TrajectoryIncompleteError,
    TrajectoryIntegrityError,
    TrajectoryVersionError,
    checked_frame,
    final_marker_path,
)

__all__ = ["ParticleTrajectoryReader"]

_MARKER_KEYS = (
    "format",
    "trajectory_schema_version",
    "artifact",
    "n_frames",
    "n_grains",
    "n_contacts",
    "metadata_sha256",
    "array_sha256",
)


def _require_version(value: Any, source: str) -> None:
    """Refuse another schema version with an actionable message."""
    if value is None or int(value) == TRAJECTORY_SCHEMA_VERSION:
        return
    raise TrajectoryVersionError(
        f"{source} was written by trajectory schema v{int(value)} and this code "
        f"reads v{TRAJECTORY_SCHEMA_VERSION}. Nothing was tampered with; "
        "regenerate the trajectory from the run its provenance names. Grain "
        "identity is never inferred for a file this code did not write."
    )


def _read_marker(path: Path) -> dict[str, Any]:
    """The finalisation marker of ``path``, refused unless complete."""
    marker_path = final_marker_path(path)
    if not marker_path.is_file():
        raise TrajectoryIncompleteError(
            f"{path.name} has no finalisation marker ({marker_path.name}); the "
            "write was interrupted, cancelled or is still running"
        )
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TrajectoryIncompleteError(
            f"{marker_path.name} is a partial or unreadable finalisation marker"
        ) from exc
    if not isinstance(marker, dict):
        raise TrajectoryIncompleteError(f"{marker_path.name} is a partial marker")
    _require_version(marker.get("trajectory_schema_version"), marker_path.name)
    missing = [key for key in _MARKER_KEYS if key not in marker]
    digests = marker.get("array_sha256")
    if isinstance(digests, dict):
        missing += [key for key in ARRAY_DATASETS if key not in digests]
    if missing:
        raise TrajectoryIncompleteError(
            f"{marker_path.name} is a partial marker, missing {missing}"
        )
    if marker["format"] != TRAJECTORY_FORMAT or marker["artifact"] != path.name:
        raise TrajectoryIncompleteError(
            f"{marker_path.name} vouches for {marker['artifact']!r} in format "
            f"{marker['format']!r}, not for {path.name!r}"
        )
    if not path.is_file():
        raise TrajectoryIncompleteError(f"{path.name} is missing beside its marker")
    return marker


def _as_text(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _stream_digest(handle: h5py.File, key: str, dtype: Any, rows: int) -> str:
    """SHA-256 of one stored array, read ``rows`` at a time."""
    digest = StreamDigest(dtype)
    dataset = handle.get(key)
    if dataset is not None:
        for start in range(0, int(dataset.shape[0]), rows):
            digest.update(dataset[start : start + rows])
    return digest.hexdigest()


def _verify_digests(
    handle: h5py.File, marker: dict[str, Any], rows: int, name: str
) -> str:
    """Check the metadata and every array against the marker.

    Returns:
        The verified metadata text.
    """
    _require_version(handle.attrs.get(VERSION_ATTR), name)
    if _as_text(handle.attrs.get(FORMAT_ATTR, "")) != TRAJECTORY_FORMAT:
        raise TrajectoryIntegrityError(f"{name} is not a particle trajectory")
    metadata = _as_text(handle.attrs.get(METADATA_ATTR, ""))
    if (
        hashlib.sha256(metadata.encode("utf-8")).hexdigest()
        != (marker["metadata_sha256"])
    ):
        raise TrajectoryIntegrityError(
            f"the metadata of {name} does not match its recorded digest; its "
            "provenance, verdict or units were changed after it was written"
        )
    for spec in ALL_SPECS:
        if (
            _stream_digest(handle, spec.key, spec.dtype, rows)
            != (marker["array_sha256"][spec.key])
        ):
            raise TrajectoryIntegrityError(
                f"{spec.key} in {name} does not match its recorded digest"
            )
    return metadata


def _length(handle: h5py.File, key: str) -> int:
    dataset = handle.get(key)
    return 0 if dataset is None else int(dataset.shape[0])


class ParticleTrajectoryReader:
    """A verified, lazily-read particle trajectory.

    Example:
        >>> with ParticleTrajectoryReader(path) as reader:
        ...     track = reader.positions_of(17)
    """

    def __init__(
        self, path: Path | str, *, rows_per_block: int = DEFAULT_ROWS_PER_CHUNK
    ) -> None:
        """Name the artifact; nothing is read until :meth:`open`.

        Raises:
            BunkerShot3DValueError: If ``rows_per_block`` is not positive.
        """
        if int(rows_per_block) < 1:
            raise BunkerShot3DValueError(
                f"rows_per_block must be positive, got {rows_per_block!r}"
            )
        self._path = Path(path)
        self._rows = int(rows_per_block)
        self._file: h5py.File | None = None
        self._metadata: dict[str, Any] = {}
        self._roster: GrainRoster | None = None
        self._time_s: NDArray[np.float64] = np.empty(0)

    def __enter__(self) -> ParticleTrajectoryReader:
        return self.open()

    def __exit__(self, *_: object) -> None:
        self.close()

    def open(self) -> ParticleTrajectoryReader:
        """Verify the artifact and keep it open for frame reads.

        Raises:
            TrajectoryIncompleteError: Missing, partial or foreign marker.
            TrajectoryVersionError: Another schema version.
            TrajectoryIntegrityError: A digest mismatch, or a roster, mask,
                time or contact that breaks the trajectory contract.
        """
        marker = _read_marker(self._path)
        handle = h5py.File(self._path, "r")
        try:
            text = _verify_digests(handle, marker, self._rows, self._path.name)
            self._adopt(handle, marker, json.loads(text))
        except BunkerShot3DValueError as exc:
            handle.close()
            if isinstance(exc, TrajectoryIntegrityError):
                raise
            raise TrajectoryIntegrityError(f"{self._path.name}: {exc}") from exc
        except BaseException:
            handle.close()
            raise
        self._file = handle
        return self

    def _adopt(
        self, handle: h5py.File, marker: dict[str, Any], metadata: dict[str, Any]
    ) -> None:
        """Check structure and contract, then keep what the getters serve."""
        roster = GrainRoster(
            grain_ids=handle["roster/grain_id"][...],
            radius_m=handle["roster/radius_m"][...],
            kind=ParticleKind(str(metadata["particle_kind"])),
        )
        n_frames = int(marker["n_frames"])
        expected = {
            "frames/t": n_frames,
            "frames/positions": n_frames * roster.n_grains,
            "contacts/t": int(marker["n_contacts"]),
        }
        actual = {key: _length(handle, key) for key in expected}
        if actual != expected or n_frames < 1:
            raise TrajectoryIntegrityError(
                f"stored lengths {actual} disagree with the marker {expected}"
            )
        times = np.asarray(handle["frames/t"][...], dtype=np.float64)
        if not bool(np.all(np.isfinite(times))) or bool(np.any(np.diff(times) <= 0)):
            raise TrajectoryIntegrityError("frame times must be strictly increasing")
        self._check_frames(handle, roster.n_grains, n_frames)
        if actual["contacts/t"] and not bool(
            np.all(np.isin(handle["contacts/grain_id"][...], roster.grain_ids))
        ):
            raise TrajectoryIntegrityError("a contact names a grain off the roster")
        self._roster = roster
        self._metadata = metadata
        self._time_s = times

    def _check_frames(self, handle: h5py.File, n_grains: int, n_frames: int) -> None:
        """Re-apply the writer's frame contract, a bounded block at a time."""
        per_block = max(1, self._rows // n_grains)
        previous_lost: NDArray[np.bool_] | None = None
        for first in range(0, n_frames, per_block):
            last = min(n_frames, first + per_block)
            positions, velocity, active, lost = self._rows_of(
                handle, first * n_grains, last * n_grains
            )
            for offset in range(last - first):
                rows = slice(offset * n_grains, (offset + 1) * n_grains)
                try:
                    state = checked_frame(
                        n_grains,
                        positions[rows],
                        velocity[rows],
                        active[rows],
                        lost[rows],
                        previous_lost,
                    )
                except BunkerShot3DValueError as exc:
                    raise TrajectoryIntegrityError(
                        f"frame {first + offset}: {exc}"
                    ) from exc
                previous_lost = state.lost

    @staticmethod
    def _rows_of(
        handle: h5py.File, start: int, stop: int
    ) -> tuple[NDArray[Any], NDArray[Any], NDArray[Any], NDArray[Any]]:
        """Positions, velocities, active and lost rows ``[start, stop)``."""
        return (
            handle["frames/positions"][start:stop],
            handle["frames/velocity"][start:stop],
            np.asarray(handle["frames/active"][start:stop], dtype=bool),
            np.asarray(handle["frames/lost"][start:stop], dtype=bool),
        )

    def close(self) -> None:
        """Release the file; idempotent."""
        if self._file is not None:
            self._file.close()
            self._file = None

    def _require_open(self) -> tuple[h5py.File, GrainRoster]:
        if self._file is None or self._roster is None:
            raise BunkerShot3DStateError("open the trajectory reader first")
        return self._file, self._roster

    @property
    def roster(self) -> GrainRoster:
        """The verified grain roster."""
        return self._require_open()[1]

    @property
    def provenance(self) -> FieldProvenance:
        """Tier, verdict and settings, exactly as written."""
        self._require_open()
        return FieldProvenance.from_dict(self._metadata["provenance"])

    @property
    def metadata(self) -> dict[str, Any]:
        """A copy of the verified metadata."""
        self._require_open()
        return json.loads(json.dumps(self._metadata))

    @property
    def schema_version(self) -> int:
        """The trajectory schema version of the file."""
        self._require_open()
        return int(self._metadata["trajectory_schema_version"])

    @property
    def n_frames(self) -> int:
        """Number of retained frames."""
        self._require_open()
        return int(self._time_s.size)

    @property
    def time_s(self) -> NDArray[np.float64]:
        """``(T,)`` retained frame times, read-only."""
        self._require_open()
        view = self._time_s.view()
        view.flags.writeable = False
        return view

    def frame(self, index: int) -> FrameState:
        """One frame in roster order, inactive grains as NaN.

        Raises:
            BunkerShot3DValueError: If ``index`` is outside the trajectory.
        """
        handle, roster = self._require_open()
        position = int(index)
        if not 0 <= position < self._time_s.size:
            raise BunkerShot3DValueError(
                f"frame {index} is outside the trajectory of {self._time_s.size}"
            )
        width = roster.n_grains
        block = self._rows_of(handle, position * width, (position + 1) * width)
        return checked_frame(width, *block, None)

    def positions_of(self, grain_id: int) -> NDArray[np.float64]:
        """``(T, 3)`` track of one grain by id, NaN while it is inactive.

        Raises:
            BunkerShot3DValueError: If the id is not on the roster.
        """
        handle, roster = self._require_open()
        row = roster.index_of(grain_id)
        step = roster.n_grains
        track = np.array(handle["frames/positions"][row::step], dtype=np.float64)
        active = np.asarray(handle["frames/active"][row::step], dtype=bool)
        track[~active] = np.nan
        return track

    def contacts(self) -> tuple[ContactEvent, ...]:
        """Every contact event, in stored order."""
        handle, _ = self._require_open()
        if _length(handle, "contacts/t") == 0:
            return ()
        columns = [
            handle[f"contacts/{leaf}"][...]
            for leaf in ("t", "grain_id", "body_id", "point", "force")
        ]
        return tuple(
            ContactEvent(
                time_s=float(t),
                grain_id=int(grain),
                body_id=int(body),
                point_m=point,
                force_n=force,
            )
            for t, grain, body, point, force in zip(*columns, strict=True)
        )
