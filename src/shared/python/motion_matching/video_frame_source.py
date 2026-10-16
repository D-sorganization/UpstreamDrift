"""One marker frame source for every engine's replayable results (#12878).

The explorer, the matched-swing browser and the export service all need the
same thing from a results archive: sample times, the model's marker positions,
the target markers and a validity mask. Engines save that in two NPZ layouts:

* the checksummed ``MatchedSwingCandidate`` package
  (``model_markers_m`` / ``target_markers_m`` / ``marker_validity``), and
* the replay record written by the per-engine forward replays
  (``markers_m`` / ``target_m`` / ``valid``).

``load_marker_frame_source`` normalises both into :class:`MarkerFrameSource`,
so exactly one writer (``motion_matching.export``) serves every engine, and an
archive that cannot be played raises :class:`VideoSourceError` naming the
missing arrays instead of leaving a blank view.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "MarkerFrameSource",
    "VideoSourceError",
    "load_marker_frame_source",
    "video_source_blocker",
]

# (model markers, target markers, validity) key triples, in preference order.
_LAYOUTS: tuple[tuple[str, str, str], ...] = (
    ("model_markers_m", "target_markers_m", "marker_validity"),
    ("markers_m", "target_m", "valid"),
)
_MIN_FRAMES = 2


class VideoSourceError(ValueError):
    """The results archive cannot supply marker frames; the message says why."""


@dataclass(frozen=True)
class MarkerFrameSource:
    """Marker trajectories ready for rendering.

    Invariants (checked on construction): ``time_s`` is ``(n,)`` with
    ``n >= 2``; both marker arrays are ``(n, k, 3)``; ``marker_validity`` is
    ``None`` or ``(n, k)`` boolean.
    """

    time_s: NDArray[np.float64]
    model_markers_m: NDArray[np.float64]
    target_markers_m: NDArray[np.float64]
    marker_validity: NDArray[np.bool_] | None = None

    def __post_init__(self) -> None:
        n_frames = len(self.time_s)
        if self.time_s.ndim != 1 or n_frames < _MIN_FRAMES:
            raise VideoSourceError(
                f"results need at least {_MIN_FRAMES} frames, got {n_frames}"
            )
        for name in ("model_markers_m", "target_markers_m"):
            array = getattr(self, name)
            if array.ndim != 3 or array.shape[0] != n_frames or array.shape[2] != 3:
                raise VideoSourceError(
                    f"{name} must have shape ({n_frames}, markers, 3), "
                    f"got {tuple(array.shape)}"
                )
        validity = self.marker_validity
        if validity is not None and validity.shape != self.model_markers_m.shape[:2]:
            raise VideoSourceError(
                f"marker validity must have shape {self.model_markers_m.shape[:2]}, "
                f"got {tuple(validity.shape)}"
            )

    @property
    def frame_count(self) -> int:
        return len(self.time_s)


def _from_npz(npz_path: Path) -> MarkerFrameSource:
    try:
        archive = np.load(npz_path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise VideoSourceError(
            f"{npz_path.name} is not a readable NPZ archive: {exc}"
        ) from exc
    with archive as data:
        keys = set(data.files)
        if "time_s" not in keys:
            raise VideoSourceError(f"{npz_path.name} has no 'time_s' array")
        for model_key, target_key, valid_key in _LAYOUTS:
            if {model_key, target_key} <= keys:
                validity = data[valid_key].astype(bool) if valid_key in keys else None
                return MarkerFrameSource(
                    time_s=np.asarray(data["time_s"], dtype=float),
                    model_markers_m=np.asarray(data[model_key], dtype=float),
                    target_markers_m=np.asarray(data[target_key], dtype=float),
                    marker_validity=validity,
                )
    raise VideoSourceError(
        f"{npz_path.name} has no marker trajectories to play "
        "(need model_markers_m + target_markers_m, or markers_m + target_m); "
        f"it contains: {', '.join(sorted(keys)) or 'nothing'}"
    )


def load_marker_frame_source(source: Any) -> MarkerFrameSource:
    """Return the marker frames of an NPZ path, candidate or frame source.

    Raises:
        FileNotFoundError: ``source`` is a path that does not exist.
        TypeError: ``source`` is none of the supported kinds.
        VideoSourceError: the archive has no playable marker trajectories.
    """
    if isinstance(source, MarkerFrameSource):
        return source
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.is_file():
            raise FileNotFoundError(f"Results file not found: {path}")
        return _from_npz(path)
    if all(
        hasattr(source, a)
        for a in ("time_s", "model_markers_m", "target_markers_m", "marker_validity")
    ):
        if source.model_markers_m is None or source.target_markers_m is None:
            raise VideoSourceError(
                "Candidate lacks marker trajectory arrays required for video export"
            )
        return MarkerFrameSource(
            time_s=np.asarray(source.time_s, dtype=float),
            model_markers_m=np.asarray(source.model_markers_m, dtype=float),
            target_markers_m=np.asarray(source.target_markers_m, dtype=float),
            marker_validity=source.marker_validity,
        )
    raise TypeError(
        f"Expected a results path or candidate, got {type(source).__name__}"
    )


def video_source_blocker(npz_path: Path | None) -> str | None:
    """Why ``npz_path`` cannot be played, or ``None`` when it can."""
    if npz_path is None:
        return "no results archive is recorded for this run"
    try:
        load_marker_frame_source(npz_path)
    except (FileNotFoundError, VideoSourceError) as exc:
        return str(exc)
    return None
