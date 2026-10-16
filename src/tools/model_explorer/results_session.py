"""Qt-free results playback state for the Model Explorer (#12878).

Holds one loaded results archive, the scrub position and the single export
path. The widget delegates here, so playback and export can be tested without
a display and the same code serves every engine (one frame source, one
writer: ``motion_matching.export``).
"""

from __future__ import annotations

from pathlib import Path

from src.shared.python.motion_matching.export import export_video
from src.shared.python.motion_matching.video_frame_source import (
    MarkerFrameSource,
    load_marker_frame_source,
)

__all__ = ["ResultsSession"]

_VIDEO_SUFFIXES = (".mp4", ".gif")


class ResultsSession:
    """A loaded results archive plus its playback cursor.

    Invariant: ``0 <= current_frame < frame_count`` and ``frame_count >= 2``.
    """

    def __init__(
        self, path: Path, source: MarkerFrameSource, engine: str = "engine"
    ) -> None:
        if not engine.strip():
            raise ValueError("engine must be a non-empty name")
        self.path = path
        self.engine = engine
        self._source = source
        self._current_frame = 0

    @classmethod
    def load(cls, path: str | Path, engine: str = "engine") -> ResultsSession:
        """Load ``path``; raise ``FileNotFoundError`` or ``VideoSourceError``."""
        resolved = Path(path)
        return cls(resolved, load_marker_frame_source(resolved), engine)

    @property
    def frame_count(self) -> int:
        return self._source.frame_count

    @property
    def current_frame(self) -> int:
        return self._current_frame

    @property
    def default_video_name(self) -> str:
        """Suggested export file name: the archive stem with ``.mp4``."""
        return f"{self.path.stem}.mp4"

    @property
    def current_time_s(self) -> float:
        return float(self._source.time_s[self._current_frame])

    def scrub(self, index: int) -> None:
        """Move the playback cursor to ``index``.

        Raises:
            ValueError: ``index`` is outside ``[0, frame_count)``.
        """
        if not 0 <= index < self.frame_count:
            raise ValueError(
                f"frame {index} is outside the results (0..{self.frame_count - 1})"
            )
        self._current_frame = index

    def export_video(self, out_path: str | Path, *, stride: int = 5) -> Path:
        """Write the whole playback to ``out_path`` (``.mp4`` or ``.gif``).

        Raises:
            ValueError: the suffix is neither ``.mp4`` nor ``.gif``.
        """
        target = Path(out_path)
        if target.suffix.lower() not in _VIDEO_SUFFIXES:
            raise ValueError(
                f"Unsupported video format '{target.suffix}'; use .mp4 or .gif"
            )
        return export_video(self._source, self.engine, target, stride=stride)
