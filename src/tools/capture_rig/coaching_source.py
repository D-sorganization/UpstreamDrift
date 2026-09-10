"""Media ownership and persistence behind the common coaching interface."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from src.motion_capture.coaching import DrawingLayer
from src.motion_capture.coaching.storage import load_layer, save_layer

from .coaching_export import export_still
from .frame_source import FrameSource
from .player import VideoReader
from .session import load_session
from .swing_export import export_swing
from .swing_export_actions import ExportJob


class CoachingSource(Protocol):
    @property
    def dirty(self) -> bool: ...

    @property
    def reader(self) -> FrameSource: ...

    @property
    def drawings(self) -> DrawingLayer: ...

    def time_at(self, index: int) -> float: ...

    def save(self, drawings: DrawingLayer) -> None: ...

    def still(self, drawings: DrawingLayer, index: int, out: Path) -> None: ...

    def export_job(self, drawings: DrawingLayer) -> ExportJob: ...


class CaptureCoachingSource:
    """Original recording and the established capture drawing/export stores."""

    def __init__(self, root: Path, view: str) -> None:
        source = load_session(root).view(view).recording
        if source is None:
            raise ValueError("This camera has no original recording")
        self.root, self.view = root, view
        self.reader = VideoReader(source)
        try:
            self.drawings = load_layer(
                root,
                view,
                self.reader.width,
                self.reader.height,
                self.reader.frame_count,
            )
        except (OSError, ValueError):
            self.reader.close()
            raise

    def time_at(self, index: int) -> float:
        """Return the original recording clock for a valid frame index."""
        if type(index) is not int or not 0 <= index < self.reader.frame_count:
            raise ValueError("Capture frame index is outside the recording")
        return index / (self.reader.fps or 30)

    @property
    def dirty(self) -> bool:
        return False

    def save(self, drawings: DrawingLayer) -> None:
        """Save original-pixel drawings after verifying recording identity."""
        self._validate(drawings)
        save_layer(self.root, drawings)

    def _validate(self, drawings: DrawingLayer) -> None:
        if drawings.with_shapes(()) != self.drawings.with_shapes(()):
            raise ValueError("Drawings belong to another source")

    def still(self, drawings: DrawingLayer, index: int, out: Path) -> None:
        """Export a PNG using the existing capture still publisher."""
        self._validate(drawings)
        export_still(self.root, drawings, index, out)

    def export_job(self, drawings: DrawingLayer) -> ExportJob:
        """Freeze drawing edits for the existing cancellable capture worker."""
        self._validate(drawings)

        def run(
            out: Path,
            cancelled: Callable[[], bool],
            progress: Callable[[int, int], None],
        ) -> None:
            export_swing(
                self.root,
                self.view,
                out,
                cancelled=cancelled,
                progress=progress,
                drawings=drawings,
            )

        return run
