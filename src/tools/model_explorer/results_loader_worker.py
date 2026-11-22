"""Load a results archive off the GUI thread (#15150, gap G3).

:class:`ResultsLoader` builds the golfer scene (the slow grip-hand solve) and
then the club-head speed in a ``QThread``: ``loaded(session)`` lets the GUI
thread create the renderer (it owns the GL context) and show the first frame,
``club_ready(series)`` follows; ``failed(message)`` replaces ``loaded`` for an
unplayable archive and ``finished`` always comes last.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.video_frame_source import VideoSourceError
from src.tools.model_explorer.golfer_playback import club_head_speed
from src.tools.model_explorer.results_session import ResultsSession

__all__ = ["LOAD_ERRORS", "ResultsLoadSettings", "ResultsLoader", "prepare_results"]

logger = get_logger(__name__)

#: What a bad or unplayable archive raises. A raise escaping ``run`` would
#: abort the application, so these are reported as ``failed`` instead.
LOAD_ERRORS: tuple[type[BaseException], ...] = (
    *(OSError, ImportError, RuntimeError, ValueError, TypeError, KeyError),
    *(IndexError, AttributeError, ArithmeticError, VideoSourceError),
)


@dataclass(frozen=True)
class ResultsLoadSettings:
    """Blocking-load bound and busy-bar width (``ValueError`` unless > 0)."""

    wait_timeout_s: float = 900.0
    progress_width_px: int = 160

    def __post_init__(self) -> None:
        if not (math.isfinite(self.wait_timeout_s) and self.wait_timeout_s > 0):
            raise ValueError(f"wait_timeout_s must be positive: {self.wait_timeout_s}")
        if self.progress_width_px <= 0:
            raise ValueError(f"progress_width_px must be positive: {self}")


def prepare_results(path: Path, engine: str | None) -> ResultsSession:
    """Read ``path`` and build its golfer scene (Qt- and GL-free, any thread).

    Raises one of :data:`LOAD_ERRORS` when the archive cannot be shown.
    """
    session = ResultsSession.load(path, engine)
    if session.pose_source is not None:
        try:
            session.prepare_scene()
        except BaseException:
            session.close()
            raise
    return session


class ResultsLoader(QThread):
    """One background load of ``path``; create it on the GUI thread."""

    loaded = pyqtSignal(object)
    club_ready = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, path: str | Path, engine: str | None = None) -> None:
        super().__init__()
        if not str(path).strip():
            raise ValueError("path must name a results archive")
        self.path = Path(path)
        self._engine = engine

    def run(self) -> None:
        try:
            session = prepare_results(self.path, self._engine)
        except LOAD_ERRORS as exc:
            logger.warning("Could not load results %s: %s", self.path, exc)
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.loaded.emit(session)
        pose = session.pose_source
        if pose is None:
            return
        try:
            series = club_head_speed(pose)
        except LOAD_ERRORS:
            logger.exception("club-head speed failed for %s", self.path)
            return
        if series is not None:
            self.club_ready.emit(series)
