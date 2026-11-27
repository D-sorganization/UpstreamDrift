"""Build the golfer scene in a ``QThread`` (#15150, G3): ``loaded`` or
``failed``, then always ``finished``; the GUI thread owns the GL renderer."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PyQt6.QtCore import QThread, pyqtSignal

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.video_frame_source import VideoSourceError
from src.tools.model_explorer.results_session import ResultsSession
from src.tools.model_explorer.scene_cache import (
    SCENE_BUILDING_MESSAGE,
    SCENE_CACHED_MESSAGE,
)

__all__ = [
    "LOAD_ERRORS",
    "ResultsLoadSettings",
    "ResultsLoader",
    "live_loader_count",
    "prepare_results",
]

logger = get_logger(__name__)

#: Unplayable-archive errors, reported as ``failed`` (escaping ``run`` aborts).
LOAD_ERRORS: tuple[type[BaseException], ...] = (
    *(OSError, ImportError, RuntimeError, ValueError, TypeError, KeyError),
    *(IndexError, AttributeError, ArithmeticError, VideoSourceError),
)


_LIVE: set[ResultsLoader] = set()


def live_loader_count() -> int:
    """Background loads still running (or not yet retired), app-wide."""
    return len(_LIVE)


@dataclass(frozen=True)
class ResultsLoadSettings:
    """Blocking-load bound, busy-bar width and poll interval (each > 0)."""

    wait_timeout_s: float = 900.0
    progress_width_px: int = 160
    #: longest the GUI thread sleeps between checks while waiting on a load
    poll_interval_ms: int = 50

    def __post_init__(self) -> None:
        if not (math.isfinite(self.wait_timeout_s) and self.wait_timeout_s > 0):
            raise ValueError(f"wait_timeout_s must be positive: {self.wait_timeout_s}")
        if self.progress_width_px <= 0:
            raise ValueError(f"progress_width_px must be positive: {self}")
        if self.poll_interval_ms <= 0:
            raise ValueError(f"poll_interval_ms must be positive: {self}")


def prepare_results(
    path: Path, engine: str | None, report: Callable[[str], None] | None = None
) -> ResultsSession:
    """Read ``path`` and build its scene on any thread; raises LOAD_ERRORS.

    ``report`` hears whether the scene is built or read from the scene cache.
    """
    session = ResultsSession.load(path, engine)
    if session.pose_source is not None:
        try:
            if report is not None:
                cached = session.scene_is_cached()
                report(SCENE_CACHED_MESSAGE if cached else SCENE_BUILDING_MESSAGE)
            session.prepare_scene()
        except BaseException:
            session.close()
            raise
    return session


class ResultsLoader(QThread):
    """One background load of ``path``; create it on the GUI thread."""

    loaded = pyqtSignal(object)
    progress = pyqtSignal(str)  # built, or read from the scene cache
    failed = pyqtSignal(str)

    def __init__(self, path: str | Path, engine: str | None = None) -> None:
        super().__init__()
        if not str(path).strip():
            raise ValueError("path must name a results archive")
        self.path = Path(path)
        self._engine = engine
        # Outlive the window: a QThread collected while running aborts.
        _LIVE.add(self)
        self.finished.connect(self._retire)

    def _retire(self) -> None:
        _LIVE.discard(self)
        self.deleteLater()

    def run(self) -> None:
        try:
            session = prepare_results(self.path, self._engine, self.progress.emit)
        except LOAD_ERRORS as exc:
            logger.warning("Could not load results %s: %s", self.path, exc)
            self.failed.emit(str(exc) or type(exc).__name__)
            return
        self.loaded.emit(session)
