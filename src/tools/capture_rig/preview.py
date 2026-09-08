"""Live camera preview for the Capture Rig tile.

Binding the plan to devices (USB topology + DirectShow listing, about 30 s)
runs in :class:`BinderThread`; then one :class:`CameraWorker` per view reads
frames from a rig :class:`FrameSource` and hands the latest image to the
panel, which shows the views side by side. :meth:`PreviewPanel.stop`
releases every camera, which the tile calls before a recording starts
(ffmpeg needs the devices) and reverses when it ends.

The source factory is injectable so the panel is tested with the rig's
synthetic sources; the default is the ffmpeg-decoded preview bound the way
the recorder binds cameras (:mod:`src.motion_capture.rig.preview_source`).
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QGridLayout,
    QLabel,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.rig.plan import CaptureMode, RigPlan
from src.motion_capture.rig.sources import FrameSource
from src.shared.python.core.contracts import require
from src.shared.python.core.process_safety import narrow_catch
from src.shared.python.theme.palette import get_current_colors

from .commands import PlanSelection

SourceFactory = Callable[[RigPlan], Mapping[str, FrameSource]]
DISPLAY_INTERVAL_S = 1 / 15  # UI refresh cap; the camera keeps its own rate
TILE_MIN = (96, 60)  # a soft floor; tiles otherwise follow the pane size


def default_sources(plan: RigPlan) -> Mapping[str, FrameSource]:
    """ffmpeg-decoded previews bound the way the recorder binds cameras."""
    from src.motion_capture.rig.preview_source import ffmpeg_preview_sources

    return ffmpeg_preview_sources(plan)


def tile_style() -> str:
    """Preview tile colours from the active theme (no literal colours here)."""
    colors = get_current_colors()
    return f"background: {colors['bg']}; color: {colors['text_secondary']};"


def bgr_to_pixmap(frame_bgr: npt.NDArray[np.uint8], width: int, height: int) -> QPixmap:
    rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
    h, w = rgb.shape[:2]
    image = QImage(rgb.tobytes(), w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(image).scaled(
        max(width, 1),
        max(height, 1),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.FastTransformation,
    )


class BinderThread(QThread):
    """Runs the slow plan-to-camera binding off the UI thread."""

    bound = pyqtSignal(object)  # Mapping[str, FrameSource]
    failed = pyqtSignal(str)

    def __init__(self, factory: SourceFactory, plan: RigPlan) -> None:
        super().__init__()
        self._factory, self._plan = factory, plan

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            self.bound.emit(dict(self._factory(self._plan)))
        except (ValueError, OSError, RuntimeError) as exc:
            self.failed.emit(str(exc))


class CameraWorker(QThread):
    """Reads one source until told to stop; emits the latest frame at ~15 Hz."""

    frame_ready = pyqtSignal(str, object)  # view, ndarray BGR
    failed = pyqtSignal(str, str)  # view, message
    opened = pyqtSignal(str, str)  # view, effective mode text

    def __init__(self, view: str, source: FrameSource, mode: CaptureMode) -> None:
        super().__init__()
        self.view, self.source, self.mode = view, source, mode
        self._stop = False
        self.frames = 0

    def stop(self) -> None:
        self._stop = True

    def run(self) -> None:  # noqa: D102 - QThread entry point
        try:
            effective = self.source.open(self.mode)
        except (ValueError, OSError, RuntimeError) as exc:
            self.failed.emit(self.view, str(exc))
            return
        self.opened.emit(
            self.view, f"{effective.width}x{effective.height}@{effective.fps}"
        )
        last = 0.0
        try:
            while not self._stop:
                frame = self.source.read()
                if frame is None:
                    time.sleep(0.005)
                    continue
                self.frames += 1
                now = time.monotonic()
                if now - last >= DISPLAY_INTERVAL_S:
                    last = now
                    self.frame_ready.emit(self.view, frame.image)
        finally:
            with narrow_catch(OSError, RuntimeError, log_message="close source"):
                self.source.close()


class PreviewPanel(QWidget):
    """Tiles of live camera frames for the planned views."""

    state_changed = pyqtSignal(bool)  # active

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        source_factory: SourceFactory = default_sources,
    ) -> None:
        super().__init__(parent)
        self._factory = source_factory
        self._binder: BinderThread | None = None
        self._plan: RigPlan | None = None
        self._workers: dict[str, CameraWorker] = {}
        self._tiles: dict[str, QLabel] = {}
        self._frames: dict[str, int] = {}
        self._last: dict[str, npt.NDArray[np.uint8]] = {}
        self.status = QLabel("preview off")
        self.grid_box = QWidget()
        self.grid = QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.grid_box, 1)
        layout.addWidget(self.status)

    # -- state ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """Binding or streaming (the cameras are, or are about to be, claimed)."""
        return bool(self._workers) or self._binder is not None

    def views(self) -> tuple[str, ...]:
        return tuple(self._tiles)

    def frames_seen(self, view: str) -> int:
        return self._frames.get(view, 0)

    # -- lifecycle --------------------------------------------------------------
    def start(self, selection: PlanSelection) -> None:
        """Bind the plan's cameras (in a thread) and start streaming each view.

        Precondition: the plan file exists. Binding failures (an unrealizable
        plan) end up on the status line and leave every camera released.
        """
        require(selection.plan.is_file(), "plan file must exist", str(selection.plan))
        self.stop()
        self._plan = RigPlan.load(selection.plan).with_overrides(
            mode=selection.mode,
            views=selection.views or None,
            controls=selection.controls if selection.controls.as_overrides() else None,
        )
        self._build_tiles(tuple(c.view for c in self._plan.cameras))
        for view, tile in self._tiles.items():
            tile.setText(f"{view}: binding camera…")
        self.status.setText("binding cameras (USB topology + DirectShow listing)…")
        self._binder = BinderThread(self._factory, self._plan)
        self._binder.bound.connect(self._on_bound)
        self._binder.failed.connect(self._on_bind_failed)
        self._binder.start()
        self.state_changed.emit(True)

    def stop(self) -> None:
        """Release every camera (blocks until the threads have closed them)."""
        binder, self._binder = self._binder, None
        if binder is not None:
            binder.wait(60000)  # binding cannot be interrupted; it is bounded
        workers = list(self._workers.values())
        for worker in workers:
            worker.stop()
        for worker in workers:
            worker.wait(5000)
        self._workers = {}
        for view, tile in self._tiles.items():
            tile.setText(f"{view}: preview off")
        if workers or binder is not None:
            self.status.setText("preview off (cameras released)")
            self.state_changed.emit(False)

    def _build_tiles(self, views: tuple[str, ...]) -> None:
        for tile in self._tiles.values():
            self.grid.removeWidget(tile)
            tile.deleteLater()
        self._tiles = {}
        self._frames = {}
        self._last = {}
        for column, view in enumerate(views):
            tile = QLabel(view)
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setMinimumSize(*TILE_MIN)
            # Ignored: the pixmap never dictates the tile's size, the pane does.
            tile.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            tile.setStyleSheet(tile_style())
            self.grid.addWidget(tile, 0, column)
            self.grid.setColumnStretch(column, 1)
            self._tiles[view] = tile

    # -- slots --------------------------------------------------------------------
    def _on_bound(self, sources: object) -> None:
        self._binder = None
        plan = self._plan
        if plan is None or not isinstance(sources, dict):
            return
        for view, source in sources.items():
            if view not in self._tiles:
                continue
            mode = next(c.mode for c in plan.cameras if c.view == view)
            worker = CameraWorker(view, source, mode)
            worker.frame_ready.connect(self._on_frame)
            worker.failed.connect(self._on_failed)
            worker.opened.connect(self._on_opened)
            self._workers[view] = worker
            self._tiles[view].setText(f"{view}: opening…")
            worker.start()
        self.status.setText("preview starting…")
        if not self._workers:
            self.state_changed.emit(False)

    def _on_bind_failed(self, message: str) -> None:
        self._binder = None
        self.status.setText(f"preview unavailable: {message}")
        for view, tile in self._tiles.items():
            tile.setText(f"{view}: not bound")
        self.state_changed.emit(False)

    def _on_frame(self, view: str, image: object) -> None:
        tile = self._tiles.get(view)
        if tile is None or not isinstance(image, np.ndarray):
            return
        self._frames[view] = self._frames.get(view, 0) + 1
        self._last[view] = image
        tile.setPixmap(bgr_to_pixmap(image, tile.width(), tile.height()))
        if self._frames[view] == 1:
            self.status.setText(
                "live: " + ", ".join(v for v, n in self._frames.items() if n)
            )

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        for view, image in self._last.items():
            tile = self._tiles.get(view)
            if tile is not None:
                tile.setPixmap(bgr_to_pixmap(image, tile.width(), tile.height()))

    def _on_opened(self, view: str, mode_text: str) -> None:
        self._tiles[view].setToolTip(f"{view} {mode_text}")

    def _on_failed(self, view: str, message: str) -> None:
        if view in self._tiles:
            self._tiles[view].setText(f"{view}: {message}")
        self.status.setText(f"{view} failed: {message}")
        self._workers.pop(view, None)
        if not self._workers:
            self.state_changed.emit(False)
