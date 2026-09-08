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
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPainter, QPixmap
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
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.palette import get_current_colors
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling
from .commands import PlanSelection

SourceFactory = Callable[[RigPlan], Mapping[str, FrameSource]]
FastFactory = Callable[[RigPlan, Mapping[str, str]], Mapping[str, FrameSource]]
DISPLAY_INTERVAL_S = 1 / 15  # UI refresh cap; the camera keeps its own rate
TILE_MIN = (96, 60)  # a soft floor; tiles otherwise follow the pane size
SNAPSHOT_POLL_MS = 100
BADGE_INSET = LayoutMetrics.SPACING_SM + 2  # from the tile's top-left corner


def default_sources(plan: RigPlan) -> Mapping[str, FrameSource]:
    """ffmpeg-decoded previews bound the way the recorder binds cameras."""
    from src.motion_capture.rig.preview_source import ffmpeg_preview_sources

    return ffmpeg_preview_sources(plan)


def sources_from_ids(
    plan: RigPlan, camera_ids: Mapping[str, str]
) -> Mapping[str, FrameSource]:
    """Previews for cameras bound earlier in this session (skips enumeration)."""
    from src.motion_capture.rig.preview_source import preview_sources_from_ids

    return preview_sources_from_ids(plan, camera_ids)


def stamp_badge(pixmap: QPixmap, text: str) -> QPixmap:
    """``text`` in a recording-red pill at the top-left of ``pixmap``.

    No-op when ``text`` is empty or the pixmap null. The pill takes the
    fleet's signal red and a text colour from the palette that reads on it.
    """
    if not text or pixmap.isNull():
        return pixmap
    colors = get_current_colors()
    pill = styling.signal_colors(colors).record
    painter = QPainter(pixmap)
    painter.setFont(get_qfont(max(Sizes.SM, pixmap.height() // 22), Weights.BOLD))
    metrics = painter.fontMetrics()
    pad = LayoutMetrics.SPACING_SM
    width = metrics.horizontalAdvance(text) + 2 * pad
    height = metrics.height() + pad
    radius = LayoutMetrics.RADIUS_SM
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(pill))
    painter.drawRoundedRect(BADGE_INSET, BADGE_INSET, width, height, radius, radius)
    painter.setPen(QColor(styling.contrast_text(pill, colors)))
    painter.drawText(BADGE_INSET + pad, BADGE_INSET + pad // 2 + metrics.ascent(), text)
    painter.end()
    return pixmap


def read_snapshot(path: Path) -> npt.NDArray[np.uint8] | None:
    """Decode a JPEG the recorder is rewriting; ``None`` if absent or torn."""
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return None if image is None else image.astype(np.uint8, copy=False)


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

    frame_ready = pyqtSignal(str, object, int)  # view, ndarray BGR, generation
    failed = pyqtSignal(str, str)  # view, message
    opened = pyqtSignal(str, str)  # view, effective mode text

    def __init__(
        self, view: str, source: FrameSource, mode: CaptureMode, generation: int = 0
    ) -> None:
        super().__init__()
        self.view, self.source, self.mode = view, source, mode
        self.generation = generation
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
                    self.frame_ready.emit(self.view, frame.image, self.generation)
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
        fast_factory: FastFactory = sources_from_ids,
    ) -> None:
        super().__init__(parent)
        self._factory = source_factory
        self._fast_factory = fast_factory
        self._generation = 0  # bumped whenever the tiles are rebuilt
        self._fast_attempt = False
        self._binder: BinderThread | None = None
        self._plan: RigPlan | None = None
        self._workers: dict[str, CameraWorker] = {}
        self._tiles: dict[str, QLabel] = {}
        self._frames: dict[str, int] = {}
        self._last: dict[str, npt.NDArray[np.uint8]] = {}
        self._badge = ""
        self._camera_ids: dict[str, str] = {}
        self._selection: PlanSelection | None = None
        self._snapshot_dir: Path | None = None
        self._snapshot_stamps: dict[str, float] = {}
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(SNAPSHOT_POLL_MS)
        self._snapshot_timer.timeout.connect(self.poll_snapshots)
        self.status = QLabel("preview off")
        self.grid_box = QWidget()
        self.grid = QGridLayout(self.grid_box)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(LayoutMetrics.SPACING_SM)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
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

    @property
    def badge(self) -> str:
        return self._badge

    def set_badge(self, text: str) -> None:
        """Stamp ``text`` (e.g. a REC readout) on every tile; empty removes it."""
        self._badge = text
        self._redraw()

    def camera_ids(self) -> dict[str, str]:
        """``{view: DirectShow instance id}`` from the last successful binding.

        Lets the recorder skip its own 30 s enumeration; empty for sources
        that are not real cameras.
        """
        return dict(self._camera_ids)

    @property
    def watching(self) -> bool:
        """Showing the recorder's snapshots rather than the cameras directly."""
        return self._snapshot_dir is not None

    # -- lifecycle --------------------------------------------------------------
    def start(self, selection: PlanSelection) -> None:
        """Bind the plan's cameras (in a thread) and start streaming each view.

        Precondition: the plan file exists. Binding failures (an unrealizable
        plan) end up on the status line and leave every camera released.
        """
        require(selection.plan.is_file(), "plan file must exist", str(selection.plan))
        self.stop()
        self._selection = selection
        self._plan = RigPlan.load(selection.plan).with_overrides(
            mode=selection.mode,
            views=selection.views or None,
            controls=selection.controls if selection.controls.as_overrides() else None,
        )
        self._build_tiles(tuple(c.view for c in self._plan.cameras))
        ids = self._camera_ids
        self._fast_attempt = bool(ids) and all(v in ids for v in self._tiles)
        if self._fast_attempt:
            factory: SourceFactory = self._fast_binder(dict(ids))
            for view, tile in self._tiles.items():
                tile.setText(f"{view}: opening camera…")
            self.status.setText("opening the cameras bound earlier…")
        else:
            factory = self._factory
            for view, tile in self._tiles.items():
                tile.setText(f"{view}: binding camera…")
            self.status.setText("binding cameras (USB topology + DirectShow listing)…")
        self._binder = BinderThread(factory, self._plan)
        self._binder.bound.connect(self._on_bound)
        self._binder.failed.connect(self._on_bind_failed)
        self._binder.start()
        self.state_changed.emit(True)

    def _fast_binder(self, ids: dict[str, str]) -> SourceFactory:
        """A factory that opens the remembered cameras without enumerating."""

        def factory(plan: RigPlan) -> Mapping[str, FrameSource]:
            return self._fast_factory(plan, ids)

        return factory

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
        self._generation += 1
        for column, view in enumerate(views):
            tile = QLabel(view)
            tile.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tile.setMinimumSize(*TILE_MIN)
            # Ignored: the pixmap never dictates the tile's size, the pane does.
            tile.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            tile.setStyleSheet(styling.tile_style())
            self.grid.addWidget(tile, 0, column)
            self.grid.setColumnStretch(column, 1)
            self._tiles[view] = tile

    # -- slots --------------------------------------------------------------------
    def _on_bound(self, sources: object) -> None:
        self._binder = None
        plan = self._plan
        if plan is None or not isinstance(sources, dict):
            return
        self._camera_ids = {
            view: str(instance)
            for view, source in sources.items()
            if (instance := getattr(source, "camera_instance_id", None))
        }
        for view, source in sources.items():
            if view not in self._tiles:
                continue
            mode = next(c.mode for c in plan.cameras if c.view == view)
            worker = CameraWorker(view, source, mode, self._generation)
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
        if self._fast_attempt and self._selection is not None:
            self._camera_ids = {}  # the remembered cameras are gone; re-enumerate
            self._fast_attempt = False
            self.start(self._selection)
            return
        self.status.setText(f"preview unavailable: {message}")
        for view, tile in self._tiles.items():
            tile.setText(f"{view}: not bound")
        self.state_changed.emit(False)

    def _on_frame(
        self, view: str, image: object, generation: int | None = None
    ) -> None:
        tile = self._tiles.get(view)
        if tile is None or not isinstance(image, np.ndarray):
            return
        if generation is not None and generation != self._generation:
            return  # a worker that was stopped, still flushing its last frame
        self._frames[view] = self._frames.get(view, 0) + 1
        self._last[view] = image
        self._draw(view)
        if self._frames[view] == 1:
            live = ", ".join(v for v, n in self._frames.items() if n)
            prefix = (
                "recording (view from the recorder): " if self.watching else "live: "
            )
            self.status.setText(prefix + live)

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        self._redraw()

    def _redraw(self) -> None:
        for view in self._last:
            self._draw(view)

    def restyle(self) -> None:
        """Tiles and badge follow a theme change."""
        style = styling.tile_style()
        for tile in self._tiles.values():
            tile.setStyleSheet(style)
        self._redraw()

    def _draw(self, view: str) -> None:
        tile, image = self._tiles.get(view), self._last.get(view)
        if tile is None or image is None:
            return
        pixmap = bgr_to_pixmap(image, tile.width(), tile.height())
        tile.setPixmap(stamp_badge(pixmap, self._badge))

    # -- recorder snapshots (the cameras belong to ffmpeg during a take) ------
    def watch_snapshots(self, directory: Path, views: tuple[str, ...]) -> None:
        """Show ``directory/<view>.jpg`` as the recorder rewrites them.

        Precondition: ``directory`` exists. Tiles are rebuilt for ``views``;
        the panel is not *active* (it holds no camera).
        """
        require(directory.is_dir(), "snapshot directory must exist", str(directory))
        self._build_tiles(views)
        for view, tile in self._tiles.items():
            tile.setText(f"{view}: waiting for the recorder…")
        self._snapshot_dir = directory
        self._snapshot_stamps = {}
        self.status.setText("recording: live view from the recorder")
        self._snapshot_timer.start()

    def stop_watching(self) -> None:
        if self._snapshot_dir is None:
            return
        self._snapshot_timer.stop()
        self._snapshot_dir = None
        self.status.setText("preview off (take finished)")

    def poll_snapshots(self) -> None:
        """Read every view's snapshot that changed since the last poll."""
        directory = self._snapshot_dir
        if directory is None:
            return
        for view in self._tiles:
            path = directory / f"{view}.jpg"
            try:
                stamp = path.stat().st_mtime_ns
            except OSError:
                continue
            if self._snapshot_stamps.get(view) == stamp:
                continue
            image = read_snapshot(path)
            if image is None:
                continue  # torn write; the next poll gets a whole file
            self._snapshot_stamps[view] = stamp
            self._on_frame(view, image)

    def _on_opened(self, view: str, mode_text: str) -> None:
        self._tiles[view].setToolTip(f"{view} {mode_text}")

    def _on_failed(self, view: str, message: str) -> None:
        if view in self._tiles:
            self._tiles[view].setText(f"{view}: {message}")
        self.status.setText(f"{view} failed: {message}")
        self._workers.pop(view, None)
        if not self._workers:
            self.state_changed.emit(False)
