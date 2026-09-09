"""Live camera preview for the Capture Rig tile.

Binding the plan to devices (USB topology + DirectShow listing, about 30 s)
runs in :class:`BinderThread`; then one :class:`CameraWorker` per view reads
frames from a rig :class:`FrameSource` and hands the latest image to the
panel, which composites the latest frame of every view into **one** canvas
through the chosen :class:`~.layout_model.LayoutSpec` (#9813) — the same
compositor playback and the video export draw with. A view that is captured
but absent from the layout keeps being captured, and the same view may
appear twice (full and a cropped detail, say). :meth:`PreviewPanel.stop`
releases every camera, which the tile calls before a recording starts
(ffmpeg needs the devices) and reverses when it ends.

The source factory is injectable so the panel is tested with the rig's
synthetic sources; the default is the ffmpeg-decoded preview bound the way
the recorder binds cameras (:mod:`src.motion_capture.rig.preview_source`).
"""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPixmap
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from src.motion_capture.rig.plan import CaptureMode, RigPlan
from src.motion_capture.rig.sources import FrameSource
from src.shared.python.core.contracts import require
from src.shared.python.core.process_safety import narrow_catch
from src.shared.python.theme.layout_metrics import LayoutMetrics
from src.shared.python.theme.palette import get_current_colors
from src.shared.python.theme.typography import Sizes, Weights, get_qfont

from . import styling
from .commands import PlanSelection
from .layout_model import LayoutSpec, SourceRef
from .layout_presets import LayoutStore as PresetStore
from .multiview import (
    CanvasLabel,
    ChooserOptions,
    LayoutChooser,
    compose_pixmap,
    live_sources,
    theme_palette,
)

SourceFactory = Callable[[RigPlan], Mapping[str, FrameSource]]
FastFactory = Callable[[RigPlan, Mapping[str, str]], Mapping[str, FrameSource]]
DISPLAY_INTERVAL_S = 1 / 15  # UI refresh cap; the camera keeps its own rate
STALL_TIMEOUT_S = 20.0  # a live view silent this long has stopped delivering;
# it must clear the slowest honest gap: ffmpeg starting and yielding frame one
STALL_POLL_MS = 1000
WORKER_JOIN_MS = 3000
SNAPSHOT_POLL_MS = 100
DEFAULT_LIVE_LAYOUT = "side_by_side"
NOTE_SEPARATOR = "\n"
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
        except ImportError as exc:
            self.failed.emit(
                f"Camera discovery dependency unavailable: {exc}. "
                "Install the project's pose dependencies to enable live preview. "
                "Recorded captures remain available in the library."
            )
        except subprocess.TimeoutExpired as exc:
            self.failed.emit(f"Camera discovery timed out: {exc}. Retry preview.")
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
        self._closed = False
        self.frames = 0
        self.last_frame_ns = time.monotonic_ns()

    def stop(self) -> None:
        """End the loop and release the camera.

        Closing the source is what frees a reader parked in a blocking
        read on a camera that has stopped sending, so this is safe (and
        necessary) to call from the GUI thread: the flag alone would
        never be seen again.
        """
        self._stop = True
        self.release()

    def release(self) -> None:
        """Close the source once; further calls do nothing."""
        if self._closed:
            return
        self._closed = True
        with narrow_catch(
            OSError, ValueError, RuntimeError, log_message="close source"
        ):
            self.source.close()

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
        self.last_frame_ns = time.monotonic_ns()
        try:
            while not self._stop:
                try:
                    frame = self.source.read()
                except (OSError, ValueError, RuntimeError):
                    break  # the source was closed under us: that is the signal
                if frame is None:
                    if self._stop:
                        break
                    time.sleep(0.005)
                    continue
                self.frames += 1
                self.last_frame_ns = time.monotonic_ns()
                now = time.monotonic()
                if now - last >= DISPLAY_INTERVAL_S:
                    last = now
                    self.frame_ready.emit(self.view, frame.image, self.generation)
        finally:
            self.release()


class PreviewPanel(QWidget):
    """One composited canvas of the live views, drawn through a layout.

    The camera workers, the recorder-snapshot mode and the REC badge are
    unchanged from the tiled preview; only the *rendering* differs: the
    latest frame of every view is composited by
    :func:`~.layout_model.compose` into a single canvas that follows the pane
    size. ``layout_store`` is injectable so tests never touch the operator's
    saved layouts.
    """

    state_changed = pyqtSignal(bool)  # active

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        source_factory: SourceFactory = default_sources,
        fast_factory: FastFactory = sources_from_ids,
        layout_store: PresetStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._factory = source_factory
        self._fast_factory = fast_factory
        self._generation = 0  # bumped whenever the view set is rebuilt
        self._fast_attempt = False
        self._binder: BinderThread | None = None
        self._plan: RigPlan | None = None
        self._workers: dict[str, CameraWorker] = {}
        self._views: tuple[str, ...] = ()
        self._notes: dict[str, str] = {}
        self._frames: dict[str, int] = {}
        self._last: dict[str, npt.NDArray[np.uint8]] = {}
        self._canvas: npt.NDArray[np.uint8] | None = None
        self._badge = ""
        self._camera_ids: dict[str, str] = {}
        self._selection: PlanSelection | None = None
        self._snapshot_dir: Path | None = None
        self._snapshot_stamps: dict[str, float] = {}
        self._stall_timeout_s = STALL_TIMEOUT_S
        self._stall_timer = QTimer(self)
        self._stall_timer.setInterval(STALL_POLL_MS)
        self._stall_timer.timeout.connect(self._check_stalls)
        self._snapshot_timer = QTimer(self)
        self._snapshot_timer.setInterval(SNAPSHOT_POLL_MS)
        self._snapshot_timer.timeout.connect(self.poll_snapshots)
        self.status = QLabel("preview off")
        self.canvas = CanvasLabel("preview off")
        self.chooser = LayoutChooser(
            options=ChooserOptions(
                store=layout_store,
                default=DEFAULT_LIVE_LAYOUT,
                editor_title="Edit live layout",
            )
        )
        self.chooser.set_frame_provider(self.frame_for)
        self.chooser.layout_changed.connect(lambda _spec: self._redraw())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        layout.addWidget(self.chooser)
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.status)

    # -- state ------------------------------------------------------------------
    @property
    def active(self) -> bool:
        """Binding or streaming (the cameras are, or are about to be, claimed)."""
        return bool(self._workers) or self._binder is not None

    def status_text(self) -> str:
        """Current camera state for the persistent capture status strip."""
        return self.status.text()

    def views(self) -> tuple[str, ...]:
        """Every view being captured, whether or not the layout shows it."""
        return self._views

    def layout_name(self) -> str:
        """The chosen layout's name (what the tile persists across restarts)."""
        return self.chooser.layout_name()

    def set_layout_name(self, name: str) -> bool:
        """Choose the layout called ``name``; ``False`` when there is none."""
        return self.chooser.set_layout_name(name)

    def layout_spec(self) -> LayoutSpec:
        """The layout the canvas is composited through."""
        return self.chooser.spec()

    def frame_for(self, source: SourceRef) -> npt.NDArray[np.uint8] | None:
        """The latest frame of ``source``'s view, for a thumbnail or a tile."""
        return self._last.get(source.view)

    def canvas_frame(self) -> npt.NDArray[np.uint8] | None:
        """The last composited canvas (BGR), or ``None`` before the first frame."""
        return self._canvas

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
        self._set_views(tuple(c.view for c in self._plan.cameras))
        ids = self._camera_ids
        self._fast_attempt = bool(ids) and all(v in ids for v in self._views)
        if self._fast_attempt:
            factory: SourceFactory = self._fast_binder(dict(ids))
            self._note_all("opening camera…")
            self.status.setText("opening the cameras bound earlier…")
        else:
            factory = self._factory
            self._note_all("binding camera…")
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
            worker.stop()  # closes the source, so a parked read returns
        lingering = [w for w in workers if not w.wait(WORKER_JOIN_MS)]
        self._workers = {}
        self._stall_timer.stop()
        self._note_all("preview off")
        if workers or binder is not None:
            self.status.setText(
                "preview off (cameras released)"
                if not lingering
                else f"preview off (cameras released; {len(lingering)} "
                "reader thread(s) still finishing)"
            )
            self.state_changed.emit(False)

    def _set_views(self, views: tuple[str, ...]) -> None:
        """Capture ``views`` from now on and offer them to the layout picker.

        Postcondition: the layout's sources are these views, so a built-in
        preset refills itself; a saved layout keeps whatever it names, and a
        view it does not show is captured all the same.
        """
        self._views = tuple(views)
        self._frames = {}
        self._last = {}
        self._canvas = None
        self._generation += 1
        self.chooser.set_sources(live_sources(self._views))
        self._note_all("")
        self._redraw()

    def _note_all(self, text: str) -> None:
        """Say the same thing about every view on the canvas placeholder."""
        self._notes = dict.fromkeys(self._views, text)
        self._show_notes()

    def _show_notes(self) -> None:
        """Put the per-view notes on the canvas while no frame has arrived."""
        if self._last:
            return
        lines = [f"{v}: {n}" for v, n in self._notes.items() if n]
        self.canvas.setPixmap(QPixmap())
        self.canvas.setText(NOTE_SEPARATOR.join(lines) or "preview off")

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
            if view not in self._views:
                continue
            mode = next(c.mode for c in plan.cameras if c.view == view)
            worker = CameraWorker(view, source, mode, self._generation)
            worker.frame_ready.connect(self._on_frame)
            worker.failed.connect(self._on_failed)
            worker.opened.connect(self._on_opened)
            self._workers[view] = worker
            self._notes[view] = "opening…"
            worker.start()
            self._stall_timer.start()
        self._show_notes()
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
        self._note_all("not bound")
        self.state_changed.emit(False)

    def _on_frame(
        self, view: str, image: object, generation: int | None = None
    ) -> None:
        if view not in self._views or not isinstance(image, np.ndarray):
            return
        if generation is not None and generation != self._generation:
            return  # a worker that was stopped, still flushing its last frame
        self._frames[view] = self._frames.get(view, 0) + 1
        self._last[view] = image
        self._draw()
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
        self._draw()

    def restyle(self) -> None:
        """Canvas, compositor palette and badge follow a theme change."""
        self.canvas.restyle()
        self._redraw()

    def _draw(self) -> None:
        """Composite the latest frame of every view into the one canvas.

        Postcondition: with no frame yet the canvas shows the per-view notes
        instead; otherwise it holds the composited image with the badge (when
        set) stamped on it, and :meth:`canvas_frame` returns those pixels.
        """
        if not self._last:
            self._show_notes()
            return
        frames = {
            SourceRef(kind="live", view=view).key: image
            for view, image in self._last.items()
        }
        canvas, pixmap = compose_pixmap(
            frames, self.chooser.spec(), self.canvas.canvas_size(), theme_palette()
        )
        self._canvas = canvas
        self.canvas.setPixmap(stamp_badge(pixmap, self._badge))

    # -- recorder snapshots (the cameras belong to ffmpeg during a take) ------
    def watch_snapshots(self, directory: Path, views: tuple[str, ...]) -> None:
        """Show ``directory/<view>.jpg`` as the recorder rewrites them.

        Precondition: ``directory`` exists. Tiles are rebuilt for ``views``;
        the panel is not *active* (it holds no camera).
        """
        require(directory.is_dir(), "snapshot directory must exist", str(directory))
        self._set_views(views)
        self._note_all("waiting for the recorder…")
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
        for view in self._views:
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
        self._notes[view] = mode_text
        self.canvas.setToolTip(
            "; ".join(f"{v} {n}" for v, n in self._notes.items() if n)
        )

    def set_stall_timeout(self, seconds: float) -> None:
        """How long a view may deliver nothing before its camera is released."""
        require(seconds > 0, "stall timeout must be positive", seconds)
        self._stall_timeout_s = seconds

    def _check_stalls(self) -> None:
        """Release any view that has stopped delivering frames.

        A camera process can stay alive while sending nothing (seen on the lab
        rig: three preview processes alive at zero CPU for over an hour). It
        keeps the device claimed, so the recorder cannot open it and every take
        comes back empty. Letting the view go frees the device and says so,
        rather than leaving a frozen picture on screen.
        """
        if not self._workers:
            self._stall_timer.stop()
            return
        limit_ns = self._stall_timeout_s * 1e9
        now = time.monotonic_ns()
        stalled = [
            view
            for view, worker in self._workers.items()
            if now - worker.last_frame_ns > limit_ns
        ]
        for view in stalled:
            worker = self._workers.pop(view)
            worker.stop()
            worker.wait(WORKER_JOIN_MS)
            self._notes[view] = "stalled: camera released"
        if not stalled:
            return
        self._show_notes()
        self.status.setText(
            "stalled, camera released: "
            + ", ".join(stalled)
            + " - press Preview cameras to reopen"
        )
        if not self._workers:
            self._stall_timer.stop()
            self.state_changed.emit(False)

    def _on_failed(self, view: str, message: str) -> None:
        if view in self._views:
            self._notes[view] = message
            self._show_notes()
        self.status.setText(f"{view} failed: {message}")
        self._workers.pop(view, None)
        if not self._workers:
            self.state_changed.emit(False)
