"""Synchronised multi-view playback of a session, drawn through a layout.

:class:`PlaybackPanel` plays several sources of one session in a single
composited canvas (#9814): the raw recording (or proxy) of a view, and the
*overlay* render of a view — the detector's pose (:mod:`.overlay`) plus the
selected variants' projected models (:mod:`.overlay_render`). Which tile
shows what is a :class:`~.layout_model.LayoutSpec`, the same model the live
preview and the composite export use, so a tile can show "raw" and the tile
beside it "overlay" of that very view.

Frame *k* means the same instant in every tile: when the session manifest
carries a strobe-alignment block each view's reader is offset by
:func:`~.mosaic.offsets_from_timing`. Readers are opened once per view and kept
open, so switching layout re-composites without re-opening any file.

Scrubbing, play/pause, speed, frame stepping and the observation-set choice
work as they always did; the current canvas can be written out as a PNG.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from src.shared.python.core.contracts import require
from src.shared.python.theme.layout_metrics import LayoutMetrics

from .layout_model import LayoutSpec, SourceRef
from .layout_presets import LayoutStore
from .mosaic import offsets_from_timing
from .multiview import (
    CanvasLabel,
    ChooserOptions,
    LayoutChooser,
    compose_pixmap,
    theme_palette,
    write_png,
)
from .overlay import PoseTrack, draw_pose
from .overlay_box import VariantOverlayBox
from .overlay_render import render_frame
from .player import VideoReader, clamp_index
from .session import SessionMedia, ViewMedia

PLAYBACK_MIN_SIZE = (320, 200)
DEFAULT_PLAYBACK_LAYOUT = "single"
SPEED_RANGE = (0.1, 4.0)
DEFAULT_FPS = 30.0
HELP: dict[str, str] = {
    "view": "Which view drives the transport: its frame count sets the "
    "slider range and its rate the playback speed.",
    "set": "Which observation set the pose overlay is drawn from.",
    "overlay": "Draw the detector's 2-D pose on every overlay tile.",
    "confidence": "Hide pose points whose detector confidence is below this.",
    "play": "Play or pause. Frame k is the same instant in every tile.",
    "speed": "Playback rate as a multiple of the recording's own frame rate.",
    "step_back": "Show the previous frame.",
    "step_forward": "Show the next frame.",
    "export": "Write the canvas exactly as it is on screen to a PNG file.",
}


def _hint(text: str, enabled: bool, why: str) -> str:
    """``text``, plus a ``Disabled:`` line when the control is greyed out."""
    return text if enabled else f"{text}\n\nDisabled: {why}."


class PlaybackPanel(QWidget):
    """Several of a session's views in one composited, frame-synced canvas.

    ``layout_store`` is injectable so tests never read the operator's saved
    layouts.
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        layout_store: LayoutStore | None = None,
    ) -> None:
        super().__init__(parent)
        self._media: SessionMedia | None = None
        self._readers: dict[str, VideoReader] = {}
        self._tracks: dict[str, PoseTrack | None] = {}
        self._offsets: dict[str, int] = {}
        self._canvas: npt.NDArray[np.uint8] | None = None
        self._index = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self.step)
        self.view_combo = QComboBox()
        self.view_combo.setToolTip(HELP["view"])
        self.view_combo.currentIndexChanged.connect(self._on_view_changed)
        self.set_combo = QComboBox()
        self.set_combo.setToolTip(HELP["set"])
        self.set_combo.currentIndexChanged.connect(self._on_set_changed)
        self.overlay_check = QCheckBox("pose overlay")
        self.overlay_check.setToolTip(HELP["overlay"])
        self.overlay_check.setChecked(True)
        self.overlay_check.toggled.connect(lambda _: self._invalidate())
        self.confidence_spin = QDoubleSpinBox()
        self.confidence_spin.setToolTip(HELP["confidence"])
        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.5)
        self.confidence_spin.valueChanged.connect(lambda _: self._invalidate())
        self.play_button = QPushButton("Play")
        self.play_button.setToolTip(HELP["play"])
        self.play_button.clicked.connect(self.toggle_play)
        self.variants = VariantOverlayBox()
        self.variants.changed.connect(self._invalidate)
        self.chooser = LayoutChooser(
            options=ChooserOptions(
                store=layout_store,
                default=DEFAULT_PLAYBACK_LAYOUT,
                editor_title="Edit playback layout",
            )
        )
        self.chooser.set_frame_provider(self._thumbnail)
        self.chooser.layout_changed.connect(lambda _spec: self._invalidate())
        self.image = CanvasLabel("no session loaded")
        self.image.setMinimumSize(*PLAYBACK_MIN_SIZE)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.valueChanged.connect(self.show_frame)
        self.status = QLabel("")
        self._build(self._transport())
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.setInterval(50)
        self._resize_timer.timeout.connect(self._invalidate)
        self.image.resized.connect(self._resize_timer.start)

    # -- construction ------------------------------------------------------
    def _transport(self) -> QHBoxLayout:
        """Play / speed / frame-step / export, under the canvas."""
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setToolTip(HELP["speed"])
        self.speed_spin.setRange(*SPEED_RANGE)
        self.speed_spin.setSingleStep(0.25)
        self.speed_spin.setValue(1.0)
        self.speed_spin.valueChanged.connect(lambda _: self._retime())
        self.step_back_button = QPushButton("◀|")
        self.step_back_button.setToolTip(_hint(HELP["step_back"], True, ""))
        self.step_back_button.clicked.connect(lambda: self.step_by(-1))
        self.step_forward_button = QPushButton("|▶")
        self.step_forward_button.setToolTip(_hint(HELP["step_forward"], True, ""))
        self.step_forward_button.clicked.connect(lambda: self.step_by(1))
        self.export_button = QPushButton("Export PNG…")
        self.export_button.setToolTip(HELP["export"])
        self.export_button.clicked.connect(self._export_clicked)
        row = QHBoxLayout()
        row.setSpacing(LayoutMetrics.SPACING_SM)
        row.addWidget(self.play_button)
        row.addWidget(self.step_back_button)
        row.addWidget(self.step_forward_button)
        row.addWidget(QLabel("speed"))
        row.addWidget(self.speed_spin)
        row.addStretch(1)
        row.addWidget(self.export_button)
        return row

    def _build(self, transport: QHBoxLayout) -> None:
        top = QHBoxLayout()
        top.setSpacing(LayoutMetrics.SPACING_SM)
        top.addWidget(QLabel("View"))
        top.addWidget(self.view_combo, 1)
        top.addWidget(QLabel("Set"))
        top.addWidget(self.set_combo, 1)
        top.addWidget(self.overlay_check)
        top.addWidget(QLabel("min conf"))
        top.addWidget(self.confidence_spin)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(LayoutMetrics.SPACING_SM)
        layout.addLayout(top)
        layout.addWidget(self.chooser)
        layout.addWidget(self.variants)
        layout.addWidget(self.image, 1)
        layout.addWidget(self.slider)
        layout.addLayout(transport)
        layout.addWidget(self.status)

    # -- state -------------------------------------------------------------
    @property
    def frame_index(self) -> int:
        return self._index

    @property
    def playing(self) -> bool:
        return self._timer.isActive()

    @property
    def _reader(self) -> VideoReader | None:
        """The transport's reader: the view chosen in the combo."""
        name = self.current_view_name()
        return None if name is None else self._readers.get(name)

    def layout_name(self) -> str:
        """The chosen layout's name (what the tile persists across restarts)."""
        return self.chooser.layout_name()

    def set_layout_name(self, name: str) -> bool:
        return self.chooser.set_layout_name(name)

    def layout_spec(self) -> LayoutSpec:
        return self.chooser.spec()

    def canvas_frame(self) -> npt.NDArray[np.uint8] | None:
        """The last composited canvas (BGR), or ``None`` before the first draw."""
        return self._canvas

    def offsets(self) -> dict[str, int]:
        """Whole-frame alignment offsets per view (all zero without timing)."""
        return dict(self._offsets)

    def current_view_name(self) -> str | None:
        view = self.view_combo.currentData()
        return None if view is None else str(view.view)

    def current_set_name(self) -> str | None:
        text = self.set_combo.currentText()
        return text or None

    # -- loading -----------------------------------------------------------
    def load(self, media: SessionMedia) -> None:
        """Offer every playable view as a raw and an overlay source.

        Postcondition: no file is open until a tile asks for a frame, and the
        first playable view drives the transport.
        """
        self.close_media()
        self.setEnabled(True)
        self._media = media
        self.variants.load(media)
        self.view_combo.blockSignals(True)
        self.view_combo.clear()
        for view in media.views:
            if view.playable is not None:
                self.view_combo.addItem(view.view, view)
        self.view_combo.blockSignals(False)
        self._offsets = offsets_from_timing(media.timing, self._rates(media))
        self.chooser.set_sources(self._sources(media))
        if self.view_combo.count():
            self._on_view_changed(0)
        else:
            self.image.setText("no playable recording in this session")

    @staticmethod
    def _rates(media: SessionMedia) -> dict[str, float]:
        return {
            v.view: float(v.fps or DEFAULT_FPS)
            for v in media.views
            if v.playable is not None
        }

    @staticmethod
    def _sources(media: SessionMedia) -> tuple[SourceRef, ...]:
        """Every playable view as an overlay source, then as a raw one.

        Overlays come first so the one-tile default keeps drawing what the
        single-view player always drew.
        """
        views = [v.view for v in media.views if v.playable is not None]
        overlays = [SourceRef(kind="overlay", view=v) for v in views]
        return (*overlays, *(SourceRef(kind="recorded", view=v) for v in views))

    def _view_media(self, view: str) -> ViewMedia | None:
        media = self._media
        if media is None:
            return None
        return next((v for v in media.views if v.view == view), None)

    def _reader_for(self, view: str) -> VideoReader | None:
        """The open reader for ``view``, opening it once and keeping it open."""
        if view in self._readers:
            return self._readers[view]
        entry = self._view_media(view)
        playable = None if entry is None else entry.playable
        if playable is None:
            return None
        self._readers[view] = VideoReader(playable)
        return self._readers[view]

    def _track_for(self, view: str) -> PoseTrack | None:
        """The pose track of ``view`` in the chosen observation set (cached)."""
        if view in self._tracks:
            return self._tracks[view]
        entry = self._view_media(view)
        path: Path | None = None
        if entry is not None:
            sets = entry.observation_sets or {}
            path = sets.get(self.current_set_name() or "") or entry.observations
        self._tracks[view] = PoseTrack.load(path) if path is not None else None
        return self._tracks[view]

    def close_media(self) -> None:
        """Stop and release every open reader."""
        self._timer.stop()
        self._resize_timer.stop()
        self.play_button.setText("Play")
        for reader in self._readers.values():
            reader.close()
        self._readers = {}
        self._tracks = {}
        self._canvas = None

    def clear_capture(self) -> None:
        """Release prior capture state so invalid selection cannot replay old data."""
        self.close_media()
        self._media = None
        self.variants.load(None)
        self._offsets = {}
        self._index = 0
        for combo in (self.view_combo, self.set_combo):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
        self.chooser.set_sources(())
        self.slider.setRange(0, 0)
        self.image.clear()
        self.image.setText("No capture loaded. Choose a valid capture from Library.")
        self.status.setText("No capture loaded")
        self.setEnabled(False)

    # -- transport ---------------------------------------------------------
    def _on_view_changed(self, index: int) -> None:
        view: ViewMedia | None = self.view_combo.itemData(index)
        if view is None or view.playable is None:
            return
        reader = self._reader_for(view.view)
        if reader is None:
            return
        self.set_combo.blockSignals(True)
        self.set_combo.clear()
        for name in view.observation_sets or {}:
            self.set_combo.addItem(name, name)
        if not self.set_combo.count() and view.observations is not None:
            self.set_combo.addItem("observations", "observations")
        self.set_combo.blockSignals(False)
        self._tracks = {}
        self.slider.setRange(0, max(reader.frame_count - 1, 0))
        self._retime()
        self.show_frame(0)

    def _on_set_changed(self, _index: int) -> None:
        self._tracks = {}
        self.show_frame(self._index)

    def _invalidate(self) -> None:
        """A drawing option changed: re-composite the frame on screen."""
        self.show_frame(self._index)

    def _retime(self) -> None:
        """Timer interval from the driving view's rate and the speed factor."""
        reader = self._reader
        entry = self._view_media(self.current_view_name() or "")
        base = (reader.fps if reader else 0.0) or (entry.fps if entry else None)
        rate = float(base or DEFAULT_FPS) * float(self.speed_spin.value())
        self._timer.setInterval(max(int(1000.0 / max(rate, 0.01)), 1))

    def toggle_play(self) -> None:
        if self._reader is None:
            return
        if self.playing:
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            self._timer.start()
            self.play_button.setText("Pause")

    def step(self) -> None:
        reader = self._reader
        if reader is None:
            return
        nxt = self._index + 1
        if nxt >= reader.frame_count:
            self.toggle_play()
            return
        self.slider.setValue(nxt)

    def step_by(self, delta: int) -> None:
        """Move ``delta`` frames (clamped); pauses first so the step sticks."""
        if self.playing:
            self.toggle_play()
        self.slider.setValue(self._index + delta)

    # -- rendering ---------------------------------------------------------
    def _thumbnail(self, source: SourceRef) -> npt.NDArray[np.uint8] | None:
        """The editor's live thumbnail of ``source`` at the current frame."""
        return self._source_frame(source, self._index)

    def _source_frame(
        self, source: SourceRef, index: int
    ) -> npt.NDArray[np.uint8] | None:
        """Frame ``index`` of ``source``, aligned and drawn on if it is an overlay.

        The alignment offset makes ``index`` the same instant in every view.
        """
        if source.is_empty:
            return None
        reader = self._reader_for(source.view)
        if reader is None:
            return None
        shifted = clamp_index(
            index + self._offsets.get(source.view, 0), reader.frame_count
        )
        frame = reader.read(shifted)
        if frame is None or source.kind != "overlay":
            return frame
        return self._overlaid(source, frame, index)

    def _overlaid(
        self, source: SourceRef, frame: npt.NDArray[np.uint8], index: int
    ) -> npt.NDArray[np.uint8]:
        """The pose and the tile's variants drawn on ``frame``."""
        track = self._track_for(source.view)
        pose = track.at(index) if track else None
        if pose is not None and self.overlay_check.isChecked():
            frame = draw_pose(
                frame,
                pose[0],
                pose[1],
                track.edges if track else (),
                min_confidence=float(self.confidence_spin.value()),
            )
        tracks = self.variants.tracks_for(source.view, source.variants)
        return render_frame(frame, tracks, index) if tracks else frame

    def show_frame(self, index: int) -> None:
        """Composite frame ``index`` (clamped) of every tile onto the canvas.

        Postcondition: :meth:`canvas_frame` holds exactly the pixels shown,
        so a PNG export and the screen agree.
        """
        reader = self._reader
        if reader is None:
            return
        self._index = clamp_index(index, reader.frame_count)
        spec = self.chooser.spec()
        frames: dict[str, npt.NDArray[np.uint8]] = {}
        for tile in spec.tiles:
            source = tile.source
            if source.is_empty or source.key in frames:
                continue
            frame = self._source_frame(source, self._index)
            if frame is not None:
                frames[source.key] = frame
        canvas, pixmap = compose_pixmap(
            frames, spec, (self.image.width(), self.image.height()), theme_palette()
        )
        self._canvas = canvas
        self.image.setPixmap(pixmap)
        self.slider.blockSignals(True)
        self.slider.setValue(self._index)
        self.slider.blockSignals(False)
        self._report(reader.frame_count)

    def _report(self, total: int) -> None:
        view = self.current_view_name()
        track = self._track_for(view) if view else None
        detected = "pose" if track and track.at(self._index) is not None else "no pose"
        extra = f" · {self.variants.error}" if self.variants.error else ""
        self.status.setText(f"frame {self._index + 1}/{total} · {detected}{extra}")

    def resizeEvent(self, event: Any) -> None:  # noqa: N802 - Qt override
        super().resizeEvent(event)
        if self._reader is not None:
            self.show_frame(self._index)

    # -- export ------------------------------------------------------------
    def export_png(self, path: Path) -> Path:
        """Write the canvas as it is on screen to ``path``.

        Precondition: a frame has been composited. Postcondition: ``path``
        exists and holds the same pixels as :meth:`canvas_frame`.
        """
        canvas = self._canvas
        require(canvas is not None, "nothing has been played back yet")
        assert canvas is not None
        return write_png(canvas, Path(path))

    def _export_clicked(self) -> None:
        if self._canvas is None:
            self.status.setText("nothing to export yet")
            return
        name, _ = QFileDialog.getSaveFileName(
            self, "Export canvas", "canvas.png", "PNG image (*.png)"
        )
        if name:
            self.status.setText(f"wrote {self.export_png(Path(name)).name}")
