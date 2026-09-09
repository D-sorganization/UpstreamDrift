"""Reference comparison workspace, controls, and reproducible exports (#9866)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import cv2
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor, QResizeEvent, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox,
    QBoxLayout,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QPlainTextEdit,
    QScrollArea,
    QSplitter,
    QTabWidget,
    QTableWidgetItem,
    QSizePolicy,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.pose_estimation.observations import CameraCalibration
from src.motion_capture.reference.evidence import CameraSnapshot
from src.motion_capture.reference.scene import session_camera, session_clock
from src.motion_capture.reference.comparison import (
    ComparisonLayer,
    ComparisonSession,
    comparison_session_path,
    load_comparison_session,
    save_comparison_session,
)
from src.motion_capture.reference.model import Asset
from src.motion_capture.coaching.storage import load_layer
from src.motion_capture.rig.edits import ViewEdit, load_edits
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
)
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_export import (
    ComparisonVideoExportOptions,
    export_comparison_video,
)

from . import styling
from .annotate_widget import ImageCanvas
from .flow_layout import FlowLayout
from .reference_controls import SpatialControls, TimeControls
from .reference_timeline import ReferenceTimeline
from .player import VideoReader
from .overlay import PoseTrack
from .reference_rendering import ComparisonRenderContext, ComparisonRenderer
from .session import load_session
from .swing_export_actions import ExportJob, ExportJobSpec, SwingExportActions


class ReferenceComparisonDialog(QDialog):
    """Reference comparison workspace: dual playback, controls, and export."""

    def __init__(
        self,
        root: Path,
        view: str,
        library: ReferenceLibrary,
        parent: QWidget | None = None,
        *,
        review_stale: bool = False,
    ) -> None:
        super().__init__(parent)
        self.root = root
        self.view = view
        self.library = library
        self._review_stale = review_stale
        self._stale_original: ComparisonSession | None = None
        media = load_session(root)
        view_media = media.view(view)
        if view_media.recording is None:
            raise ValueError("View has no original recording")

        self.reader = VideoReader(view_media.recording)
        self.fps = self.reader.fps or 30.0
        self._camera: CameraCalibration | None = None
        self._camera_snapshot: CameraSnapshot | None = None
        self._camera_problem = ""
        self._clock = session_clock(media.timing, view)
        self._load_camera()

        self.assets: list[Asset] = library.list(archived=False)
        self._current_asset: Asset | None = self.assets[0] if self.assets else None
        try:
            self._session = self._find_or_create_session()
            self._load_render_sources()
        except (ValueError, OSError):
            self.reader.close()
            raise
        self._renderer = (
            ComparisonRenderer(self._current_asset) if self._current_asset else None
        )

        self._saved_session = self._stale_original or self._session
        self._undo: list[ComparisonSession] = []
        self.canvas = ImageCanvas()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self.exporter = SwingExportActions(
            self,
            root,
            view=lambda: self.view,
            save=self.save,
            status=self._export_status,
            job=ExportJobSpec(
                self._export_job, "Export Comparison Video", "comparison.mp4"
            ),
            label="Export Comparison Video…",
        )

        self.setWindowTitle(f"Reference Comparison · {view}[*]")
        self.resize(1000, 720)
        self._build_ui()
        self._shortcuts: list[QShortcut] = []
        for key, action in (
            ("Ctrl+S", self.save),
            ("Alt+P", self._toggle_play),
            ("Alt+Left", lambda: self.slider.setValue(self.slider.value() - 1)),
            ("Alt+Right", lambda: self.slider.setValue(self.slider.value() + 1)),
            ("Alt+U", self.undo_change),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)
            self._shortcuts.append(shortcut)
        self.play_button.setToolTip(
            "Play / pause: Alt+P. Step one frame: Alt+Left / Alt+Right."
        )
        self.save_button.setToolTip(
            "Save comparison: Ctrl+S. Undo the last alignment change: Alt+U."
        )
        styling.apply_theme(self)
        self._show_frame(self.slider.minimum())

    def _load_render_sources(self) -> None:
        self._edit = load_edits(self.root).views.get(self.view, ViewEdit())
        if self._edit.first >= self.reader.frame_count or (
            self._edit.last is not None and self._edit.last >= self.reader.frame_count
        ):
            raise ValueError("Selection exceeds the decodable recording")
        crop = self._edit.crop
        if crop:
            crop.validate_size(self.reader.width, self.reader.height)
        self._drawings = load_layer(
            self.root,
            self.view,
            self.reader.width,
            self.reader.height,
            self.reader.frame_count,
        )
        observations = load_session(self.root).view(self.view).observations
        self._track = PoseTrack.load(observations) if observations else None

    def _load_camera(self) -> None:
        try:
            self._camera_snapshot = session_camera(self.root, "", self.view)
            self._camera = self._camera_snapshot.record()
        except (ValueError, OSError, KeyError) as exc:
            self._camera_problem = str(exc)
            self._camera = None

    def _find_or_create_session(self) -> ComparisonSession:
        if self._current_asset is None:
            return ComparisonSession(
                session_root=str(self.root),
                view=self.view,
                reference_id="00000000-0000-0000-0000-000000000000",
                reference_kind="motion",
            )
        path = comparison_session_path(self.root, self.view, self._current_asset.id)
        if path.is_file():
            saved = load_comparison_session(path)
            if (saved.view, saved.reference_id, saved.reference_kind) != (
                self.view,
                self._current_asset.id,
                self._current_asset.kind,
            ):
                raise ValueError(
                    "Saved comparison does not match this view and reference"
                )
            registration = saved.registration
            if registration and registration.asset_sha256:
                try:
                    registration.validate_binding(
                        self._current_asset, self._camera_snapshot, self._clock
                    )
                except ValueError as exc:
                    if not self._review_stale:
                        raise
                    answer = QMessageBox.question(
                        self,
                        "Review Changed Reference Evidence",
                        f"{exc}. Open the saved placement and events for manual review? "
                        "Saving will keep a copy of the previous settings.",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if answer != QMessageBox.StandardButton.Yes:
                        raise
                    self._stale_original = saved
                    registration = ReferenceRegistration.model_validate(
                        registration.model_dump()
                        | {
                            "asset_sha256": None,
                            "camera": None,
                            "clock": None,
                            "is_calibrated": False,
                        }
                    )
                    return saved.changed(registration=registration)
            return saved
        return ComparisonSession(
            session_root=str(self.root),
            view=self.view,
            reference_id=self._current_asset.id,
            reference_kind=self._current_asset.kind,
            registration=ReferenceRegistration(
                reference_id=self._current_asset.id,
                calibration_id="calib_session",
                is_calibrated=False,
                assumption_labels=(
                    "Manual reference-to-scene alignment",
                    self._clock.source,
                ),
            ),
            layer=ComparisonLayer(),
        )

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.asset_selector = QComboBox()
        self.asset_selector.setAccessibleName("Expert Reference Asset")
        self.asset_selector.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
        )
        self.asset_selector.setMinimumContentsLength(16)
        for asset in self.assets:
            self.asset_selector.addItem(f"{asset.title} ({asset.kind})", asset.id)
        self.asset_selector.currentIndexChanged.connect(self._asset_changed)
        header.addWidget(QLabel("Expert Reference"))
        header.addWidget(self.asset_selector, 1)
        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setMaximumWidth(360)
        self._update_status_label()
        header.addWidget(self.status_label)
        layout.addLayout(header)
        guide = QLabel(
            "1. Choose an expert  ·  2. Place the reference  ·  3. Pair swing events  ·  4. Save or export"
        )
        guide.setWordWrap(True)
        layout.addWidget(guide)
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.addWidget(self.canvas, 1)
        preview_layout.addLayout(self._playback_row())
        self.splitter.addWidget(preview)
        self.inspector = self._build_inspector()
        self.splitter.addWidget(self.inspector)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setStretchFactor(0, 3)
        self.splitter.setStretchFactor(1, 1)
        layout.addWidget(self.splitter, 1)
        buttons = FlowLayout(spacing=6)
        for label, slot in (
            ("Save Comparison", self.save),
            ("Undo Change", self.undo_change),
            ("Reset Alignment", self.reset_alignment),
        ):
            button = QPushButton(label)
            button.clicked.connect(slot)
            if label == "Save Comparison":
                self.save_button = button
            buttons.add_widget(button)
        buttons.add_widget(self.exporter.button)
        close = QPushButton("Close")
        close.clicked.connect(self.close)
        buttons.add_widget(close)
        layout.addLayout(buttons)
        self._adapt_layout()

    def _playback_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        row.addWidget(self.play_button)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Player Source Frame")
        self.slider.setRange(
            self._edit.first,
            self._edit.last
            if self._edit.last is not None
            else self.reader.frame_count - 1,
        )
        self.slider.valueChanged.connect(self._show_frame)
        row.addWidget(self.slider, 1)
        self.clock_label = QLabel()
        row.addWidget(self.clock_label)
        return row

    def _build_inspector(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setStyleSheet(styling.compact_tabs_style())
        tabs.setMinimumWidth(270)
        tabs.setMinimumHeight(220)
        registration = self._session.registration or ReferenceRegistration(
            reference_id=self._session.reference_id, calibration_id="unavailable"
        )
        asset = self._current_asset
        size = (self.reader.width, self.reader.height)
        reference_size = (
            (asset.width, asset.height) if asset and asset.kind == "video" else size
        )
        self.spatial = SpatialControls(
            registration,
            asset.kind if asset else "motion",
            size,
            reference_size=reference_size,
        )
        self.spatial.changed.connect(self._registration_applied)
        self.spatial.pending_changed.connect(self._pending_changed)
        self.scale_spin = self.spatial.scale
        self.scale_spin.valueChanged.connect(self._reg_changed)
        tabs.addTab(self._scroll(self.spatial), "Placement")
        self.timing = TimeControls(registration)
        self.timing.changed.connect(self._registration_applied)
        self.timing.pending_changed.connect(self._pending_changed)
        self.offset_spin = self.timing.offset
        self.offset_spin.valueChanged.connect(self._reg_changed)
        timing_page = QWidget()
        timing_layout = QBoxLayout(QBoxLayout.Direction.TopToBottom, timing_page)
        self.timing_layout = timing_layout
        expert_page = QWidget()
        expert_layout = QVBoxLayout(expert_page)
        expert_layout.setContentsMargins(0, 0, 0, 0)
        self.expert_timeline = (
            ReferenceTimeline(asset, registration, self._camera, size)
            if asset
            else None
        )
        if self.expert_timeline:
            pair = QPushButton("Pair Current Frames…")
            pair.clicked.connect(self._pair_frames)
            expert_layout.addWidget(pair)
            expert_layout.addWidget(self.expert_timeline)
            expert_layout.addStretch()
            timing_layout.addWidget(expert_page, 1)
        timing_layout.addWidget(self.timing, 2)
        tabs.addTab(self._scroll(timing_page), "Timing")
        tabs.addTab(self._scroll(self._appearance_page()), "Notes")
        tabs.currentChanged.connect(lambda: self._show_frame(self.slider.value()))
        return tabs

    @staticmethod
    def _scroll(content: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(content)
        area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        return area

    def _appearance_page(self) -> QWidget:
        page = QWidget()
        form = QFormLayout(page)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.visible_check = QCheckBox("Show Reference Layer")
        self.visible_check.setChecked(self._session.layer_visible)
        self.visible_check.toggled.connect(self._layer_changed)
        form.addRow(self.visible_check)
        self.colour_button = QPushButton("Reference Colour…")
        self.colour_button.clicked.connect(self._choose_colour)
        form.addRow(self.colour_button)
        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0, 1)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self._session.layer_opacity)
        self.opacity_spin.setAccessibleName("Reference Opacity")
        self.opacity_spin.valueChanged.connect(self._layer_changed)
        form.addRow("Opacity", self.opacity_spin)
        self.notes = QPlainTextEdit(self._session.notes)
        self.notes.setAccessibleName("Comparison Lesson Notes")
        self.notes.setPlaceholderText(
            "Record the coaching objective, alignment assumptions and observations…"
        )
        self.notes.textChanged.connect(self._notes_changed)
        form.addRow("Lesson Notes", self.notes)
        return page

    def _adapt_layout(self) -> None:
        compact = self.width() < 1100
        orientation = Qt.Orientation.Vertical if compact else Qt.Orientation.Horizontal
        if self.splitter.orientation() != orientation:
            self.splitter.setOrientation(orientation)
            self.splitter.setSizes([440, 260] if compact else [760, 320])
        self.timing_layout.setDirection(
            QBoxLayout.Direction.LeftToRight
            if compact
            else QBoxLayout.Direction.TopToBottom
        )
        self.inspector.setMaximumWidth(16777215 if compact else 410)
        self.inspector.setMaximumHeight(340 if compact else 16777215)

    def resizeEvent(self, event: QResizeEvent | None) -> None:  # noqa: N802
        super().resizeEvent(event)
        if hasattr(self, "splitter"):
            self._adapt_layout()

    def _update_status_label(self) -> None:
        if self._current_asset is None:
            self.status_label.setText("No reference loaded")
            self.status_label.setStyleSheet(styling.chip_style("neutral"))
            return
        self.status_label.setToolTip(self._camera_problem or self._clock.source)
        if self._current_asset.kind == "motion":
            if self._camera:
                self.status_label.setText("Camera Projection · Manual Alignment")
                self.status_label.setStyleSheet(styling.chip_style("ok"))
            else:
                self.status_label.setText("No Usable Camera · Projection Unavailable")
                self.status_label.setStyleSheet(styling.chip_style("warning"))
        else:
            self.status_label.setText("Expert Video · 2D Alignment")
            self.status_label.setStyleSheet(styling.chip_style("neutral"))

    def _asset_changed(self, index: int) -> None:
        if 0 <= index < len(self.assets):
            previous = self._current_asset
            if not self._confirm_unsaved():
                with QSignalBlocker(self.asset_selector):
                    self.asset_selector.setCurrentIndex(
                        self.assets.index(previous) if previous else -1
                    )
                return
            previous_stale = self._stale_original
            self._stale_original = None
            self._current_asset = self.assets[index]
            try:
                saved = self._find_or_create_session()
            except (ValueError, OSError) as exc:
                self._current_asset = previous
                self._stale_original = previous_stale
                with QSignalBlocker(self.asset_selector):
                    self.asset_selector.setCurrentIndex(
                        self.assets.index(previous) if previous else -1
                    )
                self.status_label.setText(f"Cannot load comparison: {exc}")
                return
            self._session = saved
            self._saved_session = self._stale_original or saved
            self._undo.clear()
            if self._renderer:
                self._renderer.close()
            self._renderer = ComparisonRenderer(self._current_asset)
            self._replace_inspector()
            self._update_status_label()
            self._show_frame(self.slider.value())

    def _replace_inspector(self) -> None:
        if self.expert_timeline:
            self.expert_timeline.dispose()
        old = self.inspector
        old.setParent(None)
        self.inspector = self._build_inspector()
        self.splitter.addWidget(self.inspector)
        old.deleteLater()
        self._adapt_layout()
        self._pending_changed()

    def _remember(self) -> None:
        self._undo.append(self._session)
        self._undo = self._undo[-100:]

    def _registration_applied(self, registration: ReferenceRegistration) -> None:
        self._remember()
        self._session = self._session.changed(registration=registration)
        self.spatial.registration = registration
        self.timing.registration = registration
        if self.expert_timeline:
            self.expert_timeline.registration = registration
        self._pending_changed()
        self._show_frame(self.slider.value())

    def _notes_changed(self) -> None:
        text = self.notes.toPlainText()
        if len(text) > 50000:
            self.status_label.setText("Keep comparison notes within 50,000 characters.")
            return
        self._session = self._session.changed(notes=text)
        self._pending_changed()

    def _dirty(self) -> bool:
        return (
            self._session != self._saved_session
            or self.spatial.pending
            or self.timing.pending
            or self.notes.toPlainText() != self._session.notes
        )

    def _pending_changed(self) -> None:
        self.setWindowModified(self._dirty())

    def _apply_pending(self) -> bool:
        if len(self.notes.toPlainText()) > 50000:
            return False
        if self.spatial.pending:
            applied = (
                self.spatial.apply_image()
                if self._current_asset and self._current_asset.kind == "video"
                else self.spatial.apply_placement()
            )
            if not applied:
                return False
        return not (self.timing.pending and not self.timing.apply_events())

    def _confirm_unsaved(self) -> bool:
        if not self._dirty():
            return True
        self._timer.stop()
        self.play_button.setText("Play")
        answer = QMessageBox.question(
            self,
            "Unsaved Comparison",
            "Save the comparison and pending alignment edits?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard or (
            answer == QMessageBox.StandardButton.Save and self.save()
        )

    def undo_change(self) -> None:
        if self.spatial.pending or self.timing.pending:
            self._replace_inspector()
        elif self._undo:
            self._session = self._undo.pop()
            self._replace_inspector()
        else:
            return
        self._show_frame(self.slider.value())

    def reset_alignment(self) -> None:
        if self._current_asset is None:
            return
        self._remember()
        registration = ReferenceRegistration(
            reference_id=self._current_asset.id,
            calibration_id="manual",
            assumption_labels=(
                "Manual reference-to-scene alignment",
                self._clock.source,
            ),
        )
        self._session = self._session.changed(registration=registration)
        self._replace_inspector()
        self._show_frame(self.slider.value())
        self.status_label.setText(
            "Alignment Reset · Undo Restores the Previous Settings"
        )

    def _pair_frames(self) -> None:
        if self.expert_timeline is None:
            return
        name, accepted = QInputDialog.getText(
            self, "Pair Swing Event", "Event Name", text="Impact"
        )
        if not accepted:
            return
        table = self.timing.events
        if table.rowCount() >= 32:
            self.status_label.setText("Use at most 32 paired events.")
            return
        row = table.rowCount()
        table.insertRow(row)
        times = (
            name.strip(),
            str(self.expert_timeline.reference_time),
            str(self._clock.player_time(self.slider.value() / self.fps)),
        )
        for column, value in enumerate(times):
            table.setItem(row, column, QTableWidgetItem(value))
        self.timing.apply_events()

    def _layer_changed(self) -> None:
        self._remember()
        self._session = self._session.with_layer(
            opacity=self.opacity_spin.value(),
            visible=self.visible_check.isChecked(),
        )
        self._show_frame(self.slider.value())

    def _reg_changed(self) -> None:
        if self._session.registration:
            reg = self._session.registration
            values = reg.model_dump()
            if self.sender() is self.scale_spin:
                values["transform"] = reg.transform.model_dump() | {
                    "scale": self.scale_spin.value()
                }
            else:
                values["time_mapping"] = reg.time_mapping.model_dump() | {
                    "offset_s": self.offset_spin.value()
                }
            new_reg = reg.model_validate(values)
            self._registration_applied(new_reg)

    def _choose_colour(self) -> None:
        col = QColorDialog.getColor(
            QColor(self._session.layer_colour), self, "Reference Colour"
        )
        if col.isValid():
            self._remember()
            self._session = self._session.with_layer(colour=col.name())
            self._show_frame(self.slider.value())

    def _show_frame(self, frame_idx: int) -> None:
        img = self.reader.read(frame_idx)
        if img is None:
            self._timer.stop()
            self.play_button.setText("Play")
            self.status_label.setText(f"Could not decode source frame {frame_idx}")
            return
        t_scene = self._clock.player_time(frame_idx / self.fps)
        self.clock_label.setText(f"{t_scene:.3f}s (f{frame_idx})")

        if self._current_asset and self._session.registration and self._renderer:
            registration = self._session.registration
            registration = registration.model_copy(update={"clock": self._clock})
            context = ComparisonRenderContext(
                self.view,
                self._current_asset,
                registration,
                self._session.layer,
                crop=self._edit.crop,
                drawings=self._drawings,
                track=self._track,
            )
            try:
                img = self._renderer.image(
                    self.reader, frame_idx, context, self._camera
                )
            except (ValueError, OSError, cv2.error) as exc:
                self._timer.stop()
                self.play_button.setText("Play")
                self.status_label.setText(f"Cannot render comparison: {exc}")
                return
        self.canvas.set_image(img)
        if self.expert_timeline:
            self.expert_timeline.follow(t_scene)
        self._pending_changed()

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(self.slider.minimum())
            self._timer.start(max(1, round(1000 / self.fps)))
            self.play_button.setText("Pause")

    def _advance(self) -> None:
        if self.slider.value() >= self.slider.maximum():
            self._toggle_play()
        else:
            self.slider.setValue(self.slider.value() + 1)

    def save(self) -> bool:
        if self._current_asset is None:
            return False
        try:
            if not self._apply_pending():
                self.status_label.setText(
                    "Resolve the pending alignment fields before saving."
                )
                return False
            registration = self._session.registration
            if registration is None:
                raise ValueError("Reference registration is unavailable")
            registration = registration.bound(
                self._current_asset, self._camera_snapshot, self._clock
            )
            saved = self._session.changed(registration=registration)
            if self._stale_original:
                path = comparison_session_path(
                    self.root, self.view, self._current_asset.id
                )
                backup = path.with_name(f"{path.stem}.before-review-{uuid4()}.json")
                with backup.open("xb") as stream:
                    stream.write(path.read_bytes())
            save_comparison_session(saved, self.root)
            self._stale_original = None
            self._session = saved
            self.spatial.registration = registration
            self.timing.registration = registration
            if self.expert_timeline:
                self.expert_timeline.registration = registration
            self._saved_session = saved
            self._pending_changed()
            self.status_label.setText("Comparison Saved")
            return True
        except (ValueError, OSError) as exc:
            self.status_label.setText(str(exc))
            return False

    def export_video(self) -> None:
        self.exporter.choose_output()

    def _export_status(self, text: str) -> None:
        self.status_label.setText(text)

    def _export_job(self) -> ExportJob:
        # Snapshot before the worker starts; later UI changes affect the next export.
        asset, registration = self._current_asset, self._session.registration
        if asset is None or registration is None:
            raise ValueError("Choose a reference with a saved registration")
        registration.validate_binding(
            self.library.load(asset.id), self._camera_snapshot, self._clock
        )
        self._load_render_sources()
        self._show_frame(self.slider.value())
        root, view, camera, layer = (
            self.root,
            self.view,
            self._camera,
            self._session.layer,
        )

        def run(
            out: Path,
            cancelled: Callable[[], bool],
            progress: Callable[[int, int], None],
        ) -> None:
            export_comparison_video(
                root,
                view,
                asset,
                registration,
                layer,
                out,
                options=ComparisonVideoExportOptions(
                    camera=camera, cancelled=cancelled, progress=progress
                ),
            )

        return run

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        self._timer.stop()
        if not self.exporter.can_close():
            event.ignore()
            return
        if not self._confirm_unsaved():
            event.ignore()
            return
        self.reader.close()
        if self.expert_timeline:
            self.expert_timeline.dispose()
        if self._renderer:
            self._renderer.close()
        event.accept()

    def reject(self) -> None:
        """Escape uses the same export cancellation guard as the close button."""
        self.close()


def show_reference_comparison(
    root: Path, library: ReferenceLibrary, parent: QWidget, view: str | None = None
) -> None:
    """Entry point from Capture Library or Rig tools."""
    try:
        if view is None:
            media = load_session(root)
            views = [item.view for item in media.views if item.recording is not None]
            if not views:
                raise ValueError("This capture has no original recordings")
            view, ok = QInputDialog.getItem(
                parent, "Compare Reference", "Camera View", views, editable=False
            )
            if not ok:
                return
        ReferenceComparisonDialog(root, view, library, parent, review_stale=True).exec()
    except (ValueError, OSError, cv2.error) as exc:
        QMessageBox.warning(parent, "Reference Comparison", str(exc))
