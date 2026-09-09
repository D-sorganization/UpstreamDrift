"""Reference comparison workspace, controls, and reproducible exports (#9866)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
from PyQt6.QtCore import QSignalBlocker, Qt, QTimer
from PyQt6.QtGui import QCloseEvent, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
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
    ) -> None:
        super().__init__(parent)
        self.root = root
        self.view = view
        self.library = library
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

        self.setWindowTitle(f"Reference Comparison · {view}")
        self.resize(1000, 720)
        self._build_ui()
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
                registration.validate_binding(
                    self._current_asset, self._camera_snapshot, self._clock
                )
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
        main_layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        self.asset_selector = QComboBox()
        for a in self.assets:
            self.asset_selector.addItem(f"{a.title} ({a.kind})", a.id)
        self.asset_selector.currentIndexChanged.connect(self._asset_changed)
        top_row.addWidget(QLabel("Reference Asset:"))
        top_row.addWidget(self.asset_selector, 1)

        self.status_label = QLabel()
        self._update_status_label()
        top_row.addWidget(self.status_label)
        main_layout.addLayout(top_row)

        main_layout.addWidget(self.canvas, 1)

        slider_row = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self._toggle_play)
        slider_row.addWidget(self.play_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(
            self._edit.first,
            self._edit.last
            if self._edit.last is not None
            else max(0, self.reader.frame_count - 1),
        )
        self.slider.valueChanged.connect(self._show_frame)
        slider_row.addWidget(self.slider, 1)

        self.clock_label = QLabel("0.000s")
        slider_row.addWidget(self.clock_label)
        main_layout.addLayout(slider_row)

        controls = QFormLayout()
        layer_row = QHBoxLayout()
        self.visible_check = QCheckBox("Visible")
        self.visible_check.setChecked(self._session.layer_visible)
        self.visible_check.toggled.connect(self._layer_changed)
        layer_row.addWidget(self.visible_check)

        self.colour_button = QPushButton("Colour…")
        self.colour_button.clicked.connect(self._choose_colour)
        layer_row.addWidget(self.colour_button)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self._session.layer_opacity)
        self.opacity_spin.valueChanged.connect(self._layer_changed)
        layer_row.addWidget(QLabel("Opacity:"))
        layer_row.addWidget(self.opacity_spin)

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-60.0, 60.0)
        self.offset_spin.setSingleStep(0.05)
        reg = self._session.registration
        self.offset_spin.setValue(reg.time_mapping.offset_s if reg else 0.0)
        self.offset_spin.valueChanged.connect(self._reg_changed)
        layer_row.addWidget(QLabel("Time Offset (s):"))
        layer_row.addWidget(self.offset_spin)

        self.scale_spin = QDoubleSpinBox()
        self.scale_spin.setRange(0.1, 5.0)
        self.scale_spin.setSingleStep(0.05)
        self.scale_spin.setValue(reg.transform.scale if reg else 1.0)
        self.scale_spin.valueChanged.connect(self._reg_changed)
        layer_row.addWidget(QLabel("Scale:"))
        layer_row.addWidget(self.scale_spin)

        controls.addRow("Layer & Sync:", layer_row)
        main_layout.addLayout(controls)

        buttons = FlowLayout(spacing=6)
        save_btn = QPushButton("Save Comparison")
        save_btn.clicked.connect(self.save)
        buttons.add_widget(save_btn)

        buttons.add_widget(self.exporter.button)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.add_widget(close_btn)
        main_layout.addLayout(buttons)

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
            self.status_label.setText("2D Homography (No 3D claims)")
            self.status_label.setStyleSheet(styling.chip_style("neutral"))

    def _asset_changed(self, index: int) -> None:
        if 0 <= index < len(self.assets):
            previous = self._current_asset
            self._current_asset = self.assets[index]
            try:
                saved = self._find_or_create_session()
            except (ValueError, OSError) as exc:
                self._current_asset = previous
                with QSignalBlocker(self.asset_selector):
                    self.asset_selector.setCurrentIndex(
                        self.assets.index(previous) if previous else -1
                    )
                self.status_label.setText(f"Cannot load comparison: {exc}")
                return
            self._session = saved
            if self._renderer:
                self._renderer.close()
            self._renderer = ComparisonRenderer(self._current_asset)
            self._update_status_label()
            registration = saved.registration
            for control, value in (
                (self.visible_check, saved.layer_visible),
                (self.opacity_spin, saved.layer_opacity),
                (
                    self.offset_spin,
                    registration.time_mapping.offset_s if registration else 0.0,
                ),
                (
                    self.scale_spin,
                    registration.transform.scale if registration else 1.0,
                ),
            ):
                with QSignalBlocker(control):
                    if isinstance(control, QCheckBox):
                        control.setChecked(bool(value))
                    else:
                        control.setValue(float(value))
            self._show_frame(self.slider.value())

    def _layer_changed(self) -> None:
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
            self._session = self._session.changed(registration=new_reg)
            self._show_frame(self.slider.value())

    def _choose_colour(self) -> None:
        col = QColorDialog.getColor(
            QColor(self._session.layer_colour), self, "Reference Colour"
        )
        if col.isValid():
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
            registration = self._session.registration
            if registration is None:
                raise ValueError("Reference registration is unavailable")
            registration = registration.bound(
                self._current_asset, self._camera_snapshot, self._clock
            )
            self._session = self._session.changed(registration=registration)
            save_comparison_session(self._session, self.root)
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
        self.reader.close()
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
        ReferenceComparisonDialog(root, view, library, parent).exec()
    except (ValueError, OSError, cv2.error) as exc:
        QMessageBox.warning(parent, "Reference Comparison", str(exc))
