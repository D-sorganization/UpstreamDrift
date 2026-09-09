"""Reference comparison workspace, controls, and reproducible exports (#9866)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal
from PyQt6.QtGui import QCloseEvent, QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison import (
    ComparisonLayer,
    ComparisonSession,
    comparison_session_path,
    load_comparison_session,
    save_comparison_session,
)
from src.motion_capture.reference.model import Asset
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
    TimeMapping,
)
from src.motion_capture.reference.storage import ReferenceLibrary
from src.tools.capture_rig.reference_export import (
    ComparisonExportWorker,
    draw_reference_overlay,
    export_comparison_video,
)

from . import styling
from .annotate_widget import ImageCanvas
from .flow_layout import FlowLayout
from .player import VideoReader
from .session import load_session


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
        self._camera: PinholeCamera | None = None
        self._load_camera()

        self.assets: list[Asset] = library.list(archived=False)
        self._current_asset: Asset | None = self.assets[0] if self.assets else None
        self._session = self._find_or_create_session()

        self.canvas = ImageCanvas()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._worker: ComparisonExportWorker | None = None

        self.setWindowTitle(f"Reference Comparison · {view}")
        self.resize(1000, 720)
        self._build_ui()
        styling.apply_theme(self)
        self._show_frame(0)

    def _load_camera(self) -> None:
        reconstruct_json = self.root / "reconstruct" / "reconstruction.json"
        if reconstruct_json.is_file():
            try:
                from src.motion_capture.reconstruct.fit import Reconstruction

                recon = Reconstruction.model_validate_json(
                    reconstruct_json.read_text(encoding="utf-8")
                )
                for cam_dict in recon.cameras:
                    if cam_dict.get("camera_id") == self.view:
                        from src.shared.python.pose_estimation.observations import (
                            CameraCalibration,
                        )

                        calib = CameraCalibration.from_dict(cam_dict)
                        self._camera = PinholeCamera.from_calibration(calib)
                        break
            except (ValueError, OSError, KeyError):
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
            try:
                return load_comparison_session(path)
            except (ValueError, OSError):
                pass
        return ComparisonSession(
            session_root=str(self.root),
            view=self.view,
            reference_id=self._current_asset.id,
            reference_kind=self._current_asset.kind,
            registration=ReferenceRegistration(
                reference_id=self._current_asset.id,
                calibration_id="calib_session",
                is_calibrated=self._camera is not None,
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
        self.slider.setRange(0, max(0, self.reader.frame_count - 1))
        self.slider.valueChanged.connect(self._show_frame)
        slider_row.addWidget(self.slider, 1)

        self.clock_label = QLabel("0.000s")
        slider_row.addWidget(self.clock_label)
        main_layout.addLayout(slider_row)

        controls = QFormLayout()
        layer_row = QHBoxLayout()
        self.visible_check = QCheckBox("Visible")
        self.visible_check.setChecked(self._session.layer.visible)
        self.visible_check.toggled.connect(self._layer_changed)
        layer_row.addWidget(self.visible_check)

        self.colour_button = QPushButton("Colour…")
        self.colour_button.clicked.connect(self._choose_colour)
        layer_row.addWidget(self.colour_button)

        self.opacity_spin = QDoubleSpinBox()
        self.opacity_spin.setRange(0.0, 1.0)
        self.opacity_spin.setSingleStep(0.05)
        self.opacity_spin.setValue(self._session.layer.opacity)
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

        export_btn = QPushButton("Export Comparison Video…")
        export_btn.clicked.connect(self.export_video)
        buttons.add_widget(export_btn)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        buttons.add_widget(close_btn)
        main_layout.addLayout(buttons)

    def _update_status_label(self) -> None:
        if self._current_asset is None:
            self.status_label.setText("No reference loaded")
            return
        if self._current_asset.kind == "motion":
            if self._camera:
                self.status_label.setText("3D Calibrated Projection")
                self.status_label.setStyleSheet("color: #44dd44;")
            else:
                self.status_label.setText("3D Uncalibrated (No camera match)")
                self.status_label.setStyleSheet("color: #ffaa00;")
        else:
            self.status_label.setText("2D Homography (No 3D claims)")
            self.status_label.setStyleSheet("color: #44aaff;")

    def _asset_changed(self, index: int) -> None:
        if 0 <= index < len(self.assets):
            self._current_asset = self.assets[index]
            self._session = self._find_or_create_session()
            self._update_status_label()
            self.visible_check.setChecked(self._session.layer.visible)
            self.opacity_spin.setValue(self._session.layer.opacity)
            reg = self._session.registration
            self.offset_spin.setValue(reg.time_mapping.offset_s if reg else 0.0)
            self.scale_spin.setValue(reg.transform.scale if reg else 1.0)
            self._show_frame(self.slider.value())

    def _layer_changed(self) -> None:
        self._session = self._session.changed(
            layer=self._session.layer.model_validate(
                {
                    "colour": self._session.layer.colour,
                    "opacity": self.opacity_spin.value(),
                    "visible": self.visible_check.isChecked(),
                    "line_width": self._session.layer.line_width,
                }
            )
        )
        self._show_frame(self.slider.value())

    def _reg_changed(self) -> None:
        if self._session.registration:
            reg = self._session.registration
            new_reg = reg.model_validate(
                {
                    "reference_id": reg.reference_id,
                    "calibration_id": reg.calibration_id,
                    "transform": ReferenceTransform(scale=self.scale_spin.value()),
                    "time_mapping": TimeMapping(offset_s=self.offset_spin.value()),
                    "is_calibrated": bool(self._camera),
                }
            )
            self._session = self._session.changed(registration=new_reg)
            self._show_frame(self.slider.value())

    def _choose_colour(self) -> None:
        col = QColorDialog.getColor(
            QColor(self._session.layer.colour), self, "Reference Colour"
        )
        if col.isValid():
            self._session = self._session.changed(
                layer=self._session.layer.model_validate(
                    self._session.layer.model_dump() | {"colour": col.name()}
                )
            )
            self._show_frame(self.slider.value())

    def _show_frame(self, frame_idx: int) -> None:
        img = self.reader.read(frame_idx)
        if img is None:
            return
        t_scene = frame_idx / self.fps
        self.clock_label.setText(f"{t_scene:.3f}s (f{frame_idx})")

        if self._current_asset and self._session.registration:
            img = draw_reference_overlay(
                img,
                self._current_asset,
                t_scene,
                self._session.registration,
                self._camera,
                self._session.layer,
            )
        self.canvas.set_image(img)

    def _toggle_play(self) -> None:
        if self._timer.isActive():
            self._timer.stop()
            self.play_button.setText("Play")
        else:
            if self.slider.value() >= self.slider.maximum():
                self.slider.setValue(0)
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
            save_comparison_session(self._session, self.root)
            self.status_label.setText("Comparison Saved")
            return True
        except (ValueError, OSError) as exc:
            self.status_label.setText(str(exc))
            return False

    def export_video(self) -> None:
        if self._current_asset is None or self._session.registration is None:
            return
        self.save()
        name, _ = QFileDialog.getSaveFileName(
            self,
            "Export Comparison Video",
            str(self.root / "comparison.mp4"),
            "MP4 Video (*.mp4);;AVI Video (*.avi)",
        )
        if not name:
            return
        out = Path(name)
        progress = QProgressDialog(
            "Exporting comparison video…", "Cancel", 0, 100, self
        )
        progress.setWindowTitle("Exporting")
        progress.setWindowModality(Qt.WindowModality.WindowModal)

        worker = ComparisonExportWorker(
            self.root,
            self.view,
            self._current_asset,
            self._session.registration,
            self._session.layer,
            out,
            self._camera,
            self,
        )
        worker.progress.connect(
            lambda done, tot: progress.setValue(int(done * 100 / tot))
        )
        progress.canceled.connect(worker.requestInterruption)
        worker.finished.connect(lambda: self._export_finished(worker, progress, out))
        worker.start()

    def _export_finished(
        self, worker: ComparisonExportWorker, progress: QProgressDialog, out: Path
    ) -> None:
        progress.close()
        if worker.error:
            QMessageBox.critical(self, "Export Failed", worker.error)
        else:
            QMessageBox.information(
                self, "Export Complete", f"Comparison video saved to:\n{out}"
            )

    def closeEvent(self, event: QCloseEvent | None) -> None:  # noqa: N802
        if event is None:
            return
        self._timer.stop()
        self.reader.close()
        event.accept()


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
