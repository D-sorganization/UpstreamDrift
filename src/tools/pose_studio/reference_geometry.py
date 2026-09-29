"""Shared geometry controls for the canonical Pose Studio world."""

from pathlib import Path

from PyQt6.QtGui import QCloseEvent
from PyQt6.QtWidgets import (
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
)

from src.motion_capture.coaching import ReferenceGeometry
from src.motion_capture.coaching.native_geometry import NativeGeometryRenderer
from src.tools.capture_rig.geometry_controls import GeometryControls

from .widgets.view_3d import View3D

POSE_STUDIO_SCENE = "pose-studio:canonical-world-v1"


class ReferenceGeometryDialog(QDialog):
    """Edit world references without coupling them to one articulated pose."""

    def __init__(self, viewport: View3D) -> None:
        super().__init__(viewport)
        self.setWindowTitle("Pose Studio · 3D References")
        self.resize(470, 700)
        self.controls = GeometryControls(ReferenceGeometry(scene_id=POSE_STUDIO_SCENE))
        self.renderer = NativeGeometryRenderer(
            viewport, scene_id=POSE_STUDIO_SCENE, frame="world_Zup"
        )
        self.status = QLabel(
            "Coordinates are entered in the shared Y-up world; the viewport shows canonical Z-up. Save references to retain them across sessions."
        )
        self.status.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.status)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(self.controls)
        layout.addWidget(area)
        buttons = QHBoxLayout()
        for title, action in (
            ("Load References…", self.load),
            ("Save References…", self.save),
            ("Close", self.close),
        ):
            button = QPushButton(title)
            button.clicked.connect(action)
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.controls.changed.connect(self._render)

    def _render(self, geometry: ReferenceGeometry) -> None:
        try:
            self.renderer.render(geometry, 0)
        except (ValueError, RuntimeError, OSError) as exc:
            self.status.setText(f"Cannot display references: {exc}")

    def load(self) -> None:
        """Load scene-matching references through the existing validated control."""
        if self.controls.pending and not self.controls.apply_selected():
            return
        path, _ = QFileDialog.getOpenFileName(
            self, "Load 3D References", "", "Reference geometry (*.json)"
        )
        if path:
            try:
                self.controls.load(Path(path))
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def save(self) -> None:
        """Save applied geometry atomically using the same portable document."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save 3D References", "references.json", "Reference geometry (*.json)"
        )
        if path:
            try:
                self.controls.save(Path(path))
                self.status.setText("References Saved")
            except (ValueError, OSError) as exc:
                self.status.setText(str(exc))

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if self.controls.pending and not self.controls.apply_selected():
            if event is not None:
                event.ignore()
            return
        super().closeEvent(event)
