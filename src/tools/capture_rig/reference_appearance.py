"""Saved motion appearance controls shared by the comparison inspector."""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QCheckBox, QFormLayout, QLabel, QWidget

from src.motion_capture.reference.comparison import ComparisonLayer

from .reference_controls import number


class MotionAppearanceControls(QWidget):
    """Compose display choices without changing the source or the body fit."""

    changed = pyqtSignal()

    def __init__(self, layer: ComparisonLayer, *, has_club: bool) -> None:
        super().__init__()
        form = QFormLayout(self)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        self.skeleton = QCheckBox("Show Stick Figure")
        self.joints = QCheckBox("Show Joints")
        self.club = QCheckBox("Show Club")
        self.club.setEnabled(has_club)
        self.club.setToolTip(
            "Requires club connectivity saved with the reference asset."
        )
        self.ellipsoids = QCheckBox("Show 3D Segment Ellipsoids")
        for field, value in (
            (self.skeleton, layer.draw_skeleton),
            (self.joints, layer.draw_joints),
            (self.club, layer.draw_club),
            (self.ellipsoids, layer.draw_ellipsoids),
        ):
            field.setChecked(value)
            field.toggled.connect(self.changed.emit)
            form.addRow(field)
        self.volume_alpha = number("Ellipsoid Opacity", (0, 1))
        self.volume_alpha.setValue(layer.ellipsoid_opacity)
        self.radius = number("Segment Radius / Length", (0.01, 0.5), 0.01)
        self.radius.setValue(layer.segment_radius_ratio)
        for label, field in (
            ("Ellipsoid Opacity", self.volume_alpha),
            ("Radius / Length", self.radius),
        ):
            field.valueChanged.connect(self.changed.emit)
            form.addRow(label, field)
        hint = QLabel(
            "Ellipsoids illustrate segment volume; they are not anatomical measurements. Club endpoints retain their source labels."
        )
        hint.setWordWrap(True)
        form.addRow(hint)

    def updated(self, layer: ComparisonLayer) -> ComparisonLayer:
        """Retain global alpha, colour and line settings edited elsewhere."""
        return ComparisonLayer.model_validate(
            layer.model_dump()
            | {
                "draw_skeleton": self.skeleton.isChecked(),
                "draw_joints": self.joints.isChecked(),
                "draw_club": self.club.isChecked(),
                "draw_ellipsoids": self.ellipsoids.isChecked(),
                "ellipsoid_opacity": self.volume_alpha.value(),
                "segment_radius_ratio": self.radius.value(),
            }
        )
