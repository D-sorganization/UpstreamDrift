"""Independent expert scrubbing to pair visible swing events with player frames."""

from bisect import bisect_left

import cv2
import numpy as np
from PyQt6.QtCore import QSignalBlocker, Qt
from PyQt6.QtWidgets import QCheckBox, QLabel, QSlider, QVBoxLayout, QWidget

from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.model import Asset, ReferenceMotion
from src.motion_capture.reference.registration import ReferenceRegistration
from src.shared.python.pose_estimation.observations import CameraCalibration

from .annotate_widget import ImageCanvas
from .reference_rendering import ComparisonRenderContext, ComparisonRenderer


class ReferenceTimeline(QWidget):
    """An optional source preview with one lazily opened decoder of its own."""

    def __init__(
        self,
        asset: Asset,
        registration: ReferenceRegistration,
        camera: CameraCalibration | None,
        scene_size: tuple[int, int],
    ) -> None:
        super().__init__()
        self.asset, self.registration = asset, registration
        self.camera, self.scene_size = camera, scene_size
        self.renderer = ComparisonRenderer(asset)
        self.scene_time = 0.0
        self.outside = False
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.linked = QCheckBox("Follow Player Clock")
        self.linked.setChecked(True)
        self.linked.toggled.connect(lambda: self.follow(self.scene_time))
        layout.addWidget(self.linked)
        self.canvas = ImageCanvas()
        self.canvas.setMinimumSize(160, 100)
        self.canvas.setMaximumHeight(170)
        layout.addWidget(self.canvas)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setAccessibleName("Expert Source Frame")
        self.slider.setRange(
            0,
            len(asset.time_s) - 1
            if isinstance(asset, ReferenceMotion)
            else asset.frames - 1,
        )
        self.slider.valueChanged.connect(self.update_preview)
        layout.addWidget(self.slider)
        self.clock = QLabel()
        self.clock.setWordWrap(True)
        layout.addWidget(self.clock)
        self.update_preview()

    @property
    def reference_time(self) -> float:
        index = self.slider.value()
        asset = self.asset
        return (
            asset.time_s[index]
            if isinstance(asset, ReferenceMotion)
            else index / asset.fps
        )

    def follow(self, scene_time: float) -> None:
        self.scene_time = scene_time
        self.outside = False
        if self.linked.isChecked():
            mapping = self.registration.time_mapping
            time = mapping.scene_to_reference(scene_time)
            asset = self.asset
            first, last = (
                (asset.time_s[0], asset.time_s[-1])
                if isinstance(asset, ReferenceMotion)
                else (0.0, (asset.frames - 1) / asset.fps)
            )
            self.outside = time < first - 1e-9 or time > last + 1e-9
            if isinstance(asset, ReferenceMotion):
                index = bisect_left(asset.time_s, time)
                if index and (
                    index == len(asset.time_s)
                    or time - asset.time_s[index - 1] < asset.time_s[index] - time
                ):
                    index -= 1
            else:
                index = round(time * asset.fps)
            with QSignalBlocker(self.slider):
                self.slider.setValue(max(0, min(index, self.slider.maximum())))
        self.slider.setEnabled(not self.linked.isChecked())
        self.update_preview()

    def update_preview(self) -> None:
        time = self.reference_time
        self.clock.setText(
            "No expert sample at this player time. Uncheck Follow Player Clock to pair frames."
            if self.outside
            else f"Expert {time:.5f} s · Frame {self.slider.value()}"
        )
        if not self.isVisible():
            return
        asset = self.asset
        width, height = (
            self.scene_size
            if isinstance(asset, ReferenceMotion)
            else (asset.width, asset.height)
        )
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if self.outside:
            self.canvas.set_image(frame)
            return
        registration = self.registration.model_copy(update={"image_transform_2d": None})
        context = ComparisonRenderContext(
            "expert", asset, registration, ComparisonLayer()
        )
        mapping = registration.time_mapping
        try:
            image = self.renderer.overlay(
                frame, mapping.reference_to_scene(time), context, self.camera
            )
            self.canvas.set_image(image)
        except (ValueError, OSError, cv2.error) as exc:
            self.clock.setText(f"Expert Preview Unavailable: {exc}")

    def dispose(self) -> None:
        """Release decoder ownership when replacing this asset or closing the dialog."""
        self.renderer.close()
