"""Original-grid comparison frames for the shared coaching drawing editor."""

from __future__ import annotations

import numpy as np

from .frame_source import FrameSource
from .overlay import draw_pose
from .reference_rendering import (
    Camera,
    ComparisonRenderContext,
    ComparisonRenderer,
    Image,
)


class ComparisonFrameSource:
    """Own source/renderer while keeping drawings between pose and reference layers.

    ``read`` supplies the original image with detected pose. The canvas calls
    ``finalize`` after drawing edits, before selection handles. Cropping,
    padding and clock labels remain export operations outside the editable grid.
    """

    def __init__(
        self, reader: FrameSource, context: ComparisonRenderContext, camera: Camera
    ) -> None:
        if not np.isfinite(reader.fps) or reader.fps <= 0:
            raise ValueError("Comparison playback requires a positive source clock")
        if min(reader.width, reader.height, reader.frame_count) <= 0:
            raise ValueError("Comparison playback requires a nonempty source")
        self.width, self.height = reader.width, reader.height
        self.fps, self.frame_count = reader.fps, reader.frame_count
        self.context, self.camera = context, camera
        self._reader = reader
        self._renderer = ComparisonRenderer(context.asset)
        self._closed = False

    def time_at(self, index: int) -> float:
        """Return registered scene seconds, rejecting invalid source indices."""
        if self._closed:
            raise ValueError("Comparison source is closed")
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise ValueError("Comparison frame is outside the source timeline")
        registration = self.context.registration
        return registration.scene_time(index / self.fps)

    def read(self, index: int) -> Image | None:
        """Return original-grid player/pose pixels without crop or drawing edits."""
        self.time_at(index)
        image = self._reader.read(index)
        track = self.context.track
        pose = track.at(index) if track else None
        if image is not None and pose is not None and track is not None:
            image = draw_pose(image, pose[0], pose[1], track.edges, min_confidence=0.5)
        return image

    def finalize(self, image: Image, index: int) -> Image:
        """Apply the established reference overlay after the canvas's drawings."""
        time = self.time_at(index)
        if image.dtype != np.uint8 or image.shape != (self.height, self.width, 3):
            raise ValueError("Comparison input must use the original BGR pixel grid")
        return self._renderer.overlay(image, time, self.context, self.camera)

    def close(self) -> None:
        """Release both owned readers; subsequent frame operations are rejected."""
        self._renderer.close()
        self._reader.close()
        self._closed = True
