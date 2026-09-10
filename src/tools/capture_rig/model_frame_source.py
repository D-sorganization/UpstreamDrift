"""Virtual model frames for the same pixel-based analysis used by capture video."""

from __future__ import annotations

from math import floor, isfinite, radians, sin
from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.motion_capture.coaching import ReferenceGeometry
from src.motion_capture.reconstruct.cameras import (
    PinholeCamera,
    intrinsics_from_fov,
    look_at,
)
from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.evidence import CameraSnapshot, asset_identity
from src.motion_capture.reference.model import ReferenceMotion
from src.motion_capture.reference.registration import ReferenceRegistration

from .reference_rendering import ComparisonRenderContext, ComparisonRenderer

MAX_DISPLAY_FRAMES = 100_000


class ModelViewRecipe(BaseModel):
    """Portable model view with an explicitly virtual camera and actual source clock."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)
    schema_version: Literal["model-view/1.0.0"] = "model-view/1.0.0"
    asset: ReferenceMotion
    registration: ReferenceRegistration
    camera: CameraSnapshot
    appearance: ComparisonLayer = Field(default_factory=ComparisonLayer)
    geometry: ReferenceGeometry | None = None
    fps: float = Field(default=30, ge=1, le=360)

    @property
    def frame_count(self) -> int:
        return (
            floor((self.asset.time_s[-1] - self.asset.time_s[0]) * self.fps + 1e-8) + 1
        )

    @model_validator(mode="after")
    def valid_binding(self) -> Self:
        duration_frames = (self.asset.time_s[-1] - self.asset.time_s[0]) * self.fps
        if not isfinite(duration_frames) or duration_frames >= MAX_DISPLAY_FRAMES:
            raise ValueError(
                "Model view exceeds 100000 display frames; trim the source"
            )
        if self.registration.reference_id != self.asset.id:
            raise ValueError("Model view registration belongs to another asset")
        if self.registration.is_calibrated or self.registration.clock is not None:
            raise ValueError("A model-only view must use an uncalibrated source clock")
        if self.geometry and self.geometry.scene_id != asset_identity(self.asset):
            raise ValueError("Model view geometry belongs to a different scene")
        width, height = self.camera.image_size_px
        if not 1 <= width <= 4096 or not 1 <= height <= 4096:
            raise ValueError(
                "Virtual image dimensions must be between 1 and 4096 pixels"
            )
        return self


class ModelFrameSource:
    """Random-access synthetic frames; rendering never edits fitted geometry.

    Samples use a uniform display clock starting at the asset's first sample.
    The fixed camera frames the complete trajectory, so scrubbing cannot change
    apparent scale. This object owns no capture file or recording manifest.
    """

    def __init__(self, recipe: ModelViewRecipe) -> None:
        if not isinstance(recipe, ModelViewRecipe):
            raise TypeError("ModelFrameSource requires a ModelViewRecipe")
        self.recipe = recipe
        self.width, self.height = recipe.camera.image_size_px
        self.fps = recipe.fps
        self.frame_count = recipe.frame_count
        self._scene_id = asset_identity(recipe.asset)
        self._renderer = ComparisonRenderer(recipe.asset)
        self._camera = recipe.camera.record()
        self._closed = False

    @classmethod
    def from_motion(
        cls,
        asset: ReferenceMotion,
        *,
        fps: float | None = None,
        size: tuple[int, int] = (960, 540),
    ) -> Self:
        """Fit one virtual camera to all finite model samples without changing units."""
        if not isinstance(asset, ReferenceMotion):
            raise TypeError("Virtual model playback requires a ReferenceMotion")
        if fps is None:
            intervals = np.diff(asset.time_s)
            fps = (
                max(1.0, min(360.0, 1 / float(np.median(intervals))))
                if len(intervals)
                else 30.0
            )
        if not isfinite(fps) or not 1 <= fps <= 360:
            raise ValueError("Display frame rate must be in [1, 360]")
        if len(size) != 2 or any(
            type(value) is not int or not 1 <= value <= 4096 for value in size
        ):
            raise ValueError("Virtual image dimensions must be integers in [1, 4096]")
        registration = ReferenceRegistration(
            reference_id=asset.id,
            calibration_id="virtual-model-view",
            is_calibrated=False,
        )
        points = np.asarray(
            [point for frame in asset.points_m for point in frame if point is not None],
            dtype=float,
        )
        if not len(points):
            raise ValueError("Model view needs at least one observed position")
        world = registration.place_points(points)
        centre = (world.min(axis=0) + world.max(axis=0)) / 2
        radius = max(float(np.linalg.norm(world - centre, axis=1).max()), 0.1)
        fov = 50.0
        limiting_angle = min(
            radians(fov / 2),
            float(np.arctan(np.tan(radians(fov / 2)) * size[1] / size[0])),
        )
        distance = radius / sin(limiting_angle) * 1.1
        direction = np.array([0.25, 0.15, 1.0])
        position = centre + direction / np.linalg.norm(direction) * distance
        camera = PinholeCamera(
            camera_id="virtual-model-view",
            matrix=intrinsics_from_fov(*size, fov),
            rotation_world_from_camera=look_at(position, centre),
            translation_world_from_camera_m=position,
            image_size_px=size,
        )
        snapshot = CameraSnapshot.from_calibration(
            camera.to_calibration(),
            provenance="Virtual model-only camera; no capture calibration implied",
        )
        return cls(
            ModelViewRecipe(
                asset=asset, registration=registration, camera=snapshot, fps=fps
            )
        )

    def time_at(self, index: int) -> float:
        """Return actual asset-clock seconds for an in-range display frame."""
        if type(index) is not int or not 0 <= index < self.frame_count:
            raise ValueError("Model frame index is outside the display timeline")
        asset = self.recipe.asset
        return asset.time_s[0] + index / self.fps

    def read(self, index: int) -> np.ndarray | None:
        """Return a fresh BGR frame, or None beyond the display timeline."""
        if self._closed:
            raise ValueError("Model frame source is closed")
        if type(index) is not int or index < 0:
            raise ValueError("Model frame index must be a nonnegative integer")
        if index >= self.frame_count:
            return None
        recipe = self.recipe
        mapping = recipe.registration.time_mapping
        time = mapping.reference_to_scene(self.time_at(index))
        context = ComparisonRenderContext(
            "virtual-model-view",
            recipe.asset,
            recipe.registration,
            recipe.appearance,
            geometry=recipe.geometry,
            scene_id=self._scene_id,
        )
        frame = np.full((self.height, self.width, 3), 32, dtype=np.uint8)
        return self._renderer.overlay(frame, time, context, self._camera)

    def close(self) -> None:
        """Release compositor ownership; subsequent reads fail explicitly."""
        self._renderer.close()
        self._closed = True
