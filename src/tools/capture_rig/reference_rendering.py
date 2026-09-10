"""One comparison compositor for preview and exported coaching clips."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from types import TracebackType
from typing import TypeAlias

import cv2
import numpy as np
import numpy.typing as npt

from src.motion_capture.coaching import DrawingLayer, ReferenceGeometry
from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.model import Asset, ReferenceMotion, ReferenceVideo
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    sample_reference_motion,
    project_reference_to_camera,
)
from src.motion_capture.rig.edits import CropRect
from src.shared.python.pose_estimation.observations import CameraCalibration

from .clips import ClipRendering, _rendered
from .geometry_rendering import render_geometry
from .overlay import PoseTrack
from .player import VideoReader
from .reference_volumes import draw_segment_volumes

Image: TypeAlias = npt.NDArray[np.uint8]
Camera: TypeAlias = PinholeCamera | CameraCalibration | None


@dataclass(frozen=True)
class ComparisonRenderContext:
    """Immutable per-view render recipe in original source coordinates."""

    view: str
    asset: Asset
    registration: ReferenceRegistration
    layer: ComparisonLayer
    crop: CropRect | None = None
    track: PoseTrack | None = None
    drawings: DrawingLayer | None = None
    geometry: ReferenceGeometry | None = None
    scene_id: str | None = None

    def __post_init__(self) -> None:
        if self.geometry is not None and self.geometry.scene_id != self.scene_id:
            raise ValueError("Reference geometry belongs to a different scene")


def _motion_image(
    frame: Image, ctx: ComparisonRenderContext, camera: Camera, time: float
) -> Image:
    if camera is None or not isinstance(ctx.asset, ReferenceMotion):
        return frame
    layer = ctx.layer
    world, mask = sample_reference_motion(ctx.asset, ctx.registration, np.array([time]))
    projected, visibility = project_reference_to_camera(world, mask, camera)
    club_edges = set(ctx.asset.club_edges)
    body_edges = tuple(edge for edge in ctx.asset.edges if edge not in club_edges)
    drawn = draw_segment_volumes(
        frame, world[0], mask[0], body_edges, camera, layer
    ).copy()
    points, visible = projected[0], visibility[0]
    if not layer.draw_club:
        club_joints = {j for edge in club_edges for j in edge}
        body_joints = {j for edge in body_edges for j in edge}
        for joint in club_joints - body_joints:
            visible[joint] = False
    pixels = [
        tuple(int(round(v)) for v in point) if valid else (0, 0)
        for point, valid in zip(points, visible, strict=True)
    ]
    for a, b in ctx.asset.edges:
        enabled = layer.draw_club if (a, b) in club_edges else layer.draw_skeleton
        if enabled and visible[a] and visible[b]:
            cv2.line(
                drawn,
                pixels[a],
                pixels[b],
                layer.colour_bgr,
                layer.line_width,
                cv2.LINE_AA,
            )
    if layer.draw_joints:
        for point, valid in zip(pixels, visible, strict=True):
            if valid:
                cv2.circle(
                    drawn,
                    point,
                    max(2, layer.line_width + 1),
                    layer.colour_bgr,
                    -1,
                    cv2.LINE_AA,
                )
    return np.asarray(
        cv2.addWeighted(drawn, layer.opacity, frame, 1 - layer.opacity, 0),
        dtype=np.uint8,
    )


def _image_matrix(
    asset: ReferenceVideo, frame: Image, registration: ReferenceRegistration
) -> npt.NDArray[np.float64]:
    height, width = frame.shape[:2]
    scale = min(width / asset.width, height / asset.height)
    default = (
        (scale, 0, (width - asset.width * scale) / 2),
        (0, scale, (height - asset.height * scale) / 2),
        (0, 0, 1),
    )
    matrix = np.asarray(registration.image_transform_2d or default, dtype=float)
    if not np.isfinite(matrix).all() or np.linalg.cond(matrix) > 1e12:
        raise ValueError("Image alignment must be a finite, invertible homography")
    corners = np.array(
        [
            [0, 0, 1],
            [asset.width, 0, 1],
            [0, asset.height, 1],
            [asset.width, asset.height, 1],
        ]
    )
    depth = corners @ matrix[2]
    if not (np.all(depth > 1e-9) or np.all(depth < -1e-9)):
        raise ValueError("Image alignment crosses a projective horizon")
    return matrix


class ComparisonRenderer:
    """Own one expert decoder; release it on asset changes or dialog/export close."""

    def __init__(self, asset: Asset) -> None:
        self.asset = asset
        self.reader: VideoReader | None = None

    def _video_frame(self, asset: ReferenceVideo, index: int) -> Image:
        if self.reader is None:
            reader = VideoReader(asset.source_path)
            if (reader.width, reader.height, reader.frame_count) != (
                asset.width,
                asset.height,
                asset.frames,
            ) or not np.isclose(reader.fps, asset.fps, rtol=1e-6, atol=1e-6):
                reader.close()
                raise ValueError("Expert video properties changed; import it again")
            self.reader = reader
        image = self.reader.read(index)
        if image is None:
            raise ValueError(f"Could not decode expert frame {index}")
        return image

    def overlay(
        self, frame: Image, time: float, ctx: ComparisonRenderContext, camera: Camera
    ) -> Image:
        """Draw reference pixels after detected pose and drawings, before crop."""
        if ctx.asset.id != self.asset.id:
            raise ValueError("Renderer belongs to a different reference asset")
        if ctx.geometry is not None:
            frame = render_geometry(frame, ctx.geometry, camera, time)
        if not ctx.layer.visible or ctx.layer.opacity <= 0:
            return frame
        asset = self.asset
        if isinstance(asset, ReferenceMotion):
            return _motion_image(frame, ctx, camera, time)
        time_mapping = ctx.registration.time_mapping
        reference_time = time_mapping.scene_to_reference(time)
        if reference_time < 0 or reference_time >= asset.frames / asset.fps:
            return frame
        index = min(round(reference_time * asset.fps), asset.frames - 1)
        image = self._video_frame(asset, index)
        matrix = _image_matrix(asset, frame, ctx.registration)
        size = (frame.shape[1], frame.shape[0])
        # Warp both premultiplied colour and coverage. Valid black expert pixels
        # remain visible; uncovered player pixels are never darkened by borders.
        warped = cv2.warpPerspective(image.astype(np.float32), matrix, size)
        mask = cv2.warpPerspective(np.ones(image.shape[:2], np.float32), matrix, size)
        alpha = ctx.layer.opacity
        mixed = warped * alpha + frame * (1 - mask[:, :, None] * alpha)
        return np.asarray(np.clip(np.rint(mixed), 0, 255), dtype=np.uint8)

    def rendering(
        self,
        ctx: ComparisonRenderContext,
        camera: Camera,
        fps: float,
        *,
        cancelled: Callable[[], bool] = lambda: False,
        progress: Callable[[int, int], None] = lambda done, total: None,
    ) -> ClipRendering:
        """Return the same pose/drawing/reference/crop/clock recipe for both paths."""
        registration = ctx.registration

        def overlay(frame: Image, index: int) -> Image:
            return self.overlay(
                frame, registration.scene_time(index / fps), ctx, camera
            )

        def timestamp(index: int, source_fps: float) -> str:
            time = registration.scene_time(index / source_fps)
            return f"{ctx.view} ref:{ctx.asset.title} f{index} t={time:.3f}s"

        return ClipRendering(
            crop=ctx.crop,
            drawings=ctx.drawings,
            strict=True,
            overlay=overlay,
            timestamp=timestamp,
            cancelled=cancelled,
            progress=progress,
        )

    def image(
        self,
        reader: VideoReader,
        index: int,
        ctx: ComparisonRenderContext,
        camera: Camera,
    ) -> Image:
        """Render a source frame for preview, including the exact export recipe."""
        rendering = self.rendering(ctx, camera, reader.fps or 30.0)
        rendering.size(reader.width, reader.height)
        image = _rendered(
            reader,
            index,
            ctx.track,
            min_confidence=0.5,
            label=ctx.view,
            rendering=rendering,
        )
        if image is None:
            raise ValueError(f"Could not decode source frame {index}")
        return image

    def close(self) -> None:
        if self.reader is not None:
            self.reader.close()
            self.reader = None

    def __enter__(self) -> ComparisonRenderer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
