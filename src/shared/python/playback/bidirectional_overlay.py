"""Bidirectional video/model overlay compositor (Epic #15660, #15669).

Extends #11356 translucent shape overlay semantics with bidirectional modes:
- MODEL_ON_VIDEO: 3D model geometry overlaid on source video.
- VIDEO_ON_MODEL: 2D source video overlaid on virtual 3D scene/model render.
- SPLIT_SCREEN: Side-by-side or stacked split comparison view.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Literal

import numpy as np

from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions


class OverlayMode(str, Enum):
    """Bidirectional overlay composition mode."""

    MODEL_ON_VIDEO = "model_on_video"
    VIDEO_ON_MODEL = "video_on_model"
    SPLIT_SCREEN = "split_screen"


@dataclass(frozen=True)
class BidirectionalOverlayConfig:
    """Settings controlling bidirectional overlay composition."""

    mode: OverlayMode = OverlayMode.MODEL_ON_VIDEO
    model_opacity: float = 1.0
    video_opacity: float = 0.5
    split_ratio: float = 0.5
    split_orientation: Literal["vertical", "horizontal"] = "vertical"

    def __post_init__(self) -> None:
        for val, name in (
            (self.model_opacity, "model_opacity"),
            (self.video_opacity, "video_opacity"),
            (self.split_ratio, "split_ratio"),
        ):
            if not (0.0 <= val <= 1.0 and math.isfinite(val)):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if self.split_orientation not in ("vertical", "horizontal"):
            raise ValueError("split_orientation must be 'vertical' or 'horizontal'")

    @classmethod
    def from_shape_options(
        cls, options: ShapeOverlayOptions
    ) -> BidirectionalOverlayConfig:
        """Construct MODEL_ON_VIDEO config adapting #11356 ShapeOverlayOptions."""
        return cls(
            mode=OverlayMode.MODEL_ON_VIDEO, model_opacity=float(options.opacity)
        )


def composite_bidirectional_overlay(
    video_frame: np.ndarray,
    model_render: np.ndarray,
    config: BidirectionalOverlayConfig,
) -> np.ndarray:
    """Composite video and model renders bidirectionally according to config."""
    video = np.asarray(video_frame, dtype=np.uint8)
    model = np.asarray(model_render, dtype=np.uint8)
    if (
        video.ndim != 3
        or video.shape[2] != 3
        or model.ndim != 3
        or model.shape[2] not in (3, 4)
    ):
        raise ValueError("Invalid video or model dimensions")
    if video.shape[:2] != model.shape[:2]:
        raise ValueError(
            f"Image dimensions must match: video {video.shape[:2]} vs model {model.shape[:2]}"
        )

    h, w, _ = video.shape
    if config.mode == OverlayMode.MODEL_ON_VIDEO:
        if config.model_opacity <= 0.0:
            return video.copy()
        alpha = (
            (model[..., 3].astype(float) / 255.0)
            if model.shape[2] == 4
            else np.any(model > 0, axis=2).astype(float)
        ) * config.model_opacity
        blended = (
            video.astype(float) * (1.0 - alpha[..., None])
            + model[..., :3].astype(float) * alpha[..., None]
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    if config.mode == OverlayMode.VIDEO_ON_MODEL:
        model_rgb = model[..., :3]
        if config.video_opacity <= 0.0:
            return model_rgb.copy()
        if config.video_opacity >= 1.0:
            return video.copy()
        a = config.video_opacity
        return np.clip(
            model_rgb.astype(float) * (1.0 - a) + video.astype(float) * a, 0, 255
        ).astype(np.uint8)

    if config.mode == OverlayMode.SPLIT_SCREEN:
        out = np.zeros_like(video)
        if config.split_orientation == "vertical":
            sx = int(round(w * config.split_ratio))
            out[:, :sx], out[:, sx:] = video[:, :sx], model[..., :3][:, sx:]
        else:
            sy = int(round(h * config.split_ratio))
            out[:sy, :], out[sy:, :] = video[:sy, :], model[..., :3][sy:, :]
        return out

    raise ValueError(f"Unsupported overlay mode: {config.mode}")
