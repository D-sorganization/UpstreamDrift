"""Pure frame-content checks for the MuJoCo viewport (UD #12884).

A rendered scene always shows sky, floor and model, so it has both a non-trivial
brightness and many distinct colours. A frame that is black (or one flat colour
with at most a small 2D overlay drawn on top) means the GL renderer produced
nothing. These helpers are pure ``numpy`` so the check is unit-testable without
an OpenGL context, and the sim widget uses the same predicate as a runtime
post-condition on its first frame after a model load.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "BLANK_MAX_DISTINCT_COLORS",
    "BLANK_MAX_MEAN_LUMINANCE",
    "FrameStats",
    "frame_stats",
    "is_blank_frame",
]

#: A frame at or below this mean luminance (0-255) is treated as black.
BLANK_MAX_MEAN_LUMINANCE: float = 5.0
#: A frame with at most this many distinct colours is featureless.
BLANK_MAX_DISTINCT_COLORS: int = 8
# Rec. 601 luma weights.
_LUMA = np.array([0.299, 0.587, 0.114])
# Sample at most this many pixels when counting colours (keeps it O(1)-ish).
_MAX_COLOR_SAMPLES = 65536


@dataclass(frozen=True)
class FrameStats:
    """Summary of a frame's content."""

    mean_luminance: float
    distinct_colors: int


def _validated_rgb(rgb: np.ndarray) -> np.ndarray:
    """Return the RGB channels of an ``HxWx3`` or ``HxWx4`` ``uint8`` frame.

    Raises:
        TypeError: if ``rgb`` is not a ``uint8`` ``numpy`` array.
        ValueError: if ``rgb`` is not a non-empty ``HxWx3``/``HxWx4`` image.
    """
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise TypeError("frame must be a uint8 numpy array")
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.size == 0:
        raise ValueError(f"frame must be a non-empty HxWx3/4 image, got {rgb.shape}")
    return rgb[..., :3]


def frame_stats(rgb: np.ndarray) -> FrameStats:
    """Return the mean luminance and (sampled) distinct-colour count of a frame.

    Postcondition: ``0 <= mean_luminance <= 255`` and ``distinct_colors >= 1``.
    """
    channels = _validated_rgb(rgb)
    mean_luminance = float(channels.reshape(-1, 3).mean(axis=0) @ _LUMA)
    pixels = channels.reshape(-1, 3)
    step = max(1, pixels.shape[0] // _MAX_COLOR_SAMPLES)
    sample = pixels[::step].astype(np.uint32)
    packed = (sample[:, 0] << 16) | (sample[:, 1] << 8) | sample[:, 2]
    return FrameStats(mean_luminance, int(np.unique(packed).size))


def is_blank_frame(
    rgb: np.ndarray,
    *,
    max_mean_luminance: float = BLANK_MAX_MEAN_LUMINANCE,
    max_distinct_colors: int = BLANK_MAX_DISTINCT_COLORS,
) -> bool:
    """Return ``True`` when ``rgb`` cannot be a rendered 3D scene.

    A frame is blank when it is (near) black *or* nearly a single colour.
    """
    stats = frame_stats(rgb)
    return (
        stats.mean_luminance <= max_mean_luminance
        or stats.distinct_colors <= max_distinct_colors
    )
