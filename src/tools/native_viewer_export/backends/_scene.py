"""Geometry helpers shared by the native viewer backends."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


def z_axis_frame(
    start: Sequence[float], end: Sequence[float]
) -> tuple[Array, Array, float]:
    """Rotation, centre and length of a capsule whose +Z axis runs start->end."""
    a, b = np.asarray(start, float), np.asarray(end, float)
    d = b - a
    length = float(np.linalg.norm(d))
    if length < 1e-12:
        raise ValueError("capsule endpoints coincide")
    z = d / length
    x = np.cross([0.0, 0.0, 1.0], z)
    x = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(x) < 1e-9 else x
    x = x / np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z]), (a + b) / 2.0, length


def y_axis_rotation(direction: Sequence[float]) -> Array:
    """Rotation taking +Y onto ``direction`` (OpenSim cylinders run along Y)."""
    d = np.asarray(direction, float)
    d = d / np.linalg.norm(d)
    y = np.array([0.0, 1.0, 0.0])
    v = np.cross(y, d)
    c = float(y @ d)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    k = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + k + k @ k / (1.0 + c)


def fit_to_size(frame: NDArray[np.uint8], width: int, height: int) -> NDArray[np.uint8]:
    """Centre-crop to the target aspect ratio, then resize to ``(height, width)``."""
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    import cv2

    h, w = frame.shape[:2]
    want = width / height
    if w / h > want:
        new_w = int(round(h * want))
        x0 = (w - new_w) // 2
        frame = frame[:, x0 : x0 + new_w]
    else:
        new_h = int(round(w / want))
        y0 = (h - new_h) // 2
        frame = frame[y0 : y0 + new_h]
    if frame.shape[:2] == (height, width):
        return np.ascontiguousarray(frame)
    return np.ascontiguousarray(
        cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    )
