"""The same source-resolution renderer serves preview and media exports."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .drawing import DrawingLayer


def render_layer(
    image: npt.NDArray[np.uint8], layer: DrawingLayer, frame: int
) -> npt.NDArray[np.uint8]:
    """Draw on a copy before any crop or resize, preserving source coordinates."""
    import cv2

    if image.shape != (layer.height, layer.width, 3):
        raise ValueError("Drawing layer does not match the source image dimensions")
    result = image.copy()
    for shape in layer.shapes:
        if not shape.at(frame):
            continue
        start = tuple(round(value) for value in shape.start)
        end = tuple(round(value) for value in shape.end)
        colour = tuple(int(shape.colour[i : i + 2], 16) for i in (5, 3, 1))
        if shape.kind == "line":
            cv2.line(result, start, end, colour, shape.stroke, cv2.LINE_AA)
        elif shape.kind == "arrow":
            cv2.arrowedLine(
                result, start, end, colour, shape.stroke, cv2.LINE_AA, tipLength=0.15
            )
        elif shape.kind == "rectangle":
            cv2.rectangle(result, start, end, colour, shape.stroke, cv2.LINE_AA)
        else:
            center = tuple(
                round((a + b) / 2) for a, b in zip(shape.start, shape.end, strict=True)
            )
            axes = tuple(
                max(1, round(abs(a - b) / 2))
                for a, b in zip(shape.start, shape.end, strict=True)
            )
            cv2.ellipse(
                result, center, axes, 0, 0, 360, colour, shape.stroke, cv2.LINE_AA
            )
    return result
