"""Plot pointwise JaxSim ZTCF parameter sensitivity for a sample trajectory."""

from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.engines.physics_engines.jaxsim.parameter_gradients import (
    DEFAULT_PARAMETER_VECTOR,
    evaluate_ztcf_parameter_sensitivity_along_trajectory,
)


def sample_measured_trajectory(samples: int = 80) -> tuple[NDArray[np.float64], ...]:
    """Return a deterministic measured-state sample for the sensitivity plot."""

    if samples < 2:
        raise ValueError("samples must be at least 2")
    t = np.linspace(0.0, 1.0, samples, dtype=np.float64)
    q = np.column_stack(
        [
            0.35 * np.sin(2.0 * np.pi * t),
            -0.25 * np.cos(2.0 * np.pi * t),
        ]
    )
    v = np.column_stack(
        [
            0.35 * 2.0 * np.pi * np.cos(2.0 * np.pi * t),
            0.25 * 2.0 * np.pi * np.sin(2.0 * np.pi * t),
        ]
    )
    return t, q, v


def write_parameter_sensitivity_plot(
    output_path: Path,
    *,
    samples: int = 80,
) -> Path:
    """Write the sample ZTCF parameter-sensitivity figure."""

    t, q, v = sample_measured_trajectory(samples)
    sensitivity = evaluate_ztcf_parameter_sensitivity_along_trajectory(
        DEFAULT_PARAMETER_VECTOR,
        q,
        v,
    )
    magnitude = np.linalg.norm(sensitivity, axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_png_plot(output_path, t, magnitude)
    return output_path


def _write_png_plot(
    output_path: Path,
    t: NDArray[np.float64],
    magnitude: NDArray[np.float64],
) -> None:
    width, height = 960, 540
    left, right, top, bottom = 84, 28, 42, 78
    image: NDArray[np.uint8] = np.full((height, width, 3), 255, dtype=np.uint8)
    colors = np.array(
        [
            [31, 119, 180],
            [214, 39, 40],
            [44, 160, 44],
            [148, 103, 189],
            [255, 127, 14],
        ],
        dtype=np.uint8,
    )

    x_min, x_max = float(np.min(t)), float(np.max(t))
    y_min = 0.0
    y_max = float(np.max(magnitude)) * 1.08
    if y_max <= 0.0:
        y_max = 1.0

    _draw_line(
        image, left, height - bottom, width - right, height - bottom, (20, 20, 20)
    )
    _draw_line(image, left, top, left, height - bottom, (20, 20, 20))
    for fraction in np.linspace(0.0, 1.0, 6):
        x = int(left + fraction * (width - left - right))
        _draw_line(image, x, top, x, height - bottom, (230, 230, 230))
    for fraction in np.linspace(0.0, 1.0, 5):
        y = int(height - bottom - fraction * (height - top - bottom))
        _draw_line(image, left, y, width - right, y, (230, 230, 230))

    for column in range(magnitude.shape[1]):
        points = [
            (
                _scale(float(t[row]), x_min, x_max, left, width - right),
                _scale(
                    float(magnitude[row, column]), y_min, y_max, height - bottom, top
                ),
            )
            for row in range(len(t))
        ]
        palette_color = colors[column % len(colors)]
        color = (
            int(palette_color[0]),
            int(palette_color[1]),
            int(palette_color[2]),
        )
        for start, end in zip(points, points[1:], strict=False):
            _draw_line(image, start[0], start[1], end[0], end[1], color)

    _write_png(output_path, image)


def _scale(
    value: float, src_min: float, src_max: float, dst_min: int, dst_max: int
) -> int:
    if src_max == src_min:
        return dst_min
    fraction = (value - src_min) / (src_max - src_min)
    return int(round(dst_min + fraction * (dst_max - dst_min)))


def _draw_line(
    image: NDArray[np.uint8],
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= y < image.shape[0] and 0 <= x < image.shape[1]:
            image[y, x] = color
            if y + 1 < image.shape[0]:
                image[y + 1, x] = color
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _write_png(output_path: Path, image: NDArray[np.uint8]) -> None:
    height, width, channels = image.shape
    if channels != 3:
        raise ValueError("image must be RGB")
    raw = b"".join(b"\x00" + image[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(raw, level=9))
    png += _png_chunk(b"IEND", b"")
    output_path.write_bytes(png)


def _png_chunk(chunk_type: bytes, data: bytes) -> bytes:
    return (
        struct.pack(">I", len(data))
        + chunk_type
        + data
        + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot pointwise JaxSim ZTCF parameter sensitivity."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/jaxsim_parameter_sensitivity.png"),
        help="PNG path to write.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=80,
        help="Number of measured-state samples.",
    )
    args = parser.parse_args()
    output_path = write_parameter_sensitivity_plot(args.output, samples=args.samples)
    print(output_path)


if __name__ == "__main__":
    main()
