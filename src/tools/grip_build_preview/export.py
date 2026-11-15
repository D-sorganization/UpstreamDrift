"""Write preview frames to GIF/MP4/still strips (imageio + native viewer writer)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from src.tools.native_viewer_export.compositor import compose_grid
from src.tools.native_viewer_export.core import imageio_writer

Image8 = NDArray[np.uint8]


def write_gif(path: Path, frames: tuple[Image8, ...], fps: int) -> None:
    if len(frames) < 1:
        raise ValueError("frames must not be empty")
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), list(frames), fps=fps)


def write_mp4(path: Path, frames: tuple[Image8, ...], fps: int) -> None:
    if len(frames) < 1:
        raise ValueError("frames must not be empty")
    writer = imageio_writer(path, fps=fps)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def write_still_strip(path: Path, frames: tuple[Image8, ...], cols: int = 4) -> None:
    if len(frames) < 1:
        raise ValueError("frames must not be empty")
    import imageio.v2 as imageio

    grid = compose_grid(list(frames), cols=min(cols, len(frames)))
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(str(path), grid)
