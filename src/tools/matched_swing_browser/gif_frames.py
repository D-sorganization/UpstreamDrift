"""Engine-free GIF frame extraction for the Matched Swing Browser (#11987).

Provides frame-exact access to a GIF animation (frame count, per-frame
durations, and a single decoded frame as PNG bytes) so the web client can
implement a browser-accurate Play/Pause/Restart control, mirroring the
desktop ``QMovie``-based playback in ``gui.py``. A browser cannot pause an
``<img>`` GIF, so the API streams individual decoded frames instead.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageSequence

__all__ = [
    "DEFAULT_FRAME_DURATION_MS",
    "gif_frame_info",
    "gif_frame_png",
]

# GIF frames may omit a duration, or record a zero/negative one. Browsers
# play such frames at 100 ms, so the web player uses the same fallback.
DEFAULT_FRAME_DURATION_MS = 100


def _validate_path(path: Path) -> None:
    """Raise the documented contract errors for a GIF path argument.

    Messages name the file only, never its absolute path, because the API
    returns them to clients.
    """
    if not isinstance(path, Path):
        raise TypeError(f"path must be a Path, got {type(path).__name__}")
    if not path.is_file():
        raise FileNotFoundError(f"GIF file not found: {path.name}")


def gif_frame_info(path: Path) -> dict[str, object]:
    """Return frame count, per-frame durations, and size for a GIF.

    Preconditions:
        ``path`` is a ``Path`` to an existing, readable GIF file.

    Postconditions:
        ``frame_count >= 1``; ``len(durations_ms) == frame_count``; every
        entry in ``durations_ms`` is a positive integer (missing or
        non-positive source durations are replaced with
        :data:`DEFAULT_FRAME_DURATION_MS`).

    Raises:
        TypeError: ``path`` is not a ``Path``.
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is not a readable GIF image.

    Returns:
        A dict with keys ``frame_count``, ``durations_ms``, ``width``, and
        ``height``.
    """
    _validate_path(path)
    try:
        with Image.open(path) as img:
            if img.format != "GIF":
                raise ValueError(f"Not a GIF image: {path.name} (format={img.format})")
            width, height = img.size
            durations_ms: list[int] = []
            for frame in ImageSequence.Iterator(img):
                duration = frame.info.get("duration")
                if not isinstance(duration, (int, float)) or duration <= 0:
                    duration = DEFAULT_FRAME_DURATION_MS
                durations_ms.append(int(duration))
    except OSError as exc:
        raise ValueError(f"Not a readable GIF image: {path.name}") from exc

    if not durations_ms:
        raise ValueError(f"GIF has no frames: {path.name}")

    return {
        "frame_count": len(durations_ms),
        "durations_ms": durations_ms,
        "width": width,
        "height": height,
    }


def gif_frame_png(path: Path, index: int) -> bytes:
    """Return a single GIF frame, converted to RGBA and encoded as PNG.

    Preconditions:
        ``path`` is a ``Path`` to an existing, readable GIF file; ``index``
        is a non-negative integer less than the GIF's frame count.

    Postconditions:
        The returned bytes begin with the PNG signature.

    Raises:
        TypeError: ``path`` is not a ``Path``.
        FileNotFoundError: ``path`` does not exist.
        ValueError: ``path`` is not a readable GIF image.
        IndexError: ``index`` is negative or out of range.

    Returns:
        PNG-encoded bytes of the requested frame.
    """
    _validate_path(path)
    try:
        with Image.open(path) as img:
            if img.format != "GIF":
                raise ValueError(f"Not a GIF image: {path.name} (format={img.format})")
            frame_count = getattr(img, "n_frames", 1)
            if index < 0 or index >= frame_count:
                raise IndexError(
                    f"Frame index {index} out of range (0..{frame_count - 1})"
                )
            img.seek(index)
            frame_rgba = img.convert("RGBA")
            buffer = io.BytesIO()
            frame_rgba.save(buffer, format="PNG")
            return buffer.getvalue()
    except OSError as exc:
        raise ValueError(f"Not a readable GIF image: {path.name}") from exc
