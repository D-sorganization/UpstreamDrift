"""Encode a hand-load comparison grid to an H.264 MP4.

ffmpeg is required. When it is missing or the encode does not produce a
readable MP4, the call raises :class:`VideoEncodeError` and leaves no file.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.hand_load_video.compare import GridSpec, Provenance, SyncedFrame, compose


class VideoEncodeError(RuntimeError):
    """ffmpeg could not write the comparison MP4."""


@dataclass(frozen=True)
class EncodeSettings:
    """Playback rate and H.264 rate factor for one comparison file.

    Preconditions: ``fps`` is a finite number above zero, and ``crf`` is an
    integer from 0 through 51.
    """

    fps: float
    crf: int = 18

    def __post_init__(self) -> None:
        if isinstance(self.fps, bool) or not isinstance(self.fps, (int, float)):
            raise TypeError("fps must be a real number")
        if not math.isfinite(self.fps) or self.fps <= 0:
            raise ValueError("fps must be finite and positive")
        if type(self.crf) is not int or not 0 <= self.crf <= 51:
            raise ValueError("crf must be an integer from 0 through 51")
        object.__setattr__(self, "fps", float(self.fps))


def ffmpeg_executable() -> str:
    """Return the ffmpeg binary imageio will invoke.

    Raises:
        VideoEncodeError: ffmpeg is not installed. No file is written.
    """
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise VideoEncodeError(
            "ffmpeg is unavailable: install imageio[ffmpeg]"
        ) from exc
    try:
        binary = imageio_ffmpeg.get_ffmpeg_exe()
    except RuntimeError as exc:
        raise VideoEncodeError(f"ffmpeg is unavailable: {exc}") from exc
    if not binary or not Path(binary).is_file():
        raise VideoEncodeError("ffmpeg is unavailable: no ffmpeg executable")
    return str(binary)


def write_comparison_mp4(
    frames: Sequence[SyncedFrame],
    provenance: Provenance,
    destination: Path,
    settings: EncodeSettings,
    spec: GridSpec | None = None,
) -> Path:
    """Encode ``frames`` to an H.264 MP4 at ``destination``.

    Each stored frame is the composed grid, so the evidence badge is on every
    frame and a missing panel stays ``UNAVAILABLE``.

    Preconditions: ``frames`` is a non-empty sequence of synced frames,
    ``destination`` is an ``.mp4`` path in an existing directory, and the
    composed frame has even width and height. ffmpeg must be installed.

    Postconditions: ``destination`` is a non-empty MP4 that imageio can read
    back with one frame per input frame.

    Raises:
        TypeError: a frame, the provenance, the destination, the settings, or
            the spec has the wrong type.
        ValueError: there are no frames, the path is not ``.mp4``, the parent
            directory is missing, or the frame size is odd.
        VideoEncodeError: ffmpeg is unavailable or does not produce a readable
            MP4. ``destination`` is not left behind.
    """
    layout = _layout(spec)
    _require_request(frames, provenance, destination, settings, layout)
    ffmpeg_executable()
    try:
        _encode(frames, provenance, destination, settings, layout)
        _require_playable(destination, len(frames))
    except VideoEncodeError:
        _discard(destination)
        raise
    except (OSError, RuntimeError, ValueError, ImportError) as exc:
        _discard(destination)
        raise VideoEncodeError(
            f"ffmpeg failed to encode {destination.name}: {exc}"
        ) from exc
    return destination


def _layout(spec: GridSpec | None) -> GridSpec:
    if spec is None:
        return GridSpec()
    if not isinstance(spec, GridSpec):
        raise TypeError("spec must be GridSpec")
    return spec


def _require_request(
    frames: Sequence[SyncedFrame],
    provenance: Provenance,
    destination: Path,
    settings: EncodeSettings,
    layout: GridSpec,
) -> None:
    if not isinstance(settings, EncodeSettings):
        raise TypeError("settings must be EncodeSettings")
    if not isinstance(provenance, Provenance):
        raise TypeError("provenance must be Provenance")
    if not isinstance(destination, Path):
        raise TypeError("destination must be a Path")
    if isinstance(frames, (str, bytes)) or not isinstance(frames, Sequence):
        raise TypeError("frames must be a sequence of SyncedFrame values")
    if len(frames) < 1:
        raise ValueError("comparison video needs at least one frame")
    if any(not isinstance(frame, SyncedFrame) for frame in frames):
        raise TypeError("frames must be SyncedFrame values")
    if destination.suffix.lower() != ".mp4":
        raise ValueError("destination must be an .mp4 path")
    if not destination.parent.is_dir():
        raise ValueError(f"destination directory does not exist: {destination.parent}")
    if layout.width % 2 or layout.height % 2:
        raise ValueError("comparison frame width and height must be even")


def _open_writer(path: Path, settings: EncodeSettings) -> Any:
    import imageio.v2 as imageio

    return imageio.get_writer(
        str(path),
        fps=settings.fps,
        codec="libx264",
        quality=None,
        pixelformat="yuv420p",
        macro_block_size=1,
        ffmpeg_log_level="error",
        output_params=["-crf", str(settings.crf), "-preset", "medium"],
    )


def _encode(
    frames: Sequence[SyncedFrame],
    provenance: Provenance,
    destination: Path,
    settings: EncodeSettings,
    layout: GridSpec,
) -> None:
    writer = _open_writer(destination, settings)
    try:
        for frame in frames:
            image, _stamp = compose(frame, provenance, layout)
            writer.append_data(image)
    finally:
        writer.close()


def _require_playable(path: Path, n_frames: int) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise VideoEncodeError(f"ffmpeg produced no video at {path.name}")
    with path.open("rb") as handle:
        header = handle.read(32)
    if b"ftyp" not in header:
        raise VideoEncodeError(
            f"ffmpeg produced a file that is not an MP4: {path.name}"
        )
    import imageio.v2 as imageio

    reader = imageio.get_reader(str(path))
    try:
        count = int(reader.count_frames())
    finally:
        reader.close()
    if count != n_frames:
        raise VideoEncodeError(f"ffmpeg wrote {count} frames, expected {n_frames}")


def _discard(path: Path) -> None:
    if path.is_file():
        path.unlink()
