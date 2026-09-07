"""What counts as a video file for the pose-estimation GUIs and tools.

One definition shared by the MediaPipe and OpenPose GUIs (issue #9611): the
capture rig writes MJPEG in Matroska (``.mkv``), OpenCV decodes it, and a
file dialog that silently hides it makes the rig's recordings unreachable.
"""

from __future__ import annotations

from pathlib import Path

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def video_dialog_filter() -> str:
    """Qt file-dialog filter listing every supported suffix, plus All Files."""
    patterns = " ".join(f"*{suffix}" for suffix in VIDEO_SUFFIXES)
    return f"Video Files ({patterns});;All Files (*)"


def is_video_file(path: Path | str) -> bool:
    """True when the suffix is one OpenCV-backed tools accept (case-insensitive)."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES
