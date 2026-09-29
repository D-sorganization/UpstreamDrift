"""What counts as a video file for the pose-estimation GUIs and tools.

One definition shared by the MediaPipe and OpenPose GUIs (issue #9611): the
capture rig writes MJPEG in Matroska (``.mkv``), OpenCV decodes it, and a
file dialog that silently hides it makes the rig's recordings unreachable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

VIDEO_SUFFIXES: tuple[str, ...] = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")


def video_dialog_filter() -> str:
    """Qt file-dialog filter listing every supported suffix, plus All Files."""
    patterns = " ".join(f"*{suffix}" for suffix in VIDEO_SUFFIXES)
    return f"Video Files ({patterns});;All Files (*)"


def is_video_file(path: Path | str) -> bool:
    """True when the suffix is one OpenCV-backed tools accept (case-insensitive)."""
    return Path(path).suffix.lower() in VIDEO_SUFFIXES


class VideoLoaderWidget(Protocol):
    """What a pose GUI must offer for :func:`load_video_into`."""

    _video_path: str

    def log(self, message: str) -> None: ...


def load_video_into(widget: Any) -> str | None:
    """Ask for a video through Qt's dialog and record it on the widget.

    Shared by the MediaPipe and OpenPose GUIs (one implementation, DRY gate).
    Returns the chosen path, or None when the dialog was cancelled. The
    widget needs ``_video_path``, ``lbl_file``, ``btn_run`` and ``log``.
    """
    from PyQt6.QtWidgets import QFileDialog

    file_name, _ = QFileDialog.getOpenFileName(
        widget, "Select Video", "", video_dialog_filter()
    )
    if not file_name:
        return None
    widget._video_path = file_name
    widget.lbl_file.setText(file_name)
    widget.btn_run.setEnabled(True)
    widget.log(f"Loaded video: {file_name}")
    return file_name
