"""Tests for engine-free GIF frame extraction (#11987).

Builds a real 3-frame GIF with Pillow (distinct colours, explicit
durations, and one frame with a missing/zero duration) and exercises the
contract documented in ``gif_frames.py``: frame count, per-frame
durations (with the documented default fallback), PNG-encoded frame
bytes, and every DbC error case.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from src.tools.matched_swing_browser.gif_frames import (
    DEFAULT_FRAME_DURATION_MS,
    gif_frame_info,
    gif_frame_png,
)

pytestmark = pytest.mark.unit

_FRAME_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]


def _make_gif(path: Path, durations_ms: list[int]) -> None:
    """Write a multi-frame GIF with distinct per-frame colours and durations."""
    frames = [
        Image.new("RGB", (4, 3), _FRAME_COLORS[i]) for i in range(len(durations_ms))
    ]
    frames[0].save(
        path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=durations_ms,
        loop=0,
    )


@pytest.fixture
def gif_path(tmp_path: Path) -> Path:
    """A 3-frame GIF: durations 40ms, 0ms (missing), 120ms."""
    path = tmp_path / "swing.gif"
    _make_gif(path, [40, 0, 120])
    return path


def test_gif_frame_info_frame_count_and_size(gif_path: Path) -> None:
    info = gif_frame_info(gif_path)
    assert info["frame_count"] == 3
    assert info["width"] == 4
    assert info["height"] == 3


def test_gif_frame_info_missing_duration_defaults_to_100ms(gif_path: Path) -> None:
    info = gif_frame_info(gif_path)
    assert info["durations_ms"] == [40, DEFAULT_FRAME_DURATION_MS, 120]


def test_gif_frame_info_postconditions(gif_path: Path) -> None:
    info = gif_frame_info(gif_path)
    assert info["frame_count"] >= 1
    assert len(info["durations_ms"]) == info["frame_count"]
    assert all(duration > 0 for duration in info["durations_ms"])


def test_gif_frame_png_has_png_signature(gif_path: Path) -> None:
    png_bytes = gif_frame_png(gif_path, 0)
    assert png_bytes.startswith(b"\x89PNG\r\n\x1a\n")


def test_gif_frame_png_frame_0_and_frame_2_differ(gif_path: Path) -> None:
    frame0 = gif_frame_png(gif_path, 0)
    frame2 = gif_frame_png(gif_path, 2)
    assert frame0 != frame2


def test_gif_frame_info_rejects_non_path() -> None:
    with pytest.raises(TypeError):
        gif_frame_info("not-a-path")  # type: ignore[arg-type]


def test_gif_frame_png_rejects_non_path() -> None:
    with pytest.raises(TypeError):
        gif_frame_png("not-a-path", 0)  # type: ignore[arg-type]


def test_gif_frame_info_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError) as excinfo:
        gif_frame_info(tmp_path / "missing.gif")
    # The API returns this message to clients: no absolute path.
    assert str(tmp_path) not in str(excinfo.value)
    assert "missing.gif" in str(excinfo.value)


def test_gif_frame_png_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        gif_frame_png(tmp_path / "missing.gif", 0)


def test_gif_frame_info_rejects_non_gif(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_gif.gif"
    Image.new("RGB", (2, 2)).save(bad_path, format="PNG")
    with pytest.raises(ValueError):
        gif_frame_info(bad_path)


def test_gif_frame_png_rejects_non_gif(tmp_path: Path) -> None:
    bad_path = tmp_path / "not_a_gif.gif"
    Image.new("RGB", (2, 2)).save(bad_path, format="PNG")
    with pytest.raises(ValueError):
        gif_frame_png(bad_path, 0)


def test_gif_frame_png_rejects_negative_index(gif_path: Path) -> None:
    with pytest.raises(IndexError):
        gif_frame_png(gif_path, -1)


def test_gif_frame_png_rejects_index_at_frame_count(gif_path: Path) -> None:
    with pytest.raises(IndexError):
        gif_frame_png(gif_path, 3)
