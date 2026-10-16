"""H.264-first MP4 writer shared by the app video exports (#12947, #12878)."""

from __future__ import annotations

import logging
import re
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="MP4 export needs opencv")

from src.shared.python.video_codec import mp4_writer as mod  # noqa: E402
from src.shared.python.video_codec.mp4_writer import Mp4Writer  # noqa: E402

SIZE = (64, 48)


def _frames(count: int = 6) -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    return [
        rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)
        for _ in range(count)
    ]


def _write(path: Path) -> Mp4Writer:
    writer = Mp4Writer(path, 10, SIZE)
    for frame in _frames():
        writer.write(frame)
    writer.release()
    return writer


def _codec_of(path: Path) -> str:
    """Codec name via ffmpeg's stream banner (ffprobe is not always installed)."""
    exe = mod.find_ffmpeg()
    assert exe is not None
    proc = subprocess.run(  # noqa: S603
        [exe, "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"Video: (\w+)", proc.stderr)
    assert match, proc.stderr
    return match.group(1)


class _FakeWriter:
    def __init__(self, opened: bool) -> None:
        self._opened = opened
        self.frames = 0
        self.released = False

    def isOpened(self) -> bool:  # noqa: N802
        return self._opened

    def write(self, frame: np.ndarray) -> None:
        self.frames += 1

    def release(self) -> None:
        self.released = True


@pytest.mark.unit
def test_prefers_avc1_and_skips_transcode_when_opencv_has_h264(tmp_path: Path) -> None:
    with (
        patch.object(
            mod, "_open_cv_writer", side_effect=lambda p, c, f, s: _FakeWriter(True)
        ) as opener,
        patch.object(mod, "_transcode_to_h264") as transcode,
    ):
        writer = Mp4Writer(tmp_path / "a.mp4", 30, SIZE)
        writer.release()
    assert writer.codec == "avc1"
    assert opener.call_args_list[0].args[1] == "avc1"
    transcode.assert_not_called()


@pytest.mark.unit
def test_falls_back_to_mp4v_when_avc1_unavailable(tmp_path: Path) -> None:
    attempts: list[str] = []

    def opener(
        path: Path, codec: str, fps: float, size: tuple[int, int]
    ) -> _FakeWriter:
        attempts.append(codec)
        return _FakeWriter(codec == "mp4v")

    with (
        patch.object(mod, "_open_cv_writer", side_effect=opener),
        patch.object(mod, "_transcode_to_h264", return_value=False),
    ):
        writer = Mp4Writer(tmp_path / "a.mp4", 30, SIZE)
        assert writer.isOpened()
        writer.release()
    assert attempts == ["avc1", "mp4v"]
    assert writer.codec == "mp4v"


@pytest.mark.unit
def test_not_opened_when_no_codec_opens(tmp_path: Path) -> None:
    with patch.object(mod, "_open_cv_writer", return_value=_FakeWriter(False)):
        writer = Mp4Writer(tmp_path / "a.mp4", 30, SIZE)
    assert not writer.isOpened()
    assert writer.codec == ""
    writer.release()  # idempotent, no error


@pytest.mark.unit
@pytest.mark.parametrize(("fps", "size"), [(0, SIZE), (30, (0, 48)), (30, (64,))])
def test_rejects_invalid_arguments(
    tmp_path: Path, fps: int, size: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        Mp4Writer(tmp_path / "a.mp4", fps, size)  # type: ignore[arg-type]


@pytest.mark.unit
def test_release_is_idempotent_and_transcodes_once(tmp_path: Path) -> None:
    calls: list[Path] = []

    def opener(
        path: Path, codec: str, fps: float, size: tuple[int, int]
    ) -> _FakeWriter:
        return _FakeWriter(codec == "mp4v")

    def transcode(path: Path) -> bool:
        calls.append(path)
        return True

    with (
        patch.object(mod, "_open_cv_writer", side_effect=opener),
        patch.object(mod, "_transcode_to_h264", side_effect=transcode),
    ):
        writer = Mp4Writer(tmp_path / "a.mp4", 30, SIZE)
        writer.release()
        writer.release()
    assert len(calls) == 1
    assert writer.codec == "avc1"  # transcode succeeded: final file is H.264


@pytest.mark.unit
def test_keeps_mp4v_and_warns_when_no_ffmpeg(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    out = tmp_path / "a.mp4"
    with (
        patch.object(mod, "find_ffmpeg", return_value=None),
        patch.object(
            mod,
            "_open_cv_writer",
            side_effect=lambda p, c, f, s: _FakeWriter(c == "mp4v"),
        ),
        caplog.at_level(logging.WARNING),
    ):
        writer = Mp4Writer(out, 30, SIZE)
        writer.release()
    assert writer.codec == "mp4v"
    assert any("mp4v" in r.message and "ffmpeg" in r.message for r in caplog.records)


@pytest.mark.unit
def test_find_ffmpeg_prefers_path_then_imageio(tmp_path: Path) -> None:
    with patch.object(mod.shutil, "which", return_value="/usr/bin/ffmpeg"):
        assert mod.find_ffmpeg() == "/usr/bin/ffmpeg"
    with (
        patch.object(mod.shutil, "which", return_value=None),
        patch.dict("sys.modules", {"imageio_ffmpeg": None}),
    ):
        assert mod.find_ffmpeg() is None


@pytest.mark.unit
def test_transcode_failure_keeps_original_file(tmp_path: Path) -> None:
    src = tmp_path / "a.mp4"
    src.write_bytes(b"orig")
    err = subprocess.CalledProcessError(1, "ffmpeg", stderr="boom")
    with (
        patch.object(mod, "find_ffmpeg", return_value="ffmpeg"),
        patch.object(mod.subprocess, "run", side_effect=err),
    ):
        assert mod._transcode_to_h264(src) is False
    assert src.read_bytes() == b"orig"
    assert not list(tmp_path.glob("*.h264tmp*"))


@pytest.mark.unit
def test_real_export_is_h264_when_any_encoder_exists(tmp_path: Path) -> None:
    if mod.find_ffmpeg() is None:
        pytest.skip("no ffmpeg on PATH or imageio-ffmpeg")
    out = tmp_path / "real.mp4"
    writer = _write(out)
    assert out.stat().st_size > 0
    assert writer.codec == "avc1"
    assert _codec_of(out) == "h264"
    capture = cv2.VideoCapture(str(out))
    try:
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
    finally:
        capture.release()


@pytest.mark.unit
def test_odd_frame_size_still_transcodes(tmp_path: Path) -> None:
    if mod.find_ffmpeg() is None:
        pytest.skip("no ffmpeg on PATH or imageio-ffmpeg")
    out = tmp_path / "odd.mp4"
    writer = Mp4Writer(out, 10, (63, 47))
    for _ in range(4):
        writer.write(np.zeros((47, 63, 3), dtype=np.uint8))
    writer.release()
    assert _codec_of(out) == "h264"
