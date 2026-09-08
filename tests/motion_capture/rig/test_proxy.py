"""H.264 proxies: argv contract, honest index, unusable recordings kept."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from src.motion_capture.rig.proxy import (
    PROXIES_FILE,
    PROXIES_SCHEMA_VERSION,
    make_proxies,
    proxy_args,
)

from .test_ingest import _bundle

pytestmark = pytest.mark.unit


def test_proxy_args_libx264_is_browser_playable(tmp_path: Path) -> None:
    args = proxy_args("ffmpeg", tmp_path / "a.mkv", tmp_path / "a.mp4")
    assert args[0] == "ffmpeg"
    assert args[args.index("-c:v") + 1] == "libx264"
    assert "-crf" in args and args[args.index("-pix_fmt") + 1] == "yuv420p"
    assert "+faststart" in args and args[-1].endswith("a.mp4")


def test_proxy_args_hardware_encoder_has_no_crf(tmp_path: Path) -> None:
    args = proxy_args(
        "ffmpeg", tmp_path / "a.mkv", tmp_path / "a.mp4", encoder="h264_nvenc"
    )
    assert "-crf" not in args and args[args.index("-c:v") + 1] == "h264_nvenc"


def test_proxy_args_rejects_unknown_encoder_and_bad_crf(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="encoder"):
        proxy_args("ffmpeg", tmp_path / "a.mkv", tmp_path / "a.mp4", encoder="vp9")
    with pytest.raises(Exception, match="crf"):
        proxy_args("ffmpeg", tmp_path / "a.mkv", tmp_path / "a.mp4", crf=99)


class _FakeRunner:
    """Stands in for ``subprocess.run``: records argv, optionally writes output."""

    def __init__(self, returncode: int, write: bool) -> None:
        self.returncode = returncode
        self.write = write
        self.calls: list[list[str]] = []

    def __call__(
        self, args: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if self.write:
            Path(args[-1]).write_bytes(b"\x00" * 16)
        return subprocess.CompletedProcess(args, self.returncode, "", "")


def test_make_proxies_indexes_every_recording(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    bundle = _bundle(tmp_path)
    runner = _FakeRunner(0, write=True)
    out = make_proxies(bundle, ffmpeg_exe="ffmpeg", runner=runner)
    assert out.schema_version == PROXIES_SCHEMA_VERSION and out.ok
    assert {p.view for p in out.proxies} == {"a", "b"}
    assert all(p.file and p.file.endswith(".mp4") for p in out.proxies)
    assert len(runner.calls) == 2
    on_disk = json.loads((bundle / PROXIES_FILE).read_text(encoding="utf-8"))
    assert on_disk["proxies"][0]["encoder"] == "libx264"


def test_make_proxies_reports_failed_transcode_honestly(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    bundle = _bundle(tmp_path)
    out = make_proxies(bundle, ffmpeg_exe="ffmpeg", runner=_FakeRunner(1, False))
    assert not out.ok
    assert all(p.file is None and "exited 1" in (p.reason or "") for p in out.proxies)


def test_make_proxies_keeps_unusable_recordings_with_reason(tmp_path: Path) -> None:
    pytest.importorskip("cv2")
    bundle = _bundle(tmp_path, break_second=True)
    runner = _FakeRunner(0, write=True)
    out = make_proxies(bundle, ffmpeg_exe="ffmpeg", runner=runner)
    by_view = {p.view: p for p in out.proxies}
    assert by_view["a"].ok and len(runner.calls) == 1
    assert not by_view["b"].ok and "not usable" in (by_view["b"].reason or "")
