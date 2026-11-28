"""The comparison grid encodes to a real MP4 (#15725)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.hand_load_video.compare import (
    COMPONENTS,
    STRATEGIES,
    GridSpec,
    Provenance,
    align,
    panel_from_arrays,
)
from src.hand_load_video.writer import (
    EncodeSettings,
    VideoEncodeError,
    write_comparison_mp4,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_RED, _GREEN = (180, 0, 0), (0, 160, 0)


def _comparison():
    panels = []
    for strategy in STRATEGIES:
        badge = "allocation-only" if strategy == "baseline" else "forward-replayed"
        for component in COMPONENTS:
            if (strategy, component) == ("baseline", "ztcf"):
                colors = ((0, 0, 0), (0, 0, 0))
            elif (strategy, component) == ("baseline", "total"):
                colors = (_RED, _GREEN)
            else:
                colors = (None, None)
            frames = [
                None if color is None else np.full((8, 8, 3), color, np.uint8)
                for color in colors
            ]
            panels.append(
                panel_from_arrays(
                    strategy,
                    component,
                    (0.0, 0.1),
                    frames,
                    badge=badge,
                    units="N",
                )
            )
    spec = GridSpec(
        cell_w=64, cell_h=48, gutter=16, banner_h=16, footer_h=16, label_w=96
    )
    synced = align(tuple(panels), (0.0, 0.1), tolerance_s=0.0)
    return synced, Provenance("fixture-swing", "toy-model", "abc123def"), spec


def _settings() -> EncodeSettings:
    return EncodeSettings(fps=8, crf=12)


def _near(pixel, rgb, tol=12) -> bool:
    return all(abs(int(pixel[channel]) - rgb[channel]) <= tol for channel in range(3))


def _region(image, spec: GridSpec, row: int, col: int):
    x, y, width, height = spec.cell_box(row, col)
    return image[y : y + height, x : x + width]


def test_missing_ffmpeg_raises_and_writes_nothing(tmp_path, monkeypatch) -> None:
    frames, provenance, spec = _comparison()
    destination = tmp_path / "comparison.mp4"
    opened: list[Path] = []

    def _missing() -> str:
        raise VideoEncodeError("ffmpeg is unavailable")

    def _open(path: Path, _settings: EncodeSettings):
        opened.append(path)
        path.write_bytes(b"")
        raise AssertionError("encoder opened without ffmpeg")

    monkeypatch.setattr("src.hand_load_video.writer.ffmpeg_executable", _missing)
    monkeypatch.setattr("src.hand_load_video.writer._open_writer", _open)
    with pytest.raises(VideoEncodeError, match="ffmpeg is unavailable"):
        write_comparison_mp4(frames, provenance, destination, _settings(), spec)
    assert opened == []
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_encode_failure_removes_an_empty_or_fake_file(tmp_path, monkeypatch) -> None:
    frames, provenance, spec = _comparison()
    destination = tmp_path / "comparison.mp4"

    class _Fake:
        def append_data(self, _frame) -> None:
            return None

        def close(self) -> None:
            destination.write_bytes(b"not-a-real-mp4")

    monkeypatch.setattr(
        "src.hand_load_video.writer.ffmpeg_executable", lambda: "ffmpeg"
    )
    monkeypatch.setattr(
        "src.hand_load_video.writer._open_writer", lambda _path, _settings: _Fake()
    )
    with pytest.raises(VideoEncodeError, match="ffmpeg"):
        write_comparison_mp4(frames, provenance, destination, _settings(), spec)
    assert not destination.exists()

    def _explode(_path: Path, _settings: EncodeSettings):
        destination.write_bytes(b"")
        raise OSError("encoder failed")

    monkeypatch.setattr("src.hand_load_video.writer._open_writer", _explode)
    with pytest.raises(VideoEncodeError, match="ffmpeg"):
        write_comparison_mp4(frames, provenance, destination, _settings(), spec)
    assert not destination.exists()


def test_invalid_request_writes_nothing(tmp_path, monkeypatch) -> None:
    frames, provenance, spec = _comparison()
    seen: list[str] = []
    monkeypatch.setattr(
        "src.hand_load_video.writer.ffmpeg_executable",
        lambda: seen.append("called") or "ffmpeg",
    )
    settings = _settings()
    destination = tmp_path / "comparison.mp4"
    with pytest.raises(ValueError, match="at least one frame"):
        write_comparison_mp4((), provenance, destination, settings, spec)
    with pytest.raises(ValueError, match=r"\.mp4"):
        write_comparison_mp4(
            frames, provenance, tmp_path / "comparison.mov", settings, spec
        )
    with pytest.raises(ValueError, match="directory"):
        write_comparison_mp4(
            frames, provenance, tmp_path / "missing" / "comparison.mp4", settings, spec
        )
    with pytest.raises(TypeError):
        write_comparison_mp4(frames, provenance, destination, object(), spec)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        EncodeSettings(fps=0)
    with pytest.raises(ValueError):
        EncodeSettings(fps=8, crf=52)
    assert seen == []
    assert list(tmp_path.iterdir()) == []


def test_writes_a_real_h264_mp4_of_the_comparison(tmp_path) -> None:
    import imageio.v2 as imageio

    frames, provenance, spec = _comparison()
    destination = tmp_path / "comparison.mp4"
    written = write_comparison_mp4(frames, provenance, destination, _settings(), spec)
    assert written == destination
    assert destination.is_file() and destination.stat().st_size > 500
    assert b"ftyp" in destination.read_bytes()[:32]
    reader = imageio.get_reader(str(destination))
    try:
        meta = reader.get_meta_data()
        decoded = [np.asarray(frame) for frame in reader]
    finally:
        reader.close()
    assert meta["codec"] == "h264"
    assert meta["fps"] == 8
    assert len(decoded) == len(frames) == 2
    assert decoded[0].shape == (spec.height, spec.width, 3)
    for image in decoded:
        assert _near(image[spec.header_sample(0)], spec.component_rgb["ztcf"])
        assert _near(image[spec.header_sample(1)], spec.component_rgb["zvcf"])
        assert _near(image[spec.header_sample(2)], spec.component_rgb["total"])
        assert _near(image[spec.badge_sample(0)], spec.badge_rgb["allocation-only"])
        assert _near(image[spec.badge_sample(1)], spec.badge_rgb["forward-replayed"])
        assert _near(image[spec.content_sample(0, 0)], (0, 0, 0), tol=8)
        assert float(_region(image, spec, 0, 1).mean()) > 20
        purple = spec.purple
        region = _region(image, spec, 1, 2)
        painted = (region[:, :, 0] > purple[0] // 2) & (
            region[:, :, 2] > purple[2] // 3
        )
        assert int(np.count_nonzero(painted)) > 100
        assert float(region.mean()) > 20
    assert _near(decoded[0][spec.content_sample(0, 2)], _RED)
    assert _near(decoded[1][spec.content_sample(0, 2)], _GREEN)
