"""Frame selection of the OSV-9 two-hand club render (#11756)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("opensim")
pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_msk_club.py"


def _script():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("render_msk_club", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_event_frames_match_the_generated_model_events() -> None:
    frames = _script().event_frames(908)
    assert frames == {"address": 0, "top": 552, "impact": 663, "finish": 907}


def test_clip_frames_play_at_half_speed() -> None:
    script = _script()
    frames = script.clip_frames(908)
    swing_s = 907 * script.SWING_DT_S
    playback_s = len(frames) / script.CLIP_FPS
    assert playback_s == pytest.approx(
        swing_s / script.SPEED, abs=1.0 / script.CLIP_FPS
    )
    assert frames[0] == 0 and frames[-1] <= 907
    assert all(b > a for a, b in zip(frames, frames[1:], strict=False))


def test_render_refuses_the_real_display(monkeypatch) -> None:  # noqa: ANN001
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("NATIVE_VIEWER_XVFB", "1")
    with pytest.raises(RuntimeError, match="real display"):
        _script().render(Path("unused.osim"), Path("unused"), None)
