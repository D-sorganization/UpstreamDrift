"""Time-based speed variants, HUD, impact window and encoding (GCV-14, #11720)."""

from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.compositor import HudInfo
from src.tools.native_viewer_export.core import (
    HQ_CRF,
    ExportSettings,
    SwingInput,
    clip_plans,
    export_swing,
    imageio_writer,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")


def _swing(steps: int = 300, dt: float = 0.001, **provenance: float) -> SwingInput:
    t = np.arange(steps + 1) * dt
    q = np.stack([t * 10.0, np.sin(t)], axis=1)
    spec = {"coordinate_order": list(COORDS)}
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=dt,
        q0=q[0],
        v0=np.zeros(2),
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=np.zeros_like(q),
        reference_engine="mujoco",
        provenance=dict(provenance),
    )
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


class _Writer:
    def __init__(self, path: Path, fps: int, sink: dict) -> None:
        self.fps, self.frames = fps, []
        sink[path.name] = self

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


class _Backend:
    engine = "fake"

    def __init__(self) -> None:
        self.swings: list[SwingInput] = []

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        self.swings.append(swing)
        for _ in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def _run(settings: ExportSettings, tmp_path: Path, swing: SwingInput | None = None):
    sink: dict = {}
    backend = _Backend()
    result = export_swing(
        backend,
        swing or _swing(),
        settings,
        tmp_path,
        writer_factory=lambda p, fps: _Writer(p, fps, sink),
    )
    return result, sink, backend


def test_defaults_are_60_fps_full_and_half_speed_at_720p() -> None:
    s = ExportSettings()
    assert (s.fps, s.speeds) == (60, (1.0, 0.5))
    assert (s.width, s.height) == (1280, 720) and s.crf <= 18
    assert (ExportSettings.preview().width, ExportSettings.preview().height) == (
        640,
        544,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"fps": 0},
        {"speeds": ()},
        {"speeds": (0.0,)},
        {"speeds": (4.5,)},
        {"speeds": (float("nan"),)},
        {"speeds": (1.0, 1.0)},
        {"crf": 60},
        {"impact_window_s": 0.0},
    ],
)
def test_settings_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        ExportSettings(**kwargs)


def test_stride_is_a_deprecated_alias() -> None:
    with pytest.warns(DeprecationWarning, match="stride"):
        s = ExportSettings(stride=40, fps=20)
    assert s.effective_speeds(0.001) == pytest.approx((0.8,))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert ExportSettings().effective_speeds(0.001) == (1.0, 0.5)


def test_one_and_half_speed_clip_sets_with_suffixes(tmp_path: Path) -> None:
    result, sink, _ = _run(ExportSettings(width=32, height=32), tmp_path)
    assert sorted(result.frames_by_suffix) == ["_0p5x", "_1x"]
    assert all(w.fps == 60 for w in sink.values())
    assert "driver_fake_face_on_1x.mp4" in sink
    assert "driver_fake_2x2_0p5x.mp4" in sink
    full, half = result.frames_by_suffix["_1x"], result.frames_by_suffix["_0p5x"]
    assert abs(half - 2 * full) <= 1
    assert result.frames == full == len(sink["driver_fake_face_on_1x.mp4"].frames)


def test_backend_gets_interpolated_poses_at_frame_times(tmp_path: Path) -> None:
    # 0.3 s at 1 kHz, 60 fps: frames fall between samples; q[:, 0] = 10 t is linear
    _, _, backend = _run(ExportSettings(width=32, height=32, speeds=(1.0,)), tmp_path)
    shown = backend.swings[0]
    times = np.asarray(shown.sample_times_s)
    np.testing.assert_allclose(np.diff(times), 1 / 60)
    np.testing.assert_allclose(shown.q[:, 0], 10.0 * times, atol=1e-9)


def test_hud_shows_speed_and_ms_from_impact() -> None:
    hud = HudInfo(0.1, "Drake", "Driver", "", speed=0.5, ms_from_impact=-12.0)
    assert hud.lines() == ["Drake | Driver | t = 0.100 s | 0.5x | -12.0 ms from impact"]
    with pytest.raises(ValueError, match="speed"):
        HudInfo(0.0, "D", "C", "", speed=0.0)


def test_impact_window_adds_slow_motion_clip(tmp_path: Path) -> None:
    swing = _swing(impact_time_s=0.25)
    settings = ExportSettings(width=32, height=32, speeds=(1.0,), impact_window_s=0.1)
    assert [p.suffix for p in clip_plans(swing, settings)] == [
        "_1x",
        "_impact_0p1x",
    ]
    result, sink, backend = _run(settings, tmp_path, swing)
    assert "driver_fake_face_on_impact_0p1x.mp4" in sink
    clip = backend.swings[1]
    times = np.asarray(clip.sample_times_s)
    assert times[0] == pytest.approx(0.2)
    np.testing.assert_allclose(np.diff(times), 0.1 / 60)
    assert times[-1] <= 0.3 + 1e-9
    assert result.frames_by_suffix["_impact_0p1x"] == len(times)


def test_impact_defaults_to_end_of_swing_and_can_be_overridden() -> None:
    swing = _swing()
    assert swing.impact_time_s == pytest.approx(0.3)
    plans = clip_plans(swing, ExportSettings(impact_time_s=0.2, impact_window_s=0.1))
    assert plans[-1].window == pytest.approx((0.15, 0.25))


def test_writer_encodes_h264_yuv420p_crf_18(monkeypatch, tmp_path: Path) -> None:
    import imageio.v2 as imageio

    seen: dict = {}
    monkeypatch.setattr(imageio, "get_writer", lambda *a, **k: seen.update(k) or "w")
    assert imageio_writer(tmp_path / "x.mp4", 60) == "w"
    assert seen["codec"] == "libx264" and seen["pixelformat"] == "yuv420p"
    params = seen["output_params"]
    assert int(params[params.index("-crf") + 1]) == HQ_CRF <= 18
    assert seen["fps"] == 60


def test_cli_speeds_preset_and_stride_warning(tmp_path: Path) -> None:
    args = cli.build_parser().parse_args(
        ["--bundle", "b", "--out", "o", "--swing", "s", "--speeds", "1,0.5,0.25"]
    )
    settings = cli.build_settings(args)
    assert settings.speeds == (1.0, 0.5, 0.25) and settings.fps == 60
    prev = cli.build_parser().parse_args(
        ["--bundle", "b", "--out", "o", "--swing", "s", "--preset", "preview"]
    )
    assert (cli.build_settings(prev).width, cli.build_settings(prev).crf) == (640, 23)
    with pytest.raises(ValueError, match="--speeds"):
        cli.parse_speeds("fast")
    bad = cli.build_parser().parse_args(
        ["--bundle", "b", "--out", "o", "--swing", "s", "--speeds", "8"]
    )
    with pytest.raises(ValueError, match="speeds"):
        cli.build_settings(bad)
