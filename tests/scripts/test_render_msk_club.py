"""Frame selection and options of the OSV-9 two-hand club render (#11756, #12042)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "render_msk_club.py"


def _script():  # noqa: ANN202
    spec = importlib.util.spec_from_file_location("render_msk_club", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Events:
    address_time = 0.5
    top_time = 1.2
    impact_time = 1.4


def test_event_frames_match_the_generated_model_events() -> None:
    frames = _script().event_frames(908)
    assert frames == {"address": 0, "top": 552, "impact": 663, "finish": 907}


def test_event_frames_follow_the_capture_events() -> None:
    script = _script()
    assert script.event_times_s(_Events()) == {
        "address": 0.5,
        "top": 1.2,
        "impact": 1.4,
    }
    frames = script.event_frames(908, _Events())
    assert frames == {"address": 250, "top": 600, "impact": 700, "finish": 907}
    clipped = script.event_frames(600, _Events())
    assert clipped["impact"] == 599


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


def test_defaults_are_the_baseline_driver_render() -> None:
    args = _script().parse_args([])
    assert (args.club, args.mode, args.capture) == ("driver", "baseline", None)
    assert args.out == _script().DEFAULT_OUT


def test_a_no_capture_iron_render_does_not_overwrite_the_driver_baseline() -> None:
    script = _script()
    args = script.parse_args(["--club", "iron7"])
    assert args.out == script.DEFAULT_OUT / "iron7_baseline"


def test_capture_defaults_to_turn_feet_and_a_club_mode_folder() -> None:
    script = _script()
    args = script.parse_args(["--capture", "c.c3d", "--club", "iron7"])
    assert args.mode == "turn_feet"
    assert args.out == script.DEFAULT_OUT / "iron7_turn_feet"
    explicit = script.parse_args(["--capture", "c.c3d", "--mode", "turn"])
    assert explicit.out == script.DEFAULT_OUT / "driver_turn"


def test_a_turn_mode_without_a_capture_is_a_usage_error() -> None:
    with pytest.raises(SystemExit) as info:
        _script().parse_args(["--mode", "turn"])
    assert info.value.code == 2
    assert _script().parse_args(["--mode", "baseline"]).mode == "baseline"


def test_unknown_club_or_mode_is_rejected() -> None:
    script = _script()
    with pytest.raises(SystemExit):
        script.parse_args(["--club", "wedge"])
    with pytest.raises(ValueError, match="club"):
        script.check_options("wedge", "baseline", False)
    with pytest.raises(ValueError, match="mode"):
        script.check_options("driver", "bogus", True)


def test_no_capture_loads_no_inputs() -> None:
    assert _script().load_capture_inputs(None) is None


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_club_model_carries_the_tracked_club(tmp_path: Path, club: str) -> None:
    from defusedxml import ElementTree as SafeET

    from src.engines.physics_engines.opensim.python import msk_club as mc

    script = _script()
    model_path = mc.MODELS_DIR / "golf_humanoid.osim"
    out = script.club_model(model_path, club, tmp_path)
    assert out.parent == tmp_path and out.name == model_path.name
    model = SafeET.parse(str(out)).getroot().find("Model")
    (body,) = [b for b in model.iter("Body") if b.get("name") == mc.CLUB_BODY]
    expected = mc.load_msk_club(club).dynamics.mass_kg
    assert float(body.findtext("mass")) == pytest.approx(expected)
    defaults = {
        c.get("name"): float(c.findtext("default_value"))
        for c in model.iter("Coordinate")
        if c.findtext("default_value") is not None
    }
    address_q = mc.load_calibration("golf_humanoid", club).address_q
    for name, value in address_q.items():
        assert defaults[name] == pytest.approx(value), name


def test_club_model_rejects_an_unknown_club(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="club"):
        _script().club_model(Path("unused.osim"), "wedge", tmp_path)


def test_render_refuses_the_real_display(monkeypatch) -> None:  # noqa: ANN001
    pytest.importorskip("opensim")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("NATIVE_VIEWER_XVFB", "1")
    with pytest.raises(RuntimeError, match="real display"):
        _script().render(Path("unused.osim"), Path("unused"), None)
