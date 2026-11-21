"""On-screen impact factors panel of native exports (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.impact_parameters import ClubheadSeries
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import impact_factors as factors
from src.tools.native_viewer_export.ball import BallPlacement
from src.tools.native_viewer_export.core import (
    ExportSettings,
    SwingInput,
    export_swing,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SPEED, IMPACT_S = 40.0, 0.1
DEFAULTS = factors.load_factor_settings()
BALL = BallPlacement((0.021335, 0.0, 0.0), 0.021335, "test")


def _series(height_m: float = 0.0) -> ClubheadSeries:
    """Face moving along +X at ``SPEED``, passing ``height_m`` above the origin
    at ``IMPACT_S`` after a slow 4 m/s move away from it (the backswing)."""
    t = np.arange(0.0, 0.2, 0.001)
    x = np.where(t < 0.05, -1.8 - 4.0 * t, SPEED * (t - IMPACT_S))
    centre = np.column_stack([x, np.zeros_like(t), np.full_like(t, height_m)])
    return ClubheadSeries(
        times_s=t,
        face_center_m=centre,
        velocity_mps=np.gradient(centre, t, axis=0),
        face_normal=np.tile([1.0, 0.0, 0.0], (len(t), 1)),
        toe_axis=np.tile([0.0, -1.0, 0.0], (len(t), 1)),
        grip_axis=np.tile([0.0, 0.0, 1.0], (len(t), 1)),
    )


def _blank() -> np.ndarray:
    return np.full((360, 640, 3), 40, np.uint8)


def test_settings_file_is_validated() -> None:
    wanted = {"clubhead_speed", "club_path_deg", "face_angle_deg", "face_to_path_deg"}
    wanted |= {"attack_angle_deg", "dynamic_loft_deg", "impact_location"}
    assert wanted <= set(DEFAULTS.rows)
    with pytest.raises(ValueError, match="rows"):
        replace(DEFAULTS, rows=("clubhead_speed", "launch_code"))
    with pytest.raises(TypeError, match="rows"):
        replace(DEFAULTS, rows="clubhead_speed")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="position"):
        replace(DEFAULTS, position="middle")
    with pytest.raises(ValueError, match="opacity"):
        replace(DEFAULTS, opacity=1.5)
    with pytest.raises(ValueError, match="positive"):
        replace(DEFAULTS, min_font_px=0)


def test_values_are_blank_then_freeze_at_impact_and_stay() -> None:
    panel = factors.panel_from_series(_series(), IMPACT_S, BALL, DEFAULTS)
    before, at, late = (panel.lines_at(t) for t in (0.02, IMPACT_S, 5.0))
    assert at == late and before != at
    assert [label for label, _ in before] == [label for label, _ in at]
    assert before[0] == ("Status", "before impact") and at[0][1] == "at impact"
    assert all(value == DEFAULTS.blank_text for _, value in before[1:])
    values = dict(at)
    speed, unit = values["Clubhead Speed"].split()
    assert unit == "mph" and float(speed) == pytest.approx(89.5, abs=0.5)
    assert values["Club Path"].endswith("deg") and values["Face Angle"].endswith("deg")
    assert "mm" in values["Impact Location (toe, high)"]


def test_panel_size_is_the_same_on_every_frame() -> None:
    panel = factors.panel_from_series(_series(), IMPACT_S, BALL, DEFAULTS)
    box = panel.bbox(_blank().shape)
    for t in (0.0, 0.05, IMPACT_S, 0.15, 9.0):
        out = panel.draw(_blank(), t)
        ys, xs = np.nonzero((out != 40).any(axis=2))
        assert (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1) == box, t
    assert box[0] > 320 and box[3] <= 360
    left = replace(panel, settings=replace(DEFAULTS, position="left"))
    assert left.bbox(_blank().shape)[2] < 320


def test_no_impact_says_so_and_shows_no_numbers() -> None:
    panel = factors.no_impact_panel(DEFAULTS)
    for t in (0.0, 1.0):
        lines = panel.lines_at(t)
        assert lines[0][1] == "no impact detected"
        assert all(value == DEFAULTS.blank_text for _, value in lines[1:])
    assert (panel.draw(_blank(), 0.0) == panel.draw(_blank(), 1.0)).all()


def test_bad_input_is_rejected() -> None:
    panel = factors.no_impact_panel(DEFAULTS)
    with pytest.raises(ValueError, match="finite"):
        panel.lines_at(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        factors.panel_from_series(_series(), float("inf"), BALL, DEFAULTS)
    with pytest.raises(TypeError, match="FactorSettings"):
        factors.panel_from_series(_series(), IMPACT_S, BALL, {})  # type: ignore[arg-type]


# --- export loop ---------------------------------------------------------------


class _Writer:
    def __init__(self, sink: list[np.ndarray]) -> None:
        self.append_data = sink.append

    def close(self) -> None:
        pass


class _Backend:
    engine = "fake"

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        for _k in indices:
            yield {v: _blank() for v in settings.views}


def _swing() -> SwingInput:
    names, q = ("A", "B"), np.zeros((21, 2))
    spec = json.dumps({"coordinate_order": list(names)}).encode()
    zeros = np.zeros((20, 2))
    bundle = InputBundle(spec, names, 0.01, q[0], q[0], zeros, q, q.copy(), "mujoco")
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


def _settings(**overrides: object) -> ExportSettings:
    base = ExportSettings(views=("face_on",), multiview=False, width=640, height=360)
    return replace(base, speeds=(1.0,), ball=False, **overrides)  # type: ignore[arg-type]


def _frames(tmp_path: Path, **overrides: object) -> list[np.ndarray]:
    sink: list[np.ndarray] = []
    export_swing(
        _Backend(),
        _swing(),
        _settings(**overrides),
        tmp_path,
        writer_factory=lambda path, fps: _Writer(sink),
    )
    return sink


def test_export_draws_the_panel_on_every_frame(tmp_path: Path, monkeypatch) -> None:
    plain = _frames(tmp_path, impact_time_s=IMPACT_S)
    monkeypatch.setattr(factors, "club_series", lambda swing: _series())
    frames = _frames(tmp_path, impact_factors=DEFAULTS, impact_time_s=IMPACT_S)
    panel = factors.panel_from_series(_series(), IMPACT_S, None, DEFAULTS)
    x0, y0, x1, y1 = panel.bbox(frames[0].shape)
    inside = [f[y0:y1, x0:x1].copy() for f in frames]
    assert len(frames) == 13 and all((part != 40).any() for part in inside)
    assert (inside[0] != inside[-1]).any()  # blank rows, then the frozen values
    assert all((part == inside[-1]).all() for part in inside[7:])
    for shown, base in zip(frames, plain, strict=True):
        shown[y0:y1, x0:x1] = base[y0:y1, x0:x1]
        assert (shown == base).all()  # nothing outside the panel box changes
    assert _frames(tmp_path, impact_factors=DEFAULTS, hud=False)[0].max() == 40


def test_export_is_unchanged_with_the_option_off(tmp_path: Path, monkeypatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("the factors panel must not be built when it is off")

    assert ExportSettings().impact_factors is None
    plain = _frames(tmp_path, impact_time_s=IMPACT_S)
    monkeypatch.setattr(factors, "factors_for_export", boom)
    off = _frames(tmp_path, impact_time_s=IMPACT_S, impact_factors=None)
    assert len(off) == len(plain)
    assert all((a == b).all() for a, b in zip(off, plain, strict=True))


def test_export_without_a_known_impact_says_no_impact(monkeypatch) -> None:
    monkeypatch.setattr(factors, "club_series", lambda swing: _series())
    panel = factors.factors_for_export(_swing(), _settings(impact_factors=DEFAULTS))
    assert panel is not None and panel.lines_at(0.19)[0][1] == "no impact detected"
    monkeypatch.undo()  # the two-coordinate test specification has no club
    failed = factors.factors_for_export(
        _swing(), _settings(impact_factors=DEFAULTS, impact_time_s=IMPACT_S)
    )
    assert failed is not None and failed.lines_at(0.19)[0][1].startswith("unavailable")
    assert all(v == DEFAULTS.blank_text for _, v in failed.lines_at(0.19)[1:])
