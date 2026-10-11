"""GCV-19 (#11725): synchronised inset plot strips."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export import core
from src.tools.native_viewer_export.core import (
    STANDARD_GRAVITY_M_S2,
    ExportSettings,
    OverlayFeed,
    SwingInput,
    default_glyph_style,
    export_swing,
)
from src.tools.native_viewer_export.inset_strip import (
    Curve,
    Panel,
    StripRenderer,
    StripSeries,
    draw_strip,
    grf_series,
    grip_series,
    render_strip,
    strip_bbox,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

MASS_KG = 80.0
WEIGHT_N = MASS_KG * STANDARD_GRAVITY_M_S2
N = 101


def _wrench(label: str, fz: float, point=(0.1, 0.2, 0.0)) -> OverlayWrench:
    return OverlayWrench(
        WrenchKind.CONTACT,
        label,
        "foot",
        point,
        force_n=(0.0, 0.0, fz),
        source="test",
    )


def _free_moment(point) -> OverlayWrench:
    return OverlayWrench(
        WrenchKind.CONTACT,
        "contact:free_moment_net",
        "system",
        point,
        torque_nm=(0.0, 0.0, 1.0),
        source="test",
    )


def _ftf(i: int, wrenches) -> ForceTorqueFrame:
    return ForceTorqueFrame(time_s=i * 0.01, engine="fake", wrenches=tuple(wrenches))


def _frame(i: int) -> ForceTorqueFrame:
    """Known values: left (1 + i/100) BW, right 0.5 BW, net their sum."""
    left, right = WEIGHT_N * (1 + i / 100), WEIGHT_N * 0.5
    wrenches = [
        _wrench("contact:grf_left", left),
        _wrench("contact:grf_net", left + right),
        _free_moment((0.01 * i, -0.02 * i, 0.0)),
    ]
    if i % 2 == 0:  # the right foot is unloaded on odd samples
        wrenches.append(_wrench("contact:grf_right", right))
    return _ftf(i, wrenches)


def _feed(**kw) -> OverlayFeed:
    base = {"frame_at": _frame, "style": default_glyph_style(body_mass_kg=MASS_KG)}
    return OverlayFeed(**{**base, **kw})


def _times() -> np.ndarray:
    return np.arange(N) * 0.01


def _analysis(fx: float | None, mz: float | None, why: str = "") -> GripAnalysis:
    return GripAnalysis(
        left=None,
        right=None,
        midpoint_m=None,
        net_force_n=None if fx is None else (fx, 0.0, 0.0),
        couple_at_midpoint_nm=None if mz is None else (0.0, 0.0, mz),
        contact_force_moment_nm=None,
        applied_free_torque_nm=None,
        mof_left_nm=None,
        mof_right_nm=None,
        split_method="unavailable",
        unavailable_reason=why,
    )


# --- series extraction ---------------------------------------------------------------


def test_grf_series_converts_to_body_weights_exactly() -> None:
    out = grf_series(_feed(), N, _times(), impact_time_s=0.8)
    assert isinstance(out, StripSeries)
    assert out.kind == "grf" and out.impact_time_s == 0.8
    grf, cop = out.panels
    left, right, net = (c.values for c in grf.curves)
    assert out.times_s.shape == left.shape
    np.testing.assert_allclose(left, 1.0 + out.times_s / 0.01 / 100.0)
    np.testing.assert_allclose(net[~np.isnan(right)], left[~np.isnan(right)] + 0.5)
    assert grf.ylabel == "Vertical GRF (BW)" and "(m)" in cop.ylabel
    x, y = (c.values for c in cop.curves)
    np.testing.assert_allclose(x, 0.01 * out.times_s / 0.01)
    np.testing.assert_allclose(y, -0.02 * out.times_s / 0.01)


def test_unloaded_foot_is_a_gap_not_zero() -> None:
    out = grf_series(_feed(), N, _times())
    right = out.panels[0].curves[1].values
    assert np.isnan(right).any() and np.isfinite(right).any()
    assert not (right == 0.0).any()


def test_missing_cop_drops_the_panel() -> None:
    def frame(i: int) -> ForceTorqueFrame:
        return _ftf(i, [_wrench("contact:grf_net", WEIGHT_N)])

    out = grf_series(_feed(frame_at=frame), N, _times())
    assert isinstance(out, StripSeries) and len(out.panels) == 1


@pytest.mark.parametrize(
    "feed, fragment",
    [
        (None, "overlays are disabled"),
        (_feed(style=default_glyph_style()), "body weight is unknown"),
        (_feed(frame_at=lambda i: _ftf(i, [])), "no sample has"),
    ],
)
def test_grf_series_reports_a_reason_when_unavailable(feed, fragment) -> None:
    out = grf_series(feed, N, _times())
    assert isinstance(out, str) and fragment in out


def test_series_use_the_source_samples_thinned_to_the_cap() -> None:
    n = 5000
    seen: list[int] = []

    def frame(i: int) -> ForceTorqueFrame:
        seen.append(i)
        return _frame(i % 100)

    out = grf_series(_feed(frame_at=frame), n, np.arange(n) * 0.001)
    assert len(seen) <= 600 and seen[0] == 0 and seen[-1] == n - 1
    assert out.times_s[0] == 0.0 and out.times_s[-1] == pytest.approx((n - 1) * 0.001)


def test_grip_series_norms_and_gaps() -> None:
    def analyses(idx):
        return [_analysis(30.0 if i else None, 4.0 if i % 2 else None) for i in idx]

    out = grip_series(_feed(grip_analyses=analyses), N, _times())
    assert isinstance(out, StripSeries) and out.kind == "grip"
    force, couple = (p.curves[0].values for p in out.panels)
    assert np.isnan(force[0]) and force[1] == pytest.approx(30.0)
    assert np.isnan(couple[0]) and couple[1] == pytest.approx(4.0)  # odd i only
    assert np.isnan(couple[-1])  # i = 100 is even: a gap, not zero
    assert out.panels[0].ylabel == "Net grip force (N)"
    assert out.panels[1].ylabel == "Grip couple (N m)"


def test_grip_series_unavailable_reasons() -> None:
    out = grip_series(_feed(), N, _times())
    assert isinstance(out, str) and "not enabled" in out

    def none(idx):
        return [_analysis(None, None, "grip solve failed: singular") for _ in idx]

    out = grip_series(_feed(grip_analyses=none), N, _times())
    assert isinstance(out, str) and "singular" in out


def test_series_reject_bad_input() -> None:
    with pytest.raises(ValueError, match="n_source"):
        grf_series(_feed(), 1, [0.0])
    with pytest.raises(ValueError, match="shape"):
        grf_series(_feed(), N, _times()[:-1])
    with pytest.raises(ValueError, match="strictly increasing"):
        StripSeries("x", np.array([0.0, 0.0]), (_panel(2),))
    with pytest.raises(ValueError, match="length"):
        StripSeries("x", np.arange(3.0), (_panel(2),))


def _panel(n: int) -> Panel:
    return Panel("y (u)", (Curve("a", np.linspace(0.0, 1.0, n), "#ffffff"),))


# --- rendering and compositing -------------------------------------------------------


def _series() -> StripSeries:
    out = grf_series(_feed(), N, _times(), impact_time_s=0.8)
    assert isinstance(out, StripSeries)
    return out


def test_render_strip_shape_dtype_and_mapping() -> None:
    strip = render_strip(_series(), 600, 120)
    assert strip.rgba.shape == (120, 600, 4) and strip.rgba.dtype == np.uint8
    assert len(strip.maps) == 2
    for m in strip.maps:
        assert 0 <= m.x0 < m.x1 <= 600 and 0 <= m.y0 < m.y1 <= 120
    assert strip.maps[0].x1 <= strip.maps[1].x0
    assert (strip.rgba[..., 3] > 0).any()  # something is drawn


def test_draw_strip_changes_only_the_strip_region() -> None:
    frame = np.full((360, 640, 3), 40, np.uint8)
    strip = render_strip(_series(), 600, 90)
    out = draw_strip(frame, strip, 0.5)
    x0, y0, x1, y1 = strip_bbox(frame.shape, strip)
    changed = (out != frame).any(axis=2)
    assert changed[y0:y1, x0:x1].any()
    mask = np.zeros_like(changed)
    mask[y0:y1, x0:x1] = True
    assert not changed[~mask].any()
    assert (frame == 40).all()  # the input is not modified
    assert y1 <= 360 - 52 and x0 >= 0 and x1 <= 640  # HUD stays uncovered


def test_cursor_moves_monotonically_with_time() -> None:
    strip = render_strip(_series(), 600, 90)
    ts = np.linspace(-0.2, 1.2, 40)  # includes out-of-range values (clamped)
    xs = np.array([strip.cursor_xs(float(t)) for t in ts])
    assert (np.diff(xs, axis=0) >= 0).all()
    assert (xs[-1] > xs[0]).all()
    frame = np.full((360, 640, 3), 40, np.uint8)
    a, b = draw_strip(frame, strip, 0.2), draw_strip(frame, strip, 0.9)
    assert (a != b).any()


def test_draw_strip_rejects_bad_input() -> None:
    strip = render_strip(_series(), 600, 90)
    frame = np.full((360, 640, 3), 40, np.uint8)
    with pytest.raises(ValueError, match="uint8"):
        draw_strip(frame.astype(float), strip, 0.1)
    with pytest.raises(ValueError, match="finite"):
        draw_strip(frame, strip, float("nan"))
    with pytest.raises(TypeError):
        draw_strip(frame, "strip", 0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="larger"):
        draw_strip(np.zeros((80, 100, 3), np.uint8), strip, 0.1)
    with pytest.raises(ValueError, match="width_px"):
        render_strip(_series(), 10, 90)
    with pytest.raises(ValueError, match="height_px"):
        render_strip(_series(), 600, 5)
    with pytest.raises(TypeError):
        render_strip("series", 600, 90)  # type: ignore[arg-type]


def test_renderer_renders_once_per_frame_size() -> None:
    renderer = StripRenderer(_series())
    frame = np.full((360, 640, 3), 40, np.uint8)
    renderer.draw(frame, 0.1)
    first = renderer._rendered[(360, 640)]
    renderer.draw(frame, 0.7)
    assert renderer._rendered[(360, 640)] is first
    renderer.draw(np.full((200, 320, 3), 40, np.uint8), 0.7)
    assert set(renderer._rendered) == {(360, 640), (200, 320)}


# --- export wiring ------------------------------------------------------------------


class _Writer:
    def __init__(self, sink: dict, path: Path) -> None:
        self.frames: list[np.ndarray] = []
        sink[path.name] = self

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        pass


class _Backend:
    engine = "fake"

    def unavailable_reason(self) -> None:
        return None

    def render(self, swing, settings, indices, overlay):
        for _k in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def _swing(steps: int = N - 1) -> SwingInput:
    names = ("A", "B")
    q = np.zeros((steps + 1, 2))
    bundle = InputBundle(
        spec_bytes=json.dumps({"coordinate_order": list(names)}).encode(),
        coordinate_order=names,
        dt_s=0.01,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )
    return SwingInput(bundle, q, "driver", "Driver", "mujoco")


def _export(tmp_path, feed, **overrides):
    settings = ExportSettings(
        views=("face_on",),
        multiview=False,
        width=640,
        height=360,
        speeds=(1.0, 0.5),
        impact_time_s=0.8,
        impact_card=False,
        ball=False,
        hud=False,
        **overrides,
    )
    sink: dict = {}
    result = export_swing(
        _Backend(),
        _swing(),
        settings,
        tmp_path,
        overlay=feed,
        writer_factory=lambda p, fps: _Writer(sink, p),
    )
    return result, sink


def test_export_draws_the_strip_on_every_frame_and_records_it(tmp_path) -> None:
    result, sink = _export(tmp_path, _feed(), inset_strip=True)
    assert len(sink) == 2  # 1x and 0.5x
    for writer in sink.values():
        assert writer.frames
        assert all((f != 40).any() for f in writer.frames)
        # the cursor advances: the first and last frames differ in the strip
        assert (writer.frames[0] != writer.frames[-1]).any()
    record = result.inset_strip
    assert isinstance(record, dict) and record["kind"] == "grf"
    assert record["samples"] == N and record["t_end_s"] == pytest.approx(1.0)
    assert [p["ylabel"] for p in record["panels"]][0] == "Vertical GRF (BW)"


def test_export_without_strip_is_unchanged(tmp_path) -> None:
    result, sink = _export(tmp_path, _feed())
    assert result.inset_strip is None
    for writer in sink.values():
        assert all((f == 40).all() for f in writer.frames)


def test_export_uses_the_grip_strip_with_the_grip_setting(tmp_path) -> None:
    feed = _feed(grip_analyses=lambda idx: [_analysis(25.0, 3.0) for _ in idx])
    result, _ = _export(tmp_path, feed, inset_strip=True, grip=True)
    assert result.inset_strip["kind"] == "grip"  # type: ignore[index]


def test_export_records_why_the_strip_is_unavailable(tmp_path) -> None:
    result, sink = _export(tmp_path, None, inset_strip=True)
    assert isinstance(result.inset_strip, str) and "overlays" in result.inset_strip
    assert all((f == 40).all() for w in sink.values() for f in w.frames)
    result, sink = _export(tmp_path, _feed(), inset_strip=True, grip=True)
    assert "not enabled" in str(result.inset_strip)
    assert all((f == 40).all() for w in sink.values() for f in w.frames)


def test_strip_is_computed_once_per_swing(tmp_path, monkeypatch) -> None:
    calls = []
    real = core.grf_series

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(core, "grf_series", counting)
    _export(tmp_path, _feed(), inset_strip=True)
    assert len(calls) == 1  # two speed clips share one series
