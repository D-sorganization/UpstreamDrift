"""Pure core of the native viewer export tool (NV-5, #11678)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.force_overlay import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.golf_view_presets import VIEW_ORDER
from src.shared.python.motion_matching.same_input import InputBundle
from src.tools.native_viewer_export.core import (
    BackendUnavailable,
    ExportSettings,
    NativeBackend,
    OverlayFeed,
    SwingInput,
    export_swing,
    frame_indices,
    load_swing_input,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")


def _bundle(steps: int = 100) -> InputBundle:
    spec = {"coordinate_order": list(COORDS)}
    q = np.zeros((steps + 1, 2))
    return InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((steps, 2)),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
    )


def _swing(steps: int = 100) -> SwingInput:
    b = _bundle(steps)
    return SwingInput(b, b.reference_q, "driver", "Driver", "mujoco")


class _Writer:
    def __init__(self, path: Path, fps: int, sink: dict) -> None:
        self.path, self.fps, self.frames, self.closed = path, fps, [], False
        sink[path.name] = self

    def append_data(self, frame: np.ndarray) -> None:
        self.frames.append(frame)

    def close(self) -> None:
        self.closed = True


class _FakeBackend:
    engine = "fake"

    def __init__(self, reason: str | None = None) -> None:
        self.reason = reason
        self.calls: list[tuple[int, ...]] = []

    def unavailable_reason(self) -> str | None:
        return self.reason

    def render(self, swing, settings, indices, overlay):
        self.calls.append(tuple(indices))
        for _k in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def test_fake_backend_satisfies_protocol() -> None:
    assert isinstance(_FakeBackend(), NativeBackend)


def test_frame_indices_stride_and_final_state() -> None:
    assert frame_indices(101, 40) == [0, 40, 80, 100]
    assert frame_indices(101, 50) == [0, 50, 100]
    assert frame_indices(5, 100) == [0]
    with pytest.raises(ValueError):
        frame_indices(0, 10)


def test_settings_validation() -> None:
    assert ExportSettings().views == VIEW_ORDER
    with pytest.raises(ValueError, match="positive"):
        ExportSettings(width=0)
    with pytest.raises(ValueError, match="unknown view"):
        ExportSettings(views=("nope",), multiview=False)
    with pytest.raises(ValueError, match="four views"):
        ExportSettings(views=("face_on",))
    with pytest.raises(ValueError, match="unique"):
        ExportSettings(views=("face_on", "face_on"), multiview=False)


def test_swing_input_checks_rollout_shape() -> None:
    b = _bundle()
    with pytest.raises(ValueError, match="shape"):
        SwingInput(b, np.zeros((3, 2)), "d", "Driver", "x")
    bad = np.zeros_like(b.reference_q)
    bad[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SwingInput(b, bad, "d", "Driver", "x")


def test_load_swing_input_reads_bundle_and_rollout(tmp_path: Path) -> None:
    b = _bundle(10)
    b.save(tmp_path / "b.npz")
    np.savez(tmp_path / "r.npz", q=np.ones_like(b.reference_q))
    s = load_swing_input(
        tmp_path / "b.npz",
        tmp_path / "r.npz",
        swing="driver",
        club="Driver",
        rollout_engine="drake",
    )
    assert s.q.max() == 1.0 and s.rollout_engine == "drake"
    s2 = load_swing_input(tmp_path / "b.npz", swing="driver", club="Driver")
    assert s2.rollout_engine == "mujoco"
    np.savez(tmp_path / "bad.npz", other=np.zeros(3))
    with pytest.raises(ValueError, match="'q'"):
        load_swing_input(tmp_path / "b.npz", tmp_path / "bad.npz", swing="d", club="D")


def test_export_clip_set_and_geometry(tmp_path: Path) -> None:
    sink: dict = {}
    result = export_swing(
        _FakeBackend(),
        _swing(),
        ExportSettings(width=80, height=60, stride=40, fps=12),
        tmp_path,
        writer_factory=lambda p, fps: _Writer(p, fps, sink),
    )
    assert not result.skipped and result.frames == 4
    assert sorted(sink) == sorted(
        [f"driver_fake_{v}.mp4" for v in (*VIEW_ORDER, "2x2")]
    )
    assert all(w.closed and len(w.frames) == 4 and w.fps == 12 for w in sink.values())
    assert sink["driver_fake_face_on.mp4"].frames[0].shape == (60, 80, 3)
    assert sink["driver_fake_2x2.mp4"].frames[0].shape == (120, 160, 3)
    # label and HUD pixels were drawn on top of the flat 40-grey tile
    assert sink["driver_fake_face_on.mp4"].frames[0].max() == 255


def test_export_without_multiview_or_overlay(tmp_path: Path) -> None:
    sink: dict = {}
    result = export_swing(
        _FakeBackend(),
        _swing(),
        ExportSettings(views=("overhead",), multiview=False, width=64, height=48),
        tmp_path,
        writer_factory=lambda p, fps: _Writer(p, fps, sink),
    )
    assert list(sink) == ["driver_fake_overhead.mp4"]
    assert result.glyph_counts == ()


def test_unavailable_backend_skips_cleanly_and_writes_nothing(tmp_path: Path) -> None:
    sink: dict = {}
    result = export_swing(
        _FakeBackend("playwright missing"),
        _swing(),
        ExportSettings(),
        tmp_path / "out",
        writer_factory=lambda p, fps: _Writer(p, fps, sink),
    )
    assert result.skipped and result.skipped_reason == "playwright missing"
    assert not sink and not (tmp_path / "out").exists()
    assert issubclass(BackendUnavailable, RuntimeError)


def test_overlay_feed_counts_glyphs_and_legend_in_hud(tmp_path: Path) -> None:
    def frame_at(k: int) -> ForceTorqueFrame:
        return ForceTorqueFrame(
            time_s=k * 0.001,
            engine="fake",
            wrenches=(
                OverlayWrench(
                    WrenchKind.CONTACT,
                    "contact:grf_r",
                    "calcn_r",
                    (0.0, 0.0, 0.0),
                    force_n=(0.0, 0.0, 700.0),
                    source="fake",
                ),
            ),
        )

    sink: dict = {}
    result = export_swing(
        _FakeBackend(),
        _swing(),
        ExportSettings(width=320, height=240, stride=50),
        tmp_path,
        overlay=OverlayFeed(frame_at),
        writer_factory=lambda p, fps: _Writer(p, fps, sink),
    )
    assert result.glyph_counts == (1, 1, 1)
    assert result.frames == 3


def test_short_backend_is_an_error(tmp_path: Path) -> None:
    class _Short(_FakeBackend):
        def render(self, swing, settings, indices, overlay):
            yield from super().render(swing, settings, indices[:1], overlay)

    with pytest.raises(RuntimeError, match="expected"):
        export_swing(
            _Short(),
            _swing(),
            ExportSettings(width=32, height=32),
            tmp_path,
            writer_factory=lambda p, fps: _Writer(p, fps, {}),
        )
