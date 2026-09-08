"""Live preview (#9813) and playback (#9814) rendered through a layout.

Both panes composite through the one :func:`layout_model.compose`, so these
tests drive them with deterministic sources and read the composed canvas
back: each view lands in its own cell, a rotation in the spec rotates the
live tile, the recorder-snapshot mode composites too (badge and all), a
layout switch re-renders without re-opening a reader, playback puts frame
*k* of every view on one canvas, an overlay tile draws the pose, the canvas
can be written out as a PNG, and the chosen layout name survives a restart.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")
pytest.importorskip("cv2")

from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication

from src.motion_capture.rig.bundle import build_index, write_bundle
from src.motion_capture.rig.plan import CameraBinding, CaptureMode, RigPlan
from src.motion_capture.rig.probe import RecordingProbe
from src.motion_capture.rig.recorder import RecordingResult
from src.motion_capture.rig.sources import Frame
from src.tools.capture_rig import gui, multiview
from src.tools.capture_rig.commands import PlanSelection
from src.tools.capture_rig.layout_model import Cell, LayoutSpec, SourceRef, Tile, preset
from src.tools.capture_rig.layout_presets import USER, LayoutStore
from src.tools.capture_rig.playback import PlaybackPanel
from src.tools.capture_rig.preview import PreviewPanel
from src.tools.capture_rig.session import load_session

pytestmark = [pytest.mark.unit, pytest.mark.ui]

VIDEO_SIZE = (320, 200)
PANEL_SIZE = (480, 360)
BLACK = (0, 0, 0)
#: BGR, chosen so one channel alone separates the two views.
COLOURS = {"cam_a": (220, 20, 20), "cam_b": (20, 220, 20)}

_APP: QApplication | None = None  # held at module level (a dropped app aborts)


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _pump(seconds: float) -> None:
    app = _app()
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def _store(tmp_path: Path) -> LayoutStore:
    """A layout store rooted in the test's own directory."""
    return LayoutStore(user_root=tmp_path / "layouts")


def _dark(spec: LayoutSpec) -> LayoutSpec:
    """The spec with a black background, so letterboxing is unambiguous."""
    return replace(spec, background=BLACK)


def _channel(region: np.ndarray, index: int) -> float:
    return float(region[:, :, index].mean())


def _centre(canvas: np.ndarray, rows: slice, cols: slice) -> np.ndarray:
    """The middle of a cell: away from the caption box and the letterbox."""
    return canvas[rows, cols]


# -- live preview ------------------------------------------------------------
class ColourSource:
    """A frame source that always yields one flat BGR colour."""

    def __init__(self, identity: str, colour: tuple[int, int, int]) -> None:
        self._identity, self._colour = identity, colour
        self._seq = 0

    @property
    def identity(self) -> str:
        return self._identity

    def open(self, mode: CaptureMode, controls: Any = None) -> CaptureMode:
        return mode

    def read(self) -> Frame:
        time.sleep(0.005)  # a real camera paces itself; do not spin the worker
        self._seq += 1
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :] = self._colour
        return Frame(image=image, seq=self._seq, t_ns=self._seq)

    def close(self) -> None:
        return None


def _plan_file(tmp_path: Path) -> Path:
    path = tmp_path / "plan.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "rig-plan/1.0.0",
                "name": "two",
                "cameras": [
                    {"view": "cam_a", "serial": "1"},
                    {"view": "cam_b", "serial": "2"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _colour_sources(plan: RigPlan) -> dict[str, ColourSource]:
    return {c.view: ColourSource(c.identity, COLOURS[c.view]) for c in plan.cameras}


def _live_panel(tmp_path: Path) -> PreviewPanel:
    _app()  # the application must exist before the first widget
    panel = PreviewPanel(source_factory=_colour_sources, layout_store=_store(tmp_path))
    panel.resize(*PANEL_SIZE)
    panel.show()
    _pump(0.2)
    return panel


def _live_canvas(panel: PreviewPanel) -> np.ndarray:
    """The composited canvas, checked to be exactly the canvas widget's size."""
    canvas = panel.canvas_frame()
    assert canvas is not None
    assert canvas.shape == (panel.canvas.height(), panel.canvas.width(), 3)
    return canvas


def test_live_preview_composites_each_view_into_its_own_cell(tmp_path: Path) -> None:
    panel = _live_panel(tmp_path)
    panel.start(PlanSelection(plan=_plan_file(tmp_path)))
    _pump(1.5)
    assert panel.views() == ("cam_a", "cam_b")
    assert all(panel.frames_seen(v) for v in panel.views())
    panel.chooser.set_spec(
        _dark(
            LayoutSpec(
                name="grid",
                rows=2,
                cols=2,
                tiles=(
                    Tile(SourceRef("live", "cam_a"), Cell(0, 0), show_label=False),
                    Tile(SourceRef("live", "cam_b"), Cell(1, 1), show_label=False),
                ),
            )
        )
    )
    canvas = _live_canvas(panel)
    h, w = canvas.shape[0] // 2, canvas.shape[1] // 2
    top_left = _centre(canvas, slice(h // 4, h - h // 4), slice(w // 3, w - w // 3))
    bottom_right = _centre(
        canvas, slice(h + h // 4, -h // 4), slice(w + w // 3, -w // 3)
    )
    empty = _centre(canvas, slice(h // 4, h - h // 4), slice(w + w // 3, -w // 3))
    assert _channel(top_left, 0) > 150 and _channel(top_left, 1) < 80  # cam_a is blue
    assert _channel(bottom_right, 1) > 150 and _channel(bottom_right, 0) < 80
    assert _channel(empty, 0) < 150 and _channel(empty, 1) < 150  # a placeholder
    panel.stop()
    panel.close()


def test_a_view_outside_the_layout_is_still_captured(tmp_path: Path) -> None:
    panel = _live_panel(tmp_path)
    panel.start(PlanSelection(plan=_plan_file(tmp_path)))
    _pump(1.2)
    panel.chooser.set_spec(preset("single", (SourceRef("live", "cam_a"),)))
    before = panel.frames_seen("cam_b")
    _pump(0.5)
    assert panel.views() == ("cam_a", "cam_b")  # cam_b is hidden, not released
    assert panel.frames_seen("cam_b") > before
    panel.stop()
    panel.close()


def _blue_centroid(canvas: np.ndarray) -> tuple[float, float]:
    """``(mean row, mean col)`` of the pixels carrying cam_a's blue."""
    ys, xs = np.nonzero(canvas[:, :, 0] > 120)
    assert ys.size, "no cam_a pixels on the canvas"
    return float(ys.mean()), float(xs.mean())


def test_rotation_in_the_spec_rotates_the_live_tile(tmp_path: Path) -> None:
    panel = _live_panel(tmp_path)
    live = tmp_path / ".live"
    live.mkdir()
    panel.watch_snapshots(live, ("cam_a",))
    portrait = np.zeros((40, 20, 3), dtype=np.uint8)
    portrait[:20, :] = COLOURS["cam_a"]  # bright TOP half of a portrait frame
    upright = _dark(preset("single", (SourceRef("live", "cam_a"),)))
    panel.chooser.set_spec(
        replace(upright, tiles=(replace(upright.tiles[0], show_label=False),))
    )
    panel._on_frame("cam_a", portrait)
    before = _blue_centroid(_live_canvas(panel))
    turned = replace(
        panel.chooser.spec(),
        tiles=(replace(panel.chooser.spec().tiles[0], rotation=90),),
    )
    panel.chooser.set_spec(turned)
    after = _blue_centroid(_live_canvas(panel))
    height = panel.canvas.height()
    assert before[0] < height / 2  # upright: the colour sits in the top half
    assert after[0] > before[0]  # rotated 90 CW: it is now centred vertically
    assert after[1] > before[1]  # ... and pushed to the right
    panel.stop_watching()
    panel.close()


def test_snapshot_mode_composites_and_keeps_the_badge(tmp_path: Path) -> None:
    import cv2

    live = tmp_path / ".live"
    live.mkdir()
    for view, colour in COLOURS.items():
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        image[:, :] = colour
        cv2.imwrite(str(live / f"{view}.jpg"), image)
    panel = _live_panel(tmp_path)
    panel.set_badge("REC 00:03")
    panel.watch_snapshots(live, ("cam_a", "cam_b"))
    assert panel.watching and not panel.active  # no camera is held during a take
    panel.chooser.set_spec(
        _dark(
            preset(
                "side_by_side",
                (SourceRef("live", "cam_a"), SourceRef("live", "cam_b")),
            )
        )
    )
    panel.poll_snapshots()
    canvas = _live_canvas(panel)
    half = canvas.shape[1] // 2
    assert _channel(canvas[:, :half], 0) > _channel(canvas[:, half:], 0)
    assert _channel(canvas[:, half:], 1) > _channel(canvas[:, :half], 1)
    assert "recording" in panel.status.text() and panel.badge == "REC 00:03"
    pixmap = panel.canvas.pixmap()  # the badge is painted onto the pixmap
    assert pixmap is not None and not pixmap.isNull()
    panel.stop_watching()
    panel.close()


def test_layout_choice_round_trips_through_qsettings(tmp_path: Path) -> None:
    ini = tmp_path / "settings.ini"

    def settings() -> QSettings:
        return QSettings(str(ini), QSettings.Format.IniFormat)

    widget = gui.CaptureRigWidget(settings=settings())
    assert widget.preview.set_layout_name("two_by_two")
    assert widget.preview.layout_name() == "two_by_two"
    assert widget.playback.set_layout_name("three_across")
    assert widget.pane_extras()["playback_layout"] == "three_across"
    widget.shutdown()  # saves the arrangement and the panes' layout names
    again = gui.CaptureRigWidget(settings=settings())
    assert again.preview.layout_name() == "two_by_two"
    assert again.playback.layout_name() == "three_across"
    again.shutdown()


def test_chooser_offers_saved_layouts_and_opens_the_editor(tmp_path: Path) -> None:
    _app()
    store = _store(tmp_path)
    store.user_root.mkdir(parents=True)
    store.save("bay", preset("two_by_two", (SourceRef("live", "cam_a"),)), USER)
    chooser = multiview.LayoutChooser(
        options=multiview.ChooserOptions(store=store),
        sources=(SourceRef("live", "cam_a"),),
    )
    assert any(e.name == "bay" and e.scope == USER for e in chooser.entries())
    seen: list[LayoutSpec] = []
    chooser.layout_changed.connect(lambda spec: seen.append(spec))
    assert chooser.set_layout_name("bay")
    assert chooser.layout_name() == "bay" and seen[-1].rows == 2
    assert not chooser.set_layout_name("no_such_layout")
    editor = chooser.open_editor()
    assert chooser.open_editor() is editor  # one window, re-used
    editor.set_grid(1, 2)
    editor.rotate(0, 1)
    assert chooser.layout_name() == multiview.EDITED
    assert chooser.spec().tiles[0].rotation == 90
    assert chooser.combo.toolTip() and chooser.edit_button.toolTip()
    editor.close()
    chooser.close()


# -- playback ----------------------------------------------------------------
def _write_video(path: Path, colour: tuple[int, int, int], frames: int) -> None:
    import cv2

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, VIDEO_SIZE
    )
    for i in range(frames):
        img = np.zeros((VIDEO_SIZE[1], VIDEO_SIZE[0], 3), dtype=np.uint8)
        img[:, :] = colour
        img[: 12 * (i + 1), :60] = (255, 255, 255)  # a bar that grows with the index
        writer.write(img)
    writer.release()


def _two_view_bundle(tmp_path: Path) -> Path:
    plan = RigPlan(
        name="rig",
        cameras=(
            CameraBinding(view="cam_a", serial="1"),
            CameraBinding(view="cam_b", serial="2"),
        ),
    )
    results = []
    for view, serial in (("cam_a", "1"), ("cam_b", "2")):
        video = tmp_path / f"{view}_{serial}.avi"
        _write_video(video, COLOURS[view], 12)
        results.append(RecordingResult(serial, video, 0, video.stat().st_size))
    index = build_index(
        plan,
        results,
        1.2,
        tmp_path,
        prober=lambda p: RecordingProbe(12, 1.2, VIDEO_SIZE[0], VIDEO_SIZE[1], 10.0),
    )
    write_bundle(tmp_path, plan, index, started_utc="2026-09-07T00:00:00+00:00")
    return tmp_path


def _pose(root: Path) -> None:
    """Observations for cam_a with a pose on frames 0, 1 and 3."""
    names = ["nose", "left_shoulder", "right_shoulder", "left_elbow"]
    rows = [
        {
            "camera_id": "1",
            "time_s": i / 10.0,
            "keypoints_px": [[160, 40], [120, 90], [200, 90], [110, 150]],
            "confidence": [0.95, 0.95, 0.95, 0.95],
        }
        for i in (0, 1, 3)
    ]
    obs = root / "observations"
    obs.mkdir()
    (obs / "cam_a.json").write_text(
        json.dumps(
            {
                "view": "cam_a",
                "identity": "1",
                "camera_id": "1",
                "fps": 10.0,
                "width": VIDEO_SIZE[0],
                "height": VIDEO_SIZE[1],
                "frames_total": 12,
                "frames_with_pose": 3,
                "detector_layout": {"name": "test", "keypoint_names": names},
                "frames": rows,
                "provenance": {"estimator": "mediapipe"},
            }
        ),
        encoding="utf-8",
    )
    (obs / "observations.json").write_text(
        json.dumps(
            {
                "plan_name": "rig",
                "views": [
                    {
                        "view": "cam_a",
                        "identity": "1",
                        "status": "available",
                        "file": "cam_a.json",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _playback(root: Path, tmp_path: Path) -> PlaybackPanel:
    _app()  # the application must exist before the first widget
    panel = PlaybackPanel(layout_store=_store(tmp_path))
    panel.resize(*PANEL_SIZE)
    panel.show()
    _pump(0.2)
    panel.load(load_session(root))
    return panel


def _playback_canvas(panel: PlaybackPanel) -> np.ndarray:
    canvas = panel.canvas_frame()
    assert canvas is not None
    assert canvas.shape == (panel.image.height(), panel.image.width(), 3)
    return canvas


def _white_pixels(canvas: np.ndarray) -> int:
    return int((canvas.min(axis=2) > 200).sum())


def test_playback_puts_frame_k_of_both_views_on_one_canvas(tmp_path: Path) -> None:
    panel = _playback(_two_view_bundle(tmp_path), tmp_path)
    assert panel.offsets() == {"cam_a": 0, "cam_b": 0}  # no timing block in this take
    panel.chooser.set_spec(
        _dark(
            preset(
                "side_by_side",
                (SourceRef("recorded", "cam_a"), SourceRef("recorded", "cam_b")),
            )
        )
    )
    panel.show_frame(4)
    canvas = _playback_canvas(panel)
    half = canvas.shape[1] // 2
    left, right = canvas[:, :half], canvas[:, half:]
    assert _channel(left, 0) > _channel(right, 0)  # cam_a: blue-dominant
    assert _channel(right, 1) > _channel(left, 1)  # cam_b: green-dominant
    tall = _white_pixels(canvas)  # frame k, not frame 0: the bar grows with k
    panel.show_frame(0)
    assert tall > _white_pixels(_playback_canvas(panel))
    panel.close_media()
    panel.close()


def test_overlay_tile_draws_the_pose_and_the_raw_tile_does_not(
    tmp_path: Path,
) -> None:
    root = _two_view_bundle(tmp_path)
    _pose(root)
    panel = _playback(root, tmp_path)
    raw = panel._source_frame(SourceRef("recorded", "cam_a"), 0)
    drawn = panel._source_frame(SourceRef("overlay", "cam_a"), 0)
    assert raw is not None and drawn is not None
    assert not np.array_equal(raw, drawn)  # the pose is drawn on the overlay tile
    panel.overlay_check.setChecked(False)
    plain = panel._source_frame(SourceRef("overlay", "cam_a"), 0)
    assert plain is not None and np.array_equal(raw, plain)
    panel.overlay_check.setChecked(True)
    panel.show_frame(0)
    assert panel.status.text().endswith("· pose")
    panel.show_frame(2)  # no pose on frame 2
    assert panel.status.text().endswith("· no pose")
    # Raw and overlay of the same view can sit side by side.
    panel.chooser.set_spec(
        _dark(
            preset(
                "side_by_side",
                (SourceRef("overlay", "cam_a"), SourceRef("recorded", "cam_a")),
            )
        )
    )
    panel.show_frame(0)
    canvas = _playback_canvas(panel)
    half = canvas.shape[1] // 2
    assert not np.array_equal(canvas[:, :half], canvas[:, half : 2 * half])
    panel.close_media()
    panel.close()


def test_layout_switch_re_renders_without_reopening_readers(tmp_path: Path) -> None:
    panel = _playback(_two_view_bundle(tmp_path), tmp_path)
    panel.show_frame(3)
    opened = dict(panel._readers)
    assert set(opened) == {"cam_a"}  # only what the one-tile default asked for
    panel.chooser.set_spec(
        _dark(
            preset(
                "side_by_side",
                (SourceRef("recorded", "cam_a"), SourceRef("recorded", "cam_b")),
            )
        )
    )
    assert set(panel._readers) == {"cam_a", "cam_b"}
    assert panel._readers["cam_a"] is opened["cam_a"]  # not re-opened
    first = _playback_canvas(panel).copy()
    panel.chooser.set_spec(_dark(preset("single", (SourceRef("recorded", "cam_b"),))))
    assert panel._readers["cam_a"] is opened["cam_a"]
    assert not np.array_equal(first, _playback_canvas(panel))
    assert panel.frame_index == 3  # switching layout does not move the playhead
    panel.close_media()
    panel.close()


def test_transport_speed_step_and_png_export(tmp_path: Path) -> None:
    panel = _playback(_two_view_bundle(tmp_path), tmp_path)
    panel.show_frame(2)
    panel.step_by(1)
    assert panel.frame_index == 3
    panel.step_by(-2)
    assert panel.frame_index == 1
    panel.step_by(-5)
    assert panel.frame_index == 0  # clamped, never negative
    before = panel._timer.interval()
    panel.speed_spin.setValue(2.0)
    assert panel._timer.interval() < before
    out = panel.export_png(tmp_path / "out" / "canvas.png")
    assert out.is_file() and out.stat().st_size > 0
    import cv2

    written = cv2.imread(str(out))
    assert written is not None and np.array_equal(written, panel.canvas_frame())
    for widget in (
        panel.play_button,
        panel.step_back_button,
        panel.step_forward_button,
        panel.speed_spin,
        panel.export_button,
        panel.view_combo,
        panel.set_combo,
        panel.overlay_check,
        panel.confidence_spin,
    ):
        assert widget.toolTip(), widget
    panel.close_media()
    panel.close()


def test_export_png_refuses_before_anything_is_played(tmp_path: Path) -> None:
    _app()
    panel = PlaybackPanel(layout_store=_store(tmp_path))
    with pytest.raises(Exception, match="nothing has been played back"):
        panel.export_png(tmp_path / "nope.png")
    panel.close()


# -- shared helpers ----------------------------------------------------------
def test_frame_offsets_turn_a_timing_block_into_whole_frames() -> None:
    timing = {
        "reference_view": "cam_a",
        "views": [
            {"view": "cam_a", "offset_ns": 0},
            {"view": "cam_b", "offset_ns": 100_000_000},  # 0.1 s later
            {"view": "cam_c", "offset_ns": None},
        ],
    }
    rates = {"cam_a": 30.0, "cam_b": 30.0, "cam_c": 30.0, "cam_d": 30.0}
    assert multiview.frame_offsets(timing, rates) == {
        "cam_a": 0,
        "cam_b": 3,
        "cam_c": 0,
        "cam_d": 0,  # a view the block does not mention
    }
    assert multiview.frame_offsets({}, rates) == dict.fromkeys(rates, 0)


def test_theme_palette_survives_a_palette_value_that_is_not_hex() -> None:
    palette = multiview.theme_palette({"bg": "#102030", "text": "transparent"})
    assert palette.background == (0x30, 0x20, 0x10)  # BGR
    assert palette.label_text == multiview.Palette().label_text  # kept the default


def test_write_png_round_trips_a_canvas(tmp_path: Path) -> None:
    import cv2

    canvas = np.zeros((8, 12, 3), dtype=np.uint8)
    canvas[:, :] = (10, 20, 30)
    out = multiview.write_png(canvas, tmp_path / "deep" / "c.png")
    assert np.array_equal(cv2.imread(str(out)), canvas)
    with pytest.raises(Exception, match="canvas must be HxWx3"):
        multiview.write_png(np.zeros((4, 4), dtype=np.uint8), tmp_path / "bad.png")
