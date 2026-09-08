"""Live camera preview panel, driven by the rig's synthetic frame sources."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication

from src.motion_capture.rig.binding import locate_plan
from src.motion_capture.rig.plan import RigPlan
from src.motion_capture.rig.sources import SyntheticFrameSource
from src.motion_capture.rig.topology import CameraLocation
from src.tools.capture_rig import gui
from src.tools.capture_rig.commands import PlanSelection
from src.tools.capture_rig.preview import PreviewPanel

pytestmark = [pytest.mark.unit, pytest.mark.ui]

_APP: QApplication | None = None


def _app() -> QApplication:
    global _APP
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv[:1])
    _APP = app  # type: ignore[assignment]
    return _APP  # type: ignore[return-value]


def _plan(tmp_path: Path) -> Path:
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


def _synthetic(plan: RigPlan) -> dict[str, SyntheticFrameSource]:
    return {c.view: SyntheticFrameSource(c.identity) for c in plan.cameras}


def _pump(app: QApplication, seconds: float) -> None:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        time.sleep(0.01)


def test_preview_streams_every_planned_view_and_releases_on_stop(
    tmp_path: Path,
) -> None:
    app = _app()
    panel = PreviewPanel(source_factory=_synthetic)
    assert not panel.active and panel.status.text() == "preview off"
    panel.start(PlanSelection(plan=_plan(tmp_path)))
    assert panel.active and panel.views() == ("cam_a", "cam_b")
    _pump(app, 1.5)  # binder thread, then the workers
    assert panel.frames_seen("cam_a") > 0 and panel.frames_seen("cam_b") > 0
    assert panel.status.text().startswith("live:")
    panel.stop()
    assert not panel.active and "released" in panel.status.text()
    # Restricting the views previews only those.
    panel.start(PlanSelection(plan=_plan(tmp_path), views=("cam_b",)))
    assert panel.views() == ("cam_b",)
    _pump(app, 0.3)
    panel.stop()
    with pytest.raises(Exception, match="plan file must exist"):
        panel.start(PlanSelection(plan=tmp_path / "missing.json"))


def test_preview_rebinds_from_remembered_ids_and_falls_back_when_they_fail(
    tmp_path: Path,
) -> None:
    app = _app()

    class Real(SyntheticFrameSource):
        camera_instance_id = "USB\\X\\1&0&0000"

    full_calls: list[str] = []
    fast_calls: list[dict[str, str]] = []

    def full(plan: RigPlan) -> dict[str, SyntheticFrameSource]:
        full_calls.append("full")
        return {c.view: Real(c.identity) for c in plan.cameras}

    fail_fast = {"on": False}

    def fast(plan: RigPlan, ids: dict[str, str]) -> dict[str, SyntheticFrameSource]:
        fast_calls.append(dict(ids))
        if fail_fast["on"]:
            raise ValueError("camera unplugged")
        return {c.view: Real(c.identity) for c in plan.cameras}

    panel = PreviewPanel(source_factory=full, fast_factory=fast)
    panel.start(PlanSelection(plan=_plan(tmp_path)))
    _pump(app, 0.6)
    assert full_calls == ["full"] and fast_calls == []
    assert panel.camera_ids() == {
        "cam_a": "USB\\X\\1&0&0000",
        "cam_b": "USB\\X\\1&0&0000",
    }
    panel.stop()
    panel.start(PlanSelection(plan=_plan(tmp_path)))  # second time: no enumeration
    assert "bound earlier" in panel.status.text()
    _pump(app, 0.6)
    assert full_calls == ["full"] and len(fast_calls) == 1
    assert all(panel.frames_seen(v) for v in panel.views())
    panel.stop()
    fail_fast["on"] = True  # the remembered cameras vanished
    panel.start(PlanSelection(plan=_plan(tmp_path)))
    _pump(app, 0.8)
    assert full_calls == ["full", "full"] and len(fast_calls) == 2
    assert all(panel.frames_seen(v) for v in panel.views())
    panel.stop()


def test_preview_drops_frames_from_a_previous_generation(tmp_path: Path) -> None:
    app = _app()
    panel = PreviewPanel(source_factory=_synthetic)
    panel.start(PlanSelection(plan=_plan(tmp_path)))
    _pump(app, 0.6)
    old = panel._generation
    panel.stop()
    live = tmp_path / ".live"
    live.mkdir()
    panel.watch_snapshots(live, ("cam_a",))
    import numpy as np

    stale = np.zeros((10, 10, 3), dtype=np.uint8)
    panel._on_frame("cam_a", stale, old)  # late delivery from a stopped worker
    assert panel.frames_seen("cam_a") == 0
    panel._on_frame("cam_a", stale)  # the snapshot path carries no generation
    assert panel.frames_seen("cam_a") == 1
    panel.stop_watching()


def test_preview_reports_an_unrealizable_plan_instead_of_raising(
    tmp_path: Path,
) -> None:
    _app()

    def refuse(plan: RigPlan) -> dict[str, SyntheticFrameSource]:
        raise ValueError("plan not realizable: missing=['cam_b']")

    panel = PreviewPanel(source_factory=refuse)
    panel.start(PlanSelection(plan=_plan(tmp_path)))
    _pump(_app(), 0.5)
    assert not panel.active and "not realizable" in panel.status.text()


def test_locate_plan_binds_views_to_cameras_with_indices(tmp_path: Path) -> None:
    plan = RigPlan.load(_plan(tmp_path))

    def cam(name: str, serial: str, index: int) -> CameraLocation:
        return CameraLocation(
            camera=name,
            composite=None,
            serial=serial,
            identity=serial,
            root_hub=f"9&{index}",
            root_port=1,
            host=None,
            hub_depth=1,
            index=index,
        )

    cams = [cam("A", "1", 0), cam("B", "2", 1)]
    located = locate_plan(plan, cams)
    assert {v: c.index for v, c in located.items()} == {"cam_a": 0, "cam_b": 1}
    with pytest.raises(ValueError, match="not realizable"):
        locate_plan(plan, cams[:1])


def test_tile_prefills_the_lab_plan_and_a_fresh_session_and_toggles_preview() -> None:
    _app()
    widget = gui.CaptureRigWidget()
    assert widget.capture.plan_edit.text().endswith("lab_three_view_sonnet.json")
    assert "sessions" in widget.capture.session_edit.text()
    assert "preview" in widget.enabled_actions()
    widget.preview._factory = _synthetic  # no cameras on the test host
    widget.toggle_preview(on=True)
    assert widget.preview.active and widget.buttons["preview"].text() == "Stop preview"
    _pump(_app(), 0.5)
    widget.toggle_preview(on=False)
    assert not widget.preview.active
    assert widget.buttons["preview"].text() == "Preview cameras"
    widget.shutdown()
