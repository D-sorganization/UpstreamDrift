"""Headless PyQt6 QTest launch-and-interact E2E for the Shadow Tracker workbench.

Drives the real ``python -m src.tools.shadow_tracker`` launch path and the review
widget with genuine ``QTest`` mouse and keyboard events (not direct handler calls):
bundle open on launch, frame scrubbing by button, slider and shortcut keys,
worst-frame triage, honest Auto-Fit refusal, and save/export through patched dialogs
(Refs #15522).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QFileDialog, QMainWindow

from src.shared.python.shadow_tracker.contracts import FrameObservation
from src.shared.python.shadow_tracker.mask_records import MaskFrame
from src.shared.python.shadow_tracker.service import DefaultShadowTrackerService
from src.shared.python.shadow_tracker.source_records import FrameIdentity, SourceAsset
from src.tools.shadow_tracker import __main__ as launcher
from src.tools.shadow_tracker.gui import ShadowTrackerWidget

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

FRAME_COUNT = 3
# Body-mask pixels set per frame (out of 16): frame-002 is the clear worst frame.
BODY_PIXELS = (16, 8, 0)


def _observation(i: int) -> FrameObservation:
    return FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id="shot-001",
        camera_id="cam-001",
        frame_id=f"frame-{i:03d}",
        pts_ticks=i * 33,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=i * 0.033,
        physical_time_reason="container_pts",
        body_mask_ref=f"mask-body-{i:03d}",
        club_mask_ref=f"mask-club-{i:03d}",
        valid_mask_ref=f"mask-valid-{i:03d}",
        confidence_provenance="ground_truth",
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="container_pts_metadata",
        decoder_name="opencv",
    )


def _mask(obs: FrameObservation, i: int) -> MaskFrame:
    return MaskFrame(
        schema_version="shadow-tracker/mask/1.0.0",
        frame=FrameIdentity(
            schema_version="shadow-tracker/frame/1.0.0",
            asset_id="asset-test",
            shot_id=obs.shot_id,
            swing_id="swing-001",
            camera_id=obs.camera_id,
            frame_id=obs.frame_id,
            pts_ticks=obs.pts_ticks,
            timebase_numerator=1,
            timebase_denominator=1000,
            physical_time_s=obs.physical_time_s,
            physical_time_reason="container_pts",
            frame_sha256="0" * 64,
            timing_mode="container_pts",
            is_timing_exact=True,
            clock_evidence="container_pts_metadata",
            decoder_name="opencv",
        ),
        width_px=4,
        height_px=4,
        body=bytes([1] * BODY_PIXELS[i] + [0] * (16 - BODY_PIXELS[i])),
        club=bytes([0] * 16),
        valid=bytes([1] * 16),
        revision_id=f"rev-{i}-0",
        parent_revision_id=None,
        producer_id="reviewer",
        correction_note="initial",
    )


def _initialize(service: DefaultShadowTrackerService) -> None:
    observations = [_observation(i) for i in range(FRAME_COUNT)]
    service.initialize_session(
        source_asset=SourceAsset(
            schema_version="shadow-tracker/source/1.0.0",
            asset_id="asset-test",
            source_uri="https://example.com/test.mp4",
            content_sha256="0" * 64,
            width_px=640,
            height_px=480,
            rights_status="permitted",
            rights_note="test asset",
        ),
        observations=observations,
        initial_masks=[_mask(o, i) for i, o in enumerate(observations)],
    )


@pytest.fixture(scope="module")
def qt_app() -> QApplication:
    return QApplication.instance() or QApplication([])  # type: ignore[return-value]


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """A saved three-frame review bundle the launcher can open with ``--bundle``."""
    service = DefaultShadowTrackerService()
    _initialize(service)
    target = tmp_path / "bundle"
    service.save_bundle(target)
    return target


@pytest.fixture
def workbench(qt_app: QApplication) -> Iterator[ShadowTrackerWidget]:
    """A shown, focused workbench holding a live three-frame session."""
    widget = ShadowTrackerWidget()
    _initialize(widget.model.service)
    widget._update_display()
    widget.resize(900, 600)
    widget.show()
    widget.setFocus()
    QTest.qWaitForWindowExposed(widget)
    try:
        yield widget
    finally:
        widget.cleanup()
        widget.close()


def _click(button: Any) -> None:
    QTest.mouseClick(button, Qt.MouseButton.LeftButton)


def _launch_and_interact(
    argv: list[str], interact: Callable[[QMainWindow, ShadowTrackerWidget], Any]
) -> Any:
    """Run the real launcher ``main`` with ``app.exec`` replaced by ``interact``."""
    captured: dict[str, Any] = {}

    def fake_exec(*_: Any) -> int:
        window = next(
            w for w in QApplication.topLevelWidgets() if isinstance(w, QMainWindow)
        )
        widget = window.centralWidget()
        assert isinstance(widget, ShadowTrackerWidget)
        QTest.qWaitForWindowExposed(window)
        captured["result"] = interact(window, widget)
        widget.cleanup()
        window.close()
        return 0

    with patch.object(QApplication, "exec", fake_exec):
        assert launcher.main(argv) == 0
    return captured["result"]


def test_launcher_opens_window_on_bundle_and_scrubs_with_buttons(
    qt_app: QApplication, bundle_dir: Path
) -> None:
    """Launch with --bundle, then click Next/Prev through the real window."""

    def interact(window: QMainWindow, widget: ShadowTrackerWidget) -> list[str]:
        seen = [window.windowTitle(), widget.lbl_frame_info.text()]
        assert window.isVisible()
        assert (widget.slider_frame.minimum(), widget.slider_frame.maximum()) == (
            0,
            FRAME_COUNT - 1,
        )
        _click(widget.btn_next)
        seen.append(widget.lbl_frame_info.text())
        _click(widget.btn_next)
        _click(widget.btn_next)  # clamped at the last frame
        seen.append(widget.lbl_frame_info.text())
        assert widget.viewport.observation is not None
        assert widget.viewport.observation.frame_id == "frame-002"
        _click(widget.btn_prev)
        seen.append(widget.lbl_frame_info.text())
        return seen

    seen = _launch_and_interact(["--bundle", str(bundle_dir)], interact)

    assert seen == [
        "Shadow Tracker Review Workbench",
        "Frame: 1 / 3",
        "Frame: 2 / 3",
        "Frame: 3 / 3",
        "Frame: 2 / 3",
    ]


def test_launcher_without_bundle_starts_empty_and_refuses_fit_honestly(
    qt_app: QApplication,
) -> None:
    """A bare launch is an empty review session; Auto-Fit says so instead of faking."""

    def interact(window: QMainWindow, widget: ShadowTrackerWidget) -> list[str]:
        states = [widget.lbl_frame_info.text(), widget.lbl_status.text()]
        _click(widget.btn_next)  # stepping an empty session must stay at 0 / 0
        states.append(widget.lbl_frame_info.text())
        _click(widget.btn_fit)
        states.append(widget.lbl_status.text())
        return states

    assert _launch_and_interact([], interact) == [
        "Frame: 0 / 0",
        "Status: Ready (Review Mode)",
        "Frame: 0 / 0",
        "Auto-Fit Unavailable: No observation loaded",
    ]


def test_slider_keys_drive_the_frame_and_viewport(
    workbench: ShadowTrackerWidget,
) -> None:
    """Keyboard focus on the slider scrubs frames and repaints the viewport."""
    slider = workbench.slider_frame
    slider.setFocus()
    QTest.keyClick(slider, Qt.Key.Key_End)
    assert workbench.model.current_frame_index == FRAME_COUNT - 1
    assert workbench.lbl_frame_info.text() == "Frame: 3 / 3"
    assert workbench.viewport.observation is not None
    assert workbench.viewport.observation.frame_id == "frame-002"

    QTest.keyClick(slider, Qt.Key.Key_Home)
    assert workbench.model.current_frame_index == 0
    assert workbench.viewport.observation is not None
    assert workbench.viewport.observation.frame_id == "frame-000"
    workbench.viewport.repaint()


def test_shortcut_keys_scrub_and_triage_worst_frame(
    workbench: ShadowTrackerWidget,
) -> None:
    """Real key events reach the workbench: arrows, Home/End, and W for worst frame."""
    QTest.keyClick(workbench, Qt.Key.Key_Right)
    assert workbench.model.current_frame_index == 1
    QTest.keyClick(workbench, Qt.Key.Key_Left)
    QTest.keyClick(workbench, Qt.Key.Key_Left)  # clamped at the first frame
    assert workbench.model.current_frame_index == 0
    QTest.keyClick(workbench, Qt.Key.Key_End)
    assert workbench.model.current_frame_index == FRAME_COUNT - 1
    QTest.keyClick(workbench, Qt.Key.Key_Home)
    assert workbench.model.current_frame_index == 0

    QTest.keyClick(workbench, Qt.Key.Key_W)
    assert workbench.model.current_frame_index == 2
    assert "frame-002" in workbench.lbl_status.text()
    assert workbench.lbl_frame_info.text() == "Frame: 3 / 3"


def test_worst_button_matches_the_w_shortcut(workbench: ShadowTrackerWidget) -> None:
    _click(workbench.btn_worst)
    assert workbench.model.current_frame_index == 2
    assert workbench.lbl_status.text().startswith("Triage: Worst frame frame-002")


def test_auto_fit_button_and_f_key_report_unqualified_backend(
    workbench: ShadowTrackerWidget,
) -> None:
    """With frames loaded, Auto-Fit refuses honestly via both click and the F key."""
    for trigger in (
        lambda: _click(workbench.btn_fit),
        lambda: QTest.keyClick(workbench, Qt.Key.Key_F),
    ):
        workbench.lbl_status.setText("")
        trigger()
        text = workbench.lbl_status.text()
        assert text.startswith("Auto-Fit Unavailable:")
        assert "unqualified" in text
    assert workbench.model.last_error_message


def test_ctrl_s_prompts_for_a_directory_then_saves_a_reloadable_bundle(
    workbench: ShadowTrackerWidget, tmp_path: Path, qt_app: QApplication
) -> None:
    """Ctrl+S on a fresh session asks for a folder, saves, and the bundle reopens."""
    target = tmp_path / "saved"
    with patch.object(
        QFileDialog, "getExistingDirectory", return_value=str(target)
    ) as dialog:
        QTest.keyClick(workbench, Qt.Key.Key_S, Qt.KeyboardModifier.ControlModifier)
    dialog.assert_called_once()
    assert workbench.lbl_status.text() == "Saved bundle to saved"
    assert workbench.model.bundle_path == target

    reopened = ShadowTrackerWidget()
    try:
        with patch.object(
            QFileDialog, "getExistingDirectory", return_value=str(target)
        ):
            _click(reopened.btn_open)
        assert reopened.lbl_status.text() == "Loaded bundle from saved"
        assert reopened.lbl_frame_info.text() == "Frame: 1 / 3"
    finally:
        reopened.cleanup()


def test_cancelled_dialogs_leave_the_session_untouched(
    workbench: ShadowTrackerWidget,
) -> None:
    """Dismissing the open, import and export dialogs changes nothing."""
    workbench.lbl_status.setText("sentinel")
    with (
        patch.object(QFileDialog, "getExistingDirectory", return_value=""),
        patch.object(QFileDialog, "getOpenFileName", return_value=("", "")),
        patch.object(QFileDialog, "getSaveFileName", return_value=("", "")),
    ):
        _click(workbench.btn_open)
        _click(workbench.btn_import)
        _click(workbench.btn_export)
    assert workbench.lbl_status.text() == "sentinel"
    assert workbench.model.frame_count == FRAME_COUNT
    assert workbench.model.current_frame_index == 0


def test_export_canonical_button_writes_json_package(
    workbench: ShadowTrackerWidget, tmp_path: Path
) -> None:
    target = tmp_path / "canonical.json"
    with patch.object(
        QFileDialog, "getSaveFileName", return_value=(str(target), "JSON")
    ):
        _click(workbench.btn_export)
    assert workbench.lbl_status.text() == "Exported canonical package to canonical.json"
    assert target.is_file()
    assert isinstance(json.loads(target.read_text(encoding="utf-8")), dict)


def test_import_button_reports_an_unreadable_video(
    workbench: ShadowTrackerWidget, tmp_path: Path
) -> None:
    """A bad video path surfaces as a status error rather than an exception."""
    missing = tmp_path / "missing.mp4"
    with patch.object(
        QFileDialog, "getOpenFileName", return_value=(str(missing), "Video")
    ):
        _click(workbench.btn_import)
    assert workbench.lbl_status.text().startswith("Error importing video:")
