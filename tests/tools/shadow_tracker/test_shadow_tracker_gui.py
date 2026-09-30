"""Unit tests for Shadow Tracker GUI workbench (MS-84, #10357)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from unittest.mock import patch
import pytest

pytest.importorskip("PyQt6", reason="the workbench shell needs a Qt binding")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets

from src.shared.python.shadow_tracker.contracts import FrameObservation
from src.tools.shadow_tracker.gui import (
    ShadowTrackerViewportWidget,
    ShadowTrackerWidget,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _dummy_observation() -> FrameObservation:
    return FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id="shot-001",
        camera_id="cam-001",
        frame_id="frame-001",
        pts_ticks=33,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=0.033,
        physical_time_reason="exact_sync_address",
        body_mask_ref="mask-body-001",
        club_mask_ref="mask-club-001",
        valid_mask_ref="mask-valid-001",
        confidence_provenance="ground_truth",
        timing_mode="exact_container_pts",
        is_timing_exact=True,
        clock_evidence="hardware_sync",
        decoder_name="opencv",
    )


def test_buttons_have_connected_receivers(qapp: Any) -> None:
    """Each toolbar button must have at least one connected receiver (MS-84)."""
    widget = ShadowTrackerWidget()
    try:
        assert widget.btn_open.receivers(widget.btn_open.clicked) > 0
        assert widget.btn_save.receivers(widget.btn_save.clicked) > 0
        assert widget.btn_export.receivers(widget.btn_export.clicked) > 0
        assert widget.btn_worst.receivers(widget.btn_worst.clicked) > 0
        assert widget.btn_prev.receivers(widget.btn_prev.clicked) > 0
        assert widget.btn_next.receivers(widget.btn_next.clicked) > 0
    finally:
        widget.cleanup()


def test_viewport_widget_type_is_not_qlabel(qapp: Any) -> None:
    """The viewport widget must not be a plain QLabel (MS-84)."""
    widget = ShadowTrackerWidget()
    try:
        assert not isinstance(widget.viewport, QtWidgets.QLabel)
        assert isinstance(widget.viewport, ShadowTrackerViewportWidget)
    finally:
        widget.cleanup()


def test_viewport_widget_rendering_empty_and_active(qapp: Any) -> None:
    """Viewport widget paints cleanly both without observation and with an observation."""
    widget = ShadowTrackerWidget()
    try:
        assert widget.viewport.observation is None
        widget.viewport.repaint()

        obs = _dummy_observation()
        widget.viewport.set_observation(obs)
        assert widget.viewport.observation == obs
        widget.viewport.repaint()
    finally:
        widget.cleanup()


def test_open_save_export_button_handlers(qapp: Any, tmp_path: Path) -> None:
    """Toolbar button handlers open, save, and export using review model APIs."""
    widget = ShadowTrackerWidget()
    try:
        with patch.object(
            QtWidgets.QFileDialog, "getExistingDirectory", return_value=str(tmp_path)
        ):
            widget._on_open_bundle()
            assert (
                "bundle" in widget.lbl_status.text().lower()
                or "ready" in widget.lbl_status.text().lower()
            )

        with patch.object(
            widget, "save_bundle", return_value=tmp_path / "saved_bundle"
        ):
            widget._on_save_bundle()
            assert "saved" in widget.lbl_status.text().lower()

        export_target = tmp_path / "export.json"
        with (
            patch.object(
                QtWidgets.QFileDialog,
                "getSaveFileName",
                return_value=(str(export_target), "JSON"),
            ),
            patch.object(
                widget, "export_canonical", return_value={"schema_version": "1.0.0"}
            ),
        ):
            widget._on_export_canonical()
            assert "exported" in widget.lbl_status.text().lower()
    finally:
        widget.cleanup()


def test_gui_keyboard_navigation_and_review_workflow(qapp: Any, tmp_path: Path) -> None:
    """Installed PyQt review journey: keyboard scrubbing, import, mask update, honest auto-fit refusal (MMR-12)."""
    from PyQt6 import QtCore, QtGui
    from src.shared.python.shadow_tracker.contracts import FitRequest
    from src.shared.python.shadow_tracker.mask_records import MaskFrame
    from src.shared.python.shadow_tracker.source_records import (
        FrameIdentity,
        SourceAsset,
    )

    widget = ShadowTrackerWidget()
    try:
        # 1. Import button exists and has connected receiver
        assert hasattr(widget, "btn_import")
        assert widget.btn_import.receivers(widget.btn_import.clicked) > 0

        # Set up a 3-frame session
        obs_list = []
        for i in range(3):
            obs_list.append(
                FrameObservation(
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
            )

        masks = []
        for i, obs in enumerate(obs_list):
            masks.append(
                MaskFrame(
                    schema_version="shadow-tracker/mask/1.0.0",
                    frame=FrameIdentity(
                        schema_version="shadow-tracker/frame/1.0.0",
                        asset_id="asset-test",
                        shot_id="shot-001",
                        swing_id="swing-001",
                        camera_id="cam-001",
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
                    body=bytes([1 if i == 0 else 0] * 16),
                    club=bytes([0] * 16),
                    valid=bytes([1] * 16),
                    revision_id=f"rev-{i}-0",
                    parent_revision_id=None,
                    producer_id="reviewer",
                    correction_note="initial",
                )
            )

        source = SourceAsset(
            schema_version="shadow-tracker/source/1.0.0",
            asset_id="asset-test",
            source_uri="https://example.com/test.mp4",
            content_sha256="0" * 64,
            width_px=640,
            height_px=480,
            rights_status="permitted",
            rights_note="test asset",
        )

        widget.model.service.initialize_session(
            source_asset=source,
            observations=obs_list,
            initial_masks=masks,
        )
        widget._update_display()

        assert widget.model.frame_count == 3
        assert widget.model.current_frame_index == 0

        # 2. Keyboard navigation: Key_Right -> frame 1, Key_End -> frame 2, Key_Home -> frame 0
        event_right = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Right,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.keyPressEvent(event_right)
        assert widget.model.current_frame_index == 1

        event_end = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_End,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.keyPressEvent(event_end)
        assert widget.model.current_frame_index == 2

        event_home = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_Home,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.keyPressEvent(event_home)
        assert widget.model.current_frame_index == 0

        # 3. Key_W jumps to worst frame
        event_w = QtGui.QKeyEvent(
            QtCore.QEvent.Type.KeyPress,
            QtCore.Qt.Key.Key_W,
            QtCore.Qt.KeyboardModifier.NoModifier,
        )
        widget.keyPressEvent(event_w)
        # Frames 1 and 2 have 0 body coverage, so worst frame is frame 1 or 2
        assert widget.model.current_frame_index in (1, 2)

        # 4. Viewport paints actively with observation
        assert widget.viewport.observation is not None
        widget.viewport.repaint()

        # 5. Mask correction marks session dirty
        assert widget.is_dirty() is False
        widget.model.update_mask(
            body=bytes([1] * 16),
            club=bytes([0] * 16),
            valid=bytes([1] * 16),
            parent_revision_id=None,
            correction_note="manual review fix",
        )
        assert widget.is_dirty() is True

        # 6. Honest refusal on automated fit
        with pytest.raises(Exception, match="unqualified|unavailable"):
            widget.model.request_fit(
                FitRequest(
                    schema_version="shadow-tracker/fit-request/1.0.0",
                    request_id="req-test",
                    shot_id="shot-001",
                    model_hash="0" * 64,
                    candidate_count=1,
                    objective_profile="silhouette_iou",
                    time_window_start_pts=0,
                    time_window_end_pts=100,
                    budget_seconds=10.0,
                    engine_capability_requirement=("forward_dynamics",),
                )
            )
        assert widget.model.last_error_message is not None
        assert "unavailable" in widget.model.last_error_message.lower()
    finally:
        widget.cleanup()


# ---------------------------------------------------------------------------
# Review-fix regressions (PR #11127 Codex findings)
# ---------------------------------------------------------------------------


def _unknown_time_observation() -> FrameObservation:
    """An imported observation with unknown physical time (default GUI import)."""
    return FrameObservation(
        schema_version="shadow-tracker/frame-observation/1.0.0",
        shot_id="shot-001",
        camera_id="cam-001",
        frame_id="frame-001",
        pts_ticks=33,
        timebase_numerator=1,
        timebase_denominator=1000,
        physical_time_s=None,
        physical_time_reason="unknown physical time without evidenced clock mapping",
        body_mask_ref="mask-body-001",
        club_mask_ref="mask-club-001",
        valid_mask_ref="mask-valid-001",
        confidence_provenance="manual_review",
        timing_mode="container_pts",
        is_timing_exact=True,
        clock_evidence="iso_bmff_pts",
        decoder_name="opencv",
    )


def test_format_clock_evidence_text_renders_unknown_time_without_none_formatting() -> (
    None
):
    """The viewport timing formatter must not format None physical time (Codex P1)."""
    from src.tools.shadow_tracker.gui import format_clock_evidence_text

    known_text = format_clock_evidence_text(_dummy_observation())
    assert "0.0330s" in known_text
    assert "PTS: 33" in known_text

    unknown_obs = _unknown_time_observation()
    unknown_text = format_clock_evidence_text(unknown_obs)
    assert "Physical Time: unknown" in unknown_text
    assert "unknown physical time without evidenced clock mapping" in unknown_text
    assert "PTS: 33" in unknown_text


def test_viewport_renders_imported_frame_with_unknown_physical_time(qapp: Any) -> None:
    """Repainting an imported (unknown-time) observation must paint cleanly, not raise."""
    widget = ShadowTrackerWidget()
    try:
        viewport = widget.viewport
        obs = _unknown_time_observation()
        viewport.set_observation(obs)
        assert viewport.observation == obs
        # Renders synchronously; a TypeError from formatting None aborts here.
        pixmap = viewport.grab()
        assert not pixmap.isNull()
    finally:
        widget.cleanup()
