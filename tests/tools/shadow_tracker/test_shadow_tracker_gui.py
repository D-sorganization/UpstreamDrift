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
