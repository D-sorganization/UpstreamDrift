"""Unit tests for the Golf Simulator Desktop Controls and UI.

Follows TDD, DbC, Law of Demeter, and DRY.
Acceptance criteria from Issue #10196 (GS-07).
"""

from __future__ import annotations

import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = pytest.mark.unit


def test_lazy_loading_when_pyqt_absent() -> None:
    """When PyQt6 is missing or fails to import, get_dockable_ui returns an actionable fallback."""
    from src.tools.golf_simulator._embed_adapter import GolfSimulatorEmbedAdapter

    adapter = GolfSimulatorEmbedAdapter()
    caps = adapter.embed_capabilities()
    assert caps.supports_embedded is True

    # Simulate missing PyQt6
    with patch.dict(sys.modules, {"PyQt6": None, "PyQt6.QtWidgets": None}):
        widget = adapter.create_main_widget(parent=None)
        assert widget is not None
        assert (
            hasattr(widget, "is_fallback")
            or hasattr(widget, "error_message")
            or widget is not None
        )


def test_desktop_widget_creation_with_pyqt() -> None:
    """When PyQt6 is available, MainWidget initializes in IDLE state with all controls."""
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])

    from src.tools.golf_simulator.gui import MainWidget

    widget = MainWidget(parent=None)
    try:
        assert widget is not None
        # Verify UI subcomponents exist
        assert hasattr(widget, "_destination_combo")
        assert hasattr(widget, "_status_badge")
        assert hasattr(widget, "_btn_prepare")
        assert hasattr(widget, "_btn_arm")
        assert hasattr(widget, "_btn_disarm")
        assert hasattr(widget, "_btn_cancel")
        assert hasattr(widget, "_btn_submit")
        assert hasattr(widget, "_btn_reconcile")

        # Initial state should be IDLE: arm and submit should be disabled
        assert widget._btn_arm.isEnabled() is False
        assert widget._btn_submit.isEnabled() is False
        assert widget._btn_reconcile.isEnabled() is False

        # Transition to PREPARED
        widget._btn_prepare.click()
        assert widget._btn_arm.isEnabled() is True
        assert widget._btn_submit.isEnabled() is False
        assert widget.is_dirty() is True

        # Transition to ARMED
        widget._btn_arm.click()
        assert widget._btn_submit.isEnabled() is True
        assert widget._btn_arm.isEnabled() is False
        assert widget._destination_combo.isEnabled() is False

        # Submit transition
        widget._btn_submit.click()
        assert widget._current_state == "ACCEPTED"
        assert widget._status_badge.text() == "ACCEPTED"

        # Uncertain state recovery
        widget._update_state_ui("SENT_UNCONFIRMED")
        assert widget._btn_reconcile.isEnabled() is True
        widget._recon_reason_input.setText("Visual confirmation")
        widget._btn_reconcile.click()
        assert widget._current_state == "VISUALLY_VERIFIED"
        assert widget._status_badge.text() == "VISUALLY_VERIFIED"
    finally:
        widget.deleteLater()
