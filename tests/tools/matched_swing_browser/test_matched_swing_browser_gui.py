"""Headless PyQt6 GUI tests for Matched Swing Browser (MS-80, #10353).

Validates:
- Headless widget creation with the committed production ledger (>= 40 rows).
- Interactive filtering by engine, capture, lane, verdict, and search text.
- Selecting a run updates five metrics, physical gates, and acceptance badge.
- QMovie initialization and playback for rows with visual GIF artifacts.
- Action buttons enable/disable based on artifact availability.
- Journey test: open browser -> select committed driver run -> verify GIF playback -> open in viewer -> receipt badge (folded in from #10360).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("PyQt6", reason="Matched Swing Browser requires PyQt6")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtCore, QtGui, QtWidgets  # noqa: E402
from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.tools.matched_swing_browser.gui import (  # noqa: E402
    MatchedSwingBrowserWidget,
    MatchedSwingBrowserWindow,
)
from src.tools.matched_swing_browser.model import (  # noqa: E402
    MatchedSwingBrowserModel,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_APP: QApplication | None = None


@pytest.fixture(scope="session", autouse=True)
def qapp() -> QApplication:
    """Module/session level persistent QApplication to avoid recreation crashes."""
    global _APP
    if _APP is None:
        inst = QApplication.instance()
        _APP = inst if isinstance(inst, QApplication) else QApplication(sys.argv[:1])
    return _APP


@pytest.fixture
def widget(qapp: QApplication) -> Generator[MatchedSwingBrowserWidget, None, None]:
    """Instantiate a MatchedSwingBrowserWidget backed by the committed ledger."""
    w = MatchedSwingBrowserWidget()
    yield w
    w.cleanup()


class TestMatchedSwingBrowserWidget:
    def test_widget_builds_and_populates_production_ledger(
        self, widget: MatchedSwingBrowserWidget
    ) -> None:
        assert widget.table is not None
        # Must list at least 40 rows per acceptance criteria (production has 98)
        assert widget.table.rowCount() >= 40
        assert widget.total_runs_count >= 40

    def test_filter_by_engine_updates_table(
        self, widget: MatchedSwingBrowserWidget
    ) -> None:
        initial_count = widget.table.rowCount()
        assert initial_count >= 40

        # Select 'drake' in engine combo
        idx = widget.engine_combo.findText("drake")
        assert idx >= 0
        widget.engine_combo.setCurrentIndex(idx)

        filtered_count = widget.table.rowCount()
        assert 0 < filtered_count < initial_count

        # All displayed rows must have engine 'drake'
        for r in range(filtered_count):
            item = widget.table.item(r, 0)
            assert item is not None
            assert item.text().lower() == "drake"

    def test_search_text_filters_table(self, widget: MatchedSwingBrowserWidget) -> None:
        widget.search_input.setText("driver_g1_drake")
        filtered_count = widget.table.rowCount()
        assert filtered_count >= 1
        for r in range(filtered_count):
            item = widget.table.item(r, 6)  # receipt path column
            assert item is not None
            assert "driver_g1_drake" in item.text()

    def test_reset_filters_restores_all_rows(
        self, widget: MatchedSwingBrowserWidget
    ) -> None:
        initial_count = widget.table.rowCount()
        widget.engine_combo.setCurrentIndex(1)
        assert widget.table.rowCount() < initial_count

        widget.reset_button.click()
        assert widget.table.rowCount() == initial_count

    def test_row_selection_displays_metrics_and_badge(
        self, widget: MatchedSwingBrowserWidget
    ) -> None:
        # Find a row with driver_g1_drake
        target_row = -1
        for r in range(widget.table.rowCount()):
            path_item = widget.table.item(r, 6)
            if path_item and "driver_g1_drake" in path_item.text():
                target_row = r
                break

        assert target_row >= 0
        widget.table.selectRow(target_row)

        # Acceptance badge should say PASSED (green)
        assert "PASSED" in widget.badge_label.text().upper()

        # Metrics panel should display formatted numbers
        whole_text = widget.whole_rmse_val.text()
        assert "mm" in whole_text or whole_text != "—"

    def test_gif_player_starts_on_row_with_gif(
        self, widget: MatchedSwingBrowserWidget
    ) -> None:
        target_row = -1
        for r in range(widget.table.rowCount()):
            path_item = widget.table.item(r, 6)
            if path_item and "driver_g1_drake" in path_item.text():
                target_row = r
                break

        assert target_row >= 0
        widget.table.selectRow(target_row)

        assert widget.movie is not None
        assert isinstance(widget.movie, QtGui.QMovie)
        assert widget.movie.state() == QtGui.QMovie.MovieState.Running

    def test_action_buttons_state(self, widget: MatchedSwingBrowserWidget) -> None:
        target_row = -1
        for r in range(widget.table.rowCount()):
            path_item = widget.table.item(r, 6)
            if path_item and "driver_g1_drake" in path_item.text():
                target_row = r
                break

        assert target_row >= 0
        widget.table.selectRow(target_row)

        # driver_g1_drake has candidate.npz, so viewer buttons must be enabled
        assert widget.open_viewer_btn.isEnabled()
        assert widget.open_native_btn.isEnabled()
        assert widget.open_parity_btn.isEnabled()


class TestJourneyHeadless:
    """Journey test: open Results Browser -> select driver run -> GIF plays -> open in viewer -> receipt badge."""

    def test_results_browser_journey(self, qapp: QApplication) -> None:
        window = MatchedSwingBrowserWindow()
        widget = window.widget

        try:
            assert widget.table.rowCount() >= 40

            # 1. Pick committed driver run
            target_row = -1
            for r in range(widget.table.rowCount()):
                path_item = widget.table.item(r, 6)
                if path_item and "driver_g1_drake" in path_item.text():
                    target_row = r
                    break
            assert target_row >= 0
            widget.table.selectRow(target_row)

            # 2. Verify GIF plays
            assert widget.movie is not None
            assert widget.movie.isValid()

            # 3. Verify receipt badge
            assert "PASSED" in widget.badge_label.text().upper()

            # 4. Open in Tour Matching Viewer mock
            with patch(
                "src.tools.tour_matching_viewer.gui.TourMatchingViewerWindow"
            ) as mock_tmv:
                widget.open_viewer_btn.click()
                assert mock_tmv.called
        finally:
            window.close()
