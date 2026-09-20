"""PyQt6 GUI for the Matched Swing Results Browser (MS-80, #10353).

Provides dual-pane browsing of matched swing runs:
- Left pane: filterable table of runs from ``reports/matched_swing_ledger.json``.
- Right pane: receipt summary, five metrics, physical gates, GIF player (QMovie),
  and launcher buttons (Tour Matching Viewer, Native Viewer MS-83, Parity Report).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PyQt6 import QtCore, QtGui, QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.ledger_schema import LedgerRow
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)

logger = get_logger(__name__)

__all__ = [
    "MatchedSwingBrowserWidget",
    "MatchedSwingBrowserWindow",
]


class MatchedSwingBrowserWidget(QWidget):
    """Dual-pane desktop widget for browsing matched-swing execution ledger."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        model: MatchedSwingBrowserModel | None = None,
        ledger_path: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model or MatchedSwingBrowserModel()
        self._ledger_path = ledger_path
        self._all_rows: list[LedgerRow] = []
        self._current_filtered_rows: list[LedgerRow] = []
        self._selected_row: LedgerRow | None = None
        self._movie: QtGui.QMovie | None = None

        self._init_ui()
        self.reload_ledger()

    @property
    def table(self) -> QTableWidget:
        return self._table

    @property
    def total_runs_count(self) -> int:
        return len(self._all_rows)

    @property
    def engine_combo(self) -> QComboBox:
        return self._engine_combo

    @property
    def search_input(self) -> QLineEdit:
        return self._search_input

    @property
    def reset_button(self) -> QPushButton:
        return self._reset_btn

    @property
    def badge_label(self) -> QLabel:
        return self._badge_lbl

    @property
    def whole_rmse_val(self) -> QLabel:
        return self._whole_rmse_lbl

    @property
    def movie(self) -> QtGui.QMovie | None:
        return self._movie

    @property
    def open_viewer_btn(self) -> QPushButton:
        return self._open_tmv_btn

    @property
    def open_native_btn(self) -> QPushButton:
        return self._open_native_btn

    @property
    def open_parity_btn(self) -> QPushButton:
        return self._open_parity_btn

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        layout.addWidget(splitter)

        left_widget = self._create_left_panel()
        right_widget = self._create_right_panel()

        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

    def _create_left_panel(self) -> QWidget:
        panel = QWidget(self)
        vbox = QVBoxLayout(panel)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(6)

        filter_group = self._create_filter_group()
        vbox.addWidget(filter_group)

        self._table = QTableWidget(panel)
        self._table.setColumnCount(7)
        self._table.setHorizontalHeaderLabels(
            [
                "Engine",
                "Capture",
                "Lane",
                "Horizon",
                "Whole RMSE",
                "Verdict",
                "Receipt Path",
            ]
        )
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            header.setStretchLastSection(True)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectionBehavior.SelectRows
        )
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.SingleSelection
        )
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers
        )
        self._table.itemSelectionChanged.connect(self._on_selection_changed)
        vbox.addWidget(self._table, stretch=1)

        self._count_lbl = QLabel("Showing 0 runs", panel)
        vbox.addWidget(self._count_lbl)
        return panel

    def _create_filter_group(self) -> QGroupBox:
        group = QGroupBox("Filter Runs", self)
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(4)

        grid.addWidget(QLabel("Engine:"), 0, 0)
        self._engine_combo = QComboBox(group)
        self._engine_combo.currentIndexChanged.connect(self._apply_filters)
        grid.addWidget(self._engine_combo, 0, 1)

        grid.addWidget(QLabel("Capture:"), 0, 2)
        self._capture_combo = QComboBox(group)
        self._capture_combo.currentIndexChanged.connect(self._apply_filters)
        grid.addWidget(self._capture_combo, 0, 3)

        grid.addWidget(QLabel("Lane:"), 1, 0)
        self._lane_combo = QComboBox(group)
        self._lane_combo.currentIndexChanged.connect(self._apply_filters)
        grid.addWidget(self._lane_combo, 1, 1)

        grid.addWidget(QLabel("Verdict:"), 1, 2)
        self._verdict_combo = QComboBox(group)
        self._verdict_combo.addItems(["All", "PASSED", "REJECTED", "UNCLASSIFIED"])
        self._verdict_combo.currentIndexChanged.connect(self._apply_filters)
        grid.addWidget(self._verdict_combo, 1, 3)

        grid.addWidget(QLabel("Search:"), 2, 0)
        self._search_input = QLineEdit(group)
        self._search_input.setPlaceholderText("Filter by path, sha, reason...")
        self._search_input.textChanged.connect(self._apply_filters)
        grid.addWidget(self._search_input, 2, 1, 1, 2)

        self._reset_btn = QPushButton("Reset", group)
        self._reset_btn.clicked.connect(self._on_reset_filters)
        grid.addWidget(self._reset_btn, 2, 3)
        return group

    def _create_right_panel(self) -> QWidget:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        container = QWidget()
        scroll.setWidget(container)

        vbox = QVBoxLayout(container)
        vbox.setContentsMargins(6, 6, 6, 6)
        vbox.setSpacing(8)

        # Header: Path, Badge, Meta
        vbox.addWidget(self._create_header_card())
        # Standard Metrics
        vbox.addWidget(self._create_metrics_card())
        # Physical Gates
        vbox.addWidget(self._create_gates_card())
        # Visual Playback
        vbox.addWidget(self._create_playback_card())
        # Action Buttons
        vbox.addWidget(self._create_actions_card())
        vbox.addStretch(1)
        return scroll

    def _create_header_card(self) -> QWidget:
        card = QGroupBox("Run Summary", self)
        vbox = QVBoxLayout(card)
        vbox.setSpacing(4)

        top_row = QHBoxLayout()
        self._receipt_path_lbl = QLabel("Select a run to view details", card)
        self._receipt_path_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self._receipt_path_lbl.setStyleSheet("font-weight: bold; font-size: 13px;")
        top_row.addWidget(self._receipt_path_lbl, stretch=1)

        self._badge_lbl = QLabel("UNCLASSIFIED", card)
        self._update_badge_style("UNCLASSIFIED")
        top_row.addWidget(self._badge_lbl)
        vbox.addLayout(top_row)

        self._candidate_lbl = QLabel("Candidate: —", card)
        self._candidate_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        vbox.addWidget(self._candidate_lbl)

        self._meta_lbl = QLabel("Horizon: — | Engine: — | Lane: —", card)
        vbox.addWidget(self._meta_lbl)
        return card

    def _create_metrics_card(self) -> QGroupBox:
        card = QGroupBox("Shared Standardized Metrics", self)
        grid = QtWidgets.QGridLayout(card)
        grid.setSpacing(4)

        grid.addWidget(QLabel("Whole Marker RMSE:"), 0, 0)
        self._whole_rmse_lbl = QLabel("—", card)
        grid.addWidget(self._whole_rmse_lbl, 0, 1)

        grid.addWidget(QLabel("Early Marker RMSE:"), 0, 2)
        self._early_rmse_lbl = QLabel("—", card)
        grid.addWidget(self._early_rmse_lbl, 0, 3)

        grid.addWidget(QLabel("Terminal Marker RMSE:"), 1, 0)
        self._term_rmse_lbl = QLabel("—", card)
        grid.addWidget(self._term_rmse_lbl, 1, 1)

        grid.addWidget(QLabel("Club Marker RMSE:"), 1, 2)
        self._club_rmse_lbl = QLabel("—", card)
        grid.addWidget(self._club_rmse_lbl, 1, 3)

        grid.addWidget(QLabel("Pelvis Yaw RMSE:"), 2, 0)
        self._yaw_rmse_lbl = QLabel("—", card)
        grid.addWidget(self._yaw_rmse_lbl, 2, 1)
        return card

    def _create_gates_card(self) -> QGroupBox:
        card = QGroupBox("Physical Gates & Verdicts", self)
        vbox = QVBoxLayout(card)
        vbox.setSpacing(4)

        self._gates_text = QTextEdit(card)
        self._gates_text.setReadOnly(True)
        self._gates_text.setMaximumHeight(100)
        self._gates_text.setPlaceholderText("No physical gates evaluated for this run.")
        vbox.addWidget(self._gates_text)

        self._rejection_lbl = QLabel("", card)
        self._rejection_lbl.setWordWrap(True)
        self._rejection_lbl.setStyleSheet("color: #d32f2f; font-size: 11px;")
        vbox.addWidget(self._rejection_lbl)
        return card

    def _create_playback_card(self) -> QGroupBox:
        card = QGroupBox("Visual Playback", self)
        vbox = QVBoxLayout(card)
        vbox.setSpacing(6)
        vbox.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._gif_lbl = QLabel("No visual playback artifact available", card)
        self._gif_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._gif_lbl.setMinimumSize(320, 240)
        self._gif_lbl.setStyleSheet(
            "border: 1px dashed #555; border-radius: 4px; background: #1a1a1a;"
        )
        vbox.addWidget(self._gif_lbl)

        ctrl_row = QHBoxLayout()
        self._play_btn = QPushButton("Play", card)
        self._play_btn.clicked.connect(self._on_play_gif)
        self._pause_btn = QPushButton("Pause", card)
        self._pause_btn.clicked.connect(self._on_pause_gif)
        self._restart_btn = QPushButton("Restart", card)
        self._restart_btn.clicked.connect(self._on_restart_gif)

        ctrl_row.addWidget(self._play_btn)
        ctrl_row.addWidget(self._pause_btn)
        ctrl_row.addWidget(self._restart_btn)
        vbox.addLayout(ctrl_row)
        return card

    def _create_actions_card(self) -> QGroupBox:
        card = QGroupBox("Actions & Viewers", self)
        grid = QtWidgets.QGridLayout(card)
        grid.setSpacing(6)

        self._open_tmv_btn = QPushButton("Open in Tour Matching Viewer", card)
        self._open_tmv_btn.clicked.connect(self._on_open_tour_matching_viewer)
        self._open_tmv_btn.setEnabled(False)
        grid.addWidget(self._open_tmv_btn, 0, 0)

        self._open_native_btn = QPushButton("Open in Native Viewer (MS-83)", card)
        self._open_native_btn.clicked.connect(self._on_open_native_viewer)
        self._open_native_btn.setEnabled(False)
        grid.addWidget(self._open_native_btn, 0, 1)

        self._open_parity_btn = QPushButton("Open Parity Report", card)
        self._open_parity_btn.clicked.connect(self._on_open_parity_report)
        self._open_parity_btn.setEnabled(False)
        grid.addWidget(self._open_parity_btn, 1, 0)

        self._view_json_btn = QPushButton("View Receipt JSON", card)
        self._view_json_btn.clicked.connect(self._on_view_receipt_json)
        self._view_json_btn.setEnabled(False)
        grid.addWidget(self._view_json_btn, 1, 1)
        return card

    def reload_ledger(self) -> None:
        """Load rows from ledger and populate table and combo filters."""
        self._all_rows = self._model.load_ledger(self._ledger_path)
        self._populate_combo_boxes()
        self._apply_filters()

    def _populate_combo_boxes(self) -> None:
        """Fill combo filters with discovered values from current ledger."""
        for combo, values in (
            (self._engine_combo, self._model.get_unique_engines(self._all_rows)),
            (self._capture_combo, self._model.get_unique_captures(self._all_rows)),
            (self._lane_combo, self._model.get_unique_lanes(self._all_rows)),
        ):
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("All")
            for val in values:
                combo.addItem(val)
            combo.blockSignals(False)

    def _apply_filters(self) -> None:
        """Filter cached rows and refresh table view."""
        engine = (
            None
            if self._engine_combo.currentIndex() <= 0
            else self._engine_combo.currentText()
        )
        capture = (
            None
            if self._capture_combo.currentIndex() <= 0
            else self._capture_combo.currentText()
        )
        lane = (
            None
            if self._lane_combo.currentIndex() <= 0
            else self._lane_combo.currentText()
        )
        verdict = (
            None
            if self._verdict_combo.currentIndex() <= 0
            else self._verdict_combo.currentText()
        )
        text = self._search_input.text()

        criteria = MatchedSwingFilter(
            engine=engine,
            capture=capture,
            lane=lane,
            verdict=verdict,
            text=text,
        )
        self._current_filtered_rows = self._model.filter_rows(self._all_rows, criteria)
        self._populate_table(self._current_filtered_rows)
        self._count_lbl.setText(
            f"Showing {len(self._current_filtered_rows)} of {len(self._all_rows)} runs"
        )

    def _populate_table(self, rows: list[LedgerRow]) -> None:
        """Render rows into QTableWidget."""
        self._table.blockSignals(True)
        self._table.setRowCount(len(rows))

        for idx, row in enumerate(rows):
            verdict = self._model.extract_verdict_string(row)
            whole_rmse = self._model.format_metric(
                row.metrics.whole_marker_rmse_m, "mm"
            )
            horizon_str = f"{row.horizon_s:.2f} s" if row.horizon_s is not None else "—"

            items = [
                QTableWidgetItem(row.engine),
                QTableWidgetItem(row.capture or "—"),
                QTableWidgetItem(row.lane),
                QTableWidgetItem(horizon_str),
                QTableWidgetItem(whole_rmse),
                QTableWidgetItem(verdict),
                QTableWidgetItem(row.receipt_path),
            ]
            for col, item in enumerate(items):
                self._table.setItem(idx, col, item)

        self._table.blockSignals(False)
        if rows:
            self._table.selectRow(0)
        else:
            self._clear_details()

    def _on_selection_changed(self) -> None:
        selected_indexes = self._table.selectedIndexes()
        if not selected_indexes:
            self._clear_details()
            return
        row_idx = selected_indexes[0].row()
        if 0 <= row_idx < len(self._current_filtered_rows):
            self._display_row_details(self._current_filtered_rows[row_idx])

    def _display_row_details(self, row: LedgerRow) -> None:
        """Render all summary panels for the newly selected row."""
        self._selected_row = row
        self._receipt_path_lbl.setText(row.receipt_path)
        verdict = self._model.extract_verdict_string(row)
        self._badge_lbl.setText(verdict)
        self._update_badge_style(verdict)

        cand = row.candidate_sha or "—"
        self._candidate_lbl.setText(f"Candidate SHA: {cand}")
        horizon = f"{row.horizon_s:.2f} s" if row.horizon_s is not None else "—"
        self._meta_lbl.setText(
            f"Horizon: {horizon} | Engine: {row.engine} | Lane: {row.lane} | Capture: {row.capture or '—'}"
        )

        m = row.metrics
        self._whole_rmse_lbl.setText(
            self._model.format_metric(m.whole_marker_rmse_m, "mm")
        )
        self._early_rmse_lbl.setText(
            self._model.format_metric(m.early_marker_rmse_m, "mm")
        )
        self._term_rmse_lbl.setText(
            self._model.format_metric(m.terminal_marker_rmse_m, "mm")
        )
        self._club_rmse_lbl.setText(
            self._model.format_metric(m.club_marker_rmse_m, "mm")
        )
        self._yaw_rmse_lbl.setText(
            self._model.format_metric(m.pelvis_yaw_rmse_rad, "deg")
        )

        self._populate_gates_info(row)
        self._load_gif_artifact(row)
        self._update_action_buttons(row)

    def _populate_gates_info(self, row: LedgerRow) -> None:
        """Format acceptance gates report for details text box."""
        if not row.acceptance or "gates" not in row.acceptance:
            self._gates_text.setPlainText("No quantitative acceptance gates recorded.")
            self._rejection_lbl.setText(row.reason or "")
            return

        lines: list[str] = []
        gates = row.acceptance.get("gates", [])
        for g in gates:
            name = g.get("name", "gate")
            status = g.get("status", "").upper()
            th = g.get("threshold", 0.0)
            ms = g.get("measured")
            unit = g.get("unit", "m")
            ms_str = f"{ms:.4f}" if ms is not None else "—"
            lines.append(
                f"• {name} [{status}]: measured={ms_str} {unit} (limit={th:.4f} {unit})"
            )

        self._gates_text.setPlainText("\n".join(lines))
        note = row.acceptance.get("qualification_note") or row.reason or ""
        self._rejection_lbl.setText(note)

    def _load_gif_artifact(self, row: LedgerRow) -> None:
        """Safely load and play GIF animation on UI without blocking."""
        self._stop_movie()
        gif_path = self._model.resolve_artifact_path(row, "gif")

        if gif_path and gif_path.is_file():
            self._movie = QtGui.QMovie(str(gif_path))
            self._gif_lbl.setMovie(self._movie)
            self._movie.start()
            self._set_playback_controls_enabled(True)
        else:
            self._gif_lbl.setText("No visual playback artifact available")
            self._set_playback_controls_enabled(False)

    def _update_action_buttons(self, row: LedgerRow) -> None:
        """Enable or disable viewer/report buttons based on artifact presence."""
        npz_path = self._model.resolve_artifact_path(row, "npz")
        has_npz = npz_path is not None and npz_path.is_file()
        self._open_tmv_btn.setEnabled(has_npz)
        self._open_native_btn.setEnabled(has_npz)

        parity_path = self._model.resolve_artifact_path(row, "parity")
        self._open_parity_btn.setEnabled(
            parity_path is not None and parity_path.is_file()
        )

        receipt_path = self._model.resolve_artifact_path(row, "receipt")
        self._view_json_btn.setEnabled(
            receipt_path is not None and receipt_path.is_file()
        )

    def _update_badge_style(self, verdict: str) -> None:
        if verdict in ("PASSED", "ACCEPTED"):
            self._badge_lbl.setStyleSheet(
                "background-color: #1b5e20; color: #a5d6a7; font-weight: bold; "
                "padding: 4px 10px; border-radius: 12px; border: 1px solid #2e7d32;"
            )
        elif verdict in ("FAILED", "REJECTED"):
            self._badge_lbl.setStyleSheet(
                "background-color: #b71c1c; color: #ef9a9a; font-weight: bold; "
                "padding: 4px 10px; border-radius: 12px; border: 1px solid #c62828;"
            )
        else:
            self._badge_lbl.setStyleSheet(
                "background-color: #424242; color: #e0e0e0; font-weight: bold; "
                "padding: 4px 10px; border-radius: 12px; border: 1px solid #616161;"
            )

    def _set_playback_controls_enabled(self, enabled: bool) -> None:
        self._play_btn.setEnabled(enabled)
        self._pause_btn.setEnabled(enabled)
        self._restart_btn.setEnabled(enabled)

    def _on_play_gif(self) -> None:
        if self._movie:
            self._movie.setPaused(False)

    def _on_pause_gif(self) -> None:
        if self._movie:
            self._movie.setPaused(True)

    def _on_restart_gif(self) -> None:
        if self._movie:
            self._movie.stop()
            self._movie.start()

    def _stop_movie(self) -> None:
        if self._movie:
            self._movie.stop()
            self._gif_lbl.clear()
            self._movie = None

    def _clear_details(self) -> None:
        self._selected_row = None
        self._receipt_path_lbl.setText("No run selected")
        self._badge_lbl.setText("—")
        self._candidate_lbl.setText("Candidate: —")
        self._meta_lbl.setText("Horizon: — | Engine: — | Lane: —")
        for lbl in (
            self._whole_rmse_lbl,
            self._early_rmse_lbl,
            self._term_rmse_lbl,
            self._club_rmse_lbl,
            self._yaw_rmse_lbl,
        ):
            lbl.setText("—")
        self._gates_text.clear()
        self._rejection_lbl.clear()
        self._stop_movie()
        self._gif_lbl.setText("No visual playback artifact available")
        self._set_playback_controls_enabled(False)
        self._open_tmv_btn.setEnabled(False)
        self._open_native_btn.setEnabled(False)
        self._open_parity_btn.setEnabled(False)
        self._view_json_btn.setEnabled(False)

    def _on_reset_filters(self) -> None:
        self._engine_combo.setCurrentIndex(0)
        self._capture_combo.setCurrentIndex(0)
        self._lane_combo.setCurrentIndex(0)
        self._verdict_combo.setCurrentIndex(0)
        self._search_input.clear()
        self._apply_filters()

    def _get_selected_npz_path(self) -> Path | None:
        if not self._selected_row:
            return None
        npz_path = self._model.resolve_artifact_path(self._selected_row, "npz")
        if not npz_path or not npz_path.is_file():
            QMessageBox.warning(
                self, "No Trajectory", "No NPZ trajectory artifact found."
            )
            return None
        return npz_path

    def _on_open_tour_matching_viewer(self) -> None:
        npz_path = self._get_selected_npz_path()
        if not npz_path:
            return

        try:
            from src.tools.tour_matching_viewer.gui import (
                TourMatchingViewerWindow,
            )

            viewer_win = TourMatchingViewerWindow(self)
            viewer_win.widget.load_file(npz_path)
            viewer_win.show()
        except Exception as exc:
            logger.exception("Failed to launch Tour Matching Viewer: %s", exc)
            QMessageBox.critical(
                self, "Viewer Error", f"Could not launch Tour Matching Viewer:\n{exc}"
            )

    def _on_open_native_viewer(self) -> None:
        npz_path = self._get_selected_npz_path()
        if not npz_path:
            return

        try:
            from src.shared.python.motion_matching.native_viewers import (
                ViewerLaunchConfig,
                ViewerUnavailableError,
                get_supported_backends,
                open_in_native_viewer,
            )
            from src.shared.python.motion_matching.visualization.simulation_viewer import (
                SimulationData,
            )

            backends = get_supported_backends()
            engine_choice, ok = QtWidgets.QInputDialog.getItem(
                self, "Select Native Viewer", "Choose backend:", backends, 0, False
            )
            if not ok or not engine_choice:
                return

            import numpy as np

            arr = np.load(npz_path)
            time_s = arr.get("time_s")
            q_coords = (
                arr.get("coordinates")
                if arr.get("coordinates") is not None
                else arr.get("q")
            )
            sim_data = SimulationData(time_s=time_s, q=q_coords)
            cfg = ViewerLaunchConfig(speed=1.0, view_mode="fitted")
            res = open_in_native_viewer(sim_data, engine_choice, config=cfg)
            if res.url:
                QMessageBox.information(
                    self, "Native Viewer", f"Viewer URL:\n{res.url}"
                )
        except ViewerUnavailableError as exc:
            QMessageBox.warning(self, "Viewer Unavailable", str(exc))
        except Exception as exc:
            logger.exception("Failed native viewer: %s", exc)
            QMessageBox.critical(
                self, "Native Viewer Error", f"Could not launch viewer:\n{exc}"
            )

    def _on_open_parity_report(self) -> None:
        if not self._selected_row:
            return
        parity_path = self._model.resolve_artifact_path(self._selected_row, "parity")
        if not parity_path or not parity_path.is_file():
            QMessageBox.information(
                self, "Parity Report", "No parity report found for this run."
            )
            return

        text = parity_path.read_text(encoding="utf-8")
        self._show_text_dialog("Parity Report (vs MuJoCo)", text)

    def _on_view_receipt_json(self) -> None:
        if not self._selected_row:
            return
        receipt_path = self._model.resolve_artifact_path(self._selected_row, "receipt")
        if not receipt_path or not receipt_path.is_file():
            return
        text = receipt_path.read_text(encoding="utf-8")
        self._show_text_dialog(f"Receipt JSON: {receipt_path.name}", text)

    def _show_text_dialog(self, title: str, content: str) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle(title)
        dialog.resize(680, 500)
        vbox = QVBoxLayout(dialog)
        edit = QTextEdit(dialog)
        edit.setReadOnly(True)
        edit.setPlainText(content)
        vbox.addWidget(edit)
        close_btn = QPushButton("Close", dialog)
        close_btn.clicked.connect(dialog.accept)
        vbox.addWidget(close_btn)
        dialog.exec()

    def cleanup(self) -> None:
        """Halt playback and clean up resources."""
        self._stop_movie()


class MatchedSwingBrowserWindow(QMainWindow):
    """Standalone desktop window hosting MatchedSwingBrowserWidget."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        model: MatchedSwingBrowserModel | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Matched Swing Results Browser")
        self.resize(1150, 750)
        self.widget = MatchedSwingBrowserWidget(self, model=model)
        self.setCentralWidget(self.widget)

    def closeEvent(self, event: QtGui.QCloseEvent | None) -> None:  # noqa: N802
        self.widget.cleanup()
        if event is not None:
            event.accept()
