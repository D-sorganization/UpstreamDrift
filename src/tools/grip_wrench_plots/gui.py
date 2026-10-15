"""Grip wrench and Loop Null-Space Exploration (NSX) plot widget (GCV-10 #11716, NSX-43 #12575).

Pure presentation over :mod:`biomechanics.grip_plot_model`, the shared
matplotlib renderer, and the framework-free NSX plot model (:mod:`.nsx_tabs`).
The first tab renders the offline grip-wrench series; the four NSX tabs
(``Invariants``, ``Envelopes``, ``Hand Split``, ``Diagnostics``) render the
NSX plot model with a registry-backed preset selector, evidence badges per
panel, a link to the user help guide, and a cancellable ``QThreadPool`` job
worker that shows invariants first.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import json
from pathlib import Path
from typing import Any

from PyQt6.QtCore import QThreadPool
from PyQt6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from src.shared.python.biomechanics.grip_plot_model import (
    GripPlotSeries,
    series_from_payload,
)
from src.shared.python.plotting.renderers.grip_wrench import (
    COUPLE_FRAMES,
    plot_grip_wrench,
)
from src.tools.grip_wrench_plots.nsx_tabs import (
    HELP_GUIDE_REL_PATH,
    NSXAnalysisWorker,
    NSXPlotTab,
    adapt_plot_model_for_preset,
    load_golden_plot_model,
    load_nsx_presets,
    validate_plot_model_dict,
)

UNAVAILABLE_TEXT = "unavailable"


class GripWrenchPlotWidget(QWidget):
    """Tabbed Grip Wrench and Loop Null-Space Exploration (NSX) widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._series: GripPlotSeries | None = None
        self._plot_model: dict[str, Any] | None = None
        self._active_worker: NSXAnalysisWorker | None = None
        self.thread_pool = QThreadPool(self)
        self._presets = load_nsx_presets()

        root = QVBoxLayout(self)
        root.addLayout(self._build_nsx_header())
        self.nsx_status_label = QLabel("NSX status: idle")
        root.addWidget(self.nsx_status_label)

        self.tabs = QTabWidget(self)
        self.grip_wrench_tab = self._build_grip_wrench_tab()
        self.invariants_tab = NSXPlotTab("invariants", self.tabs)
        self.envelopes_tab = NSXPlotTab("envelopes", self.tabs)
        self.hand_split_tab = NSXPlotTab("hand_split", self.tabs)
        self.diagnostics_tab = NSXPlotTab("diagnostics", self.tabs)

        self.tabs.addTab(self.grip_wrench_tab, "Grip Wrench")
        self.tabs.addTab(self.invariants_tab, "Invariants")
        self.tabs.addTab(self.envelopes_tab, "Envelopes")
        self.tabs.addTab(self.hand_split_tab, "Hand Split")
        self.tabs.addTab(self.diagnostics_tab, "Diagnostics")
        root.addWidget(self.tabs, 1)

    def _build_nsx_header(self) -> QHBoxLayout:
        header = QHBoxLayout()
        header.addWidget(QLabel("NSX Preset"))
        self.preset_combo = QComboBox()
        for preset in self._presets:
            name = str(preset["name"])
            label = str(preset.get("label", name))
            self.preset_combo.addItem(name, preset)
            idx = self.preset_combo.count() - 1
            self.preset_combo.setItemData(
                idx, f"{label}: {preset.get('description', '')}"
            )
        default_idx = self.preset_combo.findText("capacity_min_effort")
        if default_idx >= 0:
            self.preset_combo.setCurrentIndex(default_idx)
        self.run_nsx_button = QPushButton("Run NSX")
        self.cancel_nsx_button = QPushButton("Cancel NSX")
        self.cancel_nsx_button.setEnabled(False)
        self.help_link_label = QLabel(
            f'<a href="{HELP_GUIDE_REL_PATH}">Interpretation Guide ({HELP_GUIDE_REL_PATH})</a>'
        )
        self.help_link_label.setOpenExternalLinks(False)
        header.addWidget(self.preset_combo)
        header.addWidget(self.run_nsx_button)
        header.addWidget(self.cancel_nsx_button)
        header.addStretch(1)
        header.addWidget(self.help_link_label)

        self.preset_combo.currentTextChanged.connect(self._on_preset_changed)
        self.run_nsx_button.clicked.connect(lambda: self.start_nsx_job())
        self.cancel_nsx_button.clicked.connect(self.cancel_nsx_job)
        return header

    def _build_grip_wrench_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.frame_combo = QComboBox()
        self.frame_combo.addItems(list(COUPLE_FRAMES))
        self.frame_combo.setToolTip("Frame of the equivalent couple panel")
        self.open_button = QPushButton("Open Grip Wrench JSON...")
        controls.addWidget(QLabel("Couple frame"))
        controls.addWidget(self.frame_combo)
        controls.addStretch(1)
        controls.addWidget(self.open_button)
        layout.addLayout(controls)
        self.status_label = QLabel(f"{UNAVAILABLE_TEXT}: no grip wrench loaded")
        layout.addWidget(self.status_label)
        self.figure = Figure(figsize=(5.5, 7.0))
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        self.frame_combo.currentTextChanged.connect(self._redraw)
        # noqa: gui-thread/ok - opens a modal file dialog; the JSON is small.
        self.open_button.clicked.connect(self._choose_file)
        return tab

    def set_series(self, series: GripPlotSeries | None) -> None:
        """Show ``series`` on the Grip Wrench tab (``None`` clears to unavailable)."""
        self._series = series
        self._redraw()

    def load_json(self, path: str | Path) -> None:
        """Load a grip wrench API payload or NSX plot-model JSON file."""
        data: Any = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("grip wrench JSON must be an object")
        if "panels" in data and "time_s" in data:
            self.set_plot_model(data)
            return
        self.set_series(series_from_payload(data))

    def set_plot_model(self, plot_model: Mapping[str, Any] | None) -> None:
        """Validate and render an NSX plot model across the four NSX tabs."""
        if plot_model is None:
            self._plot_model = None
            for tab in self.nsx_tabs():
                tab.clear_panel()
            self.nsx_status_label.setText("NSX status: cleared")
            return
        validated = validate_plot_model_dict(plot_model)
        self._plot_model = validated
        panels = validated.get("panels", {})
        time_s = [float(t) for t in validated.get("time_s", ())]
        self.invariants_tab.render_panel(panels.get("invariants", {}), time_s)
        self.envelopes_tab.render_panel(panels.get("envelopes", {}), time_s)
        self.hand_split_tab.render_panel(panels.get("hand_split", {}), time_s)
        self.diagnostics_tab.render_panel(panels.get("diagnostics", {}), time_s)
        preset = str(validated.get("preset_name", self.preset_combo.currentText()))
        self.nsx_status_label.setText(
            f"NSX status: completed ({preset}, {len(time_s)} samples)"
        )

    def nsx_tabs(self) -> tuple[NSXPlotTab, NSXPlotTab, NSXPlotTab, NSXPlotTab]:
        """Return the four NSX plot tabs in display order."""
        return (
            self.invariants_tab,
            self.envelopes_tab,
            self.hand_split_tab,
            self.diagnostics_tab,
        )

    def start_nsx_job(
        self,
        preset_name: str | None = None,
        *,
        analyzer: Callable[[str], dict[str, Any]] | None = None,
        defer_completion: bool = False,
    ) -> NSXAnalysisWorker:
        """Start a cancellable NSX worker on ``self.thread_pool``."""
        if self._active_worker is not None:
            self._active_worker.cancel()
        chosen = preset_name or self.preset_combo.currentText() or "capacity_min_effort"
        worker = NSXAnalysisWorker(
            chosen, analyzer=analyzer, defer_completion=defer_completion
        )
        worker.signals.progress.connect(self._on_worker_progress)
        worker.signals.completed.connect(self._on_worker_completed)
        worker.signals.cancelled.connect(self._on_worker_cancelled)
        worker.signals.failed.connect(self._on_worker_failed)
        self._active_worker = worker
        self.cancel_nsx_button.setEnabled(True)
        self.nsx_status_label.setText(f"NSX status: running ({chosen})")
        self.thread_pool.start(worker)
        return worker

    def cancel_nsx_job(self) -> None:
        """Cancel the currently running NSX background job if any."""
        if self._active_worker is not None:
            self._active_worker.cancel()
            self.cancel_nsx_button.setEnabled(False)
            self.nsx_status_label.setText("NSX status: cancelled")

    def _on_preset_changed(self, preset_name: str) -> None:
        if not preset_name:
            return
        if self._plot_model is not None:
            base = load_golden_plot_model()
            self.set_plot_model(adapt_plot_model_for_preset(base, preset_name))

    def _on_worker_progress(self, payload: dict[str, Any]) -> None:
        inv_panel = payload.get("invariants", {})
        time_s = [float(t) for t in payload.get("time_s", ())]
        if isinstance(inv_panel, Mapping) and time_s:
            self.invariants_tab.render_panel(inv_panel, time_s)
            preset = payload.get("preset_name", self.preset_combo.currentText())
            self.nsx_status_label.setText(
                f"NSX status: partial invariants ready ({preset})"
            )

    def _on_worker_completed(self, model: dict[str, Any]) -> None:
        self.cancel_nsx_button.setEnabled(False)
        self.set_plot_model(model)

    def _on_worker_cancelled(self, preset_name: str) -> None:
        self.cancel_nsx_button.setEnabled(False)
        self.nsx_status_label.setText(f"NSX status: cancelled ({preset_name})")

    def _on_worker_failed(self, error_msg: str) -> None:
        self.cancel_nsx_button.setEnabled(False)
        self.nsx_status_label.setText(f"NSX status: failed ({error_msg})")

    def _choose_file(self) -> None:
        name, _ = QFileDialog.getOpenFileName(
            self, "Open Grip Wrench JSON", "", "JSON (*.json)"
        )
        if name:
            try:
                self.load_json(name)
            except (ValueError, OSError) as exc:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {exc}")

    def _redraw(self, *_: Any) -> None:
        series = self._series
        if series is None:
            self.figure.clear()
            self.status_label.setText(f"{UNAVAILABLE_TEXT}: no grip wrench loaded")
        else:
            plot_grip_wrench(
                self.figure, series, couple_frame=self.frame_combo.currentText()
            )
            if series.available:
                self.status_label.setText(
                    f"Split method: {series.split_method} | {len(series.time_s)} samples"
                )
            else:
                self.status_label.setText(f"{UNAVAILABLE_TEXT}: {series.reason}")
        self.canvas.draw_idle()

    def cleanup(self) -> None:
        """Release data references and cancel active jobs; safe to call repeatedly."""
        self.cancel_nsx_job()
        self.thread_pool.waitForDone(1000)
        self._series = None
        self._plot_model = None
