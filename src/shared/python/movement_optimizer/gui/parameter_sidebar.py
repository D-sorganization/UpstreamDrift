# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""ParameterSidebar: left-hand parameter panel for the main window."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if TYPE_CHECKING:
    from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
    from matplotlib.figure import Figure

    from .labelled_slider import LabelledSlider

from . import _sidebar_builders as _sb
from . import _sidebar_state as _st

logger = logging.getLogger(__name__)


class ParameterSidebar(QScrollArea):
    """Left-hand sidebar with all tuneable parameters."""

    optimize_current = pyqtSignal()
    optimize_both = pyqtSignal()
    cancel_requested = pyqtSignal()
    export_requested = pyqtSignal()
    reset_requested = pyqtSignal()
    save_solution_requested = pyqtSignal()
    load_solution_requested = pyqtSignal()
    export_video_requested = pyqtSignal()
    export_plots_requested = pyqtSignal()
    export_excel_requested = pyqtSignal()
    add_comparison_requested = pyqtSignal()
    compare_trials_requested = pyqtSignal()
    clear_comparison_requested = pyqtSignal()

    # Dynamically created widgets via _sidebar_builders
    mass_slider: LabelledSlider
    height_slider: LabelledSlider
    ll_slider: LabelledSlider
    ul_slider: LabelledSlider
    to_slider: LabelledSlider
    bar_slider: LabelledSlider
    bar_depth_slider: LabelledSlider
    bar_height_slider: LabelledSlider
    model_combo: QComboBox
    dur_slider: LabelledSlider
    smooth_slider: LabelledSlider
    opt_btn: QPushButton
    both_btn: QPushButton
    cancel_btn: QPushButton
    progress: QProgressBar
    prog_label: QLabel
    iter_label: QLabel
    cost_label: QLabel
    improve_label: QLabel
    elapsed_label: QLabel
    stall_label: QLabel
    conv_fig: Figure
    conv_canvas: FigureCanvas
    conv_ax: object  # matplotlib.axes.Axes
    result_label: QLabel
    export_btn: QPushButton
    reset_btn: QPushButton
    save_btn: QPushButton
    load_btn: QPushButton
    export_video_btn: QPushButton
    export_plots_btn: QPushButton
    export_excel_btn: QPushButton
    add_compare_btn: QPushButton
    compare_btn: QPushButton
    clear_compare_btn: QPushButton

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFixedWidth(270)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        self.main_layout = QVBoxLayout(container)
        self.main_layout.setContentsMargins(8, 8, 8, 8)
        self.main_layout.setSpacing(4)
        self.setWidget(container)

        _sb.build_body_params(self)
        _sb.build_segment_lengths(self)
        _sb.build_barbell(self)
        _sb.build_optimization(self)
        _sb.build_buttons(self)
        _sb.build_progress_panel(self)
        _sb.build_results(self)
        self.main_layout.addStretch()

    def is_3d_mode(self) -> bool:
        """Return True if the 3D model is selected."""
        return self.model_combo.currentIndex() == 1

    def connect_action_handlers(self, handlers: Mapping[str, Callable[..., None]]) -> None:
        """Connect sidebar action signals to handlers supplied by the main window."""
        self.optimize_current.connect(handlers["optimize_current"])
        self.optimize_both.connect(handlers["optimize_both"])
        self.cancel_requested.connect(handlers["cancel_requested"])
        self.export_requested.connect(handlers["export_requested"])
        self.reset_requested.connect(handlers["reset_requested"])
        self.save_solution_requested.connect(handlers["save_solution_requested"])
        self.load_solution_requested.connect(handlers["load_solution_requested"])
        self.export_video_requested.connect(handlers["export_video_requested"])
        self.export_plots_requested.connect(handlers["export_plots_requested"])
        self.export_excel_requested.connect(handlers["export_excel_requested"])
        self.add_comparison_requested.connect(handlers["add_comparison_requested"])
        self.compare_trials_requested.connect(handlers["compare_trials_requested"])
        self.clear_comparison_requested.connect(handlers["clear_comparison_requested"])

    def show_optimizing(self) -> None:
        _st.show_optimizing(self)
        self.cancel_btn.setEnabled(True)
        self.cancel_btn.setToolTip("Cancel the currently running optimization (Esc)")
        self.opt_btn.setToolTip("Optimization currently in progress. Please wait or cancel.")
        self.both_btn.setToolTip("Optimization currently in progress. Please wait or cancel.")

    def show_idle(self) -> None:
        _st.show_idle(self)

    def update_progress(self, report) -> None:
        _st.update_progress(self, report)

    def get_body_model(self):
        return _st.get_body_model(self)

    def reset_defaults(self) -> None:
        _st.reset_defaults(self)

    # ------------------------------------------------------------------
    # Façade methods — encapsulate widget-chain access so callers do not
    # need to traverse into individual child widgets (issue #263 — Law of
    # Demeter / deep object traversal).
    # ------------------------------------------------------------------

    def get_optimization_params(self) -> tuple[float, float, float]:
        """Return ``(bar_kg, duration_s, smoothness)`` from the current slider state."""
        return (
            self.bar_slider.value(),
            self.dur_slider.value(),
            self.smooth_slider.value(),
        )

    def get_segment_multipliers(self) -> dict[str, float]:
        """Return the segment-length multiplier dict from the current slider state."""
        return {
            "lower_leg": self.ll_slider.value(),
            "upper_leg": self.ul_slider.value(),
            "torso": self.to_slider.value(),
        }

    def set_progress_done(self, elapsed_str: str, n_evals: int) -> None:
        """Mark the progress bar as complete and update the label."""
        self.progress.setValue(100)
        self.prog_label.setText(f"Done in {elapsed_str} ({n_evals} evals)")

    def set_cancelled(self) -> None:
        """Update UI to the cancelled state."""
        self.prog_label.setText("Cancelled")
        self.cancel_btn.setEnabled(True)

    def set_stall_message(self, msg: str) -> None:
        """Show *msg* in the stall label."""
        self.stall_label.setText(msg)
        self.stall_label.setVisible(True)

    def clear_stall_message(self) -> None:
        """Hide the stall label."""
        self.stall_label.setVisible(False)

    def set_result_label(self, text: str) -> None:
        """Set the result summary text."""
        self.result_label.setText(text)

    def enable_post_run_buttons(self) -> None:
        """Enable export/save/compare buttons after a successful run."""
        self.export_btn.setEnabled(True)
        self.export_btn.setToolTip("Export kinematics and forces to CSV")
        self.save_btn.setEnabled(True)
        self.save_btn.setToolTip("Save the current trajectory solution to file")
        self.export_video_btn.setEnabled(True)
        self.export_video_btn.setToolTip("Export the current animation to a GIF file")
        self.export_plots_btn.setEnabled(True)
        self.export_plots_btn.setToolTip("Export the current plots to PNG/PDF files")
        self.export_excel_btn.setEnabled(True)
        self.export_excel_btn.setToolTip("Export results to an Excel workbook (.xlsx)")
        self.add_compare_btn.setEnabled(True)
        self.add_compare_btn.setToolTip("Add current trial to the comparison set")

    def get_body_params_dict(self) -> dict[str, object]:
        """Return the current body parameter dict suitable for JSON serialisation."""
        return {
            "body_mass": self.mass_slider.value(),
            "height": self.height_slider.value(),
            "seg_multipliers": self.get_segment_multipliers(),
        }

    def get_comparison_trial_data(self) -> tuple[dict[str, object], float]:
        """Return the comparison payload without exposing widget internals."""
        return self.get_body_params_dict(), self.bar_slider.value()

    def get_comparison_context(self) -> tuple[float, dict[str, object]]:
        """Return bar mass and body parameters for trial comparison records."""
        body_params, bar_mass = self.get_comparison_trial_data()
        return bar_mass, body_params

    def set_comparison_available(self, available: bool) -> None:
        """Enable or disable the comparison action."""
        self.compare_btn.setEnabled(available)
        if available:
            self.compare_btn.setToolTip("Open dialog to compare saved trials")
        else:
            self.compare_btn.setToolTip("Add multiple trials to comparison first")

    def set_clear_comparison_available(self, available: bool) -> None:
        """Enable or disable the clear comparison action."""
        self.clear_compare_btn.setEnabled(available)
        if available:
            self.clear_compare_btn.setToolTip("Clear all trials currently saved for comparison")
        else:
            self.clear_compare_btn.setToolTip("No trials currently saved to clear")

    def set_cancellation_available(self, available: bool) -> None:
        """Enable or disable the cancellation action."""
        self.cancel_btn.setEnabled(available)
        if not available:
            self.cancel_btn.setToolTip("Cancellation already requested, shutting down safely...")
        else:
            self.cancel_btn.setToolTip("Cancel the currently running optimization (Esc)")

    def set_cancelling(self) -> None:
        """Immediately reflect cancellation in the UI and flush pending events.

        Disables the Run buttons and changes the cancel button text to
        "Canceling…" so the user gets instant visual feedback, then calls
        ``QApplication.processEvents()`` to keep the UI responsive while the
        background thread winds down.
        """
        from PyQt6.QtWidgets import QApplication

        self.opt_btn.setEnabled(False)
        self.both_btn.setEnabled(False)
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.setText("Canceling…")
        self.cancel_btn.setToolTip("Cancellation already requested, shutting down safely...")
        QApplication.processEvents()
