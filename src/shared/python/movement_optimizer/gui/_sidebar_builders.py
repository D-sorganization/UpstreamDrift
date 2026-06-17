# Copyright (c) 2024-2026 D-sorganization
# SPDX-License-Identifier: MIT
#
# Movement-Optimizer — see LICENSE for full terms.
#

"""Builder functions for ParameterSidebar widget sections."""

from __future__ import annotations

import logging

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt6.QtWidgets import (
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
)

from ..constants import BAR_MASS_KG
from ..i18n import tr
from ..rendering import Palette
from .labelled_slider import LabelledSlider
from .wheel_blocker import suppress_wheel_events

logger = logging.getLogger(__name__)


def build_body_params(sidebar) -> None:
    grp = QGroupBox("Body Parameters")
    lay = QVBoxLayout(grp)
    sidebar.mass_slider = LabelledSlider(
        tr("Body Mass"),
        40,
        150,
        75,
        "kg",
        0,
        tooltip="Total body mass in kilograms (40-150 kg)",
    )
    sidebar.height_slider = LabelledSlider(
        "Height",
        1.40,
        2.10,
        1.75,
        "m",
        2,
        tooltip="Standing height in metres (1.40-2.10 m)",
    )
    lay.addWidget(sidebar.mass_slider)
    lay.addWidget(sidebar.height_slider)
    sidebar.main_layout.addWidget(grp)


def build_segment_lengths(sidebar) -> None:
    grp = QGroupBox("Segment Lengths")
    lay = QVBoxLayout(grp)
    hint = QLabel("Multipliers on base length")
    hint.setProperty("class", "dim")
    lay.addWidget(hint)
    sidebar.ll_slider = LabelledSlider(
        "Lower Leg",
        0.70,
        1.30,
        1.00,
        "x",
        2,
        tooltip="Lower leg length multiplier relative to base length (0.70-1.30 x)",
    )
    sidebar.ul_slider = LabelledSlider(
        "Upper Leg",
        0.70,
        1.30,
        1.00,
        "x",
        2,
        tooltip="Upper leg length multiplier relative to base length (0.70-1.30 x)",
    )
    sidebar.to_slider = LabelledSlider(
        "Torso",
        0.70,
        1.30,
        1.00,
        "x",
        2,
        tooltip="Torso length multiplier relative to base length (0.70-1.30 x)",
    )
    lay.addWidget(sidebar.ll_slider)
    lay.addWidget(sidebar.ul_slider)
    lay.addWidget(sidebar.to_slider)
    sidebar.main_layout.addWidget(grp)


def build_barbell(sidebar) -> None:
    grp = QGroupBox("Barbell")
    lay = QVBoxLayout(grp)
    hint = QLabel(f"Olympic bar = {BAR_MASS_KG:.0f} kg")
    hint.setProperty("class", "dim")
    lay.addWidget(hint)
    sidebar.bar_slider = LabelledSlider(
        "Total Bar + Plates",
        0,
        300,
        60,
        "kg",
        0,
        tooltip="Total barbell plus plates mass in kilograms (0-300 kg)",
    )
    lay.addWidget(sidebar.bar_slider)
    sidebar.bar_depth_slider = LabelledSlider(
        "Bar Back Offset",
        0.0,
        0.4,
        0.0,
        "m",
        2,
        tooltip="Horizontal distance the bar sits behind the shoulder joint (0.00-0.40 m)",
    )
    lay.addWidget(sidebar.bar_depth_slider)
    sidebar.bar_height_slider = LabelledSlider(
        "Bar Drop Offset",
        0.0,
        0.4,
        0.0,
        "m",
        2,
        tooltip="Vertical distance the bar drops below the shoulder joint (0.00-0.40 m)",
    )
    lay.addWidget(sidebar.bar_height_slider)
    sidebar.main_layout.addWidget(grp)


def build_optimization(sidebar) -> None:
    grp = QGroupBox("Optimization")
    lay = QVBoxLayout(grp)
    model_row = QHBoxLayout()
    model_label = QLabel("Model:")
    model_row.addWidget(model_label)
    sidebar.model_combo = QComboBox()
    sidebar.model_combo.addItems(["2D Sagittal", "3D Bilateral"])
    sidebar.model_combo.setAccessibleName("Model")
    suppress_wheel_events(sidebar.model_combo)
    model_label.setBuddy(sidebar.model_combo)
    idx_3d = sidebar.model_combo.findText("3D Bilateral")
    if idx_3d >= 0:
        from PyQt6.QtGui import QStandardItemModel

        model = sidebar.model_combo.model()
        if isinstance(model, QStandardItemModel):
            item = model.item(idx_3d)
            if item is not None:
                item.setToolTip(
                    "3D Bilateral: renders the optimised 2D trajectory "
                    "in a bilateral (two-leg) 3D pose."
                )
    model_row.addWidget(sidebar.model_combo)
    lay.addLayout(model_row)
    sidebar.dur_slider = LabelledSlider(
        "Duration",
        0.5,
        5.0,
        2.0,
        "s",
        1,
        tooltip="Total movement duration in seconds (0.5-5.0 s)",
    )
    sidebar.smooth_slider = LabelledSlider(
        "Smoothness",
        0.1,
        5.0,
        1.0,
        "x",
        1,
        tooltip="Smoothness penalty multiplier — higher values produce smoother joint torques (0.1-5.0 x)",
    )
    hint = QLabel("Higher = smoother torques")
    hint.setProperty("class", "dim")
    lay.addWidget(sidebar.dur_slider)
    lay.addWidget(sidebar.smooth_slider)
    lay.addWidget(hint)
    sidebar.main_layout.addWidget(grp)


def build_buttons(sidebar) -> None:
    sidebar.opt_btn = QPushButton(tr("Run Optimization"))
    sidebar.opt_btn.setProperty("class", "primary")
    sidebar.opt_btn.setToolTip(
        "Start trajectory optimization for the currently selected exercise tab (Ctrl+R)"
    )
    sidebar.opt_btn.setAccessibleName("Optimize Current Tab")
    sidebar.opt_btn.setAccessibleDescription(
        "Start trajectory optimization for the currently selected exercise tab."
    )
    sidebar.opt_btn.setShortcut("Ctrl+R")
    sidebar.opt_btn.clicked.connect(sidebar.optimize_current.emit)
    sidebar.main_layout.addWidget(sidebar.opt_btn)

    sidebar.both_btn = QPushButton("Optimize All Tabs")
    sidebar.both_btn.setToolTip(
        "Start trajectory optimization sequentially for all exercise tabs (Ctrl+Shift+R)"
    )
    sidebar.both_btn.setAccessibleName("Optimize All Tabs")
    sidebar.both_btn.setAccessibleDescription(
        "Start trajectory optimization sequentially for every exercise tab."
    )
    sidebar.both_btn.setShortcut("Ctrl+Shift+R")
    sidebar.both_btn.clicked.connect(sidebar.optimize_both.emit)
    sidebar.main_layout.addWidget(sidebar.both_btn)

    sidebar.cancel_btn = QPushButton(tr("Cancel"))
    sidebar.cancel_btn.setProperty("class", "cancel")
    sidebar.cancel_btn.setToolTip("Cancel the currently running optimization (Esc)")
    sidebar.cancel_btn.setAccessibleName("Cancel")
    sidebar.cancel_btn.setAccessibleDescription("Cancel the currently running optimization.")
    sidebar.cancel_btn.setShortcut("Esc")
    sidebar.cancel_btn.clicked.connect(sidebar.cancel_requested.emit)
    sidebar.cancel_btn.setVisible(False)
    sidebar.main_layout.addWidget(sidebar.cancel_btn)


def build_progress_panel(sidebar) -> None:
    grp = QGroupBox("Progress")
    lay = QVBoxLayout(grp)

    sidebar.progress = QProgressBar()
    sidebar.progress.setRange(0, 100)
    sidebar.progress.setValue(0)
    lay.addWidget(sidebar.progress)

    sidebar.prog_label = QLabel("")
    sidebar.prog_label.setProperty("class", "dim")
    lay.addWidget(sidebar.prog_label)

    sidebar.iter_label = QLabel("")
    sidebar.iter_label.setProperty("class", "dim")
    lay.addWidget(sidebar.iter_label)

    sidebar.cost_label = QLabel("")
    sidebar.cost_label.setProperty("class", "dim")
    lay.addWidget(sidebar.cost_label)

    sidebar.improve_label = QLabel("")
    sidebar.improve_label.setProperty("class", "dim")
    lay.addWidget(sidebar.improve_label)

    sidebar.elapsed_label = QLabel("")
    sidebar.elapsed_label.setProperty("class", "dim")
    lay.addWidget(sidebar.elapsed_label)

    sidebar.stall_label = QLabel("")
    sidebar.stall_label.setProperty("class", "stall-warn")
    sidebar.stall_label.setWordWrap(True)
    sidebar.stall_label.setVisible(False)
    lay.addWidget(sidebar.stall_label)

    sidebar.conv_fig = Figure(figsize=(2.4, 1.2), facecolor=Palette.BG_PANEL)
    sidebar.conv_canvas = FigureCanvas(sidebar.conv_fig)
    sidebar.conv_canvas.setFixedHeight(90)
    sidebar.conv_ax = sidebar.conv_fig.add_subplot(111)
    _style_conv_ax(sidebar)
    lay.addWidget(sidebar.conv_canvas)

    sidebar.main_layout.addWidget(grp)


def _style_conv_ax(sidebar) -> None:
    ax = sidebar.conv_ax
    ax.set_facecolor(Palette.BG_PLOT)
    ax.tick_params(colors=Palette.FG_DIM, which="both", labelsize=6)
    for sp in ("bottom", "left"):
        ax.spines[sp].set_color(Palette.FG_DIM)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("eval", fontsize=6, color=Palette.FG_DIM)
    ax.set_ylabel("cost", fontsize=6, color=Palette.FG_DIM)
    sidebar.conv_fig.tight_layout(pad=0.3)


def build_results(sidebar) -> None:
    grp = QGroupBox("Results")
    lay = QVBoxLayout(grp)
    sidebar.result_label = QLabel("(none)")
    sidebar.result_label.setProperty("class", "result")
    sidebar.result_label.setWordWrap(True)
    lay.addWidget(sidebar.result_label)
    sidebar.main_layout.addWidget(grp)

    sidebar.export_btn = QPushButton(tr("Export") + " CSV")
    sidebar.export_btn.setEnabled(False)
    sidebar.export_btn.setToolTip("Run optimization first to enable exporting kinematics to CSV")
    sidebar.export_btn.setAccessibleName("Export CSV")
    sidebar.export_btn.setAccessibleDescription("Export optimized kinematics to a CSV file.")
    sidebar.export_btn.clicked.connect(sidebar.export_requested.emit)
    sidebar.main_layout.addWidget(sidebar.export_btn)

    sidebar.reset_btn = QPushButton("Reset Defaults")
    sidebar.reset_btn.setToolTip("Reset all parameters to default values")
    sidebar.reset_btn.setAccessibleName("Reset Defaults")
    sidebar.reset_btn.setAccessibleDescription("Reset all parameters to their default values.")
    sidebar.reset_btn.clicked.connect(sidebar.reset_requested.emit)
    sidebar.main_layout.addWidget(sidebar.reset_btn)

    build_persistence_buttons(sidebar)
    build_export_buttons(sidebar)
    build_comparison_buttons(sidebar)


def build_persistence_buttons(sidebar) -> None:
    grp = QGroupBox("Solution Files")
    lay = QVBoxLayout(grp)
    sidebar.save_btn = QPushButton("Save Solution")
    sidebar.save_btn.setEnabled(False)
    sidebar.save_btn.setToolTip("Run optimization first to enable saving the trajectory solution")
    sidebar.save_btn.setAccessibleName("Save Solution")
    sidebar.save_btn.setAccessibleDescription("Save the current trajectory solution to a file.")
    sidebar.save_btn.clicked.connect(sidebar.save_solution_requested.emit)
    lay.addWidget(sidebar.save_btn)
    sidebar.load_btn = QPushButton("Load Solution")
    sidebar.load_btn.setToolTip("Load a previously saved trajectory solution file")
    sidebar.load_btn.setAccessibleName("Load Solution")
    sidebar.load_btn.setAccessibleDescription("Load a previously saved trajectory solution file.")
    sidebar.load_btn.clicked.connect(sidebar.load_solution_requested.emit)
    lay.addWidget(sidebar.load_btn)
    sidebar.main_layout.addWidget(grp)


def build_export_buttons(sidebar) -> None:
    grp = QGroupBox("Export Media")
    lay = QVBoxLayout(grp)
    sidebar.export_video_btn = QPushButton("Export Animation GIF")
    sidebar.export_video_btn.setEnabled(False)
    sidebar.export_video_btn.setToolTip("Run optimization first to enable exporting animation GIF")
    sidebar.export_video_btn.setAccessibleName("Export Animation GIF")
    sidebar.export_video_btn.setAccessibleDescription("Export the optimized animation as a GIF.")
    sidebar.export_video_btn.clicked.connect(sidebar.export_video_requested.emit)
    lay.addWidget(sidebar.export_video_btn)
    sidebar.export_plots_btn = QPushButton("Export Plots (PNG/PDF)")
    sidebar.export_plots_btn.setEnabled(False)
    sidebar.export_plots_btn.setToolTip("Run optimization first to enable exporting plots")
    sidebar.export_plots_btn.setAccessibleName("Export Plots")
    sidebar.export_plots_btn.setAccessibleDescription("Export analysis plots as PNG or PDF files.")
    sidebar.export_plots_btn.clicked.connect(sidebar.export_plots_requested.emit)
    lay.addWidget(sidebar.export_plots_btn)
    sidebar.export_excel_btn = QPushButton("Save as Excel (.xlsx)")
    sidebar.export_excel_btn.setEnabled(False)
    sidebar.export_excel_btn.setToolTip("Run optimization first to enable Excel export")
    sidebar.export_excel_btn.setAccessibleName("Save as Excel")
    sidebar.export_excel_btn.setAccessibleDescription(
        "Save optimized results to an Excel workbook."
    )
    sidebar.export_excel_btn.clicked.connect(sidebar.export_excel_requested.emit)
    lay.addWidget(sidebar.export_excel_btn)
    sidebar.main_layout.addWidget(grp)


def build_comparison_buttons(sidebar) -> None:
    grp = QGroupBox("Trial Comparison")
    lay = QVBoxLayout(grp)
    sidebar.add_compare_btn = QPushButton("Add to Comparison")
    sidebar.add_compare_btn.setEnabled(False)
    sidebar.add_compare_btn.setToolTip("Run optimization first to add current trial to comparison")
    sidebar.add_compare_btn.setAccessibleName("Add to Comparison")
    sidebar.add_compare_btn.setAccessibleDescription(
        "Add the current optimized trial to the comparison set."
    )
    sidebar.add_compare_btn.clicked.connect(sidebar.add_comparison_requested.emit)
    lay.addWidget(sidebar.add_compare_btn)
    sidebar.compare_btn = QPushButton("Compare Trials")
    sidebar.compare_btn.setEnabled(False)
    sidebar.compare_btn.setToolTip("Add multiple trials to comparison first")
    sidebar.compare_btn.setAccessibleName("Compare Trials")
    sidebar.compare_btn.setAccessibleDescription("Open the trial comparison dialog.")
    sidebar.compare_btn.clicked.connect(sidebar.compare_trials_requested.emit)
    lay.addWidget(sidebar.compare_btn)
    sidebar.clear_compare_btn = QPushButton("Clear Comparison")
    sidebar.clear_compare_btn.setToolTip("Clear all trials currently saved for comparison")
    sidebar.clear_compare_btn.setAccessibleName("Clear Comparison")
    sidebar.clear_compare_btn.setAccessibleDescription("Clear all trials from the comparison set.")
    sidebar.clear_compare_btn.clicked.connect(sidebar.clear_comparison_requested.emit)
    lay.addWidget(sidebar.clear_compare_btn)
    sidebar.main_layout.addWidget(grp)
