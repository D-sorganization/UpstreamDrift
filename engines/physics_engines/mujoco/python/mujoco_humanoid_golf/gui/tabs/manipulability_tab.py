"""Manipulability Analysis Tab for Advanced GUI.

Allows users to inspect Force and Mobility matrices and visualize ellipsoids
 for multiple body parts simultaneously (Left Hand, Club Head, etc.).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6 import QtCore, QtWidgets

from ..manipulability import ManipulabilityAnalyzer

logger = logging.getLogger(__name__)


class ManipulabilityTab(QtWidgets.QWidget):
    """Tab for manipulating and visualizing force/mobility matrices."""

    def __init__(self, sim_widget: Any, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.analyzer: ManipulabilityAnalyzer | None = None
        self.body_checkboxes: dict[str, QtWidgets.QCheckBox] = {}
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_analysis)

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        # 1. Visualization Controls Group
        vis_group = QtWidgets.QGroupBox("Visualization Settings")
        vis_layout = QtWidgets.QHBoxLayout()

        self.chk_show_mobility = QtWidgets.QCheckBox("Show Mobility Ellipsoids (Green)")
        self.chk_show_mobility.setChecked(True)
        self.chk_show_force = QtWidgets.QCheckBox("Show Force Ellipsoids (Red)")
        self.chk_show_force.setChecked(False)

        vis_layout.addWidget(self.chk_show_mobility)
        vis_layout.addWidget(self.chk_show_force)
        vis_group.setLayout(vis_layout)
        layout.addWidget(vis_group)

        # 2. Body Selection Group
        body_group = QtWidgets.QGroupBox("Body Points of Interest")
        self.body_layout = QtWidgets.QGridLayout()
        # Populated dynamically when model loads
        body_group.setLayout(self.body_layout)
        layout.addWidget(body_group)

        # 3. Matrix Inspection Group
        matrix_group = QtWidgets.QGroupBox("Live Matrix Inspection")
        matrix_layout = QtWidgets.QVBoxLayout()

        self.chk_show_matrices = QtWidgets.QCheckBox("Enable Live Matrix Updates")
        self.chk_show_matrices.setChecked(False)  # Off by default to save perf
        matrix_layout.addWidget(self.chk_show_matrices)

        self.matrix_text = QtWidgets.QTextEdit()
        self.matrix_text.setReadOnly(True)
        self.matrix_text.setFontFamily("Consolas")
        self.matrix_text.setMinimumHeight(200)
        matrix_layout.addWidget(self.matrix_text)

        matrix_group.setLayout(matrix_layout)
        layout.addWidget(matrix_group)

        layout.addStretch()

        # Start timer (15 FPS update for UI is enough, visuals run in sim loop?)
        # Actually, we'll push visuals from here to sim_widget's meshcat adapter
        self.timer.start(66)  # ~15 FPS

    def on_model_loaded(self) -> None:
        """Called when a new model is loaded into the simulation."""
        if self.sim_widget.model is None:
            return

        # Initialize Analyzer
        # We use strict _perturb_data if available for thread safety?
        # SimWidget usually runs on main thread. We can use self.sim_widget.data
        # provided we are careful not to perturb it during a step.
        # Best practice: use the widget's provided data which is synchronized.
        self.analyzer = ManipulabilityAnalyzer(self.sim_widget.model, self.sim_widget.data)

        # Populate Body Checkboxes
        self._populate_body_checkboxes()

    def _populate_body_checkboxes(self) -> None:
        """Clear and repopulate checkboxes based on model bodies."""
        # Clear existing
        for i in reversed(range(self.body_layout.count())):
            w = self.body_layout.itemAt(i).widget()
            if w:
                w.setParent(None)
        self.body_checkboxes.clear()

        if self.analyzer:
            relevant_bodies = self.analyzer.find_golf_bodies()

            # Defines defaults to check
            defaults = ["hand", "club", "head", "wrist"]

            row, col = 0, 0
            for name in relevant_bodies:
                chk = QtWidgets.QCheckBox(name)
                # Check if default
                if any(d in name.lower() for d in defaults):
                    chk.setChecked(True)

                self.body_checkboxes[name] = chk
                self.body_layout.addWidget(chk, row, col)

                col += 1
                if col > 2:  # 3 columns
                    col = 0
                    row += 1


    def _update_analysis(self) -> None:
        """Periodic update loop."""
        if not self.isVisible() or not self.analyzer or not self.sim_widget.model:
            return

        # Check if we can draw
        meshcat = self.sim_widget.meshcat_adapter

        # Prepare text report
        report_lines = []

        # Clear old ellipsoids if needed?
        # Meshcat updates in-place if names match.
        # But if we uncheck a body, we should hide it.
        # For simplicity, we might iterate all *checked* bodies.

        active_bodies = [
            name for name, chk in self.body_checkboxes.items() if chk.isChecked()
        ]

        if not active_bodies:
            if meshcat:
                meshcat.clear_ellipsoids()
            return

        for name in active_bodies:
            res = self.analyzer.compute_metrics(name)
            if not res:
                continue

            # Draw Mobility?
            if self.chk_show_mobility.isChecked() and meshcat:
                meshcat.draw_ellipsoid(
                    f"{name}_mobility",
                    res.velocity_ellipsoid.center,
                    res.velocity_ellipsoid.axes,
                    res.velocity_ellipsoid.radii,
                    color=0x00FF00,
                    opacity=0.3
                )
            
            # Draw Force?
            if self.chk_show_force.isChecked() and meshcat:
                # Force ellipsoid often visualized at same center
                meshcat.draw_ellipsoid(
                    f"{name}_force",
                    res.force_ellipsoid.center,
                    res.force_ellipsoid.axes,
                    res.force_ellipsoid.radii,
                    color=0xFF0000,
                    opacity=0.3
                )

            # Text Report?
            if self.chk_show_matrices.isChecked():
                report_lines.append(f"=== {name} ===")
                report_lines.append(f"Cond Number: {res.condition_number:.2f}")
                report_lines.append(f"Manip Index: {res.manipulability_index:.4f}")

                report_lines.append("Mobility (JJ^T):")
                with np.printoptions(precision=3, suppress=True):
                    report_lines.append(str(res.mobility_matrix))

                report_lines.append("Force ((JJ^T)^-1):")
                with np.printoptions(precision=3, suppress=True):
                    report_lines.append(str(res.force_matrix))
                report_lines.append("")

        # Hide unchecked?
        # A full clear-and-redraw is safer for toggles, but simpler:
        # We rely on meshcat persistence. If user unchecks, we should delete.
        # I'll implement a simple prune in meshcat_adapter later if needed.
        # For now, it just stops updating.

        if self.chk_show_matrices.isChecked():
            self.matrix_text.setPlainText("\n".join(report_lines))

