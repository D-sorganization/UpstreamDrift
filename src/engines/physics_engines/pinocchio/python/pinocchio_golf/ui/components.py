"""UI Components for Pinocchio Golf GUI."""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6 import QtWidgets

if TYPE_CHECKING:
    from ..gui import PinocchioGUI


class GUIBuilder:
    """Helper class to build GUI sections."""

    @staticmethod
    def setup_toolbar(gui: PinocchioGUI, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the top bar with model selector, load button, and mode selector."""
        if not (gui is not None):
            raise ValueError("gui must be provided")
        toolbar = QtWidgets.QHBoxLayout()

        gui.model_combo = QtWidgets.QComboBox()
        gui._populate_model_combo()
        toolbar.addWidget(QtWidgets.QLabel("Model:"))
        toolbar.addWidget(gui.model_combo)

        load_btn = QtWidgets.QPushButton("Load Model")
        load_btn.clicked.connect(gui._on_load_model)
        toolbar.addWidget(load_btn)

        gui.mode_selector = QtWidgets.QComboBox()
        gui.mode_selector.addItems(["Configuration", "Simulation"])
        gui.mode_selector.currentTextChanged.connect(gui._on_mode_changed)
        toolbar.addWidget(QtWidgets.QLabel("Mode:"))
        toolbar.addWidget(gui.mode_selector)

        layout.addLayout(toolbar)

    @staticmethod
    def setup_visualization_panel(
        gui: PinocchioGUI, parent_layout: QtWidgets.QVBoxLayout
    ) -> None:  # noqa: E501
        """Build the visualization group box."""
        if not (gui is not None):
            raise ValueError("gui must be provided")
        vis_group = QtWidgets.QGroupBox("Visualization Overlays")
        vis_layout = QtWidgets.QVBoxLayout()

        # ... sub-methods like _setup_overlay_checkboxes would go here ...

        vis_group.setLayout(vis_layout)
        parent_layout.addWidget(vis_group)
