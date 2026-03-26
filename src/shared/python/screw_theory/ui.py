"""Screw Theory generic UI Tab component.

Implements a unified visualization settings toggle for Instantaneous Screw Axes (ISA).
Used across MuJoCo, Pendulums, Drake, OpenSim, MyoSuite, and Pinocchio.
"""

from __future__ import annotations

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class ScrewVisualizationTab(QtWidgets.QWidget):
    """Reusable layout/tab for rendering Screw Axes and Twists across varying engines.

    Signals:
        visualization_changed (bool): Emitted when the "Show Screw Axis" toggle changes.
        target_body_changed (str): Emitted when the user specifies a specific body
            for tracking its screw axis visually.
    """

    visualization_changed = QtCore.pyqtSignal(bool)
    target_body_changed = QtCore.pyqtSignal(str)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        # Screw Axis Controls Group
        group = QtWidgets.QGroupBox("Screw Theory Kinematics")
        group_layout = QtWidgets.QVBoxLayout(group)

        self.show_screw_axis_cb = QtWidgets.QCheckBox("Show Instantaneous Screw Axis (ISA) Motion")
        self.show_screw_axis_cb.setToolTip(
            "Renders the pitch, direction, and center of screw motion for the reference frame."
        )
        self.show_screw_axis_cb.stateChanged.connect(self._on_toggle)
        group_layout.addWidget(self.show_screw_axis_cb)

        # Target Body Input/Combo Box
        body_layout = QtWidgets.QHBoxLayout()
        body_layout.addWidget(QtWidgets.QLabel("Target Body:"))

        self.target_body_input = QtWidgets.QLineEdit()
        self.target_body_input.setPlaceholderText("e.g. club_head")
        self.target_body_input.setToolTip("Name of the reference frame to analyze (ISA).")
        self.target_body_input.returnPressed.connect(self._on_body_submit)

        body_layout.addWidget(self.target_body_input)
        group_layout.addLayout(body_layout)

        layout.addWidget(group)
        layout.addStretch()

    def _on_toggle(self, state: int) -> None:
        is_checked = state == QtCore.Qt.CheckState.Checked.value
        logger.info(f"Screw Axis Visualization toggled: {is_checked}")
        self.visualization_changed.emit(is_checked)

    def _on_body_submit(self) -> None:
        body_name = self.target_body_input.text().strip()
        if body_name:
            logger.info(f"Screw Axis Target Body selected: {body_name}")
            self.target_body_changed.emit(body_name)

    def is_active(self) -> bool:
        """Return True if screw axis visualization is requested."""
        return self.show_screw_axis_cb.isChecked()

    def get_target_body(self) -> str:
        """Return the target body name."""
        return self.target_body_input.text().strip()
