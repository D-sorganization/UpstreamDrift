"""Sub-widgets for the GripModellingTab.

Contains:
- PressureVisualizationWidget: 2D heatmap of grip pressure distribution
- ContactMetricsWidget: summary metrics for contact forces
"""

from __future__ import annotations

import numpy as np
from PyQt6 import QtCore, QtGui, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.grip_contact_model import PressureVisualizationData

logger = get_logger(__name__)


class PressureVisualizationWidget(QtWidgets.QWidget):
    """Widget for visualizing grip pressure distribution.

    Issue #757: Pressure distribution visualization available in the UI.
    Displays pressure as a 2D heatmap (unwrapped grip cylinder).
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize pressure visualization widget."""
        super().__init__(parent)
        self.setMinimumSize(200, 150)
        self.pressure_data: PressureVisualizationData | None = None

        # Color map (blue -> green -> yellow -> red)
        self.color_stops = [
            (0.0, QtGui.QColor(0, 0, 255)),  # Blue (low)
            (0.33, QtGui.QColor(0, 255, 0)),  # Green
            (0.66, QtGui.QColor(255, 255, 0)),  # Yellow
            (1.0, QtGui.QColor(255, 0, 0)),  # Red (high)
        ]

    def update_pressure(self, data: PressureVisualizationData) -> None:
        """Update displayed pressure data.

        Args:
            data: New pressure visualization data
        """
        if data is None:
            raise ValueError("data must be provided")
        self.pressure_data = data
        self.update()

    def clear(self) -> None:
        """Clear pressure display."""
        self.pressure_data = None
        self.update()

    def _get_color_for_value(self, normalized_value: float) -> QtGui.QColor:
        """Get color from gradient for normalized value [0, 1]."""
        if normalized_value is None:
            raise ValueError("normalized_value must be provided")
        normalized_value = max(0.0, min(1.0, normalized_value))

        # Find surrounding color stops
        for i in range(len(self.color_stops) - 1):
            t1, c1 = self.color_stops[i]
            t2, c2 = self.color_stops[i + 1]

            if t1 <= normalized_value <= t2:
                # Interpolate
                t = (normalized_value - t1) / (t2 - t1) if t2 > t1 else 0
                r = int(c1.red() + t * (c2.red() - c1.red()))
                g = int(c1.green() + t * (c2.green() - c1.green()))
                b = int(c1.blue() + t * (c2.blue() - c1.blue()))
                return QtGui.QColor(r, g, b)

        return self.color_stops[-1][1]

    def paintEvent(self, event: QtGui.QPaintEvent | None) -> None:
        """Paint the pressure visualization."""
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)

        rect = self.rect()
        painter.fillRect(rect, QtGui.QColor(40, 40, 40))

        if self.pressure_data is None or len(self.pressure_data.pressures) == 0:
            painter.setPen(QtGui.QColor(150, 150, 150))
            painter.drawText(
                rect, QtCore.Qt.AlignmentFlag.AlignCenter, "No contact data"
            )
            return

        # Draw title
        painter.setPen(QtGui.QColor(255, 255, 255))
        painter.drawText(10, 20, f"Max: {self.pressure_data.max_pressure:.0f} Pa")
        painter.drawText(10, 35, f"Mean: {self.pressure_data.mean_pressure:.0f} Pa")

        # Draw pressure points
        margin = 50
        plot_rect = rect.adjusted(margin, margin, -margin, -20)

        if plot_rect.width() <= 0 or plot_rect.height() <= 0:
            return

        # Map grip axis position to x, angular position to y
        axis_pos = self.pressure_data.grip_axis_positions
        angles = self.pressure_data.angular_positions

        if len(axis_pos) == 0:
            return

        # Normalize positions for display
        axis_min, axis_max = np.min(axis_pos), np.max(axis_pos)
        axis_range = axis_max - axis_min if axis_max > axis_min else 1.0

        for i in range(len(self.pressure_data.pressures)):
            # Map to widget coordinates
            x_norm = (axis_pos[i] - axis_min) / axis_range
            y_norm = (angles[i] + np.pi) / (2 * np.pi)

            x = int(plot_rect.left() + x_norm * plot_rect.width())
            y = int(plot_rect.top() + y_norm * plot_rect.height())

            # Size based on pressure (larger = more pressure)
            size = int(5 + 15 * self.pressure_data.normalized_pressures[i])

            # Color based on pressure
            norm_val = self.pressure_data.normalized_pressures[i]
            color = self._get_color_for_value(norm_val)
            painter.setBrush(QtGui.QBrush(color))
            painter.setPen(QtCore.Qt.PenStyle.NoPen)
            painter.drawEllipse(x - size // 2, y - size // 2, size, size)

        # Draw axes labels
        painter.setPen(QtGui.QColor(200, 200, 200))
        painter.drawText(plot_rect.left(), rect.bottom() - 5, "Butt")
        painter.drawText(plot_rect.right() - 20, rect.bottom() - 5, "Tip")

        # Draw color legend
        legend_rect = QtCore.QRect(rect.right() - 30, margin, 15, plot_rect.height())
        for i in range(legend_rect.height()):
            t = i / legend_rect.height()
            color = self._get_color_for_value(1.0 - t)  # Flip so high is at top
            painter.setPen(color)
            painter.drawLine(
                legend_rect.left(),
                legend_rect.top() + i,
                legend_rect.right(),
                legend_rect.top() + i,
            )


class ContactMetricsWidget(QtWidgets.QWidget):
    """Widget displaying contact metrics summary.

    Issue #757: Shows contact forces, slip detection status.
    """

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize metrics widget."""
        super().__init__(parent)
        layout = QtWidgets.QFormLayout(self)

        self.lbl_normal_force = QtWidgets.QLabel("0.0 N")
        self.lbl_tangent_force = QtWidgets.QLabel("0.0 N")
        self.lbl_num_contacts = QtWidgets.QLabel("0")
        self.lbl_slip_status = QtWidgets.QLabel("No slip")
        self.lbl_slip_margin = QtWidgets.QLabel("N/A")
        self.lbl_equilibrium = QtWidgets.QLabel("Unknown")

        layout.addRow("Normal Force:", self.lbl_normal_force)
        layout.addRow("Tangent Force:", self.lbl_tangent_force)
        layout.addRow("Active Contacts:", self.lbl_num_contacts)
        layout.addRow("Slip Status:", self.lbl_slip_status)
        layout.addRow("Min Slip Margin:", self.lbl_slip_margin)
        layout.addRow("Equilibrium:", self.lbl_equilibrium)

    def update_metrics(
        self,
        normal_force: float,
        tangent_force: float,
        num_contacts: int,
        num_slipping: int,
        slip_margin: float,
        equilibrium: bool,
    ) -> None:
        """Update displayed metrics."""
        if normal_force is None:
            raise ValueError("normal_force must be provided")
        self.lbl_normal_force.setText(f"{normal_force:.1f} N")
        self.lbl_tangent_force.setText(f"{tangent_force:.1f} N")
        self.lbl_num_contacts.setText(str(num_contacts))

        if num_slipping > 0:
            self.lbl_slip_status.setText(f"SLIPPING ({num_slipping})")
            self.lbl_slip_status.setStyleSheet("color: red; font-weight: bold;")
        else:
            self.lbl_slip_status.setText("No slip")
            self.lbl_slip_status.setStyleSheet("color: green;")

        self.lbl_slip_margin.setText(f"{slip_margin:.2%}")

        if equilibrium:
            self.lbl_equilibrium.setText("Stable")
            self.lbl_equilibrium.setStyleSheet("color: green;")
        else:
            self.lbl_equilibrium.setText("Unstable")
            self.lbl_equilibrium.setStyleSheet("color: orange;")
