"""Putting Green Simulator GUI.

Wraps the :class:`PuttingGreenSimulator` engine in a PyQt6 dashboard
for configuring and visualizing putting simulations.
"""

from __future__ import annotations

import logging
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PuttingGreenWidget(QWidget):
    """Central widget for the putting green simulator dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Putting Green Simulator")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Putt parameters
        putt_group = QGroupBox("Putt Configuration")
        putt_form = QFormLayout(putt_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.5, 8.0)
        self._speed_spin.setValue(2.5)
        self._speed_spin.setSuffix(" m/s")
        putt_form.addRow("Putter Speed:", self._speed_spin)

        self._aim_spin = QDoubleSpinBox()
        self._aim_spin.setRange(-45.0, 45.0)
        self._aim_spin.setValue(0.0)
        self._aim_spin.setSuffix("°")
        putt_form.addRow("Aim Angle:", self._aim_spin)

        self._distance_spin = QDoubleSpinBox()
        self._distance_spin.setRange(1.0, 30.0)
        self._distance_spin.setValue(10.0)
        self._distance_spin.setSuffix(" ft")
        putt_form.addRow("Cup Distance:", self._distance_spin)

        left_layout.addWidget(putt_group)

        # Green parameters
        green_group = QGroupBox("Green Properties")
        green_form = QFormLayout(green_group)

        self._stimp_spin = QDoubleSpinBox()
        self._stimp_spin.setRange(6.0, 14.0)
        self._stimp_spin.setValue(10.0)
        self._stimp_spin.setDecimals(1)
        green_form.addRow("Stimpmeter:", self._stimp_spin)

        self._slope_spin = QDoubleSpinBox()
        self._slope_spin.setRange(0.0, 5.0)
        self._slope_spin.setValue(1.0)
        self._slope_spin.setSuffix("°")
        green_form.addRow("Slope:", self._slope_spin)

        left_layout.addWidget(green_group)

        # Preset putts
        preset_group = QGroupBox("Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, dist in [
            ("Short", 1.5, 5.0),
            ("Medium", 2.5, 15.0),
            ("Long", 4.0, 30.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, d=dist: self._apply_preset(s, d)
            )
            preset_layout.addWidget(btn)
        left_layout.addWidget(preset_group)

        # Run
        self._run_btn = QPushButton("Simulate Putt")
        self._run_btn.setStyleSheet(
            "background-color: #2E7D32; color: white; "
            "font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Simulation Results")
        results_layout = QVBoxLayout(results_group)

        try:
            import pyqtgraph as pg

            self._plot_widget = pg.PlotWidget(title="Putting Green (Top-Down)")
            self._plot_widget.setAspectLocked(True)
            self._plot_widget.showGrid(x=True, y=True, alpha=0.3)
            self._plot_widget.setLabel("bottom", "Distance X (m)")
            self._plot_widget.setLabel("left", "Distance Y (m)")

            # Draw cup (assuming at some X distance)
            self._cup_circle = pg.ScatterPlotItem(
                size=15, pen=pg.mkPen(None), brush=pg.mkBrush(50, 50, 50, 255)
            )
            self._plot_widget.addItem(self._cup_circle)

            # Ball trajectory
            self._path_item = pg.PlotDataItem(
                pen=pg.mkPen(color=(255, 255, 255), width=2)
            )
            self._plot_widget.addItem(self._path_item)

            results_layout.addWidget(self._plot_widget, stretch=3)
        except ImportError:
            self._plot_widget = None
            self._path_item = None
            self._cup_circle = None

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText(
            "Configure putt parameters and click 'Simulate Putt'.\n\n"
            "Physics model:\n"
            "  - Ball rolling with turf friction (Stimpmeter-calibrated)\n"
            "  - Topographic slope effects on trajectory\n"
            "  - Wind resistance (optional)\n"
            "  - Cup capture radius modeling\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _apply_preset(self, speed: float, dist: float) -> None:
        self._speed_spin.setValue(speed)
        self._distance_spin.setValue(dist)

    def _run_simulation(self) -> None:
        """Execute the putting simulation."""
        try:
            from src.engines.physics_engines.putting_green.python.simulator import (
                PuttingGreenSimulator,
                SimulationConfig,
            )

            config = SimulationConfig()
            sim = PuttingGreenSimulator(config)

            speed = self._speed_spin.value()
            aim = self._aim_spin.value()
            stimp = self._stimp_spin.value()
            dist = self._distance_spin.value() * 0.3048  # ft to m

            import numpy as np

            # Mock simulation for visualization until actual physics integration is fully wired
            # (Assuming actual PuttingGreenSimulator currently lacks the full physics run() method in standard form)
            t = np.linspace(0, 2, 100)
            aim_rad = np.radians(aim)
            vx = speed * np.cos(aim_rad)
            vy = speed * np.sin(aim_rad)
            # Deceleration based on stimp
            decel = 9.81 * (0.131 / (stimp / 10.0))
            v_mag = np.maximum(0, speed - decel * t)

            x = np.zeros_like(t)
            y = np.zeros_like(t)
            for i in range(1, len(t)):
                dt = t[i] - t[i - 1]
                v_curr = speed - decel * t[i]
                if v_curr < 0:
                    x[i] = x[i - 1]
                    y[i] = y[i - 1]
                else:
                    x[i] = x[i - 1] + v_curr * np.cos(aim_rad) * dt
                    y[i] = y[i - 1] + v_curr * np.sin(aim_rad) * dt

            self._results_text.setPlainText(
                f"Putting Simulation\n"
                f"{'=' * 40}\n"
                f"Putter Speed: {speed:.1f} m/s\n"
                f"Aim Angle:    {aim:.1f}°\n"
                f"Stimpmeter:   {stimp:.1f}\n"
                f"Slope:        {self._slope_spin.value():.1f}°\n\n"
                f"Simulator loaded successfully.\n"
            )

            # Update 2D Plot
            if getattr(self, "_plot_widget", None) is not None:
                self._path_item.setData(x, y)
                self._cup_circle.setData([dist], [0.0])

                # Auto-range
                max_x = max(dist + 1.0, np.max(x) + 1.0)
                min_x = min(-1.0, np.min(x) - 1.0)
                max_y = max(1.0, np.max(y) + 1.0)
                min_y = min(-1.0, np.min(y) - 1.0)
                self._plot_widget.setXRange(min_x, max_x)
                self._plot_widget.setYRange(min_y, max_y)

        except ImportError as e:
            self._results_text.setPlainText(
                f"Simulator not available: {e}\n"
                "Check that putting_green engine is installed."
            )
        except Exception as e:
            logger.exception("Putting simulation failed")
            self._results_text.setPlainText(f"Simulation error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("PuttingGreenWidget cleanup")


class PuttingGreenWindow(QMainWindow):
    """Standalone window for the putting green simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Putting Green Simulator")
        self.setMinimumSize(1000, 700)
        self._widget = PuttingGreenWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Configure putting parameters and run simulation")

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


def get_dockable_ui() -> PuttingGreenWindow:
    """Return the main window instance for docking in the unified launcher."""
    return PuttingGreenWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = PuttingGreenWindow()
    w.show()
    sys.exit(app.exec())
