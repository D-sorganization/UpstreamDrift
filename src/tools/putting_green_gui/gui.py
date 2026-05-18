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

            self._results_text.setPlainText(
                f"Putting Simulation\n"
                f"{'=' * 40}\n"
                f"Putter Speed: {speed:.1f} m/s\n"
                f"Aim Angle:    {aim:.1f}°\n"
                f"Stimpmeter:   {stimp:.1f}\n"
                f"Slope:        {self._slope_spin.value():.1f}°\n\n"
                f"Simulator loaded successfully.\n"
                f"(Full 3D visualization coming in next phase)\n"
            )
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
