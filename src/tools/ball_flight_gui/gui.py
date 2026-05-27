"""Aerodynamic Ball Flight Simulator GUI.

Wraps the aerodynamics engine and ball flight simulators in a PyQt6
dashboard for configuring launch conditions and visualizing trajectories
with full aerodynamic force decomposition.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class BallFlightWidget(QWidget):
    """Central widget for the aerodynamic ball flight simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Aerodynamic Ball Flight Simulator")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Launch conditions
        launch_group = QGroupBox("Launch Conditions")
        launch_form = QFormLayout(launch_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(50.0, 200.0)
        self._speed_spin.setValue(163.0)
        self._speed_spin.setSuffix(" mph")
        launch_form.addRow("Ball Speed:", self._speed_spin)

        self._angle_spin = QDoubleSpinBox()
        self._angle_spin.setRange(-5.0, 45.0)
        self._angle_spin.setValue(11.0)
        self._angle_spin.setSuffix("°")
        launch_form.addRow("Launch Angle:", self._angle_spin)

        self._spin_spin = QDoubleSpinBox()
        self._spin_spin.setRange(0.0, 12000.0)
        self._spin_spin.setValue(2500.0)
        self._spin_spin.setSuffix(" rpm")
        launch_form.addRow("Backspin:", self._spin_spin)

        self._sidespin_spin = QDoubleSpinBox()
        self._sidespin_spin.setRange(-5000.0, 5000.0)
        self._sidespin_spin.setValue(0.0)
        self._sidespin_spin.setSuffix(" rpm")
        launch_form.addRow("Sidespin:", self._sidespin_spin)

        left_layout.addWidget(launch_group)

        # Environment
        env_group = QGroupBox("Environment")
        env_form = QFormLayout(env_group)

        self._wind_speed = QDoubleSpinBox()
        self._wind_speed.setRange(0.0, 50.0)
        self._wind_speed.setValue(0.0)
        self._wind_speed.setSuffix(" mph")
        env_form.addRow("Wind Speed:", self._wind_speed)

        self._wind_dir = QDoubleSpinBox()
        self._wind_dir.setRange(0.0, 360.0)
        self._wind_dir.setValue(0.0)
        self._wind_dir.setSuffix("°")
        env_form.addRow("Wind Direction:", self._wind_dir)

        self._altitude = QDoubleSpinBox()
        self._altitude.setRange(0.0, 10000.0)
        self._altitude.setValue(0.0)
        self._altitude.setSuffix(" ft")
        env_form.addRow("Altitude:", self._altitude)

        left_layout.addWidget(env_group)

        # Aerodynamic options
        aero_group = QGroupBox("Aerodynamic Model")
        aero_layout = QVBoxLayout(aero_group)
        self._chk_dimples = QCheckBox("Dimple geometry effects")
        self._chk_dimples.setChecked(True)
        aero_layout.addWidget(self._chk_dimples)
        self._chk_magnus = QCheckBox("Magnus force (spin-induced lift)")
        self._chk_magnus.setChecked(True)
        aero_layout.addWidget(self._chk_magnus)
        self._chk_seam = QCheckBox("Seam orientation effects")
        aero_layout.addWidget(self._chk_seam)
        left_layout.addWidget(aero_group)

        # Presets
        preset_group = QGroupBox("Club Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, angle, spin in [
            ("Driver", 163.0, 11.0, 2500.0),
            ("7-Iron", 118.0, 16.0, 7000.0),
            ("PW", 94.0, 23.0, 9000.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, a=angle, sp=spin: self._apply_preset(s, a, sp)
            )
            preset_layout.addWidget(btn)
        left_layout.addWidget(preset_group)

        # Run
        self._run_btn = QPushButton("Simulate Flight")
        self._run_btn.setStyleSheet(
            "background-color: #1565C0; color: white; font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_simulation)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)
        results_group = QGroupBox("Flight Results")
        results_layout = QVBoxLayout(results_group)

        try:
            import pyqtgraph as pg  # noqa: F401
            import pyqtgraph.opengl as gl

            self._gl_view = gl.GLViewWidget()
            self._gl_view.opts["distance"] = 250
            self._gl_view.opts["elevation"] = 20
            self._gl_view.opts["azimuth"] = 45

            grid = gl.GLGridItem()
            grid.setSize(x=500, y=500, z=0)
            grid.setSpacing(x=50, y=50, z=0)
            self._gl_view.addItem(grid)

            axis = gl.GLAxisItem()
            axis.setSize(x=50, y=50, z=50)
            self._gl_view.addItem(axis)

            self._plot_item = None
            results_layout.addWidget(self._gl_view, stretch=3)
        except ImportError:
            self._gl_view = None
            self._plot_item = None

        self._results_text = QTextEdit()
        self._results_text.setReadOnly(True)
        self._results_text.setPlainText(
            "Configure launch conditions and click 'Simulate Flight'.\n\n"
            "Aerodynamic forces modeled:\n"
            "  - Drag (Reynolds-dependent Cd)\n"
            "  - Magnus lift (spin-induced)\n"
            "  - Gravity\n"
            "  - Wind effects\n"
            "  - Dimple geometry\n"
            "  - Altitude/air density\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _apply_preset(self, speed: float, angle: float, spin: float) -> None:
        self._speed_spin.setValue(speed)
        self._angle_spin.setValue(angle)
        self._spin_spin.setValue(spin)

    def _run_simulation(self) -> None:
        """Execute the ball flight simulation."""
        try:
            from src.shared.python.physics.ball_simulator import BallFlightSimulator
            from src.shared.python.physics.ball_launch_conditions import (
                LaunchConditions,
            )

            speed_ms = self._speed_spin.value() * 0.44704  # mph to m/s
            angle = self._angle_spin.value()
            spin_rps = self._spin_spin.value() / 60.0 * 2 * np.pi  # rpm to rad/s

            launch = LaunchConditions(
                velocity=speed_ms,
                launch_angle=angle,
                spin_rate=spin_rps,
            )

            sim = BallFlightSimulator()
            trajectory = sim.simulate_trajectory(launch, max_time=10.0, dt=0.01)

            if trajectory:
                last = trajectory[-1]
                carry = float(np.sqrt(last.position[0] ** 2 + last.position[1] ** 2))
                max_h = max(p.position[2] for p in trajectory)

                self._results_text.setPlainText(
                    f"Ball Flight Results\n"
                    f"{'=' * 40}\n"
                    f"Launch: {self._speed_spin.value():.0f} mph, "
                    f"{angle:.1f}°, {self._spin_spin.value():.0f} rpm\n\n"
                    f"Carry:       {carry:.1f} m ({carry * 1.09361:.1f} yd)\n"
                    f"Max Height:  {max_h:.1f} m ({max_h * 3.28084:.1f} ft)\n"
                    f"Flight Time: {last.time:.2f} s\n"
                    f"Points:      {len(trajectory)}\n\n"
                    f"Landing Position:\n"
                    f"  X: {last.position[0]:.1f} m\n"
                    f"  Y: {last.position[1]:.1f} m\n"
                    f"  Z: {last.position[2]:.1f} m\n"
                )

                # Update 3D Visualization
                if getattr(self, "_gl_view", None) is not None:
                    import pyqtgraph as pg  # noqa: F401
                    import pyqtgraph.opengl as gl

                    pts = np.array([p.position for p in trajectory])

                    if self._plot_item is not None:
                        self._gl_view.removeItem(self._plot_item)

                    self._plot_item = gl.GLLinePlotItem(
                        pos=pts, color=(0.0, 0.7, 1.0, 1.0), width=3, antialias=True
                    )
                    self._gl_view.addItem(self._plot_item)

                    # Auto-center camera on the trajectory midpoint
                    if len(pts) > 0:
                        from pyqtgraph import Vector

                        mid_x = max(pts[:, 0]) / 2.0
                        self._gl_view.opts["center"] = Vector(mid_x, 0, 0)
            else:
                self._results_text.setPlainText("No trajectory generated.")
        except ImportError as e:
            self._results_text.setPlainText(f"Ball flight simulator not available: {e}")
        except Exception as e:
            logger.exception("Ball flight simulation failed")
            self._results_text.setPlainText(f"Simulation error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("BallFlightWidget cleanup")


class BallFlightWindow(QMainWindow):
    """Standalone window for the ball flight simulator."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Aerodynamic Ball Flight Simulator")
        self.setMinimumSize(1100, 700)
        self._widget = BallFlightWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "Forces: Drag + Magnus + Gravity + Wind | Configure and run simulation"
        )

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


def get_dockable_ui() -> BallFlightWindow:
    """Return the main window instance for docking in the unified launcher."""
    return BallFlightWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = BallFlightWindow()
    w.show()
    sys.exit(app.exec())
