"""Swing-to-Flight Pipeline GUI — end-to-end simulation visualizer.

Wraps :class:`SwingBallFlightPipeline` in a PyQt6 dashboard that lets
the user configure swing parameters, run the full pipeline
(swing → impact → ball flight), and visualize the trajectory.

Implements the GUI tile for the swing-to-flight pipeline registered
in ``models.yaml`` under the ``simulation`` category.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
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


class SwingFlightWidget(QWidget):
    """Central widget for the swing-to-flight pipeline dashboard."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: Any = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: controls
        left = QWidget()
        left_layout = QVBoxLayout(left)

        title = QLabel("Swing → Impact → Ball Flight Pipeline")
        title_font = title.font()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title.setFont(title_font)
        left_layout.addWidget(title)

        # Swing parameters
        swing_group = QGroupBox("Swing Parameters")
        swing_form = QFormLayout(swing_group)

        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(20.0, 60.0)
        self._speed_spin.setValue(45.0)
        self._speed_spin.setSuffix(" m/s")
        swing_form.addRow("Clubhead Speed:", self._speed_spin)

        self._loft_spin = QDoubleSpinBox()
        self._loft_spin.setRange(5.0, 60.0)
        self._loft_spin.setValue(10.5)
        self._loft_spin.setSuffix("°")
        swing_form.addRow("Loft Angle:", self._loft_spin)

        self._mass_spin = QDoubleSpinBox()
        self._mass_spin.setRange(0.100, 0.400)
        self._mass_spin.setValue(0.200)
        self._mass_spin.setDecimals(3)
        self._mass_spin.setSuffix(" kg")
        swing_form.addRow("Clubhead Mass:", self._mass_spin)

        left_layout.addWidget(swing_group)

        # Engine selector
        engine_group = QGroupBox("Physics Engine Source")
        engine_layout = QVBoxLayout(engine_group)
        self._engine_combo = QComboBox()
        self._engine_combo.addItems(["mujoco", "drake", "pinocchio", "manual"])
        self._engine_combo.setCurrentText("manual")
        engine_layout.addWidget(self._engine_combo)
        left_layout.addWidget(engine_group)

        # Presets
        preset_group = QGroupBox("Club Presets")
        preset_layout = QHBoxLayout(preset_group)
        for name, speed, loft in [
            ("Driver", 50.0, 10.5),
            ("7-Iron", 35.0, 34.0),
            ("PW", 28.0, 46.0),
        ]:
            btn = QPushButton(name)
            btn.clicked.connect(
                lambda checked, s=speed, loft_val=loft: self._apply_preset(s, loft_val)
            )
            preset_layout.addWidget(btn)
        left_layout.addWidget(preset_group)

        # Run button
        self._run_btn = QPushButton("Run Full Pipeline")
        self._run_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; "
            "font-weight: bold; padding: 12px;"
        )
        self._run_btn.clicked.connect(self._run_pipeline)
        left_layout.addWidget(self._run_btn)

        left_layout.addStretch()
        splitter.addWidget(left)

        # Right: results
        right = QWidget()
        right_layout = QVBoxLayout(right)

        results_group = QGroupBox("Pipeline Results")
        results_layout = QVBoxLayout(results_group)

        try:
            import pyqtgraph as pg
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
            "Configure swing parameters and click 'Run Full Pipeline' "
            "to execute the end-to-end simulation.\n\n"
            "Pipeline stages:\n"
            "  1. Swing State → Pre-Impact kinematics\n"
            "  2. Impact Model → Post-impact ball velocity + spin\n"
            "  3. Launch Conditions → Aerodynamic trajectory\n"
            "  4. Metrics extraction (carry, height, flight time)\n"
        )
        results_layout.addWidget(self._results_text)
        right_layout.addWidget(results_group)
        splitter.addWidget(right)

        splitter.setSizes([350, 650])
        layout.addWidget(splitter)

    def _apply_preset(self, speed: float, loft: float) -> None:
        self._speed_spin.setValue(speed)
        self._loft_spin.setValue(loft)

    def _run_pipeline(self) -> None:
        """Execute the swing-to-flight pipeline."""
        try:
            from src.shared.python.physics.swing_ball_flight_pipeline import (
                SwingBallFlightPipeline,
                SwingState,
            )

            speed = self._speed_spin.value()
            loft = self._loft_spin.value()
            mass = self._mass_spin.value()
            engine = self._engine_combo.currentText()

            # Build swing state from UI parameters
            velocity = np.array([speed, 0.0, 0.0])
            swing = SwingState(
                clubhead_velocity=velocity,
                clubhead_angular_velocity=np.zeros(3),
                clubhead_orientation=np.array([0.0, 0.0, 1.0]),
                clubhead_mass=mass,
                clubhead_loft_deg=loft,
                engine_name=engine,
            )

            pipeline = SwingBallFlightPipeline()
            result = pipeline.run(swing)
            self._result = result

            self._results_text.setPlainText(
                f"Pipeline Complete\n"
                f"{'=' * 40}\n"
                f"Engine: {result.swing_state.engine_name}\n\n"
                f"Impact Results:\n"
                f"  Ball speed: {np.linalg.norm(result.impact_state.ball_velocity):.1f} m/s\n"
                f"  Ball spin:  {np.linalg.norm(result.impact_state.ball_angular_velocity):.0f} rad/s\n\n"
                f"Launch Conditions:\n"
                f"  Speed:      {result.launch_conditions.velocity:.1f} m/s\n"
                f"  Angle:      {result.launch_conditions.launch_angle:.1f}°\n"
                f"  Spin Rate:  {result.launch_conditions.spin_rate:.0f} rad/s\n\n"
                f"Flight Results:\n"
                f"  Carry:      {result.carry_m:.1f} m ({result.carry_m * 1.09361:.1f} yd)\n"
                f"  Max Height: {result.max_height_m:.1f} m\n"
                f"  Flight Time:{result.flight_time_s:.2f} s\n"
                f"  Landing:    {result.landing_angle_deg:.1f}°\n\n"
                f"Trajectory: {len(result.trajectory)} points\n"
            )

            # Update 3D Visualization
            if getattr(self, "_gl_view", None) is not None and result.trajectory:
                import pyqtgraph as pg
                import pyqtgraph.opengl as gl

                pts = np.array([p.position for p in result.trajectory])

                if self._plot_item is not None:
                    self._gl_view.removeItem(self._plot_item)

                self._plot_item = gl.GLLinePlotItem(
                    pos=pts, color=(1.0, 0.5, 0.0, 1.0), width=3, antialias=True
                )
                self._gl_view.addItem(self._plot_item)

                # Auto-center camera
                if len(pts) > 0:
                    from pyqtgraph import Vector

                    mid_x = max(pts[:, 0]) / 2.0
                    self._gl_view.opts["center"] = Vector(mid_x, 0, 0)

        except ImportError as e:
            self._results_text.setPlainText(
                f"Pipeline not available: {e}\n\n"
                "The SwingBallFlightPipeline module may not be merged yet.\n"
                "Check branch: feat/5337-swing-ball-flight-pipeline"
            )
        except Exception as e:
            logger.exception("Pipeline execution failed")
            self._results_text.setPlainText(f"Pipeline error: {e}")

    def cleanup(self) -> None:
        """Release resources."""
        logger.debug("SwingFlightWidget cleanup")


class SwingFlightWindow(QMainWindow):
    """Standalone window for the swing-to-flight pipeline."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Swing → Flight Pipeline")
        self.setMinimumSize(1100, 700)
        self._widget = SwingFlightWidget(self)
        self.setCentralWidget(self._widget)
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage("Configure swing parameters and run the pipeline")

    def closeEvent(self, event: Any) -> None:
        self._widget.cleanup()
        super().closeEvent(event)


def get_dockable_ui() -> SwingFlightWindow:
    """Return the main window instance for docking in the unified launcher."""
    return SwingFlightWindow()


if __name__ == "__main__":
    import sys

    app = QApplication(sys.argv)
    w = SwingFlightWindow()
    w.show()
    sys.exit(app.exec())
"""Swing-to-Flight Pipeline GUI."""
