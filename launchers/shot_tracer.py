"""Shot Tracer Visualization GUI.

A standalone PyQt6 application for visualizing golf ball trajectories.
Allows manual input of launch parameters (ball speed, launch angle, spin rate,
spin axis) and displays the resulting 3D trajectory.

This serves as both a validation tool for the ball flight model and a
visualization utility for shot analysis.
"""

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
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
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False

from ball_flight_physics import (
    BallFlightSimulator,
    LaunchConditions,
)

logger = logging.getLogger(__name__)


@dataclass
class LaunchParameters:
    """User-configurable launch parameters for the shot tracer."""

    ball_speed_mph: float = 163.0  # mph
    launch_angle_deg: float = 11.0  # degrees
    spin_rate_rpm: float = 2500.0  # rpm
    azimuth_angle_deg: float = 0.0  # degrees (direction)
    side_spin_rpm: float = 0.0  # rpm (positive = fade, negative = draw)

    def to_launch_conditions(self) -> LaunchConditions:
        """Convert to LaunchConditions for the simulator."""
        # Convert units
        ball_speed_ms = self.ball_speed_mph * 0.44704  # mph to m/s
        launch_angle_rad = math.radians(self.launch_angle_deg)
        azimuth_rad = math.radians(self.azimuth_angle_deg)

        # Calculate spin axis from backspin and sidespin
        # Backspin axis is perpendicular to flight direction (pointing left)
        # Sidespin axis is vertical
        total_spin = math.sqrt(self.spin_rate_rpm**2 + self.side_spin_rpm**2)
        if total_spin > 0:
            # Axis is tilted based on ratio of backspin to sidespin
            backspin_component = self.spin_rate_rpm / total_spin
            sidespin_component = self.side_spin_rpm / total_spin
            # Default backspin axis is [0, -1, 0], sidespin tilts it
            spin_axis = np.array([sidespin_component, -backspin_component, 0.0])
            spin_axis = spin_axis / np.linalg.norm(spin_axis)
        else:
            spin_axis = np.array([0.0, -1.0, 0.0])

        return LaunchConditions(
            velocity=ball_speed_ms,
            launch_angle=launch_angle_rad,
            azimuth_angle=azimuth_rad,
            spin_rate=total_spin,
            spin_axis=spin_axis,
        )


class ShotTracerWidget(QWidget):
    """Main widget for the shot tracer visualization."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        """Initialize the shot tracer widget."""
        super().__init__(parent)
        self.params = LaunchParameters()
        self.trajectory: list[Any] = []
        self.animation_timer = QTimer()
        self.animation_index = 0

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self) -> None:
        """Setup the user interface."""
        main_layout = QHBoxLayout(self)

        # Create splitter for resizable panels
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel: Controls
        left_panel = self._create_controls_panel()
        splitter.addWidget(left_panel)

        # Right panel: 3D Visualization
        right_panel = self._create_visualization_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (30% controls, 70% visualization)
        splitter.setSizes([300, 700])

        main_layout.addWidget(splitter)

    def _create_controls_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Title
        title = QLabel("Shot Tracer")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Launch Parameters Group
        launch_group = QGroupBox("Launch Parameters")
        form = QFormLayout()

        # Ball Speed
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(50.0, 200.0)
        self.speed_spin.setValue(self.params.ball_speed_mph)
        self.speed_spin.setSuffix(" mph")
        self.speed_spin.setDecimals(1)
        form.addRow("Ball Speed:", self.speed_spin)

        # Launch Angle
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-10.0, 45.0)
        self.angle_spin.setValue(self.params.launch_angle_deg)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setDecimals(1)
        form.addRow("Launch Angle:", self.angle_spin)

        # Spin Rate
        self.spin_spin = QDoubleSpinBox()
        self.spin_spin.setRange(0.0, 12000.0)
        self.spin_spin.setValue(self.params.spin_rate_rpm)
        self.spin_spin.setSuffix(" rpm")
        self.spin_spin.setDecimals(0)
        form.addRow("Backspin:", self.spin_spin)

        # Azimuth Angle
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(-45.0, 45.0)
        self.azimuth_spin.setValue(self.params.azimuth_angle_deg)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setDecimals(1)
        form.addRow("Direction:", self.azimuth_spin)

        # Side Spin
        self.sidespin_spin = QDoubleSpinBox()
        self.sidespin_spin.setRange(-5000.0, 5000.0)
        self.sidespin_spin.setValue(self.params.side_spin_rpm)
        self.sidespin_spin.setSuffix(" rpm")
        self.sidespin_spin.setDecimals(0)
        form.addRow("Sidespin (+fade/-draw):", self.sidespin_spin)

        launch_group.setLayout(form)
        layout.addWidget(launch_group)

        # Presets Group
        presets_group = QGroupBox("Club Presets")
        presets_layout = QVBoxLayout()

        driver_btn = QPushButton("Driver")
        driver_btn.clicked.connect(lambda: self._apply_preset("driver"))
        presets_layout.addWidget(driver_btn)

        iron7_btn = QPushButton("7-Iron")
        iron7_btn.clicked.connect(lambda: self._apply_preset("7iron"))
        presets_layout.addWidget(iron7_btn)

        pw_btn = QPushButton("Pitching Wedge")
        pw_btn.clicked.connect(lambda: self._apply_preset("pw"))
        presets_layout.addWidget(pw_btn)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Action Buttons
        button_layout = QHBoxLayout()

        self.simulate_btn = QPushButton("Simulate Flight")
        self.simulate_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        button_layout.addWidget(self.simulate_btn)

        self.animate_btn = QPushButton("Animate")
        self.animate_btn.setEnabled(False)
        button_layout.addWidget(self.animate_btn)

        layout.addLayout(button_layout)

        # Results Group
        self.results_group = QGroupBox("Flight Results")
        results_layout = QFormLayout()

        self.carry_label = QLabel("--")
        results_layout.addRow("Carry Distance:", self.carry_label)

        self.max_height_label = QLabel("--")
        results_layout.addRow("Max Height:", self.max_height_label)

        self.flight_time_label = QLabel("--")
        results_layout.addRow("Flight Time:", self.flight_time_label)

        self.landing_angle_label = QLabel("--")
        results_layout.addRow("Landing Angle:", self.landing_angle_label)

        self.results_group.setLayout(results_layout)
        layout.addWidget(self.results_group)

        # Stretch to push everything to top
        layout.addStretch()

        return panel

    def _create_visualization_panel(self) -> QWidget:
        """Create the 3D visualization panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        if PYQTGRAPH_AVAILABLE:
            # Create OpenGL widget for 3D visualization
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setCameraPosition(distance=400, elevation=25, azimuth=45)

            # Add grid
            grid = gl.GLGridItem()
            grid.setSize(300, 50, 1)
            grid.setSpacing(25, 25, 25)
            grid.translate(150, 0, 0)
            self.gl_widget.addItem(grid)

            # Create placeholder for trajectory
            self.trajectory_line: gl.GLLinePlotItem | None = None
            self.ball_scatter: gl.GLScatterPlotItem | None = None

            layout.addWidget(self.gl_widget)
        else:
            # Fallback 2D visualization using pyqtgraph PlotWidget
            try:
                self.plot_widget = pg.PlotWidget()
                self.plot_widget.setLabel("left", "Height", units="m")
                self.plot_widget.setLabel("bottom", "Distance", units="m")
                self.plot_widget.setTitle("Ball Flight Trajectory (Side View)")
                self.plot_widget.showGrid(x=True, y=True)
                self.trajectory_curve: pg.PlotDataItem | None = None
                layout.addWidget(self.plot_widget)
                self.use_2d_fallback = True
            except Exception:
                # Final fallback - just a label
                label = QLabel(
                    "3D Visualization requires pyqtgraph with OpenGL support.\n"
                    "Install with: pip install pyqtgraph PyOpenGL"
                )
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
                self.use_2d_fallback = False

        return panel

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        self.simulate_btn.clicked.connect(self._run_simulation)
        self.animate_btn.clicked.connect(self._start_animation)
        self.animation_timer.timeout.connect(self._update_animation)

    def _apply_preset(self, club: str) -> None:
        """Apply preset values for a club type."""
        presets = {
            "driver": (163.0, 11.0, 2500.0),
            "7iron": (118.0, 16.0, 7000.0),
            "pw": (94.0, 23.0, 9000.0),
        }

        if club in presets:
            speed, angle, spin = presets[club]
            self.speed_spin.setValue(speed)
            self.angle_spin.setValue(angle)
            self.spin_spin.setValue(spin)
            self.azimuth_spin.setValue(0.0)
            self.sidespin_spin.setValue(0.0)

    def _run_simulation(self) -> None:
        """Run the ball flight simulation with current parameters."""
        # Update parameters from widgets
        self.params.ball_speed_mph = self.speed_spin.value()
        self.params.launch_angle_deg = self.angle_spin.value()
        self.params.spin_rate_rpm = self.spin_spin.value()
        self.params.azimuth_angle_deg = self.azimuth_spin.value()
        self.params.side_spin_rpm = self.sidespin_spin.value()

        # Create simulator and run
        simulator = BallFlightSimulator()
        launch = self.params.to_launch_conditions()

        try:
            self.trajectory = simulator.simulate_trajectory(launch, max_time=10.0)
            analysis = simulator.analyze_trajectory(self.trajectory)

            # Update results
            carry_yards = analysis["carry_distance"] * 1.09361
            self.carry_label.setText(f"{carry_yards:.1f} yards")
            self.max_height_label.setText(f"{analysis['max_height']:.1f} m")
            self.flight_time_label.setText(f"{analysis['flight_time']:.2f} s")
            self.landing_angle_label.setText(f"{analysis['landing_angle']:.1f}°")

            # Update visualization
            self._update_visualization()
            self.animate_btn.setEnabled(True)

        except Exception as e:
            logger.exception("Simulation failed")
            QMessageBox.warning(self, "Simulation Error", str(e))

    def _update_visualization(self) -> None:
        """Update the 3D visualization with the trajectory."""
        if not self.trajectory:
            return

        # Extract positions
        positions = np.array([p.position for p in self.trajectory])

        if PYQTGRAPH_AVAILABLE:
            # Remove old items
            if self.trajectory_line is not None:
                self.gl_widget.removeItem(self.trajectory_line)
            if self.ball_scatter is not None:
                self.gl_widget.removeItem(self.ball_scatter)

            # Create new trajectory line
            self.trajectory_line = gl.GLLinePlotItem(
                pos=positions, color=(1, 0.5, 0, 1), width=3, antialias=True
            )
            self.gl_widget.addItem(self.trajectory_line)

            # Add ball at landing position
            landing_pos = positions[-1:]
            self.ball_scatter = gl.GLScatterPlotItem(
                pos=landing_pos, color=(1, 1, 1, 1), size=10
            )
            self.gl_widget.addItem(self.ball_scatter)

            # Center camera on trajectory
            self.gl_widget.setCameraPosition(
                distance=max(200, positions[:, 0].max() * 1.2),
                elevation=20,
                azimuth=45,
            )
        elif hasattr(self, "use_2d_fallback") and self.use_2d_fallback:
            # 2D fallback - plot side view
            self.plot_widget.clear()
            self.trajectory_curve = self.plot_widget.plot(
                positions[:, 0],
                positions[:, 2],
                pen=pg.mkPen(color="orange", width=2),
            )
            # Add landing point
            self.plot_widget.plot(
                [positions[-1, 0]],
                [positions[-1, 2]],
                pen=None,
                symbol="o",
                symbolBrush="white",
            )

    def _start_animation(self) -> None:
        """Start animating the ball flight."""
        if not self.trajectory:
            return

        self.animation_index = 0
        self.animation_timer.start(20)  # 50 fps

    def _update_animation(self) -> None:
        """Update animation frame."""
        if self.animation_index >= len(self.trajectory):
            self.animation_timer.stop()
            return

        if PYQTGRAPH_AVAILABLE and self.ball_scatter is not None:
            pos = self.trajectory[self.animation_index].position
            self.ball_scatter.setData(pos=np.array([pos]))

        self.animation_index += 2  # Skip frames for speed


class ShotTracerWindow(QMainWindow):
    """Main window for the Shot Tracer application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Golf Shot Tracer - Ball Flight Visualization")
        self.setMinimumSize(1200, 800)

        # Central widget
        self.central_widget = ShotTracerWidget()
        self.setCentralWidget(self.central_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Waterloo/Penner Ball Flight Model | Enter parameters and click 'Simulate Flight'"
        )


def main() -> None:
    """Launch the Shot Tracer application."""
    logging.basicConfig(level=logging.INFO)

    app = QApplication(sys.argv)
    app.setStyle("Fusion")

    # Apply dark theme
    app.setStyleSheet(
        """
        QMainWindow, QWidget {
            background-color: #2b2b2b;
            color: #ffffff;
        }
        QGroupBox {
            border: 1px solid #555;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
            font-weight: bold;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QDoubleSpinBox, QSpinBox {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 3px;
            padding: 5px;
        }
        QPushButton {
            background-color: #3c3c3c;
            border: 1px solid #555;
            border-radius: 5px;
            padding: 8px;
        }
        QPushButton:hover {
            background-color: #4c4c4c;
        }
        QPushButton:pressed {
            background-color: #2c2c2c;
        }
        QLabel {
            color: #ffffff;
        }
        """
    )

    window = ShotTracerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
