"""Multi-Model Shot Tracer Visualization GUI.

A standalone PyQt6 application for visualizing golf ball trajectories
using multiple physics models for comparison.

Supports:
- Waterloo/Penner model (quadratic Cd/Cl)
- MacDonald-Hanzely model (1991 physics paper)
- Nathan model (libgolf-inspired, Reynolds-dependent)

This serves as both a validation tool for comparing models and a
visualization utility for shot analysis.
"""

import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Add shared directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "shared" / "python"))

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

try:
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl

    PYQTGRAPH_AVAILABLE = True
except ImportError:
    PYQTGRAPH_AVAILABLE = False
    pg = None  # type: ignore[assignment]
    gl = None  # type: ignore[assignment]

from flight_models import (
    FlightModelRegistry,
    FlightModelType,
    FlightResult,
    UnifiedLaunchConditions,
    compare_models,
)

logger = logging.getLogger(__name__)

# Color palette for multiple trajectories
TRAJECTORY_COLORS = [
    (1.0, 0.5, 0.0, 1.0),  # Orange - Waterloo/Penner
    (0.0, 0.7, 1.0, 1.0),  # Cyan - MacDonald-Hanzely
    (0.0, 1.0, 0.5, 1.0),  # Green - Nathan
    (1.0, 0.0, 0.5, 1.0),  # Magenta - future models
    (1.0, 1.0, 0.0, 1.0),  # Yellow - future models
]


class MultiModelShotTracerWidget(QWidget):
    """Main widget for multi-model shot tracer visualization."""

    def __init__(self, parent: "QWidget | None" = None) -> None:
        """Initialize the shot tracer widget."""
        super().__init__(parent)
        self.results: dict[str, FlightResult] = {}
        self.trajectory_plots: dict[str, Any] = {}
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

        # Right panel: Visualization + Results
        right_panel = self._create_visualization_panel()
        splitter.addWidget(right_panel)

        # Set initial sizes (35% controls, 65% visualization)
        splitter.setSizes([350, 650])

        main_layout.addWidget(splitter)

    def _create_controls_panel(self) -> QWidget:
        """Create the left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(10)

        # Title
        title = QLabel("Multi-Model Shot Tracer")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Model Selection Group
        model_group = QGroupBox("Physics Models")
        model_layout = QVBoxLayout()

        self.model_checkboxes: dict[str, QCheckBox] = {}
        for model_type in FlightModelType:
            model = FlightModelRegistry.get_model(model_type)
            checkbox = QCheckBox(f"{model.name}")
            checkbox.setToolTip(f"{model.description}\nRef: {model.reference}")
            checkbox.setChecked(True)  # All models enabled by default
            self.model_checkboxes[model_type.value] = checkbox
            model_layout.addWidget(checkbox)

        model_group.setLayout(model_layout)
        layout.addWidget(model_group)

        # Launch Parameters Group
        launch_group = QGroupBox("Launch Parameters")
        form = QFormLayout()

        # Ball Speed
        self.speed_spin = QDoubleSpinBox()
        self.speed_spin.setRange(50.0, 200.0)
        self.speed_spin.setValue(163.0)
        self.speed_spin.setSuffix(" mph")
        self.speed_spin.setDecimals(1)
        form.addRow("Ball Speed:", self.speed_spin)

        # Launch Angle
        self.angle_spin = QDoubleSpinBox()
        self.angle_spin.setRange(-10.0, 45.0)
        self.angle_spin.setValue(11.0)
        self.angle_spin.setSuffix("°")
        self.angle_spin.setDecimals(1)
        form.addRow("Launch Angle:", self.angle_spin)

        # Spin Rate
        self.spin_spin = QDoubleSpinBox()
        self.spin_spin.setRange(0.0, 12000.0)
        self.spin_spin.setValue(2500.0)
        self.spin_spin.setSuffix(" rpm")
        self.spin_spin.setDecimals(0)
        form.addRow("Backspin:", self.spin_spin)

        # Azimuth Angle
        self.azimuth_spin = QDoubleSpinBox()
        self.azimuth_spin.setRange(-45.0, 45.0)
        self.azimuth_spin.setValue(0.0)
        self.azimuth_spin.setSuffix("°")
        self.azimuth_spin.setDecimals(1)
        form.addRow("Direction:", self.azimuth_spin)

        # Spin Axis Angle (for sidespin)
        self.spin_axis_spin = QDoubleSpinBox()
        self.spin_axis_spin.setRange(-45.0, 45.0)
        self.spin_axis_spin.setValue(0.0)
        self.spin_axis_spin.setSuffix("°")
        self.spin_axis_spin.setDecimals(1)
        self.spin_axis_spin.setToolTip(
            "Spin axis tilt: 0° = pure backspin, ±45° = fade/draw"
        )
        form.addRow("Spin Axis Tilt:", self.spin_axis_spin)

        launch_group.setLayout(form)
        layout.addWidget(launch_group)

        # Presets Group
        presets_group = QGroupBox("Club Presets")
        presets_layout = QHBoxLayout()

        driver_btn = QPushButton("Driver")
        driver_btn.clicked.connect(lambda: self._apply_preset("driver"))
        presets_layout.addWidget(driver_btn)

        iron7_btn = QPushButton("7-Iron")
        iron7_btn.clicked.connect(lambda: self._apply_preset("7iron"))
        presets_layout.addWidget(iron7_btn)

        pw_btn = QPushButton("PW")
        pw_btn.clicked.connect(lambda: self._apply_preset("pw"))
        presets_layout.addWidget(pw_btn)

        presets_group.setLayout(presets_layout)
        layout.addWidget(presets_group)

        # Action Buttons
        button_layout = QHBoxLayout()

        self.simulate_btn = QPushButton("Compare Models")
        self.simulate_btn.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 10px;"
        )
        button_layout.addWidget(self.simulate_btn)

        self.clear_btn = QPushButton("Clear")
        button_layout.addWidget(self.clear_btn)

        layout.addLayout(button_layout)

        # Legend
        legend_group = QGroupBox("Legend")
        legend_layout = QVBoxLayout()

        self.legend_labels: list[QLabel] = []
        for i, model_type in enumerate(FlightModelType):
            model = FlightModelRegistry.get_model(model_type)
            color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]
            rgb = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
            label = QLabel(f"● {model.name}")
            label.setStyleSheet(f"color: {rgb}; font-weight: bold;")
            legend_layout.addWidget(label)
            self.legend_labels.append(label)

        legend_group.setLayout(legend_layout)
        layout.addWidget(legend_group)

        layout.addStretch()

        return panel

    def _create_visualization_panel(self) -> QWidget:
        """Create the visualization and results panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # 3D Visualization (top)
        if PYQTGRAPH_AVAILABLE:
            self.gl_widget = gl.GLViewWidget()
            self.gl_widget.setCameraPosition(distance=400, elevation=25, azimuth=45)
            self.gl_widget.setMinimumHeight(400)

            # Add grid
            grid = gl.GLGridItem()
            grid.setSize(300, 100, 1)
            grid.setSpacing(25, 25, 25)
            grid.translate(150, 0, 0)
            self.gl_widget.addItem(grid)

            layout.addWidget(self.gl_widget, stretch=2)
        else:
            label = QLabel(
                "3D Visualization requires pyqtgraph with OpenGL support.\n"
                "Install with: pip install pyqtgraph PyOpenGL"
            )
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(label, stretch=2)

        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        layout.addWidget(separator)

        # Results Table (bottom)
        results_group = QGroupBox("Model Comparison Results")
        results_layout = QVBoxLayout()

        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels(
            ["Model", "Carry (yd)", "Max Height (m)", "Time (s)", "Landing (°)"]
        )
        header = self.results_table.horizontalHeader()
        if header is not None:
            header.setStretchLastSection(True)
        self.results_table.setMinimumHeight(120)
        results_layout.addWidget(self.results_table)

        results_group.setLayout(results_layout)
        layout.addWidget(results_group, stretch=1)

        return panel

    def _connect_signals(self) -> None:
        """Connect widget signals to handlers."""
        self.simulate_btn.clicked.connect(self._run_comparison)
        self.clear_btn.clicked.connect(self._clear_visualization)

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
            self.spin_axis_spin.setValue(0.0)

    def _get_selected_models(self) -> list[FlightModelType]:
        """Get list of selected model types."""
        selected = []
        for model_type in FlightModelType:
            checkbox = self.model_checkboxes.get(model_type.value)
            if checkbox and checkbox.isChecked():
                selected.append(model_type)
        return selected

    def _run_comparison(self) -> None:
        """Run all selected models and compare results."""
        # Create launch conditions
        launch = UnifiedLaunchConditions.from_imperial(
            ball_speed_mph=self.speed_spin.value(),
            launch_angle_deg=self.angle_spin.value(),
            spin_rate_rpm=self.spin_spin.value(),
            azimuth_angle_deg=self.azimuth_spin.value(),
            spin_axis_angle_deg=self.spin_axis_spin.value(),
        )

        # Get selected models
        selected_types = self._get_selected_models()
        if not selected_types:
            QMessageBox.warning(self, "No Models", "Please select at least one model.")
            return

        models = [FlightModelRegistry.get_model(t) for t in selected_types]

        try:
            self.results = compare_models(launch, models)
            self._update_visualization()
            self._update_results_table()
        except Exception as e:
            logger.exception("Comparison failed")
            QMessageBox.warning(self, "Simulation Error", str(e))

    def _update_visualization(self) -> None:
        """Update the 3D visualization with all trajectories."""
        if not PYQTGRAPH_AVAILABLE:
            return

        # Clear old trajectories
        for plot_item in self.trajectory_plots.values():
            if plot_item is not None:
                self.gl_widget.removeItem(plot_item)
        self.trajectory_plots.clear()

        # Plot each model's trajectory
        for i, (model_name, result) in enumerate(self.results.items()):
            positions = result.to_position_array()
            color = TRAJECTORY_COLORS[i % len(TRAJECTORY_COLORS)]

            line = gl.GLLinePlotItem(
                pos=positions, color=color, width=3, antialias=True
            )
            self.gl_widget.addItem(line)
            self.trajectory_plots[model_name] = line

        # Adjust camera to fit all trajectories
        if self.results:
            all_positions = np.vstack(
                [r.to_position_array() for r in self.results.values()]
            )
            max_x = np.max(all_positions[:, 0])
            self.gl_widget.setCameraPosition(
                distance=max(200, max_x * 1.2), elevation=20, azimuth=45
            )

    def _update_results_table(self) -> None:
        """Update the results comparison table."""
        self.results_table.setRowCount(len(self.results))

        for row, (model_name, result) in enumerate(self.results.items()):
            carry_yd = result.carry_distance * 1.09361

            self.results_table.setItem(row, 0, QTableWidgetItem(model_name))
            self.results_table.setItem(row, 1, QTableWidgetItem(f"{carry_yd:.1f}"))
            self.results_table.setItem(
                row, 2, QTableWidgetItem(f"{result.max_height:.1f}")
            )
            self.results_table.setItem(
                row, 3, QTableWidgetItem(f"{result.flight_time:.2f}")
            )
            self.results_table.setItem(
                row, 4, QTableWidgetItem(f"{result.landing_angle:.1f}")
            )

        self.results_table.resizeColumnsToContents()

    def _clear_visualization(self) -> None:
        """Clear all trajectories and results."""
        if PYQTGRAPH_AVAILABLE:
            for plot_item in self.trajectory_plots.values():
                if plot_item is not None:
                    self.gl_widget.removeItem(plot_item)
            self.trajectory_plots.clear()

        self.results.clear()
        self.results_table.setRowCount(0)


class MultiModelShotTracerWindow(QMainWindow):
    """Main window for the Multi-Model Shot Tracer application."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Golf Shot Tracer - Multi-Model Comparison")
        self.setMinimumSize(1300, 900)

        # Central widget
        self.central_widget = MultiModelShotTracerWidget()
        self.setCentralWidget(self.central_widget)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Models: Waterloo/Penner, MacDonald-Hanzely, Nathan | "
            "Select models, enter parameters, and click 'Compare Models'"
        )


def main() -> None:
    """Launch the Multi-Model Shot Tracer application."""
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
        QDoubleSpinBox, QSpinBox, QComboBox {
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
        QCheckBox {
            color: #ffffff;
        }
        QTableWidget {
            background-color: #3c3c3c;
            color: #ffffff;
            gridline-color: #555;
        }
        QHeaderView::section {
            background-color: #2b2b2b;
            color: #ffffff;
            padding: 5px;
            border: 1px solid #555;
        }
        """
    )

    window = MultiModelShotTracerWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
