"""Entry point and main window for the MuJoCo golf pendulum demo."""

import sys
from pathlib import Path

# Add suite root to sys.path to allow imports from shared provided that this file
# is 5 levels deep from the root:
# .../engines/physics_engines/mujoco/python/mujoco_humanoid_golf/__main__.py
try:
    suite_root = Path(__file__).resolve().parents[5]
    if str(suite_root) not in sys.path:
        sys.path.insert(0, str(suite_root))
except IndexError:
    pass

from PyQt6 import QtCore, QtWidgets

from .advanced_gui import AdvancedGolfAnalysisWindow

# Legacy simple window for backwards compatibility
from .models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    FULL_BODY_GOLF_SWING_XML,
    TRIPLE_PENDULUM_XML,
    UPPER_BODY_GOLF_SWING_XML,
)
from .sim_widget import MuJoCoSimWidget


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
    ) -> None:
        """Docstring for __init__."""
        super().__init__()

        self.setWindowTitle("MuJoCo Golf Swing Models")
        self.resize(1200, 700)

        central = QtWidgets.QWidget(self)
        self.setCentralWidget(central)

        h_layout = QtWidgets.QHBoxLayout(central)

        # Left: MuJoCo viewer widget (larger for better visualization)
        self.sim_widget = MuJoCoSimWidget(width=800, height=600, fps=60)
        h_layout.addWidget(self.sim_widget, stretch=2)

        # Right: control panel with scroll area
        control_panel = QtWidgets.QFrame(self)
        control_panel.setFrameShape(QtWidgets.QFrame.Shape.StyledPanel)
        control_layout = QtWidgets.QVBoxLayout(control_panel)
        control_layout.setContentsMargins(8, 8, 8, 8)

        # Model selector
        model_group = QtWidgets.QGroupBox("Golf Swing Model")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.addItem("Control Demo: Chaotic driven pendulum (2 DOF)")
        self.model_combo.addItem("Simple: Double pendulum (2 DOF)")
        self.model_combo.addItem("Simple: Triple pendulum (3 DOF)")
        self.model_combo.addItem("Advanced: Upper body + arms (10 DOF)")
        self.model_combo.addItem("Complete: Full body with legs (15 DOF)")
        self.model_combo.addItem("Research: Advanced biomechanical (28 DOF)")
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)
        control_layout.addWidget(model_group)

        # Play / Pause / Reset
        buttons_group = QtWidgets.QGroupBox("Simulation Control")
        buttons_layout = QtWidgets.QHBoxLayout(buttons_group)

        self.play_pause_btn = QtWidgets.QPushButton("Pause")
        self.play_pause_btn.setCheckable(True)
        self.play_pause_btn.toggled.connect(self.on_play_pause_toggled)

        self.reset_btn = QtWidgets.QPushButton("Reset")
        self.reset_btn.clicked.connect(self.on_reset_clicked)

        buttons_layout.addWidget(self.play_pause_btn)
        buttons_layout.addWidget(self.reset_btn)
        control_layout.addWidget(buttons_group)

        # Scrollable area for actuator controls
        scroll_area = QtWidgets.QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarPolicy.ScrollBarAlwaysOff,
        )

        self.actuator_container = QtWidgets.QWidget()
        self.actuator_layout = QtWidgets.QVBoxLayout(self.actuator_container)
        scroll_area.setWidget(self.actuator_container)

        control_layout.addWidget(scroll_area, stretch=1)

        h_layout.addWidget(control_panel, stretch=1)

        # Store sliders and labels for dynamic updates
        self.actuator_sliders: list[QtWidgets.QSlider] = []
        self.actuator_labels: list[QtWidgets.QLabel] = []
        self.actuator_groups: list[QtWidgets.QGroupBox] = []

        # Model configurations
        self.model_configs = [
            {
                "name": "chaotic_pendulum",
                "xml": CHAOTIC_PENDULUM_XML,
                "actuators": ["Base Drive (Forcing)", "Pendulum Control"],
            },
            {
                "name": "double",
                "xml": DOUBLE_PENDULUM_XML,
                "actuators": ["Shoulder", "Wrist"],
            },
            {
                "name": "triple",
                "xml": TRIPLE_PENDULUM_XML,
                "actuators": ["Shoulder", "Elbow", "Wrist"],
            },
            {
                "name": "upper_body",
                "xml": UPPER_BODY_GOLF_SWING_XML,
                "actuators": [
                    "Spine Rotation",
                    "L Shoulder Swing",
                    "L Shoulder Lift",
                    "L Elbow",
                    "L Wrist",
                    "R Shoulder Swing",
                    "R Shoulder Lift",
                    "R Elbow",
                    "R Wrist",
                    "Club Wrist",
                ],
            },
            {
                "name": "full_body",
                "xml": FULL_BODY_GOLF_SWING_XML,
                "actuators": [
                    "L Ankle",
                    "L Knee",
                    "R Ankle",
                    "R Knee",
                    "Spine Bend",
                    "Spine Rotation",
                    "L Shoulder Swing",
                    "L Shoulder Lift",
                    "L Elbow",
                    "L Wrist",
                    "R Shoulder Swing",
                    "R Shoulder Lift",
                    "R Elbow",
                    "R Wrist",
                    "Club Wrist",
                ],
            },
            {
                "name": "advanced_biomech",
                "xml": ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
                "actuators": [
                    "L Ankle Plantar",
                    "L Ankle Invert",
                    "L Knee",
                    "R Ankle Plantar",
                    "R Ankle Invert",
                    "R Knee",
                    "Spine Lateral",
                    "Spine Sagittal",
                    "Spine Rotation",
                    "L Scap Elev",
                    "L Scap Prot",
                    "L Shldr Flex",
                    "L Shldr Abd",
                    "L Shldr Rot",
                    "L Elbow",
                    "L Wrist Flex",
                    "L Wrist Dev",
                    "R Scap Elev",
                    "R Scap Prot",
                    "R Shldr Flex",
                    "R Shldr Abd",
                    "R Shldr Rot",
                    "R Elbow",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "Shaft Upper",
                    "Shaft Middle",
                    "Shaft Tip",
                ],
            },
        ]

        # Load default model
        self.load_current_model()
        self.sim_widget.reset_state()

    # -------- Model management --------

    def load_current_model(self) -> None:
        """Load selected model and recreate actuator controls."""
        index = self.model_combo.currentIndex()
        config = self.model_configs[index]

        # Clear existing controls
        self._clear_actuator_controls()

        # Load the model XML
        xml_str: str = str(config["xml"])
        self.sim_widget.load_model_from_xml(xml_str)

        # Create new actuator controls
        actuator_list: list[str] = list(config["actuators"])
        self._create_actuator_controls(actuator_list)

    def _clear_actuator_controls(self) -> None:
        """Remove all existing actuator control widgets."""
        self.actuator_sliders.clear()
        self.actuator_labels.clear()

        for group in self.actuator_groups:
            self.actuator_layout.removeWidget(group)
            group.deleteLater()
        self.actuator_groups.clear()

    def _create_actuator_controls(self, actuator_names: list[str]) -> None:
        """Create sliders for all actuators with logical grouping."""
        # Group actuators by body part
        groups = self._group_actuators(actuator_names)

        for group_name, actuators in groups.items():
            group_box = QtWidgets.QGroupBox(group_name)
            group_layout = QtWidgets.QFormLayout(group_box)

            for actuator_name in actuators:
                slider, label = self._create_slider_control(actuator_name)
                self.actuator_sliders.append(slider)
                self.actuator_labels.append(label)

                row = QtWidgets.QHBoxLayout()
                row.addWidget(slider, stretch=3)
                row.addWidget(label, stretch=1)
                group_layout.addRow(f"{actuator_name}:", row)

            self.actuator_groups.append(group_box)
            self.actuator_layout.addWidget(group_box)

        # Add stretch at the end
        self.actuator_layout.addStretch(1)

    def _group_actuators(self, actuator_names: list[str]) -> dict[str, list[str]]:
        """Group actuators by body part for organized display."""
        groups: dict[str, list[str]] = {
            "Control Inputs": [],
            "Legs": [],
            "Torso/Spine": [],
            "Left Scapula": [],
            "Right Scapula": [],
            "Left Arm": [],
            "Right Arm": [],
            "Club/Shaft": [],
            "Simple Joints": [],
        }

        for name in actuator_names:
            name_lower = name.lower()
            if (
                "base drive" in name_lower
                or "forcing" in name_lower
                or "pendulum control" in name_lower
            ):
                groups["Control Inputs"].append(name)
            elif "ankle" in name_lower or "knee" in name_lower:
                groups["Legs"].append(name)
            elif "spine" in name_lower:
                groups["Torso/Spine"].append(name)
            elif name.startswith("L Scap") or (
                "scap" in name_lower and name.startswith("L")
            ):
                groups["Left Scapula"].append(name)
            elif name.startswith("R Scap") or (
                "scap" in name_lower and name.startswith("R")
            ):
                groups["Right Scapula"].append(name)
            elif name.startswith("L "):
                groups["Left Arm"].append(name)
            elif name.startswith("R "):
                groups["Right Arm"].append(name)
            elif "club" in name_lower or "shaft" in name_lower:
                groups["Club/Shaft"].append(name)
            else:
                groups["Simple Joints"].append(name)

        # Remove empty groups
        return {k: v for k, v in groups.items() if v}

    def _create_slider_control(
        self,
        actuator_name: str,  # Reserved for future use
    ) -> tuple[QtWidgets.QSlider, QtWidgets.QLabel]:
        """Create a slider and label for a single actuator."""
        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        slider.setMinimum(-100)
        slider.setMaximum(100)
        slider.setValue(0)
        slider.setSingleStep(1)
        slider.valueChanged.connect(self.on_actuator_changed)

        label = QtWidgets.QLabel("0 Nm")
        label.setMinimumWidth(60)

        return slider, label

    # -------- Callbacks --------

    def on_model_changed(
        self,
        index: int,
    ) -> None:  # Required by Qt signal
        """Handle model selection change."""
        self.load_current_model()
        self.sim_widget.reset_state()

    def on_play_pause_toggled(
        self,
        checked: bool,
    ) -> None:  # Required by Qt signal
        """Handle play/pause button toggle."""
        if checked:
            self.play_pause_btn.setText("Play")
            self.sim_widget.set_running(False)
        else:
            self.play_pause_btn.setText("Pause")
            self.sim_widget.set_running(True)

    def on_reset_clicked(self) -> None:
        """Reset simulation to initial state."""
        self.sim_widget.reset_state()
        # Reset all sliders to zero
        for slider in self.actuator_sliders:
            slider.setValue(0)

    def on_actuator_changed(self) -> None:
        """Update actuator torques when any slider changes."""
        for i, (slider, label) in enumerate(
            zip(self.actuator_sliders, self.actuator_labels, strict=False),
        ):
            value = float(slider.value())
            label.setText(f"{value:.0f} Nm")
            self.sim_widget.set_joint_torque(i, value)


def main() -> None:
    """Launch the advanced golf swing analysis application."""
    app = QtWidgets.QApplication(sys.argv)

    # Use advanced GUI by default
    win = AdvancedGolfAnalysisWindow()

    win.show()
    sys.exit(app.exec())


def main_simple() -> None:
    """Launch the simple/legacy golf swing application."""
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
