"""Dialog for selecting MATLAB Simscape Models."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QFont, QIcon
from PyQt6.QtWidgets import (
    QDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from src.launchers.ui_components import ASSETS_DIR

# Define the models internally
MATLAB_MODELS = [
    {
        "id": "simscape_2d",
        "name": "Simscape 2D",
        "description": "2D Simscape Multibody Golf Swing Model (.slx)",
        "type": "matlab_file",
        "path": "src/engines/Simscape_Multibody_Models/2D_Golf_Model/matlab/GolfSwingZVCF.slx",
    },
    {
        "id": "simscape_3d",
        "name": "Simscape 3D",
        "description": "3D Simscape Multibody Golf Swing Model (.slx)",
        "type": "matlab_file",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/model/GolfSwing3D_Kinetic.slx",
    },
    {
        "id": "dataset_generator",
        "name": "Dataset Generator",
        "description": "Forward Dynamics Dataset Generator GUI (.m)",
        "type": "matlab_file",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/scripts/dataset_generator/Dataset_GUI.m",
    },
    {
        "id": "matlab_analysis",
        "name": "Analysis GUI",
        "description": "Golf Swing Analysis and Plotting Suite (.m)",
        "type": "matlab_file",
        "path": "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/2D GUI/main_scripts/golf_swing_analysis_gui.m",
    },
]


class MockModelConfig:
    """Mock config for launching."""

    def __init__(self, data: dict[str, str]):
        self.id = data["id"]
        self.name = data["name"]
        self.description = data["description"]
        self.type = data["type"]
        self.path = data["path"]
        self.source_root = None
        self.provider = None
        self.working_dir = None


class MatlabSuiteWidget(QWidget):
    """Widget to select and launch MATLAB/Simscape models."""

    def __init__(self, parent_launcher: Any, parent_dialog: Any = None) -> None:
        super().__init__(parent_launcher)
        self.parent_launcher = parent_launcher
        self.parent_dialog = parent_dialog
        self.setup_ui()

    def setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        title = QLabel("Select a MATLAB / Simscape Model to Launch")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Try to load matlab logo
        matlab_logo_path = ASSETS_DIR / "matlab_logo.png"
        has_logo = matlab_logo_path.exists()

        for model_data in MATLAB_MODELS:
            # Create button
            btn = QPushButton()
            btn.setMinimumHeight(60)
            btn.setCursor(Qt.CursorShape.PointingHandCursor)

            # Apply styling
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #2D2D2D;
                    color: white;
                    border: 1px solid #3D3D3D;
                    border-radius: 8px;
                    text-align: left;
                    padding: 10px;
                }
                QPushButton:hover {
                    background-color: #3D3D3D;
                    border: 1px solid #5D5D5D;
                }
            """)

            if has_logo:
                btn.setIcon(QIcon(str(matlab_logo_path)))
                btn.setIconSize(QSize(40, 40))

            # Set text with name and description
            btn.setText(f"  {model_data['name']} \n  {model_data['description']}")
            btn.setFont(QFont("Segoe UI", 10))

            # Connect signal
            btn.clicked.connect(lambda checked, m=model_data: self.launch_model(m))
            layout.addWidget(btn)

        if self.parent_dialog is not None:
            close_btn = QPushButton("Close")
            close_btn.setMinimumHeight(40)
            close_btn.clicked.connect(self.parent_dialog.accept)
            layout.addWidget(close_btn)

    def launch_model(self, model_data: dict[str, str]) -> None:
        """Launch the selected model."""
        model_obj = MockModelConfig(model_data)
        self.parent_launcher._launch_matlab_app(model_obj)
        if self.parent_dialog is not None:
            self.parent_dialog.accept()


class MatlabSuiteDialog(QDialog):
    """Dialog to select and launch MATLAB/Simscape models."""

    def __init__(self, parent_launcher: Any) -> None:
        super().__init__(parent_launcher)
        self.setWindowTitle("Matlab Simscape Models")
        self.setMinimumWidth(600)
        self.setup_ui(parent_launcher)

    def setup_ui(self, parent_launcher: Any) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.widget = MatlabSuiteWidget(parent_launcher, self)
        layout.addWidget(self.widget)

    def launch_model(self, model_data: dict[str, str]) -> None:
        """Forward launch to inner widget for backward compatibility."""
        self.widget.launch_model(model_data)
