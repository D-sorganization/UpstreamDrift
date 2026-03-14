"""Main Window for Pinocchio Golf GUI."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pinocchio as pin  # type: ignore
from PyQt6 import QtWidgets

from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)
from src.shared.python.ui.simulation_gui_base import SimulationGUIBase

from ..manipulability import PinocchioManipulabilityAnalyzer
from ..pinocchio_recorder import PinocchioRecorder

# Mixin imports (relative to this file)
from .analysis_mixin import AnalysisMixin
from .simulation_mixin import SimulationMixin
from .ui_setup_mixin import UISetupMixin
from .visualization_mixin import VisualizationMixin

# Check meshcat availability
try:
    import meshcat.geometry as g
    import meshcat.visualizer as viz

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    g = None  # type: ignore
    viz = None  # type: ignore

if MESHCAT_AVAILABLE:
    from pinocchio.visualize import MeshcatVisualizer
else:
    MeshcatVisualizer = object  # Dummy class if missing

# Set up logging using centralized module
configure_gui_logging()
logger = get_logger(__name__)

# Constants
DT_DEFAULT = 0.01
SLIDER_RANGE_RAD = 10.0
SLIDER_SCALE = 100.0
COM_SPHERE_RADIUS = 0.02
COM_COLOR = 0xFFFF00


class PinocchioGUI(
    UISetupMixin,
    SimulationMixin,
    VisualizationMixin,
    AnalysisMixin,
    SimulationGUIBase,
):
    """Main GUI widget for Pinocchio robot visualization and computation."""

    WINDOW_TITLE = "Pinocchio Golf Model (Dynamics & Kinematics)"
    WINDOW_WIDTH = 1000
    WINDOW_HEIGHT = 900

    def __init__(self) -> None:
        """Initialize the Pinocchio GUI."""
        self.dt = DT_DEFAULT
        self._init_internal_state()

        super().__init__()

        pin_version = getattr(pin, "__version__", "unknown")
        logger.info(f"Pinocchio Version: {pin_version}")

        self._init_meshcat_viewer()

        self.available_models: list[dict[str, Any]] = []
        self._scan_urdf_models()

        self._setup_ui()
        self._load_default_model()

    def _init_internal_state(self) -> None:
        """Initialize physics, visualization, and data components."""
        self.model: pin.Model | None = None
        self.data: pin.Data | None = None
        self.visual_model: pin.VisualModel | None = None
        self.collision_model: pin.CollisionModel | None = None
        self.viz: MeshcatVisualizer | None = None
        self.q: np.ndarray | None = None
        self.v: np.ndarray | None = None

        self.analyzer: Any | None = None
        self.latest_induced: dict[str, np.ndarray] | None = None
        self.latest_cf: dict[str, np.ndarray] | None = None

        self.manip_analyzer: PinocchioManipulabilityAnalyzer | None = None
        self.manip_checkboxes: dict[str, QtWidgets.QCheckBox] = {}

        self.recorder = PinocchioRecorder(engine=self)
        self.sim_time = 0.0

        self.joint_sliders: list[QtWidgets.QSlider] = []
        self.joint_spinboxes: list[QtWidgets.QDoubleSpinBox] = []
        self.joint_names: list[str] = []

    def _load_default_model(self) -> None:
        """Load the default golfer model if available."""
        # Use relative path from this file's directory
        default_urdf = (
            Path(__file__).parent / "../../../models/generated/golfer.urdf"
        ).resolve()

        if default_urdf.exists():
            self.available_models.insert(
                0, {"name": "Default: Golfer", "path": str(default_urdf)}
            )
            self.load_urdf(str(default_urdf))
        else:
            self.available_models.insert(0, {"name": "Select Model...", "path": None})

    def get_joint_names(self) -> list[str]:
        """Return joint names for the Live Analysis widget."""
        return self.joint_names

    def log_write(self, text: str) -> None:
        """Append a message to the UI log panel and logger."""
        assert text is not None, "text must be provided"
        if hasattr(self, "log"):
            self.log.append(text)
        logger.info(text)

    def _populate_model_combo(self) -> None:
        """Populate the model dropdown list."""
        if hasattr(self, "model_combo"):
            self.model_combo.clear()
            for model in self.available_models:
                self.model_combo.addItem(model["name"])

    def _on_model_combo_changed(self, index: int) -> None:
        """Handle model selection from the dropdown."""
        if 0 <= index < len(self.available_models):
            path = self.available_models[index]["path"]
            if path:
                self.load_urdf(path)

    # ==================================================================
    # SimulationGUIBase Abstract Overrides
    # ==================================================================

    def _build_base_ui(self) -> None:
        """Override base UI construction."""
        pass

    def step_simulation(self) -> None:
        """Advance the Pinocchio simulation (SimulationMixin)."""
        SimulationMixin.step_simulation(self)

    def reset_simulation(self) -> None:
        """Reset the simulation."""
        if self.model is None:
            return
        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.is_running = False
        self.sim_time = 0.0
        self.btn_run.setText("Run Simulation")
        self.btn_run.setChecked(False)
        self._update_viewer()
        self._sync_kinematic_controls()
        self.recorder.reset()
        self.lbl_rec_status.setText("Frames: 0")
        self.log_write("Simulation Reset.")

    def update_visualization(self) -> None:
        """Refresh visualizations (VisualizationMixin)."""
        self._update_viewer()

    def load_model(self, index: int) -> None:
        """Load selected model."""
        self._on_model_combo_changed(index)

    def sync_kinematic_controls(self) -> None:
        """Sync sliders."""
        self._sync_kinematic_controls()

    def start_recording(self) -> None:
        """Start recording."""
        self.recorder.start_recording()
        self.log_write("Recording Started.")

    def stop_recording(self) -> None:
        """Stop recording."""
        self.recorder.stop_recording()
        self.log_write(f"Recording Stopped. Frames: {self.recorder.get_num_frames()}")

    def get_recording_frame_count(self) -> int:
        return self.recorder.get_num_frames()

    def export_data(self, filename: str) -> None:
        self._export_statistics()

    def on_model_loaded(self) -> None:
        self.log_write(f"Model Loaded: {self.model.name if self.model else 'None'}")

    def on_live_analysis_toggled(self, checked: bool) -> None:
        status = "Enabled" if checked else "Disabled"
        self.log_write(f"Live Analysis {status}")


def main() -> None:
    """Main entry point for the GUI application."""
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    window = PinocchioGUI()
    window.show()
    sys.exit(app.exec())
