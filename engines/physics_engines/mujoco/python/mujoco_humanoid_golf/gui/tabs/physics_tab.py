from __future__ import annotations

import logging
import typing
from pathlib import Path

from PyQt6 import QtCore, QtWidgets

from ...common_utils import get_shared_urdf_path
from ...linkage_mechanisms import LINKAGE_CATALOG
from ...models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    FULL_BODY_GOLF_SWING_XML,
    MYOARM_SIMPLE_PATH,
    MYOBODY_PATH,
    MYOUPPERBODY_PATH,
    TRIPLE_PENDULUM_XML,
    UPPER_BODY_GOLF_SWING_XML,
)
from ...sim_widget import MuJoCoSimWidget

if typing.TYPE_CHECKING:
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = logging.getLogger(__name__)


class PhysicsTab(QtWidgets.QWidget):
    """Tab for physics engine configuration and model selection."""

    # Signal emitted when model changes
    # Arguments: model_name, config_dict
    model_changed = QtCore.pyqtSignal(str, dict)

    # Signal emitted when operating mode changes
    # Arguments: mode ("dynamic" or "kinematic")
    mode_changed = QtCore.pyqtSignal(str)

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window

        self.model_configs: list[dict] = []
        self.model_descriptions: dict[int, str] = {}

        self._init_model_configs()
        self._setup_ui()

    def _init_model_configs(self) -> None:
        """Initialize the list of available physics models."""
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
            {
                "name": "myoupperbody",
                "xml_path": MYOUPPERBODY_PATH,
                "actuators": [
                    "R Shoulder Flex",
                    "R Shoulder Add",
                    "R Shoulder Rot",
                    "R Elbow",
                    "R Forearm",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "L Shoulder Flex",
                    "L Shoulder Add",
                    "L Shoulder Rot",
                    "L Elbow",
                    "L Forearm",
                    "L Wrist Flex",
                    "L Wrist Dev",
                    "R Erector Spinae",
                    "L Erector Spinae",
                    "R Int Oblique",
                    "L Int Oblique",
                    "R Ext Oblique",
                    "L Ext Oblique",
                ],
            },
            {
                "name": "myobody",
                "xml_path": MYOBODY_PATH,
                "actuators": [f"Muscle {i + 1}" for i in range(290)],
            },
            {
                "name": "myoarm_simple",
                "xml_path": MYOARM_SIMPLE_PATH,
                "actuators": [
                    "R Shoulder Flex",
                    "R Shoulder Add",
                    "R Shoulder Rot",
                    "R Elbow",
                    "R Forearm",
                    "R Wrist Flex",
                    "R Wrist Dev",
                    "L Shoulder Flex",
                    "L Shoulder Add",
                    "L Shoulder Rot",
                    "L Elbow",
                    "L Forearm",
                    "L Wrist Flex",
                    "L Wrist Dev",
                ],
            },
        ]

        # Add linkage mechanisms
        for mech_name, mech_config in LINKAGE_CATALOG.items():
            self.model_configs.append(
                {
                    "name": mech_name.lower()
                    .replace(" ", "_")
                    .replace(":", "")
                    .replace("(", "")
                    .replace(")", ""),
                    "xml": mech_config["xml"],
                    "actuators": mech_config["actuators"],
                    "category": mech_config.get("category", "Mechanisms"),
                    "description": mech_config.get("description", ""),
                }
            )

        # Add shared URDF models
        self._load_shared_urdfs()

        # Connect to sim_widget loading signals
        if hasattr(self.sim_widget, "loading_started"):
            self.sim_widget.loading_started.connect(self._on_loading_started)
            self.sim_widget.loading_finished.connect(self._on_loading_finished)

    def _load_shared_urdfs(self) -> None:
        """Load URDF models from shared/urdf directory."""
        base_dir = Path(get_shared_urdf_path()).parent
        if not base_dir.exists():
            return

        for urdf_file in base_dir.glob("*.urdf"):
            name = urdf_file.stem
            self.model_configs.append(
                {
                    "name": name,
                    "xml_path": urdf_file,
                    "actuators": ["Joint 1", "Joint 2"],  # Generic fallback
                    "description": "Shared URDF Model",
                }
            )

    def _setup_ui(self) -> None:
        """Create the physics configuration UI."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # 1. Model Selector
        model_group = QtWidgets.QGroupBox("Physics Models & Mechanisms")
        model_layout = QtWidgets.QVBoxLayout(model_group)
        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setToolTip(
            "Select a physics model to simulate.\n"
            "DOF = Degrees of Freedom\n"
            "Higher DOF = more complex/realistic model"
        )
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self.on_model_changed)
        model_layout.addWidget(self.model_combo)

        # Model description label
        self.model_description_label = QtWidgets.QLabel()
        self.model_description_label.setWordWrap(True)
        # Style set in dark_theme.qss
        self.model_description_label.setObjectName("descriptionLabel")
        model_layout.addWidget(self.model_description_label)
        self._update_model_description(0)
        main_layout.addWidget(model_group)

        # 2. Operating Mode Selector
        mode_group = QtWidgets.QGroupBox("Operating Mode")
        mode_layout = QtWidgets.QHBoxLayout(mode_group)
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(
            ["Dynamic (Torque Control)", "Kinematic (Pose Adjustment)"]
        )
        self.mode_combo.setToolTip(
            "Dynamic: Physics-driven simulation using torques.\n"
            "Kinematic: Direct control of joint positions (pose)."
        )
        self.mode_combo.currentIndexChanged.connect(self._on_operating_mode_changed)
        mode_layout.addWidget(QtWidgets.QLabel("Mode:"))
        mode_layout.addWidget(self.mode_combo)
        main_layout.addWidget(mode_group)

        # Add stretch to push everything to top
        main_layout.addStretch(1)

        # Select default model (Full Body Golf Swing)
        # Indices: 0=Chaotic, 1=Double, 2=Triple, 3=Upper, 4=Full
        default_index = 4
        if self.model_combo.count() > default_index:
            self.model_combo.setCurrentIndex(default_index)

    def _populate_model_combo(self) -> None:
        # Add basic described models first
        desc_map = {
            0: "Simple driven pendulum showing chaotic behavior.",
            1: "Basic swing with shoulder and wrist joints. Simplest realistic model.",
            2: "Adds elbow joint for more realistic arm mechanics.",
            3: "Upper body model with spine rotation and both arms.",
            4: "Full body model including leg drive and weight transfer.",
            5: "Most detailed model with scapulae, 3-DOF shoulders, flexible shaft.",
            6: "Muscle-actuated upper body. Independent muscle control.",
            7: "Complete musculoskeletal model. Very complex - for advanced users.",
            8: "Both arms with muscle actuation. Good for arm mechanics study.",
        }

        for i, config in enumerate(self.model_configs):
            display_name = config["name"].replace("_", " ").title()
            if "category" in config:
                display_name = f"{config['category']}: {display_name}"
            elif i < 9:  # The basic golf/myo models
                prefix = (
                    "Golf"
                    if "golf" in config["name"] or "pendulum" in config["name"]
                    else "Musculoskeletal"
                )
                display_name = (
                    f"{prefix}: {display_name} ({len(config['actuators'])} DOF)"
                )

            self.model_combo.addItem(display_name)

            # Store description
            mapped_desc = desc_map.get(i)
            desc = str(mapped_desc) if mapped_desc else None

            if not desc:
                desc = str(config.get("description", "Imported model"))
            self.model_descriptions[i] = desc

    def _update_model_description(self, index: int) -> None:
        if index in self.model_descriptions:
            self.model_description_label.setText(self.model_descriptions[index])
        else:
            self.model_description_label.setText("")

    def on_model_changed(self, index: int) -> None:
        """Handle model selection change."""
        self.load_current_model()
        self._update_model_description(index)

    def _on_loading_started(self) -> None:
        """Handle start of model loading."""
        self.model_combo.setEnabled(False)
        self.mode_combo.setEnabled(False)
        if hasattr(self.main_window, "statusBar"):
            self.main_window.statusBar().showMessage("Loading physics model...")

    def _on_loading_finished(self, success: bool) -> None:
        """Handle completion of model loading."""
        self.model_combo.setEnabled(True)
        self.mode_combo.setEnabled(True)

        if hasattr(self.main_window, "statusBar"):
            if success:
                self.main_window.statusBar().showMessage(
                    "Model loaded successfully.", 3000
                )
            else:
                self.main_window.statusBar().showMessage("Model load failed.", 5000)

        if success:
            self._finalize_model_change()

    def _finalize_model_change(self) -> None:
        """Post-load configuration update."""
        index = self.model_combo.currentIndex()
        if index < 0 or index >= len(self.model_configs):
            return

        config = self.model_configs[index]

        # Handle actuator count mismatch
        model = self.sim_widget.model
        if model:
            if model.nu != len(config["actuators"]):
                # Fix up
                diff = model.nu - len(config["actuators"])
                if diff > 0:
                    config["actuators"].extend(
                        [
                            f"Actuator {i}"
                            for i in range(len(config["actuators"]), model.nu)
                        ]
                    )
                else:
                    config["actuators"] = config["actuators"][: model.nu]

        self.sim_widget.verify_control_system()

        # Emit signal so other tabs can update
        self.model_changed.emit(config["name"], config)

        # Trigger mode update too
        self._on_operating_mode_changed(self.mode_combo.currentIndex())

    def load_current_model(self) -> None:
        """Load selected model and emit change signal."""
        index = self.model_combo.currentIndex()
        if index < 0 or index >= len(self.model_configs):
            return

        config = self.model_configs[index]

        try:
            if hasattr(self.sim_widget, "load_model_async"):
                if "xml_path" in config:
                    self.sim_widget.load_model_async(
                        str(config["xml_path"]), is_file=True
                    )
                elif "xml" in config:
                    self.sim_widget.load_model_async(str(config["xml"]), is_file=False)
                else:
                    raise ValueError(f"Invalid config: {config['name']}")
            else:
                # Fallback to sync
                if "xml_path" in config:
                    self.sim_widget.load_model_from_file(str(config["xml_path"]))
                elif "xml" in config:
                    self.sim_widget.load_model_from_xml(str(config["xml"]))
                else:
                    raise ValueError(f"Invalid config: {config['name']}")

                # If sync, manually trigger finalize
                self._finalize_model_change()

        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error", f"Failed to load model: {e}")
            logger.error("Failed to load model: %s", e)
            return

    def _on_operating_mode_changed(self, index: int) -> None:
        """Handle operating mode change (Dynamic vs Kinematic)."""
        mode = "dynamic" if index == 0 else "kinematic"
        self.sim_widget.set_operating_mode(mode)
        self.mode_changed.emit(mode)
