"""Pinocchio Pose Editor Tab.

Provides a full-featured pose editing interface for Pinocchio models including:
- Joint manipulation with sliders and spinboxes
- Gravity toggle for static posing
- Pose library (save/load/export/import/interpolate)
- Preset poses for common configurations
- Kinematic mode for direct joint control
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Bootstrap: add repo root to sys.path for src.* imports
_root = next(
    (p for p in Path(__file__).resolve().parents if (p / "pyproject.toml").exists()),
    Path(__file__).resolve().parent,
)
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(__file__)

from src.shared.python.engine_core.engine_availability import (  # noqa: E402
    PINOCCHIO_AVAILABLE,
    PYQT6_AVAILABLE,
)
from src.shared.python.logging_pkg.logging_config import get_logger  # noqa: E402
from src.shared.python.pose_editor.core import (  # noqa: E402
    BasePoseEditor,
    JointInfo,
    JointType,
)
from src.shared.python.pose_editor.library import PoseLibrary, StoredPose  # noqa: E402
from src.shared.python.pose_editor.widgets import (  # noqa: E402
    GravityControlWidget,
    JointSliderWidget,
    PoseLibraryWidget,
    SignalBlocker,
)

logger = get_logger(__name__)

# PyQt6 imports
if PYQT6_AVAILABLE:
    from PyQt6 import QtCore, QtGui, QtWidgets
else:
    QtCore = None  # type: ignore[misc, assignment]
    QtGui = None  # type: ignore[misc, assignment]
    QtWidgets = None  # type: ignore[misc, assignment]

# Pinocchio imports
if PINOCCHIO_AVAILABLE:
    import pinocchio as pin
else:
    pin = None  # type: ignore[misc, assignment]


class PinocchioPoseEditor(BasePoseEditor):
    """Pose editor implementation for Pinocchio physics engine."""

    def __init__(
        self,
        model: Any | None = None,
        data: Any | None = None,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> None:
        """Initialize the Pinocchio pose editor.

        Args:
            model: Pinocchio model
            data: Pinocchio data
            q: Initial position configuration
            v: Initial velocity configuration
        """
        super().__init__()
        self._model = model
        self._data = data
        self._q = q
        self._v = v
        self._original_gravity = np.array([0, 0, -9.81])
        self._update_callback: Any = None
        self._viz: Any = None  # Visualizer reference

        if model:
            self._initialize_joint_info()

    def set_model_and_data(
        self,
        model: Any,
        data: Any,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> None:
        """Set or update the model and data.

        Args:
            model: Pinocchio model
            data: Pinocchio data
            q: Position configuration (optional)
            v: Velocity configuration (optional)
        """
        self._model = model
        self._data = data
        self._q = q if q is not None else pin.neutral(model)
        self._v = v if v is not None else np.zeros(model.nv)
        self._original_gravity = model.gravity.linear.copy()
        self._initialize_joint_info()

    def set_visualizer(self, viz: Any) -> None:
        """Set the visualizer for updates.

        Args:
            viz: Pinocchio MeshcatVisualizer or similar
        """
        self._viz = viz

    def set_update_callback(self, callback: Any) -> None:
        """Set callback for visualization updates.

        Args:
            callback: Function to call when pose changes
        """
        self._update_callback = callback

    def _initialize_joint_info(self) -> None:
        """Initialize joint information from the model."""
        if self._model is None:
            return

        self._joint_info = []

        # Skip universe joint (index 0)
        for i in range(1, self._model.njoints):
            joint = self._model.joints[i]
            joint_name = self._model.names[i]

            # Determine joint type
            nq = joint.nq
            nv = joint.nv

            if nv == 1 and nq == 1:
                joint_type = JointType.REVOLUTE
            elif nv == 1 and nq == 1:
                joint_type = JointType.PRISMATIC
            elif nv == 3:
                joint_type = JointType.SPHERICAL
            elif nv == 6:
                joint_type = JointType.FREE
            else:
                joint_type = JointType.UNKNOWN

            # Get limits from model
            idx_q = joint.idx_q
            lower_limit = -np.pi
            upper_limit = np.pi

            if self._model.lowerPositionLimit is not None and idx_q < len(
                self._model.lowerPositionLimit
            ):
                lower = self._model.lowerPositionLimit[idx_q]
                upper = self._model.upperPositionLimit[idx_q]

                # Check for reasonable limits
                if np.isfinite(lower) and np.isfinite(upper):
                    lower_limit = lower
                    upper_limit = upper

            # Categorize joint
            group = self._categorize_joint(joint_name)

            info = JointInfo(
                name=joint_name,
                index=i,
                joint_type=joint_type,
                position_index=idx_q,
                velocity_index=joint.idx_v,
                num_positions=nq,
                num_velocities=nv,
                lower_limit=lower_limit,
                upper_limit=upper_limit,
                group=group,
                unit="rad" if joint_type == JointType.REVOLUTE else "m",
            )

            self._joint_info.append(info)

        # Initialize state
        if self._q is not None:
            self._state.joint_positions = self._q.copy()
        if self._v is not None:
            self._state.joint_velocities = self._v.copy()

        logger.info(
            "Initialized %d joints for Pinocchio pose editing", len(self._joint_info)
        )

    def _categorize_joint(self, name: str) -> str:
        """Categorize a joint into a group based on its name."""
        name_lower = name.lower()

        if any(
            x in name_lower for x in ["shoulder", "humerus", "elbow", "wrist", "arm"]
        ):
            if "l" in name_lower[:2]:
                return "Left Arm"
            else:
                return "Right Arm"

        if any(x in name_lower for x in ["hip", "knee", "ankle", "leg", "foot"]):
            if "l" in name_lower[:2]:
                return "Left Leg"
            else:
                return "Right Leg"

        if any(
            x in name_lower
            for x in ["spine", "back", "torso", "trunk", "lowerback", "upperback"]
        ):
            return "Spine"

        if any(x in name_lower for x in ["neck", "head"]):
            return "Head"

        if any(x in name_lower for x in ["pelvis", "root", "base"]):
            return "Pelvis"

        if any(x in name_lower for x in ["club", "shaft", "grip"]):
            return "Club"

        return "Other"

    def get_joint_info(self) -> list[JointInfo]:
        """Get information about all joints."""
        return self._joint_info

    def get_joint_position(self, joint_index: int) -> float | np.ndarray:
        """Get the current position of a joint."""
        if self._model is None or self._q is None:
            return 0.0

        for info in self._joint_info:
            if info.index == joint_index:
                if info.num_positions == 1:
                    return float(self._q[info.position_index])
                else:
                    return self._q[
                        info.position_index : info.position_index + info.num_positions
                    ]

        return 0.0

    def set_joint_position(self, joint_index: int, value: float | np.ndarray) -> None:
        """Set the position of a joint."""
        if self._model is None or self._q is None:
            return

        for info in self._joint_info:
            if info.index == joint_index:
                if info.num_positions == 1:
                    self._q[info.position_index] = float(value)
                else:
                    self._q[
                        info.position_index : info.position_index + info.num_positions
                    ] = value

                self._state.joint_positions = self._q.copy()
                self._notify("pose_changed", self._q)
                break

    def get_all_positions(self) -> np.ndarray:
        """Get all joint positions."""
        if self._q is None:
            return np.array([])
        return self._q.copy()

    def set_all_positions(self, positions: np.ndarray) -> None:
        """Set all joint positions."""
        if self._model is None:
            return

        self._q = positions.copy()
        self._state.joint_positions = positions.copy()
        self._notify("pose_changed", positions)

    def get_all_velocities(self) -> np.ndarray:
        """Get all joint velocities."""
        if self._v is None:
            return np.array([])
        return self._v.copy()

    def set_all_velocities(self, velocities: np.ndarray) -> None:
        """Set all joint velocities."""
        if self._model is None:
            return

        self._v = velocities.copy()
        self._state.joint_velocities = velocities.copy()

    def set_gravity_enabled(self, enabled: bool) -> None:
        """Enable or disable gravity."""
        if self._model is None:
            return

        try:
            if enabled:
                self._model.gravity.linear = self._original_gravity
                logger.info("Gravity enabled: %s", self._original_gravity)
            else:
                self._model.gravity.linear = np.zeros(3)
                logger.info("Gravity disabled")

            self._state.gravity_enabled = enabled
            self._notify("gravity_changed", enabled)

        except (ValueError, TypeError, RuntimeError) as e:
            logger.warning("Could not modify gravity: %s", e)

    def is_gravity_enabled(self) -> bool:
        """Check if gravity is enabled."""
        return self._state.gravity_enabled

    def _get_gravity_magnitude(self) -> float:
        """Get current gravity magnitude."""
        if self._model is None:
            return 9.81
        return float(np.linalg.norm(self._model.gravity.linear))

    def update_visualization(self) -> None:
        """Update visualization to reflect current pose."""
        if self._viz is not None and self._q is not None:
            try:
                self._viz.display(self._q)
            except (RuntimeError, ValueError, OSError) as e:
                logger.debug("Visualization update failed: %s", e)

        if self._update_callback:
            self._update_callback()

    def get_body_names(self) -> list[str]:
        """Get list of body names."""
        if self._model is None:
            return []
        return list(self._model.names)[1:]  # Skip universe

    def get_body_position(self, body_name: str) -> np.ndarray | None:
        """Get world position of a body."""
        if self._model is None or self._data is None or self._q is None:
            return None

        try:
            # Update kinematics
            pin.forwardKinematics(self._model, self._data, self._q)
            pin.updateFramePlacements(self._model, self._data)

            # Find frame
            for i, name in enumerate(self._model.names):
                if name == body_name:
                    return self._data.oMi[i].translation.copy()

            return None
        except (RuntimeError, ValueError, OSError):
            return None


class PinocchioPoseEditorTab(QtWidgets.QWidget):  # type: ignore[misc]
    """PyQt6 tab widget for Pinocchio pose editing.

    Provides a complete pose editing interface that can be added
    to the Pinocchio GUI application.
    """

    pose_changed = QtCore.pyqtSignal(np.ndarray)
    gravity_changed = QtCore.pyqtSignal(bool)

    def __init__(
        self,
        model: Any | None = None,
        data: Any | None = None,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the pose editor tab.

        Args:
            model: Pinocchio model
            data: Pinocchio data
            q: Initial position configuration
            v: Initial velocity configuration
            parent: Parent widget
        """
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for PinocchioPoseEditorTab")

        super().__init__(parent)

        self._editor = PinocchioPoseEditor(model, data, q, v)
        self._library = PoseLibrary()
        self._joint_widgets: dict[int, JointSliderWidget] = {}
        self._group_widgets: dict[str, QtWidgets.QWidget] = {}

        self._setup_ui()

        if model:
            self._build_joint_controls()

    def _setup_ui(self) -> None:
        """Create the user interface."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(QtWidgets.QLabel("<b>Pose Editor</b>"))
        header_layout.addStretch()

        self.lbl_mode = QtWidgets.QLabel("Mode: Kinematic")
        self.lbl_mode.setStyleSheet("color: #4CAF50; font-weight: bold;")
        header_layout.addWidget(self.lbl_mode)

        main_layout.addLayout(header_layout)

        # Create splitter
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)

        # Top: Gravity and actions
        top_widget = QtWidgets.QWidget()
        top_layout = QtWidgets.QVBoxLayout(top_widget)
        top_layout.setContentsMargins(0, 0, 0, 0)

        # Gravity control
        self.gravity_widget = GravityControlWidget()
        self.gravity_widget.gravity_changed.connect(self._on_gravity_changed)
        top_layout.addWidget(self.gravity_widget)

        # Quick actions
        actions_group = QtWidgets.QGroupBox("Quick Actions")
        actions_layout = QtWidgets.QGridLayout(actions_group)

        self.btn_reset = QtWidgets.QPushButton("Reset to Neutral")
        self.btn_reset.setToolTip("Reset to neutral configuration")
        self.btn_reset.clicked.connect(self._on_reset)
        actions_layout.addWidget(self.btn_reset, 0, 0)

        self.btn_zero_vel = QtWidgets.QPushButton("Zero Velocities")
        self.btn_zero_vel.setToolTip("Set all velocities to zero")
        self.btn_zero_vel.clicked.connect(self._on_zero_velocities)
        actions_layout.addWidget(self.btn_zero_vel, 0, 1)

        self.btn_t_pose = QtWidgets.QPushButton("T-Pose")
        self.btn_t_pose.clicked.connect(lambda: self._load_preset("T-Pose"))
        actions_layout.addWidget(self.btn_t_pose, 1, 0)

        self.btn_address = QtWidgets.QPushButton("Address")
        self.btn_address.clicked.connect(lambda: self._load_preset("Address"))
        actions_layout.addWidget(self.btn_address, 1, 1)

        top_layout.addWidget(actions_group)
        splitter.addWidget(top_widget)

        # Middle: Joint controls
        joints_group = QtWidgets.QGroupBox("Joint Controls")
        joints_layout = QtWidgets.QVBoxLayout(joints_group)

        # Filter
        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        self.txt_filter = QtWidgets.QLineEdit()
        self.txt_filter.setPlaceholderText("Search joints...")
        self.txt_filter.setClearButtonEnabled(True)
        self.txt_filter.textChanged.connect(self._filter_joints)
        filter_layout.addWidget(self.txt_filter)

        self.combo_group = QtWidgets.QComboBox()
        self.combo_group.addItem("All Groups")
        self.combo_group.currentTextChanged.connect(self._filter_joints)
        filter_layout.addWidget(self.combo_group)

        joints_layout.addLayout(filter_layout)

        # Scroll area
        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)

        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        self.slider_layout.setSpacing(2)

        self.scroll_area.setWidget(self.slider_container)
        joints_layout.addWidget(self.scroll_area)

        splitter.addWidget(joints_group)

        # Bottom: Pose library
        self.library_widget = PoseLibraryWidget(self._library)
        self.library_widget.pose_loaded.connect(self._on_pose_loaded)
        self.library_widget.interpolation_requested.connect(self._on_interpolation)
        self.library_widget.save_pose_requested = self._save_current_pose
        self.library_widget.preset_load_requested = self._load_preset_from_data
        splitter.addWidget(self.library_widget)

        splitter.setSizes([150, 300, 250])

    def set_model_and_data(
        self,
        model: Any,
        data: Any,
        q: np.ndarray | None = None,
        v: np.ndarray | None = None,
    ) -> None:
        """Update the model and data.

        Args:
            model: Pinocchio model
            data: Pinocchio data
            q: Position configuration
            v: Velocity configuration
        """
        self._editor.set_model_and_data(model, data, q, v)
        self._build_joint_controls()

    def set_visualizer(self, viz: Any) -> None:
        """Set the visualizer for updates."""
        self._editor.set_visualizer(viz)

    def set_update_callback(self, callback: Any) -> None:
        """Set callback for visualization updates."""
        self._editor.set_update_callback(callback)

    def set_state_references(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set direct references to q and v arrays.

        This allows the editor to modify the same arrays used by the main GUI.

        Args:
            q: Position configuration array
            v: Velocity configuration array
        """
        self._editor._q = q
        self._editor._v = v

    def _build_joint_controls(self) -> None:
        """Build joint control widgets."""
        # Clear existing
        for widget in self._joint_widgets.values():
            widget.deleteLater()
        self._joint_widgets.clear()
        self._group_widgets.clear()

        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

        joints = self._editor.get_joint_info()
        if not joints:
            self.slider_layout.addWidget(
                QtWidgets.QLabel("No joints available. Load a model first.")
            )
            return

        # Populate group filter
        groups = sorted(set(j.group for j in joints))
        self.combo_group.clear()
        self.combo_group.addItem("All Groups")
        for group in groups:
            self.combo_group.addItem(group)

        # Group joints
        grouped: dict[str, list[JointInfo]] = {}
        for joint in joints:
            if not joint.is_single_dof():
                continue
            if joint.group not in grouped:
                grouped[joint.group] = []
            grouped[joint.group].append(joint)

        # Create widgets by group
        for group_name in sorted(grouped.keys()):
            # Group header
            header = QtWidgets.QLabel(f"<b>{group_name}</b>")
            header.setStyleSheet("margin-top: 8px; padding: 4px; background: #e0e0e0;")
            header.setProperty("group", group_name)
            self.slider_layout.addWidget(header)
            self._group_widgets[group_name] = header

            for joint in grouped[group_name]:
                widget = JointSliderWidget(joint)
                widget.value_changed.connect(self._on_joint_changed)
                widget.setProperty("group", group_name)
                widget.setProperty("joint_name", joint.name)

                # Set initial value
                value = self._editor.get_joint_position(joint.index)
                if isinstance(value, np.ndarray):
                    value = float(value[0])
                widget.set_value(value)

                self.slider_layout.addWidget(widget)
                self._joint_widgets[joint.index] = widget

        self.slider_layout.addStretch()
        logger.info("Built controls for %d Pinocchio joints", len(self._joint_widgets))

    def _filter_joints(self, text: str = "") -> None:
        """Filter displayed joints."""
        search_text = self.txt_filter.text().lower()
        selected_group = self.combo_group.currentText()

        for i in range(self.slider_layout.count()):
            item = self.slider_layout.itemAt(i)
            if item is None:
                continue

            widget = item.widget()
            if widget is None:
                continue

            group = widget.property("group")
            joint_name = widget.property("joint_name")

            show = True

            if selected_group != "All Groups" and group != selected_group:
                show = False

            if search_text and joint_name:
                if search_text not in joint_name.lower():
                    show = False

            widget.setVisible(show)

    def _on_joint_changed(self, joint_index: int, value: float) -> None:
        """Handle joint value change."""
        self._editor.set_joint_position(joint_index, value)
        self._editor.update_visualization()
        self.pose_changed.emit(self._editor.get_all_positions())

    def _on_gravity_changed(self, enabled: bool) -> None:
        """Handle gravity toggle."""
        self._editor.set_gravity_enabled(enabled)
        self.gravity_changed.emit(enabled)

    def _on_reset(self) -> None:
        """Reset to neutral configuration."""
        if self._editor._model is not None:
            neutral = pin.neutral(self._editor._model)
            self._editor.set_all_positions(neutral)
            self._editor.zero_velocities()
            self._sync_sliders()
            self._editor.update_visualization()

    def _on_zero_velocities(self) -> None:
        """Zero all velocities."""
        self._editor.zero_velocities()

    def _on_pose_loaded(self, pose: StoredPose) -> None:
        """Handle pose loaded from library."""
        if len(pose.joint_positions) > 0:
            self._editor.set_all_positions(pose.joint_positions)
            if pose.joint_velocities is not None:
                self._editor.set_all_velocities(pose.joint_velocities)
            self._sync_sliders()
            self._editor.update_visualization()

    def _on_interpolation(self, pose_a: str, pose_b: str, alpha: float) -> None:
        """Handle interpolation request."""
        positions = self._library.interpolate(pose_a, pose_b, alpha)
        if positions is not None:
            self._editor.set_all_positions(positions)
            self._sync_sliders()
            self._editor.update_visualization()

    def _save_current_pose(self, name: str, description: str) -> None:
        """Save current pose to library."""
        positions = self._editor.get_all_positions()
        velocities = self._editor.get_all_velocities()

        named_positions = {}
        for joint in self._editor.get_joint_info():
            if joint.is_single_dof():
                named_positions[joint.name] = float(positions[joint.position_index])

        self._library.save_pose(
            name=name,
            positions=positions,
            velocities=velocities,
            description=description,
            named_positions=named_positions,
        )
        self.library_widget.refresh()

    def _load_preset(self, preset_name: str) -> None:
        """Load a preset pose by name."""
        from src.shared.python.pose_editor.library import get_preset_pose

        preset_data = get_preset_pose(preset_name)
        if preset_data:
            self._load_preset_from_data(preset_name, preset_data)

    def _load_preset_from_data(self, name: str, data: dict[str, Any]) -> None:
        """Load preset pose from data dictionary."""
        joints = self._editor.get_joint_info()
        positions = self._editor.get_all_positions()

        for joint in joints:
            if joint.name in data:
                value = data[joint.name]
                if isinstance(value, (int, float)):
                    positions[joint.position_index] = value

        self._editor.set_all_positions(positions)
        self._editor.zero_velocities()
        self._sync_sliders()
        self._editor.update_visualization()
        logger.info("Loaded preset: %s", name)

    def _sync_sliders(self) -> None:
        """Sync all sliders with current editor state."""
        for joint_idx, widget in self._joint_widgets.items():
            value = self._editor.get_joint_position(joint_idx)
            if isinstance(value, np.ndarray):
                value = float(value[0])
            with SignalBlocker(widget.slider, widget.spinbox):
                widget.set_value(value)

    def refresh(self) -> None:
        """Refresh the widget from editor state."""
        self._sync_sliders()
        self.library_widget.refresh()

    @property
    def library(self) -> PoseLibrary:
        """Get the pose library."""
        return self._library

    @property
    def editor(self) -> PinocchioPoseEditor:
        """Get the pose editor."""
        return self._editor
