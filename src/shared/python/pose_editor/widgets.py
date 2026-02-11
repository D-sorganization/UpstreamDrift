"""PyQt6 widgets for pose editing.

Provides reusable widgets for pose manipulation across all physics engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

# PyQt6 imports
try:
    from PyQt6 import QtCore, QtGui, QtWidgets

    PYQT6_AVAILABLE = True
except ImportError:
    PYQT6_AVAILABLE = False
    QtCore = None  # type: ignore[misc, assignment]
    QtGui = None  # type: ignore[misc, assignment]
    QtWidgets = None  # type: ignore[misc, assignment]

if TYPE_CHECKING:
    from .core import JointInfo, PoseEditorInterface
    from .library import PoseLibrary, StoredPose


class SignalBlocker:
    """Context manager to temporarily block widget signals."""

    def __init__(self, *widgets: Any) -> None:
        self.widgets = widgets

    def __enter__(self) -> None:
        for w in self.widgets:
            if w is not None:
                w.blockSignals(True)

    def __exit__(self, *args: Any) -> None:
        for w in self.widgets:
            if w is not None:
                w.blockSignals(False)


class JointSliderWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Widget for controlling a single joint with slider and spinbox."""

    value_changed = QtCore.pyqtSignal(int, float)  # joint_index, value

    SLIDER_SCALE = 1000.0  # Scale factor for int slider to float conversion

    def __init__(
        self,
        joint_info: JointInfo,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the joint slider widget.

        Args:
            joint_info: Information about the joint
            parent: Parent widget
        """
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for JointSliderWidget")

        super().__init__(parent)
        self.joint_info = joint_info
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the UI components."""
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(4)

        # Label
        self.label = QtWidgets.QLabel(self.joint_info.display_name)
        self.label.setMinimumWidth(100)
        self.label.setToolTip(f"Joint: {self.joint_info.name}")
        layout.addWidget(self.label)

        # Slider
        self.slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)

        lower = self.joint_info.lower_limit
        upper = self.joint_info.upper_limit
        if isinstance(lower, np.ndarray):
            lower = float(lower[0])
            upper = float(upper[0])

        self.slider.setMinimum(int(lower * self.SLIDER_SCALE))
        self.slider.setMaximum(int(upper * self.SLIDER_SCALE))
        self.slider.setValue(0)
        self.slider.setToolTip(
            f"Range: {lower:.2f} to {upper:.2f} {self.joint_info.unit}"
        )
        self.slider.valueChanged.connect(self._on_slider_changed)
        layout.addWidget(self.slider, stretch=1)

        # Spinbox
        self.spinbox = QtWidgets.QDoubleSpinBox()
        self.spinbox.setMinimum(lower)
        self.spinbox.setMaximum(upper)
        self.spinbox.setSingleStep(0.01)
        self.spinbox.setDecimals(3)
        self.spinbox.setSuffix(f" {self.joint_info.unit}")
        self.spinbox.setMinimumWidth(90)
        self.spinbox.valueChanged.connect(self._on_spinbox_changed)
        layout.addWidget(self.spinbox)

        # Reset button
        self.btn_reset = QtWidgets.QPushButton("0")
        self.btn_reset.setMaximumWidth(30)
        self.btn_reset.setToolTip("Reset to zero")
        self.btn_reset.clicked.connect(self._reset_to_zero)
        layout.addWidget(self.btn_reset)

    def _on_slider_changed(self, value: int) -> None:
        """Handle slider value change."""
        float_value = value / self.SLIDER_SCALE
        with SignalBlocker(self.spinbox):
            self.spinbox.setValue(float_value)
        self.value_changed.emit(self.joint_info.index, float_value)

    def _on_spinbox_changed(self, value: float) -> None:
        """Handle spinbox value change."""
        int_value = int(value * self.SLIDER_SCALE)
        with SignalBlocker(self.slider):
            self.slider.setValue(int_value)
        self.value_changed.emit(self.joint_info.index, value)

    def set_value(self, value: float) -> None:
        """Set the current value.

        Args:
            value: Joint position value
        """
        with SignalBlocker(self.slider, self.spinbox):
            self.slider.setValue(int(value * self.SLIDER_SCALE))
            self.spinbox.setValue(value)

    def get_value(self) -> float:
        """Get the current value.

        Returns:
            Current joint position value
        """
        return self.spinbox.value()

    def _reset_to_zero(self) -> None:
        """Reset to zero position."""
        self.set_value(0.0)
        self.value_changed.emit(self.joint_info.index, 0.0)


class GravityControlWidget(QtWidgets.QGroupBox):  # type: ignore[misc]
    """Widget for controlling gravity during pose editing."""

    gravity_changed = QtCore.pyqtSignal(bool)  # enabled state

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        """Initialize the gravity control widget."""
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for GravityControlWidget")

        super().__init__("Gravity Control", parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the UI components."""
        layout = QtWidgets.QVBoxLayout(self)

        # Enable/disable checkbox
        self.chk_gravity = QtWidgets.QCheckBox("Gravity Enabled")
        self.chk_gravity.setChecked(True)
        self.chk_gravity.setToolTip(
            "Disable gravity to pose the model without it falling"
        )
        self.chk_gravity.toggled.connect(self._on_gravity_toggled)
        layout.addWidget(self.chk_gravity)

        # Quick toggle button
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_toggle = QtWidgets.QPushButton("Toggle Gravity (G)")
        self.btn_toggle.setShortcut(QtGui.QKeySequence("G"))
        self.btn_toggle.setToolTip("Quickly toggle gravity on/off")
        self.btn_toggle.clicked.connect(self._toggle_gravity)
        btn_layout.addWidget(self.btn_toggle)
        layout.addLayout(btn_layout)

        # Status indicator
        self.lbl_status = QtWidgets.QLabel("Gravity: ON")
        self.lbl_status.setStyleSheet(
            "font-weight: bold; color: #2e7d32;"
        )  # Green for ON
        layout.addWidget(self.lbl_status)

        # Info label
        info = QtWidgets.QLabel(
            "<i>Tip: Disable gravity to pose the model<br>"
            "without it collapsing. Re-enable before<br>"
            "running simulation.</i>"
        )
        info.setStyleSheet("color: #666; font-size: 10px;")
        layout.addWidget(info)

    def _on_gravity_toggled(self, enabled: bool) -> None:
        """Handle gravity checkbox toggle."""
        self._update_status(enabled)
        self.gravity_changed.emit(enabled)

    def _toggle_gravity(self) -> None:
        """Toggle gravity state."""
        self.chk_gravity.setChecked(not self.chk_gravity.isChecked())

    def _update_status(self, enabled: bool) -> None:
        """Update the status display."""
        if enabled:
            self.lbl_status.setText("Gravity: ON")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #2e7d32;")
        else:
            self.lbl_status.setText("Gravity: OFF")
            self.lbl_status.setStyleSheet("font-weight: bold; color: #c62828;")

    def set_gravity_enabled(self, enabled: bool) -> None:
        """Set the gravity state.

        Args:
            enabled: True to enable gravity
        """
        with SignalBlocker(self.chk_gravity):
            self.chk_gravity.setChecked(enabled)
        self._update_status(enabled)

    def is_gravity_enabled(self) -> bool:
        """Check if gravity is enabled.

        Returns:
            True if gravity is enabled
        """
        return self.chk_gravity.isChecked()


class PoseLibraryWidget(QtWidgets.QGroupBox):  # type: ignore[misc]
    """Widget for managing pose library (save/load/export/import)."""

    pose_loaded = QtCore.pyqtSignal(object)  # Emits StoredPose
    interpolation_requested = QtCore.pyqtSignal(
        str, str, float
    )  # pose_a, pose_b, alpha

    def __init__(
        self,
        library: PoseLibrary,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the pose library widget.

        Args:
            library: PoseLibrary instance to manage
            parent: Parent widget
        """
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for PoseLibraryWidget")

        super().__init__("Pose Library", parent)
        self.library = library
        self._setup_ui()
        self._refresh_pose_list()

    def _setup_ui(self) -> None:
        """Create the UI components."""
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(6)

        # Save section
        save_layout = QtWidgets.QHBoxLayout()
        self.txt_pose_name = QtWidgets.QLineEdit()
        self.txt_pose_name.setPlaceholderText("Pose name...")
        self.txt_pose_name.setClearButtonEnabled(True)
        save_layout.addWidget(self.txt_pose_name)

        self.btn_save = QtWidgets.QPushButton("Save Pose")
        self.btn_save.setToolTip("Save current pose to library")
        self.btn_save.clicked.connect(self._on_save_pose)
        save_layout.addWidget(self.btn_save)
        layout.addLayout(save_layout)

        # Description
        self.txt_description = QtWidgets.QLineEdit()
        self.txt_description.setPlaceholderText("Description (optional)...")
        layout.addWidget(self.txt_description)

        # Pose list
        layout.addWidget(QtWidgets.QLabel("Saved Poses:"))
        self.list_poses = QtWidgets.QListWidget()
        self.list_poses.setMaximumHeight(150)
        self.list_poses.setSelectionMode(
            QtWidgets.QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.list_poses.itemDoubleClicked.connect(self._on_pose_double_clicked)
        layout.addWidget(self.list_poses)

        # Action buttons
        btn_layout = QtWidgets.QGridLayout()

        self.btn_load = QtWidgets.QPushButton("Load")
        self.btn_load.setToolTip("Apply selected pose")
        self.btn_load.clicked.connect(self._on_load_pose)
        btn_layout.addWidget(self.btn_load, 0, 0)

        self.btn_delete = QtWidgets.QPushButton("Delete")
        self.btn_delete.setToolTip("Delete selected pose")
        self.btn_delete.clicked.connect(self._on_delete_pose)
        btn_layout.addWidget(self.btn_delete, 0, 1)

        self.btn_export = QtWidgets.QPushButton("Export All")
        self.btn_export.setToolTip("Export library to JSON file")
        self.btn_export.clicked.connect(self._on_export)
        btn_layout.addWidget(self.btn_export, 1, 0)

        self.btn_import = QtWidgets.QPushButton("Import")
        self.btn_import.setToolTip("Import poses from JSON file")
        self.btn_import.clicked.connect(self._on_import)
        btn_layout.addWidget(self.btn_import, 1, 1)

        layout.addLayout(btn_layout)

        # Interpolation section
        interp_group = QtWidgets.QGroupBox("Pose Interpolation")
        interp_layout = QtWidgets.QVBoxLayout(interp_group)

        interp_info = QtWidgets.QLabel(
            "Select two poses in the list to blend between them"
        )
        interp_info.setStyleSheet("font-style: italic; font-size: 10px;")
        interp_layout.addWidget(interp_info)

        slider_layout = QtWidgets.QHBoxLayout()
        slider_layout.addWidget(QtWidgets.QLabel("Blend:"))

        self.slider_interp = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.slider_interp.setMinimum(0)
        self.slider_interp.setMaximum(100)
        self.slider_interp.setValue(0)
        self.slider_interp.valueChanged.connect(self._on_interpolation_changed)
        slider_layout.addWidget(self.slider_interp)

        self.lbl_interp = QtWidgets.QLabel("0%")
        self.lbl_interp.setMinimumWidth(40)
        slider_layout.addWidget(self.lbl_interp)

        interp_layout.addLayout(slider_layout)
        layout.addWidget(interp_group)

        # Presets section
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Presets:"))

        self.combo_presets = QtWidgets.QComboBox()
        self.combo_presets.addItem("-- Select Preset --")
        self._populate_presets()
        preset_layout.addWidget(self.combo_presets)

        self.btn_load_preset = QtWidgets.QPushButton("Load")
        self.btn_load_preset.clicked.connect(self._on_load_preset)
        preset_layout.addWidget(self.btn_load_preset)

        layout.addLayout(preset_layout)

    def _populate_presets(self) -> None:
        """Populate the presets combo box."""
        from .library import list_preset_poses

        for name in list_preset_poses():
            self.combo_presets.addItem(name)

    def _refresh_pose_list(self) -> None:
        """Refresh the pose list widget."""
        self.list_poses.clear()
        for pose_name in self.library.list_poses():
            pose = self.library.load_pose(pose_name)
            if pose:
                item = QtWidgets.QListWidgetItem(pose_name)
                item.setToolTip(pose.description or "No description")
                item.setData(QtCore.Qt.ItemDataRole.UserRole, pose)
                self.list_poses.addItem(item)

    def _on_save_pose(self) -> None:
        """Handle save pose button click."""
        name = self.txt_pose_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(
                self, "No Name", "Please enter a name for the pose."
            )
            return

        # Check for overwrite
        if name in self.library.list_poses():
            reply = QtWidgets.QMessageBox.question(
                self,
                "Overwrite?",
                f"Pose '{name}' already exists. Overwrite?",
                QtWidgets.QMessageBox.StandardButton.Yes
                | QtWidgets.QMessageBox.StandardButton.No,
            )
            if reply != QtWidgets.QMessageBox.StandardButton.Yes:
                return

        # This will be connected externally to actually save the pose
        self.save_pose_requested(name, self.txt_description.text().strip())

        self.txt_pose_name.clear()
        self.txt_description.clear()
        self._refresh_pose_list()

    def save_pose_requested(self, name: str, description: str) -> None:
        """Override this to handle pose saving.

        Args:
            name: Pose name
            description: Pose description
        """
        # Default implementation does nothing
        # Should be connected to editor to get current positions

    def _on_load_pose(self) -> None:
        """Handle load pose button click."""
        selected = self.list_poses.currentItem()
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "No Selection", "Please select a pose to load."
            )
            return

        pose = selected.data(QtCore.Qt.ItemDataRole.UserRole)
        if pose:
            self.pose_loaded.emit(pose)

    def _on_pose_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        """Handle double-click on pose item."""
        pose = item.data(QtCore.Qt.ItemDataRole.UserRole)
        if pose:
            self.pose_loaded.emit(pose)

    def _on_delete_pose(self) -> None:
        """Handle delete pose button click."""
        selected = self.list_poses.currentItem()
        if not selected:
            return

        name = selected.text()
        reply = QtWidgets.QMessageBox.question(
            self,
            "Delete Pose?",
            f"Are you sure you want to delete '{name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            self.library.delete_pose(name)
            self._refresh_pose_list()

    def _on_export(self) -> None:
        """Handle export button click."""
        if len(self.library.list_poses()) == 0:
            QtWidgets.QMessageBox.warning(self, "No Poses", "No poses to export.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pose Library",
            "poses.json",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                count = self.library.export_to_json(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Exported {count} poses to {filename}",
                )
            except (RuntimeError, ValueError, OSError) as e:
                QtWidgets.QMessageBox.critical(
                    self, "Export Error", f"Failed to export: {e}"
                )

    def _on_import(self) -> None:
        """Handle import button click."""
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Pose Library",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                count = self.library.import_from_json(filename)
                self._refresh_pose_list()
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Complete",
                    f"Imported {count} poses from {filename}",
                )
            except ImportError as e:
                QtWidgets.QMessageBox.critical(
                    self, "Import Error", f"Failed to import: {e}"
                )

    def _on_interpolation_changed(self, value: int) -> None:
        """Handle interpolation slider change."""
        alpha = value / 100.0
        self.lbl_interp.setText(f"{value}%")

        # Get selected poses
        selected_items = self.list_poses.selectedItems()
        if len(selected_items) != 2:
            return

        pose_a = selected_items[0].text()
        pose_b = selected_items[1].text()

        self.interpolation_requested.emit(pose_a, pose_b, alpha)

    def _on_load_preset(self) -> None:
        """Handle load preset button click."""
        from .library import get_preset_pose

        preset_name = self.combo_presets.currentText()
        if preset_name == "-- Select Preset --":
            return

        preset_data = get_preset_pose(preset_name)
        if preset_data:
            # Emit the preset data for the editor to apply
            self.preset_load_requested(preset_name, preset_data)

    def preset_load_requested(self, name: str, data: dict[str, Any]) -> None:
        """Override this to handle preset loading.

        Args:
            name: Preset name
            data: Preset joint values
        """

    def refresh(self) -> None:
        """Refresh the pose list from the library."""
        self._refresh_pose_list()


class PoseEditorWidget(QtWidgets.QWidget):  # type: ignore[misc]
    """Main widget for pose editing.

    Combines joint sliders, gravity control, and pose library into
    a complete pose editing interface.
    """

    pose_changed = QtCore.pyqtSignal(np.ndarray)  # Emits new positions

    def __init__(
        self,
        editor: PoseEditorInterface | None = None,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        """Initialize the pose editor widget.

        Args:
            editor: Pose editor implementation (optional, can be set later)
            parent: Parent widget
        """
        if not PYQT6_AVAILABLE:
            raise ImportError("PyQt6 is required for PoseEditorWidget")

        super().__init__(parent)
        self._editor = editor
        self._joint_widgets: dict[int, JointSliderWidget] = {}
        self._library = None

        self._setup_ui()

        if editor:
            self.set_editor(editor)

    def _setup_ui(self) -> None:
        """Create the UI components."""
        main_layout = QtWidgets.QVBoxLayout(self)
        main_layout.setContentsMargins(4, 4, 4, 4)

        # Header
        header_layout = QtWidgets.QHBoxLayout()
        header_layout.addWidget(QtWidgets.QLabel("<b>Pose Editor</b>"))
        header_layout.addStretch()

        self.btn_reset = QtWidgets.QPushButton("Reset All")
        self.btn_reset.setToolTip("Reset all joints to zero")
        self.btn_reset.clicked.connect(self._on_reset_all)
        header_layout.addWidget(self.btn_reset)

        main_layout.addLayout(header_layout)

        # Gravity control
        self.gravity_widget = GravityControlWidget()
        self.gravity_widget.gravity_changed.connect(self._on_gravity_changed)
        main_layout.addWidget(self.gravity_widget)

        # Joint sliders in scroll area
        scroll_group = QtWidgets.QGroupBox("Joint Controls")
        scroll_layout = QtWidgets.QVBoxLayout(scroll_group)

        self.scroll_area = QtWidgets.QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setMinimumHeight(200)

        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        self.slider_layout.setSpacing(2)

        self.scroll_area.setWidget(self.slider_container)
        scroll_layout.addWidget(self.scroll_area)

        main_layout.addWidget(scroll_group)

        # Pose library
        from .library import PoseLibrary

        self._library = PoseLibrary()
        self.library_widget = PoseLibraryWidget(self._library)
        self.library_widget.pose_loaded.connect(self._on_pose_loaded)
        self.library_widget.interpolation_requested.connect(self._on_interpolation)
        self.library_widget.save_pose_requested = self._save_current_pose
        self.library_widget.preset_load_requested = self._load_preset
        main_layout.addWidget(self.library_widget)

        # Zero velocities button
        self.btn_zero_vel = QtWidgets.QPushButton("Zero Velocities")
        self.btn_zero_vel.setToolTip("Set all joint velocities to zero")
        self.btn_zero_vel.clicked.connect(self._on_zero_velocities)
        main_layout.addWidget(self.btn_zero_vel)

    def set_editor(self, editor: PoseEditorInterface) -> None:
        """Set the pose editor implementation.

        Args:
            editor: Pose editor implementation
        """
        self._editor = editor
        self._build_joint_sliders()

    def _build_joint_sliders(self) -> None:
        """Build joint slider widgets from editor."""
        # Clear existing
        for widget in self._joint_widgets.values():
            widget.deleteLater()
        self._joint_widgets.clear()

        if self._editor is None:
            return

        # Get joint info
        joints = self._editor.get_joint_info()

        # Group joints by category
        groups: dict[str, list] = {}
        for joint in joints:
            if not joint.is_single_dof():
                continue
            group = joint.group
            if group not in groups:
                groups[group] = []
            groups[group].append(joint)

        # Create widgets by group
        for group_name in sorted(groups.keys()):
            # Group header
            header = QtWidgets.QLabel(f"<b>{group_name}</b>")
            header.setStyleSheet("margin-top: 8px;")
            self.slider_layout.addWidget(header)

            for joint in groups[group_name]:
                widget = JointSliderWidget(joint)
                widget.value_changed.connect(self._on_joint_value_changed)

                # Set initial value
                value = self._editor.get_joint_position(joint.index)
                if isinstance(value, np.ndarray):
                    value = float(value[0])
                widget.set_value(value)

                self.slider_layout.addWidget(widget)
                self._joint_widgets[joint.index] = widget

        # Add spacer
        self.slider_layout.addStretch()

    def _on_joint_value_changed(self, joint_index: int, value: float) -> None:
        """Handle joint value change from slider."""
        if self._editor:
            self._editor.set_joint_position(joint_index, value)
            self._editor.update_visualization()
            self.pose_changed.emit(self._editor.get_all_positions())

    def _on_gravity_changed(self, enabled: bool) -> None:
        """Handle gravity toggle."""
        if self._editor:
            self._editor.set_gravity_enabled(enabled)

    def _on_reset_all(self) -> None:
        """Reset all joints to zero."""
        if self._editor:
            positions = np.zeros_like(self._editor.get_all_positions())
            self._editor.set_all_positions(positions)
            self._sync_sliders_from_editor()
            self._editor.update_visualization()

    def _on_zero_velocities(self) -> None:
        """Zero all velocities."""
        if self._editor:
            velocities = np.zeros_like(self._editor.get_all_velocities())
            self._editor.set_all_velocities(velocities)

    def _on_pose_loaded(self, pose: StoredPose) -> None:
        """Handle pose loaded from library."""
        if self._editor and len(pose.joint_positions) > 0:
            self._editor.set_all_positions(pose.joint_positions)
            if pose.joint_velocities is not None:
                self._editor.set_all_velocities(pose.joint_velocities)
            self._sync_sliders_from_editor()
            self._editor.update_visualization()

    def _on_interpolation(self, pose_a: str, pose_b: str, alpha: float) -> None:
        """Handle interpolation request."""
        if self._editor and self._library:
            positions = self._library.interpolate(pose_a, pose_b, alpha)
            if positions is not None:
                self._editor.set_all_positions(positions)
                self._sync_sliders_from_editor()
                self._editor.update_visualization()

    def _save_current_pose(self, name: str, description: str) -> None:
        """Save the current pose to the library."""
        if self._editor and self._library:
            positions = self._editor.get_all_positions()
            velocities = self._editor.get_all_velocities()
            self._library.save_pose(
                name=name,
                positions=positions,
                velocities=velocities,
                description=description,
            )

    def _load_preset(self, name: str, data: dict[str, Any]) -> None:
        """Load a preset pose."""
        if self._editor:
            joints = self._editor.get_joint_info()
            positions = self._editor.get_all_positions()

            for joint in joints:
                if joint.name in data:
                    value = data[joint.name]
                    if isinstance(value, (int, float)):
                        positions[joint.position_index] = value

            self._editor.set_all_positions(positions)
            self._sync_sliders_from_editor()
            self._editor.update_visualization()

    def _sync_sliders_from_editor(self) -> None:
        """Sync all sliders with current editor state."""
        if self._editor is None:
            return

        for joint_idx, widget in self._joint_widgets.items():
            value = self._editor.get_joint_position(joint_idx)
            if isinstance(value, np.ndarray):
                value = float(value[0])
            with SignalBlocker(widget.slider, widget.spinbox):
                widget.set_value(value)

    def refresh(self) -> None:
        """Refresh the widget from editor state."""
        self._sync_sliders_from_editor()
        if self._library:
            self.library_widget.refresh()

    @property
    def library(self) -> PoseLibrary | None:
        """Get the pose library."""
        return self._library
