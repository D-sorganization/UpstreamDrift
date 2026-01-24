from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco
from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_config import get_logger

from ...interactive_manipulation import ConstraintType
from ...sim_widget import MuJoCoSimWidget

if TYPE_CHECKING:
    from ..advanced_gui import AdvancedGolfAnalysisWindow

logger = get_logger(__name__)


class ManipulationTab(QtWidgets.QWidget):
    """Tab for interactive manipulation and pose management."""

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        main_window: AdvancedGolfAnalysisWindow,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self.main_window = main_window
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the interactive manipulation interface."""
        manip_layout = QtWidgets.QVBoxLayout(self)
        manip_layout.setContentsMargins(8, 8, 8, 8)

        # Global Body Selection
        sel_group = QtWidgets.QGroupBox("Target Selection")
        sel_layout = QtWidgets.QHBoxLayout(sel_group)
        sel_layout.addWidget(QtWidgets.QLabel("Select Body:"))
        self.manip_body_combo = QtWidgets.QComboBox()
        self.manip_body_combo.currentIndexChanged.connect(self.on_manip_body_selected)
        sel_layout.addWidget(self.manip_body_combo, stretch=1)
        manip_layout.addWidget(sel_group)

        # Drag mode controls
        drag_group = QtWidgets.QGroupBox("Drag Mode")
        drag_layout = QtWidgets.QVBoxLayout(drag_group)

        self.enable_drag_cb = QtWidgets.QCheckBox("Enable Drag Manipulation")
        self.enable_drag_cb.setChecked(True)
        self.enable_drag_cb.stateChanged.connect(self.on_drag_enabled_changed)
        drag_layout.addWidget(self.enable_drag_cb)

        self.maintain_orientation_cb = QtWidgets.QCheckBox(
            "Maintain Orientation While Dragging"
        )
        self.maintain_orientation_cb.stateChanged.connect(
            self.on_maintain_orientation_changed
        )
        drag_layout.addWidget(self.maintain_orientation_cb)

        self.nullspace_posture_cb = QtWidgets.QCheckBox(
            "Use Nullspace Posture Optimization"
        )
        self.nullspace_posture_cb.setChecked(True)
        self.nullspace_posture_cb.stateChanged.connect(
            self.on_nullspace_posture_changed
        )
        drag_layout.addWidget(self.nullspace_posture_cb)

        manip_layout.addWidget(drag_group)

        # Manual Transform Controls
        transform_group = QtWidgets.QGroupBox("Manual Transform (Selected Body)")
        transform_layout = QtWidgets.QVBoxLayout(transform_group)

        # Position
        pos_layout = QtWidgets.QHBoxLayout()
        pos_layout.addWidget(QtWidgets.QLabel("Pos:"))
        self.trans_x = QtWidgets.QDoubleSpinBox()
        self.trans_x.setRange(-10, 10)
        self.trans_x.setSingleStep(0.01)
        self.trans_x.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 0, v)
        )
        pos_layout.addWidget(self.trans_x)

        self.trans_y = QtWidgets.QDoubleSpinBox()
        self.trans_y.setRange(-10, 10)
        self.trans_y.setSingleStep(0.01)
        self.trans_y.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 1, v)
        )
        pos_layout.addWidget(self.trans_y)

        self.trans_z = QtWidgets.QDoubleSpinBox()
        self.trans_z.setRange(-10, 10)
        self.trans_z.setSingleStep(0.01)
        self.trans_z.valueChanged.connect(
            lambda v: self.on_manual_transform("pos", 2, v)
        )
        pos_layout.addWidget(self.trans_z)
        transform_layout.addLayout(pos_layout)

        # Rotation (Euler)
        rot_layout = QtWidgets.QHBoxLayout()
        rot_layout.addWidget(QtWidgets.QLabel("Rot:"))
        self.trans_roll = QtWidgets.QDoubleSpinBox()  # X
        self.trans_roll.setRange(-180, 180)
        self.trans_roll.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 0, v)
        )
        rot_layout.addWidget(self.trans_roll)

        self.trans_pitch = QtWidgets.QDoubleSpinBox()  # Y
        self.trans_pitch.setRange(-180, 180)
        self.trans_pitch.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 1, v)
        )
        rot_layout.addWidget(self.trans_pitch)

        self.trans_yaw = QtWidgets.QDoubleSpinBox()  # Z
        self.trans_yaw.setRange(-180, 180)
        self.trans_yaw.valueChanged.connect(
            lambda v: self.on_manual_transform("rot", 2, v)
        )
        rot_layout.addWidget(self.trans_yaw)
        transform_layout.addLayout(rot_layout)

        # Helper button to get current values from selection
        self.refresh_trans_btn = QtWidgets.QPushButton("Refresh from Selection")
        self.refresh_trans_btn.clicked.connect(self.update_manual_transform_values)
        transform_layout.addWidget(self.refresh_trans_btn)

        manip_layout.addWidget(transform_group)

        # Constraint controls
        constraint_group = QtWidgets.QGroupBox("Body Constraints")
        constraint_layout = QtWidgets.QVBoxLayout(constraint_group)

        # Body selection
        body_select_layout = QtWidgets.QHBoxLayout()
        body_label = QtWidgets.QLabel("Body:")
        self.constraint_body_combo = QtWidgets.QComboBox()
        body_label.setBuddy(self.constraint_body_combo)
        body_select_layout.addWidget(body_label)
        self.constraint_body_combo.setMinimumWidth(150)
        body_select_layout.addWidget(self.constraint_body_combo, stretch=1)
        constraint_layout.addLayout(body_select_layout)

        # Constraint type
        type_layout = QtWidgets.QHBoxLayout()
        type_label = QtWidgets.QLabel("Type:")
        self.constraint_type_combo = QtWidgets.QComboBox()
        type_label.setBuddy(self.constraint_type_combo)
        type_layout.addWidget(type_label)
        self.constraint_type_combo.addItems(["Fixed in Space", "Relative to Body"])
        type_layout.addWidget(self.constraint_type_combo, stretch=1)
        constraint_layout.addLayout(type_layout)

        # Reference body (for relative constraints)
        self.ref_body_layout = QtWidgets.QHBoxLayout()
        ref_label = QtWidgets.QLabel("Reference:")
        self.ref_body_combo = QtWidgets.QComboBox()
        ref_label.setBuddy(self.ref_body_combo)
        self.ref_body_layout.addWidget(ref_label)
        self.ref_body_combo.setMinimumWidth(150)
        self.ref_body_layout.addWidget(self.ref_body_combo, stretch=1)
        self.ref_body_widget = QtWidgets.QWidget()
        self.ref_body_widget.setLayout(self.ref_body_layout)
        self.ref_body_widget.setVisible(False)
        constraint_layout.addWidget(self.ref_body_widget)

        self.constraint_type_combo.currentIndexChanged.connect(
            lambda idx: self.ref_body_widget.setVisible(idx == 1)
        )

        # Constraint buttons
        constraint_btn_layout = QtWidgets.QHBoxLayout()
        self.add_constraint_btn = QtWidgets.QPushButton("Add Constraint")
        self.add_constraint_btn.clicked.connect(self.on_add_constraint)
        self.remove_constraint_btn = QtWidgets.QPushButton("Remove Constraint")
        self.remove_constraint_btn.clicked.connect(self.on_remove_constraint)
        constraint_btn_layout.addWidget(self.add_constraint_btn)
        constraint_btn_layout.addWidget(self.remove_constraint_btn)
        constraint_layout.addLayout(constraint_btn_layout)

        # Clear all constraints button
        self.clear_constraints_btn = QtWidgets.QPushButton("Clear All Constraints")
        self.clear_constraints_btn.clicked.connect(self.on_clear_constraints)
        self.clear_constraints_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #d62728;
            }
            QPushButton:hover {
                background-color: #a81f20;
            }
        """
        )
        constraint_layout.addWidget(self.clear_constraints_btn)

        # Constrained bodies list
        constraint_layout.addWidget(QtWidgets.QLabel("Active Constraints:"))
        self.constraints_list = QtWidgets.QListWidget()
        self.constraints_list.setMaximumHeight(100)
        constraint_layout.addWidget(self.constraints_list)

        manip_layout.addWidget(constraint_group)

        # Pose library controls
        pose_group = QtWidgets.QGroupBox("Pose Library")
        pose_layout = QtWidgets.QVBoxLayout(pose_group)

        # Save pose
        save_layout = QtWidgets.QHBoxLayout()
        self.pose_name_input = QtWidgets.QLineEdit()
        self.pose_name_input.setPlaceholderText("Pose name...")
        self.pose_name_input.setClearButtonEnabled(True)
        self.pose_name_input.setAccessibleName("Pose Name")
        save_layout.addWidget(self.pose_name_input)
        self.save_pose_btn = QtWidgets.QPushButton("Save Pose")
        self.save_pose_btn.setToolTip(
            "Save the current body configuration to the library"
        )
        self.save_pose_btn.clicked.connect(self.on_save_pose)
        save_layout.addWidget(self.save_pose_btn)
        pose_layout.addLayout(save_layout)

        # Pose list
        self.pose_list = QtWidgets.QListWidget()
        self.pose_list.setMaximumHeight(120)
        pose_layout.addWidget(self.pose_list)

        # Pose actions
        pose_btn_layout = QtWidgets.QGridLayout()
        self.load_pose_btn = QtWidgets.QPushButton("Load")
        self.load_pose_btn.setToolTip("Apply the selected pose to the model")
        self.load_pose_btn.clicked.connect(self.on_load_pose)
        self.delete_pose_btn = QtWidgets.QPushButton("Delete")
        self.delete_pose_btn.setToolTip("Remove the selected pose from the library")
        self.delete_pose_btn.clicked.connect(self.on_delete_pose)
        self.export_poses_btn = QtWidgets.QPushButton("Export Library")
        self.export_poses_btn.setToolTip("Save all poses to a JSON file")
        self.export_poses_btn.clicked.connect(self.on_export_poses)
        self.import_poses_btn = QtWidgets.QPushButton("Import Library")
        self.import_poses_btn.setToolTip("Load poses from a JSON file")
        self.import_poses_btn.clicked.connect(self.on_import_poses)

        pose_btn_layout.addWidget(self.load_pose_btn, 0, 0)
        pose_btn_layout.addWidget(self.delete_pose_btn, 0, 1)
        pose_btn_layout.addWidget(self.export_poses_btn, 1, 0)
        pose_btn_layout.addWidget(self.import_poses_btn, 1, 1)
        pose_layout.addLayout(pose_btn_layout)

        # Pose interpolation
        interp_group = QtWidgets.QGroupBox("Pose Interpolation")
        interp_layout = QtWidgets.QVBoxLayout(interp_group)

        interp_slider_layout = QtWidgets.QFormLayout()
        self.interp_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.interp_slider.setMinimum(0)
        self.interp_slider.setMaximum(100)
        self.interp_slider.setValue(0)
        self.interp_slider.valueChanged.connect(self.on_interpolate_poses)
        self.interp_label = QtWidgets.QLabel("0%")
        interp_slider_layout.addRow("Blend:", self.interp_slider)
        interp_slider_layout.addRow("", self.interp_label)
        interp_layout.addLayout(interp_slider_layout)

        interp_note = QtWidgets.QLabel("Select two poses in library to interpolate")
        interp_note.setStyleSheet("font-style: italic; font-size: 9pt;")
        interp_layout.addWidget(interp_note)

        pose_layout.addWidget(interp_group)

        manip_layout.addWidget(pose_group)

        # IK settings
        ik_group = QtWidgets.QGroupBox("IK Solver Settings (Advanced)")
        ik_layout = QtWidgets.QFormLayout(ik_group)

        self.ik_damping_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ik_damping_slider.setMinimum(1)
        self.ik_damping_slider.setMaximum(100)
        self.ik_damping_slider.setValue(5)
        self.ik_damping_slider.setToolTip(
            "Adjust damping factor for IK solver to improve stability"
        )
        self.ik_damping_slider.setAccessibleName("IK Damping")
        self.ik_damping_slider.valueChanged.connect(self.on_ik_damping_changed)
        self.ik_damping_label = QtWidgets.QLabel("0.05")
        ik_layout.addRow("Damping:", self.ik_damping_slider)
        ik_layout.addRow("", self.ik_damping_label)

        self.ik_step_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.ik_step_slider.setMinimum(1)
        self.ik_step_slider.setMaximum(100)
        self.ik_step_slider.setValue(30)
        self.ik_step_slider.setToolTip("Adjust step size for IK solver convergence")
        self.ik_step_slider.setAccessibleName("IK Step Size")
        self.ik_step_slider.valueChanged.connect(self.on_ik_step_changed)
        self.ik_step_label = QtWidgets.QLabel("0.30")
        ik_layout.addRow("Step Size:", self.ik_step_slider)
        ik_layout.addRow("", self.ik_step_label)

        manip_layout.addWidget(ik_group)

        # Instructions
        instructions = QtWidgets.QLabel(
            "<b>Quick Start:</b><br>"
            "• Click and drag any body part to move it<br>"
            "• Scroll wheel to zoom camera<br>"
            "• Add constraints to fix bodies in space<br>"
            "• Save poses for later use"
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet(
            "padding: 10px; background-color: #e8f4f8; border-radius: 5px;"
        )
        manip_layout.addWidget(instructions)

        manip_layout.addStretch(1)

    def update_body_lists(self) -> None:
        """Update body selection combo boxes."""
        if self.sim_widget.model is None:
            return

        # Clear existing items
        self.constraint_body_combo.clear()
        self.ref_body_combo.clear()
        self.manip_body_combo.clear()

        # Add all bodies
        for body_id in range(1, self.sim_widget.model.nbody):  # Skip world (0)
            body_name = mujoco.mj_id2name(
                self.sim_widget.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name:
                item_text = f"{body_id}: {body_name}"
                self.constraint_body_combo.addItem(item_text)
                self.ref_body_combo.addItem(item_text)
                self.manip_body_combo.addItem(item_text)

    # -------- Callbacks --------

    def on_drag_enabled_changed(self, state: int) -> None:
        """Handle drag manipulation setting."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.enable_drag(enabled)

    def on_maintain_orientation_changed(self, state: int) -> None:
        """Handle maintain orientation setting."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.maintain_orientation = enabled

    def on_nullspace_posture_changed(self, state: int) -> None:
        """Handle nullspace posture optimization setting."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.use_nullspace_posture = enabled

    def on_manip_body_selected(self, index: int) -> None:
        """Handle body selection from combo box."""
        if index < 0:
            return

        text = self.manip_body_combo.itemText(index)
        try:
            body_id = int(text.split(":")[0])
            manipulator = self.sim_widget.get_manipulator()
            if manipulator:
                # Manually set selection
                manipulator.selected_body_id = body_id
                # Trigger update of manual transform values
                self.update_manual_transform_values()
                self.sim_widget._render_once()
        except ValueError:
            pass

    def on_manual_transform(self, type_: str, axis: int, value: float) -> None:
        """Handle manual transform changes."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator or manipulator.selected_body_id is None:
            return

        # We need to know if it's a mocap body or free joint
        body_id = manipulator.selected_body_id
        model = self.sim_widget.model
        data = self.sim_widget.data

        if model is None or data is None:
            return

        mocap_id = model.body_mocapid[body_id]

        if mocap_id != -1:
            # It's a mocap body
            if type_ == "pos":
                data.mocap_pos[mocap_id][axis] = value
            elif type_ == "rot":
                # Warn user about unimplemented rotation
                QtWidgets.QMessageBox.warning(
                    self,
                    "Not Implemented",
                    "Rotation transformation for manual sliders is not implemented.",
                )
        else:
            # Not a mocap body
            if self.main_window.statusBar():
                self.main_window.statusBar().showMessage(
                    "Manual transform only available for mocap bodies", 3000
                )

        self.sim_widget._render_once()

    def update_manual_transform_values(self) -> None:
        """Update sliders from current selection."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator or manipulator.selected_body_id is None:
            return

        body_id = manipulator.selected_body_id
        model = self.sim_widget.model
        data = self.sim_widget.data

        if model is None or data is None:
            return

        # Determine if the body is controlled by mocap
        mocap_id = model.body_mocapid[body_id]
        if mocap_id != -1:
            # Mocap body: get pose from mocap_pos
            pos = data.mocap_pos[mocap_id]
            self.trans_x.setValue(pos[0])
            self.trans_y.setValue(pos[1])
            self.trans_z.setValue(pos[2])

    def on_add_constraint(self) -> None:
        """Add a constraint to the selected body."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        body_text = self.constraint_body_combo.currentText()
        if not body_text:
            return

        body_id = int(body_text.split(":")[0])

        constraint_type_idx = self.constraint_type_combo.currentIndex()
        if constraint_type_idx == 0:
            # Fixed in space
            manipulator.add_constraint(body_id, ConstraintType.FIXED_IN_SPACE)
        else:
            # Relative to body
            ref_text = self.ref_body_combo.currentText()
            if not ref_text:
                QtWidgets.QMessageBox.warning(
                    self,
                    "No Reference Body",
                    "Please select a reference body.",
                )
                return

            ref_body_id = int(ref_text.split(":")[0])
            manipulator.add_constraint(
                body_id,
                ConstraintType.RELATIVE_TO_BODY,
                reference_body_id=ref_body_id,
            )

        self.update_constraints_list()

    def on_remove_constraint(self) -> None:
        """Remove constraint from selected body."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        body_text = self.constraint_body_combo.currentText()
        if not body_text:
            return

        body_id = int(body_text.split(":")[0])
        manipulator.remove_constraint(body_id)
        self.update_constraints_list()

    def on_clear_constraints(self) -> None:
        """Clear all constraints."""
        manipulator = self.sim_widget.get_manipulator()
        if manipulator:
            manipulator.clear_constraints()
            self.update_constraints_list()

    def update_constraints_list(self) -> None:
        """Update the list of active constraints."""
        self.constraints_list.clear()

        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        for body_id in manipulator.get_constrained_bodies():
            body_name = manipulator.get_body_name(body_id)
            self.constraints_list.addItem(f"{body_id}: {body_name}")

    def on_save_pose(self) -> None:
        """Save current pose to library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        pose_name = self.pose_name_input.text().strip()
        if not pose_name:
            QtWidgets.QMessageBox.warning(
                self,
                "No Name",
                "Please enter a name for the pose.",
            )
            return

        manipulator.save_pose(pose_name)
        self.update_pose_list()
        self.pose_name_input.clear()

        logger.info("Pose '%s' saved successfully", pose_name)
        if self.main_window.statusBar():
            self.main_window.statusBar().showMessage(
                f"Pose '{pose_name}' saved successfully.", 3000
            )

    def on_load_pose(self) -> None:
        """Load selected pose from library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        current_item = self.pose_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select a pose to load.",
            )
            return

        pose_name = current_item.text()
        if manipulator.load_pose(pose_name):
            logger.info("Pose '%s' loaded successfully", pose_name)
            if self.main_window.statusBar():
                self.main_window.statusBar().showMessage(
                    f"Pose '{pose_name}' loaded successfully.", 3000
                )
        else:
            QtWidgets.QMessageBox.warning(
                self,
                "Load Failed",
                f"Failed to load pose '{pose_name}'.",
            )

    def on_delete_pose(self) -> None:
        """Delete selected pose from library."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        current_item = self.pose_list.currentItem()
        if not current_item:
            QtWidgets.QMessageBox.warning(
                self,
                "No Selection",
                "Please select a pose to delete.",
            )
            return

        pose_name = current_item.text()
        reply = QtWidgets.QMessageBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete pose '{pose_name}'?",
            QtWidgets.QMessageBox.StandardButton.Yes
            | QtWidgets.QMessageBox.StandardButton.No,
        )

        if reply == QtWidgets.QMessageBox.StandardButton.Yes:
            if manipulator.delete_pose(pose_name):
                self.update_pose_list()
                logger.info("Pose '%s' deleted successfully", pose_name)
                if self.main_window.statusBar():
                    self.main_window.statusBar().showMessage(
                        f"Pose '{pose_name}' deleted successfully.", 3000
                    )

    def on_export_poses(self) -> None:
        """Export pose library to file."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        if len(manipulator.list_poses()) == 0:
            QtWidgets.QMessageBox.warning(self, "No Poses", "No poses to export.")
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Pose Library",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                manipulator.export_pose_library(filename)
                QtWidgets.QMessageBox.information(
                    self,
                    "Export Successful",
                    f"Pose library exported to {filename}",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Export Error",
                    f"Error exporting poses: {e!s}",
                )

    def on_import_poses(self) -> None:
        """Import pose library from file."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import Pose Library",
            "",
            "JSON Files (*.json)",
        )

        if filename:
            try:
                count = manipulator.import_pose_library(filename)
                self.update_pose_list()
                QtWidgets.QMessageBox.information(
                    self,
                    "Import Successful",
                    f"Imported {count} poses from {filename}",
                )
            except Exception as e:
                QtWidgets.QMessageBox.critical(
                    self,
                    "Import Error",
                    f"Error importing poses: {e!s}",
                )

    def update_pose_list(self) -> None:
        """Update the pose library list."""
        self.pose_list.clear()
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return
        for pose_name in manipulator.list_poses():
            self.pose_list.addItem(pose_name)

    def on_interpolate_poses(self, value: int) -> None:
        """Interpolate between two selected poses."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return

        alpha = value / 100.0
        self.interp_label.setText(f"{value}%")

        selected_items = self.pose_list.selectedItems()
        if len(selected_items) != 2:
            return

        pose_a = selected_items[0].text()
        pose_b = selected_items[1].text()
        manipulator.interpolate_poses(pose_a, pose_b, alpha)

    def on_ik_damping_changed(self, value: int) -> None:
        """Handle IK damping slider change."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return
        damping = value / 100.0
        manipulator.ik_damping = damping
        self.ik_damping_label.setText(f"{damping:.2f}")

    def on_ik_step_changed(self, value: int) -> None:
        """Handle IK step size slider change."""
        manipulator = self.sim_widget.get_manipulator()
        if not manipulator:
            return
        step_size = value / 100.0
        manipulator.ik_step_size = step_size
        self.ik_step_label.setText(f"{step_size:.2f}")
