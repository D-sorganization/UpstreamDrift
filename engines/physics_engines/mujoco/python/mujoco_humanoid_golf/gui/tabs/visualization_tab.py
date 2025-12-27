from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import mujoco
from PyQt6 import QtCore, QtGui, QtWidgets

if TYPE_CHECKING:
    from ...sim_widget import MuJoCoSimWidget

logger = logging.getLogger(__name__)


class VisualizationTab(QtWidgets.QWidget):
    """Tab for visualization settings and camera controls."""

    def __init__(
        self,
        sim_widget: MuJoCoSimWidget,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.sim_widget = sim_widget
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Create the visualization settings UI."""
        viz_layout = QtWidgets.QVBoxLayout(self)
        viz_layout.setContentsMargins(8, 8, 8, 8)

        # Camera controls
        camera_group = QtWidgets.QGroupBox("Camera View")
        camera_layout = QtWidgets.QVBoxLayout(camera_group)

        # Preset camera views
        preset_layout = QtWidgets.QHBoxLayout()
        preset_layout.addWidget(QtWidgets.QLabel("Preset:"))
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(["side", "front", "top", "follow", "down-the-line"])
        self.camera_combo.currentTextChanged.connect(self.on_camera_changed)
        preset_layout.addWidget(self.camera_combo)
        camera_layout.addLayout(preset_layout)

        # Reset camera button
        reset_cam_btn = QtWidgets.QPushButton("Reset Camera")
        reset_cam_btn.clicked.connect(self.on_reset_camera)
        camera_layout.addWidget(reset_cam_btn)

        # Advanced camera controls
        advanced_cam_group = QtWidgets.QGroupBox("Advanced Camera Controls")
        advanced_cam_layout = QtWidgets.QFormLayout(advanced_cam_group)

        # Azimuth (rotation around vertical axis)
        self.azimuth_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.azimuth_slider.setMinimum(0)
        self.azimuth_slider.setMaximum(360)
        self.azimuth_slider.setValue(90)
        self.azimuth_slider.setToolTip(
            "Rotate camera around the vertical axis (0-360\u00b0)"
        )
        self.azimuth_slider.setAccessibleName("Camera Azimuth")
        self.azimuth_slider.valueChanged.connect(self.on_azimuth_changed)
        self.azimuth_label = QtWidgets.QLabel("90\u00b0")
        advanced_cam_layout.addRow("Azimuth:", self.azimuth_slider)
        advanced_cam_layout.addRow("", self.azimuth_label)

        # Elevation (up/down angle)
        self.elevation_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.elevation_slider.setMinimum(-90)
        self.elevation_slider.setMaximum(90)
        self.elevation_slider.setValue(-20)
        self.elevation_slider.setToolTip(
            "Adjust camera vertical angle (-90\u00b0 to 90\u00b0)"
        )
        self.elevation_slider.setAccessibleName("Camera Elevation")
        self.elevation_slider.valueChanged.connect(self.on_elevation_changed)
        self.elevation_label = QtWidgets.QLabel("-20\u00b0")
        advanced_cam_layout.addRow("Elevation:", self.elevation_slider)
        advanced_cam_layout.addRow("", self.elevation_label)

        # Distance slider for zoom control
        self.distance_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.distance_slider.setMinimum(1)
        self.distance_slider.setMaximum(500)
        self.distance_slider.setValue(30)
        self.distance_slider.setToolTip("Zoom camera in/out")
        self.distance_slider.setAccessibleName("Camera Distance")
        self.distance_slider.valueChanged.connect(self.on_distance_changed)
        self.distance_label = QtWidgets.QLabel("3.0")
        advanced_cam_layout.addRow("Distance:", self.distance_slider)
        advanced_cam_layout.addRow("", self.distance_label)

        # Lookat position (X, Y, Z)
        lookat_layout = QtWidgets.QHBoxLayout()
        self.lookat_x_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_x_spin.setRange(-10.0, 10.0)
        self.lookat_x_spin.setSingleStep(0.1)
        self.lookat_x_spin.setValue(0.0)
        self.lookat_x_spin.setToolTip("Camera target X coordinate")
        self.lookat_x_spin.setAccessibleName("Lookat X")
        self.lookat_x_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("X:"))
        lookat_layout.addWidget(self.lookat_x_spin)

        self.lookat_y_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_y_spin.setRange(-10.0, 10.0)
        self.lookat_y_spin.setSingleStep(0.1)
        self.lookat_y_spin.setValue(0.0)
        self.lookat_y_spin.setToolTip("Camera target Y coordinate")
        self.lookat_y_spin.setAccessibleName("Lookat Y")
        self.lookat_y_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("Y:"))
        lookat_layout.addWidget(self.lookat_y_spin)

        self.lookat_z_spin = QtWidgets.QDoubleSpinBox()
        self.lookat_z_spin.setRange(-10.0, 10.0)
        self.lookat_z_spin.setSingleStep(0.1)
        self.lookat_z_spin.setValue(1.0)
        self.lookat_z_spin.setToolTip("Camera target Z coordinate")
        self.lookat_z_spin.setAccessibleName("Lookat Z")
        self.lookat_z_spin.valueChanged.connect(self.on_lookat_changed)
        lookat_layout.addWidget(QtWidgets.QLabel("Z:"))
        lookat_layout.addWidget(self.lookat_z_spin)

        advanced_cam_layout.addRow("Lookat:", lookat_layout)

        # Mouse controls info
        mouse_info = QtWidgets.QLabel(
            "Mouse Controls:\n"
            "\u2022 Left Drag: Rotate camera\n"
            "\u2022 Right/Ctrl+Left: Rotate camera\n"
            "\u2022 Middle/Shift+Left: Pan camera\n"
            "\u2022 Wheel: Zoom",
        )
        mouse_info.setWordWrap(True)
        # Style set in dark_theme.qss
        mouse_info.setObjectName("helpLabel")
        advanced_cam_layout.addRow("", mouse_info)

        camera_layout.addWidget(advanced_cam_group)
        viz_layout.addWidget(camera_group)

        # Background color controls
        bg_group = QtWidgets.QGroupBox("Background Color")
        bg_layout = QtWidgets.QVBoxLayout(bg_group)

        # Sky color
        sky_layout = QtWidgets.QHBoxLayout()
        sky_layout.addWidget(QtWidgets.QLabel("Sky Color:"))
        self.sky_color_btn = QtWidgets.QPushButton()
        self.sky_color_btn.setMinimumSize(60, 30)
        self.sky_color_btn.setStyleSheet("background-color: rgb(51, 77, 102);")
        self.sky_color_btn.setToolTip("Click to change sky color")
        self.sky_color_btn.setAccessibleName("Sky Color")
        self.sky_color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.sky_color_btn.clicked.connect(self.on_sky_color_clicked)
        sky_layout.addWidget(self.sky_color_btn)
        sky_layout.addStretch()
        bg_layout.addLayout(sky_layout)

        # Ground color
        ground_layout = QtWidgets.QHBoxLayout()
        ground_layout.addWidget(QtWidgets.QLabel("Ground Color:"))
        self.ground_color_btn = QtWidgets.QPushButton()
        self.ground_color_btn.setMinimumSize(60, 30)
        self.ground_color_btn.setStyleSheet("background-color: rgb(51, 51, 51);")
        self.ground_color_btn.setToolTip("Click to change ground color")
        self.ground_color_btn.setAccessibleName("Ground Color")
        self.ground_color_btn.setCursor(QtCore.Qt.CursorShape.PointingHandCursor)
        self.ground_color_btn.clicked.connect(self.on_ground_color_clicked)
        ground_layout.addWidget(self.ground_color_btn)
        ground_layout.addStretch()
        bg_layout.addLayout(ground_layout)

        # Reset to defaults button
        reset_bg_btn = QtWidgets.QPushButton("Reset to Defaults")
        reset_bg_btn.clicked.connect(self.on_reset_background)
        bg_layout.addWidget(reset_bg_btn)

        viz_layout.addWidget(bg_group)

        # Meshcat Visualization
        meshcat_group = QtWidgets.QGroupBox("Web Visualization (Meshcat)")
        meshcat_layout = QtWidgets.QVBoxLayout(meshcat_group)

        btn_meshcat = QtWidgets.QPushButton("Open Web Visualizer")
        btn_meshcat.clicked.connect(self.on_open_meshcat)
        meshcat_layout.addWidget(btn_meshcat)

        viz_layout.addWidget(meshcat_group)

        # Force/Torque visualization
        force_group = QtWidgets.QGroupBox("Force & Torque Visualization")
        force_layout = QtWidgets.QVBoxLayout(force_group)

        # Torque vectors
        self.show_torques_cb = QtWidgets.QCheckBox("Show Joint Torque Vectors")
        self.show_torques_cb.stateChanged.connect(self.on_show_torques_changed)
        force_layout.addWidget(self.show_torques_cb)

        torque_scale_layout = QtWidgets.QFormLayout()
        self.torque_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.torque_scale_slider.setMinimum(1)
        self.torque_scale_slider.setMaximum(100)
        self.torque_scale_slider.setValue(10)
        self.torque_scale_slider.setToolTip(
            "Adjust the visual length of torque vectors"
        )
        self.torque_scale_slider.setAccessibleName("Torque Scale")
        self.torque_scale_slider.valueChanged.connect(self.on_torque_scale_changed)
        self.torque_scale_label = QtWidgets.QLabel("1.0%")
        torque_scale_layout.addRow("Torque Scale:", self.torque_scale_slider)
        torque_scale_layout.addRow("", self.torque_scale_label)
        force_layout.addLayout(torque_scale_layout)

        # Force vectors
        self.show_forces_cb = QtWidgets.QCheckBox("Show Constraint Forces")
        self.show_forces_cb.stateChanged.connect(self.on_show_forces_changed)
        force_layout.addWidget(self.show_forces_cb)

        force_scale_layout = QtWidgets.QFormLayout()
        self.force_scale_slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        self.force_scale_slider.setMinimum(1)
        self.force_scale_slider.setMaximum(100)
        self.force_scale_slider.setValue(10)
        self.force_scale_slider.setToolTip("Adjust the visual length of force vectors")
        self.force_scale_slider.setAccessibleName("Force Scale")
        self.force_scale_slider.valueChanged.connect(self.on_force_scale_changed)
        self.force_scale_label = QtWidgets.QLabel("10%")
        force_scale_layout.addRow("Force Scale:", self.force_scale_slider)
        force_scale_layout.addRow("", self.force_scale_label)
        force_layout.addLayout(force_scale_layout)

        # Contact forces
        self.show_contacts_cb = QtWidgets.QCheckBox("Show Contact Forces")
        self.show_contacts_cb.stateChanged.connect(self.on_show_contacts_changed)
        force_layout.addWidget(self.show_contacts_cb)

        # Ellipsoids
        ellipsoid_group = QtWidgets.QGroupBox("Ellipsoids")
        ellipsoid_layout = QtWidgets.QVBoxLayout(ellipsoid_group)
        self.show_mobility_ellipsoid_cb = QtWidgets.QCheckBox(
            "Show Mobility Ellipsoid (Green)"
        )
        self.show_mobility_ellipsoid_cb.stateChanged.connect(
            self.on_ellipsoid_visualization_changed
        )
        ellipsoid_layout.addWidget(self.show_mobility_ellipsoid_cb)

        self.show_force_ellipsoid_cb = QtWidgets.QCheckBox("Show Force Ellipsoid (Red)")
        self.show_force_ellipsoid_cb.stateChanged.connect(
            self.on_ellipsoid_visualization_changed
        )
        ellipsoid_layout.addWidget(self.show_force_ellipsoid_cb)
        viz_layout.addWidget(ellipsoid_group)

        viz_layout.addWidget(force_group)

        # Matrix Analysis
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.jacobian_cond_label = QtWidgets.QLabel("Condition: --")
        self.constraint_rank_label = QtWidgets.QLabel("Rank: --")
        self.nefc_label = QtWidgets.QLabel("Constraints: --")

        matrix_layout.addRow("Jacobian Cond:", self.jacobian_cond_label)
        matrix_layout.addRow("Constraint Rank:", self.constraint_rank_label)
        matrix_layout.addRow("Active Constraints:", self.nefc_label)
        viz_layout.addWidget(matrix_group)

        # Body Appearance Controls
        appearance_group = QtWidgets.QGroupBox("Body Appearance")
        appearance_layout = QtWidgets.QVBoxLayout(appearance_group)

        # Body selector
        body_sel_layout = QtWidgets.QHBoxLayout()
        body_sel_layout.addWidget(QtWidgets.QLabel("Body:"))
        self.viz_body_combo = QtWidgets.QComboBox()
        self.viz_body_combo.setMinimumWidth(150)
        body_sel_layout.addWidget(self.viz_body_combo, stretch=1)
        appearance_layout.addLayout(body_sel_layout)

        # Color picker
        color_layout = QtWidgets.QHBoxLayout()
        self.viz_color_btn = QtWidgets.QPushButton("Change Color")
        self.viz_color_btn.clicked.connect(self.on_change_body_color)
        color_layout.addWidget(self.viz_color_btn)

        self.viz_reset_color_btn = QtWidgets.QPushButton("Reset Color")
        self.viz_reset_color_btn.clicked.connect(self.on_reset_body_color)
        color_layout.addWidget(self.viz_reset_color_btn)

        appearance_layout.addLayout(color_layout)
        viz_layout.addWidget(appearance_group)

        viz_layout.addStretch(1)

    # -------- Callbacks --------

    def on_camera_changed(self, camera_name: str) -> None:
        """Handle camera view change."""
        self.sim_widget.set_camera(camera_name)
        # Update sliders to match camera preset
        self.update_camera_sliders()

    def update_camera_sliders(self) -> None:
        """Update camera control sliders to match current camera state."""
        if self.sim_widget.camera is not None:
            # Update azimuth (0-360)
            az = self.sim_widget.camera.azimuth % 360
            self.azimuth_slider.setValue(int(az))
            self.azimuth_label.setText(f"{az:.1f}\u00b0")

            # Update elevation
            el = self.sim_widget.camera.elevation
            self.elevation_slider.setValue(int(el))
            self.elevation_label.setText(f"{el:.1f}\u00b0")

            # Update distance (convert to slider scale: 1-500 represents 0.1-50.0)
            dist = self.sim_widget.camera.distance
            slider_val = int((dist - 0.1) / (50.0 - 0.1) * 499) + 1
            self.distance_slider.setValue(slider_val)
            self.distance_label.setText(f"{dist:.2f}")

            # Update lookat
            lookat = self.sim_widget.camera.lookat
            self.lookat_x_spin.setValue(lookat[0])
            self.lookat_y_spin.setValue(lookat[1])
            self.lookat_z_spin.setValue(lookat[2])

    def on_azimuth_changed(self, value: int) -> None:
        """Handle azimuth slider change."""
        self.sim_widget.set_camera_azimuth(float(value))
        self.azimuth_label.setText(f"{value}\u00b0")

    def on_elevation_changed(self, value: int) -> None:
        """Handle elevation slider change."""
        self.sim_widget.set_camera_elevation(float(value))
        self.elevation_label.setText(f"{value}\u00b0")

    def on_distance_changed(self, value: int) -> None:
        """Handle distance slider change."""
        # Convert slider value (1-500) to distance (0.1-50.0)
        distance = 0.1 + (value - 1) / 499.0 * (50.0 - 0.1)
        self.sim_widget.set_camera_distance(distance)
        self.distance_label.setText(f"{distance:.2f}")

    def on_lookat_changed(self) -> None:
        """Handle lookat position change."""
        x = self.lookat_x_spin.value()
        y = self.lookat_y_spin.value()
        z = self.lookat_z_spin.value()
        self.sim_widget.set_camera_lookat(x, y, z)

    def on_reset_camera(self) -> None:
        """Reset camera to default position."""
        self.sim_widget.reset_camera()
        self.update_camera_sliders()

    def on_sky_color_clicked(self) -> None:
        """Handle sky color button click - open color picker."""
        current_color = QtGui.QColor(
            int(self.sim_widget.sky_color[0] * 255),
            int(self.sim_widget.sky_color[1] * 255),
            int(self.sim_widget.sky_color[2] * 255),
        )
        color = QtWidgets.QColorDialog.getColor(current_color, self, "Select Sky Color")
        if color.isValid():
            rgba = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0,
                1.0,
            ]
            self.sim_widget.set_background_color(sky_color=rgba)
            # Update button color
            self.sky_color_btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, "
                f"{color.blue()});",
            )

    def on_ground_color_clicked(self) -> None:
        """Handle ground color button click - open color picker."""
        current_color = QtGui.QColor(
            int(self.sim_widget.ground_color[0] * 255),
            int(self.sim_widget.ground_color[1] * 255),
            int(self.sim_widget.ground_color[2] * 255),
        )
        color = QtWidgets.QColorDialog.getColor(
            current_color,
            self,
            "Select Ground Color",
        )
        if color.isValid():
            rgba = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0,
                1.0,
            ]
            self.sim_widget.set_background_color(ground_color=rgba)
            # Update button color
            self.ground_color_btn.setStyleSheet(
                f"background-color: rgb({color.red()}, {color.green()}, "
                f"{color.blue()});",
            )

    def on_reset_background(self) -> None:
        """Reset background colors to defaults."""
        default_sky = [0.2, 0.3, 0.4, 1.0]
        default_ground = [0.2, 0.2, 0.2, 1.0]
        self.sim_widget.set_background_color(
            sky_color=default_sky,
            ground_color=default_ground,
        )
        # Update button colors
        self.sky_color_btn.setStyleSheet("background-color: rgb(51, 77, 102);")
        self.ground_color_btn.setStyleSheet("background-color: rgb(51, 51, 51);")

    def on_open_meshcat(self) -> None:
        """Open the Meshcat visualizer in the default browser."""
        if (
            hasattr(self.sim_widget, "meshcat_adapter")
            and self.sim_widget.meshcat_adapter
        ):
            self.sim_widget.meshcat_adapter.open_browser()
        else:
            QtWidgets.QMessageBox.warning(
                self, "Meshcat", "Meshcat adapter not initialized or not available."
            )

    def on_show_torques_changed(self, state: int) -> None:
        """Handle torque visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_torque_visualization(enabled)

    def on_torque_scale_changed(self, value: int) -> None:
        """Handle torque scale slider change."""
        scale = value / 100.0  # Convert to 0.01 - 1.0
        self.torque_scale_label.setText(f"{scale:.2f}%")
        self.sim_widget.set_torque_visualization(
            self.show_torques_cb.isChecked(),
            scale * 0.01,
        )

    def on_show_forces_changed(self, state: int) -> None:
        """Handle force visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_force_visualization(enabled)

    def on_force_scale_changed(self, value: int) -> None:
        """Handle force scale slider change."""
        scale = value / 10.0
        self.force_scale_label.setText(f"{scale:.1f}%")
        self.sim_widget.set_force_visualization(
            self.show_forces_cb.isChecked(),
            scale * 0.1,
        )

    def on_show_contacts_changed(self, state: int) -> None:
        """Handle contact force visualization toggle."""
        enabled = state == QtCore.Qt.CheckState.Checked.value
        self.sim_widget.set_contact_force_visualization(enabled)

    def on_ellipsoid_visualization_changed(self, state: int) -> None:
        """Handle ellipsoid visualization toggle."""
        show_mobility = self.show_mobility_ellipsoid_cb.isChecked()
        show_force = self.show_force_ellipsoid_cb.isChecked()
        self.sim_widget.set_ellipsoid_visualization(show_mobility, show_force)

    def on_change_body_color(self) -> None:
        """Handle body color change."""
        body_name = self.viz_body_combo.currentText()
        if not body_name:
            return

        color = QtWidgets.QColorDialog.getColor(
            QtCore.Qt.GlobalColor.white,
            self,
            f"Select Color for {body_name}",
        )
        if color.isValid():
            rgba = [
                color.red() / 255.0,
                color.green() / 255.0,
                color.blue() / 255.0,
                1.0,
            ]
            self.sim_widget.set_body_color(body_name, rgba)

    def on_reset_body_color(self) -> None:
        """Handle body color reset."""
        body_name = self.viz_body_combo.currentText()
        if not body_name:
            return

        self.sim_widget.reset_body_color(body_name)

    def update_matrix_metrics(
        self,
        cond: float | str,
        rank: int | str,
        nefc: int | str,
    ) -> None:
        """Update matrix analysis labels."""
        self.jacobian_cond_label.setText(f"Condition: {cond}")
        self.constraint_rank_label.setText(f"Rank: {rank}")
        self.nefc_label.setText(f"Constraints: {nefc}")

    def update_body_list(self) -> None:
        """Update body selection list."""
        if self.sim_widget.model is None:
            return

        self.viz_body_combo.clear()
        for body_id in range(1, self.sim_widget.model.nbody):
            body_name = mujoco.mj_id2name(
                self.sim_widget.model,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
            )
            if body_name:
                self.viz_body_combo.addItem(f"{body_id}: {body_name}")
