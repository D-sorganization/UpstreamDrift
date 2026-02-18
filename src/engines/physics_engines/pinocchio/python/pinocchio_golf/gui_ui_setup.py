"""Pinocchio GUI UI setup mixin.

Extracts all widget/layout construction, kinematic controls, slider/spinbox
wiring, and mode switching from PinocchioGUI (gui.py).
"""

from __future__ import annotations

from typing import Any

from PyQt6 import QtCore, QtWidgets

from src.shared.python.dashboard.widgets import LivePlotWidget
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.ui.widgets import LogPanel, SignalBlocker

logger = get_logger(__name__)

# Constants (duplicated from gui.py to avoid circular imports)
SLIDER_RANGE_RAD = 10.0
SLIDER_SCALE = 100.0

__all__ = ["UISetupMixin"]


class UISetupMixin:
    """Mixin providing all UI construction methods for PinocchioGUI.

    Provides:
    - ``_setup_ui``: Main UI construction entry point
    - ``_setup_toolbar``: Model selector and mode combo
    - ``_setup_simulation_tab``: Simulation tab with controls
    - ``_setup_visualization_panel``: Visualization group box
    - ``_setup_overlay_checkboxes``: Frame/COM/force/torque toggles
    - ``_setup_ellipsoid_controls``: Manipulability analysis controls
    - ``_setup_advanced_vectors``: Induced/counterfactual vector controls
    - ``_setup_vector_scales``: Force/torque scale spinboxes
    - ``_setup_matrix_analysis_panel``: Jacobian/mass matrix display
    - ``_setup_dynamic_tab``: Dynamic mode controls
    - ``_setup_kinematic_tab``: Kinematic mode slider area
    - ``_build_kinematic_controls``: Joint slider/spinbox creation
    - ``_add_joint_control_widget``: Single joint control widget
    - ``_sync_kinematic_controls``: Sync sliders with model state
    - ``_on_slider`` / ``_on_spin``: Slider and spinbox handlers
    - ``_update_q``: Update joint configuration
    - ``_on_mode_changed``: Mode switch handler
    - ``_populate_model_combo``: Populate model dropdown
    - ``_on_model_combo_changed``: Handle model selection
    """

    def _setup_ui(self: Any) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # 1. Top Bar: Load & Mode
        self._setup_toolbar(layout)

        # 2. Controls Stack (Main Tabs)
        self.main_tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.main_tabs)

        # Tab 1: Control & Simulation
        self._setup_simulation_tab()

        # Tab 2: Live Analysis (LivePlotWidget)
        if LivePlotWidget is not None:
            self.live_tab = QtWidgets.QWidget()
            live_layout = QtWidgets.QVBoxLayout(self.live_tab)
            self.live_plot = LivePlotWidget(self.recorder)
            live_layout.addWidget(self.live_plot)
            self.main_tabs.addTab(self.live_tab, "Live Analysis")

        # Tab 3: Post-Hoc Analysis & Plotting
        self._setup_analysis_tab()

    def _setup_toolbar(self: Any, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the top bar with model selector, load button, and mode selector."""
        top_layout = QtWidgets.QHBoxLayout()

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(200)
        self._populate_model_combo()
        self.model_combo.currentIndexChanged.connect(self._on_model_combo_changed)
        top_layout.addWidget(self.model_combo)

        self.load_btn = QtWidgets.QPushButton("Load File...")
        self.load_btn.clicked.connect(lambda: self.load_urdf())
        top_layout.addWidget(self.load_btn)

        top_layout.addStretch()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_layout.addWidget(QtWidgets.QLabel("Mode:"))
        top_layout.addWidget(self.mode_combo)

        layout.addLayout(top_layout)

    def _setup_simulation_tab(self: Any) -> None:
        """Build the simulation tab with controls and viz."""
        sim_tab = QtWidgets.QWidget()
        sim_layout = QtWidgets.QVBoxLayout(sim_tab)

        self.controls_stack = QtWidgets.QStackedWidget()
        sim_layout.addWidget(self.controls_stack)

        self._setup_dynamic_tab()
        self._setup_kinematic_tab()

        # Visualization panel
        self._setup_visualization_panel(sim_layout)

        # Matrix Analysis Panel
        self._setup_matrix_analysis_panel(sim_layout)

        self.log = LogPanel()
        sim_layout.addWidget(self.log)

        self.main_tabs.addTab(sim_tab, "Simulation")

    def _setup_visualization_panel(
        self: Any, sim_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """Build the visualization group box."""
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()

        # Checkboxes row
        self._setup_overlay_checkboxes(vis_layout)

        # Ellipsoids & Body Selection
        self._setup_ellipsoid_controls(vis_layout)

        # Advanced Vectors
        self._setup_advanced_vectors(vis_layout)

        # Vector Scales
        self._setup_vector_scales(vis_layout)

        # Live Analysis Toggle
        self.chk_live_analysis = QtWidgets.QCheckBox("Live Analysis (Induced/CF)")
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time "
            "(Can slow down sim)"
        )
        self.chk_live_analysis.toggled.connect(self._on_live_analysis_toggled)
        vis_layout.addWidget(self.chk_live_analysis)

        vis_group.setLayout(vis_layout)
        sim_layout.addWidget(vis_group)

    def _setup_overlay_checkboxes(self: Any, vis_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the frame/COM/force/torque overlay checkboxes."""
        chk_layout = QtWidgets.QHBoxLayout()
        self.chk_frames = QtWidgets.QCheckBox("Show Frames")
        self.chk_frames.toggled.connect(self._toggle_frames)
        chk_layout.addWidget(self.chk_frames)

        self.chk_coms = QtWidgets.QCheckBox("Show COMs")
        self.chk_coms.toggled.connect(self._toggle_coms)
        chk_layout.addWidget(self.chk_coms)

        self.chk_forces = QtWidgets.QCheckBox("Show Forces")
        self.chk_forces.toggled.connect(self._toggle_forces)
        chk_layout.addWidget(self.chk_forces)

        self.chk_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_torques.toggled.connect(self._toggle_torques)
        chk_layout.addWidget(self.chk_torques)
        vis_layout.addLayout(chk_layout)

    def _setup_ellipsoid_controls(self: Any, vis_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the manipulability ellipsoid toggles and body selection grid."""
        ellip_group = QtWidgets.QGroupBox("Manipulability Analysis")
        ellip_layout = QtWidgets.QVBoxLayout()

        # Toggles
        toggles_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Mobility (Green)")
        self.chk_mobility.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Force (Red)")
        self.chk_force_ellip.toggled.connect(self._update_viewer)
        toggles_layout.addWidget(self.chk_force_ellip)
        ellip_layout.addLayout(toggles_layout)

        # Body Grid
        self.manip_body_layout = QtWidgets.QGridLayout()
        body_container = QtWidgets.QWidget()
        body_container.setLayout(self.manip_body_layout)
        ellip_layout.addWidget(QtWidgets.QLabel("Points of Interest:"))
        ellip_layout.addWidget(body_container)

        ellip_group.setLayout(ellip_layout)
        vis_layout.addWidget(ellip_group)

    def _setup_advanced_vectors(self: Any, vis_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the induced acceleration and counterfactual vector controls."""
        adv_vec_layout = QtWidgets.QHBoxLayout()
        self.chk_induced = QtWidgets.QCheckBox("Induced Accel")
        self.chk_induced.toggled.connect(self._update_viewer)

        self.combo_induced = QtWidgets.QComboBox()
        self.combo_induced.setEditable(True)
        self.combo_induced.addItems(["gravity", "velocity", "total"])
        self.combo_induced.setToolTip(
            "Select source (e.g. gravity) or type "
            "specific torque vector in comma-sep form"
        )

        # Use lineEdit signal to avoid lag on keystrokes
        if line_edit := self.combo_induced.lineEdit():
            line_edit.editingFinished.connect(self._update_viewer)
        self.combo_induced.currentIndexChanged.connect(self._update_viewer)

        self.chk_cf = QtWidgets.QCheckBox("Counterfactuals")
        self.chk_cf.toggled.connect(self._update_viewer)

        self.combo_cf = QtWidgets.QComboBox()
        self.combo_cf.addItems(["ztcf_accel", "zvcf_torque"])
        self.combo_cf.currentTextChanged.connect(self._update_viewer)

        adv_vec_layout.addWidget(self.chk_induced)
        adv_vec_layout.addWidget(self.combo_induced)
        adv_vec_layout.addWidget(self.chk_cf)
        adv_vec_layout.addWidget(self.combo_cf)
        vis_layout.addLayout(adv_vec_layout)

    def _setup_vector_scales(self: Any, vis_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the force and torque scale spinboxes."""
        scale_layout = QtWidgets.QHBoxLayout()
        self.spin_force_scale = QtWidgets.QDoubleSpinBox()
        self.spin_force_scale.setRange(0.01, 10.0)
        self.spin_force_scale.setSingleStep(0.05)
        self.spin_force_scale.setValue(0.1)
        self.spin_force_scale.setPrefix("Scale: ")
        self.spin_force_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spin_force_scale)

        self.spin_torque_scale = QtWidgets.QDoubleSpinBox()
        self.spin_torque_scale.setRange(0.01, 10.0)
        self.spin_torque_scale.setSingleStep(0.05)
        self.spin_torque_scale.setValue(0.1)
        self.spin_torque_scale.setPrefix("T Scale: ")
        self.spin_torque_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spin_torque_scale)
        vis_layout.addLayout(scale_layout)

    def _setup_matrix_analysis_panel(
        self: Any, sim_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """Build the matrix analysis group box."""
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Mass Matrix Rank:", self.lbl_rank)
        sim_layout.addWidget(matrix_group)

    def _setup_dynamic_tab(self: Any) -> None:
        dyn_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dyn_page)

        # Run Controls
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self._toggle_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._reset_simulation)
        btn_layout.addWidget(self.btn_reset)
        dyn_layout.addLayout(btn_layout)

        # Recording Controls
        rec_layout = QtWidgets.QHBoxLayout()
        self.btn_record = QtWidgets.QPushButton("Record")
        self.btn_record.setCheckable(True)
        self.btn_record.setStyleSheet(
            "QPushButton:checked { background-color: #ffcccc; }"
        )
        self.btn_record.clicked.connect(self._toggle_recording)
        rec_layout.addWidget(self.btn_record)

        self.lbl_rec_status = QtWidgets.QLabel("Frames: 0")
        rec_layout.addWidget(self.lbl_rec_status)
        dyn_layout.addLayout(rec_layout)

        dyn_layout.addStretch()
        self.controls_stack.addWidget(dyn_page)

    def _setup_kinematic_tab(self: Any) -> None:
        kin_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kin_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        scroll.setWidget(self.slider_container)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kin_page)

    def _build_kinematic_controls(self: Any) -> None:
        if self.model is None:
            return

        # Clear layout
        while self.slider_layout.count():
            item = self.slider_layout.takeAt(0)
            if item is None:
                break
            widget = item.widget()
            if widget:
                widget.deleteLater()

        self.joint_sliders = []
        self.joint_spinboxes = []
        self.joint_names = []

        # Populate joint_names for joints 1..N (excluding Universe at index 0).
        self.joint_names = list(self.model.names)[1:]

        # Update joint selection combo for analysis
        self.joint_select_combo.clear()
        self.joint_select_combo.addItems(self.joint_names)

        # Iterate joints (skip universe)
        for i in range(1, self.model.njoints):
            self._add_joint_control_widget(i)

    def _add_joint_control_widget(self: Any, i: int) -> None:
        if self.model is None:
            return

        joint_name = self.model.names[i]
        # Simple assumption: 1 DOF per joint for sliders.
        nq_joint = self.model.joints[i].nq

        if nq_joint != 1:
            msg = (
                f"Skipping joint '{joint_name}' (index {i}): "
                f"{nq_joint} DOFs not supported in kinematic controls."
            )
            self.log_write(msg)
            return

        # joint_names is pre-populated above; widgets are only created for supported
        # 1-DOF joints
        row = QtWidgets.QWidget()
        r_layout = QtWidgets.QHBoxLayout(row)
        r_layout.setContentsMargins(0, 0, 0, 0)

        r_layout.addWidget(QtWidgets.QLabel(f"{joint_name}:"))

        slider = QtWidgets.QSlider(QtCore.Qt.Orientation.Horizontal)
        # Range +/- SLIDER_RANGE_RAD
        slider_min = int(-SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider_max = int(SLIDER_RANGE_RAD * SLIDER_SCALE)
        slider.setRange(slider_min, slider_max)
        slider.setValue(0)

        spin = QtWidgets.QDoubleSpinBox()
        spin.setRange(-SLIDER_RANGE_RAD, SLIDER_RANGE_RAD)
        spin.setSingleStep(0.1)

        # Connect
        idx_q = self.model.joints[i].idx_q
        idx = int(idx_q)  # Capture index into q vector

        slider.valueChanged.connect(
            lambda val, s=spin, k=idx: self._on_slider(val, s, k)
        )
        spin.valueChanged.connect(lambda val, s=slider, k=idx: self._on_spin(val, s, k))

        r_layout.addWidget(slider)
        r_layout.addWidget(spin)
        self.slider_layout.addWidget(row)

        self.joint_sliders.append(slider)
        self.joint_spinboxes.append(spin)

    def _sync_kinematic_controls(self: Any) -> None:
        """Synchronize sliders/spinboxes with current model state q."""
        if self.model is None or self.q is None:
            return

        slider_idx = 0
        for i in range(1, self.model.njoints):
            # Must match the filtering in _build_kinematic_controls
            if self.model.joints[i].nq != 1:
                continue

            idx_q = self.model.joints[i].idx_q
            val = self.q[idx_q]

            if slider_idx < len(self.joint_sliders):
                slider = self.joint_sliders[slider_idx]
                spin = self.joint_spinboxes[slider_idx]

                with SignalBlocker(slider, spin):
                    slider.setValue(int(val * SLIDER_SCALE))
                    spin.setValue(val)

                slider_idx += 1

    def _on_slider(
        self: Any, val: int, spin: QtWidgets.QDoubleSpinBox, idx: int
    ) -> None:
        angle = val / SLIDER_SCALE
        with SignalBlocker(spin):
            spin.setValue(angle)
        self._update_q(idx, angle)

    def _on_spin(self: Any, val: float, slider: QtWidgets.QSlider, idx: int) -> None:
        with SignalBlocker(slider):
            slider.setValue(int(val * SLIDER_SCALE))
        self._update_q(idx, val)

    def _update_q(self: Any, idx: int, val: float) -> None:
        if self.operating_mode != "kinematic":
            return
        if self.q is not None:
            self.q[idx] = val
            self._update_viewer()

    def _on_mode_changed(self: Any, mode_text: str) -> None:
        if "Dynamic" in mode_text:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
        else:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            # Stop simulation when entering kinematic mode
            self.is_running = False
            self.btn_run.setText("Run Simulation")
            self.btn_run.setChecked(False)
            self._sync_kinematic_controls()

    def _populate_model_combo(self: Any) -> None:
        """Populate the model dropdown."""
        self.model_combo.clear()
        for model in self.available_models:
            self.model_combo.addItem(model["name"])

    def _on_model_combo_changed(self: Any, index: int) -> None:
        """Handle model selection."""
        if index < 0 or index >= len(self.available_models):
            return

        path = self.available_models[index]["path"]
        if path:
            self.load_urdf(path)
