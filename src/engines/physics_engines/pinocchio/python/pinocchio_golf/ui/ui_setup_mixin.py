"""UI Setup mixin for Pinocchio GUI."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from PyQt6 import QtWidgets

from src.shared.python.dashboard.widgets import LivePlotWidget
from src.shared.python.ui.widgets import LogPanel

if TYPE_CHECKING:
    from ..gui import PinocchioGUI

logger = logging.getLogger(__name__)


class UISetupMixin:
    """Mixin containing UI construction methods."""

    def _setup_ui(self: PinocchioGUI) -> None:
        """Build the PyQt Interface."""
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

    def _setup_toolbar(self: PinocchioGUI, layout: QtWidgets.QVBoxLayout) -> None:
        """Build the top bar with model selector, load button, and mode selector."""
        if not (layout is not None):
            raise ValueError("layout must be provided")
        if not (layout is not None):
            raise ValueError("layout must be provided")
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

    def _setup_simulation_tab(self: PinocchioGUI) -> None:
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
        self: PinocchioGUI, sim_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """Build the visualization group box."""
        if not (sim_layout is not None):
            raise ValueError("sim_layout must be provided")
        if not (sim_layout is not None):
            raise ValueError("sim_layout must be provided")
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
        self.chk_live_analysis.toggled.connect(self.on_live_analysis_toggled)
        vis_layout.addWidget(self.chk_live_analysis)

        vis_group.setLayout(vis_layout)
        sim_layout.addWidget(vis_group)

    def _setup_overlay_checkboxes(self, vis_layout: QtWidgets.QVBoxLayout) -> None:
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
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

    def _setup_ellipsoid_controls(self, vis_layout: QtWidgets.QVBoxLayout) -> None:
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        ellip_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Show Mobility Ellipsoid (Green)")
        self.chk_mobility.toggled.connect(self._update_viewer)
        ellip_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Show Force Ellipsoid (Red)")
        self.chk_force_ellip.toggled.connect(self._update_viewer)
        ellip_layout.addWidget(self.chk_force_ellip)
        vis_layout.addLayout(ellip_layout)

        self.manip_body_group = QtWidgets.QGroupBox("Manipulability Targets")
        self.manip_body_layout = QtWidgets.QGridLayout()
        self.manip_body_group.setLayout(self.manip_body_layout)
        vis_layout.addWidget(self.manip_body_group)

    def _setup_advanced_vectors(self, vis_layout: QtWidgets.QVBoxLayout) -> None:
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        vec_grid = QtWidgets.QGridLayout()

        self.chk_induced = QtWidgets.QCheckBox("Induced Vectors")
        self.chk_induced.toggled.connect(self._update_viewer)

        self.combo_induced = QtWidgets.QComboBox()
        self.combo_induced.setEditable(True)
        self.combo_induced.addItems(["gravity", "velocity", "total"])
        if line_edit := self.combo_induced.lineEdit():
            line_edit.editingFinished.connect(self._update_viewer)
        self.combo_induced.currentIndexChanged.connect(self._update_viewer)

        self.chk_cf = QtWidgets.QCheckBox("CF Vectors")
        self.chk_cf.toggled.connect(self._update_viewer)

        vec_grid.addWidget(self.chk_induced, 0, 0)
        vec_grid.addWidget(self.combo_induced, 0, 1)
        vec_grid.addWidget(self.chk_cf, 1, 0)

        vis_layout.addLayout(vec_grid)

    def _setup_vector_scales(self, vis_layout: QtWidgets.QVBoxLayout) -> None:
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        if not (vis_layout is not None):
            raise ValueError("vis_layout must be provided")
        scale_layout = QtWidgets.QHBoxLayout()
        scale_layout.addWidget(QtWidgets.QLabel("Force Scale:"))
        self.spn_f_scale = QtWidgets.QDoubleSpinBox()
        self.spn_f_scale.setRange(0.001, 100.0)
        self.spn_f_scale.setValue(1.0)
        self.spn_f_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spn_f_scale)

        scale_layout.addWidget(QtWidgets.QLabel("Torque Scale:"))
        self.spn_t_scale = QtWidgets.QDoubleSpinBox()
        self.spn_t_scale.setRange(0.001, 100.0)
        self.spn_t_scale.setValue(1.0)
        self.spn_t_scale.valueChanged.connect(self._update_viewer)
        scale_layout.addWidget(self.spn_t_scale)

        vis_layout.addLayout(scale_layout)

    def _setup_matrix_analysis_panel(self, sim_layout: QtWidgets.QVBoxLayout) -> None:
        if not (sim_layout is not None):
            raise ValueError("sim_layout must be provided")
        if not (sim_layout is not None):
            raise ValueError("sim_layout must be provided")
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        form_layout = QtWidgets.QFormLayout()
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        form_layout.addRow("Jacobian Cond:", self.lbl_cond)
        form_layout.addRow("Mass Matrix Rank:", self.lbl_rank)
        matrix_group.setLayout(form_layout)
        sim_layout.addWidget(matrix_group)

    def _setup_dynamic_tab(self: PinocchioGUI) -> None:
        dyn_page = QtWidgets.QWidget()
        dyn_layout = QtWidgets.QVBoxLayout(dyn_page)

        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.clicked.connect(self._toggle_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self.reset_simulation)
        btn_layout.addWidget(self.btn_reset)
        dyn_layout.addLayout(btn_layout)

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

    def _setup_kinematic_tab(self: PinocchioGUI) -> None:
        kin_page = QtWidgets.QWidget()
        kin_layout = QtWidgets.QVBoxLayout(kin_page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.slider_container = QtWidgets.QWidget()
        self.slider_layout = QtWidgets.QVBoxLayout(self.slider_container)
        scroll.setWidget(self.slider_container)

        kin_layout.addWidget(scroll)
        self.controls_stack.addWidget(kin_page)
