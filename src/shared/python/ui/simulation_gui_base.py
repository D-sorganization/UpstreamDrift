"""Base class for physics engine simulation GUIs.

This module provides SimulationGUIBase, an abstract base class that
extracts common functionality shared across MuJoCo, Drake, and Pinocchio
engine GUIs, including:

- Tab-based layout management
- Simulation controls (play/pause/reset)
- Mode switching (dynamic/kinematic)
- Recording controls with frame counting
- Data export infrastructure
- Visualization toggles (forces, torques, ellipsoids)
- Matrix analysis display (Jacobian condition, mass matrix rank)
- Status bar management

Engine-specific subclasses override abstract methods for their unique
simulation stepping, visualization, and model loading behaviors.

Phase 5 of the decoupling plan (docs/plans/decoupling-plan.md).

Usage:
    from src.shared.python.ui.simulation_gui_base import SimulationGUIBase

    class MyEngineGUI(SimulationGUIBase):
        def _step_simulation(self) -> None:
            ...
        def _reset_simulation_state(self) -> None:
            ...
        # etc.
"""

from __future__ import annotations

from abc import abstractmethod

from PyQt6 import QtCore, QtWidgets

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class SimulationGUIBase(QtWidgets.QMainWindow):
    """Abstract base class for physics engine simulation GUIs.

    Provides common tab management, simulation controls, recording,
    visualization toggles, and export infrastructure. Subclasses must
    implement engine-specific abstract methods.

    Attributes:
        operating_mode: Current mode, either "dynamic" or "kinematic".
        is_running: Whether the simulation is actively running.
        sim_time: Current simulation time in seconds.
    """

    # -- Class-level configuration (override in subclasses) ---------------

    WINDOW_TITLE: str = "Simulation GUI"
    WINDOW_WIDTH: int = 1000
    WINDOW_HEIGHT: int = 800
    TIMER_INTERVAL_MS: int = 10  # ~100 Hz default

    # Styles for run/stop button
    STYLE_BUTTON_RUN: str = "QPushButton { background-color: #4CAF50; color: white; }"
    STYLE_BUTTON_STOP: str = "QPushButton { background-color: #f44336; color: white; }"

    def __init__(self) -> None:
        """Initialize the base simulation GUI.

        Sets up the main window, common state variables, and builds the
        shared UI framework. Subclasses should call ``super().__init__()``
        then perform engine-specific initialization.

        Raises:
            TypeError: If instantiated directly (abstract base class).
        """
        if type(self) is SimulationGUIBase:
            msg = (
                "Cannot instantiate SimulationGUIBase directly. "
                "Subclass it and implement all abstract methods."
            )
            raise TypeError(msg)
        super().__init__()
        self.setWindowTitle(self.WINDOW_TITLE)
        self.resize(self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

        # -- Common simulation state ------------------------------------
        self.operating_mode: str = "dynamic"
        self.is_running: bool = False
        self.sim_time: float = 0.0

        # -- Timer (subclass starts it after engine init) ---------------
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._game_loop)

        # -- Build shared UI skeleton -----------------------------------
        self._build_base_ui()

    # ==================================================================
    # UI construction
    # ==================================================================

    def _build_base_ui(self) -> None:
        """Build the shared UI skeleton.

        Creates:
        - Central widget with a top-level tab widget
        - Simulation control tab (dynamic + kinematic stacked pages)
        - Visualization group with common toggles
        - Matrix analysis panel
        - Status bar
        """
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        # -- Top bar: model selector + mode combo ----------------------
        top_layout = QtWidgets.QHBoxLayout()

        self.model_combo = QtWidgets.QComboBox()
        self.model_combo.setMinimumWidth(200)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        top_layout.addWidget(QtWidgets.QLabel("Model:"))
        top_layout.addWidget(self.model_combo)

        top_layout.addStretch()

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["Dynamic (Physics)", "Kinematic (Pose)"])
        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        top_layout.addWidget(QtWidgets.QLabel("Mode:"))
        top_layout.addWidget(self.mode_combo)

        root_layout.addLayout(top_layout)

        # -- Main tab widget -------------------------------------------
        self.main_tab_widget = QtWidgets.QTabWidget()
        root_layout.addWidget(self.main_tab_widget)

        # Tab 1: Simulation Control
        sim_tab = QtWidgets.QWidget()
        sim_layout = QtWidgets.QVBoxLayout(sim_tab)

        # Stacked widget for dynamic / kinematic controls
        self.controls_stack = QtWidgets.QStackedWidget()
        sim_layout.addWidget(self.controls_stack)

        self._build_dynamic_controls_page()
        self._build_kinematic_controls_page()

        # Visualization group
        self._build_visualization_group(sim_layout)

        # Matrix analysis
        self._build_matrix_analysis_group(sim_layout)

        self.main_tab_widget.addTab(sim_tab, "Simulation")

        # Status bar
        self._build_status_bar()

    def _build_dynamic_controls_page(self) -> None:
        """Build the dynamic (physics) controls page."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        # Run / Pause button
        btn_layout = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run Simulation")
        self.btn_run.setCheckable(True)
        self.btn_run.setStyleSheet(self.STYLE_BUTTON_RUN)
        self.btn_run.clicked.connect(self._toggle_run)
        btn_layout.addWidget(self.btn_run)

        self.btn_reset = QtWidgets.QPushButton("Reset")
        self.btn_reset.clicked.connect(self._on_reset_clicked)
        btn_layout.addWidget(self.btn_reset)
        layout.addLayout(btn_layout)

        # Recording controls
        rec_group = QtWidgets.QGroupBox("Recording")
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
        rec_group.setLayout(rec_layout)
        layout.addWidget(rec_group)

        # Export button
        self.btn_export = QtWidgets.QPushButton("Export Data")
        self.btn_export.clicked.connect(self._on_export_clicked)
        layout.addWidget(self.btn_export)

        layout.addStretch()
        self.controls_stack.addWidget(page)

    def _build_kinematic_controls_page(self) -> None:
        """Build the kinematic (pose) controls page with a scroll area."""
        page = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(page)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        self.kinematic_content = QtWidgets.QWidget()
        self.kinematic_layout = QtWidgets.QVBoxLayout(self.kinematic_content)
        scroll.setWidget(self.kinematic_content)
        layout.addWidget(scroll)

        self.controls_stack.addWidget(page)

    def _build_visualization_group(self, parent_layout: QtWidgets.QVBoxLayout) -> None:
        """Build the common visualization toggles group."""
        vis_group = QtWidgets.QGroupBox("Visualization")
        vis_layout = QtWidgets.QVBoxLayout()

        # Checkboxes row
        chk_layout = QtWidgets.QHBoxLayout()
        self.chk_show_forces = QtWidgets.QCheckBox("Show Forces")
        self.chk_show_forces.toggled.connect(self._on_visualization_changed)
        chk_layout.addWidget(self.chk_show_forces)

        self.chk_show_torques = QtWidgets.QCheckBox("Show Torques")
        self.chk_show_torques.toggled.connect(self._on_visualization_changed)
        chk_layout.addWidget(self.chk_show_torques)
        vis_layout.addLayout(chk_layout)

        # Ellipsoids
        ellip_layout = QtWidgets.QHBoxLayout()
        self.chk_mobility = QtWidgets.QCheckBox("Mobility Ellipsoid")
        self.chk_mobility.toggled.connect(self._on_visualization_changed)
        ellip_layout.addWidget(self.chk_mobility)

        self.chk_force_ellip = QtWidgets.QCheckBox("Force Ellipsoid")
        self.chk_force_ellip.toggled.connect(self._on_visualization_changed)
        ellip_layout.addWidget(self.chk_force_ellip)
        vis_layout.addLayout(ellip_layout)

        # Live analysis toggle
        self.chk_live_analysis = QtWidgets.QCheckBox("Live Analysis (Induced/CF)")
        self.chk_live_analysis.setToolTip(
            "Compute Induced Accelerations and Counterfactuals in real-time "
            "(may slow down simulation)"
        )
        self.chk_live_analysis.toggled.connect(self._on_live_analysis_toggled)
        vis_layout.addWidget(self.chk_live_analysis)

        vis_group.setLayout(vis_layout)
        parent_layout.addWidget(vis_group)

    def _build_matrix_analysis_group(
        self, parent_layout: QtWidgets.QVBoxLayout
    ) -> None:
        """Build the matrix analysis info panel."""
        matrix_group = QtWidgets.QGroupBox("Matrix Analysis")
        matrix_layout = QtWidgets.QFormLayout(matrix_group)
        self.lbl_cond = QtWidgets.QLabel("--")
        self.lbl_rank = QtWidgets.QLabel("--")
        matrix_layout.addRow("Jacobian Cond:", self.lbl_cond)
        matrix_layout.addRow("Mass Matrix Rank:", self.lbl_rank)
        parent_layout.addWidget(matrix_group)

    def _build_status_bar(self) -> None:
        """Create a basic status bar."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage("Ready")

    # ==================================================================
    # Mode and model management
    # ==================================================================

    def _on_mode_changed(self, text: str) -> None:
        """Handle mode combo change between Dynamic and Kinematic."""
        if "Kinematic" in text:
            self.operating_mode = "kinematic"
            self.controls_stack.setCurrentIndex(1)
            self.is_running = False
            self.btn_run.setChecked(False)
            self.btn_run.setStyleSheet(self.STYLE_BUTTON_RUN)
            self.btn_run.setText("Run Simulation")
            self._update_status("Mode: Kinematic Control")
            self.sync_kinematic_controls()
        else:
            self.operating_mode = "dynamic"
            self.controls_stack.setCurrentIndex(0)
            self._update_status("Mode: Dynamic Simulation")
            if self.is_running:
                self.btn_run.setText("Pause Simulation")
                self.btn_run.setChecked(True)
                self.btn_run.setStyleSheet(self.STYLE_BUTTON_STOP)
            else:
                self.btn_run.setText("Run Simulation")
                self.btn_run.setChecked(False)
                self.btn_run.setStyleSheet(self.STYLE_BUTTON_RUN)

    def _on_model_changed(self, index: int) -> None:
        """Handle model selection change.

        Subclasses should override ``load_model`` to perform
        engine-specific model loading.
        """
        self.load_model(index)

    # ==================================================================
    # Simulation controls
    # ==================================================================

    def _toggle_run(self, checked: bool) -> None:
        """Toggle simulation running state."""
        self.is_running = checked
        if checked:
            self.btn_run.setText("Pause Simulation")
            self.btn_run.setStyleSheet(self.STYLE_BUTTON_STOP)
            self._update_status("Simulation Running...")
        else:
            self.btn_run.setText("Run Simulation")
            self.btn_run.setStyleSheet(self.STYLE_BUTTON_RUN)
            self._update_status("Simulation Paused")

    def _on_reset_clicked(self) -> None:
        """Handle reset button click."""
        self.is_running = False
        self.btn_run.setChecked(False)
        self.btn_run.setText("Run Simulation")
        self.btn_run.setStyleSheet(self.STYLE_BUTTON_RUN)
        self.sim_time = 0.0
        self._update_status("Simulation Reset")
        self.reset_simulation()
        self._reset_recording_ui()

    def _reset_recording_ui(self) -> None:
        """Reset recording UI elements to initial state."""
        self.lbl_rec_status.setText("Frames: 0")
        if self.btn_record.isChecked():
            self.btn_record.setChecked(False)
            self.btn_record.setText("Record")

    # ==================================================================
    # Recording
    # ==================================================================

    def _toggle_recording(self) -> None:
        """Toggle recording state."""
        if self.btn_record.isChecked():
            self.start_recording()
            self.btn_record.setText("Stop Recording")
            self._update_status("Recording started...")
        else:
            self.stop_recording()
            self.btn_record.setText("Record")
            self._update_status(
                f"Recording stopped. Frames: {self.get_recording_frame_count()}"
            )

    def update_recording_label(self) -> None:
        """Update the recording frame count label.

        Call this from the game loop during recording.
        """
        self.lbl_rec_status.setText(f"Frames: {self.get_recording_frame_count()}")

    # ==================================================================
    # Game loop
    # ==================================================================

    def _game_loop(self) -> None:
        """Main timer-driven loop dispatching to subclass logic.

        Calls ``step_simulation`` when in dynamic mode and running,
        then always calls ``update_visualization``.
        """
        # Always update live plot widget if present
        if hasattr(self, "live_plot"):
            self.live_plot.update_plot()

        if self.operating_mode == "dynamic" and self.is_running:
            self.step_simulation()

        self.update_visualization()

    # ==================================================================
    # Visualization change handler
    # ==================================================================

    def _on_visualization_changed(self) -> None:
        """Handle any visualization toggle change."""
        self.update_visualization()

    def _on_live_analysis_toggled(self, checked: bool) -> None:
        """Handle live analysis toggle."""
        if checked:
            self._update_status("Live Analysis Enabled")
        else:
            self._update_status("Live Analysis Disabled")

    # ==================================================================
    # Export
    # ==================================================================

    def _on_export_clicked(self) -> None:
        """Handle export button click."""
        frame_count = self.get_recording_frame_count()
        if frame_count == 0:
            QtWidgets.QMessageBox.warning(
                self, "No Data", "No recorded data to export."
            )
            return

        filename, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save Data", "", "All Files (*)"
        )
        if not filename:
            return

        try:
            self.export_data(filename)
            self._update_status(f"Data exported to {filename}")
        except (RuntimeError, ValueError, OSError) as exc:
            QtWidgets.QMessageBox.critical(self, "Export Error", str(exc))
            logger.exception("Export failed")

    # ==================================================================
    # Status helpers
    # ==================================================================

    def _update_status(self, message: str) -> None:
        """Update the status bar with a message."""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message)

    # ==================================================================
    # Abstract methods -- subclasses MUST implement
    # ==================================================================

    @abstractmethod
    def step_simulation(self) -> None:
        """Advance the simulation by one time step.

        Called by the game loop when in dynamic mode and running.
        Must handle physics integration and recording.
        """

    @abstractmethod
    def reset_simulation(self) -> None:
        """Reset the simulation to its initial state.

        Must reset physics state, recorder, and sync kinematic controls.
        """

    @abstractmethod
    def update_visualization(self) -> None:
        """Refresh all engine-specific visualizations.

        Called every game loop tick. Should handle 3D viewer updates,
        overlay rendering, force/torque vectors, and ellipsoids based
        on the current checkbox states.
        """

    @abstractmethod
    def load_model(self, index: int) -> None:
        """Load a model selected by the model combo at *index*.

        Should handle engine-specific model loading and rebuild
        kinematic controls.
        """

    @abstractmethod
    def sync_kinematic_controls(self) -> None:
        """Synchronize kinematic slider/spinbox values with model state.

        Called when switching to kinematic mode or after model load.
        """

    @abstractmethod
    def start_recording(self) -> None:
        """Start recording simulation data."""

    @abstractmethod
    def stop_recording(self) -> None:
        """Stop recording simulation data."""

    @abstractmethod
    def get_recording_frame_count(self) -> int:
        """Return the number of recorded frames."""

    @abstractmethod
    def export_data(self, filename: str) -> None:
        """Export recorded data to the given *filename*.

        May export in multiple formats (CSV, Parquet, etc.).
        """

    @abstractmethod
    def get_joint_names(self) -> list[str]:
        """Return joint names for plotting and analysis widgets."""

    # ==================================================================
    # Optional hooks (override for engine-specific behavior)
    # ==================================================================

    def on_model_loaded(self) -> None:
        """Hook called after a model has been successfully loaded.

        Override to perform post-load setup (e.g., populate
        manipulability checkboxes, update live plot joint names).
        """

    def on_live_analysis_toggled(self, checked: bool) -> None:
        """Hook called when the live analysis checkbox is toggled.

        Override to enable/disable real-time analysis computation.
        """
