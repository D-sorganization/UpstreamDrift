"""Advanced professional GUI for golf swing analysis.

This module provides a comprehensive interface with:
- Simulation controls and visualization
- Real-time biomechanical analysis
- Advanced plotting and data export
- Force/torque vector visualization
- Camera controls
"""

from __future__ import annotations

import logging
import typing
from pathlib import Path

from PyQt6 import QtCore, QtGui, QtWidgets
from shared.python.dashboard.widgets import LivePlotWidget

from .advanced_gui_methods import AdvancedGuiMethodsMixin
from .grip_modelling_tab import GripModellingTab
from .gui.tabs.analysis_tab import AnalysisTab
from .gui.tabs.controls_tab import ControlsTab
from .gui.tabs.manipulability_tab import ManipulabilityTab
from .gui.tabs.manipulation_tab import ManipulationTab
from .gui.tabs.physics_tab import PhysicsTab
from .gui.tabs.plotting_tab import PlottingTab
from .gui.tabs.visualization_tab import VisualizationTab
from .sim_widget import MuJoCoSimWidget

logger = logging.getLogger(__name__)


class AdvancedGolfAnalysisWindow(QtWidgets.QMainWindow, AdvancedGuiMethodsMixin):
    """Professional golf swing analysis application with comprehensive features."""

    SIMPLIFIED_ACTUATOR_THRESHOLD: typing.Final[int] = 60

    def __init__(self) -> None:
        """Initialize the AdvancedGolfAnalysisWindow.

        Sets up the main window, models, and UI components including
        controls, visualization, analysis, and plotting tabs.
        """
        super().__init__()

        self.setWindowTitle("Golf Swing Biomechanical Analysis Suite")
        self.resize(1600, 900)

        # Apply Global Stylesheet
        # Apply Global Stylesheet
        self._load_stylesheet()

    def _load_stylesheet(self) -> None:
        """Load and apply the external QSS stylesheet."""
        try:
            style_path = Path(__file__).parent / "gui" / "styles" / "dark_theme.qss"
            if style_path.exists():
                with open(style_path) as f:
                    self.setStyleSheet(f.read())
            else:
                logger.warning(
                    "Stylesheet not found: %s; using default Qt styling", style_path
                )
        except Exception:
            logger.exception("Failed to load stylesheet, using default Qt styling")

        # Model configurations

        # Create central tab widget
        self.main_tab_widget = QtWidgets.QTabWidget()
        self.setCentralWidget(self.main_tab_widget)

        # --- Tab 1: Golf Swing Analysis ---
        self.golf_analysis_widget = QtWidgets.QWidget()
        self.main_tab_widget.addTab(self.golf_analysis_widget, "Golf Swing Analysis")

        main_layout = QtWidgets.QHBoxLayout(self.golf_analysis_widget)

        # Main horizontal splitter: simulation | controls/analysis
        main_splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)

        # Left: Simulation view
        self.sim_widget = MuJoCoSimWidget(width=900, height=700, fps=60)
        main_splitter.addWidget(self.sim_widget)

        # Right: Tabbed interface for controls and analysis
        self.tab_widget = QtWidgets.QTabWidget()
        self.tab_widget.setMinimumWidth(400)
        main_splitter.addWidget(self.tab_widget)

        # Set splitter proportions (70% simulation, 30% controls)
        main_splitter.setSizes([1100, 500])

        main_layout.addWidget(main_splitter)

        # --- Tab 2: Grip Modelling ---
        self.grip_modelling_tab = GripModellingTab()
        self.main_tab_widget.addTab(self.grip_modelling_tab, "Grip Modelling")

        # Create tabs
        # Physics Configuration Tab
        self.physics_tab = PhysicsTab(self.sim_widget, self)
        self.tab_widget.addTab(self.physics_tab, "Physics")

        # Simulation Controls Tab
        self.controls_tab = ControlsTab(self.sim_widget, self)
        self.tab_widget.addTab(self.controls_tab, "Controls")

        # Connect signals
        self.physics_tab.model_changed.connect(self.controls_tab.on_model_loaded)
        self.physics_tab.model_changed.connect(self.on_model_changed_signal)
        self.physics_tab.mode_changed.connect(self.controls_tab.on_mode_changed)

        # Connect live analysis toggle
        if hasattr(self.controls_tab, "chk_live_analysis"):
            self.controls_tab.chk_live_analysis.toggled.connect(
                self.on_live_analysis_toggled
            )

        # Visualization Tab
        self.visualization_tab = VisualizationTab(self.sim_widget, self)
        self.tab_widget.addTab(self.visualization_tab, "Visualization")
        self.analysis_tab = AnalysisTab(self.sim_widget, self)
        self.tab_widget.addTab(self.analysis_tab, "Analysis")
        self.plotting_tab = PlottingTab(self.sim_widget, self)
        self.tab_widget.addTab(self.plotting_tab, "Plotting")

        # Live Analysis Tab
        recorder = self.sim_widget.get_recorder()
        # Ensure recorder has engine reference for joint names
        recorder.engine = self.sim_widget.engine
        self.live_plot = LivePlotWidget(recorder)
        self.tab_widget.addTab(self.live_plot, "Live Analysis")

        self.manipulation_tab = ManipulationTab(self.sim_widget, self)
        self.tab_widget.addTab(self.manipulation_tab, "Interactive Pose")

        # Manipulability & Force Tab
        self.manipulability_tab = ManipulabilityTab(self.sim_widget, self)
        self.tab_widget.addTab(self.manipulability_tab, "Manipulability")

        # Connect model loaded signal to manipulability tab
        self.physics_tab.model_changed.connect(
            lambda n, c: self.manipulability_tab.on_model_loaded()
        )

        # Connect grip modelling tab to simulation widget
        self.grip_modelling_tab.connect_sim_widget(self.sim_widget)

        self.sim_widget.reset_state()

        # Apply professional styling
        self._apply_styling()

        # Auto-load configuration if present (overrides defaults if config found)
        self._load_launch_config()

        # Create status bar
        self._create_status_bar()

        # Start status bar update timer
        self.status_timer = QtCore.QTimer(self)
        self.status_timer.timeout.connect(self._update_status_bar)
        self.status_timer.start(200)  # Update every 200ms

        # Connect live plot update to simulation timer
        # This ensures the plot updates whenever the simulation steps/renders
        if hasattr(self.sim_widget, "timer"):
            self.sim_widget.timer.timeout.connect(self.live_plot.update_plot)

    @property
    def model_configs(self) -> list[dict]:
        """Expose model configs from PhysicsTab for mixin compatibility."""
        if hasattr(self, "physics_tab"):
            return self.physics_tab.model_configs
        return []

    @property
    def model_combo(self) -> QtWidgets.QComboBox | None:
        """Expose model combo from PhysicsTab for mixin compatibility."""
        if hasattr(self, "physics_tab"):
            return self.physics_tab.model_combo
        return None

    def on_model_changed_signal(self, model_name: str, config: dict) -> None:
        """Handle model change signal from PhysicsTab."""
        # Update body lists for interactive manipulation
        self.update_body_lists()

        # Update camera controls to match new model
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.update_camera_sliders()

        # Update plotting tab joints
        if hasattr(self, "plotting_tab"):
            self.plotting_tab.update_joint_list()

        # Update status bar immediately
        self._update_status_bar()

    def _on_quick_camera_clicked(self, preset_name: str) -> None:
        """Handle quick camera button click."""
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.camera_combo.setCurrentText(preset_name)

    def on_reset_camera(self) -> None:
        """Reset camera to default position."""
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.on_reset_camera()
        else:
            self.sim_widget.reset_camera()

    def keyPressEvent(self, event: QtGui.QKeyEvent | None) -> None:  # type: ignore[override]
        """Handle keyboard shortcuts."""
        if event is None:
            return
        key = event.key()

        # Camera preset shortcuts (1-5)
        camera_shortcuts = {
            QtCore.Qt.Key.Key_1: "side",
            QtCore.Qt.Key.Key_2: "front",
            QtCore.Qt.Key.Key_3: "top",
            QtCore.Qt.Key.Key_4: "follow",
            QtCore.Qt.Key.Key_5: "down-the-line",
        }

        if key in camera_shortcuts:
            # Enum key lookup issue with Int key
            preset = camera_shortcuts[key]  # type: ignore[index]
            self._on_quick_camera_clicked(preset)
            return

        # Space key: Play/Pause
        if key == QtCore.Qt.Key.Key_Space:
            if hasattr(self.controls_tab, "play_pause_btn"):
                self.controls_tab.play_pause_btn.toggle()
            return

        # R key: Reset
        if key == QtCore.Qt.Key.Key_R:
            if hasattr(self.controls_tab, "reset_btn"):
                self.controls_tab.reset_btn.click()
            return

        # H key: Toggle help panel
        if key == QtCore.Qt.Key.Key_H:
            # Find the help group and toggle it
            if hasattr(self.controls_tab, "help_group"):
                self.controls_tab.help_group.setChecked(
                    not self.controls_tab.help_group.isChecked(),
                )
            return

        super().keyPressEvent(event)

    def _create_status_bar(self) -> None:
        """Create a status bar showing simulation information."""
        status_bar = self.statusBar()
        if status_bar is None:
            return
        status_bar.setStyleSheet(
            """
            QStatusBar {
                background-color: #2c3e50;
                color: white;
                font-weight: bold;
            }
            QStatusBar::item {
                border: none;
            }
        """,
        )

        # Model info label
        self.status_model_label = QtWidgets.QLabel("Model: --")
        self.status_model_label.setStyleSheet("color: #3498db; padding: 0 10px;")
        status_bar.addWidget(self.status_model_label)

        # Separator
        sep1 = QtWidgets.QLabel("|")
        sep1.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep1)

        # Time label
        self.status_time_label = QtWidgets.QLabel("Time: 0.00s")
        self.status_time_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        status_bar.addWidget(self.status_time_label)

        # Separator
        sep2 = QtWidgets.QLabel("|")
        sep2.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep2)

        # Camera info label
        self.status_camera_label = QtWidgets.QLabel("Camera: side")
        self.status_camera_label.setStyleSheet("color: #9b59b6; padding: 0 10px;")
        status_bar.addWidget(self.status_camera_label)

        # Separator
        sep3 = QtWidgets.QLabel("|")
        sep3.setStyleSheet("color: #7f8c8d;")
        status_bar.addWidget(sep3)

        # Simulation state label
        self.status_state_label = QtWidgets.QLabel("Running")
        self.status_state_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        status_bar.addWidget(self.status_state_label)

        # Permanent widget for recording status (right side)
        self.status_recording_label = QtWidgets.QLabel("")
        self.status_recording_label.setStyleSheet("color: #e74c3c; padding: 0 10px;")
        status_bar.addPermanentWidget(self.status_recording_label)

    def _update_status_bar(self) -> None:
        """Update status bar with current simulation info."""
        if self.sim_widget.model is None:
            return

        # Update model info
        if hasattr(self, "physics_tab") and hasattr(self.physics_tab, "model_configs"):
            config_idx = self.physics_tab.model_combo.currentIndex()
            if config_idx < len(self.physics_tab.model_configs):
                model_name = self.physics_tab.model_configs[config_idx]["name"]
                num_actuators = self.sim_widget.model.nu
                self.status_model_label.setText(
                    f"Model: {model_name} ({num_actuators} actuators)",
                )

        # Update time
        if self.sim_widget.data is not None:
            time = self.sim_widget.data.time
            self.status_time_label.setText(f"Time: {time:.2f}s")

        # Update camera info
        if self.sim_widget.camera is not None:
            az = self.sim_widget.camera.azimuth
            el = self.sim_widget.camera.elevation
            dist = self.sim_widget.camera.distance
            self.status_camera_label.setText(
                f"Camera: Az={az:.0f}° El={el:.0f}° D={dist:.1f}",
            )

        # Update simulation state
        if self.sim_widget.running:
            self.status_state_label.setText("Running")
            self.status_state_label.setStyleSheet("color: #2ecc71; padding: 0 10px;")
        else:
            self.status_state_label.setText("Paused")
            self.status_state_label.setStyleSheet("color: #f39c12; padding: 0 10px;")

        # Update recording status
        recorder = self.sim_widget.get_recorder()
        if recorder.is_recording:
            frames = recorder.get_num_frames()
            duration = recorder.get_duration()
            self.status_recording_label.setText(
                f"RECORDING: {frames} frames ({duration:.1f}s)",
            )
            self.status_recording_label.setStyleSheet(
                "color: #e74c3c; font-weight: bold; padding: 0 10px;",
            )
        else:
            frames = recorder.get_num_frames()
            if frames > 0:
                self.status_recording_label.setText(f"Recorded: {frames} frames")
                self.status_recording_label.setStyleSheet(
                    "color: #f39c12; padding: 0 10px;",
                )
            else:
                self.status_recording_label.setText("")

    def _apply_styling(self) -> None:
        """Apply professional styling to the application."""
        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #f0f0f0;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #cccccc;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QPushButton {
                padding: 5px 10px;
                border-radius: 3px;
                background-color: #1f77b4;
                color: white;
            }
            QPushButton:hover {
                background-color: #1a5f8f;
            }
            QPushButton:pressed {
                background-color: #144a6e;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: white;
            }
            QTabBar::tab {
                padding: 8px 16px;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background-color: #1f77b4;
                color: white;
                font-weight: bold;
            }
```
        """,
        )

    def on_live_analysis_toggled(self, checked: bool) -> None:
        """Handle live analysis toggle."""
        self.sim_widget.enable_live_analysis = checked
        status_bar = self.statusBar()
        if checked:
            if status_bar:
                status_bar.showMessage(
                    "Live Biomechanical Analysis Enabled (Performance may drop)", 3000
                )
        else:
            if status_bar:
                status_bar.showMessage("Live Biomechanical Analysis Disabled", 3000)

    # -------- Model management --------

    def _update_camera_sliders(self) -> None:
        """Update camera control sliders to match current camera state."""
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.update_camera_sliders()

    # -------- Interactive manipulation event handlers --------

    def update_body_lists(self) -> None:
        """Update body selection combo boxes."""
        if hasattr(self, "manipulation_tab"):
            self.manipulation_tab.update_body_lists()
        if hasattr(self, "visualization_tab"):
            self.visualization_tab.update_body_list()
