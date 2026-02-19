"""UISetupMixin -- Tab and widget setup methods for HumanoidLauncher."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.dashboard.widgets import LivePlotWidget
from src.shared.python.theme.style_constants import Styles

logger = logging.getLogger(__name__)


class UISetupMixin:
    """Tab and widget setup methods for HumanoidLauncher."""

    def setup_ui(self) -> None:
        """Build the main window layout with tabs and log area."""
        central_widget = QWidget()

        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)

        main_layout.setContentsMargins(20, 20, 20, 20)

        main_layout.setSpacing(20)

        # Header

        header_label = QLabel("Humanoid Golf Simulation")

        header_label.setStyleSheet(Styles.HEADER_TITLE_LARGE)

        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(header_label)

        subtitle_label = QLabel("Advanced biomechanical golf swing analysis")

        subtitle_label.setStyleSheet(Styles.HEADER_SUBTITLE)

        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        main_layout.addWidget(subtitle_label)

        # Tabs

        self.tabs = QTabWidget()

        self.tabs.setStyleSheet("""

            QTabWidget::pane { border: 1px solid #444; background: #2b2b2b; }

            QTabBar::tab { background: #333; color: #ccc; padding: 10px 20px; }

            QTabBar::tab:selected { background: #0078d4; color: white; }

            QTabBar::tab:hover { background: #444; }

        """)

        self.setup_sim_tab()

        self.setup_appearance_tab()

        self.setup_equip_tab()

        self.setup_live_analysis_tab()

        main_layout.addWidget(self.tabs)

        # Footer / Log

        self.setup_log_area(main_layout)

    def setup_sim_tab(self) -> None:
        """Build the simulation settings and control tab."""
        tab = QWidget()

        layout = QVBoxLayout(tab)

        layout.setSpacing(15)

        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(self._setup_sim_settings_group())

        layout.addWidget(self._setup_sim_state_group())

        layout.addLayout(self._setup_sim_action_buttons())

        layout.addLayout(self._setup_sim_results_buttons())

        layout.addStretch()

        self.tabs.addTab(tab, "Simulation")

    def _setup_sim_settings_group(self) -> QGroupBox:
        """Create the simulation settings group with control mode and live view."""

        settings_group = QGroupBox("Simulation Settings")

        settings_group.setStyleSheet(Styles.GROUPBOX_DARK)

        settings_layout = QGridLayout()

        settings_layout.setSpacing(10)

        # Signal Generator Buttons (only shown for poly mode)

        self.btn_poly_generator = QPushButton("📊 Polynomial Generator")

        self.btn_poly_generator.setStyleSheet(Styles.BTN_POLY_GENERATOR)

        self.btn_poly_generator.clicked.connect(self.open_polynomial_generator)

        self.btn_signal_toolkit = QPushButton("🔧 Signal Toolkit")

        self.btn_signal_toolkit.setStyleSheet(Styles.BTN_SIGNAL_TOOLKIT)

        self.btn_signal_toolkit.clicked.connect(self.open_signal_toolkit)

        # Control Mode

        settings_layout.addWidget(QLabel("Control Mode:"), 0, 0)

        self.combo_control = QComboBox()

        self.combo_control.addItems(["pd", "lqr", "poly"])

        # Help text for control mode

        self.mode_help_label = QLabel()

        self.mode_help_label.setStyleSheet(Styles.TEXT_HELP_HINT)

        self.mode_help_label.setWordWrap(True)

        # Connect to updated help text method

        self.combo_control.currentTextChanged.connect(self.on_control_mode_changed)

        self.combo_control.setCurrentText(
            str(getattr(self.config, "control_mode", "pd"))
        )

        settings_layout.addWidget(self.combo_control, 0, 1)

        settings_layout.addWidget(self.btn_poly_generator, 0, 2)

        settings_layout.addWidget(self.btn_signal_toolkit, 0, 3)

        settings_layout.addWidget(self.mode_help_label, 1, 1, 1, 3)

        # Trigger initial update after button exists

        self.on_control_mode_changed(self.combo_control.currentText())

        # Live View

        self.chk_live = QCheckBox("Live Interactive View (requires X11/VcXsrv)")

        self.chk_live.setChecked(self.config.live_view)

        settings_layout.addWidget(self.chk_live, 1, 0, 1, 3)

        settings_group.setLayout(settings_layout)

        return settings_group

    def _setup_sim_state_group(self) -> QGroupBox:
        """Create the state management group with load/save path controls."""

        from PyQt6.QtWidgets import QLineEdit

        state_group = QGroupBox("State Management")

        state_group.setStyleSheet(Styles.GROUPBOX_DARK)

        state_layout = QGridLayout()

        # Load Path

        state_layout.addWidget(QLabel("Load State:"), 0, 0)

        self.txt_load_path = QLineEdit(self.config.load_state_path)

        state_layout.addWidget(self.txt_load_path, 0, 1)

        btn_browse_load = QPushButton("Browse")

        btn_browse_load.clicked.connect(lambda: self.browse_file(self.txt_load_path))

        state_layout.addWidget(btn_browse_load, 0, 2)

        # Save Path

        state_layout.addWidget(QLabel("Save State:"), 1, 0)

        self.txt_save_path = QLineEdit(self.config.save_state_path)

        state_layout.addWidget(self.txt_save_path, 1, 1)

        btn_browse_save = QPushButton("Browse")

        btn_browse_save.clicked.connect(
            lambda: self.browse_file(self.txt_save_path, save=True)
        )

        state_layout.addWidget(btn_browse_save, 1, 2)

        state_group.setLayout(state_layout)

        return state_group

    def _setup_sim_action_buttons(self) -> QHBoxLayout:
        """Create the run/stop/rebuild action button row."""

        btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("RUN SIMULATION")

        self.btn_run.setStyleSheet(Styles.BTN_DOCKER_RUN)

        self.btn_run.clicked.connect(self.start_simulation)

        self.btn_stop = QPushButton("STOP")

        self.btn_stop.setStyleSheet(Styles.BTN_DOCKER_STOP)

        self.btn_stop.setEnabled(False)

        self.btn_stop.clicked.connect(self.stop_simulation)

        self.btn_rebuild = QPushButton("UPDATE ENV")

        self.btn_rebuild.setStyleSheet(Styles.BTN_DOCKER_REBUILD)

        self.btn_rebuild.clicked.connect(self.rebuild_docker)

        btn_layout.addWidget(self.btn_run)

        btn_layout.addWidget(self.btn_stop)

        btn_layout.addWidget(self.btn_rebuild)

        return btn_layout

    def _setup_sim_results_buttons(self) -> QHBoxLayout:
        """Create the results button row (video, data, IAA plot)."""

        results_layout = QHBoxLayout()

        results_layout.addWidget(QLabel("Results:"))

        self.btn_video = QPushButton("Open Video")

        self.btn_video.setEnabled(False)

        self.btn_video.clicked.connect(self.open_video)

        self.btn_data = QPushButton("Open Data (CSV)")

        self.btn_data.setEnabled(False)

        self.btn_data.clicked.connect(self.open_data)

        self.btn_plot_iaa = QPushButton("Plot IAA")

        self.btn_plot_iaa.setEnabled(False)

        self.btn_plot_iaa.clicked.connect(self.plot_induced_acceleration)

        self.btn_plot_iaa.setToolTip("Plot Induced Acceleration Analysis")

        results_layout.addWidget(self.btn_video)

        results_layout.addWidget(self.btn_data)

        results_layout.addWidget(self.btn_plot_iaa)

        results_layout.addStretch()

        return results_layout

    def setup_live_analysis_tab(self) -> None:
        """Setup the live analysis tab with plot widget."""

        tab = QWidget()

        layout = QVBoxLayout(tab)

        layout.setContentsMargins(10, 10, 10, 10)

        self.live_plot = LivePlotWidget(self.recorder)

        layout.addWidget(self.live_plot)

        self.tabs.addTab(tab, "Live Analysis")

    def enable_results(self, enabled: bool) -> None:
        """Enable or disable the result viewing buttons."""
        from src.shared.python.engine_core.engine_availability import (
            MATPLOTLIB_AVAILABLE,
        )

        self.btn_video.setEnabled(enabled)

        self.btn_data.setEnabled(enabled)

        self.btn_plot_iaa.setEnabled(enabled and MATPLOTLIB_AVAILABLE)

    def _build_dimensions_group(self) -> QGroupBox:
        """Build the physical dimensions group box with height and weight controls.

        Returns:
            Configured QGroupBox widget.
        """
        dim_group = QGroupBox("📏 Physical Dimensions")
        dim_layout = QGridLayout()

        dim_layout.addWidget(QLabel("Height (m):"), 0, 0)
        self.spin_height = QDoubleSpinBox()
        self.spin_height.setRange(0.5, 3.0)
        self.spin_height.setSingleStep(0.05)
        self.spin_height.setValue(self.config.height_m)
        dim_layout.addWidget(self.spin_height, 0, 1)

        dim_layout.addWidget(QLabel("Weight (%):"), 1, 0)
        self.slider_weight = QSlider(Qt.Orientation.Horizontal)
        self.slider_weight.setRange(50, 200)
        self.slider_weight.setValue(int(self.config.weight_percent))
        self.lbl_weight_val = QLabel(f"{self.slider_weight.value()}%")
        self.slider_weight.valueChanged.connect(
            lambda v: self.lbl_weight_val.setText(f"{v}%")
        )
        dim_layout.addWidget(self.slider_weight, 1, 1)
        dim_layout.addWidget(self.lbl_weight_val, 1, 2)

        dim_group.setLayout(dim_layout)
        return dim_group

    def _build_colors_group(self) -> QGroupBox:
        """Build the body colors group box with color picker buttons.

        Returns:
            Configured QGroupBox widget.
        """
        color_group = QGroupBox("🎨 Body Colors")
        self.color_layout = QGridLayout()
        self.color_buttons: dict[str, QPushButton] = {}

        parts = [
            ("Shirt", "shirt"),
            ("Pants", "pants"),
            ("Shoes", "shoes"),
            ("Skin", "skin"),
            ("Club", "club"),
        ]

        for i, (name, key) in enumerate(parts):
            self.color_layout.addWidget(QLabel(name), i, 0)
            btn = QPushButton()
            btn.setFixedSize(50, 25)
            rgba = self.config.colors.get(key, [1, 1, 1, 1])
            self.set_btn_color(btn, rgba)
            btn.clicked.connect(lambda checked, k=key, b=btn: self.pick_color(k, b))
            self.color_layout.addWidget(btn, i, 1)
            self.color_buttons[key] = btn

        color_group.setLayout(self.color_layout)
        return color_group

    def setup_appearance_tab(self) -> None:
        """Build the humanoid appearance customization tab."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        layout.addWidget(self._build_dimensions_group())
        layout.addWidget(self._build_colors_group())

        btn_save = QPushButton("💾 Save Appearance Settings")
        btn_save.setStyleSheet(Styles.BTN_SAVE)
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        layout.addStretch()
        self.tabs.addTab(tab, "Appearance")

    def setup_equip_tab(self) -> None:
        """Build the equipment configuration tab."""
        tab = QWidget()

        layout = QVBoxLayout(tab)

        layout.setContentsMargins(20, 20, 20, 20)

        # Club Params

        club_group = QGroupBox("Golf Club Parameters")

        club_layout = QGridLayout()

        club_layout.addWidget(QLabel("Club Length (m):"), 0, 0)

        self.slider_length = QSlider(Qt.Orientation.Horizontal)

        self.slider_length.setRange(50, 150)  # 0.5 to 1.5 * 100

        self.slider_length.setValue(int(self.config.club_length * 100))

        self.lbl_length_val = QLabel(f"{self.slider_length.value() / 100:.2f} m")

        self.slider_length.valueChanged.connect(
            lambda v: self.lbl_length_val.setText(f"{v / 100:.2f} m")
        )

        club_layout.addWidget(self.slider_length, 0, 1)

        club_layout.addWidget(self.lbl_length_val, 0, 2)

        club_layout.addWidget(QLabel("Club Mass (kg):"), 1, 0)

        self.slider_mass = QSlider(Qt.Orientation.Horizontal)

        self.slider_mass.setRange(10, 200)  # 0.1 to 2.0 * 100

        self.slider_mass.setValue(int(self.config.club_mass * 100))

        self.lbl_mass_val = QLabel(f"{float(self.slider_mass.value()) / 100:.2f} kg")

        self.slider_mass.valueChanged.connect(
            lambda v: self.lbl_mass_val.setText(f"{float(v) / 100:.2f} kg")
        )

        club_layout.addWidget(self.slider_mass, 1, 1)

        club_layout.addWidget(self.lbl_mass_val, 1, 2)

        club_group.setLayout(club_layout)

        layout.addWidget(club_group)

        # Advanced Features

        feat_group = QGroupBox("Advanced Model Features")

        feat_layout = QVBoxLayout()

        self.chk_two_hand = QCheckBox("Two-Handed Grip (Constrained)")

        self.chk_two_hand.setChecked(self.config.two_handed)

        feat_layout.addWidget(self.chk_two_hand)

        self.chk_face = QCheckBox("Enhanced Face (Nose, Mouth)")

        self.chk_face.setChecked(self.config.enhance_face)

        feat_layout.addWidget(self.chk_face)

        self.chk_fingers = QCheckBox("Articulated Fingers (Segments)")

        self.chk_fingers.setChecked(self.config.articulated_fingers)

        feat_layout.addWidget(self.chk_fingers)

        feat_group.setLayout(feat_layout)

        layout.addWidget(feat_group)

        # Save Button

        btn_save = QPushButton("Save Equipment Settings")

        btn_save.setStyleSheet(Styles.BTN_SAVE)

        btn_save.clicked.connect(self.save_config)

        layout.addWidget(btn_save)

        layout.addStretch()

        self.tabs.addTab(tab, "Equipment")

    def setup_log_area(self, parent_layout: QVBoxLayout) -> None:
        """Build the simulation log output area."""
        log_group = QGroupBox("Simulation Log")

        log_layout = QVBoxLayout()

        header_layout = QHBoxLayout()

        header_layout.addWidget(QLabel("Real-time simulation output:"))

        btn_clear = QPushButton("Clear Log")

        btn_clear.clicked.connect(self.clear_log)

        header_layout.addWidget(btn_clear)

        header_layout.addStretch()

        log_layout.addLayout(header_layout)

        self.txt_log = QTextEdit()

        self.txt_log.setReadOnly(True)

        self.txt_log.setStyleSheet(Styles.CONSOLE_LOG_DARK)

        log_layout.addWidget(self.txt_log)

        log_group.setLayout(log_layout)

        parent_layout.addWidget(log_group)
