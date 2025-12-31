import csv
import datetime
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

# Third-party imports
try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QPalette
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# Ensure shared modules are importable
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent
# Add Project Root to sys.path so 'shared' package is discoverable
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from shared.python.configuration_manager import ConfigurationManager  # noqa: E402
from shared.python.process_worker import ProcessWorker  # noqa: E402

# Polynomial generator widget imported lazily to avoid MuJoCo DLL issues
# Import happens in open_polynomial_generator() only when needed


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Default Configuration
# Logger initialized above


class ModernDarkPalette(QPalette):
    """Custom Dark Palette for a modern look."""

    def __init__(self):
        super().__init__()
        self.setColor(QPalette.ColorRole.Window, QColor(43, 43, 43))
        self.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Base, QColor(30, 30, 30))
        self.setColor(QPalette.ColorRole.AlternateBase, QColor(43, 43, 43))
        self.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.Button, QColor(53, 53, 53))
        self.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        self.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        self.setColor(QPalette.ColorRole.Link, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.Highlight, QColor(42, 130, 218))
        self.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.black)


class HumanoidLauncher(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Humanoid Golf Simulation Suite")
        self.setMinimumSize(1000, 800)

        # Apply Theme
        QApplication.setStyle("Fusion")
        self.setPalette(ModernDarkPalette())

        # Paths
        self.current_dir = Path(__file__).parent.resolve()
        # engines/physics_engines/mujoco/python -> engines/physics_engines/mujoco
        self.repo_path = self.current_dir.parent
        # Save config in docker/src where the humanoid_golf simulation expects it
        self.config_path = self.repo_path / "docker" / "src" / "simulation_config.json"

        # State
        self.config_manager = ConfigurationManager(self.config_path)
        self.config = self.config_manager.load()

        self.simulation_thread = None

        # Load Config - Already done above
        # self.load_config()

        # UI Setup
        self.setup_ui()

    def setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        # Header
        header_label = QLabel("Humanoid Golf Simulation")
        header_label.setStyleSheet(
            "font-size: 24px; font-weight: bold; color: #ffffff;"
        )
        header_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header_label)

        subtitle_label = QLabel("Advanced biomechanical golf swing analysis")
        subtitle_label.setStyleSheet("font-size: 14px; color: #cccccc;")
        subtitle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(subtitle_label)

        # Tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(
            """
            QTabWidget::pane { border: 1px solid #444; background: #2b2b2b; }
            QTabBar::tab { background: #333; color: #ccc; padding: 10px 20px; }
            QTabBar::tab:selected { background: #0078d4; color: white; }
            QTabBar::tab:hover { background: #444; }
        """
        )

        self.setup_sim_tab()
        self.setup_appearance_tab()
        self.setup_equip_tab()

        main_layout.addWidget(self.tabs)

        # Footer / Log
        self.setup_log_area(main_layout)

    def setup_sim_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Simulation Settings Group
        settings_group = QGroupBox("Simulation Settings")
        settings_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "margin-top: 10px; }\n"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 5px; }"
        )
        settings_layout = QGridLayout()
        settings_layout.setSpacing(10)

        # Polynomial Generator Button (only shown for poly mode)
        self.btn_poly_generator = QPushButton("📊 Configure Polynomial")
        self.btn_poly_generator.setStyleSheet(
            "background-color: #0078d4; color: white; padding: 8px; font-weight: bold;"
        )
        self.btn_poly_generator.clicked.connect(self.open_polynomial_generator)
        # Helper for control mode help updates
        # Check initial state later

        # Control Mode
        settings_layout.addWidget(QLabel("Control Mode:"), 0, 0)
        self.combo_control = QComboBox()
        self.combo_control.addItems(["pd", "lqr", "poly"])

        # Help text for control mode
        self.mode_help_label = QLabel()
        self.mode_help_label.setStyleSheet(
            "color: #aaa; font-style: italic; font-size: 11px;"
        )
        self.mode_help_label.setWordWrap(True)

        # Connect to updated help text method
        self.combo_control.currentTextChanged.connect(self.on_control_mode_changed)
        self.combo_control.setCurrentText(
            str(getattr(self.config, "control_mode", "pd"))
        )
        settings_layout.addWidget(self.combo_control, 0, 1)
        settings_layout.addWidget(self.btn_poly_generator, 0, 2)
        settings_layout.addWidget(self.mode_help_label, 1, 1, 1, 2)

        # Trigger initial update after button exists
        self.on_control_mode_changed(self.combo_control.currentText())

        # Live View
        self.chk_live = QCheckBox("Live Interactive View (requires X11/VcXsrv)")
        self.chk_live.setChecked(self.config.live_view)
        settings_layout.addWidget(self.chk_live, 1, 0, 1, 3)

        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)

        # State Management Group
        state_group = QGroupBox("State Management")
        state_group.setStyleSheet(
            "QGroupBox { font-weight: bold; border: 1px solid #555; "
            "margin-top: 10px; }\n"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; "
            "padding: 0 5px; }"
        )
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
        layout.addWidget(state_group)

        # Action Buttons
        btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("RUN SIMULATION")
        self.btn_run.setStyleSheet(
            "background-color: #107c10; color: white;"
            "padding: 12px; font-weight: bold; font-size: 14px;"
        )
        self.btn_run.clicked.connect(self.start_simulation)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet(
            "background-color: #d13438; color: white;"
            "padding: 12px; font-weight: bold; font-size: 14px;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.stop_simulation)

        self.btn_rebuild = QPushButton("UPDATE ENV")
        self.btn_rebuild.setStyleSheet(
            "background-color: #8b5cf6; color: white;padding: 12px; font-weight: bold;"
        )
        self.btn_rebuild.clicked.connect(self.rebuild_docker)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_rebuild)

        layout.addLayout(btn_layout)

        # Results Buttons
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

        layout.addLayout(results_layout)
        layout.addStretch()

        self.tabs.addTab(tab, "Simulation")

    def enable_results(self, enabled: bool) -> None:
        self.btn_video.setEnabled(enabled)
        self.btn_data.setEnabled(enabled)
        self.btn_plot_iaa.setEnabled(enabled and HAS_MATPLOTLIB)

    def setup_appearance_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(20, 20, 20, 20)

        # Dimensions
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
        layout.addWidget(dim_group)

        # Colors
        color_group = QGroupBox("🎨 Body Colors")
        self.color_layout = QGridLayout()
        self.color_buttons = {}

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
        layout.addWidget(color_group)

        # Save Button
        btn_save = QPushButton("💾 Save Appearance Settings")
        btn_save.setStyleSheet(
            "background-color: #107c10; color: white; padding: 10px;"
        )
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        layout.addStretch()
        self.tabs.addTab(tab, "Appearance")

    def setup_equip_tab(self):
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
        btn_save.setStyleSheet(
            "background-color: #107c10; color: white; padding: 10px;"
        )
        btn_save.clicked.connect(self.save_config)
        layout.addWidget(btn_save)

        layout.addStretch()
        self.tabs.addTab(tab, "Equipment")

    def setup_log_area(self, parent_layout):
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
        self.txt_log.setStyleSheet(
            "background-color: #1e1e1e; color: #ddd;"
            "font-family: Consolas; font-size: 10pt;"
        )
        log_layout.addWidget(self.txt_log)

        log_group.setLayout(log_layout)
        parent_layout.addWidget(log_group)

    def log(self, msg):
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")
        self.txt_log.ensureCursorVisible()

    def clear_log(self):
        self.txt_log.clear()
        self.log("Log cleared.")

    def set_btn_color(self, btn, rgba):
        from collections.abc import Sequence

        rgba_seq: Sequence[float] = rgba  # type: ignore[assignment]
        r, g, b = (int(c * 255) for c in rgba_seq[:3])
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
        )

    def pick_color(self, key, btn):
        current = self.config.colors.get(key, [1.0, 1.0, 1.0, 1.0])
        initial = QColor(
            int(current[0] * 255),
            int(current[1] * 255),
            int(current[2] * 255),
        )

        color = QColorDialog.getColor(initial, self, f"Choose {key} Color")
        if color.isValid():
            new_rgba = [color.redF(), color.greenF(), color.blueF(), 1.0]
            self.config.colors[key] = new_rgba
            self.set_btn_color(btn, new_rgba)
            self.save_config()

    def on_control_mode_changed(self, mode: str) -> None:
        """Update the help text and enable/disable polynomial generator button
        based on the selected control mode."""
        descriptions = {
            "pd": "Proportional-Derivative control (Target Pose tracking).",
            "lqr": "Linear Quadratic Regulator (Optimal control).",
            "poly": "Polynomial trajectory tracking (Time-varying torque).",
        }
        self.mode_help_label.setText(descriptions.get(mode, ""))
        self.btn_poly_generator.setEnabled(mode == "poly")

    def open_polynomial_generator(self):
        """Open polynomial generator dialog."""
        # Lazy import to avoid MuJoCo DLL initialization on Windows
        try:
            # We import directly from the file to avoid importing the package
            # 'mujoco_humanoid_golf', which triggers 'import mujoco' in its __init__.
            # This allows the polynomial generator (which uses only matplotlib/numpy)
            # to work even if MuJoCo DLLs are missing or incompatible locally.
            import importlib.util

            target_file = (
                self.current_dir / "mujoco_humanoid_golf" / "polynomial_generator.py"
            )
            if not target_file.exists():
                raise FileNotFoundError(f"File not found: {target_file}")

            # Use a stable module name to allow caching and avoid memory leaks
            module_name = "polynomial_generator_widget"

            if module_name in sys.modules:
                module = sys.modules[module_name]
            else:
                spec = importlib.util.spec_from_file_location(module_name, target_file)
                if spec is None or spec.loader is None:
                    raise ImportError(f"Could not load spec from {target_file}")

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)

            PolynomialGeneratorWidget = module.PolynomialGeneratorWidget
        except ImportError as e:
            QMessageBox.warning(
                self,
                "Polynomial Generator Unavailable",
                f"The polynomial generator widget is not available.\n\nError: {e}\n\n"
                "Please ensure mujoco_humanoid_golf/polynomial_generator.py exists.",
            )
            return
        except Exception as e:
            QMessageBox.warning(
                self,
                "Loading Error",
                f"Failed to load generator widget.\n\nError: {e}",
            )
            return

        # Create dialog
        dialog = QDialog(self)
        dialog.setWindowTitle("Polynomial Function Generator")
        dialog.setMinimumSize(900, 700)

        layout = QVBoxLayout(dialog)

        # Add polynomial generator widget
        poly_widget = PolynomialGeneratorWidget(dialog)

        # Set available joints (humanoid joint names)
        joints = [
            "lowerbackrx",
            "upperbackrx",
            "rtibiarx",
            "ltibiarx",
            "rfemurrx",
            "lfemurrx",
            "rfootrx",
            "lfootrx",
            "rhumerusrx",
            "lhumerusrx",
            "rhumerusrz",
            "lhumerusrz",
            "rhumerusry",
            "lhumerusry",
            "rradiusrx",
            "lradiusrx",
        ]
        poly_widget.set_joints(joints)

        # Connect signal to save coefficients
        def on_polynomial_generated(joint_name, coefficients):
            """Save generated polynomial coefficients to config."""
            self.config.polynomial_coefficients[joint_name] = coefficients
            self.save_config()
            self.log(f"Polynomial generated for {joint_name}: {coefficients}")

        poly_widget.polynomial_generated.connect(on_polynomial_generated)

        layout.addWidget(poly_widget)

        # Add close button
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.exec()

    def browse_file(self, line_edit, save=False):
        if save:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save State", "", "JSON State (*.json)"
            )
        else:
            path, _ = QFileDialog.getOpenFileName(
                self, "Load State", "", "JSON State (*.json)"
            )

        if path:
            line_edit.setText(path)

    def load_config(self):
        """Deprecated: Config is loaded in __init__. Kept for compatibility."""
        pass

    def save_config(self):
        """Save current configuration to file."""
        try:
            # Update config object from UI
            self.config.height_m = self.spin_height.value()
            self.config.weight_percent = self.slider_weight.value()
            self.config.club_length = self.slider_length.value() / 100.0
            self.config.club_mass = self.slider_mass.value() / 100.0

            # Features
            self.config.two_handed = self.chk_two_hand.isChecked()
            self.config.enhance_face = self.chk_face.isChecked()
            self.config.articulated_fingers = self.chk_fingers.isChecked()

            # Paths
            self.config.save_state_path = self.txt_save_path.text()
            self.config.load_state_path = self.txt_load_path.text()

            # Live View
            self.config.live_view = self.chk_live.isChecked()

            # Control Mode
            text = self.combo_control.currentText()
            if "PD" in text:
                self.config.control_mode = "pd"
            elif "LQR" in text:
                self.config.control_mode = "lqr"
            elif "Polynomial" in text:
                self.config.control_mode = "poly"

            self.config_manager.save(self.config)
            self.log(f"Config saved to {self.config_path}")
        except Exception as e:
            self.log(f"Error saving config: {e}")

    def get_simulation_command(self) -> tuple[list[str], dict[str, str] | None]:
        """Construct the command to run the simulation.

        Returns:
            Tuple of (command_list, environment_dict).
        """
        # Check if running inside Docker container
        in_docker = Path("/.dockerenv").exists()
        env = None

        if in_docker:
            # Running inside the Mujoco Docker container:
            # - CWD set to /workspace/engines/physics_engines/mujoco/python
            # - Module exists at ../docker/src/humanoid_golf/sim.py relative to here
            cmd = ["python", "-m", "humanoid_golf.sim"]

            # Ensure PYTHONPATH includes the directory containing 'humanoid_golf'
            # We add '../docker/src' to PYTHONPATH.
            # Note: We must also include existing PYTHONPATH if any.
            # Using os.environ logic in ProcessWorker handles the merge,
            # we just provide the override/addition here.
            env = {"PYTHONPATH": "../docker/src"}
            return cmd, env

        # HOST-SIDE EXECUTION (Legacy/Dev)
        is_windows = platform.system() == "Windows"
        abs_repo_path = str(self.repo_path.resolve())

        cmd = []
        mount_path = abs_repo_path

        if is_windows:
            drive, tail = os.path.splitdrive(abs_repo_path)
            if drive:
                drive_letter = drive[0].lower()
                rel_path = tail.replace("\\", "/")
                wsl_path = f"/mnt/{drive_letter}{rel_path}"
                cmd = ["wsl", "docker", "run"]
                mount_path = wsl_path
            else:
                logging.warning(
                    "Repository path '%s' does not start with a drive letter; "
                    "using absolute path directly for Docker mount.",
                    abs_repo_path,
                )
                cmd = ["docker", "run"]
                mount_path = abs_repo_path.replace("\\", "/")
        else:
            cmd = ["docker", "run"]

        cmd.extend(
            ["--rm", "-v", f"{mount_path}:/workspace", "-w", "/workspace/docker/src"]
        )

        # Display settings
        if self.config.live_view:
            if is_windows:
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                # Fix pixelated/jumbled display by disabling Qt DPI scaling over X11
                cmd.extend(["-e", "QT_AUTO_SCREEN_SCALE_FACTOR=0"])
                cmd.extend(["-e", "QT_SCALE_FACTOR=1"])
                cmd.extend(["-e", "QT_QPA_PLATFORM=xcb"])
                # Note: Resurrecting LIBGL_ALWAYS_INDIRECT=1 as it was present in the
                # "working state" from 2 days ago. Some Windows/VcXsrv configs need it.
                cmd.extend(["-e", "LIBGL_ALWAYS_INDIRECT=1"])
            else:
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])
        else:
            cmd.extend(["-e", "MUJOCO_GL=osmesa"])

        # Image and Command
        cmd.extend(["robotics_env", "/opt/mujoco-env/bin/python", "-u"])

        # Always use the working humanoid golf simulation that supports customization
        # Module path: docker/src/humanoid_golf/sim.py
        # (separate from mujoco_humanoid_golf package)
        # This reads simulation_config.json and applies all GUI settings:
        # - Body segment colors (shirt, pants, shoes, skin, eyes, club)
        # - Height and weight scaling
        # - Club parameters (length, mass)
        # - Control mode (PD, LQR, Polynomial)
        # - State save/load
        # - Live view vs headless mode
        cmd.extend(["-m", "humanoid_golf.sim"])

        return cmd, None

    def plot_induced_acceleration(self):
        """Plot Induced Acceleration Analysis from CSV."""
        if not HAS_MATPLOTLIB:
            return

        csv_path = self.current_dir.parent / "docker" / "src" / "golf_data.csv"
        if not csv_path.exists():
            QMessageBox.warning(self, "No Data", "golf_data.csv not found.")
            return

        try:
            # Read CSV headers to find joints
            joints = set()
            with open(csv_path) as f:
                reader = csv.reader(f)
                headers = next(reader)

            # Headers like iaa_{joint}_{type}
            # type is g, c, t, total
            for h in headers:
                if h.startswith("iaa_") and h.endswith("_total"):
                    # Extract joint name: iaa_HEAD_total
                    parts = h.split("_")
                    if len(parts) >= 3:
                        joint = "_".join(parts[1:-1])
                        joints.add(joint)

            if not joints:
                QMessageBox.warning(
                    self,
                    "No IAA Data",
                    "No Induced Acceleration data found in CSV.",
                )
                return

            sorted_joints = sorted(list(joints))

            # Ask user for joint
            joint, ok = QInputDialog.getItem(
                self, "Select Joint", "Joint:", sorted_joints, 0, False
            )
            if not ok or not joint:
                return

            # Read Data
            times: list[float] = []
            g_vals: list[float] = []
            c_vals: list[float] = []
            t_vals: list[float] = []
            tot_vals: list[float] = []

            # Column names
            col_g = f"iaa_{joint}_g"
            col_c = f"iaa_{joint}_c"
            col_t = f"iaa_{joint}_t"
            col_tot = f"iaa_{joint}_total"

            with open(csv_path) as f:
                reader = csv.DictReader(f)  # type: ignore[assignment]
                for row_dict in reader:
                    if isinstance(row_dict, dict):
                        try:
                            times.append(float(row_dict.get("time", "0")))
                            g_vals.append(float(row_dict.get(col_g, "0")))
                            c_vals.append(float(row_dict.get(col_c, "0")))
                            t_vals.append(float(row_dict.get(col_t, "0")))
                            tot_vals.append(float(row_dict.get(col_tot, "0")))
                        except (ValueError, KeyError):
                            continue

            if not times:
                return

            # Plot
            plt.figure(figsize=(10, 6))
            plt.plot(times, g_vals, label="Gravity", linestyle="--")
            plt.plot(times, c_vals, label="Velocity (Coriolis)", linestyle="-.")
            plt.plot(times, t_vals, label="Control", linestyle=":")
            plt.plot(times, tot_vals, label="Total", color="k", linewidth=1)

            plt.title(f"Induced Accelerations: {joint}")
            plt.xlabel("Time [s]")
            plt.ylabel("Acceleration [rad/s^2]")
            plt.legend()
            plt.grid(True)
            plt.show()

        except Exception as e:
            QMessageBox.critical(self, "Plot Error", str(e))

    def start_simulation(self):
        self.save_config()
        self.log("Starting simulation...")

        cmd, env = self.get_simulation_command()

        self.simulation_thread = ProcessWorker(cmd, env=env)
        self.simulation_thread.log_signal.connect(self.log)
        self.simulation_thread.finished_signal.connect(self.on_simulation_finished)

        self.simulation_thread.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def stop_simulation(self):
        if self.simulation_thread:
            self.log("Stopping simulation...")
            self.simulation_thread.stop()

    def on_simulation_finished(self, code, stderr):
        if code == 0:
            self.log("Simulation finished successfully.")
            self.btn_video.setEnabled(True)
            self.btn_data.setEnabled(True)
        elif code == 139:
            self.log(f"Simulation failed with code {code} (Segmentation Fault).")
            self.log(
                "⚠️ COMMON CAUSE: X11 Display Server not found or "
                "configured incorrectly."
            )
            self.log("1. Ensure VcXsrv (XLaunch) is running.")
            self.log("2. Ensure 'Disable access control' is CHECKED in VcXsrv.")
            self.log(
                "3. If you don't need the live GUI, uncheck 'Live Interactive View'."
            )
        else:
            self.log(f"Simulation failed with code {code}.")

        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

        # Handle segmentation fault with user prompt for headless mode
        if code == 139:
            reply = QMessageBox.question(
                self,
                "Simulation Crashed (X11 Error)",
                "The simulation crashed due to a display error "
                "(Segmentation Fault).\n\n"
                "This usually means the X11 server (VcXsrv) is not running or "
                "blocked.\n\n"
                "Would you like to try running in Headless Mode instead?\n"
                "(This will disable the live view but still generate video results)",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )

            if reply == QMessageBox.StandardButton.Yes:
                self.log("Switching to Headless Mode and retrying...")
                self.chk_live.setChecked(False)  # Uncheck box
                # Config is saved automatically in start_simulation
                self.start_simulation()

    def rebuild_docker(self):
        reply = QMessageBox.question(
            self,
            "Rebuild Environment",
            "This will rebuild the Docker environment. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            self.log(
                "Rebuilding Docker environment... "
                "(This functionality is simplified here, check terminal)"
            )

            docker_dir = self.repo_path / "docker"
            cmd = ["docker", "build", "-t", "robotics_env", "."]

            # Start worker for build
            self.build_thread = ProcessWorker(cmd, cwd=str(docker_dir))
            self.build_thread.log_signal.connect(self.log)
            self.build_thread.finished_signal.connect(
                lambda c, e: self.log(f"Build complete with code {c}")
            )
            self.build_thread.start()

    def open_video(self):
        vid_path = self.repo_path / "docker" / "src" / "humanoid_golf.mp4"
        self._open_file(vid_path)

    def open_data(self):
        csv_path = self.repo_path / "docker" / "src" / "golf_data.csv"
        self._open_file(csv_path)

    def _open_file(self, path):
        if not path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return

        if platform.system() == "Windows" and hasattr(os, "startfile"):
            # Ensure path exists before opening
            if path.exists():
                os.startfile(str(path))
            else:
                logging.error(f"Cannot open non-existent file: {path}")
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    # Global Stylesheet for Rounded Buttons and Modern Look
    app.setStyleSheet(
        """
        QPushButton {
            border-radius: 5px;
            padding: 5px;
        }
        QPushButton:hover {
            border: 1px solid #4a90e2;
        }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            border-radius: 4px;
            padding: 4px;
            border: 1px solid #555;
            background-color: #333;
            color: white;
            selection-background-color: #4a90e2;
        }
        QComboBox QAbstractItemView {
            background-color: #333;
            color: white;
            selection-background-color: #4a90e2;
            border: 1px solid #555;
        }
        QLabel, QCheckBox, QRadioButton, QGroupBox {
            color: white;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border: 1px solid #777;
            background: #2b2b2b;
        }
        QCheckBox::indicator:checked {
            background: #4a90e2;
            border: 1px solid #4a90e2;
        }
        QPushButton {
            color: white; /* Ensure button text is visible */
            background-color: #444; /* Default dark button bg */
        }
        /* Fix QMessageBox readability: prevent white text on white/system background */
        QMessageBox {
            background-color: #333; /* Dark background matching app */
        }
        QMessageBox QLabel {
            color: white; /* White text on dark background */
        }
    """
    )

    window = HumanoidLauncher()
    window.show()
    sys.exit(app.exec())
