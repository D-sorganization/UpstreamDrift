"""Humanoid Configuration Tab for AdvancedGolfAnalysisWindow.

Absorbs the settings previously found in the standalone humanoid_launcher.py:
- Simulation settings (control mode, Docker launch, state management)
- Appearance settings (height, weight, body colors)
- Equipment settings (club params, advanced model features)

This eliminates the need for the intermediate humanoid_launcher window,
allowing users to go directly from the main launcher tile to the
full simulation with all config options available.

Resolves: #1213, #1214
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from src.shared.python.config.configuration_manager import ConfigurationManager
from src.shared.python.data_io.path_utils import get_repo_root
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.ui.qt.process_worker import ProcessWorker

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

PROJECT_ROOT = get_repo_root()


class HumanoidConfigTab(QWidget):
    """Combined configuration tab for humanoid simulation settings.

    Provides sub-tabs for:
    - Docker Simulation (control mode, run/stop, state management)
    - Appearance (height, weight, body colors)
    - Equipment (club parameters, advanced features)
    """

    config_saved = pyqtSignal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Paths
        self._mujoco_dir = (
            Path(__file__).resolve().parent.parent.parent.parent  # mujoco/python
        )
        self.config_path = (
            self._mujoco_dir / "docker" / "src" / "simulation_config.json"
        )

        # State
        self.config_manager = ConfigurationManager(self.config_path)
        try:
            self.config = self.config_manager.load()
        except (RuntimeError, ValueError, OSError):
            from src.shared.python.config.configuration_manager import SimulationConfig

            self.config = SimulationConfig()

        self.simulation_thread: ProcessWorker | None = None

        self._setup_ui()

    def _setup_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Sub-tabs for the three config sections
        self.sub_tabs = QTabWidget()
        layout.addWidget(self.sub_tabs)

        self._setup_simulation_tab()
        self._setup_appearance_tab()
        self._setup_equipment_tab()

    # ------------------------------------------------------------------
    # Docker Simulation Tab
    # ------------------------------------------------------------------

    def _setup_simulation_tab(self) -> None:
        tab = QWidget()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(tab)

        tab_layout = QVBoxLayout(tab)
        tab_layout.setSpacing(10)
        tab_layout.setContentsMargins(10, 10, 10, 10)

        # -- Simulation Settings --
        settings_group = QGroupBox("Simulation Settings")
        settings_layout = QGridLayout()
        settings_layout.setSpacing(8)

        # Control Mode
        settings_layout.addWidget(QLabel("Control Mode:"), 0, 0)
        self.combo_control = QComboBox()
        self.combo_control.addItems(["pd", "lqr", "poly"])
        self.combo_control.setCurrentText(
            str(getattr(self.config, "control_mode", "pd"))
        )
        self.combo_control.currentTextChanged.connect(self._on_control_mode_changed)
        settings_layout.addWidget(self.combo_control, 0, 1)

        # Signal generator buttons
        self.btn_poly_generator = QPushButton("Polynomial Generator")
        self.btn_poly_generator.clicked.connect(self._open_polynomial_generator)
        settings_layout.addWidget(self.btn_poly_generator, 0, 2)

        self.btn_signal_toolkit = QPushButton("Signal Toolkit")
        self.btn_signal_toolkit.clicked.connect(self._open_signal_toolkit)
        settings_layout.addWidget(self.btn_signal_toolkit, 0, 3)

        # Mode help
        self.mode_help_label = QLabel()
        self.mode_help_label.setWordWrap(True)
        settings_layout.addWidget(self.mode_help_label, 1, 1, 1, 3)

        # Trigger initial state
        self._on_control_mode_changed(self.combo_control.currentText())

        # Live view checkbox
        self.chk_live = QCheckBox("Live Interactive View (requires X11/VcXsrv)")
        self.chk_live.setChecked(self.config.live_view)
        settings_layout.addWidget(self.chk_live, 2, 0, 1, 4)

        settings_group.setLayout(settings_layout)
        tab_layout.addWidget(settings_group)

        # -- State Management --
        state_group = QGroupBox("State Management")
        state_layout = QGridLayout()

        state_layout.addWidget(QLabel("Load State:"), 0, 0)
        self.txt_load_path = QLineEdit(self.config.load_state_path)
        state_layout.addWidget(self.txt_load_path, 0, 1)
        btn_browse_load = QPushButton("Browse")
        btn_browse_load.clicked.connect(lambda: self._browse_file(self.txt_load_path))
        state_layout.addWidget(btn_browse_load, 0, 2)

        state_layout.addWidget(QLabel("Save State:"), 1, 0)
        self.txt_save_path = QLineEdit(self.config.save_state_path)
        state_layout.addWidget(self.txt_save_path, 1, 1)
        btn_browse_save = QPushButton("Browse")
        btn_browse_save.clicked.connect(
            lambda: self._browse_file(self.txt_save_path, save=True)
        )
        state_layout.addWidget(btn_browse_save, 1, 2)

        state_group.setLayout(state_layout)
        tab_layout.addWidget(state_group)

        # -- Action Buttons --
        btn_layout = QHBoxLayout()

        self.btn_run = QPushButton("RUN DOCKER SIMULATION")
        self.btn_run.setStyleSheet(
            "background-color: #107c10; color: white;padding: 10px; font-weight: bold;"
        )
        self.btn_run.clicked.connect(self._start_simulation)

        self.btn_stop = QPushButton("STOP")
        self.btn_stop.setStyleSheet(
            "background-color: #d13438; color: white;padding: 10px; font-weight: bold;"
        )
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._stop_simulation)

        self.btn_rebuild = QPushButton("UPDATE ENV")
        self.btn_rebuild.setStyleSheet(
            "background-color: #8b5cf6; color: white;padding: 10px; font-weight: bold;"
        )
        self.btn_rebuild.clicked.connect(self._rebuild_docker)

        btn_layout.addWidget(self.btn_run)
        btn_layout.addWidget(self.btn_stop)
        btn_layout.addWidget(self.btn_rebuild)
        tab_layout.addLayout(btn_layout)

        # -- Results --
        results_layout = QHBoxLayout()
        results_layout.addWidget(QLabel("Results:"))

        self.btn_video = QPushButton("Open Video")
        self.btn_video.setEnabled(False)
        self.btn_video.clicked.connect(self._open_video)
        results_layout.addWidget(self.btn_video)

        self.btn_data = QPushButton("Open Data (CSV)")
        self.btn_data.setEnabled(False)
        self.btn_data.clicked.connect(self._open_data)
        results_layout.addWidget(self.btn_data)

        results_layout.addStretch()
        tab_layout.addLayout(results_layout)

        # -- Simulation Log --
        log_group = QGroupBox("Simulation Log")
        log_layout = QVBoxLayout()
        header = QHBoxLayout()
        header.addWidget(QLabel("Docker simulation output:"))
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(lambda: self.txt_log.clear())
        header.addWidget(btn_clear)
        header.addStretch()
        log_layout.addLayout(header)

        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setMaximumHeight(200)
        self.txt_log.setStyleSheet("font-family: Consolas, monospace; font-size: 10pt;")
        log_layout.addWidget(self.txt_log)
        log_group.setLayout(log_layout)
        tab_layout.addWidget(log_group)

        tab_layout.addStretch()
        self.sub_tabs.addTab(scroll, "Docker Simulation")

    # ------------------------------------------------------------------
    # Appearance Tab
    # ------------------------------------------------------------------

    def _setup_appearance_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        # Dimensions
        dim_group = QGroupBox("Physical Dimensions")
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
        color_group = QGroupBox("Body Colors")
        color_layout = QGridLayout()
        self.color_buttons: dict[str, QPushButton] = {}

        parts = [
            ("Shirt", "shirt"),
            ("Pants", "pants"),
            ("Shoes", "shoes"),
            ("Skin", "skin"),
            ("Club", "club"),
        ]

        for i, (name, key) in enumerate(parts):
            color_layout.addWidget(QLabel(name), i, 0)
            btn = QPushButton()
            btn.setFixedSize(50, 25)
            rgba = self.config.colors.get(key, [1, 1, 1, 1])
            self._set_btn_color(btn, rgba)
            btn.clicked.connect(lambda checked, k=key, b=btn: self._pick_color(k, b))
            color_layout.addWidget(btn, i, 1)
            self.color_buttons[key] = btn

        color_group.setLayout(color_layout)
        layout.addWidget(color_group)

        # Save button
        btn_save = QPushButton("Save Appearance")
        btn_save.setStyleSheet("background-color: #107c10; color: white; padding: 8px;")
        btn_save.clicked.connect(self._save_config)
        layout.addWidget(btn_save)

        layout.addStretch()
        self.sub_tabs.addTab(tab, "Appearance")

    # ------------------------------------------------------------------
    # Equipment Tab
    # ------------------------------------------------------------------

    def _setup_equipment_tab(self) -> None:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(10, 10, 10, 10)

        # Club parameters
        club_group = QGroupBox("Golf Club Parameters")
        club_layout = QGridLayout()

        club_layout.addWidget(QLabel("Club Length (m):"), 0, 0)
        self.slider_length = QSlider(Qt.Orientation.Horizontal)
        self.slider_length.setRange(50, 150)
        self.slider_length.setValue(int(self.config.club_length * 100))
        self.lbl_length_val = QLabel(f"{self.slider_length.value() / 100:.2f} m")
        self.slider_length.valueChanged.connect(
            lambda v: self.lbl_length_val.setText(f"{v / 100:.2f} m")
        )
        club_layout.addWidget(self.slider_length, 0, 1)
        club_layout.addWidget(self.lbl_length_val, 0, 2)

        club_layout.addWidget(QLabel("Club Mass (kg):"), 1, 0)
        self.slider_mass = QSlider(Qt.Orientation.Horizontal)
        self.slider_mass.setRange(10, 200)
        self.slider_mass.setValue(int(self.config.club_mass * 100))
        self.lbl_mass_val = QLabel(f"{float(self.slider_mass.value()) / 100:.2f} kg")
        self.slider_mass.valueChanged.connect(
            lambda v: self.lbl_mass_val.setText(f"{float(v) / 100:.2f} kg")
        )
        club_layout.addWidget(self.slider_mass, 1, 1)
        club_layout.addWidget(self.lbl_mass_val, 1, 2)

        club_group.setLayout(club_layout)
        layout.addWidget(club_group)

        # Advanced features
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

        # Save button
        btn_save = QPushButton("Save Equipment")
        btn_save.setStyleSheet("background-color: #107c10; color: white; padding: 8px;")
        btn_save.clicked.connect(self._save_config)
        layout.addWidget(btn_save)

        layout.addStretch()
        self.sub_tabs.addTab(tab, "Equipment")

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def _save_config(self) -> None:
        """Persist current UI state to simulation_config.json."""
        try:
            self.config.height_m = self.spin_height.value()
            self.config.weight_percent = self.slider_weight.value()
            self.config.club_length = self.slider_length.value() / 100.0
            self.config.club_mass = self.slider_mass.value() / 100.0
            self.config.two_handed = self.chk_two_hand.isChecked()
            self.config.enhance_face = self.chk_face.isChecked()
            self.config.articulated_fingers = self.chk_fingers.isChecked()
            self.config.save_state_path = self.txt_save_path.text()
            self.config.load_state_path = self.txt_load_path.text()
            self.config.live_view = self.chk_live.isChecked()
            self.config.control_mode = self.combo_control.currentText()

            self.config_manager.save(self.config)
            self._log(f"Config saved to {self.config_path}")
            self.config_saved.emit()
        except (RuntimeError, ValueError, OSError) as e:
            self._log(f"Error saving config: {e}")

    # ------------------------------------------------------------------
    # Control mode
    # ------------------------------------------------------------------

    def _on_control_mode_changed(self, mode: str) -> None:
        descriptions = {
            "pd": "Proportional-Derivative control (Target Pose tracking).",
            "lqr": "Linear Quadratic Regulator (Optimal control).",
            "poly": "Polynomial trajectory tracking (Time-varying torque).",
        }
        self.mode_help_label.setText(descriptions.get(mode, ""))
        self.btn_poly_generator.setEnabled(mode == "poly")
        self.btn_signal_toolkit.setEnabled(mode == "poly")

    # ------------------------------------------------------------------
    # Signal generators (lazy imports to avoid MuJoCo DLL issues)
    # ------------------------------------------------------------------

    _HUMANOID_JOINTS = [
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

    def _open_polynomial_generator(self) -> None:
        try:
            import importlib.util

            target_file = (
                self._mujoco_dir / "mujoco_humanoid_golf" / "polynomial_generator.py"
            )
            if not target_file.exists():
                raise FileNotFoundError(f"File not found: {target_file}")

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

            PolynomialGeneratorWidget = module.PolynomialGeneratorWidget  # type: ignore[attr-defined]
        except ImportError as e:
            QMessageBox.warning(self, "Unavailable", str(e))
            return

        from PyQt6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Polynomial Function Generator")
        dialog.setMinimumSize(900, 700)
        dlg_layout = QVBoxLayout(dialog)
        poly_widget = PolynomialGeneratorWidget(dialog)
        poly_widget.set_joints(self._HUMANOID_JOINTS)

        def on_generated(joint_name: str, coefficients: list[float]) -> None:
            self.config.polynomial_coefficients[joint_name] = coefficients
            self._save_config()
            self._log(f"Polynomial generated for {joint_name}: {coefficients}")

        poly_widget.polynomial_generated.connect(on_generated)
        dlg_layout.addWidget(poly_widget)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        dlg_layout.addWidget(btn_close)
        dialog.exec()

    def _open_signal_toolkit(self) -> None:
        try:
            from src.shared.python.ui.qt.widgets.signal_toolkit_widget import (
                SignalToolkitWidget,
            )
        except ImportError as e:
            QMessageBox.warning(self, "Unavailable", str(e))
            return

        from PyQt6.QtWidgets import QDialog

        dialog = QDialog(self)
        dialog.setWindowTitle("Signal Processing Toolkit")
        dialog.setMinimumSize(1200, 800)
        dlg_layout = QVBoxLayout(dialog)
        toolkit_widget = SignalToolkitWidget(dialog)
        toolkit_widget.set_joints(self._HUMANOID_JOINTS)

        def on_generated(joint_name: str, coefficients: list[float]) -> None:
            self.config.polynomial_coefficients[joint_name] = coefficients
            self._save_config()
            self._log(f"Signal generated for {joint_name}: {coefficients}")

        toolkit_widget.signal_generated.connect(on_generated)
        dlg_layout.addWidget(toolkit_widget)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        dlg_layout.addWidget(btn_close)
        dialog.exec()

    # ------------------------------------------------------------------
    # Docker simulation launch
    # ------------------------------------------------------------------

    def _get_simulation_command(self) -> tuple[list[str], dict[str, str] | None]:
        """Construct the Docker command to run the humanoid simulation."""
        in_docker = Path("/.dockerenv").exists()
        env = None

        if in_docker:
            cmd = ["python", "-m", "humanoid_golf.sim"]
            env = {"PYTHONPATH": "../docker/src"}
            return cmd, env

        is_windows = platform.system() == "Windows"
        repo_path = str(self._mujoco_dir.resolve())
        cmd: list[str] = []
        mount_path = repo_path

        if is_windows:
            drive, tail = os.path.splitdrive(repo_path)
            if drive:
                drive_letter = drive[0].lower()
                rel_path = tail.replace("\\", "/")
                wsl_path = f"/mnt/{drive_letter}{rel_path}"
                cmd = ["wsl", "docker", "run"]
                mount_path = wsl_path
            else:
                cmd = ["docker", "run"]
                mount_path = repo_path.replace("\\", "/")
        else:
            cmd = ["docker", "run"]

        cmd.extend(
            ["--rm", "-v", f"{mount_path}:/workspace", "-w", "/workspace/docker/src"]
        )

        if self.config.live_view:
            if is_windows:
                cmd.extend(["-e", "DISPLAY=host.docker.internal:0"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-e", "QT_AUTO_SCREEN_SCALE_FACTOR=0"])
                cmd.extend(["-e", "QT_SCALE_FACTOR=1"])
                cmd.extend(["-e", "QT_QPA_PLATFORM=xcb"])
            else:
                cmd.extend(["-e", f"DISPLAY={os.environ.get('DISPLAY', ':0')}"])
                cmd.extend(["-e", "MUJOCO_GL=glfw"])
                cmd.extend(["-e", "PYOPENGL_PLATFORM=glx"])
                cmd.extend(["-v", "/tmp/.X11-unix:/tmp/.X11-unix"])  # nosec B108
        else:
            cmd.extend(["-e", "MUJOCO_GL=osmesa"])

        cmd.extend(["robotics_env", "/opt/mujoco-env/bin/python", "-u"])
        cmd.extend(["-m", "humanoid_golf.sim"])
        return cmd, None

    def _start_simulation(self) -> None:
        self._save_config()
        self._log("Starting Docker simulation...")

        cmd, env = self._get_simulation_command()
        self.simulation_thread = ProcessWorker(cmd, env=env)
        self.simulation_thread.log_signal.connect(self._log)
        self.simulation_thread.finished_signal.connect(self._on_simulation_finished)
        self.simulation_thread.start()

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

    def _stop_simulation(self) -> None:
        if self.simulation_thread:
            self._log("Stopping simulation...")
            self.simulation_thread.stop()

    def _on_simulation_finished(self, code: int, stderr: str) -> None:
        if code == 0:
            self._log("Simulation finished successfully.")
            self.btn_video.setEnabled(True)
            self.btn_data.setEnabled(True)
        elif code == 139:
            self._log(f"Simulation crashed (code {code}) - X11 display issue.")
            reply = QMessageBox.question(
                self,
                "Simulation Crashed (X11 Error)",
                "The simulation crashed due to a display error.\n\n"
                "Would you like to retry in Headless Mode?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.chk_live.setChecked(False)
                self._start_simulation()
                return
        else:
            self._log(f"Simulation failed with code {code}.")

        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)

    def _rebuild_docker(self) -> None:
        reply = QMessageBox.question(
            self,
            "Rebuild Environment",
            "This will rebuild the Docker environment. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._log("Rebuilding Docker environment...")
            docker_dir = self._mujoco_dir / "docker"
            cmd = ["docker", "build", "-t", "robotics_env", "."]
            build_thread = ProcessWorker(cmd, cwd=str(docker_dir))
            build_thread.log_signal.connect(self._log)
            build_thread.finished_signal.connect(
                lambda c, e: self._log(f"Build complete with code {c}")
            )
            build_thread.start()
            # Store reference to prevent GC
            self._build_thread = build_thread

    # ------------------------------------------------------------------
    # File helpers
    # ------------------------------------------------------------------

    def _open_video(self) -> None:
        vid_path = self._mujoco_dir / "docker" / "src" / "humanoid_golf.mp4"
        self._open_file(vid_path)

    def _open_data(self) -> None:
        csv_path = self._mujoco_dir / "docker" / "src" / "golf_data.csv"
        self._open_file(csv_path)

    def _open_file(self, path: Path) -> None:
        if not path.exists():
            QMessageBox.warning(self, "Error", f"File not found: {path}")
            return
        if platform.system() == "Windows" and hasattr(os, "startfile"):
            os.startfile(str(path))
        elif platform.system() == "Darwin":
            subprocess.run(["open", str(path)], check=False)
        else:
            subprocess.run(["xdg-open", str(path)], check=False)

    def _browse_file(self, line_edit: QLineEdit, save: bool = False) -> None:
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

    def _set_btn_color(self, btn: QPushButton, rgba: Sequence[float]) -> None:
        r, g, b = (int(c * 255) for c in rgba[:3])
        btn.setStyleSheet(
            f"background-color: rgb({r},{g},{b}); border: 1px solid #555;"
        )

    def _pick_color(self, key: str, btn: QPushButton) -> None:
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
            self._set_btn_color(btn, new_rgba)
            self._save_config()

    def _log(self, msg: str) -> None:
        import datetime

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.txt_log.append(f"[{timestamp}] {msg}")
        self.txt_log.ensureCursorVisible()
