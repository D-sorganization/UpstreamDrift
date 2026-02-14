"""AnalysisMixin -- Analysis, config, and dialog methods for HumanoidLauncher."""

from __future__ import annotations

import csv
import datetime
import json
import logging
import sys
from collections.abc import Sequence
from pathlib import Path

from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QColorDialog,
    QDialog,
    QFileDialog,
    QInputDialog,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from src.shared.python.engine_core.engine_availability import MATPLOTLIB_AVAILABLE
from src.shared.python.theme.style_constants import Styles

HAS_MATPLOTLIB = MATPLOTLIB_AVAILABLE

if HAS_MATPLOTLIB:
    import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AnalysisMixin:
    """Analysis, configuration, and dialog methods for HumanoidLauncher."""

    def log(self, msg: str) -> None:
        """Append a message to the log or process a data stream packet."""
        # Check for JSON data stream

        if msg.startswith("DATA_JSON:"):
            try:
                json_str = msg.split("DATA_JSON:", 1)[1]

                data = json.loads(json_str)

                self.recorder.process_packet(data)

                self.live_plot.update_plot()

            except (ValueError, KeyError, json.JSONDecodeError) as e:
                # Fallback logging if parsing fails

                timestamp = datetime.datetime.now().strftime("%H:%M:%S")

                self.txt_log.append(f"[{timestamp}] Error parsing stream: {e}")

            return

        timestamp = datetime.datetime.now().strftime("%H:%M:%S")

        self.txt_log.append(f"[{timestamp}] {msg}")

        self.txt_log.ensureCursorVisible()

    def clear_log(self) -> None:
        """Clear the simulation log text area."""
        self.txt_log.clear()

        self.log("Log cleared.")

    def set_btn_color(self, btn: QPushButton, rgba: Sequence[float]) -> None:
        """Apply an RGBA color swatch to a button's background."""
        r, g, b = (int(c * 255) for c in rgba[:3])

        btn.setStyleSheet(Styles.color_swatch(r, g, b))

    def pick_color(self, key: str, btn: QPushButton) -> None:
        """Open a color picker dialog and store the chosen color."""
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
        """Update the help text and enable/disable signal generator buttons

        based on the selected control mode."""

        descriptions = {
            "pd": "Proportional-Derivative control (Target Pose tracking).",
            "lqr": "Linear Quadratic Regulator (Optimal control).",
            "poly": "Polynomial trajectory tracking (Time-varying torque).",
        }

        self.mode_help_label.setText(descriptions.get(mode, ""))

        self.btn_poly_generator.setEnabled(mode == "poly")

        self.btn_signal_toolkit.setEnabled(mode == "poly")

    def _load_polynomial_generator_class(self) -> type | None:
        try:
            import importlib.util

            target_file = (
                self.current_dir / "mujoco_humanoid_golf" / "polynomial_generator.py"
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

            return module.PolynomialGeneratorWidget  # type: ignore[attr-defined]

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Polynomial Generator Unavailable",
                f"The polynomial generator widget is not available.\n\nError: {e}\n\n"
                "Please ensure mujoco_humanoid_golf/polynomial_generator.py exists.",
            )
            return None

        except (RuntimeError, TypeError, AttributeError) as e:
            QMessageBox.warning(
                self,
                "Loading Error",
                f"Failed to load generator widget.\n\nError: {e}",
            )
            return None

    def _get_humanoid_joint_names(self) -> list[str]:
        return [
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

    def open_polynomial_generator(self) -> None:
        """Open polynomial generator dialog."""

        PolynomialGeneratorWidget = self._load_polynomial_generator_class()
        if PolynomialGeneratorWidget is None:
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Polynomial Function Generator")
        dialog.setMinimumSize(900, 700)
        layout = QVBoxLayout(dialog)

        poly_widget = PolynomialGeneratorWidget(dialog)
        poly_widget.set_joints(self._get_humanoid_joint_names())

        def on_polynomial_generated(joint_name: str, coefficients: list[float]) -> None:
            """Save generated polynomial coefficients to config."""
            self.config.polynomial_coefficients[joint_name] = coefficients
            self.save_config()
            self.log(f"Polynomial generated for {joint_name}: {coefficients}")

        poly_widget.polynomial_generated.connect(on_polynomial_generated)
        layout.addWidget(poly_widget)

        btn_close = QPushButton("Close")
        btn_close.clicked.connect(dialog.accept)
        layout.addWidget(btn_close)

        dialog.exec()

    def open_signal_toolkit(self) -> None:
        """Open signal processing toolkit dialog."""

        try:
            from src.shared.python.ui.qt.widgets.signal_toolkit_widget import (
                SignalToolkitWidget,
            )

        except ImportError as e:
            QMessageBox.warning(
                self,
                "Signal Toolkit Unavailable",
                f"The signal toolkit widget is not available.\n\nError: {e}\n\n"
                "Please ensure matplotlib and PyQt6 are installed.",
            )

            return

        except (RuntimeError, TypeError, AttributeError) as e:
            QMessageBox.warning(
                self,
                "Loading Error",
                f"Failed to load signal toolkit widget.\n\nError: {e}",
            )

            return

        # Create dialog

        dialog = QDialog(self)

        dialog.setWindowTitle("Signal Processing Toolkit")

        dialog.setMinimumSize(1200, 800)

        layout = QVBoxLayout(dialog)

        # Add signal toolkit widget

        toolkit_widget = SignalToolkitWidget(dialog)

        # Set available joints (humanoid joint names)

        joints = self._get_humanoid_joint_names()

        toolkit_widget.set_joints(joints)

        # Connect signal to save coefficients

        def on_signal_generated(joint_name: str, coefficients: list[float]) -> None:
            """Save generated polynomial coefficients to config."""

            self.config.polynomial_coefficients[joint_name] = coefficients

            self.save_config()

            self.log(f"Signal generated for {joint_name}: {coefficients}")

        toolkit_widget.signal_generated.connect(on_signal_generated)

        layout.addWidget(toolkit_widget)

        # Add close button

        btn_close = QPushButton("Close")

        btn_close.clicked.connect(dialog.accept)

        layout.addWidget(btn_close)

        dialog.exec()

    def browse_file(self, line_edit: QLineEdit, save: bool = False) -> None:
        """Open a file dialog and write the selected path to a line edit."""
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

    def load_config(self) -> None:
        """Deprecated: Config is loaded in __init__. Kept for compatibility.

        This method is a no-op since config is now loaded automatically during
        __init__ via ConfigurationManager. Calling this method has no effect.
        """

        # No-op: config is loaded in __init__ via self.config_manager.load()

    def save_config(self) -> None:
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

        except ImportError as e:
            self.log(f"Error saving config: {e}")

    def _extract_iaa_joints_from_headers(self, headers: list[str]) -> list[str]:
        joints = set()
        for h in headers:
            if h.startswith("iaa_") and h.endswith("_total"):
                parts = h.split("_")
                if len(parts) >= 3:
                    joint = "_".join(parts[1:-1])
                    joints.add(joint)
        return sorted(joints)

    def _read_iaa_data(
        self, csv_path: Path, joint: str
    ) -> tuple[list[float], list[float], list[float], list[float], list[float]]:
        times: list[float] = []
        g_vals: list[float] = []
        c_vals: list[float] = []
        t_vals: list[float] = []
        tot_vals: list[float] = []

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

        return times, g_vals, c_vals, t_vals, tot_vals

    def _render_iaa_plot(
        self,
        joint: str,
        times: list[float],
        g_vals: list[float],
        c_vals: list[float],
        t_vals: list[float],
        tot_vals: list[float],
    ) -> None:
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

    def plot_induced_acceleration(self) -> None:
        """Plot Induced Acceleration Analysis from CSV."""

        if not HAS_MATPLOTLIB:
            return

        csv_path = self.current_dir.parent / "docker" / "src" / "golf_data.csv"

        if not csv_path.exists():
            QMessageBox.warning(self, "No Data", "golf_data.csv not found.")
            return

        try:
            with open(csv_path) as f:
                reader = csv.reader(f)
                headers = next(reader)

            sorted_joints = self._extract_iaa_joints_from_headers(headers)

            if not sorted_joints:
                QMessageBox.warning(
                    self,
                    "No IAA Data",
                    "No Induced Acceleration data found in CSV.",
                )
                return

            joint, ok = QInputDialog.getItem(
                self, "Select Joint", "Joint:", sorted_joints, 0, False
            )
            if not ok or not joint:
                return

            times, g_vals, c_vals, t_vals, tot_vals = self._read_iaa_data(
                csv_path, joint
            )

            if not times:
                return

            self._render_iaa_plot(joint, times, g_vals, c_vals, t_vals, tot_vals)

        except (FileNotFoundError, PermissionError, OSError) as e:
            QMessageBox.critical(self, "Plot Error", str(e))
