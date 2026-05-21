"""
Lower Body Model - PyQt6/PyQt6 GUI Launcher
"""

# mypy: ignore-errors

import logging
import sys
import threading
import time

from mujoco import viewer
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QApplication,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMainWindow,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .builder import build_lower_body_xml
from .simulator import LowerBodySimulator
from src.shared.python.theme.integration import ThemedWindowMixin

_LOG = logging.getLogger(__name__)


class ControlPanel(ThemedWindowMixin, QMainWindow):
    """
    Control Panel for Lower Body Simulator.
    Allows real-time tweaking of initial posture and applies basic PD stability.
    """

    def __init__(self, sim: LowerBodySimulator, mujoco_viewer: viewer.Handle) -> None:
        super().__init__()
        self.setup_theme_support()
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.FramelessWindowHint)
        self.sim = sim
        self.viewer = mujoco_viewer

        self.setWindowTitle("Lower Body Control Panel")
        self.setMinimumWidth(400)
        self.setGeometry(50, 50, 400, 700)
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.WindowStaysOnTopHint)

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QVBoxLayout(main_widget)

        # Posture Controls
        posture_group = QGroupBox("Initial Posture Parameters (Degrees)")
        posture_layout = QFormLayout(posture_group)

        # 1. Anterior Tilt
        self.tilt_slider = QSlider(Qt.Orientation.Horizontal)
        self.tilt_slider.setMinimum(-45)
        self.tilt_slider.setMaximum(45)
        self.tilt_slider.setValue(0)
        self.tilt_slider.setTickInterval(5)
        self.tilt_lbl = QLabel("0")
        self.tilt_slider.valueChanged.connect(lambda v: self.tilt_lbl.setText(str(v)))
        posture_layout.addRow("Hip Anterior Tilt:", self.tilt_slider)
        posture_layout.addRow("", self.tilt_lbl)

        # 2. Knee Flexion
        self.knee_slider = QSlider(Qt.Orientation.Horizontal)
        self.knee_slider.setMinimum(0)
        self.knee_slider.setMaximum(160)
        self.knee_slider.setValue(0)
        self.knee_slider.setTickInterval(10)
        self.knee_lbl = QLabel("0")
        self.knee_slider.valueChanged.connect(lambda v: self.knee_lbl.setText(str(v)))
        posture_layout.addRow("Knee Flexion:", self.knee_slider)
        posture_layout.addRow("", self.knee_lbl)

        # 3. Foot Angle
        self.foot_slider = QSlider(Qt.Orientation.Horizontal)
        self.foot_slider.setMinimum(-45)
        self.foot_slider.setMaximum(45)
        self.foot_slider.setValue(0)
        self.foot_slider.setTickInterval(5)
        self.foot_lbl = QLabel("0")
        self.foot_slider.valueChanged.connect(lambda v: self.foot_lbl.setText(str(v)))
        posture_layout.addRow("Foot Extern Rot:", self.foot_slider)
        posture_layout.addRow("", self.foot_lbl)
        layout.addWidget(posture_group)

        # PD Gains
        pd_group = QGroupBox("Control Gains")
        pd_layout = QFormLayout(pd_group)

        self.kp_slider = QSlider(Qt.Orientation.Horizontal)
        self.kp_slider.setMinimum(0)
        self.kp_slider.setMaximum(2000)
        self.kp_slider.setValue(int(self.sim.kp_stability))
        self.kp_lbl = QLabel(str(int(self.sim.kp_stability)))
        self.kp_slider.valueChanged.connect(self.update_gains)
        pd_layout.addRow("Kp Stiffness:", self.kp_slider)
        pd_layout.addRow("", self.kp_lbl)

        self.kd_slider = QSlider(Qt.Orientation.Horizontal)
        self.kd_slider.setMinimum(0)
        self.kd_slider.setMaximum(300)
        self.kd_slider.setValue(int(self.sim.kd_stability))
        self.kd_lbl = QLabel(str(int(self.sim.kd_stability)))
        self.kd_slider.valueChanged.connect(self.update_gains)
        pd_layout.addRow("Kd Damping:", self.kd_slider)
        pd_layout.addRow("", self.kd_lbl)

        layout.addWidget(pd_group)

        # Playback Controls
        play_group = QGroupBox("Playback")
        play_layout = QVBoxLayout(play_group)

        self.is_playing = False
        self.play_btn = QPushButton("Play")
        self.play_btn.clicked.connect(self.toggle_play)
        play_layout.addWidget(self.play_btn)

        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(0)
        self.timeline_slider.setEnabled(True)
        self.timeline_slider.valueChanged.connect(self.scrub_timeline)
        play_layout.addWidget(QLabel("Scrub History:"))
        play_layout.addWidget(self.timeline_slider)

        layout.addWidget(play_group)

        # Re-apply Button
        apply_btn = QPushButton("Apply Initial Stance & Reset")
        apply_btn.clicked.connect(self.apply_stance)
        layout.addWidget(apply_btn)

        self.full_reset_btn = QPushButton("Full Reset Simulation")
        self.full_reset_btn.clicked.connect(self.full_reset_simulation)
        layout.addWidget(self.full_reset_btn)

        # IAA Demo Button
        iaa_btn = QPushButton("Run Induced Acceleration Trigger (Console)")
        iaa_btn.clicked.connect(self.run_iaa)
        layout.addWidget(iaa_btn)

        # Torque Functions Button
        torque_group = QGroupBox("Dynamic Control Generation")
        torque_layout = QVBoxLayout(torque_group)
        self.func_btn = QPushButton("Open Polynomial Designer")
        self.func_btn.clicked.connect(self.open_function_generator)
        torque_layout.addWidget(self.func_btn)
        layout.addWidget(torque_group)

        # Diagnostics Area
        diag_group = QGroupBox("System Diagnostics")
        diag_layout = QVBoxLayout(diag_group)
        self.diag_txt = QTextEdit()
        self.diag_txt.setReadOnly(True)
        self.diag_txt.setText("Initializing Telemetry...")
        diag_layout.addWidget(self.diag_txt)
        layout.addWidget(diag_group)

        # Low frequency UI updater
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_ui_state)
        self.timer.start(100)

        # Lock for sim access
        self.sim_lock = threading.RLock()

        self.ui_update_counter = 0

        # Background decoupled physics thread
        self.physics_thread = threading.Thread(target=self.physics_loop, daemon=True)
        self.physics_thread.start()

        # Initial apply
        self.apply_stance()

    def update_gains(self) -> None:
        val_kp = self.kp_slider.value()
        val_kd = self.kd_slider.value()
        self.kp_lbl.setText(str(val_kp))
        self.kd_lbl.setText(str(val_kd))
        with self.sim_lock:
            self.sim.kp_stability = float(val_kp)
            self.sim.kd_stability = float(val_kd)

    def toggle_play(self) -> None:
        self.is_playing = not self.is_playing
        self.play_btn.setText("Pause" if self.is_playing else "Play")
        self.timeline_slider.setEnabled(not self.is_playing)
        with self.sim_lock:
            if not self.is_playing and self.sim.history:
                self.timeline_slider.setMaximum(len(self.sim.history) - 1)
                self.timeline_slider.setValue(len(self.sim.history) - 1)

    def scrub_timeline(self) -> None:
        if self.is_playing:
            return
        idx = self.timeline_slider.value()
        with self.sim_lock:
            self.sim.restore_frame(idx)
            self.viewer.sync()

    def apply_stance(self) -> None:
        self.full_reset_simulation()

    def full_reset_simulation(self) -> None:
        """Reset simulation, playback, history, and current target pose."""
        tilt = self.tilt_slider.value()
        knee = self.knee_slider.value()
        foot = self.foot_slider.value()

        with self.sim_lock:
            self.is_playing = False
            self.play_btn.setText("Play")
            self.timeline_slider.setEnabled(True)

            self.sim.reset()
            self.sim.clear_history()

            try:
                self.sim.setup_initial_pose(
                    hip_anterior_tilt=tilt,
                    knee_flexion=knee,
                    foot_angle=foot,
                )
            except ValueError as exc:
                _LOG.warning("Infeasible pose: %s", exc)
                return
            if self.sim.hip_rotation_target is not None:
                self.sim.apply_hip_rotation_target(0.0)
                self.sim.qpos_target = self.sim.data.qpos.copy()

            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setMaximum(0)
            self.timeline_slider.setValue(0)
            self.timeline_slider.blockSignals(False)
            self.viewer.sync()

    def run_iaa(self) -> None:
        with self.sim_lock:
            _LOG.info("--- Induced Acceleration Analysis ---")
            iaa = self.sim.analyze_induced_acceleration("act_r_hip_x", 10.0)
            for k, v in iaa.items():
                _LOG.info(f"  {k}: {v:.4f}")

    def update_ui_state(self) -> None:
        """Runs on the main PyQt thread just to synchronize UI and check exit conditions."""
        if not self.viewer.is_running():
            QApplication.quit()
            return

        with self.sim_lock:
            # Update scrubber bounds gracefully preventing deadlocks
            if (
                not self.is_playing
                and self.sim.history
                and not self.timeline_slider.underMouse()
            ):
                current_max = self.timeline_slider.maximum()
                actual_max = len(self.sim.history) - 1
                if current_max != actual_max:
                    self.timeline_slider.blockSignals(True)
                    self.timeline_slider.setMaximum(actual_max)
                    self.timeline_slider.blockSignals(False)

            # Update Diagnostics
            if hasattr(self, "diag_txt"):
                diag = self.sim.compute_diagnostics()
                if not diag["is_diverged"]:
                    grf = diag.get("grf", {})
                    grf_text = f"GRF Z | R: {grf.get('right_z', 0):.1f} N | L: {grf.get('left_z', 0):.1f} N"

                    t_text = "\n".join(
                        [
                            f"  {k}: {v:.1f} Nm"
                            for k, v in diag.get("joint_torques", {}).items()
                            if v > 0.01
                        ]
                    )
                    if not t_text:
                        t_text = "  None Active"

                    text = (
                        f"Time: {diag['time_sec']:.2f} s | Frames: {diag['history_frames']}\n"
                        f"Z Height: {diag['pelvis_z_m']:.3f} m\n"
                        f"R. Knee: {diag['r_knee_deg']:.1f} deg\n"
                        f"Tracking Err: {diag['max_tracking_err_deg']:.1f} deg\n"
                        f"Total Torque: {diag['total_applied_torque_nm']:.1f} Nm\n"
                        f"{grf_text}\n"
                        f"Joint Torques:\n{t_text}"
                    )
                else:
                    text = "STATUS: DIVERGED (NaN)"

                # prevent selecting scrolling causing freezes
                scroll = self.diag_txt.verticalScrollBar().value()
                self.diag_txt.setPlainText(text)
                self.diag_txt.verticalScrollBar().setValue(scroll)

    def open_function_generator(self) -> None:
        try:
            import importlib
            import sys
            from pathlib import Path

            # Allow path resolution to module
            mod_path = Path(__file__).resolve().parent.parent.parent
            if str(mod_path) not in sys.path:
                sys.path.insert(0, str(mod_path))

            dialog_module = importlib.import_module(
                "pendulum_simulator.src.double_pendulum_golf.gui."
                "function_generator_dialog"
            )
            FunctionGeneratorDialog = dialog_module.FunctionGeneratorDialog
        except ImportError as e:
            _LOG.error(f"Could not import FunctionGeneratorDialog: {e}")
            return

        dlg = FunctionGeneratorDialog(
            self,
            joint_names=[
                "r_hip_x",
                "r_hip_y",
                "r_hip_z",
                "r_knee",
                "r_ankle_x",
                "r_ankle_y",
                "l_hip_x",
                "l_hip_y",
                "l_hip_z",
                "l_knee",
                "l_ankle_x",
                "l_ankle_y",
            ],
        )
        dlg.torque_imported.connect(self.on_torque_imported)
        dlg.exec()

    def on_torque_imported(self, joint_name: str, coeffs: object) -> None:
        with self.sim_lock:
            # Safely cast coeffs to list
            try:
                c = [float(x) for x in coeffs]
                self.sim.set_joint_polynomial(joint_name, c)
                _LOG.info(f"Imported torque polynomial for {joint_name}: {c}")
            except Exception as e:  # noqa: BLE001 — caller-supplied data may be any type
                _LOG.error(f"Failed to set polynomial: {e}")

    def physics_loop(self) -> None:
        """Dedicated background thread immune to PySide deadlocks."""
        while self.viewer.is_running():
            start_t = time.perf_counter()
            with self.sim_lock:
                if self.is_playing:
                    for _ in range(5):
                        self.sim.step()

                # Periodically sync to the visualizer
                self.viewer.sync()

            # Throttle accurately to 40Hz (25ms) minus execution time overhead
            elapsed = time.perf_counter() - start_t
            sleep_time = max(0.001, 0.025 - elapsed)
            time.sleep(sleep_time)


def main() -> None:
    app = QApplication(sys.argv)

    xml = build_lower_body_xml()
    sim = LowerBodySimulator(xml)

    # Launch passive viewer in its own thread block
    mj_viewer = viewer.launch_passive(sim.model, sim.data)

    # Create PyQt interface
    window = ControlPanel(sim, mj_viewer)
    window.show()

    # Block on UI until closed
    app.exec()

    # Close viewer when UI is closed
    if mj_viewer.is_running():
        mj_viewer.close()


if __name__ == "__main__":
    _LOG.basicConfig(level=logging.INFO)
    main()
