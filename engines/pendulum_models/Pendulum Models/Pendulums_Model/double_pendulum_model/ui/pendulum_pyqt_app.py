from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

from double_pendulum_model.physics.double_pendulum import (
    DoublePendulumDynamics,
    DoublePendulumParameters,
    DoublePendulumState,
    compile_forcing_functions,
)
from double_pendulum_model.physics.triple_pendulum import (
    PolynomialProfile,
    TriplePendulumDynamics,
    TriplePendulumParameters,
    TriplePendulumState,
)

TIME_STEP = 0.01


@dataclass
class SimulationConfig:
    model: str
    gravity_enabled: bool
    constrained_to_plane: bool
    forward_mode: bool
    torque_expressions: tuple[str, str, str]
    velocity_polynomials: tuple[str, str, str]


class PendulumCanvas(FigureCanvasQTAgg):
    def __init__(self) -> None:
        fig = Figure(figsize=(8, 8), dpi=100)
        self.ax = fig.add_subplot(111, projection="3d")
        super().__init__(fig)
        self._configure_axes()

    def _configure_axes(self) -> None:
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=25, azim=-60)
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([-1.5, 1.5])

    def draw_chain(self, points: np.ndarray) -> None:
        self.ax.cla()
        self._configure_axes()
        xs, ys, zs = points.T
        self.ax.plot(xs, ys, zs, marker="o", linestyle="-", linewidth=2)
        self.draw()


class PendulumController(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Driven Pendulum Explorer — PyQt6")
        self.resize(1500, 950)

        self.canvas = PendulumCanvas()
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._on_step)

        self.state_double = DoublePendulumState(
            theta1=-0.5, theta2=-1.2, omega1=0.0, omega2=0.0
        )
        self.state_triple = TriplePendulumState(
            theta1=-0.5, theta2=-0.8, theta3=-0.6, omega1=0.0, omega2=0.0, omega3=0.0
        )

        self.double_params = DoublePendulumParameters.default()
        self.triple_params = TriplePendulumParameters.default()
        self.double_dynamics = DoublePendulumDynamics(self.double_params)
        self.triple_dynamics = TriplePendulumDynamics(self.triple_params)

        self.time = 0.0
        self._build_layout()
        self._update_plot()

    def _build_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.canvas, stretch=3)

        control_panel = QtWidgets.QScrollArea()
        control_panel.setWidgetResizable(True)
        control_contents = QtWidgets.QWidget()
        form_layout = QtWidgets.QFormLayout(control_contents)

        self.model_selector = QtWidgets.QComboBox()
        self.model_selector.addItems(["Double", "Triple"])

        self.gravity_checkbox = QtWidgets.QCheckBox("Enable gravity")
        self.gravity_checkbox.setChecked(True)

        self.plane_checkbox = QtWidgets.QCheckBox("Constrain to swing plane")
        self.plane_checkbox.setChecked(True)

        self.mode_selector = QtWidgets.QComboBox()
        self.mode_selector.addItems(
            ["Forward dynamics (torques)", "Inverse dynamics (velocity polynomials)"]
        )

        self.torque_inputs: dict[str, QtWidgets.QLineEdit] = {}
        for label, default in (
            ("Shoulder torque (N·m)", "0"),
            ("Wrist torque (N·m)", "0"),
            ("Elbow torque (N·m)", "0"),
        ):
            entry = QtWidgets.QLineEdit(default)
            self.torque_inputs[label] = entry
            form_layout.addRow(label, entry)

        self.velocity_inputs: dict[str, QtWidgets.QLineEdit] = {}
        for label, default in (
            ("Shoulder ω polynomial", "0"),
            ("Wrist ω polynomial", "0"),
            ("Elbow ω polynomial", "0"),
        ):
            entry = QtWidgets.QLineEdit(default)
            self.velocity_inputs[label] = entry
            form_layout.addRow(label, entry)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QtWidgets.QPushButton("Pause")
        self.stop_button.clicked.connect(self._pause)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.clicked.connect(self._reset)

        for widget_label, widget in (
            ("Model", self.model_selector),
            ("Gravity", self.gravity_checkbox),
            ("Plane constraint", self.plane_checkbox),
            ("Mode", self.mode_selector),
        ):
            form_layout.insertRow(0, widget_label, widget)

        button_row = QtWidgets.QHBoxLayout()
        button_row.addWidget(self.start_button)
        button_row.addWidget(self.stop_button)
        button_row.addWidget(self.reset_button)
        form_layout.addRow(button_row)

        control_panel.setWidget(control_contents)
        layout.addWidget(control_panel, stretch=1)

    def _current_config(self) -> SimulationConfig:
        return SimulationConfig(
            model=self.model_selector.currentText(),
            gravity_enabled=self.gravity_checkbox.isChecked(),
            constrained_to_plane=self.plane_checkbox.isChecked(),
            forward_mode=self.mode_selector.currentIndex() == 0,
            torque_expressions=(
                self.torque_inputs["Shoulder torque (N·m)"].text(),
                self.torque_inputs["Wrist torque (N·m)"].text(),
                self.torque_inputs["Elbow torque (N·m)"].text(),
            ),
            velocity_polynomials=(
                self.velocity_inputs["Shoulder ω polynomial"].text(),
                self.velocity_inputs["Wrist ω polynomial"].text(),
                self.velocity_inputs["Elbow ω polynomial"].text(),
            ),
        )

    def _start(self) -> None:
        self.time = 0.0
        self.timer.start(int(TIME_STEP * 1000))

    def _pause(self) -> None:
        self.timer.stop()

    def _reset(self) -> None:
        self.timer.stop()
        self.time = 0.0
        self.state_double = DoublePendulumState(
            theta1=-0.5, theta2=-1.2, omega1=0.0, omega2=0.0
        )
        self.state_triple = TriplePendulumState(
            theta1=-0.5, theta2=-0.8, theta3=-0.6, omega1=0.0, omega2=0.0, omega3=0.0
        )
        self._update_plot()

    def _on_step(self) -> None:
        config = self._current_config()
        self.double_params.gravity_enabled = config.gravity_enabled
        self.triple_params.gravity_enabled = config.gravity_enabled
        self.double_params.constrained_to_plane = config.constrained_to_plane

        if config.model == "Double":
            if config.forward_mode:
                forcing = compile_forcing_functions(
                    config.torque_expressions[0], config.torque_expressions[1]
                )
                self.double_dynamics.forcing_functions = forcing
                self.state_double = self.double_dynamics.step(
                    self.time, self.state_double, TIME_STEP
                )
            else:
                profiles = self._polynomial_profiles(config.velocity_polynomials[:2])
                self.state_double = self._apply_inverse_profile_double(
                    self.state_double, profiles
                )
            self.time += TIME_STEP
            self._update_plot()
        else:
            if config.forward_mode:
                torques = tuple(
                    self._safe_eval(expr) for expr in config.torque_expressions
                )
                self.state_triple = self.triple_dynamics.step(
                    self.time, self.state_triple, TIME_STEP, torques
                )
            else:
                profiles = self._polynomial_profiles(config.velocity_polynomials)
                self.state_triple = self._apply_inverse_profile_triple(
                    self.state_triple, profiles
                )
            self.time += TIME_STEP
            self._update_plot()

    def _safe_eval(self, expression: str) -> float:
        try:
            return float(
                eval(
                    expression,
                    {
                        "__builtins__": {},
                        "pi": math.pi,
                        "sin": math.sin,
                        "cos": math.cos,
                    },
                )
            )
        except Exception:
            return 0.0

    def _polynomial_profiles(
        self, expressions: tuple[str, ...]
    ) -> tuple[PolynomialProfile, ...]:
        profiles = []
        for expr in expressions:
            cleaned = expr.replace(" ", "")
            if not cleaned:
                profiles.append(PolynomialProfile((0.0,)))
                continue
            try:
                coefficients = tuple(float(c) for c in cleaned.split("+"))
            except ValueError:
                coefficients = (0.0,)
            profiles.append(PolynomialProfile(coefficients))
        return tuple(profiles)

    def _apply_inverse_profile_double(
        self,
        state: DoublePendulumState,
        profiles: tuple[PolynomialProfile, PolynomialProfile],
    ) -> DoublePendulumState:
        omega1 = profiles[0].omega(self.time)
        omega2 = profiles[1].omega(self.time)
        alpha1 = profiles[0].alpha(self.time)
        alpha2 = profiles[1].alpha(self.time)

        state_with_profile = DoublePendulumState(
            theta1=state.theta1,
            theta2=state.theta2,
            omega1=omega1,
            omega2=omega2,
        )

        accelerations = (alpha1, alpha2)
        torques = self.double_dynamics.inverse_dynamics(
            state_with_profile, accelerations
        )
        self.double_dynamics.forcing_functions = (
            lambda _t, _s: torques[0],
            lambda _t, _s: torques[1],
        )
        return self.double_dynamics.step(self.time, state_with_profile, TIME_STEP)

    def _apply_inverse_profile_triple(
        self,
        state: TriplePendulumState,
        profiles: tuple[PolynomialProfile, PolynomialProfile, PolynomialProfile],
    ) -> TriplePendulumState:
        omega = [profile.omega(self.time) for profile in profiles]
        alpha = [profile.alpha(self.time) for profile in profiles]
        accelerations = tuple(alpha)
        state_with_profile = TriplePendulumState(
            theta1=state.theta1,
            theta2=state.theta2,
            theta3=state.theta3,
            omega1=omega[0],
            omega2=omega[1],
            omega3=omega[2],
        )
        torques = self.triple_dynamics.inverse_dynamics(
            state_with_profile, accelerations
        )
        return self.triple_dynamics.step(
            self.time, state_with_profile, TIME_STEP, torques
        )

    def _update_plot(self) -> None:
        config = self._current_config()
        if config.model == "Double":
            points = self._points_double(self.state_double)
        else:
            points = self._points_triple(self.state_triple)
        self.canvas.draw_chain(points)

    def _points_double(self, state: DoublePendulumState) -> np.ndarray:
        plane_rotation = self._plane_rotation(self.double_params.plane_inclination_deg)
        shoulder = np.array([0.0, 0.0, 0.0])
        upper = self._point_from_angles(
            state.theta1, plane_rotation, self.double_params.upper_segment.length_m
        )
        lower = upper + self._point_from_angles(
            state.theta1 + state.theta2,
            plane_rotation,
            self.double_params.lower_segment.length_m,
        )
        return np.vstack([shoulder, upper, lower])

    def _points_triple(self, state: TriplePendulumState) -> np.ndarray:
        shoulder = np.array([0.0, 0.0, 0.0])
        params = self.triple_params.segments
        plane_rotation = self._plane_rotation(35.0)
        p1 = self._point_from_angles(state.theta1, plane_rotation, params[0].length_m)
        p2 = p1 + self._point_from_angles(
            state.theta1 + state.theta2, plane_rotation, params[1].length_m
        )
        p3 = p2 + self._point_from_angles(
            state.theta1 + state.theta2 + state.theta3,
            plane_rotation,
            params[2].length_m,
        )
        return np.vstack([shoulder, p1, p2, p3])

    def _point_from_angles(
        self, angle: float, rotation: np.ndarray, length: float
    ) -> np.ndarray:
        local = np.array(
            [
                length * math.sin(angle),
                0.0,
                -length * math.cos(angle),
            ]
        )
        return rotation @ local

    def _plane_rotation(self, inclination_deg: float) -> np.ndarray:
        inclination_rad = math.radians(inclination_deg)
        cos_inc = math.cos(inclination_rad)
        sin_inc = math.sin(inclination_rad)
        return np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_inc, -sin_inc],
                [0.0, sin_inc, cos_inc],
            ]
        )


def run_app() -> None:
    app = QtWidgets.QApplication([])
    controller = PendulumController()
    controller.show()
    app.exec()


if __name__ == "__main__":
    run_app()
