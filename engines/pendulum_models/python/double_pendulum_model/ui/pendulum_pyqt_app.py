from __future__ import annotations

import ast
import functools
import logging
import math
import typing
from dataclasses import dataclass

import numpy as np  # noqa: TID253
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
from double_pendulum_model.ui.validation import (
    validate_polynomial_text,
    validate_torque_text,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6 import QtCore, QtWidgets

# Security: Use simpleeval for safe expression evaluation
from simpleeval import SimpleEval

logger = logging.getLogger(__name__)

TIME_STEP = 0.01

# Initialize shared evaluator for constant expressions
# Reusing the instance avoids overhead of re-registering functions
_EVALUATOR = SimpleEval()
_EVALUATOR.functions = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "log": math.log,
    "exp": math.exp,
}
_EVALUATOR.names = {
    "pi": math.pi,
}


def _validate_math_ast(node: ast.AST) -> None:
    allowed_nodes = {
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Name,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Call,
        ast.Constant,
        ast.BitXor,
    }
    allowed_names = {"pi", "sin", "cos"}

    for child in ast.walk(node):
        if type(child) not in allowed_nodes:
            msg = f"Disallowed syntax: {type(child).__name__}"
            raise ValueError(msg)
        if isinstance(child, ast.Name) and child.id not in allowed_names:
            msg = f"Unknown variable: {child.id}"
            raise ValueError(msg)
        if isinstance(child, ast.Call):
            if not isinstance(child.func, ast.Name):
                msg = "Only direct function calls allowed"
                raise TypeError(msg)
            if child.func.id not in allowed_names:
                msg = f"Disallowed function: {child.func.id}"
                raise ValueError(msg)


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
        self.ax: typing.Any = fig.add_subplot(111, projection="3d")
        super().__init__(fig)  # type: ignore[no-untyped-call]
        self._configure_axes()

    def _configure_axes(self) -> None:
        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.view_init(elev=25, azim=-60)
        self.ax.set_xlim([-2.5, 2.5])
        self.ax.set_ylim([-2.5, 2.5])
        self.ax.set_zlim([-1.5, 1.5])

    def draw_chain(self, points: np.ndarray[typing.Any, typing.Any]) -> None:
        self.ax.cla()
        self._configure_axes()
        xs, ys, zs = points.T
        self.ax.plot(xs, ys, zs, marker="o", linestyle="-", linewidth=2)
        self.draw()  # type: ignore[no-untyped-call]


class PendulumController(QtWidgets.QWidget):  # type: ignore[misc]
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

    def _validate_polynomial_input(
        self, text: str, widget: QtWidgets.QLineEdit
    ) -> None:
        error_msg = validate_polynomial_text(text)
        self._set_input_error(widget, error=error_msg)

    def _validate_torque_input(self, text: str, widget: QtWidgets.QLineEdit) -> None:
        error_msg = validate_torque_text(text)
        self._set_input_error(widget, error=error_msg)

    def _set_input_error(
        self, widget: QtWidgets.QLineEdit, *, error: str | None = None
    ) -> None:
        if error:
            widget.setStyleSheet("background-color: #ffcccc;")
            widget.setToolTip(error)
        else:
            widget.setStyleSheet("")
            widget.setToolTip("")

    def _build_layout(self) -> None:
        layout = QtWidgets.QHBoxLayout(self)
        layout.addWidget(self.canvas, stretch=3)

        control_panel = QtWidgets.QScrollArea()
        control_panel.setWidgetResizable(True)
        control_contents = QtWidgets.QWidget()

        self._build_form(control_contents)

        control_panel.setWidget(control_contents)
        layout.addWidget(control_panel, stretch=1)

    def _build_form(self, parent: QtWidgets.QWidget) -> None:
        form_layout = QtWidgets.QFormLayout(parent)

        self.status_group = QtWidgets.QGroupBox("Simulation Status")
        status_layout = QtWidgets.QVBoxLayout()
        self.status_label = QtWidgets.QLabel("Ready")
        self.status_label.setStyleSheet("font-family: monospace;")
        status_layout.addWidget(self.status_label)
        self.status_group.setLayout(status_layout)
        form_layout.addRow(self.status_group)

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

        self._add_torque_inputs(form_layout)
        self._add_velocity_inputs(form_layout)

        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.setToolTip("Start the simulation")
        self.start_button.clicked.connect(self._start)
        self.stop_button = QtWidgets.QPushButton("Pause")
        self.stop_button.setToolTip("Pause the simulation")
        self.stop_button.clicked.connect(self._pause)
        self.reset_button = QtWidgets.QPushButton("Reset")
        self.reset_button.setToolTip("Reset simulation to initial state")
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

    def _add_torque_inputs(self, layout: QtWidgets.QFormLayout) -> None:
        self.torque_inputs: dict[str, QtWidgets.QLineEdit] = {}
        for label, default in (
            ("Shoulder torque (N·m)", "0"),
            ("Wrist torque (N·m)", "0"),
            ("Elbow torque (N·m)", "0"),
        ):
            entry = QtWidgets.QLineEdit(default)
            entry.setToolTip("Constant value (e.g. 10.5) or math expression")
            entry.textChanged.connect(
                functools.partial(self._validate_torque_input, widget=entry)
            )
            self.torque_inputs[label] = entry
            layout.addRow(label, entry)

    def _add_velocity_inputs(self, layout: QtWidgets.QFormLayout) -> None:
        self.velocity_inputs: dict[str, QtWidgets.QLineEdit] = {}
        for label, default in (
            ("Shoulder ω polynomial", "0"),
            ("Wrist ω polynomial", "0"),
            ("Elbow ω polynomial", "0"),
        ):
            entry = QtWidgets.QLineEdit(default)
            entry.setToolTip(
                "Polynomial coefficients separated by '+' (e.g. 1.0+0.5+0.1)"
            )
            entry.textChanged.connect(
                functools.partial(self._validate_polynomial_input, widget=entry)
            )
            self.velocity_inputs[label] = entry
            layout.addRow(label, entry)

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
        """Safely evaluate mathematical expression using simpleeval.

        Security: Replaced eval() with simpleeval to eliminate code injection risk.
        """
        try:
            result = _EVALUATOR.eval(expression)
            return float(result)
        except (ValueError, TypeError, SyntaxError, NameError):
            logger.exception("Error evaluating expression: %s", expression)
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
            self._update_status(self.state_double)
        else:
            points = self._points_triple(self.state_triple)
            self._update_status(self.state_triple)
        self.canvas.draw_chain(points)

    def _update_status(self, state: DoublePendulumState | TriplePendulumState) -> None:
        status_text = f"Time: {self.time:.3f} s\n"

        def fmt(val: float) -> str:
            return f"{val:>7.3f}"

        status_text += f"θ1: {fmt(state.theta1)} | ω1: {fmt(state.omega1)}\n"
        status_text += f"θ2: {fmt(state.theta2)} | ω2: {fmt(state.omega2)}\n"

        if isinstance(state, TriplePendulumState):
            status_text += f"θ3: {fmt(state.theta3)} | ω3: {fmt(state.omega3)}\n"

        self.status_label.setText(status_text)

    def _points_double(
        self, state: DoublePendulumState
    ) -> np.ndarray[typing.Any, typing.Any]:
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

    def _points_triple(
        self, state: TriplePendulumState
    ) -> np.ndarray[typing.Any, typing.Any]:
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
        self, angle: float, rotation: np.ndarray[typing.Any, typing.Any], length: float
    ) -> np.ndarray[typing.Any, typing.Any]:
        local = np.array(
            [
                length * math.sin(angle),
                0.0,
                -length * math.cos(angle),
            ]
        )
        result = rotation @ local
        return np.array(result, dtype=np.float64)

    def _plane_rotation(
        self, inclination_deg: float
    ) -> np.ndarray[typing.Any, typing.Any]:
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
