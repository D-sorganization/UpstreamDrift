"""
Driven double pendulum dynamics with control-affine structure.

This module models a two-link planar manipulator (shoulder + wrist) swinging on a
user-specified plane (e.g., a golf swing plane). It exposes control-affine
dynamics, supports arbitrary user forcing functions, and reports joint torques
for educational demonstrations of chaos and control.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass

# Physical constants with documented units and references
# International gravity standard at 45 degrees latitude (m/s^2)
GRAVITATIONAL_ACCELERATION = 9.80665

# Typical anatomical and equipment references for defaults
DEFAULT_ARM_LENGTH_M = 0.75  # Representative combined arm length (meters)
DEFAULT_ARM_MASS_KG = 7.5  # Nominal combined mass of both arms (kilograms)
DEFAULT_ARM_CENTER_OF_MASS_RATIO = 0.45  # Dimensionless fraction of length
DEFAULT_ARM_INERTIA_SCALING = 1.0 / 12.0  # Uniform rod inertia coefficient about COM

DEFAULT_SHAFT_LENGTH_M = 1.0  # Representative golf shaft length (meters)
DEFAULT_SHAFT_MASS_KG = 0.35  # Typical steel shaft mass (kilograms)
DEFAULT_CLUBHEAD_MASS_KG = 0.20  # Typical driver clubhead mass (kilograms)
DEFAULT_SHAFT_COM_RATIO = 0.43  # Empirical COM ratio for a driver (dimensionless)
DEFAULT_PLANE_INCLINATION_DEG = 35.0  # Golf swing plane tilt from vertical (degrees)

# Damping constants for energy dissipation in joints (N·m·s/rad)
DEFAULT_DAMPING_SHOULDER = 0.4
DEFAULT_DAMPING_WRIST = 0.25

# Numerical tolerance for detecting singular mass matrices (dimensionless)
MASS_MATRIX_SINGULAR_TOLERANCE = 1e-12


class ExpressionFunction:
    """Safe evaluation of user-provided expressions.

    The expression can use standard math functions, state variables, and time
    (``t``). Only a curated subset of ``ast`` nodes are accepted to prevent
    arbitrary code execution.
    """

    _ALLOWED_NODES = {
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
        ast.Attribute,
        ast.BitXor,
    }

    _ALLOWED_NAMES = {
        name: getattr(math, name)
        for name in (
            "sin",
            "cos",
            "tan",
            "asin",
            "acos",
            "atan",
            "atan2",
            "sqrt",
            "log",
            "log10",
            "exp",
            "pi",
            "tau",
            "fabs",
        )
    }

    def __init__(self, expression: str) -> None:
        self.expression = expression.strip()
        parsed = ast.parse(self.expression, mode="eval")
        self._validate_ast(parsed)
        self._code = compile(parsed, filename="<ExpressionFunction>", mode="eval")

    def __call__(self, t: float, state: DoublePendulumState) -> float:
        context: dict[str, float] = {
            "t": t,
            "theta1": state.theta1,
            "theta2": state.theta2,
            "omega1": state.omega1,
            "omega2": state.omega2,
            **self._ALLOWED_NAMES,
        }
        result = eval(self._code, {"__builtins__": {}}, context)
        return float(result)

    def _validate_ast(self, node: ast.AST) -> None:
        for child in ast.walk(node):
            if (
                type(child) not in self._ALLOWED_NODES
            ):  # noqa: E721 - type comparison is intentional
                raise ValueError(
                    f"Disallowed syntax in expression: {type(child).__name__}"
                )
            if isinstance(child, ast.Name) and child.id not in {
                "t",
                "theta1",
                "theta2",
                "omega1",
                "omega2",
                *self._ALLOWED_NAMES,
            }:
                raise ValueError(f"Use of unknown variable '{child.id}' in expression")
            if isinstance(child, ast.Call):
                if not isinstance(child.func, ast.Name | ast.Attribute):
                    raise ValueError("Only direct function calls are permitted")
                if (
                    isinstance(child.func, ast.Name)
                    and child.func.id not in self._ALLOWED_NAMES
                ):
                    raise ValueError(f"Function '{child.func.id}' is not permitted")


@dataclass
class SegmentProperties:
    """Physical properties of a single pendulum segment."""

    length_m: float
    mass_kg: float
    center_of_mass_ratio: float
    inertia_about_com: float

    @property
    def center_of_mass_distance(self) -> float:
        return self.length_m * self.center_of_mass_ratio

    @property
    def inertia_about_proximal_joint(self) -> float:
        return self.inertia_about_com + self.mass_kg * self.center_of_mass_distance**2


@dataclass
class LowerSegmentProperties:
    """Composite properties for a golf-club-like lower segment."""

    length_m: float
    shaft_mass_kg: float
    clubhead_mass_kg: float
    shaft_com_ratio: float

    @property
    def total_mass(self) -> float:
        return self.shaft_mass_kg + self.clubhead_mass_kg

    @property
    def center_of_mass_distance(self) -> float:
        shaft_com = self.length_m * self.shaft_com_ratio
        weighted_sum = (
            shaft_com * self.shaft_mass_kg + self.length_m * self.clubhead_mass_kg
        )
        return weighted_sum / self.total_mass

    @property
    def inertia_about_com(self) -> float:
        shaft_inertia_com = (1.0 / 12.0) * self.shaft_mass_kg * self.length_m**2
        # Shaft COM is at shaft_com_ratio * length, not at midpoint
        shaft_com_position = self.length_m * self.shaft_com_ratio
        shaft_offset = (shaft_com_position - self.center_of_mass_distance) ** 2
        clubhead_offset = (self.length_m - self.center_of_mass_distance) ** 2
        parallel_axis = (
            self.shaft_mass_kg * shaft_offset + self.clubhead_mass_kg * clubhead_offset
        )
        return shaft_inertia_com + parallel_axis

    @property
    def inertia_about_proximal_joint(self) -> float:
        return (
            self.inertia_about_com + self.total_mass * self.center_of_mass_distance**2
        )


@dataclass
class DoublePendulumParameters:
    """Configuration for the double pendulum."""

    upper_segment: SegmentProperties
    lower_segment: LowerSegmentProperties
    plane_inclination_deg: float = DEFAULT_PLANE_INCLINATION_DEG
    damping_shoulder: float = DEFAULT_DAMPING_SHOULDER
    damping_wrist: float = DEFAULT_DAMPING_WRIST
    gravity_m_s2: float = GRAVITATIONAL_ACCELERATION
    gravity_enabled: bool = True
    constrained_to_plane: bool = True

    @classmethod
    def default(cls) -> DoublePendulumParameters:
        upper_inertia = (
            DEFAULT_ARM_INERTIA_SCALING * DEFAULT_ARM_MASS_KG * DEFAULT_ARM_LENGTH_M**2
        )
        upper_segment = SegmentProperties(
            length_m=DEFAULT_ARM_LENGTH_M,
            mass_kg=DEFAULT_ARM_MASS_KG,
            center_of_mass_ratio=DEFAULT_ARM_CENTER_OF_MASS_RATIO,
            inertia_about_com=upper_inertia,
        )
        lower_segment = LowerSegmentProperties(
            length_m=DEFAULT_SHAFT_LENGTH_M,
            shaft_mass_kg=DEFAULT_SHAFT_MASS_KG,
            clubhead_mass_kg=DEFAULT_CLUBHEAD_MASS_KG,
            shaft_com_ratio=DEFAULT_SHAFT_COM_RATIO,
        )
        return cls(upper_segment=upper_segment, lower_segment=lower_segment)

    @property
    def plane_inclination_rad(self) -> float:
        return math.radians(self.plane_inclination_deg)

    @property
    def projected_gravity(self) -> float:
        if not self.gravity_enabled:
            return 0.0
        if not self.constrained_to_plane:
            return self.gravity_m_s2
        return self.gravity_m_s2 * math.cos(self.plane_inclination_rad)


@dataclass
class DoublePendulumState:
    """Dynamic state of the pendulum."""

    theta1: float  # Angle of upper segment from vertical (in-plane)
    theta2: float  # Relative angle of lower segment from upper segment (in-plane)
    omega1: float  # Angular velocity of upper segment
    omega2: float  # Angular velocity of lower segment
    phi: float = 0.0  # Out-of-plane angle (above/below plane, in radians)
    omega_phi: float = 0.0  # Out-of-plane angular velocity


@dataclass
class JointTorques:
    """Torque decomposition at the joints."""

    applied: tuple[float, float]
    gravitational: tuple[float, float]
    damping: tuple[float, float]
    coriolis_centripetal: tuple[float, float]


class DoublePendulumDynamics:
    """Control-affine driven double pendulum."""

    def __init__(
        self,
        parameters: DoublePendulumParameters | None = None,
        forcing_functions: (
            tuple[Callable[[float, DoublePendulumState], float], ...] | None
        ) = None,
    ) -> None:
        self.parameters = parameters or DoublePendulumParameters.default()

        def zero_input(_: float, __: DoublePendulumState) -> float:
            return 0.0

        self.forcing_functions = forcing_functions or (zero_input, zero_input)

    def mass_matrix(
        self, theta2: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        p = self.parameters
        m1 = p.upper_segment.mass_kg
        m2 = p.lower_segment.total_mass
        l1 = p.upper_segment.length_m
        lc1 = p.upper_segment.center_of_mass_distance
        lc2 = p.lower_segment.center_of_mass_distance
        i1 = p.upper_segment.inertia_about_proximal_joint
        i2 = p.lower_segment.inertia_about_proximal_joint
        cos_theta2 = math.cos(theta2)

        m11 = i1 + i2 + m1 * lc1**2 + m2 * (l1**2 + lc2**2 + 2 * l1 * lc2 * cos_theta2)
        m12 = i2 + m2 * (lc2**2 + l1 * lc2 * cos_theta2)
        m22 = i2 + m2 * lc2**2
        return ((m11, m12), (m12, m22))

    def coriolis_vector(
        self, theta2: float, omega1: float, omega2: float
    ) -> tuple[float, float]:
        p = self.parameters
        m2 = p.lower_segment.total_mass
        l1 = p.upper_segment.length_m
        lc2 = p.lower_segment.center_of_mass_distance
        sin_theta2 = math.sin(theta2)
        h = -m2 * l1 * lc2 * sin_theta2
        c1 = h * (2 * omega1 * omega2 + omega2**2)
        c2 = h * omega1**2
        return c1, c2

    def gravity_vector(self, theta1: float, theta2: float) -> tuple[float, float]:
        p = self.parameters
        m1 = p.upper_segment.mass_kg
        m2 = p.lower_segment.total_mass
        l1 = p.upper_segment.length_m
        lc1 = p.upper_segment.center_of_mass_distance
        lc2 = p.lower_segment.center_of_mass_distance
        g = p.projected_gravity
        g1 = (m1 * lc1 + m2 * l1) * g * math.sin(theta1) + m2 * lc2 * g * math.sin(
            theta1 + theta2
        )
        g2 = m2 * lc2 * g * math.sin(theta1 + theta2)
        return g1, g2

    def damping_vector(self, omega1: float, omega2: float) -> tuple[float, float]:
        p = self.parameters
        d1 = p.damping_shoulder * omega1
        d2 = p.damping_wrist * omega2
        return d1, d2

    def _invert_mass_matrix(
        self, theta2: float
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        mass = self.mass_matrix(theta2)
        determinant = mass[0][0] * mass[1][1] - mass[0][1] * mass[1][0]
        if abs(determinant) <= MASS_MATRIX_SINGULAR_TOLERANCE:
            raise ZeroDivisionError(
                "Mass matrix determinant is too close to zero; check pendulum parameters"
            )
        inv_m = (
            (mass[1][1] / determinant, -mass[0][1] / determinant),
            (-mass[1][0] / determinant, mass[0][0] / determinant),
        )
        return mass, inv_m

    def control_affine(
        self, state: DoublePendulumState
    ) -> tuple[tuple[float, ...], tuple[tuple[float, ...], ...]]:
        c1, c2 = self.coriolis_vector(state.theta2, state.omega1, state.omega2)
        g1, g2 = self.gravity_vector(state.theta1, state.theta2)
        d1, d2 = self.damping_vector(state.omega1, state.omega2)
        mass, inv_m = self._invert_mass_matrix(state.theta2)

        drift_acc1 = -(inv_m[0][0] * (c1 + g1 + d1) + inv_m[0][1] * (c2 + g2 + d2))
        drift_acc2 = -(inv_m[1][0] * (c1 + g1 + d1) + inv_m[1][1] * (c2 + g2 + d2))
        f = (state.omega1, state.omega2, drift_acc1, drift_acc2)

        g_matrix = (
            (inv_m[0][0], inv_m[0][1]),
            (inv_m[1][0], inv_m[1][1]),
        )
        control_matrix = (
            (0.0, 0.0),
            (0.0, 0.0),
            g_matrix[0],
            g_matrix[1],
        )
        return f, control_matrix

    def applied_torques(
        self, t: float, state: DoublePendulumState
    ) -> tuple[float, float]:
        tau1 = self.forcing_functions[0](t, state)
        tau2 = self.forcing_functions[1](t, state)
        return tau1, tau2

    def inverse_dynamics(
        self, state: DoublePendulumState, accelerations: tuple[float, float]
    ) -> tuple[float, float]:
        """Compute joint torques required to realize the provided accelerations."""

        c1, c2 = self.coriolis_vector(state.theta2, state.omega1, state.omega2)
        g1, g2 = self.gravity_vector(state.theta1, state.theta2)
        d1, d2 = self.damping_vector(state.omega1, state.omega2)
        mass, _ = self._invert_mass_matrix(state.theta2)

        acc1, acc2 = accelerations
        tau1 = mass[0][0] * acc1 + mass[0][1] * acc2 + c1 + g1 + d1
        tau2 = mass[1][0] * acc1 + mass[1][1] * acc2 + c2 + g2 + d2
        return tau1, tau2

    def joint_torque_breakdown(
        self, state: DoublePendulumState, control: tuple[float, float]
    ) -> JointTorques:
        c1, c2 = self.coriolis_vector(state.theta2, state.omega1, state.omega2)
        g1, g2 = self.gravity_vector(state.theta1, state.theta2)
        d1, d2 = self.damping_vector(state.omega1, state.omega2)
        return JointTorques(
            applied=control,
            gravitational=(g1, g2),
            damping=(d1, d2),
            coriolis_centripetal=(c1, c2),
        )

    def derivatives(
        self, t: float, state: DoublePendulumState
    ) -> tuple[float, float, float, float]:
        tau1, tau2 = self.applied_torques(t, state)
        c1, c2 = self.coriolis_vector(state.theta2, state.omega1, state.omega2)
        g1, g2 = self.gravity_vector(state.theta1, state.theta2)
        d1, d2 = self.damping_vector(state.omega1, state.omega2)
        _, inv_m = self._invert_mass_matrix(state.theta2)
        acc1 = inv_m[0][0] * (tau1 - c1 - g1 - d1) + inv_m[0][1] * (tau2 - c2 - g2 - d2)
        acc2 = inv_m[1][0] * (tau1 - c1 - g1 - d1) + inv_m[1][1] * (tau2 - c2 - g2 - d2)
        return state.omega1, state.omega2, acc1, acc2

    def step(
        self, t: float, state: DoublePendulumState, dt: float
    ) -> DoublePendulumState:
        def rk4_increment(
            current_state: DoublePendulumState, scale: float, derivs: Iterable[float]
        ) -> DoublePendulumState:
            dtheta1, dtheta2, domega1, domega2 = derivs
            # Preserve phi and omega_phi (out-of-plane motion not yet in dynamics)
            phi = getattr(current_state, "phi", 0.0)
            omega_phi = getattr(current_state, "omega_phi", 0.0)
            return DoublePendulumState(
                theta1=current_state.theta1 + scale * dtheta1,
                theta2=current_state.theta2 + scale * dtheta2,
                omega1=current_state.omega1 + scale * domega1,
                omega2=current_state.omega2 + scale * domega2,
                phi=phi,
                omega_phi=omega_phi,
            )

        k1 = self.derivatives(t, state)
        k2 = self.derivatives(t + dt / 2.0, rk4_increment(state, dt / 2.0, k1))
        k3 = self.derivatives(t + dt / 2.0, rk4_increment(state, dt / 2.0, k2))
        k4 = self.derivatives(t + dt, rk4_increment(state, dt, k3))

        new_theta1 = state.theta1 + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        new_theta2 = state.theta2 + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        new_omega1 = state.omega1 + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        new_omega2 = state.omega2 + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        # Preserve phi and omega_phi
        phi = getattr(state, "phi", 0.0)
        omega_phi = getattr(state, "omega_phi", 0.0)

        return DoublePendulumState(
            theta1=new_theta1,
            theta2=new_theta2,
            omega1=new_omega1,
            omega2=new_omega2,
            phi=phi,
            omega_phi=omega_phi,
        )


def compile_forcing_functions(
    shoulder_expression: str, wrist_expression: str
) -> tuple[
    Callable[[float, DoublePendulumState], float],
    Callable[[float, DoublePendulumState], float],
]:
    return ExpressionFunction(shoulder_expression), ExpressionFunction(wrist_expression)
