from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import sympy as sp

GRAVITATIONAL_ACCELERATION = 9.80665
DAMPING_DEFAULT = (0.35, 0.3, 0.25)


@dataclass
class TripleSegmentProperties:
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
class TriplePendulumParameters:
    segments: tuple[
        TripleSegmentProperties, TripleSegmentProperties, TripleSegmentProperties
    ]
    damping: tuple[float, float, float] = DAMPING_DEFAULT
    gravity_enabled: bool = True
    gravity_m_s2: float = GRAVITATIONAL_ACCELERATION

    @classmethod
    def default(cls) -> TriplePendulumParameters:
        seg1 = TripleSegmentProperties(
            length_m=0.75,
            mass_kg=7.5,
            center_of_mass_ratio=0.45,
            inertia_about_com=(1.0 / 12.0) * 7.5 * 0.75**2,
        )
        seg2 = TripleSegmentProperties(
            length_m=0.6,
            mass_kg=1.8,
            center_of_mass_ratio=0.46,
            inertia_about_com=(1.0 / 12.0) * 1.8 * 0.6**2,
        )
        seg3 = TripleSegmentProperties(
            length_m=1.0,
            mass_kg=0.55,
            center_of_mass_ratio=0.43,
            inertia_about_com=(1.0 / 12.0) * 0.55 * 1.0**2,
        )
        return cls((seg1, seg2, seg3))

    @property
    def gravity(self) -> float:
        return self.gravity_m_s2 if self.gravity_enabled else 0.0


@dataclass
class TriplePendulumState:
    theta1: float
    theta2: float
    theta3: float
    omega1: float
    omega2: float
    omega3: float


@dataclass
class TripleJointTorques:
    applied: tuple[float, float, float]
    gravitational: tuple[float, float, float]
    damping: tuple[float, float, float]
    coriolis_centripetal: tuple[float, float, float]


@dataclass
class PolynomialProfile:
    coefficients: tuple[float, ...]

    def omega(self, t: float) -> float:
        poly = np.poly1d(self.coefficients)
        return float(poly(t))

    def alpha(self, t: float) -> float:
        derivative = np.polyder(self.coefficients)
        return float(np.poly1d(derivative)(t))


@functools.lru_cache(maxsize=1)
def _symbolic_triple_functions() -> (
    tuple[
        Callable[..., np.ndarray], Callable[..., np.ndarray], Callable[..., np.ndarray]
    ]
):
    theta1, theta2, theta3 = sp.symbols("theta1 theta2 theta3")
    omega1, omega2, omega3 = sp.symbols("omega1 omega2 omega3")
    alpha1, alpha2, alpha3 = sp.symbols("alpha1 alpha2 alpha3")

    l1, l2, l3 = sp.symbols("l1 l2 l3")
    lc1, lc2, lc3 = sp.symbols("lc1 lc2 lc3")
    m1, m2, m3 = sp.symbols("m1 m2 m3")
    i1_sym, i2_sym, i3_sym = sp.symbols("I1 I2 I3")
    g = sp.symbols("g")

    q = sp.Matrix([theta1, theta2, theta3])
    qd = sp.Matrix([omega1, omega2, omega3])
    qdd = sp.Matrix([alpha1, alpha2, alpha3])

    phi1 = theta1
    phi2 = theta1 + theta2
    phi3 = theta1 + theta2 + theta3

    x1 = lc1 * sp.sin(phi1)
    y1 = lc1 * sp.cos(phi1)

    x2 = l1 * sp.sin(phi1) + lc2 * sp.sin(phi2)
    y2 = l1 * sp.cos(phi1) + lc2 * sp.cos(phi2)

    x3 = l1 * sp.sin(phi1) + l2 * sp.sin(phi2) + lc3 * sp.sin(phi3)
    y3 = l1 * sp.cos(phi1) + l2 * sp.cos(phi2) + lc3 * sp.cos(phi3)

    vx1 = sp.diff(x1, theta1) * omega1
    vy1 = sp.diff(y1, theta1) * omega1

    vx2 = x2.diff(theta1) * omega1 + x2.diff(theta2) * omega2
    vy2 = y2.diff(theta1) * omega1 + y2.diff(theta2) * omega2

    vx3 = x3.diff(theta1) * omega1 + x3.diff(theta2) * omega2 + x3.diff(theta3) * omega3
    vy3 = y3.diff(theta1) * omega1 + y3.diff(theta2) * omega2 + y3.diff(theta3) * omega3

    kinetic_energy = (
        sp.Rational(1, 2) * m1 * (vx1**2 + vy1**2)
        + sp.Rational(1, 2) * i1_sym * omega1**2
        + sp.Rational(1, 2) * m2 * (vx2**2 + vy2**2)
        + sp.Rational(1, 2) * i2_sym * (omega1 + omega2) ** 2
        + sp.Rational(1, 2) * m3 * (vx3**2 + vy3**2)
        + sp.Rational(1, 2) * i3_sym * (omega1 + omega2 + omega3) ** 2
    )

    potential_energy = m1 * g * y1 + m2 * g * y2 + m3 * g * y3
    lagrangian = kinetic_energy - potential_energy

    tau = []
    for i in range(3):
        generalized_velocity_term = sp.diff(lagrangian, qd[i])
        time_derivative_term = (
            sp.diff(generalized_velocity_term, theta1) * omega1
            + sp.diff(generalized_velocity_term, theta2) * omega2
            + sp.diff(generalized_velocity_term, theta3) * omega3
        )
        time_derivative_term += (
            sp.diff(generalized_velocity_term, omega1) * qdd[0]
            + sp.diff(generalized_velocity_term, omega2) * qdd[1]
            + sp.diff(generalized_velocity_term, omega3) * qdd[2]
        )
        generalized_coordinate_term = sp.diff(lagrangian, q[i])
        tau.append(time_derivative_term - generalized_coordinate_term)

    tau_vec = sp.Matrix(tau)
    mass_matrix_sym = tau_vec.jacobian(qdd)
    bias = sp.simplify(tau_vec.subs({alpha1: 0, alpha2: 0, alpha3: 0}))

    symbols = (
        theta1,
        theta2,
        theta3,
        omega1,
        omega2,
        omega3,
        l1,
        l2,
        l3,
        lc1,
        lc2,
        lc3,
        m1,
        m2,
        m3,
        i1_sym,
        i2_sym,
        i3_sym,
        g,
    )

    mass_func = sp.lambdify(symbols[:6] + symbols[6:], mass_matrix_sym, "numpy")
    bias_func = sp.lambdify(symbols[:6] + symbols[6:], bias, "numpy")
    gravity_func = sp.lambdify(
        (
            theta1,
            theta2,
            theta3,
            l1,
            l2,
            l3,
            lc1,
            lc2,
            lc3,
            m1,
            m2,
            m3,
            i1_sym,
            i2_sym,
            i3_sym,
            g,
        ),
        bias.subs({omega1: 0, omega2: 0, omega3: 0}),
        "numpy",
    )
    return mass_func, bias_func, gravity_func


class TriplePendulumDynamics:
    def __init__(self, parameters: TriplePendulumParameters | None = None) -> None:
        self.parameters = parameters or TriplePendulumParameters.default()
        self._mass_func, self._bias_func, self._gravity_func = (
            _symbolic_triple_functions()
        )

    def _parameter_vector(self) -> tuple[float, ...]:
        segs = self.parameters.segments
        return (
            segs[0].length_m,
            segs[1].length_m,
            segs[2].length_m,
            segs[0].center_of_mass_distance,
            segs[1].center_of_mass_distance,
            segs[2].center_of_mass_distance,
            segs[0].mass_kg,
            segs[1].mass_kg,
            segs[2].mass_kg,
            segs[0].inertia_about_com,
            segs[1].inertia_about_com,
            segs[2].inertia_about_com,
            self.parameters.gravity,
        )

    def mass_matrix(self, state: TriplePendulumState) -> np.ndarray:
        params = self._parameter_vector()
        theta = (state.theta1, state.theta2, state.theta3)
        omega = (state.omega1, state.omega2, state.omega3)
        mass = self._mass_func(*theta, *omega, *params)
        return np.array(mass, dtype=float)

    def bias_vector(self, state: TriplePendulumState) -> np.ndarray:
        params = self._parameter_vector()
        theta = (state.theta1, state.theta2, state.theta3)
        omega = (state.omega1, state.omega2, state.omega3)
        bias = self._bias_func(*theta, *omega, *params)
        damping = np.array(self.parameters.damping, dtype=float) * np.array(
            omega, dtype=float
        )
        return np.array(bias, dtype=float).flatten() + damping

    def forward_dynamics(
        self, state: TriplePendulumState, control: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        accelerations = np.linalg.solve(mass, np.array(control, dtype=float) - bias)
        return tuple(float(a) for a in accelerations)  # type: ignore[return-value]

    def inverse_dynamics(
        self, state: TriplePendulumState, accelerations: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        torques = mass @ np.array(accelerations, dtype=float) + bias
        return tuple(float(t) for t in torques)  # type: ignore[return-value]

    def joint_torque_breakdown(
        self, state: TriplePendulumState, control: tuple[float, float, float]
    ) -> TripleJointTorques:
        theta = (state.theta1, state.theta2, state.theta3)
        params = self._parameter_vector()
        gravity_components = np.array(
            self._gravity_func(*theta, *params), dtype=float
        ).flatten()
        damping_components = tuple(
            float(self.parameters.damping[i] * state_component)
            for i, state_component in enumerate(
                (state.omega1, state.omega2, state.omega3)
            )
        )
        coriolis_bias = (
            self.bias_vector(state)
            - gravity_components
            - np.array(damping_components, dtype=float)
        )
        return TripleJointTorques(
            applied=control,
            gravitational=tuple(float(c) for c in gravity_components),  # type: ignore[arg-type]
            damping=damping_components,  # type: ignore[arg-type]
            coriolis_centripetal=tuple(float(c) for c in coriolis_bias),  # type: ignore[arg-type]
        )

    def step(
        self,
        t: float,
        state: TriplePendulumState,
        dt: float,
        control: tuple[float, float, float],
    ) -> TriplePendulumState:
        def rk4_increment(
            current_state: TriplePendulumState,
            scale: float,
            derivs: tuple[float, float, float, float, float, float],
        ) -> TriplePendulumState:
            dtheta1, dtheta2, dtheta3, domega1, domega2, domega3 = derivs
            return TriplePendulumState(
                theta1=current_state.theta1 + scale * dtheta1,
                theta2=current_state.theta2 + scale * dtheta2,
                theta3=current_state.theta3 + scale * dtheta3,
                omega1=current_state.omega1 + scale * domega1,
                omega2=current_state.omega2 + scale * domega2,
                omega3=current_state.omega3 + scale * domega3,
            )

        def derivatives(
            current_state: TriplePendulumState,
        ) -> tuple[float, float, float, float, float, float]:
            acc1, acc2, acc3 = self.forward_dynamics(current_state, control)
            return (
                current_state.omega1,
                current_state.omega2,
                current_state.omega3,
                acc1,
                acc2,
                acc3,
            )

        k1 = derivatives(state)
        k2 = derivatives(rk4_increment(state, dt / 2.0, k1))
        k3 = derivatives(rk4_increment(state, dt / 2.0, k2))
        k4 = derivatives(rk4_increment(state, dt, k3))

        new_theta1 = state.theta1 + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        new_theta2 = state.theta2 + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        new_theta3 = state.theta3 + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])

        new_omega1 = state.omega1 + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        new_omega2 = state.omega2 + dt / 6.0 * (k1[4] + 2 * k2[4] + 2 * k3[4] + k4[4])
        new_omega3 = state.omega3 + dt / 6.0 * (k1[5] + 2 * k2[5] + 2 * k3[5] + k4[5])

        return TriplePendulumState(
            theta1=new_theta1,
            theta2=new_theta2,
            theta3=new_theta3,
            omega1=new_omega1,
            omega2=new_omega2,
            omega3=new_omega3,
        )
