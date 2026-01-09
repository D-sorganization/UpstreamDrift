# ruff: noqa: PLR0913, ARG001, N803, TID253
from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    import collections.abc

from shared.python.constants import GRAVITY_M_S2

GRAVITATIONAL_ACCELERATION = GRAVITY_M_S2
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
            mass_kg=0.35,
            # Derived from Shaft(0.15kg, com=0.43) + Head(0.20kg, com=1.0)
            center_of_mass_ratio=0.7557142857142858,
            # Composite inertia of shaft + head
            inertia_about_com=0.04034857142857143,
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


def _calc_mass_matrix(
    theta1: float,
    theta2: float,
    theta3: float,
    omega1: float,
    omega2: float,
    omega3: float,
    l1: float,
    l2: float,
    l3: float,
    lc1: float,
    lc2: float,
    lc3: float,
    m1: float,
    m2: float,
    m3: float,
    I1: float,
    I2: float,
    I3: float,
    g: float,
) -> np.ndarray:
    mass = np.zeros((3, 3))
    mass[0, 0] = (
        I1
        + I2
        + I3
        + lc1**2 * m1
        + m2 * (l1**2 + 2 * l1 * lc2 * np.cos(theta2) + lc2**2)
        + m3
        * (
            l1**2
            + 2 * l1 * l2 * np.cos(theta2)
            + 2 * l1 * lc3 * np.cos(theta2 + theta3)
            + l2**2
            + 2 * l2 * lc3 * np.cos(theta3)
            + lc3**2
        )
    )
    mass[0, 1] = (
        I2
        + I3
        + lc2 * m2 * (l1 * np.cos(theta2) + lc2)
        + m3
        * (
            l1 * l2 * np.cos(theta2)
            + l1 * lc3 * np.cos(theta2 + theta3)
            + l2**2
            + 2 * l2 * lc3 * np.cos(theta3)
            + lc3**2
        )
    )
    mass[0, 2] = I3 + lc3 * m3 * (
        l1 * np.cos(theta2 + theta3) + l2 * np.cos(theta3) + lc3
    )
    mass[1, 0] = (
        I2
        + I3
        + lc2 * m2 * (l1 * np.cos(theta2) + lc2)
        + m3
        * (
            l1 * l2 * np.cos(theta2)
            + l1 * lc3 * np.cos(theta2 + theta3)
            + l2**2
            + 2 * l2 * lc3 * np.cos(theta3)
            + lc3**2
        )
    )
    mass[1, 1] = (
        I2 + I3 + lc2**2 * m2 + m3 * (l2**2 + 2 * l2 * lc3 * np.cos(theta3) + lc3**2)
    )
    mass[1, 2] = I3 + lc3 * m3 * (l2 * np.cos(theta3) + lc3)
    mass[2, 0] = I3 + lc3 * m3 * (
        l1 * np.cos(theta2 + theta3) + l2 * np.cos(theta3) + lc3
    )
    mass[2, 1] = I3 + lc3 * m3 * (l2 * np.cos(theta3) + lc3)
    mass[2, 2] = I3 + lc3**2 * m3
    return mass


def _calc_bias_vector(
    theta1: float,
    theta2: float,
    theta3: float,
    omega1: float,
    omega2: float,
    omega3: float,
    l1: float,
    l2: float,
    l3: float,
    lc1: float,
    lc2: float,
    lc3: float,
    m1: float,
    m2: float,
    m3: float,
    I1: float,
    I2: float,
    I3: float,
    g: float,
) -> np.ndarray:
    bias = np.zeros((3,))
    bias[0] = (
        g * l1 * m2 * np.sin(theta1)
        + g * l1 * m3 * np.sin(theta1)
        + g * l2 * m3 * np.sin(theta1 + theta2)
        + g * lc1 * m1 * np.sin(theta1)
        + g * lc2 * m2 * np.sin(theta1 + theta2)
        + g * lc3 * m3 * np.sin(theta1 + theta2 + theta3)
        - 2 * l1 * l2 * m3 * omega1 * omega2 * np.sin(theta2)
        - l1 * l2 * m3 * omega2**2 * np.sin(theta2)
        - 2 * l1 * lc2 * m2 * omega1 * omega2 * np.sin(theta2)
        - l1 * lc2 * m2 * omega2**2 * np.sin(theta2)
        - 2 * l1 * lc3 * m3 * omega1 * omega2 * np.sin(theta2 + theta3)
        - 2 * l1 * lc3 * m3 * omega1 * omega3 * np.sin(theta2 + theta3)
        - l1 * lc3 * m3 * omega2**2 * np.sin(theta2 + theta3)
        - 2 * l1 * lc3 * m3 * omega2 * omega3 * np.sin(theta2 + theta3)
        - l1 * lc3 * m3 * omega3**2 * np.sin(theta2 + theta3)
        - 2 * l2 * lc3 * m3 * omega1 * omega3 * np.sin(theta3)
        - 2 * l2 * lc3 * m3 * omega2 * omega3 * np.sin(theta3)
        - l2 * lc3 * m3 * omega3**2 * np.sin(theta3)
    )
    bias[1] = (
        g * l2 * m3 * np.sin(theta1 + theta2)
        + g * lc2 * m2 * np.sin(theta1 + theta2)
        + g * lc3 * m3 * np.sin(theta1 + theta2 + theta3)
        + l1 * l2 * m3 * omega1**2 * np.sin(theta2)
        + l1 * lc2 * m2 * omega1**2 * np.sin(theta2)
        + l1 * lc3 * m3 * omega1**2 * np.sin(theta2 + theta3)
        - 2 * l2 * lc3 * m3 * omega1 * omega3 * np.sin(theta3)
        - 2 * l2 * lc3 * m3 * omega2 * omega3 * np.sin(theta3)
        - l2 * lc3 * m3 * omega3**2 * np.sin(theta3)
    )
    bias[2] = (
        lc3
        * m3
        * (
            g * np.sin(theta1 + theta2 + theta3)
            + l1 * omega1**2 * np.sin(theta2 + theta3)
            + l2 * omega1**2 * np.sin(theta3)
            + 2 * l2 * omega1 * omega2 * np.sin(theta3)
            + l2 * omega2**2 * np.sin(theta3)
        )
    )
    return bias


def _calc_gravity_vector(
    theta1: float,
    theta2: float,
    theta3: float,
    l1: float,
    l2: float,
    l3: float,
    lc1: float,
    lc2: float,
    lc3: float,
    m1: float,
    m2: float,
    m3: float,
    I1: float,
    I2: float,
    I3: float,
    g: float,
) -> np.ndarray:
    gravity = np.zeros((3,))
    gravity[0] = (
        g * l1 * m2 * np.sin(theta1)
        + g * l1 * m3 * np.sin(theta1)
        + g * l2 * m3 * np.sin(theta1 + theta2)
        + g * lc1 * m1 * np.sin(theta1)
        + g * lc2 * m2 * np.sin(theta1 + theta2)
        + g * lc3 * m3 * np.sin(theta1 + theta2 + theta3)
    )
    gravity[1] = (
        g * l2 * m3 * np.sin(theta1 + theta2)
        + g * lc2 * m2 * np.sin(theta1 + theta2)
        + g * lc3 * m3 * np.sin(theta1 + theta2 + theta3)
    )
    gravity[2] = g * lc3 * m3 * np.sin(theta1 + theta2 + theta3)
    return gravity


def _hardcoded_triple_functions() -> tuple[
    collections.abc.Callable[..., np.ndarray],
    collections.abc.Callable[..., np.ndarray],
    collections.abc.Callable[..., np.ndarray],
]:
    return _calc_mass_matrix, _calc_bias_vector, _calc_gravity_vector


class TriplePendulumDynamics:
    def __init__(self, parameters: TriplePendulumParameters | None = None) -> None:
        self.parameters = parameters or TriplePendulumParameters.default()
        self._mass_func, self._bias_func, self._gravity_func = (
            _hardcoded_triple_functions()
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
        result = np.array(bias, dtype=float).flatten() + damping
        return np.array(result, dtype=np.float64)

    def forward_dynamics(
        self, state: TriplePendulumState, control: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        accelerations = np.linalg.solve(mass, np.array(control, dtype=float) - bias)
        return typing.cast(
            "tuple[float, float, float]", tuple(float(a) for a in accelerations)
        )

    def inverse_dynamics(
        self, state: TriplePendulumState, accelerations: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        torques = mass @ np.array(accelerations, dtype=float) + bias
        return typing.cast(
            "tuple[float, float, float]", tuple(float(t) for t in torques)
        )

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
            gravitational=typing.cast(
                "tuple[float, float, float]",
                tuple(float(c) for c in gravity_components),
            ),
            damping=typing.cast("tuple[float, float, float]", damping_components),
            coriolis_centripetal=typing.cast(
                "tuple[float, float, float]", tuple(float(c) for c in coriolis_bias)
            ),
        )

    def step(
        self,
        _t: float,
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
