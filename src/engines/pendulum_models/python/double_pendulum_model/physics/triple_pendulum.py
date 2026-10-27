"""Triple-pendulum physics model: equations of motion and numerical integration."""

# ruff: noqa: PLR0913, ARG001, N803, TID253
from __future__ import annotations

import typing
from dataclasses import dataclass

import numpy as np

if typing.TYPE_CHECKING:
    import collections.abc

from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)

from ._rk4 import rk4_step

GRAVITATIONAL_ACCELERATION = TRIPLE_PENDULUM_SPEC.gravity_m_s2
DAMPING_DEFAULT = TRIPLE_PENDULUM_SPEC.damping_nm_s_rad


@dataclass
class TripleSegmentProperties:
    length_m: float
    mass_kg: float
    center_of_mass_ratio: float
    inertia_about_com: float

    @property
    def center_of_mass_distance(self) -> float:
        """Return distance from proximal joint to center of mass."""
        return self.length_m * self.center_of_mass_ratio

    @property
    def inertia_about_proximal_joint(self) -> float:
        """Compute inertia about the proximal joint via parallel axis."""
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
        """Create parameters with default three-segment properties."""
        segments = tuple(
            TripleSegmentProperties(
                length_m=segment.length_m,
                mass_kg=segment.mass_kg,
                center_of_mass_ratio=segment.center_of_mass_ratio,
                inertia_about_com=segment.inertia_about_com,
            )
            for segment in TRIPLE_PENDULUM_SPEC.segments
        )
        return cls((segments[0], segments[1], segments[2]))

    @property
    def gravity(self) -> float:
        """Return effective gravitational acceleration."""
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

    def __post_init__(self) -> None:
        # Build the polynomial and its derivative once instead of rebuilding a
        # fresh np.poly1d (and recomputing np.polyder) on every omega()/alpha()
        # call. Evaluation results are unchanged.
        self._poly = np.poly1d(self.coefficients)
        self._deriv = np.poly1d(np.polyder(self.coefficients))

    def omega(self, t: float) -> float:
        """Evaluate angular velocity polynomial at time t."""
        if not (t is not None):
            raise ValueError("t must be provided")
        return float(self._poly(t))

    def alpha(self, t: float) -> float:
        """Evaluate angular acceleration polynomial at time t."""
        if not (t is not None):
            raise ValueError("t must be provided")
        return float(self._deriv(t))


def _calc_mass_matrix(
    theta: tuple[float, float, float],
    omega: tuple[float, float, float],
    params: tuple[float, ...],
) -> np.ndarray:
    if theta is None:
        raise ValueError("theta must be provided")
    theta1, theta2, theta3 = theta
    omega1, omega2, omega3 = omega
    l1, l2, l3, lc1, lc2, lc3, m1, m2, m3, I1, I2, I3, g = params
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
    theta: tuple[float, float, float],
    omega: tuple[float, float, float],
    params: tuple[float, ...],
) -> np.ndarray:
    if theta is None:
        raise ValueError("theta must be provided")
    theta1, theta2, theta3 = theta
    omega1, omega2, omega3 = omega
    l1, l2, l3, lc1, lc2, lc3, m1, m2, m3, I1, I2, I3, g = params
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
    theta: tuple[float, float, float],
    params: tuple[float, ...],
) -> np.ndarray:
    if theta is None:
        raise ValueError("theta must be provided")
    theta1, theta2, theta3 = theta
    l1, l2, l3, lc1, lc2, lc3, m1, m2, m3, I1, I2, I3, g = params
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
        """Compute the 3x3 mass matrix for the current state."""
        if not (state is not None):
            raise ValueError("state must be provided")
        params = self._parameter_vector()
        theta = (state.theta1, state.theta2, state.theta3)
        omega = (state.omega1, state.omega2, state.omega3)
        mass = self._mass_func(theta, omega, params)
        return np.array(mass, dtype=float)

    def bias_vector(self, state: TriplePendulumState) -> np.ndarray:
        """Compute bias vector including Coriolis, gravity, and damping."""
        if not (state is not None):
            raise ValueError("state must be provided")
        params = self._parameter_vector()
        theta = (state.theta1, state.theta2, state.theta3)
        omega = (state.omega1, state.omega2, state.omega3)
        bias = self._bias_func(theta, omega, params)
        damping = np.array(self.parameters.damping, dtype=float) * np.array(
            omega, dtype=float
        )
        result = np.array(bias, dtype=float).flatten() + damping
        return np.array(result, dtype=np.float64)

    def forward_dynamics(
        self, state: TriplePendulumState, control: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Solve for joint accelerations given applied torques."""
        if not (state is not None):
            raise ValueError("state must be provided")
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        accelerations = np.linalg.solve(mass, np.array(control, dtype=float) - bias)
        return typing.cast(
            "tuple[float, float, float]", tuple(float(a) for a in accelerations)
        )

    def inverse_dynamics(
        self, state: TriplePendulumState, accelerations: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Compute torques required to produce the given accelerations."""
        if not (state is not None):
            raise ValueError("state must be provided")
        mass = self.mass_matrix(state)
        bias = self.bias_vector(state)
        torques = mass @ np.array(accelerations, dtype=float) + bias
        return typing.cast(
            "tuple[float, float, float]", tuple(float(t) for t in torques)
        )

    def joint_torque_breakdown(
        self, state: TriplePendulumState, control: tuple[float, float, float]
    ) -> TripleJointTorques:
        """Decompose joint torques into applied, gravity, damping, and Coriolis."""
        if not (state is not None):
            raise ValueError("state must be provided")
        theta = (state.theta1, state.theta2, state.theta3)
        params = self._parameter_vector()
        gravity_components = np.array(
            self._gravity_func(theta, params), dtype=float
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
        integrator: str = "rk4",
    ) -> TriplePendulumState:
        """Advance the state by one integration step."""
        if not (_t is not None):
            raise ValueError("_t must be provided")

        if integrator != "rk4":
            if integrator in ("semi_implicit", "semi_implicit_euler", "euler"):
                acc1, acc2, acc3 = self.forward_dynamics(state, control)
                w1, w2, w3 = (
                    state.omega1 + dt * acc1,
                    state.omega2 + dt * acc2,
                    state.omega3 + dt * acc3,
                )
                is_semi = integrator != "euler"
                return TriplePendulumState(
                    theta1=state.theta1 + dt * (w1 if is_semi else state.omega1),
                    theta2=state.theta2 + dt * (w2 if is_semi else state.omega2),
                    theta3=state.theta3 + dt * (w3 if is_semi else state.omega3),
                    omega1=w1,
                    omega2=w2,
                    omega3=w3,
                )
            raise ValueError(f"unknown integrator: {integrator!r}")

        def _derivs(_time: float, vec: tuple[float, ...]) -> tuple[float, ...]:
            """Velocity and acceleration derivatives for the state vector."""
            theta1, theta2, theta3, omega1, omega2, omega3 = vec
            current = TriplePendulumState(
                theta1=theta1,
                theta2=theta2,
                theta3=theta3,
                omega1=omega1,
                omega2=omega2,
                omega3=omega3,
            )
            acc1, acc2, acc3 = self.forward_dynamics(current, control)
            return (omega1, omega2, omega3, acc1, acc2, acc3)

        new_theta1, new_theta2, new_theta3, new_omega1, new_omega2, new_omega3 = (
            rk4_step(
                _derivs,
                (
                    state.theta1,
                    state.theta2,
                    state.theta3,
                    state.omega1,
                    state.omega2,
                    state.omega3,
                ),
                _t,
                dt,
            )
        )

        return TriplePendulumState(
            theta1=new_theta1,
            theta2=new_theta2,
            theta3=new_theta3,
            omega1=new_omega1,
            omega2=new_omega2,
            omega3=new_omega3,
        )
