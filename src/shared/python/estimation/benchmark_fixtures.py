"""Shared deterministic benchmark fixtures for estimation and matching (#11422).

Provides fixed-base pendulum, underactuated analytic, and native stance fixtures
with exact truth derivation, strict SI units, and energy/contact invariants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from src.shared.python.core.contracts import check_finite, require


@dataclass(frozen=True)
class BenchmarkTrajectory:
    """Deterministic trajectory states with strict SI units."""

    t: np.ndarray  # (N,) seconds
    q: np.ndarray  # (N, D) rad or metres
    qdot: np.ndarray  # (N, D) rad/s or m/s
    qddot: np.ndarray  # (N, D) rad/s^2 or m/s^2

    def __post_init__(self) -> None:
        require(self.t.ndim == 1, "time must be 1D")
        require(self.q.ndim == 2, "q must be 2D (N, D)")
        require(self.qdot.ndim == 2, "qdot must be 2D (N, D)")
        require(self.qddot.ndim == 2, "qddot must be 2D (N, D)")
        n = len(self.t)
        require(len(self.q) == n, "q length mismatch")
        require(len(self.qdot) == n, "qdot length mismatch")
        require(len(self.qddot) == n, "qddot length mismatch")
        require(check_finite(self.t), "t must be finite")
        require(check_finite(self.q), "q must be finite")
        require(check_finite(self.qdot), "qdot must be finite")
        require(check_finite(self.qddot), "qddot must be finite")


@dataclass(frozen=True)
class BenchmarkFixture:
    """Benchmark problem fixture with dynamics, controls, and ground truth."""

    name: str
    trajectory: BenchmarkTrajectory
    controls: np.ndarray  # (N, M) N*m or N
    num_dofs: int
    actuated_mask: tuple[bool, ...]
    units: dict[str, str]
    forces_type: Any  # ForceMeasurementType
    contact_forces: np.ndarray  # (N, 3) [Fx, Fy, Fz] in Newtons
    friction_coefficient: float = 0.6
    parameters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require(self.num_dofs > 0, "num_dofs must be positive")
        require(
            len(self.actuated_mask) == self.num_dofs, "actuated_mask length mismatch"
        )
        require(self.controls.ndim == 2, "controls must be 2D (N, M)")
        require(
            len(self.controls) == len(self.trajectory.t), "controls length mismatch"
        )
        require(check_finite(self.controls), "controls must be finite")
        require(check_finite(self.contact_forces), "contact_forces must be finite")

    def total_energy(self) -> np.ndarray:
        """Calculate total mechanical energy (kinetic + potential) along trajectory."""
        m = float(self.parameters.get("mass_kg", 1.0))
        length_m = float(self.parameters.get("length_m", 1.0))
        g = float(self.parameters.get("gravity_mps2", 9.81))

        # For single DOF fixed-base pendulum:
        # Kinetic: T = 0.5 * m * length_m^2 * qdot^2
        # Potential: V = -m * g * length_m * cos(q) (zero at horizontal)
        q = self.trajectory.q[:, 0]
        qdot = self.trajectory.qdot[:, 0]
        kinetic = 0.5 * m * (length_m**2) * (qdot**2)
        potential = -m * g * length_m * np.cos(q)
        return kinetic + potential


_STANDARD_BENCHMARK_UNITS: dict[str, str] = {
    "position": "m",
    "angle": "rad",
    "torque": "N*m",
    "force": "N",
    "time": "s",
}


def _assemble_fixture(
    name: str,
    t_arr: np.ndarray,
    q_arr: np.ndarray,
    qdot_arr: np.ndarray,
    qddot_arr: np.ndarray,
    controls: np.ndarray,
    actuated_mask: tuple[bool, ...],
    contact_forces: np.ndarray,
    friction_coefficient: float = 0.6,
    parameters: dict[str, Any] | None = None,
) -> BenchmarkFixture:
    from src.shared.python.estimation.benchmark_manifest import ForceMeasurementType

    trajectory = BenchmarkTrajectory(
        t=t_arr,
        q=q_arr,
        qdot=qdot_arr,
        qddot=qddot_arr,
    )
    return BenchmarkFixture(
        name=name,
        trajectory=trajectory,
        controls=controls,
        num_dofs=len(actuated_mask),
        actuated_mask=actuated_mask,
        units=_STANDARD_BENCHMARK_UNITS,
        forces_type=ForceMeasurementType.MEASURED,
        contact_forces=contact_forces,
        friction_coefficient=friction_coefficient,
        parameters=parameters or {},
    )


def _rk4_step(
    f: Any,
    t: float,
    y: np.ndarray,
    dt: float,
    *args: Any,
) -> np.ndarray:
    k1 = f(t, y, *args)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1, *args)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2, *args)
    k4 = f(t + dt, y + dt * k3, *args)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def make_deterministic_pendulum_fixture(
    n_frames: int = 60,
    dt: float = 1.0 / 60.0,
    seed: int = 42,
) -> BenchmarkFixture:
    """Build a fixed-base conservative single pendulum fixture.

    Parameters: mass m=1.0 kg, length l=1.0 m, gravity g=9.81 m/s^2.
    Integrates via RK4 with substepping to ensure energy conservation.
    """
    require(n_frames >= 2, "n_frames must be >= 2")
    require(dt > 0.0, "dt must be positive")

    rng = np.random.default_rng(seed)
    q0 = float(rng.uniform(0.3, 0.7))
    qdot0 = 0.0

    m = 1.0
    length_m = 1.0
    g = 9.81

    def ode(_t: float, y: np.ndarray) -> np.ndarray:
        # y = [q, qdot]
        # dq/dt = qdot
        # dqdot/dt = -(g/length_m) * sin(q)
        q_val = y[0]
        qdot_val = y[1]
        qddot_val = -(g / length_m) * np.sin(q_val)
        return np.array([qdot_val, qddot_val], dtype=np.float64)

    t_arr = np.linspace(0.0, (n_frames - 1) * dt, n_frames)
    q_arr = np.zeros((n_frames, 1), dtype=np.float64)
    qdot_arr = np.zeros((n_frames, 1), dtype=np.float64)
    qddot_arr = np.zeros((n_frames, 1), dtype=np.float64)

    y = np.array([q0, qdot0], dtype=np.float64)
    substeps = 20
    sub_dt = dt / substeps

    for i in range(n_frames):
        q_arr[i, 0] = y[0]
        qdot_arr[i, 0] = y[1]
        qddot_arr[i, 0] = -(g / length_m) * np.sin(y[0])

        if i < n_frames - 1:
            for _ in range(substeps):
                y = _rk4_step(ode, 0.0, y, sub_dt)

    return _assemble_fixture(
        name="deterministic-fixed-base-pendulum",
        t_arr=t_arr,
        q_arr=q_arr,
        qdot_arr=qdot_arr,
        qddot_arr=qddot_arr,
        controls=np.zeros((n_frames, 1), dtype=np.float64),
        actuated_mask=(True,),
        contact_forces=np.zeros((n_frames, 3), dtype=np.float64),
        friction_coefficient=0.6,
        parameters={
            "mass_kg": m,
            "length_m": length_m,
            "gravity_mps2": g,
            "seed": seed,
        },
    )


def make_underactuated_analytic_fixture(
    n_frames: int = 60,
    dt: float = 1.0 / 60.0,
    seed: int = 42,
) -> BenchmarkFixture:
    """Build a 2-DOF underactuated analytic fixture with passive 2nd DOF."""
    from src.shared.python.estimation.benchmark_manifest import ForceMeasurementType

    require(n_frames >= 2, "n_frames must be >= 2")
    require(dt > 0.0, "dt must be positive")

    rng = np.random.default_rng(seed)
    q0 = np.array([float(rng.uniform(0.1, 0.4)), 0.0], dtype=np.float64)
    qdot0 = np.zeros(2, dtype=np.float64)

    t_arr = np.linspace(0.0, (n_frames - 1) * dt, n_frames)
    q_arr = np.zeros((n_frames, 2), dtype=np.float64)
    qdot_arr = np.zeros((n_frames, 2), dtype=np.float64)
    qddot_arr = np.zeros((n_frames, 2), dtype=np.float64)

    # Simplified 2-DOF planar model where joint 1 is driven and joint 2 oscillates passively
    omega1 = 1.5
    omega2 = 3.0
    for i, t_val in enumerate(t_arr):
        q_arr[i, 0] = q0[0] * np.cos(omega1 * t_val)
        q_arr[i, 1] = 0.2 * np.sin(omega2 * t_val)
        qdot_arr[i, 0] = -q0[0] * omega1 * np.sin(omega1 * t_val)
        qdot_arr[i, 1] = 0.2 * omega2 * np.cos(omega2 * t_val)
        qddot_arr[i, 0] = -q0[0] * (omega1**2) * np.cos(omega1 * t_val)
        qddot_arr[i, 1] = -0.2 * (omega2**2) * np.sin(omega2 * t_val)

    # Passive joint control torque is strictly zero
    controls = np.zeros((n_frames, 2), dtype=np.float64)
    controls[:, 0] = 0.5 * np.cos(omega1 * t_arr)  # Actuated joint 1
    controls[:, 1] = 0.0  # Passive joint 2

    return _assemble_fixture(
        name="underactuated-analytic-fixture",
        t_arr=t_arr,
        q_arr=q_arr,
        qdot_arr=qdot_arr,
        qddot_arr=qddot_arr,
        controls=controls,
        actuated_mask=(True, False),  # Joint 2 is passive!
        contact_forces=np.zeros((n_frames, 3), dtype=np.float64),
        friction_coefficient=0.6,
        parameters={"seed": seed},
    )


def make_native_stance_fixture(
    n_frames: int = 30,
    dt: float = 1.0 / 60.0,
    seed: int = 42,
) -> BenchmarkFixture:
    """Build a native stance fixture with ground contact forces and friction cone."""
    require(n_frames >= 2, "n_frames must be >= 2")
    require(dt > 0.0, "dt must be positive")

    t_arr = np.linspace(0.0, (n_frames - 1) * dt, n_frames)
    q_arr = np.zeros((n_frames, 3), dtype=np.float64)
    qdot_arr = np.zeros((n_frames, 3), dtype=np.float64)
    qddot_arr = np.zeros((n_frames, 3), dtype=np.float64)

    # Stance subject mass = 75 kg, normal force ~ 735 N
    body_weight_n = 75.0 * 9.81
    mu = 0.6

    contact_forces = np.zeros((n_frames, 3), dtype=np.float64)
    contact_forces[:, 2] = body_weight_n  # F_z >= 0
    # Tangential forces within friction cone
    contact_forces[:, 0] = 0.2 * body_weight_n * np.sin(2.0 * np.pi * t_arr)
    contact_forces[:, 1] = 0.1 * body_weight_n * np.cos(2.0 * np.pi * t_arr)

    return _assemble_fixture(
        name="native-stance-fixture",
        t_arr=t_arr,
        q_arr=q_arr,
        qdot_arr=qdot_arr,
        qddot_arr=qddot_arr,
        controls=np.zeros((n_frames, 3), dtype=np.float64),
        actuated_mask=(True, True, True),
        contact_forces=contact_forces,
        friction_coefficient=mu,
        parameters={"body_weight_n": body_weight_n, "seed": seed},
    )
