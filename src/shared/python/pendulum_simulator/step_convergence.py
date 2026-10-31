"""Shared per-engine step-size convergence study harness and exact switch-time handling (SAP-3, #13792).

Design by Contract
------------------
- Pre: Breakpoints in aligned mode must align with grid dt (fail closed).
- Pre: Valid finite physical parameters and positive step sizes.
- Post: ConvergenceReceipt records engine, integrator, dt, substeps, interpolation mode,
  reference tolerance, errors, and observed convergence order.

Law of Demeter & DRY
--------------------
- Single shared convergence harness drives any engine conforming to TriplePendulumEngineAdapter.
- Exact switch time logic is centralized in BreakpointTorqueProfile.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np

from double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)
from src.shared.python.pendulum_simulator.physics_base import (
    kinetic_energy_from_M,
    total_energy_from_parts,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)

#: Shoulder pivot height (m) of the canonical model, matching the MuJoCo XML.
PIVOT_HEIGHT_M = 1.2
_SEGMENTS = TRIPLE_PENDULUM_SPEC.segments
_LENGTHS_M = np.array([seg.length_m for seg in _SEGMENTS])
_MASSES_KG = np.array([seg.mass_kg for seg in _SEGMENTS])
_COM_RATIOS = np.array([seg.center_of_mass_ratio for seg in _SEGMENTS])


@dataclass(frozen=True)
class BreakpointTorqueProfile:
    """Torque profile with exact switch breakpoints and alignment enforcement."""

    initial_torque: tuple[float, float, float]
    breakpoints: tuple[tuple[float, tuple[float, float, float]], ...] = ()
    interpolation_mode: str = "zoh"

    def __post_init__(self) -> None:
        if self.interpolation_mode not in {"zoh", "linear"}:
            raise ValueError(f"unknown interpolation_mode: {self.interpolation_mode!r}")
        for t_bp, _ in self.breakpoints:
            if not math.isfinite(t_bp) or t_bp < 0:
                raise ValueError("breakpoint time must be finite and non-negative")

    def evaluate(self, t: float) -> tuple[float, float, float]:
        """Evaluate continuous torque at exact time t."""
        current_torque = self.initial_torque
        for t_bp, next_torque in sorted(self.breakpoints, key=lambda x: x[0]):
            if t >= t_bp:
                current_torque = next_torque
            else:
                break
        return current_torque

    def validate_grid_alignment(
        self, dt: float, mode: str = "aligned", tol: float = 1e-9
    ) -> None:
        """Validate that all breakpoints align with the step grid in aligned mode."""
        if mode == "aligned":
            for t_bp, _ in self.breakpoints:
                steps = round(t_bp / dt)
                if abs(t_bp - steps * dt) > tol:
                    raise ValueError(
                        f"Breakpoint {t_bp} is not aligned to grid dt={dt} (mode={mode})"
                    )

    def plan_step(
        self, t_current: float, dt: float, mode: str = "event_located"
    ) -> list[tuple[float, float, tuple[float, float, float]]]:
        """Plan step sub-intervals for exact breakpoint handling."""
        if mode == "aligned":
            self.validate_grid_alignment(dt, mode="aligned")
            return [(t_current, dt, self.evaluate(t_current))]

        if mode == "event_located":
            t_end = t_current + dt
            bps = [
                t_bp
                for t_bp, _ in sorted(self.breakpoints, key=lambda x: x[0])
                if t_current < t_bp < t_end
            ]
            if not bps:
                return [(t_current, dt, self.evaluate(t_current))]

            segments: list[tuple[float, float, tuple[float, float, float]]] = []
            t_sub = t_current
            for bp in bps:
                sub_dt = bp - t_sub
                if sub_dt > 1e-12:
                    segments.append((t_sub, sub_dt, self.evaluate(t_sub)))
                t_sub = bp
            if t_end - t_sub > 1e-12:
                segments.append((t_sub, t_end - t_sub, self.evaluate(t_sub)))
            return segments

        raise ValueError(f"unknown switch mode: {mode!r}")


@runtime_checkable
class TriplePendulumEngineAdapter(Protocol):
    """Public step API protocol for triple pendulum engines."""

    @property
    def engine_name(self) -> str: ...

    def reset(
        self,
        initial_angles: tuple[float, float, float] = (-1.5, 0.5, 0.2),
        initial_velocities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None: ...

    def configure(self, integrator: str, dt: float) -> None: ...

    def step(
        self, t: float, dt: float, control: tuple[float, float, float]
    ) -> None: ...

    def tip_position(self) -> np.ndarray: ...


class InHouseTripleAdapter:
    """Adapter driving in-house TriplePendulumDynamics through public step API."""

    def __init__(self, dynamics: TriplePendulumDynamics | None = None) -> None:
        self.dynamics = dynamics or TriplePendulumDynamics()
        self.state = TriplePendulumState(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self.integrator = "rk4"
        self.dt = 0.001

    @property
    def engine_name(self) -> str:
        return "in_house_triple"

    def reset(
        self,
        initial_angles: tuple[float, float, float] = (-1.5, 0.5, 0.2),
        initial_velocities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.state = TriplePendulumState(*initial_angles, *initial_velocities)

    def configure(self, integrator: str, dt: float) -> None:
        self.integrator = integrator
        self.dt = dt

    def step(self, t: float, dt: float, control: tuple[float, float, float]) -> None:
        self.state = self.dynamics.step(
            t, self.state, dt, control, integrator=self.integrator
        )

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Relative joint angles (rad) and rates (rad/s)."""
        s = self.state
        q = np.array([s.theta1, s.theta2, s.theta3], dtype=np.float64)
        return q, np.array([s.omega1, s.omega2, s.omega3], dtype=np.float64)

    def tip_position(self) -> np.ndarray:
        angles, _ = self._absolute()
        tip_x = -float(np.sum(_LENGTHS_M * np.sin(angles)))
        tip_z = PIVOT_HEIGHT_M - float(np.sum(_LENGTHS_M * np.cos(angles)))
        return np.array([tip_x, 0.0, tip_z], dtype=np.float64)

    def tip_velocity(self) -> np.ndarray:
        """World-frame tip velocity (m/s), the time derivative of tip_position."""
        angles, rates = self._absolute()
        vx = -float(np.sum(_LENGTHS_M * np.cos(angles) * rates))
        vz = float(np.sum(_LENGTHS_M * np.sin(angles) * rates))
        return np.array([vx, 0.0, vz], dtype=np.float64)

    def total_energy(self) -> float:
        """Kinetic plus gravitational energy (J), world z = 0 datum as in MuJoCo."""
        _, v = self.joint_state()
        kinetic = kinetic_energy_from_M(self.dynamics.mass_matrix(self.state), v)
        angles, _ = self._absolute()
        joint_z = PIVOT_HEIGHT_M - np.concatenate(
            ([0.0], np.cumsum(_LENGTHS_M * np.cos(angles))[:-1])
        )
        com_z = joint_z - _COM_RATIOS * _LENGTHS_M * np.cos(angles)
        g = TRIPLE_PENDULUM_SPEC.gravity_m_s2
        return total_energy_from_parts(kinetic, float(np.sum(_MASSES_KG * g * com_z)))

    def _absolute(self) -> tuple[np.ndarray, np.ndarray]:
        q, v = self.joint_state()
        return np.cumsum(q), np.cumsum(v)


class MuJoCoTripleAdapter:
    """Adapter driving MuJoCo canonical 3DOF model through public step API."""

    def __init__(self) -> None:
        import mujoco

        from src.engines.physics_engines.mujoco.pendulum_models_xml import (
            TRIPLE_PENDULUM_XML,
        )

        self._mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(TRIPLE_PENDULUM_XML)
        self.data = mujoco.MjData(self.model)
        self._head_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "club_head"
        )
        self.dt = 0.001

    @property
    def engine_name(self) -> str:
        return "mujoco_triple"

    def reset(
        self,
        initial_angles: tuple[float, float, float] = (-1.5, 0.5, 0.2),
        initial_velocities: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.data.qpos[:] = initial_angles
        self.data.qvel[:] = initial_velocities
        self.data.ctrl[:] = (0.0, 0.0, 0.0)
        self._mujoco.mj_forward(self.model, self.data)

    def configure(self, integrator: str, dt: float) -> None:
        opt = self.model.opt
        types = self._mujoco.mjtIntegrator
        if integrator == "rk4":
            opt.integrator = types.mjINT_RK4
        elif integrator in ("euler", "semi_implicit", "semi_implicit_euler"):
            opt.integrator = types.mjINT_EULER
        else:
            raise ValueError(f"unknown mujoco integrator: {integrator!r}")
        opt.timestep = dt
        self.dt = dt

    def step(self, t: float, dt: float, control: tuple[float, float, float]) -> None:
        self.data.ctrl[:] = control
        opt = self.model.opt
        opt.timestep = dt
        self._mujoco.mj_step(self.model, self.data)

    def tip_position(self) -> np.ndarray:
        return np.array(self.data.geom_xpos[self._head_geom_id], dtype=np.float64)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Relative joint angles (rad) and rates (rad/s)."""
        return (
            np.array(self.data.qpos, dtype=np.float64),
            np.array(self.data.qvel, dtype=np.float64),
        )

    def tip_velocity(self) -> np.ndarray:
        """World-frame linear velocity (m/s) of the club-head geom."""
        self._mujoco.mj_forward(self.model, self.data)
        res = np.zeros(6)
        self._mujoco.mj_objectVelocity(
            self.model,
            self.data,
            self._mujoco.mjtObj.mjOBJ_GEOM,
            self._head_geom_id,
            res,
            0,
        )
        return res[3:].copy()

    def total_energy(self) -> float:
        """MuJoCo's own kinetic plus potential energy (J)."""
        self._mujoco.mj_forward(self.model, self.data)
        self._mujoco.mj_energyPos(self.model, self.data)
        self._mujoco.mj_energyVel(self.model, self.data)
        return float(np.sum(self.data.energy))


@dataclass(frozen=True)
class ConvergenceReceipt:
    """Structured convergence study evidence receipt."""

    engine: str
    integrator: str
    dts_s: tuple[float, ...]
    substeps: int
    interpolation_mode: str
    reference_tolerance: float
    observed_order: float
    errors_m: tuple[float, ...]
    tip_error_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "engine": self.engine,
            "integrator": self.integrator,
            "dts_s": list(self.dts_s),
            "substeps": self.substeps,
            "interpolation_mode": self.interpolation_mode,
            "reference_tolerance": self.reference_tolerance,
            "observed_order": self.observed_order,
            "errors_m": list(self.errors_m),
            "tip_error_m": self.tip_error_m,
            "metadata": self.metadata,
        }


def run_step_convergence_study(
    engine: TriplePendulumEngineAdapter,
    timesteps: Sequence[float] = (0.004, 0.002, 0.001, 0.0005),
    integrator: str = "rk4",
    t_end: float = 0.2,
    torque_profile: BreakpointTorqueProfile | None = None,
    interpolation_mode: str = "zoh",
    reference_dt: float = 2.5e-5,
    substeps: int = 1,
) -> ConvergenceReceipt:
    """Run step-size convergence study driving engine through public step API."""
    profile = torque_profile or BreakpointTorqueProfile(
        initial_torque=(80.0, 30.0, 10.0),
        interpolation_mode=interpolation_mode,
    )

    def _execute_run(dt_step: float, integ: str) -> np.ndarray:
        engine.reset()
        engine.configure(integ, dt_step / substeps)
        n_steps = int(round(t_end / dt_step))
        for k in range(n_steps):
            t_base = k * dt_step
            segments = profile.plan_step(
                t_base, dt_step, mode="aligned" if substeps == 1 else "event_located"
            )
            for t_sub, dt_sub, ctrl in segments:
                sub_dt = dt_sub / substeps
                for s in range(substeps):
                    t_eval = t_sub + s * sub_dt
                    engine.step(t_eval, sub_dt, ctrl)
        return engine.tip_position()

    ref_tip = _execute_run(reference_dt, "rk4")
    errors = [
        float(np.linalg.norm(_execute_run(dt, integrator) - ref_tip))
        for dt in timesteps
    ]
    observed_order, _ = np.polyfit(np.log(timesteps), np.log(errors), 1)

    return ConvergenceReceipt(
        engine=engine.engine_name,
        integrator=integrator,
        dts_s=tuple(float(dt) for dt in timesteps),
        substeps=substeps,
        interpolation_mode=profile.interpolation_mode,
        reference_tolerance=reference_dt,
        observed_order=float(observed_order),
        errors_m=tuple(errors),
        tip_error_m=errors[0],
        metadata={"t_end_s": t_end, "nominal_torque": profile.initial_torque},
    )
