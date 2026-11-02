"""Shared-core ``PhysicsEngine`` adapter for the Putting Green engine.

#14141 step 10a. Thin wrapper over ``src.shared.python.putting_dynamics``
(the mode machine already used by green reading, the SG optimizer, and
the MCP putting tools, #14141 steps 7-9) so
:func:`src.engines.loaders.load_putting_green_engine` can hand out a
shared-core engine instead of the legacy, hand-rolled simulator, whose
duplicate physics a later #14141 step removes (ADR-0058).

``set_state(q, v)`` precomputes the *entire* putt once via
:func:`~src.shared.python.putting_dynamics.simulate_ball`, anchored at
the current time. ``step(dt)`` only advances time and samples that
trajectory (linear interpolation; secant velocity; held at rest past
the end). Dynamics queries (mass matrix, bias/gravity, drift, the
counterfactuals) are simplified 2-DOF point-mass forms: gravity from
``HeightField.gradient`` and deceleration from the shared core's
rolling-friction law (:data:`USGA_STIMP_ROLL_MODEL`) — a ``(q, v)``
pair has no slide/spin channel, so these queries assume the steady
rolling branch even though the sampled trajectory itself always runs
the full airborne/slide/roll mode machine.
"""

from __future__ import annotations

import bisect
import math

import numpy as np

from src.shared.python.contracts import require, require_finite, require_positive
from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG, GRAVITY_M_S2
from src.shared.python.engine_core.base_physics_engine import BasePhysicsEngine
from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint
from src.shared.python.putting_dynamics import (
    USGA_STIMP_ROLL_MODEL,
    BallState,
    SurfaceSpec,
    simulate_ball,
)
from src.shared.python.putting_dynamics.friction import rolling_mu
from src.shared.python.putting_dynamics.solver import DT_S
from src.shared.python.putting_dynamics.types import PuttResult, TrajectorySample

__all__ = ["SharedPuttingEngine"]

#: Engine-type identifier recorded on checkpoints (distinct from the
#: legacy simulator's own checkpoint provenance, #14141 step 3).
ENGINE_TYPE = "putting_green_shared"

_BALL_MASS_KG = float(GOLF_BALL_MASS_KG)
_G = float(GRAVITY_M_S2)


class SharedPuttingEngine(BasePhysicsEngine):
    """``PhysicsEngine`` adapter wrapping the shared putting-dynamics core."""

    def __init__(self, surface: SurfaceSpec | None = None) -> None:
        super().__init__()
        self._surface = surface or SurfaceSpec.flat_uniform()
        self._height = self._surface.height
        self._friction_field = self._surface.friction_field
        self.model = self._surface
        self.model_name_str = "putting_green"
        self._is_initialized = True
        self._q: np.ndarray = np.zeros(2)
        self._v: np.ndarray = np.zeros(2)
        self._v0: np.ndarray = np.zeros(2)
        self._time = 0.0
        self._traj_t0 = 0.0
        self._result: PuttResult | None = None
        self._sample_times: tuple[float, ...] = ()
        self._recompute_trajectory()

    @property
    def engine_type(self) -> str:
        """Checkpoint/identification type, distinct from the legacy engine."""
        return ENGINE_TYPE

    @property
    def ball_mass(self) -> float:
        """Ball mass [kg] (single source: ``physics_constants``)."""
        return _BALL_MASS_KG

    @property
    def roll_model(self) -> str:
        """Roll model backing every trajectory this engine samples."""
        return USGA_STIMP_ROLL_MODEL

    def get_capabilities(self) -> EngineCapabilities:
        """Report Putting Green (shared-core) engine capabilities."""
        return EngineCapabilities(
            engine_name="PuttingGreenShared",
            mass_matrix=CapabilityLevel.FULL,
            jacobian=CapabilityLevel.FULL,
            contact_forces=CapabilityLevel.PARTIAL,
            inverse_dynamics=CapabilityLevel.FULL,
            drift_acceleration=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            contact_step=CapabilityLevel.FULL,
            model_positioning=CapabilityLevel.FULL,
            measurements=CapabilityLevel.FULL,
        )

    def _load_from_path_impl(self, path: str) -> None:
        """No-op: configure via the ``surface`` constructor argument."""

    def _load_from_string_impl(self, content: str, extension: str | None) -> None:
        """No-op: configure via the ``surface`` constructor argument."""

    def load_from_path(self, path: str) -> None:
        """No-op: SharedPuttingEngine is analytic (no model file to load)."""

    def load_from_string(self, content: str, extension: str | None = None) -> None:
        """No-op: SharedPuttingEngine is analytic (no model content to load)."""

    def reset(self) -> None:
        """Reset to the origin at rest, time zero."""
        self._time = 0.0
        self.set_state(np.zeros(2), np.zeros(2))

    def step(self, dt: float | None = None) -> None:
        """Advance time by ``dt`` and sample the precomputed trajectory."""
        step_s = DT_S if dt is None else float(dt)
        require_finite(step_s, "dt")
        require_positive(step_s, "dt")
        self._time += step_s
        self._sample_trajectory()

    def forward(self) -> None:
        """No cached derived quantities to refresh; queries are on-demand."""

    def get_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the current ``(position, velocity)`` pair."""
        return self._q.copy(), self._v.copy()

    def set_state(self, q: np.ndarray, v: np.ndarray) -> None:
        """Set ``(position, velocity)`` and precompute the resulting putt."""
        if q is None or v is None:
            raise ValueError("q and v must be provided")
        q_arr = np.asarray(q, dtype=float)
        v_arr = np.asarray(v, dtype=float)
        require(
            len(q_arr) == 2 and len(v_arr) == 2,
            "state must be 2-DOF (x, y)",
            (q_arr, v_arr),
        )
        require_finite(q_arr, "q")
        require_finite(v_arr, "v")
        self._q = q_arr
        self._v = v_arr
        self._recompute_trajectory()

    def set_control(self, u: np.ndarray) -> None:
        """Apply an impulsive force over one default step, then re-plan."""
        if u is None:
            raise ValueError("u must be provided")
        u_arr = np.asarray(u, dtype=float)
        require(len(u_arr) == 2, "u must be 2-DOF (fx, fy)", u_arr)
        require_finite(u_arr, "u")
        new_v = self._v + (u_arr / self.ball_mass) * DT_S
        self.set_state(self._q, new_v)

    def get_time(self) -> float:
        """Current simulation time [s]."""
        return self._time

    def restore_checkpoint(self, checkpoint: StateCheckpoint) -> None:
        """Restore ``(q, v, t)`` and re-plan the trajectory from there."""
        if checkpoint is None:
            raise ValueError("checkpoint must be provided")
        self._time = checkpoint.timestamp
        self.set_state(checkpoint.get_q(), checkpoint.get_v())

    def compute_mass_matrix(self) -> np.ndarray:
        """Scalar ball mass on both diagonal entries."""
        return np.eye(2) * self.ball_mass

    def compute_bias_forces(self) -> np.ndarray:
        """Mass-scaled drift acceleration (friction + slope) at the current state."""
        return self.ball_mass * self.compute_drift_acceleration()

    def compute_gravity_forces(self) -> np.ndarray:
        """Mass-scaled slope acceleration at the current position."""
        return self.ball_mass * self._slope_accel(float(self._q[0]), float(self._q[1]))

    def compute_contact_forces(self) -> np.ndarray:
        """Green reaction on the ball, ``(fx, fy, fz)`` [N].

        Rolling friction in the plane plus the normal support ``m g``
        (small-slope form; a ``(q, v)`` state has no slide channel).
        """
        q_x, q_y = float(self._q[0]), float(self._q[1])
        friction = self.ball_mass * (
            self.compute_drift_acceleration() - self._slope_accel(q_x, q_y)
        )
        return np.array([friction[0], friction[1], self.ball_mass * _G])

    def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
        """Force required for ``qacc`` on a point mass: ``M @ qacc``."""
        if qacc is None:
            raise ValueError("qacc must be provided")
        return self.ball_mass * np.asarray(qacc, dtype=float)

    def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
        """Identity Jacobian for the ``"ball"`` body; ``None`` otherwise."""
        if body_name is None:
            raise ValueError("body_name must be provided")
        if body_name != "ball":
            return None
        return {"linear": np.eye(2), "angular": np.zeros((1, 2))}

    def compute_drift_acceleration(self) -> np.ndarray:
        """Passive acceleration (gravity + rolling friction) at the current state."""
        return self._rolling_drift_accel(
            float(self._q[0]), float(self._q[1]), float(self._v[0]), float(self._v[1])
        )

    def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
        """Point-mass response to an applied force: ``tau / mass``."""
        if tau is None:
            raise ValueError("tau must be provided")
        return np.asarray(tau, dtype=float) / self.ball_mass

    def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Zero-torque counterfactual: drift acceleration at ``(q, v)``."""
        if q is None or v is None:
            raise ValueError("q and v must be provided")
        return self._rolling_drift_accel(
            float(q[0]), float(q[1]), float(v[0]), float(v[1])
        )

    def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
        """Zero-velocity counterfactual: slope acceleration at ``q`` alone."""
        if q is None:
            raise ValueError("q must be provided")
        return self._slope_accel(float(q[0]), float(q[1]))

    def _recompute_trajectory(self) -> None:
        """Re-plan the whole putt from the current state via ``simulate_ball``."""
        self._v0 = self._v.copy()
        self._traj_t0 = self._time
        ball = BallState(
            x_m=float(self._q[0]),
            y_m=float(self._q[1]),
            vx_mps=float(self._v[0]),
            vy_mps=float(self._v[1]),
        )
        self._result = simulate_ball(ball, self._surface)
        self._sample_times = tuple(s.t_s for s in self._result.samples)
        self._sample_trajectory()

    def _sample_trajectory(self) -> None:
        """Interpolate ``(q, v)`` from the precomputed trajectory at ``self._time``."""
        if self._result is None or not self._sample_times:
            return
        samples = self._result.samples
        times = self._sample_times
        rel_t = self._time - self._traj_t0
        if rel_t <= times[0]:
            self._q = np.array([samples[0].x_m, samples[0].y_m])
            self._v = self._v0.copy()
            return
        if rel_t >= times[-1]:
            self._q = np.array([samples[-1].x_m, samples[-1].y_m])
            self._v = self._rest_velocity(samples)
            return
        idx = bisect.bisect_right(times, rel_t) - 1
        lo, hi = samples[idx], samples[idx + 1]
        span = hi.t_s - lo.t_s
        frac = (rel_t - lo.t_s) / span if span > 0.0 else 0.0
        self._q = np.array(
            [lo.x_m + frac * (hi.x_m - lo.x_m), lo.y_m + frac * (hi.y_m - lo.y_m)]
        )
        self._v = np.array(
            [
                (hi.x_m - lo.x_m) / span if span > 0.0 else 0.0,
                (hi.y_m - lo.y_m) / span if span > 0.0 else 0.0,
            ]
        )

    @staticmethod
    def _rest_velocity(samples: tuple[TrajectorySample, ...]) -> np.ndarray:
        """Secant velocity into the final sample; zero once truly at rest."""
        last = samples[-1]
        if last.speed_mps <= 0.0 or len(samples) < 2:
            return np.zeros(2)
        prev = samples[-2]
        span = last.t_s - prev.t_s
        if span <= 0.0:
            return np.zeros(2)
        return np.array([(last.x_m - prev.x_m) / span, (last.y_m - prev.y_m) / span])

    def _slope_accel(self, x: float, y: float) -> np.ndarray:
        """``-g * grad(height)`` at ``(x, y)`` (shared ``HeightField``)."""
        grad = self._height.gradient(x, y)
        return np.array([-_G * grad[0], -_G * grad[1]])

    def _rolling_drift_accel(
        self, x: float, y: float, vx: float, vy: float
    ) -> np.ndarray:
        """Gravity plus rolling-friction deceleration at ``(x, y, vx, vy)``."""
        accel = self._slope_accel(x, y)
        speed = math.hypot(vx, vy)
        if speed <= 0.0:
            return accel
        heading = math.atan2(vy, vx)
        roll_mult, _slide_mult = self._friction_field.multipliers(x, y)
        mu = rolling_mu(self._surface.friction, speed, roll_mult, heading)
        return accel + np.array([-mu * _G * vx / speed, -mu * _G * vy / speed])
