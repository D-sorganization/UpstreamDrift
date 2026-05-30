"""ODE reference backend — the analytical safety net.

``ODEBackend`` wraps the existing, well-tested analytical dynamics
(:class:`DoublePendulumDynamics`) behind the frozen
:class:`~simulation_backends.protocol.SimulationBackend` and
:class:`~simulation_backends.protocol.DynamicsProvider` Protocols.

Because it is a thin shim over the closed-form equations of motion, this backend
is the *ground truth* against which the MuJoCo CPU and GPU backends are
cross-validated (epic tasks M5/M7). It has no optional dependencies and always
runs on CPU.

State convention (shared by every backend)::

    q = [theta1, theta2]            # shoulder, wrist angles [rad]
    v = [omega1, omega2]            # angular velocities [rad/s]
    u = [tau_shoulder, tau_wrist]   # applied joint torques [N*m]

with ``nq == nv == nu == 2``. The analytical model also carries an out-of-plane
``phi``/``omega_phi`` pair, but the in-plane golf model holds them at zero, so
this backend never exposes them.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .protocol import BackendCapabilities, SimState, Trace

if TYPE_CHECKING:
    from .model_params import GolfModelParams

logger = get_logger(__name__)

#: Generalised-coordinate dimension of the in-plane golf double pendulum.
_DOF = 2


class ODEBackend:
    """Reference backend wrapping the analytical RK4 double-pendulum dynamics.

    Satisfies both :class:`~simulation_backends.protocol.SimulationBackend`
    (integration / stepping) and
    :class:`~simulation_backends.protocol.DynamicsProvider` (mass matrix and
    bias forces), giving an independent derivation of the equations of motion
    for cross-validating the MuJoCo backends.

    Args:
        params: Single-source-of-truth model parameters.
        dt: Default integration step [s] used by :meth:`step` when no override
            is supplied. Must be strictly positive.

    Raises:
        ValueError: If ``dt`` is not strictly positive or not finite.
    """

    def __init__(self, params: GolfModelParams, *, dt: float = 0.01) -> None:
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DoublePendulumDynamics,
            DoublePendulumState,
        )

        require(
            float(dt) > 0.0 and np.isfinite(dt),
            f"dt must be a positive, finite step size; got {dt!r}",
            value=dt,
        )
        self._params = params
        self._dt = float(dt)
        self._dyn = DoublePendulumDynamics(params.to_double_pendulum_parameters())
        self._state = DoublePendulumState(
            theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0
        )
        self._time = 0.0
        # Annotate as a general ndarray so later slice-copies (which mypy widens
        # to an any-rank shape under numpy's shape-typed stubs) remain compatible.
        self._u: np.ndarray = np.zeros(_DOF, dtype=float)
        # Wire the analytical forcing functions to read the live control vector
        # so that the RK4 integrator applies the torques set via set_control.
        self._dyn.forcing_functions = (
            lambda _t, _s: float(self._u[0]),
            lambda _t, _s: float(self._u[1]),
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capability description (CPU, non-batched, dynamics provider)."""
        return BackendCapabilities(
            name="ode",
            device="cpu",
            supports_batched=False,
            is_differentiable=False,
            provides_dynamics=True,
        )

    def reset(self, state: SimState | None = None) -> None:
        """Reset the simulation to ``state`` (or the zero state if ``None``).

        Args:
            state: Target state. ``state.q`` / ``state.v`` must each have at
                least :data:`_DOF` entries; only the first two are used. If
                ``None``, all coordinates, velocities and the clock are zeroed.

        Postcondition:
            The control vector is cleared to zero and the clock is set to
            ``state.time`` (or ``0.0``).
        """
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DoublePendulumState,
        )

        if state is None:
            self._state = DoublePendulumState(
                theta1=0.0, theta2=0.0, omega1=0.0, omega2=0.0
            )
            self._time = 0.0
        else:
            require(
                state.q.size >= _DOF and state.v.size >= _DOF,
                f"reset state must carry at least {_DOF} coordinates; "
                f"got q={state.q.shape}, v={state.v.shape}",
                value=(state.q.shape, state.v.shape),
            )
            self._state = DoublePendulumState(
                theta1=float(state.q[0]),
                theta2=float(state.q[1]),
                omega1=float(state.v[0]),
                omega2=float(state.v[1]),
            )
            self._time = float(state.time)
        self._u = np.zeros(_DOF, dtype=float)

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one RK4 step.

        Args:
            dt: Step size [s]; defaults to the constructor ``dt`` when ``None``.
                Must be strictly positive and finite.

        Raises:
            ValueError: If an explicit ``dt`` is non-positive or not finite.
        """
        step_dt = self._dt if dt is None else float(dt)
        require(
            step_dt > 0.0 and np.isfinite(step_dt),
            f"step dt must be a positive, finite step size; got {dt!r}",
            value=dt,
        )
        self._state = self._dyn.step(self._time, self._state, step_dt)
        self._time += step_dt

    def get_state(self) -> SimState:
        """Return the current state as a :class:`SimState` (in-plane DOFs only)."""
        return SimState(
            q=np.array([self._state.theta1, self._state.theta2], dtype=float),
            v=np.array([self._state.omega1, self._state.omega2], dtype=float),
            time=self._time,
        )

    def set_control(self, u: np.ndarray) -> None:
        """Set the joint-torque vector applied during subsequent steps.

        Args:
            u: Control vector ``[tau_shoulder, tau_wrist]``; must have at least
                :data:`_DOF` finite entries. A private copy is stored.

        Raises:
            ValueError: If ``u`` has fewer than two entries or is non-finite.
        """
        u_arr = np.asarray(u, dtype=float).reshape(-1)
        require(
            u_arr.size >= _DOF,
            f"control vector must have at least {_DOF} entries; got {u_arr.size}",
            value=u_arr.size,
        )
        require(
            bool(np.all(np.isfinite(u_arr[:_DOF]))),
            "control vector must be finite",
            value=u,
        )
        self._u = u_arr[:_DOF].copy()

    def get_time(self) -> float:
        """Return the current simulation time [s]."""
        return self._time

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return joint accelerations ``qacc`` for state ``(q, v)`` under ``u``.

        Solves ``M(q) qacc = tau - bias(q, v)`` where ``bias`` is the sum of
        Coriolis/centripetal, gravity, and damping torques — matching the
        analytical equations of motion exactly.

        Args:
            q: Joint positions ``[theta1, theta2]`` (>= 2 entries used).
            v: Joint velocities ``[omega1, omega2]`` (>= 2 entries used).
            u: Applied torques ``[tau_shoulder, tau_wrist]``; ``None`` means
                passive (zero torque).

        Returns:
            Acceleration vector ``qacc`` of shape ``(2,)``.

        Raises:
            ValueError: If ``q`` or ``v`` (or a non-``None`` ``u``) has fewer
                than two entries.
        """
        q_arr, v_arr = self._require_state_pair(q, v)
        tau: np.ndarray
        if u is None:
            tau = np.zeros(_DOF, dtype=float)
        else:
            tau = np.asarray(u, dtype=float).reshape(-1)
            require(
                tau.size >= _DOF,
                f"control vector must have at least {_DOF} entries; got {tau.size}",
                value=tau.size,
            )
            tau = tau[:_DOF]
        mass = self.mass_matrix(q_arr)
        bias = self.bias_forces(q_arr, v_arr)
        return np.linalg.solve(mass, tau - bias)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the dense joint-space inertia matrix ``M(q)``, shape ``(2, 2)``.

        Args:
            q: Joint positions; only ``q[1]`` (the wrist angle ``theta2``) enters
                the mass matrix. Must carry at least two entries.

        Returns:
            Symmetric positive-definite ``(2, 2)`` inertia matrix.

        Raises:
            ValueError: If ``q`` has fewer than two entries.
        """
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        require(
            q_arr.size >= _DOF,
            f"q must have at least {_DOF} entries; got {q_arr.size}",
            value=q_arr.size,
        )
        return np.array(self._dyn.mass_matrix(float(q_arr[1])), dtype=float)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return bias forces ``C(q,v)v + g(q) + damping``, shape ``(2,)``.

        This is the analytical analogue of MuJoCo's ``qfrc_bias - qfrc_passive``
        and is the term moved to the right-hand side in ``forward_dynamics``.

        Args:
            q: Joint positions ``[theta1, theta2]`` (>= 2 entries used).
            v: Joint velocities ``[omega1, omega2]`` (>= 2 entries used).

        Returns:
            Bias-force vector of shape ``(2,)``.

        Raises:
            ValueError: If ``q`` or ``v`` has fewer than two entries.
        """
        q_arr, v_arr = self._require_state_pair(q, v)
        theta1, theta2 = float(q_arr[0]), float(q_arr[1])
        omega1, omega2 = float(v_arr[0]), float(v_arr[1])
        c1, c2 = self._dyn.coriolis_vector(theta2, omega1, omega2)
        g1, g2 = self._dyn.gravity_vector(theta1, theta2)
        d1, d2 = self._dyn.damping_vector(omega1, omega2)
        return np.array([c1 + g1 + d1, c2 + g2 + d2], dtype=float)

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        """Integrate ``horizon`` steps and return the recorded :class:`Trace`.

        Follows the shared rollout contract: the returned trace has
        ``horizon + 1`` samples — the initial state at ``t = 0`` followed by the
        state after each of ``horizon`` steps, so
        ``t == [0, dt, ..., horizon*dt]``.

        Args:
            controls: Prescribed control history of shape ``(horizon, 2)``;
                ``controls[k]`` is applied *during* step ``k``. ``None`` means a
                passive (zero-torque) rollout.
            horizon: Number of integration steps (``> 0``).
            dt: Integration step size [s] (``> 0``).

        Returns:
            A :class:`Trace` with ``backend == "ode"``, ``t`` of length
            ``horizon + 1``, and ``q`` / ``v`` of shape ``(horizon + 1, 2)``.
            When ``controls`` is given, ``u`` has shape ``(horizon + 1, 2)``
            with ``u[k]`` the control applied during step ``k`` and a
            zero-padded final row (no step departs the terminal sample). This
            time-aligned layout matches the MuJoCo backend so traces can be
            cross-validated row-for-row.

        Raises:
            ValueError: If ``horizon`` or ``dt`` is non-positive, ``dt`` is not
                finite, or ``controls`` is given with the wrong shape.
        """
        require(horizon > 0, f"horizon must be positive; got {horizon}", value=horizon)
        require(
            float(dt) > 0.0 and np.isfinite(dt),
            f"dt must be a positive, finite step size; got {dt!r}",
            value=dt,
        )
        controls_arr = self._validate_controls(controls, horizon)

        num_samples = horizon + 1
        q_hist = np.empty((num_samples, _DOF), dtype=float)
        v_hist = np.empty((num_samples, _DOF), dtype=float)
        t_hist = np.arange(num_samples, dtype=float) * float(dt)
        # Time-aligned control history: zero-padded final row matches the MuJoCo
        # backend layout (Trace.u must share the time axis length).
        u_hist = None if controls_arr is None else np.zeros((num_samples, _DOF))

        state = self._state
        time = self._time
        q_hist[0] = (state.theta1, state.theta2)
        v_hist[0] = (state.omega1, state.omega2)

        zero_u = np.zeros(_DOF, dtype=float)
        for k in range(horizon):
            step_u = zero_u if controls_arr is None else controls_arr[k]
            self._u = step_u.copy()
            if u_hist is not None:
                u_hist[k] = step_u
            state = self._dyn.step(time, state, float(dt))
            time += float(dt)
            q_hist[k + 1] = (state.theta1, state.theta2)
            v_hist[k + 1] = (state.omega1, state.omega2)

        # Commit the integrated trajectory back onto the live backend state.
        self._state = state
        self._time = time

        return Trace(
            t=t_hist,
            q=q_hist,
            v=v_hist,
            u=u_hist,
            dt=float(dt),
            backend="ode",
            meta={"integrator": "rk4", "dof": _DOF},
        )

    @staticmethod
    def _require_state_pair(
        q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coerce and validate a ``(q, v)`` pair to length-``_DOF`` float arrays."""
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        v_arr = np.asarray(v, dtype=float).reshape(-1)
        require(
            q_arr.size >= _DOF and v_arr.size >= _DOF,
            f"q and v must each have at least {_DOF} entries; "
            f"got q={q_arr.size}, v={v_arr.size}",
            value=(q_arr.size, v_arr.size),
        )
        return q_arr, v_arr

    @staticmethod
    def _validate_controls(
        controls: np.ndarray | None, horizon: int
    ) -> np.ndarray | None:
        """Validate the control history shape; return a ``(horizon, 2)`` array."""
        if controls is None:
            return None
        controls_arr = np.asarray(controls, dtype=float)
        require(
            controls_arr.shape == (horizon, _DOF),
            f"controls must have shape ({horizon}, {_DOF}); got {controls_arr.shape}",
            value=controls_arr.shape,
        )
        return controls_arr
