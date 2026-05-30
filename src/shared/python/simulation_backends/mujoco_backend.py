"""MuJoCo CPU backend — the cross-validation lynchpin.

This backend wraps the reference C++ ``mujoco`` bindings on the CPU. It is the
*lynchpin* of the epic's cross-validation strategy: it satisfies both
:class:`~simulation_backends.protocol.SimulationBackend` (full simulation loop)
**and** :class:`~simulation_backends.protocol.DynamicsProvider` (``mass_matrix``
/ ``bias_forces``), giving an independent derivation of the equations of motion
that must agree with the analytical model to ``~1e-9``.

The MuJoCo model is rendered mechanically from the single source of truth via
:func:`simulation_backends.mjcf.params_to_mjcf`, so the MuJoCo and analytical
dynamics cannot silently drift apart (epic task M2.3).

State convention (shared by every backend):

* ``q = [theta1, theta2]`` — shoulder angle (upper link from vertical) and wrist
  hinge angle (lower link *relative* to the upper). This maps one-to-one onto
  MuJoCo's two hinge ``qpos`` entries.
* ``v = [omega1, omega2]`` — corresponding joint velocities.
* ``u = [tau_shoulder, tau_wrist]`` — joint torques applied through the two
  ``motor`` actuators (``gear=1``), here injected via ``qfrc_applied`` so the
  generalised force is exactly ``u``.

All arithmetic is float64 (the MuJoCo CPU default). No GPU synchronisation is
required on the CPU path.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.logging_pkg.logging_config import get_logger

from .capabilities import require_mujoco
from .mjcf import params_to_mjcf
from .protocol import BackendCapabilities, SimState, Trace

if TYPE_CHECKING:
    from .model_params import GolfModelParams

logger = get_logger(__name__)

#: Stable backend identifier (matches the factory registry key).
_BACKEND_NAME = "mujoco"


class MuJoCoBackend:
    """CPU MuJoCo backend exposing simulation *and* dynamics primitives.

    Implements both the :class:`~simulation_backends.protocol.SimulationBackend`
    and :class:`~simulation_backends.protocol.DynamicsProvider` Protocols.

    Args:
        params: The single-source-of-truth model parameters.
        dt: Default integration timestep [s] (strictly positive). Written into
            both the MJCF ``<option>`` and ``model.opt.timestep``; individual
            :meth:`step` / :meth:`rollout` calls may override it.

    Raises:
        TypeError: If ``params`` is not a
            :class:`~simulation_backends.model_params.GolfModelParams`.
        ValueError: If ``dt`` is not strictly positive.
        BackendNotAvailableError: If the ``mujoco`` package is not installed.
    """

    def __init__(self, params: GolfModelParams, *, dt: float = 0.01) -> None:
        # Validate the optional dependency *first* with an actionable message.
        require_mujoco()

        # Imported lazily so package import never requires the mujoco wheel.
        import mujoco

        # DbC preconditions for the public constructor.
        from .model_params import GolfModelParams as _GolfModelParams

        if not isinstance(params, _GolfModelParams):
            raise TypeError(
                f"params must be a GolfModelParams, got {type(params).__name__}"
            )
        if not (isinstance(dt, (int, float)) and dt > 0.0):
            raise ValueError(f"dt must be a positive float, got {dt!r}")

        self._mujoco = mujoco
        self._params = params
        self._model = mujoco.MjModel.from_xml_string(
            params_to_mjcf(params, timestep_s=float(dt))
        )
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = float(dt)
        self._u = np.zeros(self._model.nu, dtype=float)

        # Bring qpos/qvel/derived quantities to a consistent initial state.
        mujoco.mj_forward(self._model, self._data)

    # ------------------------------------------------------------------ #
    # Capabilities
    # ------------------------------------------------------------------ #
    @property
    def capabilities(self) -> BackendCapabilities:
        """Return the static capability description for this backend."""
        return BackendCapabilities(
            name=_BACKEND_NAME,
            device="cpu",
            supports_batched=False,
            is_differentiable=False,
            provides_dynamics=True,
        )

    # ------------------------------------------------------------------ #
    # DynamicsProvider
    # ------------------------------------------------------------------ #
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the dense joint-space inertia matrix ``M(q)``.

        Args:
            q: Joint positions, shape ``(nv,)``.

        Returns:
            The symmetric positive-definite mass matrix, shape ``(nv, nv)``.

        Raises:
            ValueError: If ``q`` does not have shape ``(nv,)``.
        """
        q_arr = self._as_state_vector(q, "q")
        d = self._data
        d.qpos[:] = q_arr
        d.qvel[:] = 0.0
        self._mujoco.mj_forward(self._model, d)
        m = np.zeros((self._model.nv, self._model.nv), dtype=float)
        self._mujoco.mj_fullM(self._model, m, d.qM)
        return m

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return bias forces ``C(q,v) v + g(q) (+ damping)``.

        This is MuJoCo's ``qfrc_bias`` minus ``qfrc_passive`` (the latter holds
        the joint-damping contribution), matching the analytical
        ``coriolis + gravity + damping`` decomposition.

        Args:
            q: Joint positions, shape ``(nv,)``.
            v: Joint velocities, shape ``(nv,)``.

        Returns:
            Bias-force vector, shape ``(nv,)``.

        Raises:
            ValueError: If ``q`` or ``v`` does not have shape ``(nv,)``.
        """
        q_arr = self._as_state_vector(q, "q")
        v_arr = self._as_state_vector(v, "v")
        d = self._data
        d.qpos[:] = q_arr
        d.qvel[:] = v_arr
        d.qfrc_applied[:] = 0.0
        self._mujoco.mj_forward(self._model, d)
        return (d.qfrc_bias - d.qfrc_passive).copy()

    # ------------------------------------------------------------------ #
    # SimulationBackend
    # ------------------------------------------------------------------ #
    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return joint accelerations ``qacc`` for state ``(q, v)`` under ``u``.

        Args:
            q: Joint positions, shape ``(nv,)``.
            v: Joint velocities, shape ``(nv,)``.
            u: Applied joint torques, shape ``(nu,)``. ``None`` means passive
                (zero applied torque).

        Returns:
            Joint accelerations, shape ``(nv,)``.

        Raises:
            ValueError: If any argument has the wrong shape.
        """
        q_arr = self._as_state_vector(q, "q")
        v_arr = self._as_state_vector(v, "v")
        if u is None:
            applied = np.zeros(self._model.nv, dtype=float)
        else:
            applied = self._as_control_vector(u)
        d = self._data
        d.qpos[:] = q_arr
        d.qvel[:] = v_arr
        d.qfrc_applied[:] = applied
        self._mujoco.mj_forward(self._model, d)
        return d.qacc.copy()

    def reset(self, state: SimState | None = None) -> None:
        """Reset to ``state`` (or the canonical zero state if ``None``).

        Args:
            state: Target state. Its ``q`` / ``v`` must each have length ``nv``.

        Raises:
            TypeError: If ``state`` is neither ``None`` nor a :class:`SimState`.
            ValueError: If ``state.q`` / ``state.v`` have the wrong length.
        """
        d = self._data
        if state is None:
            d.qpos[:] = 0.0
            d.qvel[:] = 0.0
        else:
            if not isinstance(state, SimState):
                raise TypeError(
                    f"state must be a SimState or None, got {type(state).__name__}"
                )
            d.qpos[:] = self._as_state_vector(state.q, "state.q")
            d.qvel[:] = self._as_state_vector(state.v, "state.v")
        d.time = 0.0
        d.qfrc_applied[:] = 0.0
        self._u = np.zeros(self._model.nu, dtype=float)
        self._mujoco.mj_forward(self._model, d)

    def set_control(self, u: np.ndarray) -> None:
        """Set the generalised torque vector applied by subsequent steps.

        Args:
            u: Joint torques with at least ``nu`` entries.

        Raises:
            ValueError: If ``u`` has fewer than ``nu`` entries.
        """
        self._u = self._as_control_vector(u)

    def step(self, dt: float | None = None) -> None:
        """Advance the simulation by one step.

        Args:
            dt: Step size [s]. ``None`` reuses ``model.opt.timestep``; a positive
                override updates ``model.opt.timestep`` before stepping.

        Raises:
            ValueError: If ``dt`` is provided and not strictly positive.
        """
        if dt is not None:
            if not (isinstance(dt, (int, float)) and dt > 0.0):
                raise ValueError(f"dt must be a positive float, got {dt!r}")
            if dt != self._model.opt.timestep:
                self._model.opt.timestep = float(dt)
        self._data.qfrc_applied[:] = self._u
        self._mujoco.mj_step(self._model, self._data)

    def get_state(self) -> SimState:
        """Return the current :class:`SimState` (arrays copied)."""
        d = self._data
        return SimState(q=d.qpos.copy(), v=d.qvel.copy(), time=float(d.time))

    def get_time(self) -> float:
        """Return the current simulation time [s]."""
        return float(self._data.time)

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        """Integrate ``horizon`` steps from the current state.

        Honours the shared rollout contract: the returned :class:`Trace` has
        ``horizon + 1`` samples — the initial state at ``t = 0`` followed by the
        state after each of ``horizon`` steps. ``controls[k]`` is applied during
        step ``k``; ``None`` means a passive (zero-torque) rollout.

        Args:
            controls: Prescribed control history ``(horizon, nu)`` or ``None``.
            horizon: Number of steps to integrate (``> 0``).
            dt: Integration step size [s] (``> 0``).

        Returns:
            A :class:`Trace` with ``t``/``q``/``v`` of length ``horizon + 1`` and
            ``u`` of shape ``(horizon + 1, nu)`` (zero-padded final row) when
            ``controls`` is given, else ``None``.

        Raises:
            ValueError: If ``horizon``/``dt`` are non-positive or ``controls`` has
                the wrong shape.
        """
        if not (isinstance(horizon, int) and horizon > 0):
            raise ValueError(f"horizon must be a positive int, got {horizon!r}")
        if not (isinstance(dt, (int, float)) and dt > 0.0):
            raise ValueError(f"dt must be a positive float, got {dt!r}")

        nu = int(self._model.nu)
        nv = int(self._model.nv)
        controls_arr = self._validate_controls(controls, horizon, nu)

        # Reset only the integrator clock so the rollout starts at t=0 from the
        # current configuration (matches the ODE backend contract).
        self._data.time = 0.0
        self._data.qfrc_applied[:] = 0.0

        q_hist = np.empty((horizon + 1, nv), dtype=float)
        v_hist = np.empty((horizon + 1, nv), dtype=float)
        t_hist = np.empty(horizon + 1, dtype=float)
        u_hist = None if controls_arr is None else np.zeros((horizon + 1, nu))

        # Record the initial state (t = 0) before any step.
        self._record(0, t_hist, q_hist, v_hist)

        zero_u = np.zeros(nu, dtype=float)
        for k in range(horizon):
            step_u = zero_u if controls_arr is None else controls_arr[k]
            self.set_control(step_u)
            if u_hist is not None:
                u_hist[k] = step_u
            self.step(dt)
            self._record(k + 1, t_hist, q_hist, v_hist)

        return Trace(
            t=t_hist,
            q=q_hist,
            v=v_hist,
            u=u_hist,
            dt=float(dt),
            backend=_BACKEND_NAME,
            meta={"nq": nv, "nv": nv, "nu": nu},
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _record(
        self,
        index: int,
        t_hist: np.ndarray,
        q_hist: np.ndarray,
        v_hist: np.ndarray,
    ) -> None:
        """Copy the current MuJoCo state into the pre-allocated history rows."""
        d = self._data
        t_hist[index] = float(d.time)
        q_hist[index] = d.qpos
        v_hist[index] = d.qvel

    def _as_state_vector(self, x: np.ndarray, name: str) -> np.ndarray:
        """Coerce ``x`` to a float ``(nv,)`` vector, validating its length."""
        arr = np.asarray(x, dtype=float).reshape(-1)
        nv = int(self._model.nv)
        if arr.shape != (nv,):
            raise ValueError(f"{name} must have shape ({nv},), got {tuple(arr.shape)}")
        return arr

    def _as_control_vector(self, u: np.ndarray) -> np.ndarray:
        """Coerce ``u`` to a float ``(nu,)`` vector, requiring at least ``nu``."""
        arr = np.asarray(u, dtype=float).reshape(-1)
        nu = int(self._model.nu)
        if arr.shape[0] < nu:
            raise ValueError(
                f"control must have at least {nu} entries, got {arr.shape[0]}"
            )
        return arr[:nu].copy()

    def _validate_controls(
        self, controls: np.ndarray | None, horizon: int, nu: int
    ) -> np.ndarray | None:
        """Validate and coerce a rollout control history to ``(horizon, nu)``."""
        if controls is None:
            return None
        arr = np.asarray(controls, dtype=float)
        if arr.shape != (horizon, nu):
            raise ValueError(
                f"controls must have shape ({horizon}, {nu}), got {tuple(arr.shape)}"
            )
        return arr
