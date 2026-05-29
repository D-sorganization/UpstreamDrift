"""MuJoCo Warp (MJWarp) GPU backend for massively parallel batched rollouts.

This backend wraps `mujoco_warp <https://github.com/google-deepmind/mujoco_warp>`_
("MJWarp"), the GPU re-implementation of MuJoCo's pipeline built on NVIDIA Warp.
Its reason for existing is :meth:`MJWarpBackend.rollout_batch`: thousands of
independent double-pendulum worlds stepped in parallel on a CUDA device, which
the CPU ``ode`` / ``mujoco`` backends cannot do at scale.

Status, versioning and stability (ADR-0023)
--------------------------------------------
MJWarp is **alpha and under active development**; its public API still moves
between releases. Because of that, the GPU stack is an *optional* extra rather
than a hard dependency: the package imports and the full test suite runs on a
CPU-only machine via the ``ode`` and ``mujoco`` backends. The exact pinned
versions of ``warp-lang`` and ``mujoco-warp`` that this backend is validated
against — and the rationale for treating it as alpha — are recorded in
``docs/adr/0023-mujoco-warp-backend.md``. Update that ADR (not this docstring)
when the pin changes.

float32 vs float64 (epic task M4.3)
-----------------------------------
MJWarp computes in **single precision (float32)** on the GPU, whereas the
analytical / MuJoCo-CPU backends compute in float64. Cross-validation against
those backends therefore uses *tolerance-based* comparisons (``np.allclose``
with a relaxed ``atol``/``rtol``), never bit-exact equality. Host-side arrays
returned by this backend are upcast to float64 to match the frozen
:class:`~simulation_backends.protocol.SimState` / :class:`Trace` convention, but
the underlying integration error floor is set by float32, not by the upcast.

Reading results back to the host (synchronisation requirement)
--------------------------------------------------------------
Warp kernels are launched **asynchronously** on the CUDA stream. Any device
array (``qpos``/``qvel``/...) read back to host memory *before the stream has
drained* may contain stale or partially-written values. Every method in this
backend that copies device state to NumPy therefore calls ``wp.synchronize()``
**first**. This is a correctness requirement, not an optimisation: omitting it
yields silently wrong traces.

Capabilities
------------
``provides_dynamics`` is ``False``: MJWarp is a black-box stepper here and does
not expose dense ``M(q)`` / bias primitives, so :meth:`mass_matrix` and
:meth:`bias_forces` raise :class:`BackendCapabilityError` directing callers to
the ``mujoco`` backend for those.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from .exceptions import BackendCapabilityError
from .mjcf import params_to_mjcf
from .protocol import BackendCapabilities, BatchTrace, SimState, Trace

if TYPE_CHECKING:
    from .model_params import GolfModelParams

logger = get_logger(__name__)

#: Backend identifier registered in the factory (epic convention: lowercase).
_BACKEND_NAME = "mjwarp"

#: State dimensions of the planar driven double pendulum (shoulder + wrist).
_NQ = 2
_NV = 2
_NU = 2


class MJWarpBackend:
    """GPU MuJoCo-Warp backend (single-env convenience + batched rollouts).

    Satisfies both :class:`~simulation_backends.protocol.SimulationBackend` and
    :class:`~simulation_backends.protocol.BatchedBackend`. The single-env methods
    (:meth:`reset`, :meth:`step`, :meth:`rollout`, ...) are thin conveniences
    implemented as a one-world batch; the value of this backend is
    :meth:`rollout_batch`.

    Constructing this backend **requires** the optional GPU stack (``warp`` +
    ``mujoco_warp`` + a CUDA device); see :func:`capabilities.require_warp`. The
    class itself imports with no GPU present so that capability flags can be
    inspected via :meth:`describe_capabilities` on any machine.

    Args:
        params: The single-source-of-truth model parameters.
        dt: Default integration step [s] written into the MJCF and used by
            :meth:`step` when no per-call ``dt`` is given (``> 0``).
        device: Warp device string, e.g. ``"cuda"`` or ``"cuda:0"``.

    Raises:
        BackendNotAvailableError: If the ``[warp]`` extra is not installed.
        TypeError: If ``params`` is not a :class:`GolfModelParams`.
        ValueError: If ``dt`` is not strictly positive.
    """

    def __init__(
        self,
        params: GolfModelParams,
        *,
        dt: float = 0.01,
        device: str = "cuda",
    ) -> None:
        # Gate on the optional GPU stack *first*: this raises
        # BackendNotAvailableError on a CPU-only machine (this one), so nothing
        # below ever executes here. Import lazily so module import stays clean.
        from .capabilities import require_warp

        require_warp()

        # --- GPU-ONLY PATH (not executable on this CPU-only machine) ---------
        # Everything from here down is reached only when warp + mujoco_warp +
        # CUDA are present. It cannot be run or unit-tested on this box; it is
        # exercised by the @requires_gpu smoke tests on a CUDA runner.
        from .model_params import GolfModelParams as _GolfModelParams

        require(
            isinstance(params, _GolfModelParams),
            "params must be a GolfModelParams",
        )
        require(dt > 0.0, "dt must be > 0")

        import mujoco  # noqa: PLC0415 - lazy GPU-only import
        import mujoco_warp as mjw  # noqa: PLC0415 - lazy GPU-only import
        import warp as wp  # noqa: PLC0415 - lazy GPU-only import

        wp.init()

        self._params = params
        self._dt = float(dt)
        self._device = device
        self._mjw = mjw
        self._wp = wp

        # Build the CPU MuJoCo model from the shared MJCF, then upload it once.
        mjcf_xml = params_to_mjcf(params, timestep_s=self._dt)
        cpu_model = mujoco.MjModel.from_xml_string(mjcf_xml)
        self._cpu_model = cpu_model
        self._m = mjw.put_model(cpu_model)

        # Single-env working data; (re)created on reset.
        self._d: Any = None
        self._control = np.zeros(_NU, dtype=np.float64)
        self.reset()

    # ------------------------------------------------------------------ #
    # Capabilities                                                       #
    # ------------------------------------------------------------------ #
    @staticmethod
    def describe_capabilities() -> BackendCapabilities:
        """Return the static capability descriptor for the MJWarp backend.

        Implemented as a ``staticmethod`` (no instance, no GPU) so callers and
        tests can assert backend flags on a machine without CUDA/warp.

        Returns:
            The frozen :class:`BackendCapabilities` for ``mjwarp``: a CUDA,
            batched, non-differentiable backend that does not expose dynamics
            primitives.
        """
        return BackendCapabilities(
            name=_BACKEND_NAME,
            device="cuda",
            supports_batched=True,
            is_differentiable=False,
            provides_dynamics=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capability description for this backend instance."""
        return self.describe_capabilities()

    # ------------------------------------------------------------------ #
    # Single-env convenience (SimulationBackend protocol)                #
    # ------------------------------------------------------------------ #
    def reset(self, state: SimState | None = None) -> None:
        """Reset the single-env world to ``state`` (or zeros if ``None``).

        Args:
            state: Initial state; ``q``/``v`` must each have length 2. ``None``
                resets to the canonical zero configuration (both links down).

        Raises:
            ValueError: If ``state`` has the wrong dimension.
        """
        # GPU-ONLY: builds a one-world device Data buffer.
        self._d = self._mjw.make_data(self._m, nworld=1)
        if state is not None:
            self._validate_state(state)
            self._write_world_state(0, np.asarray(state.q), np.asarray(state.v))
            self._set_world_time(float(state.time))
        self._control = np.zeros(_NU, dtype=np.float64)
        self._write_world_control(0, self._control)

    def step(self, dt: float | None = None) -> None:
        """Advance the single-env world by one step.

        Args:
            dt: Step size [s]; if ``None`` the construction-time ``dt`` is used.
                If given it must be ``> 0`` and overrides ``opt.timestep``.

        Raises:
            ValueError: If ``dt`` is given and not strictly positive.
        """
        if dt is not None:
            require(dt > 0.0, "dt must be > 0")
            self._set_timestep(float(dt))
        # GPU-ONLY: launch + drain the stream before any later host read.
        self._mjw.step(self._m, self._d)
        self._wp.synchronize()

    def get_state(self) -> SimState:
        """Return the current single-env :class:`SimState` (read from device)."""
        q, v = self._read_world_state(0)
        return SimState(q=q, v=v, time=self._read_world_time())

    def set_control(self, u: np.ndarray) -> None:
        """Set the torque vector ``[tau_shoulder, tau_wrist]`` for next steps.

        Args:
            u: Control vector of length 2 [N*m].

        Raises:
            ValueError: If ``u`` does not have length 2.
        """
        ctrl = np.asarray(u, dtype=np.float64).reshape(-1)
        require(ctrl.size == _NU, f"control must have length {_NU}, got {ctrl.size}")
        self._control = ctrl
        self._write_world_control(0, ctrl)

    def get_time(self) -> float:
        """Return the current single-env simulation time [s]."""
        return self._read_world_time()

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return joint accelerations ``qacc`` for state ``(q, v)`` under ``u``.

        Computes a single forward dynamics pass (``mjw.forward``) on the device
        and copies ``qacc`` back to the host after synchronising.

        Args:
            q: Joint positions, length 2 [rad].
            v: Joint velocities, length 2 [rad/s].
            u: Optional torque vector, length 2 [N*m]; ``None`` means zero.

        Returns:
            ``qacc`` of shape ``(2,)`` [rad/s^2], dtype float64.

        Raises:
            ValueError: If ``q``/``v``/``u`` have the wrong length.
        """
        q_arr = np.asarray(q, dtype=np.float64).reshape(-1)
        v_arr = np.asarray(v, dtype=np.float64).reshape(-1)
        require(q_arr.size == _NQ, f"q must have length {_NQ}, got {q_arr.size}")
        require(v_arr.size == _NV, f"v must have length {_NV}, got {v_arr.size}")
        if u is None:
            u_arr = np.zeros(_NU, dtype=np.float64)
        else:
            u_arr = np.asarray(u, dtype=np.float64).reshape(-1)
            require(u_arr.size == _NU, f"u must have length {_NU}, got {u_arr.size}")

        self._write_world_state(0, q_arr, v_arr)
        self._write_world_control(0, u_arr)
        # GPU-ONLY: forward pass populates d.qacc; synchronise before reading.
        self._mjw.forward(self._m, self._d)
        self._wp.synchronize()
        qacc = self._device_to_host(self._d.qacc)[0]
        return np.asarray(qacc, dtype=np.float64).reshape(-1)

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        """Integrate ``horizon`` steps for one env and return a :class:`Trace`.

        Implemented as a one-env :meth:`rollout_batch`. Honours the rollout
        contract: returns ``horizon + 1`` samples (initial state then one per
        step), with ``t == [0, dt, ..., horizon*dt]``.

        Args:
            controls: Control history ``(horizon, 2)`` applied during each step,
                or ``None`` for a passive (zero-torque) rollout.
            horizon: Number of steps (``> 0``).
            dt: Step size [s] (``> 0``).

        Returns:
            A :class:`Trace` with ``q``/``v`` of shape ``(horizon + 1, 2)``.

        Raises:
            ValueError: If ``horizon``/``dt`` are non-positive or ``controls``
                has the wrong shape.
        """
        batch = self.rollout_batch(controls, horizon, dt, num_envs=1)
        return batch.env(0)

    # ------------------------------------------------------------------ #
    # Batched rollout (BatchedBackend protocol) -- the headline feature  #
    # ------------------------------------------------------------------ #
    def rollout_batch(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> BatchTrace:
        """Integrate ``num_envs`` worlds in parallel and return a BatchTrace.

        Steps the MJWarp pipeline ``horizon`` times over ``num_envs`` worlds,
        recording state after each step. Calls ``wp.synchronize()`` before every
        host read so the assembled arrays are never stale (see module docstring).

        Args:
            controls: One of

                * ``None`` — passive (zero torque) for all worlds;
                * shape ``(horizon, 2)`` — shared history broadcast to all worlds;
                * shape ``(num_envs, horizon, 2)`` — independent per-world history.

                ``controls[..., k, :]`` is the torque applied **during** step
                ``k`` (``k = 0 .. horizon - 1``).
            horizon: Number of steps to integrate (``> 0``).
            dt: Integration step size [s] (``> 0``).
            num_envs: Number of parallel worlds (``> 0``).

        Returns:
            A :class:`BatchTrace` with ``q``/``v`` of shape
            ``(num_envs, horizon + 1, 2)`` and ``t`` of length ``horizon + 1``.
            ``u`` is ``None`` when ``controls`` is ``None``; otherwise it is the
            broadcast per-env history of shape ``(num_envs, horizon, 2)``.

        Raises:
            ValueError: If ``horizon``/``dt``/``num_envs`` are non-positive or
                ``controls`` has an unsupported shape.
        """
        require(horizon > 0, f"horizon must be > 0, got {horizon}")
        require(dt > 0.0, f"dt must be > 0, got {dt}")
        require(num_envs > 0, f"num_envs must be > 0, got {num_envs}")
        control_seq = self._normalise_controls(controls, horizon, num_envs)

        # GPU-ONLY PATH (not executable on this CPU-only machine) ------------
        self._set_timestep(float(dt))
        data = self._mjw.make_data(self._m, nworld=num_envs)

        nq, nv = _NQ, _NV
        q_hist = np.zeros((num_envs, horizon + 1, nq), dtype=np.float64)
        v_hist = np.zeros((num_envs, horizon + 1, nv), dtype=np.float64)

        # Record the initial state (t = 0) before stepping.
        self._wp.synchronize()
        q_hist[:, 0, :] = self._device_to_host(data.qpos)[:, :nq]
        v_hist[:, 0, :] = self._device_to_host(data.qvel)[:, :nv]

        for k in range(horizon):
            if control_seq is not None:
                self._write_all_controls(data, control_seq[:, k, :])
            self._mjw.step(self._m, data)
            # Synchronise BEFORE copying device state to the host buffer.
            self._wp.synchronize()
            q_hist[:, k + 1, :] = self._device_to_host(data.qpos)[:, :nq]
            v_hist[:, k + 1, :] = self._device_to_host(data.qvel)[:, :nv]

        t = np.arange(horizon + 1, dtype=np.float64) * float(dt)
        u_out = None if control_seq is None else control_seq
        return BatchTrace(
            t=t,
            q=q_hist,
            v=v_hist,
            u=u_out,
            dt=float(dt),
            backend=_BACKEND_NAME,
            meta={"device": self._device, "precision": "float32"},
        )

    # ------------------------------------------------------------------ #
    # Dynamics primitives -- unsupported (provides_dynamics=False)       #
    # ------------------------------------------------------------------ #
    def mass_matrix(self, q: np.ndarray) -> np.ndarray:  # noqa: ARG002 - interface
        """Not supported by MJWarp; use the ``mujoco`` backend.

        Raises:
            BackendCapabilityError: Always — ``provides_dynamics`` is ``False``.
        """
        raise BackendCapabilityError(
            "mjwarp does not expose dynamics primitives; "
            "use the mujoco backend for M(q)/bias"
        )

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:  # noqa: ARG002 - interface
        """Not supported by MJWarp; use the ``mujoco`` backend.

        Raises:
            BackendCapabilityError: Always — ``provides_dynamics`` is ``False``.
        """
        raise BackendCapabilityError(
            "mjwarp does not expose dynamics primitives; "
            "use the mujoco backend for M(q)/bias"
        )

    # ------------------------------------------------------------------ #
    # Internal helpers (all GPU-only; rely on warp/mujoco_warp)          #
    # ------------------------------------------------------------------ #
    @staticmethod
    def _validate_state(state: SimState) -> None:
        """Validate a single-env state has 2-vector ``q`` and ``v``."""
        require(
            state.q.size == _NQ and state.v.size == _NV,
            f"state q/v must each have length {_NQ}",
        )

    @staticmethod
    def _normalise_controls(
        controls: np.ndarray | None,
        horizon: int,
        num_envs: int,
    ) -> np.ndarray | None:
        """Broadcast ``controls`` to a ``(num_envs, horizon, 2)`` float array.

        Args:
            controls: ``None``, shared ``(horizon, 2)``, or per-env
                ``(num_envs, horizon, 2)``.
            horizon: Expected number of steps.
            num_envs: Expected number of worlds.

        Returns:
            ``None`` if ``controls`` is ``None`` (passive); otherwise a
            contiguous ``(num_envs, horizon, 2)`` float64 array.

        Raises:
            ValueError: If ``controls`` has an unsupported shape.
        """
        if controls is None:
            return None
        arr = np.asarray(controls, dtype=np.float64)
        if arr.shape == (horizon, _NU):
            arr = np.broadcast_to(arr, (num_envs, horizon, _NU))
        elif arr.shape != (num_envs, horizon, _NU):
            raise ValueError(
                "controls must be None, "
                f"({horizon}, {_NU}) shared, or "
                f"({num_envs}, {horizon}, {_NU}) per-env; got {arr.shape}"
            )
        return np.ascontiguousarray(arr, dtype=np.float64)

    def _device_to_host(self, device_array: Any) -> np.ndarray:
        """Copy a warp device array to a 2-D ``(nworld, dim)`` NumPy array.

        Assumes the caller has already invoked ``wp.synchronize()``. Upcasts to
        float64 to match the frozen :class:`SimState`/:class:`Trace` convention.
        """
        host = np.asarray(device_array.numpy(), dtype=np.float64)
        return np.atleast_2d(host)

    def _set_timestep(self, dt: float) -> None:
        """Set the integrator timestep on both the host and device models."""
        self._cpu_model.opt.timestep = dt
        self._m.opt.timestep = dt

    # The remaining single-world accessors index world 0 of ``self._d`` and are
    # only ever reached on the GPU path (after a successful require_warp()).
    def _write_world_state(self, world: int, q: np.ndarray, v: np.ndarray) -> None:
        """Write ``q``/``v`` into ``self._d`` for one world (host->device)."""
        qpos = self._device_to_host(self._d.qpos)
        qvel = self._device_to_host(self._d.qvel)
        qpos[world, :_NQ] = q
        qvel[world, :_NV] = v
        self._d.qpos = self._wp.array(qpos.astype(np.float32), dtype=self._wp.float32)
        self._d.qvel = self._wp.array(qvel.astype(np.float32), dtype=self._wp.float32)

    def _write_world_control(self, world: int, u: np.ndarray) -> None:
        """Write control vector ``u`` into ``self._d.ctrl`` for one world."""
        ctrl = self._device_to_host(self._d.ctrl)
        ctrl[world, :_NU] = u
        self._d.ctrl = self._wp.array(ctrl.astype(np.float32), dtype=self._wp.float32)

    def _write_all_controls(self, data: Any, controls: np.ndarray) -> None:
        """Write a ``(num_envs, 2)`` control slice into all worlds of ``data``."""
        ctrl = np.ascontiguousarray(controls, dtype=np.float32)
        data.ctrl = self._wp.array(ctrl, dtype=self._wp.float32)

    def _read_world_state(self, world: int) -> tuple[np.ndarray, np.ndarray]:
        """Read ``(q, v)`` for one world (device->host, after synchronise)."""
        self._wp.synchronize()
        q = self._device_to_host(self._d.qpos)[world, :_NQ]
        v = self._device_to_host(self._d.qvel)[world, :_NV]
        return np.asarray(q, dtype=np.float64), np.asarray(v, dtype=np.float64)

    def _read_world_time(self) -> float:
        """Read the simulation time of world 0 (device->host)."""
        self._wp.synchronize()
        time_host = np.atleast_1d(np.asarray(self._d.time.numpy(), dtype=np.float64))
        return float(time_host.reshape(-1)[0])

    def _set_world_time(self, time: float) -> None:
        """Set the simulation time across worlds of ``self._d``."""
        time_host = np.atleast_1d(np.asarray(self._d.time.numpy(), dtype=np.float64))
        time_host[...] = time
        self._d.time = self._wp.array(
            time_host.astype(np.float32), dtype=self._wp.float32
        )
