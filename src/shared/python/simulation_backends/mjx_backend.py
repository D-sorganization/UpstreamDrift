"""MJX (MuJoCo on JAX) backend for differentiable batched rollouts.

MJX is the JAX-native MuJoCo pipeline. This adapter keeps it behind the same
``SimulationBackend`` / ``BatchedBackend`` contracts as the ODE, MuJoCo CPU and
MJWarp adapters:

* model source is still :func:`simulation_backends.mjcf.params_to_mjcf`;
* host-facing rollout results still use :class:`Trace` / :class:`BatchTrace`;
* JAX/MJX imports stay lazy and optional;
* differentiable code paths remain available before conversion to NumPy through
  :meth:`rollout_batch_arrays` and :meth:`final_state_control_jacobian`.

``provides_dynamics`` is ``False`` because this bounded slice exposes pointwise
``forward_dynamics`` and differentiable rollouts, not dense ``M(q)`` / bias
force primitives. Use the ``ode`` or ``mujoco`` backend for those.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
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

_BACKEND_NAME = "mjx"
_NQ = 2
_NV = 2
_NU = 2


@dataclass(frozen=True)
class _MJXApis:
    """Lazy imports grouped for test injection and narrow adapter plumbing."""

    jax: Any
    jnp: Any
    mujoco: Any
    mjx: Any


class MJXBackend:
    """MJX/JAX backend with batched and differentiable rollout surfaces.

    Args:
        params: The single-source-of-truth model parameters.
        dt: Default integration step [s] written into the MJCF and MJX model.

    Raises:
        BackendNotAvailableError: If the optional ``[mjx]`` stack is missing.
        TypeError: If ``params`` is not a :class:`GolfModelParams`.
        ValueError: If ``dt`` is not strictly positive.
    """

    def __init__(self, params: GolfModelParams, *, dt: float = 0.01) -> None:
        from .capabilities import require_mjx
        from .model_params import GolfModelParams as _GolfModelParams

        require_mjx()
        if not isinstance(params, _GolfModelParams):
            raise TypeError(
                f"params must be a GolfModelParams, got {type(params).__name__}"
            )
        require(dt > 0.0, "dt must be > 0")

        self._apis = _load_mjx_apis()
        self._params = params
        self._dt = float(dt)
        self._cpu_model = self._apis.mujoco.MjModel.from_xml_string(
            params_to_mjcf(params, timestep_s=self._dt)
        )
        self._m = self._apis.mjx.put_model(self._cpu_model)
        self._data = self._apis.mjx.make_data(self._m)
        self._control = np.zeros(_NU, dtype=np.float64)
        self.reset()

    @staticmethod
    def describe_capabilities() -> BackendCapabilities:
        """Return MJX's static capability descriptor without importing JAX."""
        return BackendCapabilities(
            name=_BACKEND_NAME,
            device="jax",
            supports_batched=True,
            is_differentiable=True,
            provides_dynamics=False,
        )

    @property
    def capabilities(self) -> BackendCapabilities:
        """Static capability description for this backend instance."""
        return self.describe_capabilities()

    def reset(self, state: SimState | None = None) -> None:
        """Reset the single-env working state to ``state`` or zeros."""
        data = self._apis.mjx.make_data(self._m)
        if state is not None:
            _validate_state(state)
            data = _replace_data(
                data,
                qpos=self._jnp_array(state.q),
                qvel=self._jnp_array(state.v),
                time=self._jnp_array(float(state.time)),
            )
        self._data = data
        self._control = np.zeros(_NU, dtype=np.float64)
        self._write_control(self._control)

    def step(self, dt: float | None = None) -> None:
        """Advance the single-env working state by one MJX step."""
        if dt is not None:
            require(dt > 0.0, "dt must be > 0")
            self._set_timestep(float(dt))
        self._write_control(self._control)
        self._data = self._apis.mjx.step(self._m, self._data)

    def get_state(self) -> SimState:
        """Return the current single-env :class:`SimState`."""
        q = np.asarray(self._data.qpos, dtype=np.float64).reshape(-1)[:_NQ]
        v = np.asarray(self._data.qvel, dtype=np.float64).reshape(-1)[:_NV]
        time = float(np.asarray(self._data.time, dtype=np.float64).reshape(-1)[0])
        return SimState(q=q, v=v, time=time)

    def set_control(self, u: np.ndarray) -> None:
        """Set the torque vector ``[tau_shoulder, tau_wrist]`` for stepping."""
        control = _as_fixed_vector(u, _NU, "control")
        self._control = control
        self._write_control(control)

    def get_time(self) -> float:
        """Return the current single-env simulation time [s]."""
        return self.get_state().time

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        """Return MJX forward dynamics acceleration for ``(q, v, u)``."""
        q_arr = _as_fixed_vector(q, _NQ, "q")
        v_arr = _as_fixed_vector(v, _NV, "v")
        u_arr = np.zeros(_NU, dtype=np.float64)
        if u is not None:
            u_arr = _as_fixed_vector(u, _NU, "u")

        data = self._apis.mjx.make_data(self._m)
        data = _replace_data(
            data,
            qpos=self._jnp_array(q_arr),
            qvel=self._jnp_array(v_arr),
            ctrl=self._jnp_array(u_arr),
        )
        data = self._apis.mjx.forward(self._m, data)
        return np.asarray(data.qacc, dtype=np.float64).reshape(-1)[:_NV]

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        """Integrate one differentiable MJX rollout from the current state."""
        current = self.get_state()
        control_seq = _normalise_controls(controls, horizon, 1)
        _t, q, v, u = self.rollout_batch_arrays(
            control_seq,
            horizon=horizon,
            dt=dt,
            num_envs=1,
            initial_q=current.q.reshape(1, _NQ),
            initial_v=current.v.reshape(1, _NV),
        )
        u_trace = None if u is None else np.asarray(u[0], dtype=np.float64)
        trace = Trace(
            t=np.arange(horizon + 1, dtype=np.float64) * float(dt),
            q=np.asarray(q[0], dtype=np.float64),
            v=np.asarray(v[0], dtype=np.float64),
            u=u_trace,
            dt=float(dt),
            backend=_BACKEND_NAME,
            meta=_trace_meta(self._device_kind()),
        )
        self.reset(trace.final_state())
        return trace

    def rollout_batch(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> BatchTrace:
        """Integrate ``num_envs`` MJX worlds in parallel."""
        control_seq = _normalise_controls(controls, horizon, num_envs)
        t, q, v, u = self.rollout_batch_arrays(
            control_seq,
            horizon=horizon,
            dt=dt,
            num_envs=num_envs,
        )
        return BatchTrace(
            t=np.asarray(t, dtype=np.float64),
            q=np.asarray(q, dtype=np.float64),
            v=np.asarray(v, dtype=np.float64),
            u=None if u is None else np.asarray(u, dtype=np.float64),
            dt=float(dt),
            backend=_BACKEND_NAME,
            meta=_trace_meta(self._device_kind()),
        )

    def rollout_batch_arrays(
        self,
        controls: Any,
        *,
        horizon: int,
        dt: float,
        num_envs: int,
        initial_q: np.ndarray | None = None,
        initial_v: np.ndarray | None = None,
    ) -> tuple[Any, Any, Any, Any | None]:
        """Return JAX arrays for a batched rollout before host conversion.

        This is the differentiable surface: callers that need gradients should
        compose over these arrays (or use :meth:`final_state_control_jacobian`)
        rather than the NumPy-backed :class:`BatchTrace`.
        """
        _validate_rollout_args(horizon, dt, num_envs)
        self._set_timestep(float(dt))

        jnp = self._apis.jnp
        q0 = (
            jnp.zeros((num_envs, _NQ), dtype=jnp.float64)
            if initial_q is None
            else jnp.asarray(initial_q, dtype=jnp.float64)
        )
        v0 = (
            jnp.zeros((num_envs, _NV), dtype=jnp.float64)
            if initial_v is None
            else jnp.asarray(initial_v, dtype=jnp.float64)
        )
        _validate_initial_batch(q0, v0, num_envs)

        control_seq = None
        u_hist = None
        if controls is not None:
            control_seq = jnp.asarray(controls, dtype=jnp.float64)
            if tuple(control_seq.shape) != (num_envs, horizon, _NU):
                raise ValueError(
                    "controls must have shape "
                    f"({num_envs}, {horizon}, {_NU}), got {control_seq.shape}"
                )
            terminal = jnp.zeros((num_envs, 1, _NU), dtype=jnp.float64)
            u_hist = jnp.concatenate([control_seq, terminal], axis=1)

        zero_control = jnp.zeros((num_envs, _NU), dtype=jnp.float64)
        data = self._apis.mjx.make_data(self._m)
        data = _replace_data(
            data,
            qpos=q0,
            qvel=v0,
            ctrl=zero_control,
            time=jnp.zeros((num_envs,), dtype=jnp.float64),
        )

        q_rows = [data.qpos]
        v_rows = [data.qvel]
        for step_index in range(horizon):
            step_control = (
                zero_control if control_seq is None else control_seq[:, step_index, :]
            )
            data = _replace_data(data, ctrl=step_control)
            data = self._apis.mjx.step(self._m, data)
            q_rows.append(data.qpos)
            v_rows.append(data.qvel)

        t = jnp.asarray(np.arange(horizon + 1, dtype=np.float64) * float(dt))
        q_hist = jnp.stack(q_rows, axis=1)
        v_hist = jnp.stack(v_rows, axis=1)
        return t, q_hist, v_hist, u_hist

    def final_state_control_jacobian(
        self,
        controls: np.ndarray,
        *,
        horizon: int,
        dt: float,
        num_envs: int,
    ) -> np.ndarray:
        """Return ``d final_state / d controls`` for a batched MJX rollout."""
        control_seq = _normalise_controls(controls, horizon, num_envs)
        if control_seq is None:
            raise ValueError("controls are required for control jacobian")
        flat_shape = (num_envs * horizon * _NU,)
        jnp = self._apis.jnp

        def _final_state(flat_controls: Any) -> Any:
            reshaped = jnp.asarray(flat_controls, dtype=jnp.float64).reshape(
                num_envs, horizon, _NU
            )
            _t, q, v, _u = self.rollout_batch_arrays(
                reshaped,
                horizon=horizon,
                dt=dt,
                num_envs=num_envs,
            )
            return jnp.concatenate([q[:, -1, :], v[:, -1, :]], axis=1).reshape(-1)

        flat_controls = jnp.asarray(control_seq.reshape(flat_shape), dtype=jnp.float64)
        jacobian = self._apis.jax.jacrev(_final_state)(flat_controls)
        return np.asarray(jacobian, dtype=np.float64)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:  # noqa: ARG002 - protocol
        """Not supported by this MJX slice; use ``ode`` or ``mujoco``."""
        raise BackendCapabilityError(
            "mjx does not expose dense dynamics primitives in this adapter; "
            "use the ode or mujoco backend for M(q)/bias"
        )

    def bias_forces(
        self,
        q: np.ndarray,  # noqa: ARG002 - protocol
        v: np.ndarray,  # noqa: ARG002 - protocol
    ) -> np.ndarray:
        """Not supported by this MJX slice; use ``ode`` or ``mujoco``."""
        raise BackendCapabilityError(
            "mjx does not expose dense dynamics primitives in this adapter; "
            "use the ode or mujoco backend for M(q)/bias"
        )

    def _write_control(self, control: np.ndarray) -> None:
        self._data = _replace_data(self._data, ctrl=self._jnp_array(control))

    def _set_timestep(self, dt: float) -> None:
        if dt == self._dt:
            return
        self._dt = dt
        self._cpu_model.opt.timestep = dt
        self._m = self._apis.mjx.put_model(self._cpu_model)

    def _jnp_array(self, value: object) -> Any:
        return self._apis.jnp.asarray(value, dtype=self._apis.jnp.float64)

    def _device_kind(self) -> str:
        devices = getattr(self._apis.jax, "devices", None)
        if not callable(devices):
            return "unknown"
        try:
            first = devices()[0]
        except (IndexError, RuntimeError):
            return "unknown"
        return str(getattr(first, "platform", getattr(first, "device_kind", "unknown")))


def _load_mjx_apis() -> _MJXApis:
    try:
        return _MJXApis(
            jax=importlib.import_module("jax"),
            jnp=importlib.import_module("jax.numpy"),
            mujoco=importlib.import_module("mujoco"),
            mjx=importlib.import_module("mujoco.mjx"),
        )
    except ImportError as exc:
        raise ImportError(
            "MJX is not installed. Install with `pip install upstream-drift[mjx]`."
        ) from exc


def _validate_state(state: SimState) -> None:
    require(
        state.q.size == _NQ and state.v.size == _NV,
        f"state q/v must each have length {_NQ}",
    )


def _validate_rollout_args(horizon: int, dt: float, num_envs: int) -> None:
    if not isinstance(horizon, int) or horizon <= 0:
        raise ValueError(f"horizon must be a positive int, got {horizon!r}")
    if not isinstance(num_envs, int) or num_envs <= 0:
        raise ValueError(f"num_envs must be a positive int, got {num_envs!r}")
    if not isinstance(dt, (int, float)) or dt <= 0.0 or not np.isfinite(dt):
        raise ValueError(f"dt must be a positive finite float, got {dt!r}")


def _validate_initial_batch(q: Any, v: Any, num_envs: int) -> None:
    if tuple(q.shape) != (num_envs, _NQ):
        raise ValueError(f"initial_q must have shape ({num_envs}, {_NQ})")
    if tuple(v.shape) != (num_envs, _NV):
        raise ValueError(f"initial_v must have shape ({num_envs}, {_NV})")


def _normalise_controls(
    controls: np.ndarray | None,
    horizon: int,
    num_envs: int,
) -> np.ndarray | None:
    _validate_rollout_args(horizon, 1.0, num_envs)
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
    if not np.all(np.isfinite(arr)):
        raise ValueError("controls must be finite")
    return np.ascontiguousarray(arr, dtype=np.float64)


def _as_fixed_vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.shape != (size,):
        raise ValueError(f"{name} must have shape ({size},), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _replace_data(data: Any, **kwargs: Any) -> Any:
    replace = getattr(data, "replace", None)
    if callable(replace):
        return replace(**kwargs)
    raise TypeError("MJX data object must provide a replace(**kwargs) method")


def _trace_meta(device_kind: str) -> dict[str, object]:
    return {
        "device": "jax",
        "jax_device": device_kind,
        "mjcf_source": "GolfModelParams",
        "differentiable": True,
        "precision": "jax-default",
    }
