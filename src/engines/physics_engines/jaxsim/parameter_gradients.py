"""JAX-backed parameter sensitivities for the JaxSim gradient gate.

The functions in this module evaluate the zero-torque counterfactual (ZTCF)
drift field pointwise along measured states and differentiate it with respect
to anthropometric parameters. They intentionally do not integrate a
counterfactual rollout: every output row is the instantaneous sensitivity at
the matching input ``(q, v)`` sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

PARAMETER_NAMES: tuple[str, ...] = (
    "upper_length_m",
    "upper_mass_kg",
    "lower_length_m",
    "shaft_mass_kg",
    "clubhead_mass_kg",
)
"""Anthropometric parameter vector order used by this module."""

DEFAULT_PARAMETER_VECTOR: NDArray[np.float64] = np.array(
    [0.72, 4.5, 1.10, 0.18, 0.20],
    dtype=np.float64,
)
"""Stable defaults used by examples without importing optional pendulum code."""

_STATE_SIZE = 2
_PARAMETER_SIZE = len(PARAMETER_NAMES)
_GRAVITY_M_S2 = 9.80665
_DAMPING = np.array([0.05, 0.02], dtype=np.float64)

GradientMode = Literal["forward", "reverse"]


@dataclass(frozen=True)
class ParameterGradientValidation:
    """Comparison between JAX autodiff and finite-difference gradients."""

    autodiff_jacobian: NDArray[np.float64]
    finite_difference_jacobian: NDArray[np.float64]
    max_abs_error: float
    max_rel_error: float


def ztcf_drift_field(
    parameter_vector: ArrayLike,
    q: ArrayLike,
    v: ArrayLike,
) -> NDArray[np.float64]:
    """Evaluate the pointwise ZTCF drift field for one measured state.

    Args:
        parameter_vector: Five positive anthropometric parameters in
            :data:`PARAMETER_NAMES` order.
        q: Two joint positions ``[theta_shoulder, theta_wrist]``.
        v: Two joint velocities ``[omega_shoulder, omega_wrist]``.

    Returns:
        Drift acceleration ``solve(M(q), -bias(q, v))`` as a ``(2,)`` array.
    """

    jnp, _jax = _require_jax()
    params = _as_jax_parameter_vector(parameter_vector, jnp)
    q_vec = _as_jax_state_vector(q, "q", jnp)
    v_vec = _as_jax_state_vector(v, "v", jnp)
    return np.asarray(_ztcf_drift_field_jax(params, q_vec, v_vec, jnp), dtype=float)


def parameter_jacobian(
    parameter_vector: ArrayLike,
    q: ArrayLike,
    v: ArrayLike,
    *,
    mode: GradientMode = "forward",
) -> NDArray[np.float64]:
    """Differentiate the ZTCF drift field with respect to parameters.

    Args:
        parameter_vector: Five positive anthropometric parameters.
        q: Two joint positions.
        v: Two joint velocities.
        mode: ``"forward"`` for :func:`jax.jacfwd`, ``"reverse"`` for
            :func:`jax.jacrev`.

    Returns:
        Jacobian with shape ``(2, 5)``: drift acceleration rows by parameter
        columns.
    """

    jnp, jax = _require_jax()
    params = _as_jax_parameter_vector(parameter_vector, jnp)
    q_vec = _as_jax_state_vector(q, "q", jnp)
    v_vec = _as_jax_state_vector(v, "v", jnp)

    def drift(p: Any) -> Any:
        return _ztcf_drift_field_jax(p, q_vec, v_vec, jnp)

    if mode == "forward":
        jacobian = jax.jacfwd(drift)(params)
    elif mode == "reverse":
        jacobian = jax.jacrev(drift)(params)
    else:
        raise ValueError(f"unsupported gradient mode: {mode!r}")
    return np.asarray(jacobian, dtype=float)


def finite_difference_parameter_jacobian(
    parameter_vector: ArrayLike,
    q: ArrayLike,
    v: ArrayLike,
    *,
    step: float = 1.0e-5,
) -> NDArray[np.float64]:
    """Compute a central finite-difference parameter Jacobian.

    Args:
        parameter_vector: Five positive anthropometric parameters.
        q: Two joint positions.
        v: Two joint velocities.
        step: Positive perturbation size applied per parameter.

    Returns:
        Numeric Jacobian with shape ``(2, 5)``.
    """

    params = _as_numpy_parameter_vector(parameter_vector)
    q_vec = _as_numpy_state_vector(q, "q")
    v_vec = _as_numpy_state_vector(v, "v")
    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("step must be positive and finite")

    jacobian: NDArray[np.float64] = np.empty(
        (_STATE_SIZE, _PARAMETER_SIZE), dtype=np.float64
    )
    for index in range(_PARAMETER_SIZE):
        delta: NDArray[np.float64] = np.zeros(_PARAMETER_SIZE, dtype=np.float64)
        delta[index] = step
        plus = ztcf_drift_field(params + delta, q_vec, v_vec)
        minus = ztcf_drift_field(params - delta, q_vec, v_vec)
        jacobian[:, index] = (plus - minus) / (2.0 * step)
    return jacobian


def validate_parameter_jacobian(
    parameter_vector: ArrayLike,
    q: ArrayLike,
    v: ArrayLike,
    *,
    mode: GradientMode = "forward",
    finite_difference_step: float = 1.0e-5,
) -> ParameterGradientValidation:
    """Validate autodiff gradients against central finite differences."""

    autodiff = parameter_jacobian(parameter_vector, q, v, mode=mode)
    finite_diff = finite_difference_parameter_jacobian(
        parameter_vector,
        q,
        v,
        step=finite_difference_step,
    )
    abs_error = np.abs(autodiff - finite_diff)
    scale = np.maximum(np.abs(finite_diff), 1.0e-12)
    rel_error = abs_error / scale
    return ParameterGradientValidation(
        autodiff_jacobian=autodiff,
        finite_difference_jacobian=finite_diff,
        max_abs_error=float(np.max(abs_error)),
        max_rel_error=float(np.max(rel_error)),
    )


def evaluate_ztcf_parameter_sensitivity_along_trajectory(
    parameter_vector: ArrayLike,
    q_traj: ArrayLike,
    v_traj: ArrayLike,
    *,
    mode: GradientMode = "forward",
) -> NDArray[np.float64]:
    """Evaluate parameter sensitivities pointwise along measured states.

    Args:
        parameter_vector: Five positive anthropometric parameters.
        q_traj: Measured joint positions with shape ``(T, 2)``.
        v_traj: Measured joint velocities with shape ``(T, 2)``.
        mode: JAX autodiff mode.

    Returns:
        Array with shape ``(T, 2, 5)``.
    """

    params = _as_numpy_parameter_vector(parameter_vector)
    q_mat = _as_trajectory(q_traj, "q_traj")
    v_mat = _as_trajectory(v_traj, "v_traj")
    if q_mat.shape != v_mat.shape:
        raise ValueError(
            f"q_traj and v_traj must share shape; got {q_mat.shape} vs {v_mat.shape}"
        )
    out = np.empty((q_mat.shape[0], _STATE_SIZE, _PARAMETER_SIZE), dtype=np.float64)
    for row in range(q_mat.shape[0]):
        out[row] = parameter_jacobian(params, q_mat[row], v_mat[row], mode=mode)
    return out


def _ztcf_drift_field_jax(
    parameter_vector: Any,
    q: Any,
    v: Any,
    jnp: Any,
) -> Any:
    """JAX-traceable drift field implementation."""

    upper_length, upper_mass, lower_length, shaft_mass, clubhead_mass = parameter_vector
    lower_mass = shaft_mass + clubhead_mass
    upper_com = 0.45 * upper_length
    lower_com = (
        shaft_mass * (0.5 * lower_length) + clubhead_mass * lower_length
    ) / lower_mass
    upper_inertia = upper_mass * upper_length**2 / 12.0
    shaft_inertia_about_lower_com = (
        shaft_mass * lower_length**2 / 12.0
        + shaft_mass * (lower_com - 0.5 * lower_length) ** 2
    )
    clubhead_inertia_about_lower_com = clubhead_mass * (lower_length - lower_com) ** 2
    lower_inertia = shaft_inertia_about_lower_com + clubhead_inertia_about_lower_com

    theta1, theta2 = q
    omega1, omega2 = v
    cos_theta2 = jnp.cos(theta2)
    sin_theta2 = jnp.sin(theta2)

    coupling = lower_mass * upper_length * lower_com
    m11 = (
        upper_inertia
        + lower_inertia
        + upper_mass * upper_com**2
        + lower_mass
        * (upper_length**2 + lower_com**2 + 2.0 * upper_length * lower_com * cos_theta2)
    )
    m12 = lower_inertia + lower_mass * (
        lower_com**2 + upper_length * lower_com * cos_theta2
    )
    m22 = lower_inertia + lower_mass * lower_com**2
    mass_matrix = jnp.array([[m11, m12], [m12, m22]])

    coriolis = jnp.array(
        [
            -coupling * sin_theta2 * (2.0 * omega1 * omega2 + omega2**2),
            coupling * sin_theta2 * omega1**2,
        ]
    )
    gravity = jnp.array(
        [
            (upper_mass * upper_com + lower_mass * upper_length)
            * _GRAVITY_M_S2
            * jnp.sin(theta1)
            + lower_mass * lower_com * _GRAVITY_M_S2 * jnp.sin(theta1 + theta2),
            lower_mass * lower_com * _GRAVITY_M_S2 * jnp.sin(theta1 + theta2),
        ]
    )
    damping = jnp.asarray(_DAMPING) * v
    return jnp.linalg.solve(mass_matrix, -(coriolis + gravity + damping))


def _require_jax() -> tuple[Any, Any]:
    try:
        import jax  # type: ignore[import-not-found]

        jax.config.update("jax_enable_x64", True)
        import jax.numpy as jnp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "JAX is required for JaxSim parameter gradients. "
            "Install with `pip install upstream-drift[jaxsim]`."
        ) from exc
    return jnp, jax


def _as_numpy_parameter_vector(value: ArrayLike) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (_PARAMETER_SIZE,):
        raise ValueError(f"parameter_vector must have shape ({_PARAMETER_SIZE},)")
    if not np.all(np.isfinite(array)):
        raise ValueError("parameter_vector must contain only finite values")
    if np.any(array <= 0.0):
        raise ValueError("parameter_vector entries must be positive")
    return array


def _as_numpy_state_vector(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.shape != (_STATE_SIZE,):
        raise ValueError(f"{name} must have shape ({_STATE_SIZE},)")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_jax_parameter_vector(value: ArrayLike, jnp: Any) -> Any:
    return jnp.asarray(_as_numpy_parameter_vector(value))


def _as_jax_state_vector(value: ArrayLike, name: str, jnp: Any) -> Any:
    return jnp.asarray(_as_numpy_state_vector(value, name))


def _as_trajectory(value: ArrayLike, name: str) -> NDArray[np.float64]:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != _STATE_SIZE:
        raise ValueError(f"{name} must have shape (T, {_STATE_SIZE})")
    if array.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one sample")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array
