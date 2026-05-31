"""ZTCF / ZVCF counterfactual-acceleration primitives (epic task M7).

This module reproduces the **ZTCF** (zero-torque counterfactual) and **ZVCF**
(zero-velocity counterfactual) decompositions of the golf double-pendulum's
acceleration, expressed entirely in terms of the backend-agnostic
:class:`~simulation_backends.protocol.DynamicsProvider` primitives
(``mass_matrix`` and ``bias_forces``). Because the ODE reference backend and the
MuJoCo CPU backend both implement that Protocol, the *same* functions evaluate
on either engine and must agree (cross-validation, epic task M7).

Definitions (pointwise, at a single measured state ``(q, v)`` with control
``tau``)::

    ZTCF accel:  qddot = solve(M(q), -bias(q, v))         # the drift field f(x)
    ZVCF accel:  qddot = solve(M(q),  tau - bias(q, 0))   # velocity zeroed

For the planar double pendulum the bias force is ``C(q, v) v + g(q) + d(v)``.
At ``v = 0`` the Coriolis term (quadratic in velocity) and the viscous damping
term (linear in velocity) both vanish, so ``bias(q, 0) == g(q)``. The ZVCF
expression therefore reduces to ``solve(M, tau - g(q))``, matching the
analytical ground truth in
:meth:`PendulumPhysicsEngine.compute_zvcf`.

# AGENT-NOTE: These are POINTWISE / INSTANTANEOUS decompositions evaluated at
# each measured state along a trajectory -- they are NOT forward-integrated
# counterfactual rollouts. ``evaluate_ztcf_along_trajectory`` maps the ZTCF
# operator over the *measured* (q, v) samples and returns the instantaneous
# drift acceleration at each one; it does NOT integrate a zero-torque system
# forward in time. Do not "fix" this into a time integration -- the pointwise
# semantics are the whole point (epic task M7.3). The same caveat applies to
# every function in this module.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from typing import TYPE_CHECKING

import numpy as np

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    from .protocol import DynamicsProvider

logger = get_logger(__name__)

__all__ = [
    "CanonicalDynamicsTrajectory",
    "ZtcfZvcfResult",
    "drift_and_control_split",
    "evaluate_ztcf_along_trajectory",
    "evaluate_ztcf_zvcf_on_canonical_trajectory",
    "persist_ztcf_zvcf_analysis",
    "ztcf_acceleration",
    "zvcf_acceleration",
]

_CANONICAL_V2 = "canonical-v2"
_WORLD_Z_UP = "world_Zup"
_SI_UNITS = "SI"
_ANALYSIS_SCHEMA_VERSION = "2.0.0"


@dataclass(frozen=True)
class CanonicalDynamicsTrajectory:
    """Pointwise canonical-v2 state samples for ZTCF/ZVCF analysis.

    This is an analysis-boundary adapter: engine adapters convert native state
    layouts to ``canonical-v2`` before constructing this value, and the math in
    this module still runs only through ``DynamicsProvider`` primitives.

    ``q`` is configuration-space data and may be longer than ``v`` in
    canonical-v2 because a floating-base quaternion has one redundant
    coordinate (``nq = nv + 1``). ``v`` and ``tau`` live in tangent space.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray | None = None
    convention: str = _CANONICAL_V2
    frame: str = _WORLD_Z_UP
    units: str = _SI_UNITS
    meta: Mapping[str, object] | None = None

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float).reshape(-1)
        q = np.asarray(self.q, dtype=float)
        v = np.asarray(self.v, dtype=float)
        require(t.size > 0, "t must be non-empty", value=t.shape)
        require(q.ndim == 2, "q must be 2-D (T, nq)", value=q.shape)
        require(v.ndim == 2, "v must be 2-D (T, nv)", value=v.shape)
        require(
            q.shape[0] == t.size and v.shape[0] == t.size,
            "t, q, and v must share the same sample count",
            value=(t.shape, q.shape, v.shape),
        )
        require(q.shape[1] > 0, "q must have at least one coordinate", value=q.shape)
        require(v.shape[1] > 0, "v must have at least one velocity", value=v.shape)
        require(
            check_finite(t) and check_finite(q) and check_finite(v),
            "t, q, and v must contain only finite values",
        )

        tau = None
        if self.tau is not None:
            tau = np.asarray(self.tau, dtype=float)
            require(tau.ndim == 2, "tau must be 2-D (T, nv)", value=tau.shape)
            require(
                tau.shape == v.shape,
                f"tau must match v shape {v.shape}; got {tau.shape}",
                value=tau.shape,
            )
            require(check_finite(tau), "tau must contain only finite values")

        require(
            self.convention == _CANONICAL_V2,
            f"convention must be {_CANONICAL_V2!r}",
            value=self.convention,
        )
        require(
            self.frame == _WORLD_Z_UP,
            f"frame must be {_WORLD_Z_UP!r}",
            value=self.frame,
        )
        require(self.units == _SI_UNITS, f"units must be {_SI_UNITS!r}")

        object.__setattr__(self, "t", t)
        object.__setattr__(self, "q", q)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "tau", tau)
        object.__setattr__(self, "meta", dict(self.meta or {}))

    @property
    def num_steps(self) -> int:
        """Number of sampled states."""
        return int(self.t.shape[0])

    @property
    def tangent_dim(self) -> int:
        """Dimension of tangent-space vectors (``nv``)."""
        return int(self.v.shape[1])

    def controls_or_zeros(self) -> np.ndarray:
        """Return applied controls, defaulting passive samples to zeros."""
        if self.tau is None:
            return np.zeros_like(self.v)
        return self.tau.copy()


@dataclass(frozen=True)
class ZtcfZvcfResult:
    """Pointwise ZTCF/ZVCF and affine drift/control analysis result."""

    t: np.ndarray
    ztcf_acceleration: np.ndarray
    zvcf_acceleration: np.ndarray
    drift_acceleration: np.ndarray
    control_acceleration: np.ndarray
    convention: str = _CANONICAL_V2
    frame: str = _WORLD_Z_UP
    units: str = _SI_UNITS

    def __post_init__(self) -> None:
        t = np.asarray(self.t, dtype=float).reshape(-1)
        arrays = {
            "ztcf_acceleration": np.asarray(self.ztcf_acceleration, dtype=float),
            "zvcf_acceleration": np.asarray(self.zvcf_acceleration, dtype=float),
            "drift_acceleration": np.asarray(self.drift_acceleration, dtype=float),
            "control_acceleration": np.asarray(self.control_acceleration, dtype=float),
        }
        require(t.size > 0, "t must be non-empty", value=t.shape)
        for name, value in arrays.items():
            require(value.ndim == 2, f"{name} must be 2-D (T, nv)", value=value.shape)
            require(
                value.shape[0] == t.size,
                f"{name} sample count must match t",
                value=(value.shape, t.shape),
            )
            require(check_finite(value), f"{name} must contain only finite values")
        expected_shape = arrays["ztcf_acceleration"].shape
        for name, value in arrays.items():
            require(
                value.shape == expected_shape,
                f"{name} must share shape {expected_shape}; got {value.shape}",
                value=value.shape,
            )
        require(
            self.convention == _CANONICAL_V2,
            f"convention must be {_CANONICAL_V2!r}",
            value=self.convention,
        )
        require(
            self.frame == _WORLD_Z_UP,
            f"frame must be {_WORLD_Z_UP!r}",
            value=self.frame,
        )
        require(self.units == _SI_UNITS, f"units must be {_SI_UNITS!r}")

        object.__setattr__(self, "t", t)
        for name, value in arrays.items():
            object.__setattr__(self, name, value)


def _as_state_vector(name: str, value: np.ndarray) -> np.ndarray:
    """Coerce ``value`` to a finite 1-D float vector (shared precondition guard).

    Args:
        name: Argument name, used in diagnostic messages.
        value: Array-like to validate and coerce.

    Returns:
        A contiguous 1-D ``float`` :class:`numpy.ndarray`.

    Raises:
        ValueError: If ``value`` is not 1-D or contains non-finite entries.
    """
    arr = np.asarray(value, dtype=float).reshape(-1)
    require(arr.size > 0, f"{name} must be non-empty", value=arr.shape)
    require(check_finite(arr), f"{name} must contain only finite values", value=value)
    return arr


def _solve_mass(
    provider: DynamicsProvider, q: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve ``M(q) x = rhs`` using the provider's inertia matrix.

    Args:
        provider: Any object satisfying :class:`DynamicsProvider` (LOD: only its
            ``mass_matrix`` method is touched here).
        q: Joint positions ``(n,)`` at which to evaluate ``M``.
        rhs: Right-hand side ``(n,)``.

    Returns:
        The solution ``x = M(q)^-1 rhs``, shape ``(n,)``.

    Raises:
        ValueError: If ``M(q)`` is not an ``(n, n)`` matrix matching ``rhs`` or
            the resulting acceleration is non-finite (e.g. singular inertia).
    """
    mass = np.asarray(provider.mass_matrix(q), dtype=float)
    n = rhs.shape[0]
    require(
        mass.shape == (n, n),
        f"mass_matrix(q) must be ({n}, {n}); got {mass.shape}",
        value=mass.shape,
    )
    qddot = np.linalg.solve(mass, rhs)
    # Postcondition: a finite acceleration (a singular/ill-conditioned M would
    # surface here rather than silently propagating NaN downstream).
    require(
        check_finite(qddot),
        "solve(M(q), rhs) produced non-finite acceleration (singular inertia?)",
        value=qddot,
    )
    return qddot


def ztcf_acceleration(
    provider: DynamicsProvider, q: np.ndarray, v: np.ndarray
) -> np.ndarray:
    """Zero-torque counterfactual (ZTCF) acceleration at a single state.

    Computes the *instantaneous* drift-field acceleration ``f(x)`` -- the
    acceleration the mechanism would experience at this exact ``(q, v)`` with
    **zero applied torque**::

        qddot = solve(M(q), -bias(q, v))

    # AGENT-NOTE: Pointwise / instantaneous, evaluated at the supplied measured
    # state. This is NOT a forward-integrated zero-torque rollout.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces`` (the
            :class:`DynamicsProvider` Protocol; ODE or MuJoCo CPU backend).
        q: Joint positions ``(n,)`` [rad].
        v: Joint velocities ``(n,)`` [rad/s].

    Returns:
        Drift acceleration ``(n,)`` [rad/s^2].

    Postcondition:
        Result has shape ``(n,)`` and is finite.

    Raises:
        ValueError: If ``q``/``v`` are empty, non-finite, or differ in length.
    """
    q_arr = _as_state_vector("q", q)
    v_arr = _as_state_vector("v", v)
    bias = np.asarray(provider.bias_forces(q_arr, v_arr), dtype=float)
    require(
        bias.shape == v_arr.shape,
        f"bias_forces(q, v) must be {v_arr.shape}; got {bias.shape}",
        value=bias.shape,
    )
    return _solve_mass(provider, q_arr, -bias)


def zvcf_acceleration(
    provider: DynamicsProvider, q: np.ndarray, tau: np.ndarray
) -> np.ndarray:
    """Zero-velocity counterfactual (ZVCF) acceleration at a single state.

    Computes the *instantaneous* acceleration with the **velocity zeroed but the
    applied control preserved**::

        qddot = solve(M(q), tau - bias(q, 0))

    With ``v = 0`` the Coriolis and viscous-damping contributions vanish, so
    ``bias(q, 0)`` reduces to the gravity vector ``g(q)`` -- this matches
    :meth:`PendulumPhysicsEngine.compute_zvcf` (which uses ``-g(q) + tau``)
    exactly.

    # AGENT-NOTE: Pointwise / instantaneous, evaluated at the supplied measured
    # position with velocity set to zero. This is NOT a forward-integrated
    # zero-velocity rollout.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces`` (the
            :class:`DynamicsProvider` Protocol; ODE or MuJoCo CPU backend).
        q: Joint positions ``(n,)`` [rad].
        tau: Applied generalised control/torque ``(n,)`` [N*m].

    Returns:
        Zero-velocity acceleration ``(n,)`` [rad/s^2].

    Postcondition:
        Result has shape ``(n,)`` and is finite.

    Raises:
        ValueError: If ``q``/``tau`` are empty, non-finite, or differ in length.
    """
    q_arr = _as_state_vector("q", q)
    tau_arr = _as_state_vector("tau", tau)
    bias_zero_v = np.asarray(
        provider.bias_forces(q_arr, np.zeros_like(tau_arr)), dtype=float
    )
    require(
        bias_zero_v.shape == tau_arr.shape,
        f"bias_forces(q, 0) must be {tau_arr.shape}; got {bias_zero_v.shape}",
        value=bias_zero_v.shape,
    )
    return _solve_mass(provider, q_arr, tau_arr - bias_zero_v)


def evaluate_ztcf_along_trajectory(
    provider: DynamicsProvider, q_traj: np.ndarray, v_traj: np.ndarray
) -> np.ndarray:
    """Evaluate the ZTCF acceleration pointwise at every sampled state.

    Maps :func:`ztcf_acceleration` over each ``(q_traj[t], v_traj[t])`` sample
    and stacks the results. The output row ``t`` is the *instantaneous* drift
    acceleration at measured sample ``t``.

    # AGENT-NOTE: This is a POINTWISE evaluation along an ALREADY-MEASURED
    # trajectory -- each row is the instantaneous zero-torque drift acceleration
    # at that measured state. It is NOT a forward integration of a zero-torque
    # system, and must not be "fixed" into one (epic task M7.3). The measured
    # trajectory is the input; the drift field sampled on it is the output.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
        q_traj: Position history, shape ``(T, n)`` [rad].
        v_traj: Velocity history, shape ``(T, n)`` [rad/s].

    Returns:
        ZTCF accelerations, shape ``(T, n)`` [rad/s^2].

    Postcondition:
        Output shape equals ``q_traj``'s shape and every entry is finite.

    Raises:
        ValueError: If ``q_traj``/``v_traj`` are not 2-D, disagree in shape, or
            contain non-finite entries.
    """
    q_mat = np.asarray(q_traj, dtype=float)
    v_mat = np.asarray(v_traj, dtype=float)
    require(q_mat.ndim == 2, "q_traj must be 2-D (T, n)", value=q_mat.shape)
    require(v_mat.ndim == 2, "v_traj must be 2-D (T, n)", value=v_mat.shape)
    require(
        q_mat.shape[0] == v_mat.shape[0],
        "q_traj and v_traj must share sample count",
        value=(q_mat.shape, v_mat.shape),
    )
    require(
        check_finite(q_mat) and check_finite(v_mat),
        "q_traj and v_traj must contain only finite values",
    )

    out = np.empty_like(v_mat)
    for idx in range(q_mat.shape[0]):
        out[idx] = ztcf_acceleration(provider, q_mat[idx], v_mat[idx])
    return out


def drift_and_control_split(
    provider: DynamicsProvider,
    q: np.ndarray,
    v: np.ndarray,
    tau: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Split the acceleration into its drift and control contributions.

    Because the equation of motion is affine in the control, the total
    acceleration ``solve(M, tau - bias(q, v))`` decomposes additively into:

    * the **drift** term ``f = solve(M(q), -bias(q, v))`` -- identical to
      :func:`ztcf_acceleration`; and
    * the **control** term ``solve(M(q), tau)``.

    Their sum is the actual acceleration under ``tau``, so this is the
    instantaneous ``qddot = f(x) + g(x) u`` decomposition at the measured state.

    # AGENT-NOTE: Pointwise / instantaneous decomposition at a single measured
    # state -- NOT a forward-integrated counterfactual. Do not turn this into a
    # time integration (epic task M7.3).

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces``.
        q: Joint positions ``(n,)`` [rad].
        v: Joint velocities ``(n,)`` [rad/s].
        tau: Applied generalised control/torque ``(n,)`` [N*m].

    Returns:
        A ``(drift, control)`` tuple, each shape ``(n,)`` [rad/s^2], where
        ``drift`` is the ZTCF acceleration and ``control`` is ``M(q)^-1 tau``.

    Postcondition:
        ``drift + control`` equals ``solve(M(q), tau - bias(q, v))``.

    Raises:
        ValueError: If ``q``/``v``/``tau`` are empty, non-finite, or differ in
            length.
    """
    drift = ztcf_acceleration(provider, q, v)
    q_arr = _as_state_vector("q", q)
    tau_arr = _as_state_vector("tau", tau)
    control = _solve_mass(provider, q_arr, tau_arr)
    return drift, control


def evaluate_ztcf_zvcf_on_canonical_trajectory(
    provider: DynamicsProvider,
    trajectory: CanonicalDynamicsTrajectory,
) -> ZtcfZvcfResult:
    """Evaluate pointwise ZTCF/ZVCF on canonical-v2 state samples.

    Args:
        provider: Backend exposing ``mass_matrix`` and ``bias_forces`` in the
            same canonical-v2 coordinates as ``trajectory.q``/``trajectory.v``.
        trajectory: Canonical-v2 state samples and optional controls.

    Returns:
        Pointwise accelerations and affine split arrays, all shape ``(T, nv)``.

    Postcondition:
        ``drift_acceleration`` equals ``ztcf_acceleration`` sample-by-sample.
    """
    require(
        isinstance(trajectory, CanonicalDynamicsTrajectory),
        "trajectory must be a CanonicalDynamicsTrajectory",
        value=type(trajectory).__name__,
    )
    tau = trajectory.controls_or_zeros()
    ztcf = np.empty_like(trajectory.v)
    zvcf = np.empty_like(trajectory.v)
    drift = np.empty_like(trajectory.v)
    control = np.empty_like(trajectory.v)

    for idx in range(trajectory.num_steps):
        q_i = trajectory.q[idx]
        v_i = trajectory.v[idx]
        tau_i = tau[idx]
        ztcf[idx] = ztcf_acceleration(provider, q_i, v_i)
        zvcf[idx] = zvcf_acceleration(provider, q_i, tau_i)
        drift[idx], control[idx] = drift_and_control_split(provider, q_i, v_i, tau_i)

    return ZtcfZvcfResult(
        t=trajectory.t,
        ztcf_acceleration=ztcf,
        zvcf_acceleration=zvcf,
        drift_acceleration=drift,
        control_acceleration=control,
        convention=trajectory.convention,
        frame=trajectory.frame,
        units=trajectory.units,
    )


def _validate_path(path: str | os.PathLike[str]) -> str:
    """Return ``path`` as a filesystem string after basic DbC validation."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be str or os.PathLike, got {type(path).__name__}")
    path_str = os.fspath(path)
    require(bool(path_str.strip()), "path must be a non-empty filesystem path")
    return path_str


def _write_scalar_meta(handle: Any, meta: Mapping[str, object]) -> None:
    """Persist scalar metadata as root attributes using the CC-4 meta prefix."""
    for key, value in meta.items():
        if isinstance(value, (str, bool, int, float)):
            handle.attrs[f"meta_{key}"] = value


def persist_ztcf_zvcf_analysis(
    trajectory: CanonicalDynamicsTrajectory,
    result: ZtcfZvcfResult,
    path: str | os.PathLike[str],
    *,
    backend: str,
) -> None:
    """Persist canonical-v2 ZTCF/ZVCF arrays into a CC-4-style HDF5 file.

    The file uses the shared CC-4 root attributes and required ``t``/``q``/``v``
    datasets, then adds the analysis datasets ``ztcf_acceleration``,
    ``zvcf_acceleration``, ``drift_acceleration``, and
    ``control_acceleration``. It is intentionally an analysis artifact rather
    than a forward rollout trace.
    """
    require(
        isinstance(trajectory, CanonicalDynamicsTrajectory),
        "trajectory must be a CanonicalDynamicsTrajectory",
        value=type(trajectory).__name__,
    )
    require(
        isinstance(result, ZtcfZvcfResult),
        "result must be a ZtcfZvcfResult",
        value=type(result).__name__,
    )
    require(
        bool(backend.strip()),
        "backend must be a non-empty backend identifier",
        value=backend,
    )
    require(
        result.t.shape == trajectory.t.shape
        and result.ztcf_acceleration.shape == trajectory.v.shape,
        "result must match trajectory sample count and tangent dimension",
        value=(result.t.shape, result.ztcf_acceleration.shape, trajectory.v.shape),
    )
    path_str = _validate_path(path)
    import h5py

    with h5py.File(path_str, "w") as handle:
        handle.attrs["schema_version"] = _ANALYSIS_SCHEMA_VERSION
        handle.attrs["backend"] = backend
        handle.attrs["dt"] = (
            float(np.median(np.diff(trajectory.t))) if trajectory.t.size > 1 else 0.0
        )
        handle.attrs["kind"] = "ztcf_zvcf_analysis"
        handle.attrs["convention"] = trajectory.convention
        handle.attrs["frame"] = trajectory.frame
        handle.attrs["units"] = trajectory.units
        _write_scalar_meta(handle, trajectory.meta or {})

        handle.create_dataset("t", data=trajectory.t)
        handle.create_dataset("q", data=trajectory.q)
        handle.create_dataset("v", data=trajectory.v)
        handle.create_dataset("u", data=trajectory.controls_or_zeros())
        handle.create_dataset("ztcf_acceleration", data=result.ztcf_acceleration)
        handle.create_dataset("zvcf_acceleration", data=result.zvcf_acceleration)
        handle.create_dataset("drift_acceleration", data=result.drift_acceleration)
        handle.create_dataset("control_acceleration", data=result.control_acceleration)
