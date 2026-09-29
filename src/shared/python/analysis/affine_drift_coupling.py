"""AffineDrift coupling for canonical double-pendulum kinematics.

The coupling is intentionally pointwise: it samples the double-pendulum drift
and control-affine terms on an already estimated kinematic trajectory. It does
not integrate a new counterfactual trajectory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import h5py
import numpy as np

from src.shared.python.core.contracts import check_finite, require
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.ztcf_zvcf import ztcf_acceleration

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.shared.python.simulation_backends.protocol import DynamicsProvider

__all__ = [
    "AffineDriftCouplingResult",
    "couple_trace_to_affine_drift",
    "extract_double_pendulum_kinematics",
    "read_affine_drift_coupling",
    "write_affine_drift_coupling",
]

_RESULT_SCHEMA_VERSION = "1.0.0"
_DEFAULT_DOF = 2


@dataclass(frozen=True)
class AffineDriftCouplingResult:
    """Pointwise AffineDrift surface sampled from estimated kinematics.

    Attributes:
        t: Sample times, shape ``(T,)`` [s].
        q: Extracted double-pendulum joint positions, shape ``(T, 2)`` [rad].
        v: Extracted double-pendulum joint velocities, shape ``(T, 2)`` [rad/s].
        tau: Extracted applied torques, shape ``(T, 2)`` [N*m].
        drift_acceleration: Zero-torque acceleration ``f_v(x)``, shape
            ``(T, 2)`` [rad/s^2].
        control_acceleration: Applied-control acceleration contribution
            ``M(q)^-1 tau``, shape ``(T, 2)`` [rad/s^2].
        total_acceleration: ``drift_acceleration + control_acceleration``.
        affine_drift: First-order state drift ``[v, drift_acceleration]``,
            shape ``(T, 4)``.
        affine_control_matrix: Control map ``g(x)`` for state
            ``x=[q0,q1,v0,v1]``, shape ``(T, 4, 2)``. The top position rows are
            zero and the bottom rows are ``M(q)^-1``.
        source_backend: Backend label from the source :class:`Trace`.
    """

    t: np.ndarray
    q: np.ndarray
    v: np.ndarray
    tau: np.ndarray
    drift_acceleration: np.ndarray
    control_acceleration: np.ndarray
    total_acceleration: np.ndarray
    affine_drift: np.ndarray
    affine_control_matrix: np.ndarray
    source_backend: str = "unknown"
    schema_version: str = _RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate coupling result arrays are finite and shape-consistent."""
        object.__setattr__(self, "t", np.asarray(self.t, dtype=float).reshape(-1))
        for name in (
            "q",
            "v",
            "tau",
            "drift_acceleration",
            "control_acceleration",
            "total_acceleration",
            "affine_drift",
        ):
            value = np.asarray(getattr(self, name), dtype=float)
            object.__setattr__(self, name, np.atleast_2d(value))
        control_matrix = np.asarray(self.affine_control_matrix, dtype=float)
        object.__setattr__(self, "affine_control_matrix", control_matrix)

        n = self.t.shape[0]
        dof = self.q.shape[1]
        require(dof == _DEFAULT_DOF, "q must have shape (T, 2)", value=self.q.shape)
        for name in (
            "q",
            "v",
            "tau",
            "drift_acceleration",
            "control_acceleration",
            "total_acceleration",
        ):
            value = getattr(self, name)
            require(
                value.shape == (n, dof),
                f"{name} must have shape ({n}, {dof}); got {value.shape}",
                value=value.shape,
            )
            require(check_finite(value), f"{name} must be finite", value=value)
        require(
            self.affine_drift.shape == (n, dof * 2),
            f"affine_drift must have shape ({n}, {dof * 2}); "
            f"got {self.affine_drift.shape}",
            value=self.affine_drift.shape,
        )
        require(
            self.affine_control_matrix.shape == (n, dof * 2, dof),
            f"affine_control_matrix must have shape ({n}, {dof * 2}, {dof}); "
            f"got {self.affine_control_matrix.shape}",
            value=self.affine_control_matrix.shape,
        )
        require(
            check_finite(self.t)
            and check_finite(self.affine_drift)
            and check_finite(self.affine_control_matrix),
            "all coupling arrays must be finite",
        )


def extract_double_pendulum_kinematics(
    trace: Trace,
    *,
    q_indices: Sequence[int] | None = None,
    v_indices: Sequence[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract the two double-pendulum coordinates from a Trace.

    The default conversion is deterministic for both accepted layouts:

    * 2-DOF double-pendulum traces use all ``q``/``v`` columns.
    * canonical-v2-style floating-base traces use the last two internal-joint
      columns from ``q`` and ``v``.

    Args:
        trace: Estimated kinematics in the shared Trace schema.
        q_indices: Optional explicit source ``q`` columns.
        v_indices: Optional explicit source ``v`` columns.

    Returns:
        ``(q, v)`` arrays, each shape ``(T, 2)``.

    Raises:
        TypeError: If ``trace`` is not a Trace.
        ValueError: If the selected columns are invalid or non-finite.
    """
    if not isinstance(trace, Trace):
        raise TypeError(f"trace must be Trace, got {type(trace).__name__}")
    q_cols = _resolve_indices("q_indices", q_indices, trace.q.shape[1])
    v_cols = _resolve_indices("v_indices", v_indices, trace.v.shape[1])
    q = trace.q[:, q_cols]
    v = trace.v[:, v_cols]
    _require_pair("extracted q", "extracted v", q, v)
    return q.copy(), v.copy()


def couple_trace_to_affine_drift(
    provider: DynamicsProvider,
    trace: Trace,
    *,
    q_indices: Sequence[int] | None = None,
    v_indices: Sequence[int] | None = None,
    control_indices: Sequence[int] | None = None,
) -> AffineDriftCouplingResult:
    """Couple estimated kinematics to a double-pendulum AffineDrift surface.

    Args:
        provider: Dynamics provider exposing ``mass_matrix`` and ``bias_forces``.
        trace: Source kinematics, usually from canonical-v2 estimation or a
            double-pendulum backend rollout.
        q_indices: Optional source ``q`` columns for the two pendulum angles.
        v_indices: Optional source ``v`` columns for the two pendulum speeds.
        control_indices: Optional source ``u``/``torques`` columns for torques.

    Returns:
        An :class:`AffineDriftCouplingResult` containing pointwise drift and
        control-affine terms.

    Raises:
        ValueError: If selected kinematics or controls are not finite, or if the
            provider returns malformed dynamics primitives.
    """
    q, v = extract_double_pendulum_kinematics(
        trace, q_indices=q_indices, v_indices=v_indices
    )
    tau = _extract_control(trace, control_indices)
    drift = np.empty_like(q)
    control = np.empty_like(q)
    control_matrix = np.zeros((trace.num_steps, _DEFAULT_DOF * 2, _DEFAULT_DOF))

    for idx in range(trace.num_steps):
        drift[idx] = ztcf_acceleration(provider, q[idx], v[idx])
        inverse_mass = _inverse_mass(provider, q[idx])
        control[idx] = inverse_mass @ tau[idx]
        control_matrix[idx, _DEFAULT_DOF:, :] = inverse_mass

    total = drift + control
    affine_drift = np.column_stack((v, drift))
    return AffineDriftCouplingResult(
        t=trace.t.copy(),
        q=q,
        v=v,
        tau=tau,
        drift_acceleration=drift,
        control_acceleration=control,
        total_acceleration=total,
        affine_drift=affine_drift,
        affine_control_matrix=control_matrix,
        source_backend=trace.backend,
    )


def write_affine_drift_coupling(
    result: AffineDriftCouplingResult, path: str | os.PathLike[str]
) -> None:
    """Persist an AffineDrift coupling result to HDF5."""
    _validate_path(path)
    with h5py.File(os.fspath(path), "w") as handle:
        handle.attrs["schema_version"] = result.schema_version
        handle.attrs["kind"] = "affine_drift_coupling"
        handle.attrs["source_backend"] = result.source_backend
        for name in (
            "t",
            "q",
            "v",
            "tau",
            "drift_acceleration",
            "control_acceleration",
            "total_acceleration",
            "affine_drift",
            "affine_control_matrix",
        ):
            handle.create_dataset(name, data=getattr(result, name))


def read_affine_drift_coupling(
    path: str | os.PathLike[str],
) -> AffineDriftCouplingResult:
    """Read an HDF5 coupling artifact written by ``write_affine_drift_coupling``."""
    _validate_path(path)
    with h5py.File(os.fspath(path), "r") as handle:
        kind = handle.attrs.get("kind", "")
        if kind != "affine_drift_coupling":
            raise ValueError(f"expected affine_drift_coupling artifact, got {kind!r}")
        source_backend = str(handle.attrs.get("source_backend", "unknown"))
        schema_version = str(handle.attrs.get("schema_version", _RESULT_SCHEMA_VERSION))
        return AffineDriftCouplingResult(
            t=np.asarray(handle["t"][()], dtype=float),
            q=np.asarray(handle["q"][()], dtype=float),
            v=np.asarray(handle["v"][()], dtype=float),
            tau=np.asarray(handle["tau"][()], dtype=float),
            drift_acceleration=np.asarray(
                handle["drift_acceleration"][()], dtype=float
            ),
            control_acceleration=np.asarray(
                handle["control_acceleration"][()], dtype=float
            ),
            total_acceleration=np.asarray(
                handle["total_acceleration"][()], dtype=float
            ),
            affine_drift=np.asarray(handle["affine_drift"][()], dtype=float),
            affine_control_matrix=np.asarray(
                handle["affine_control_matrix"][()], dtype=float
            ),
            source_backend=source_backend,
            schema_version=schema_version,
        )


def _resolve_indices(
    name: str, indices: Sequence[int] | None, width: int
) -> tuple[int, int]:
    """Resolve explicit or default two-column index selection."""
    if indices is None:
        require(
            width >= _DEFAULT_DOF,
            f"{name} source width must be at least {_DEFAULT_DOF}; got {width}",
            value=width,
        )
        if width == _DEFAULT_DOF:
            return (0, 1)
        return (width - _DEFAULT_DOF, width - 1)
    resolved = tuple(int(i) for i in indices)
    require(
        len(resolved) == _DEFAULT_DOF,
        f"{name} must select exactly {_DEFAULT_DOF} columns",
        value=resolved,
    )
    for idx in resolved:
        require(0 <= idx < width, f"{name} index {idx} out of range", value=width)
    return (resolved[0], resolved[1])


def _extract_control(trace: Trace, control_indices: Sequence[int] | None) -> np.ndarray:
    """Return time-aligned applied torques, falling back to zeros."""
    source = trace.u if trace.u is not None else trace.torques
    if source is None:
        return np.zeros((trace.num_steps, _DEFAULT_DOF), dtype=float)
    columns = _resolve_indices("control_indices", control_indices, source.shape[1])
    tau = np.asarray(source[:, columns], dtype=float)
    require(
        tau.shape == (trace.num_steps, _DEFAULT_DOF),
        f"control selection must produce ({trace.num_steps}, {_DEFAULT_DOF}); "
        f"got {tau.shape}",
        value=tau.shape,
    )
    require(check_finite(tau), "control history must be finite", value=tau)
    return tau.copy()


def _inverse_mass(provider: DynamicsProvider, q: np.ndarray) -> np.ndarray:
    """Return ``M(q)^-1`` with shape and finiteness checks."""
    mass = np.asarray(provider.mass_matrix(q), dtype=float)
    require(
        mass.shape == (_DEFAULT_DOF, _DEFAULT_DOF),
        f"mass_matrix(q) must be ({_DEFAULT_DOF}, {_DEFAULT_DOF}); got {mass.shape}",
        value=mass.shape,
    )
    inverse = np.linalg.solve(mass, np.eye(_DEFAULT_DOF))
    require(check_finite(inverse), "inverse mass matrix must be finite", value=inverse)
    return inverse


def _require_pair(q_name: str, v_name: str, q: np.ndarray, v: np.ndarray) -> None:
    """Validate extracted kinematics pair."""
    require(
        q.shape == v.shape == (q.shape[0], _DEFAULT_DOF),
        f"{q_name}/{v_name} must both have shape (T, {_DEFAULT_DOF}); "
        f"got {q.shape} and {v.shape}",
        value=(q.shape, v.shape),
    )
    require(
        check_finite(q) and check_finite(v),
        f"{q_name}/{v_name} must be finite",
        value=(q, v),
    )


def _validate_path(path: str | os.PathLike[str]) -> None:
    """Validate persistence paths."""
    if not isinstance(path, (str, os.PathLike)):
        raise TypeError(f"path must be str or os.PathLike, got {type(path).__name__}")
    if not str(os.fspath(path)).strip():
        raise ValueError("path must be non-empty")
