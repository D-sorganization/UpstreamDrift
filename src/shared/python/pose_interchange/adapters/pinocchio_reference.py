"""Pinocchio reference adapter for canonical-v2 engine boundaries.

Canonical-v2 stores a floating base as ``[xyz, quat_wxyz]`` and six base
velocity/acceleration entries as ``[linear; angular]`` with linear velocity in
the WORLD frame and angular velocity in the BODY (local) frame
(``docs/conventions/canonical-v2.md`` §2). Pinocchio's free-flyer joint uses
the same ``[linear; angular]`` ordering but expresses **both** blocks in the
LOCAL frame, so the base linear velocity is rotated, ``v_local = R^T v_world``
(issue #12605), and the base linear acceleration picks up the ``-[w]x R^T v``
drift term. Those maps are delegated to the NSX-50 dual dynamic maps
(:mod:`pose_interchange.dynamic_maps`, the Pinocchio row of the spec §9 table)
rather than re-derived here. The quaternion is reordered (canonical
``w``-first vs Pinocchio ``w``-last). Generalized forces (``tau``) and the
gradient blocks stay in Pinocchio native coordinates; the canonical dual is
``tau_c = T^T tau_n``. The module runs without importing the Pinocchio wheel.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from src.shared.python.engine_core.capabilities import (
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.engine_core.finite_difference import (
    require_enough_frames_for_finite_diff as _require_enough_frames_for_finite_diff,
)
from src.shared.python.pose_interchange.adapters.pinocchio import PinocchioAdapter
from src.shared.python.pose_interchange.canonical_layout import (
    ANGULAR,
    LINEAR,
    FREE_FLYER_Q as _FREE_FLYER_Q,
    FREE_FLYER_V as _FREE_FLYER_V,
)
from src.shared.python.pose_interchange.dynamic_maps import (
    canonical_acceleration,
    canonical_velocity,
    native_acceleration,
    native_velocity,
    velocity_map_matrices,
)
from src.shared.python.pose_interchange.protocol import JointSlot

_DEFAULT_EPS = 1.0e-6
# Free-flyer block only: joint speeds pass through, so no joint slots are mapped.
_NO_JOINTS: Mapping[str, JointSlot] = {}


class PinocchioReferenceCapability(str, Enum):
    """CC-9 capability names exposed by the Pinocchio reference adapter."""

    INVERSE_DYN = "INVERSE_DYN"
    FORWARD_DYN = "FORWARD_DYN"
    GRADIENTS = "GRADIENTS"


@dataclass(frozen=True)
class CanonicalV2State:
    """Canonical-v2 q/v/a state.

    ``q`` begins with ``[x, y, z, qw, qx, qy, qz]``. ``v`` and ``a`` begin with
    ``[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]`` with the
    linear block in the WORLD frame and the angular block in the BODY (local)
    frame, per ``docs/conventions/canonical-v2.md`` §2.
    """

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        _require_vector(self.q, "q", min_size=_FREE_FLYER_Q)
        _require_vector(self.v, "v", min_size=_FREE_FLYER_V)
        _require_vector(self.a, "a", expected_size=self.v.shape[0])
        if self.q.shape[0] != self.v.shape[0] + 1:
            raise ValueError(
                "canonical-v2 free-flyer q must have one more entry than v "
                f"(got q={self.q.shape[0]}, v={self.v.shape[0]})"
            )


@dataclass(frozen=True)
class PinocchioNativeState:
    """Pinocchio-native q/v/a state.

    ``q`` begins with ``[x, y, z, qx, qy, qz, qw]`` (quaternion ``w``-last).
    ``v`` and ``a`` begin with
    ``[linear_x, linear_y, linear_z, angular_x, angular_y, angular_z]`` with
    BOTH blocks in the LOCAL frame (canonical-v2's linear block is WORLD).
    """

    q: npt.NDArray[np.float64]
    v: npt.NDArray[np.float64]
    a: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        _require_vector(self.q, "q", min_size=_FREE_FLYER_Q)
        _require_vector(self.v, "v", min_size=_FREE_FLYER_V)
        _require_vector(self.a, "a", expected_size=self.v.shape[0])
        if self.q.shape[0] != self.v.shape[0] + 1:
            raise ValueError(
                "Pinocchio free-flyer q must have one more entry than v "
                f"(got q={self.q.shape[0]}, v={self.v.shape[0]})"
            )


@dataclass(frozen=True)
class InverseDynamicsBatch:
    """Batched inverse-dynamics output in canonical-v2 ordering."""

    qdot: npt.NDArray[np.float64]
    qddot: npt.NDArray[np.float64]
    tau: npt.NDArray[np.float64]
    backend: str


@dataclass(frozen=True)
class InverseDynamicsGradients:
    """Finite or analytic derivative blocks for ``tau = rnea(q, v, a)``."""

    dtau_dq: npt.NDArray[np.float64]
    dtau_dv: npt.NDArray[np.float64]
    dtau_da: npt.NDArray[np.float64]
    dtau_dinertial: npt.NDArray[np.float64]
    backend: str


@runtime_checkable
class PinocchioReferenceBackend(Protocol):
    """Minimal Pinocchio-like backend required by the CC-9 adapter."""

    def fk(
        self, q: npt.NDArray[np.float64], frame_name: str
    ) -> npt.NDArray[np.float64]:
        """Return a 4x4 world transform for ``frame_name``."""

    def jacobian(
        self, q: npt.NDArray[np.float64], frame_name: str
    ) -> npt.NDArray[np.float64]:
        """Return Pinocchio spatial Jacobian rows as ``[linear; angular]``."""

    def rnea(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        a: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return inverse-dynamics generalized forces."""

    def aba(
        self,
        q: npt.NDArray[np.float64],
        v: npt.NDArray[np.float64],
        tau: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Return forward-dynamics generalized accelerations."""


class PinocchioReferenceAdapter:
    """CC-9 canonical-v2 boundary for Pinocchio as the reference engine."""

    engine_name = "pinocchio"

    def __init__(self, backend: PinocchioReferenceBackend) -> None:
        self._backend = backend

    def capabilities(self) -> frozenset[PinocchioReferenceCapability]:
        """Return the CC-9 capability taxonomy implemented by this adapter."""

        return frozenset(
            {
                PinocchioReferenceCapability.INVERSE_DYN,
                PinocchioReferenceCapability.FORWARD_DYN,
                PinocchioReferenceCapability.GRADIENTS,
            }
        )

    def engine_capabilities(self) -> EngineCapabilities:
        """Return the legacy capability report with CC-9 fields marked full."""

        return EngineCapabilities(
            engine_name="Pinocchio",
            jacobian=CapabilityLevel.FULL,
            inverse_dynamics=CapabilityLevel.FULL,
            forward_sim=CapabilityLevel.FULL,
            parameter_gradients=CapabilityLevel.FULL,
            state_control_gradients=CapabilityLevel.FULL,
            extra={
                "cc9_reference_capabilities": sorted(
                    capability.value for capability in self.capabilities()
                ),
                "quat_order": "xyzw",
                "canonical_quat_order": "wxyz",
            },
        )

    def from_canonical_v2(self, state: CanonicalV2State) -> PinocchioNativeState:
        """Map canonical-v2 q/v/a into Pinocchio native ordering and frames."""

        q_native = _canonical_q_to_pinocchio(state.q)
        return PinocchioNativeState(
            q=q_native,
            v=_canonical_velocity_to_pinocchio(q_native, state.v),
            a=_canonical_acceleration_to_pinocchio(q_native, state.a, state.v),
        )

    def to_canonical_v2(self, state: PinocchioNativeState) -> CanonicalV2State:
        """Map Pinocchio native q/v/a into canonical-v2 ordering and frames."""

        v_canonical = _pinocchio_velocity_to_canonical(state.q, state.v)
        return CanonicalV2State(
            q=_pinocchio_q_to_canonical(state.q),
            v=v_canonical,
            a=_pinocchio_acceleration_to_canonical(state.q, state.a, v_canonical),
        )

    def fk(self, state: CanonicalV2State, frame_name: str) -> npt.NDArray[np.float64]:
        """Return frame FK as a 4x4 transform using canonical-v2 input."""

        native = self.from_canonical_v2(state)
        transform = np.asarray(self._backend.fk(native.q, frame_name), dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"fk must return shape (4, 4), got {transform.shape}")
        return transform.copy()

    def jacobian(
        self, state: CanonicalV2State, frame_name: str
    ) -> Mapping[str, npt.NDArray[np.float64]]:
        """Return Jacobian blocks in canonical-v2 ``[linear; angular]`` order.

        Pinocchio emits spatial Jacobian rows as ``[linear; angular]`` (the
        canonical-v2 row order) with columns for the native velocity. The base
        columns are composed with the NSX-50 map, ``J_c = J_n T``, so
        ``spatial @ v`` agrees with the canonical ``v`` (issue #12605).
        """

        native = self.from_canonical_v2(state)
        pin_jacobian = np.asarray(
            self._backend.jacobian(native.q, frame_name), dtype=np.float64
        )
        if (
            pin_jacobian.ndim != 2
            or pin_jacobian.shape[0] != _FREE_FLYER_V
            or pin_jacobian.shape[1] < _FREE_FLYER_V
        ):
            raise ValueError(
                "jacobian must be a 6-row matrix in Pinocchio [linear; angular] "
                f"order with free-flyer columns, got {pin_jacobian.shape}"
            )
        t_mat, _ = _base_maps(native.q)
        canonical_jacobian = pin_jacobian.copy()
        canonical_jacobian[:, :_FREE_FLYER_V] = pin_jacobian[:, :_FREE_FLYER_V] @ t_mat
        linear = canonical_jacobian[LINEAR, :].copy()
        angular = canonical_jacobian[ANGULAR, :].copy()
        return {
            "linear": linear,
            "angular": angular,
            "spatial": np.vstack([linear, angular]),
        }

    def inverse_dynamics(self, state: CanonicalV2State) -> npt.NDArray[np.float64]:
        """Compute RNEA generalized forces (Pinocchio native coordinates)."""

        native = self.from_canonical_v2(state)
        return _as_tau(self._backend.rnea(native.q, native.v, native.a), native.v)

    def forward_dynamics(
        self, state: CanonicalV2State, tau: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Compute ABA acceleration and return the canonical-v2 acceleration.

        ``tau`` is in Pinocchio native generalized-force coordinates (the output
        of :meth:`inverse_dynamics`).
        """

        native = self.from_canonical_v2(state)
        tau_array = _require_vector(
            np.asarray(tau, dtype=np.float64),
            "tau",
            expected_size=native.v.shape[0],
        )
        native_accel = np.asarray(
            self._backend.aba(native.q, native.v, tau_array), dtype=np.float64
        )
        return _pinocchio_acceleration_to_canonical(
            native.q,
            _require_vector(native_accel, "aba", expected_size=native.v.shape[0]),
            state.v,
        )

    def inverse_dynamics_trajectory(
        self,
        q: npt.ArrayLike,
        times: npt.ArrayLike,
        *,
        qdot: npt.ArrayLike | None = None,
        qddot: npt.ArrayLike | None = None,
    ) -> InverseDynamicsBatch:
        """Run inverse dynamics over a canonical-v2 q trajectory.

        When the optional ``upstream_pinocchio_id`` wheel is importable this
        routes finite differencing and the frame loop through the Rust driver.
        Otherwise it uses the NumPy fallback with the same callback boundary.

        Canonical ``qdot``/``qddot`` overrides are mapped per frame into the
        Pinocchio LOCAL base frame (issue #12605) and returned unchanged; a
        ``qddot`` override needs its ``qdot`` because the base acceleration map
        depends on velocity. Without overrides the returned derivatives are the
        raw native finite differences.
        """

        q_canonical = _require_matrix(np.asarray(q, dtype=np.float64), "q")
        time_array = _require_vector(np.asarray(times, dtype=np.float64), "times")
        if q_canonical.shape[0] != time_array.shape[0]:
            raise ValueError("q rows must match times length")
        # Precondition (issue #7146): refuse to silently zero-fill derivatives
        # for trajectories too short to differentiate. Callers wanting statics
        # must pass explicit qdot/qddot overrides.
        _require_enough_frames_for_finite_diff(
            n_frames=q_canonical.shape[0],
            need_qdot=qdot is None,
            need_qddot=qddot is None,
        )
        q_native = _map_rows(q_canonical, _canonical_q_to_pinocchio)
        qdot_canonical = _optional_motion_matrix(qdot, "qdot", q_native.shape[0])
        qddot_canonical = _optional_motion_matrix(qddot, "qddot", q_native.shape[0])
        qdot_native: npt.NDArray[np.float64] | None = None
        qddot_native: npt.NDArray[np.float64] | None = None
        if qdot_canonical is not None:
            qdot_native = np.vstack(
                [
                    _canonical_velocity_to_pinocchio(q_row, v_row)
                    for q_row, v_row in zip(q_native, qdot_canonical, strict=True)
                ]
            )
        if qddot_canonical is not None:
            if qdot_canonical is None:
                raise ValueError(
                    "a qddot override requires the matching canonical qdot "
                    "override: the Pinocchio base acceleration map depends on v"
                )
            qddot_native = np.vstack(
                [
                    _canonical_acceleration_to_pinocchio(q_row, a_row, v_row)
                    for q_row, a_row, v_row in zip(
                        q_native, qddot_canonical, qdot_canonical, strict=True
                    )
                ]
            )

        rust = _load_rust_driver()
        if rust is not None:
            qdot_out, qddot_out, tau = rust.inverse_dynamics(
                q_native,
                time_array,
                self._backend.rnea,
                qdot_native,
                qddot_native,
            )
            return InverseDynamicsBatch(
                qdot=_derivative_output(qdot_canonical, qdot_out),
                qddot=_derivative_output(qddot_canonical, qddot_out),
                tau=np.asarray(tau, dtype=np.float64),
                backend="rust",
            )

        qdot_fallback, qddot_fallback = _finite_difference(
            q_native, time_array, qdot_native, qddot_native
        )
        tau = np.empty_like(qdot_fallback)
        for index in range(q_native.shape[0]):
            tau[index] = _as_tau(
                self._backend.rnea(
                    q_native[index], qdot_fallback[index], qddot_fallback[index]
                ),
                qdot_fallback[index],
            )
        return InverseDynamicsBatch(
            qdot=_derivative_output(qdot_canonical, qdot_fallback),
            qddot=_derivative_output(qddot_canonical, qddot_fallback),
            tau=tau,
            backend="numpy",
        )

    def inverse_dynamics_gradients(
        self,
        state: CanonicalV2State,
        *,
        inertial_parameters: npt.ArrayLike | None = None,
        set_inertial_parameters: (
            Callable[[npt.NDArray[np.float64]], None] | None
        ) = None,
        epsilon: float = _DEFAULT_EPS,
    ) -> InverseDynamicsGradients:
        """Return analytic gradients when supplied by backend, else finite diff.

        The blocks differentiate the native ``rnea`` with respect to the
        Pinocchio native ``q``/``v``/``a``, not the canonical-v2 coordinates.
        """

        native = self.from_canonical_v2(state)
        analytic = getattr(self._backend, "rnea_gradients", None)
        if callable(analytic):
            result = analytic(native.q, native.v, native.a)
            return InverseDynamicsGradients(
                dtau_dq=np.asarray(result["dtau_dq"], dtype=np.float64),
                dtau_dv=np.asarray(result["dtau_dv"], dtype=np.float64),
                dtau_da=np.asarray(result["dtau_da"], dtype=np.float64),
                dtau_dinertial=np.asarray(
                    result.get("dtau_dinertial", np.zeros((native.v.shape[0], 0))),
                    dtype=np.float64,
                ),
                backend="analytic",
            )

        tau_size = native.v.shape[0]
        return InverseDynamicsGradients(
            dtau_dq=_central_difference(
                native.q,
                lambda q_value: _as_tau(
                    self._backend.rnea(q_value, native.v, native.a), native.v
                ),
                tau_size=tau_size,
                epsilon=epsilon,
            ),
            dtau_dv=_central_difference(
                native.v,
                lambda v_value: _as_tau(
                    self._backend.rnea(native.q, v_value, native.a), native.v
                ),
                tau_size=tau_size,
                epsilon=epsilon,
            ),
            dtau_da=_central_difference(
                native.a,
                lambda a_value: _as_tau(
                    self._backend.rnea(native.q, native.v, a_value), native.v
                ),
                tau_size=tau_size,
                epsilon=epsilon,
            ),
            dtau_dinertial=_inertial_parameter_gradient(
                inertial_parameters,
                set_inertial_parameters,
                lambda: _as_tau(
                    self._backend.rnea(native.q, native.v, native.a), native.v
                ),
                tau_size=tau_size,
                epsilon=epsilon,
            ),
            backend="numpy",
        )


def _canonical_q_to_pinocchio(
    q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    q_array = _require_vector(q, "q", min_size=_FREE_FLYER_Q)
    result = q_array.copy()
    result[3:7] = q_array[[4, 5, 6, 3]]
    return result


def _pinocchio_q_to_canonical(
    q: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    q_array = _require_vector(q, "q", min_size=_FREE_FLYER_Q)
    result = q_array.copy()
    result[3:7] = q_array[[6, 3, 4, 5]]
    return result


def _base_maps(
    q_native: npt.NDArray[np.float64],
    v_canonical: npt.NDArray[np.float64] | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Return the 6x6 free-flyer ``(T, Tdot)`` with ``v_native = T v_canonical``.

    Delegates to the NSX-50 :func:`velocity_map_matrices` with the Pinocchio
    adapter's dynamic layout (linear and angular both LOCAL) so the frame
    convention lives in one place. Precondition: unit base quaternion.
    """

    base_v = None if v_canonical is None else v_canonical[:_FREE_FLYER_V]
    return velocity_map_matrices(
        PinocchioAdapter.dynamic_layout, q_native, _NO_JOINTS, base_v
    )


def _canonical_velocity_to_pinocchio(
    q_native: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Canonical ``v`` (world linear) -> Pinocchio ``v`` (``R^T`` linear)."""

    result = _require_vector(v, "v", min_size=_FREE_FLYER_V).copy()
    t_mat, _ = _base_maps(q_native)
    result[:_FREE_FLYER_V] = native_velocity(t_mat, result[:_FREE_FLYER_V])
    return result


def _pinocchio_velocity_to_canonical(
    q_native: npt.NDArray[np.float64], v: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Inverse of :func:`_canonical_velocity_to_pinocchio`."""

    result = _require_vector(v, "v", min_size=_FREE_FLYER_V).copy()
    t_mat, _ = _base_maps(q_native)
    result[:_FREE_FLYER_V] = canonical_velocity(t_mat, result[:_FREE_FLYER_V])
    return result


def _canonical_acceleration_to_pinocchio(
    q_native: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    v_canonical: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Canonical ``a`` -> Pinocchio ``a`` (``T a + Tdot v`` on the base block)."""

    result = _require_vector(a, "a", min_size=_FREE_FLYER_V).copy()
    t_mat, t_dot = _base_maps(q_native, v_canonical)
    result[:_FREE_FLYER_V] = native_acceleration(
        t_mat, t_dot, result[:_FREE_FLYER_V], v_canonical[:_FREE_FLYER_V]
    )
    return result


def _pinocchio_acceleration_to_canonical(
    q_native: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
    v_canonical: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Inverse of :func:`_canonical_acceleration_to_pinocchio`."""

    result = _require_vector(a, "a", min_size=_FREE_FLYER_V).copy()
    t_mat, t_dot = _base_maps(q_native, v_canonical)
    result[:_FREE_FLYER_V] = canonical_acceleration(
        t_mat, t_dot, result[:_FREE_FLYER_V], v_canonical[:_FREE_FLYER_V]
    )
    return result


def _derivative_output(
    canonical_override: npt.NDArray[np.float64] | None,
    native_estimate: npt.ArrayLike,
) -> npt.NDArray[np.float64]:
    """Return the canonical override if supplied, else the native estimate."""

    if canonical_override is not None:
        return canonical_override.copy()
    return np.asarray(native_estimate, dtype=np.float64)


def _finite_difference(
    q: npt.NDArray[np.float64],
    times: npt.NDArray[np.float64],
    qdot_override: npt.NDArray[np.float64] | None,
    qddot_override: npt.NDArray[np.float64] | None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    qdot = (
        qdot_override.copy()
        if qdot_override is not None
        else _finite_diff_qdot(q, times)
    )
    qddot = (
        qddot_override.copy()
        if qddot_override is not None
        else _finite_diff_qddot(q, times)
    )
    return qdot, qddot


def _finite_diff_qdot(
    q: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    qdot = np.zeros((q.shape[0], q.shape[1] - 1), dtype=np.float64)
    q_like = q[:, : qdot.shape[1]]
    for index in range(1, q.shape[0] - 1):
        dt = times[index + 1] - times[index - 1]
        if dt > 0.0:
            qdot[index] = (q_like[index + 1] - q_like[index - 1]) / dt
    if q.shape[0] >= 2:
        qdot[0] = (q_like[1] - q_like[0]) / max(times[1] - times[0], 1.0e-9)
        qdot[-1] = (q_like[-1] - q_like[-2]) / max(times[-1] - times[-2], 1.0e-9)
    return qdot


def _finite_diff_qddot(
    q: npt.NDArray[np.float64], times: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    qddot = np.zeros((q.shape[0], q.shape[1] - 1), dtype=np.float64)
    q_like = q[:, : qddot.shape[1]]
    for index in range(1, q.shape[0] - 1):
        dt_b = times[index] - times[index - 1]
        dt_f = times[index + 1] - times[index]
        if dt_b > 0.0 and dt_f > 0.0:
            qddot[index] = (
                2.0
                * (
                    q_like[index + 1] * dt_b
                    - q_like[index] * (dt_b + dt_f)
                    + q_like[index - 1] * dt_f
                )
                / (dt_b * dt_f * (dt_b + dt_f))
            )
    if q.shape[0] >= 3:
        qddot[0] = qddot[1]
        qddot[-1] = qddot[-2]
    return qddot


def _central_difference(
    x: npt.NDArray[np.float64],
    evaluate: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
    *,
    tau_size: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive")
    gradient = np.empty((tau_size, x.shape[0]), dtype=np.float64)
    for index in range(x.shape[0]):
        step = np.zeros_like(x)
        step[index] = epsilon
        gradient[:, index] = (evaluate(x + step) - evaluate(x - step)) / (2.0 * epsilon)
    return gradient


def _inertial_parameter_gradient(
    inertial_parameters: npt.ArrayLike | None,
    set_inertial_parameters: Callable[[npt.NDArray[np.float64]], None] | None,
    evaluate: Callable[[], npt.NDArray[np.float64]],
    *,
    tau_size: int,
    epsilon: float,
) -> npt.NDArray[np.float64]:
    if inertial_parameters is None:
        return np.zeros((tau_size, 0), dtype=np.float64)
    if set_inertial_parameters is None:
        raise ValueError(
            "set_inertial_parameters is required when inertial_parameters is provided"
        )
    params = _require_vector(
        np.asarray(inertial_parameters, dtype=np.float64), "params"
    )
    gradient = np.empty((tau_size, params.shape[0]), dtype=np.float64)
    for index in range(params.shape[0]):
        step = np.zeros_like(params)
        step[index] = epsilon
        set_inertial_parameters(params + step)
        plus = evaluate()
        set_inertial_parameters(params - step)
        minus = evaluate()
        gradient[:, index] = (plus - minus) / (2.0 * epsilon)
    set_inertial_parameters(params)
    return gradient


def _load_rust_driver() -> Any | None:
    try:
        import upstream_pinocchio_id  # type: ignore[import-not-found]
    except ImportError:
        return None
    return upstream_pinocchio_id


def _optional_motion_matrix(
    value: npt.ArrayLike | None, name: str, n_rows: int
) -> npt.NDArray[np.float64] | None:
    if value is None:
        return None
    matrix = _require_matrix(np.asarray(value, dtype=np.float64), name)
    if matrix.shape[0] != n_rows:
        raise ValueError(f"{name} rows must match q rows ({n_rows})")
    return matrix


def _map_rows(
    matrix: npt.NDArray[np.float64],
    transform: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]],
) -> npt.NDArray[np.float64]:
    return np.vstack([transform(row) for row in matrix])


def _as_tau(
    value: npt.ArrayLike, reference: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    return _require_vector(
        np.asarray(value, dtype=np.float64),
        "tau",
        expected_size=reference.shape[0],
    ).copy()


def _require_vector(
    value: npt.NDArray[np.float64],
    name: str,
    *,
    expected_size: int | None = None,
    min_size: int | None = None,
) -> npt.NDArray[np.float64]:
    if value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {value.shape}")
    if expected_size is not None and value.shape[0] != expected_size:
        raise ValueError(
            f"{name} must have length {expected_size}, got {value.shape[0]}"
        )
    if min_size is not None and value.shape[0] < min_size:
        raise ValueError(f"{name} must have at least {min_size} entries")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _require_matrix(
    value: npt.NDArray[np.float64], name: str
) -> npt.NDArray[np.float64]:
    if value.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must contain only finite values")
    return value


__all__ = [
    "CanonicalV2State",
    "InverseDynamicsBatch",
    "InverseDynamicsGradients",
    "PinocchioNativeState",
    "PinocchioReferenceAdapter",
    "PinocchioReferenceBackend",
    "PinocchioReferenceCapability",
]
