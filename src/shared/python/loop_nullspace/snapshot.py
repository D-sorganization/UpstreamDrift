"""Build a :class:`ClosedLoopSnapshot` from an engine force adapter (NSX-13).

One engine seam: the snapshot is assembled from the native
``BaseEngineForceAdapter`` that the torque allocator already uses, not from a
parallel protocol.  The adapter must declare
``Capability.CLOSED_LOOP_SNAPSHOT`` at ``CapabilityLevel.FULL`` and provide the
optional closed-loop methods (see :class:`ClosedLoopSnapshotAdapter` in
``multi_engine_torque_allocator``): ``compute_closure_drift``,
``native_closure_frame``, ``closure_wrench_frame``, ``club_inertia_split``,
``closure_topology`` and the ``native_closure_convention`` attribute.  The
native closure Jacobian and drift are converted to the canonical grip-frame
convention with the NSX-02 converter; row order, sign and frame are never
re-derived here.  No engine is imported.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace.closure_convention import to_canonical_closure
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    GripFrame,
    InputProvenance,
    ModelProvenance,
)

if TYPE_CHECKING:
    from src.shared.python.loop_nullspace.evidence import Array

__all__ = [
    "canonical_closure_jacobian",
    "parity_effort",
    "has_closed_loop_protocol",
    "snapshot_from_adapter",
]

_CLOSED_LOOP_METHODS = (
    "compute_closure_drift",
    "native_closure_frame",
    "closure_wrench_frame",
    "club_inertia_split",
    "closure_topology",
)


def has_closed_loop_protocol(adapter: Any) -> bool:
    """Whether ``adapter`` offers the native-convention and frame methods.

    True when it exposes ``native_closure_convention`` and the two frame
    methods needed to convert its closure rows to the canonical convention.
    """
    return all(
        hasattr(adapter, name)
        for name in (
            "native_closure_convention",
            "native_closure_frame",
            "closure_wrench_frame",
        )
    )


def _canonicalise(
    adapter: Any, q: ArrayLike, jacobian: ArrayLike, drift: ArrayLike
) -> tuple[Array, Array, GripFrame]:
    native_rot, native_point = adapter.native_closure_frame(q)
    grip_rot, grip_point = adapter.closure_wrench_frame(q)
    j_c, jdot_v = to_canonical_closure(
        jacobian,
        drift,
        native=adapter.native_closure_convention,
        native_frame_rotation=native_rot,
        grip_frame_rotation=grip_rot,
        native_reference_point=native_point,
        grip_reference_point=grip_point,
    )
    return j_c, jdot_v, GripFrame(rotation=grip_rot, origin=grip_point)


def canonical_closure_jacobian(adapter: Any, q: ArrayLike) -> Array:
    """Canonical closure Jacobian ``J_c`` of ``adapter`` at ``q`` (NSX-13).

    Used by the torque allocator so its result does not depend on the engine's
    native row order, frame, reference point or sign (R8, #12551).  An adapter
    that does not implement the closed-loop frame protocol is a legacy
    passthrough: its ``compute_grip_jacobian`` is returned unchanged.

    Postcondition: ``(6, nv)`` finite array.
    """
    native = adapter.compute_grip_jacobian(q)
    if not has_closed_loop_protocol(adapter):
        return np.asarray(native, dtype=np.float64)
    j_c, _, _ = _canonicalise(adapter, q, native, np.zeros(6))
    return np.asarray(j_c, dtype=np.float64)


def parity_effort(
    adapter: Any, tau_applied: ArrayLike, j_grip: ArrayLike, lambda_grip: ArrayLike
) -> Array:
    """Effort to pass to ``verify_acceleration_parity`` for ``adapter``.

    A closed-loop adapter (``has_closed_loop_protocol``) solves for the closure
    reaction itself, so it receives the applied efforts only.  A legacy adapter
    audits the raw equation and needs ``J^T lambda`` folded in.
    """
    tau = np.asarray(tau_applied, dtype=np.float64)
    if has_closed_loop_protocol(adapter):
        return tau
    return tau + np.asarray(j_grip).T @ np.asarray(lambda_grip)


def _require_full(adapter: Any) -> None:
    ask = getattr(adapter, "capability", None)
    level = ask(Capability.CLOSED_LOOP_SNAPSHOT) if callable(ask) else None
    if level is not CapabilityLevel.FULL:
        shown = "undeclared" if level is None else level.name
        raise CapabilityUnavailable(
            f"{Capability.CLOSED_LOOP_SNAPSHOT.value} requires CapabilityLevel.FULL; "
            f"{type(adapter).__name__} declares {shown}"
        )
    missing = [
        n for n in _CLOSED_LOOP_METHODS if not callable(getattr(adapter, n, None))
    ]
    if missing or not has_closed_loop_protocol(adapter):
        raise CapabilityUnavailable(
            f"{type(adapter).__name__} declares FULL closed_loop_snapshot but lacks "
            f"{missing or 'native_closure_convention'}"
        )


def _provenance_version(adapter: Any) -> str:
    """Snapshot version tag, plus the drift method when the adapter declares one.

    A closure drift obtained numerically (NSX-51: MuJoCo) is declared by the
    adapter's ``closure_drift_method`` and recorded as ``;drift=<method>``.  An
    adapter that exposes ``kkt_regularization`` (MyoSuite, NSX-52) has it
    recorded as ``;kkt_regularization=<value>``.
    """
    method = getattr(adapter, "closure_drift_method", None)
    regularization = getattr(adapter, "kkt_regularization", None)
    tag = "closed_loop_snapshot_v1"
    if method:
        tag += f";drift={method}"
    if regularization is not None:
        tag += f";kkt_regularization={regularization:g}"
    return tag


def snapshot_from_adapter(
    adapter: Any,
    q: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    *,
    input_provenance: InputProvenance,
) -> ClosedLoopSnapshot:
    """Assemble a validated :class:`ClosedLoopSnapshot` at ``(q, v, a)``.

    ``M a + h = S^T tau + J_c^T lambda`` holds with ``M, h`` from
    ``adapter.compute_mass_and_bias`` and ``J_c, Jdot_v`` converted from the
    adapter's native closure convention to the canonical one.  ``J_g`` (ground
    contact) is left ``None``; foot contacts are not part of this seam.

    Preconditions: ``adapter`` declares ``CLOSED_LOOP_SNAPSHOT`` at ``FULL``
    (else :class:`CapabilityUnavailable`); ``q, v, a`` are finite vectors of
    length ``adapter.nv`` (else ``ValueError``).  Postcondition: the returned
    snapshot passed every NSX-10 contract check.  The state is *not* required
    to satisfy the closure; admission (NSX-14) decides that.
    """
    _require_full(adapter)
    nv = adapter.nv
    vectors = [np.asarray(x, dtype=np.float64) for x in (q, v, a)]
    for name, vec in zip(("q", "v", "a"), vectors, strict=True):
        if vec.shape != (nv,):
            raise ValueError(f"{name} must have shape ({nv},), got {vec.shape}")
        if not np.isfinite(vec).all():
            raise ValueError(f"{name} must be finite")
    q_arr, v_arr, a_arr = vectors
    mass, bias = adapter.compute_mass_and_bias(q_arr, v_arr)
    topology = adapter.closure_topology()
    j_c, jdot_v, grip = _canonicalise(
        adapter,
        q_arr,
        adapter.compute_grip_jacobian(q_arr),
        adapter.compute_closure_drift(q_arr, v_arr),
    )
    bare, welded = adapter.club_inertia_split()
    selection = np.eye(nv)[list(topology.actuated_indices)]
    return ClosedLoopSnapshot(
        q=q_arr,
        v=v_arr,
        a=a_arr,
        M=mass,
        h=bias,
        S=selection,
        J_c=j_c,
        Jdot_v=jdot_v,
        grip_frame=grip,
        bare_club=bare,
        welded_hand_inertias=tuple(welded),
        model_provenance=ModelProvenance(
            name=adapter.engine_type.value,
            version=_provenance_version(adapter),
            content_hash=adapter.model_hash,
        ),
        input_provenance=input_provenance,
        topology=topology,
    )
