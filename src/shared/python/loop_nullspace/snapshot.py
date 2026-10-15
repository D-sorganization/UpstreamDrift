"""Build canonical ClosedLoopSnapshot from a native BaseEngineForceAdapter (#12558, NSX-13).

Enforces the single-engine-seam contract: snapshots are constructed from an
adapter advertising ``Capability.CLOSED_LOOP_SNAPSHOT == CapabilityLevel.FULL``,
with native closure Jacobians and drifts canonicalized into the R8 grip frame
via :func:`src.shared.python.loop_nullspace.closure_convention.to_canonical_closure`.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.biomechanics.club_solids_partition import SolidSpatialInertia
from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
)
from src.shared.python.loop_nullspace.closure_convention import (
    CANONICAL_CLOSURE_CONVENTION,
    to_canonical_closure,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    FloatArray,
    FootContact,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    BaseEngineForceAdapter,
)

__all__ = [
    "CapabilityUnavailable",
    "CapabilityUnavailableError",
    "snapshot_from_adapter",
]


class CapabilityUnavailableError(RuntimeError):
    """Raised when an adapter does not support a required capability at FULL level."""


CapabilityUnavailable = CapabilityUnavailableError


def _query_snapshot_capability(adapter: Any) -> CapabilityLevel:
    """Return the adapter's support level for Capability.CLOSED_LOOP_SNAPSHOT."""
    if hasattr(adapter, "capability"):
        level = adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT)
        if isinstance(level, CapabilityLevel):
            return level
    if hasattr(adapter, "closed_loop_snapshot_capability"):
        level = adapter.closed_loop_snapshot_capability()
        if isinstance(level, CapabilityLevel):
            return level
    if hasattr(adapter, "level_for"):
        level = adapter.level_for(Capability.CLOSED_LOOP_SNAPSHOT)
        if isinstance(level, CapabilityLevel):
            return level
    return CapabilityLevel.NONE


def _to_spatial_inertia(
    item: SpatialInertia | SolidSpatialInertia,
    default_solids: tuple[str, ...] = (),
) -> SpatialInertia:
    """Coerce a SolidSpatialInertia or SpatialInertia into a contracts.SpatialInertia."""
    if isinstance(item, SpatialInertia):
        return item
    if isinstance(item, SolidSpatialInertia):
        leaf = item.name.rsplit("/", 1)[-1]
        solids = default_solids or ((leaf,) if leaf else ())
        return SpatialInertia(
            mass_kg=float(item.mass_kg),
            com_m=np.asarray(item.com_body_m, dtype=np.float64),
            inertia_com_kg_m2=np.asarray(item.inertia_com_body_kg_m2, dtype=np.float64),
            solid_names=solids,
        )
    raise TypeError(
        f"Expected SpatialInertia or SolidSpatialInertia, got {type(item).__name__}"
    )


def _selection_matrix(actuated_indices: tuple[int, ...], nv: int) -> FloatArray:
    """Construct (n_tau, nv) actuation selection matrix S."""
    s_mat = np.zeros((len(actuated_indices), nv), dtype=np.float64)
    for row_idx, col_idx in enumerate(actuated_indices):
        s_mat[row_idx, col_idx] = 1.0
    return s_mat


def _extract_contacts(
    adapter: Any, q_arr: FloatArray, nv: int
) -> tuple[FloatArray | None, tuple[FootContact, ...]]:
    """Extract optional ground contact Jacobian J_g and FootContact metadata."""
    if hasattr(adapter, "foot_contacts_snapshot"):
        return adapter.foot_contacts_snapshot(q_arr)
    n_spheres = int(getattr(adapter, "n_contact_spheres", 0))
    if n_spheres <= 0:
        return None, ()
    j_g = np.asarray(adapter.compute_contact_jacobian(q_arr), dtype=np.float64)
    if j_g.size == 0 or j_g.shape[1] != nv:
        return None, ()
    if hasattr(adapter, "foot_contacts_for_state"):
        contacts = tuple(adapter.foot_contacts_for_state(q_arr, j_g))
        return j_g, contacts
    return j_g, ()


def _build_topology(
    adapter: Any,
    nv: int,
    actuated: tuple[int, ...],
    j_c: FloatArray,
) -> LoopTopology:
    """Build or retrieve the LoopTopology for the adapter's kinematic model."""
    if hasattr(adapter, "topology") and isinstance(adapter.topology, LoopTopology):
        return adapter.topology
    act_set = set(actuated)
    unactuated = tuple(i for i in range(nv) if i not in act_set)
    if hasattr(adapter, "loop_support_indices"):
        support = tuple(int(i) for i in adapter.loop_support_indices)
    else:
        nonzero_cols = tuple(
            int(j) for j in range(nv) if float(np.linalg.norm(j_c[:, j])) > 1e-12
        )
        support = nonzero_cols or actuated
    kind = getattr(adapter, "closure_kind", ClosureKind.WELD6)
    return LoopTopology(
        closure_kind=ClosureKind(kind),
        closing_side="right",
        holding_side="left",
        actuated_indices=actuated,
        unactuated_indices=unactuated,
        loop_support_indices=support,
        coordinate_order=tuple(str(c) for c in adapter.coordinate_order),
    )


def _resolve_provenance(
    adapter: Any,
    input_provenance: InputProvenance | None,
    model_provenance: ModelProvenance | None,
) -> tuple[ModelProvenance, InputProvenance]:
    """Resolve ModelProvenance and InputProvenance for snapshot construction."""
    if model_provenance is not None:
        resolved_model = model_provenance
    elif hasattr(adapter, "model_provenance") and isinstance(
        adapter.model_provenance, ModelProvenance
    ):
        resolved_model = adapter.model_provenance
    else:
        resolved_model = ModelProvenance(
            spec_sha256=str(getattr(adapter, "model_hash", "") or "pinocchio-spec"),
            inertial_source="full_body_spec_solids",
            bare_club_bench_id="bare_club_partition",
            engine_name=str(
                getattr(getattr(adapter, "engine_type", None), "value", "pinocchio")
            ),
        )
    resolved_input = (
        input_provenance
        if input_provenance is not None
        else InputProvenance(
            kinematic_source="adapter_state",
            derivative_method="analytical",
            time_sync_status="SYNCHRONIZED",
        )
    )
    return resolved_model, resolved_input


def _canonical_closure_terms(
    adapter: BaseEngineForceAdapter,
    q_arr: FloatArray,
    v_arr: FloatArray,
) -> tuple[GripFrame, FloatArray, FloatArray]:
    """Compute GripFrame and canonical (J_c, Jdot_v) from native adapter methods."""
    j_native = np.asarray(adapter.compute_grip_jacobian(q_arr), dtype=np.float64)
    jdot_v_native = np.asarray(
        adapter.compute_closure_drift(q_arr, v_arr), dtype=np.float64
    )
    r_wg, p_g = adapter.closure_wrench_frame(q_arr)
    grip_frame = GripFrame(rotation=r_wg, origin=p_g)
    conv = getattr(
        adapter,
        "native_closure_convention",
        getattr(adapter, "NATIVE_CLOSURE_CONVENTION", CANONICAL_CLOSURE_CONVENTION),
    )
    p_ref = (
        adapter.native_closure_reference_point(q_arr)
        if hasattr(adapter, "native_closure_reference_point")
        else p_g
    )
    j_c, jdot_v = (
        to_canonical_closure(
            j_native,
            jdot_v_native,
            native=conv,
            grip_frame_rotation=r_wg,
            grip_origin=p_g,
            reference_point=p_ref,
        )
        if j_native.shape[0] == 6
        else (j_native, jdot_v_native)
    )
    return grip_frame, j_c, jdot_v


def snapshot_from_adapter(
    adapter: BaseEngineForceAdapter,
    q: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    *,
    input_provenance: InputProvenance | None = None,
    model_provenance: ModelProvenance | None = None,
    grf_observed: ArrayLike | None = None,
) -> ClosedLoopSnapshot:
    """Build a canonical :class:`ClosedLoopSnapshot` from a native force adapter."""
    level = _query_snapshot_capability(adapter)
    if level is not CapabilityLevel.FULL:
        engine_label = getattr(adapter, "engine_type", type(adapter).__name__)
        raise CapabilityUnavailableError(
            f"Adapter {engine_label!r} does not support Capability.CLOSED_LOOP_SNAPSHOT "
            f"at FULL level (reported {level.name})"
        )

    nv = int(adapter.nv)
    q_arr = np.asarray(q, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    a_arr = np.asarray(a, dtype=np.float64)
    if q_arr.ndim != 1 or q_arr.shape[0] < nv:
        raise ValueError(f"q must be 1-D with length >= {nv}, got {q_arr.shape}")
    if v_arr.shape != (nv,) or a_arr.shape != (nv,):
        raise ValueError(
            f"v and a must have shape ({nv},), got {v_arr.shape} and {a_arr.shape}"
        )

    mass, bias = adapter.compute_mass_and_bias(q_arr, v_arr)
    grip_frame, j_c, jdot_v = _canonical_closure_terms(adapter, q_arr, v_arr)

    bare_raw, hands_raw = adapter.club_inertia_split()
    bare_club = _to_spatial_inertia(
        bare_raw, default_solids=("Clubhead", "Rigid Shaft", "Grip")
    )
    welded_hands = tuple(_to_spatial_inertia(h) for h in hands_raw)
    actuated = tuple(int(i) for i in adapter.actuated_indices)
    topology = _build_topology(adapter, nv, actuated, j_c)
    j_g, foot_contacts = _extract_contacts(adapter, q_arr, nv)
    resolved_model, resolved_input = _resolve_provenance(
        adapter, input_provenance, model_provenance
    )

    grf_arr = (
        None if grf_observed is None else np.asarray(grf_observed, dtype=np.float64)
    )
    snapshot = ClosedLoopSnapshot(
        q=q_arr,
        v=v_arr,
        a=a_arr,
        M=mass,
        h=bias,
        S=_selection_matrix(actuated, nv),
        J_c=j_c,
        Jdot_v=jdot_v,
        grip_frame=grip_frame,
        bare_club=bare_club,
        welded_hand_inertias=welded_hands,
        model_provenance=resolved_model,
        input_provenance=resolved_input,
        topology=topology,
        J_g=j_g,
        foot_contacts=foot_contacts,
        grf_observed=grf_arr,
    )
    if hasattr(adapter, "closure_pose_residual"):
        try:
            pose_res = np.asarray(
                adapter.closure_pose_residual(q_arr), dtype=np.float64
            )
            pose_res.setflags(write=False)
            object.__setattr__(snapshot, "_closure_pose_residual", pose_res)
        except (ValueError, RuntimeError, TypeError, AttributeError):
            pass
    return snapshot
