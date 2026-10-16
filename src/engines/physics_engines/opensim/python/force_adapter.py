"""OpenSim closed-loop force adapter for the NSX snapshot seam (NSX-54, #12582).

A thin delegating layer over :class:`OpenSimFullBodyParityAdapter`: every
engine evaluation (``calcM``, inverse dynamics at zero acceleration, frame
Jacobians and their bias) already lives there; this class only maps it onto the
``BaseEngineForceAdapter`` / ``ClosedLoopSnapshotAdapter`` protocols so that
``loop_nullspace.snapshot.snapshot_from_adapter`` can build a snapshot.

State vectors are the spec's named scalar coordinates in ``coordinate_order``;
the velocity is ``dq/dt`` of those coordinates (the Simbody ``u`` slot of each
one-mobility coordinate), not a Simbody FreeJoint mobilizer speed, so no base
twist map from ``pose_interchange.dynamic_maps`` applies.  The native closure
rows follow ``CLOSURE_CONVENTION`` (WORLD frame, ``[linear; angular]``,
``V_b - V_a``); the canonical grip frame is the hand-side weld frame, which
coincides with the club-side one on a closed loop.  No muscles, no ``calcG``.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.engines.physics_engines.opensim.python.closed_loop_derivatives import (
    OpenSimClosedLoopDerivatives,
    opensim_closed_loop_derivatives,
)
from src.engines.physics_engines.opensim.python.full_body_parity import (
    CLOSURE_CONVENTION,
    OpenSimFullBodyParityAdapter,
)
from src.shared.python.biomechanics.club_inertia_partition import (
    RigidInertia,
    club_solids_partition,
)
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosureKind,
    LoopTopology,
    SpatialInertia,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import EngineType
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

Array = NDArray[np.float64]


class OpenSimNativeForceAdapter:
    """Native OpenSim/Simbody force adapter with ``CLOSED_LOOP_SNAPSHOT`` FULL.

    Args:
        spec_bytes: Full-body spec document (``full-body-v1``).

    Postconditions: ``coordinate_order`` is the spec order; ``actuated_indices``
    are the coordinates outside the six unactuated root coordinates.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        self._parity = OpenSimFullBodyParityAdapter(spec_bytes)
        self._spec = json.loads(spec_bytes)
        self._hash = hashlib.sha256(spec_bytes).hexdigest()
        order = self._parity.coordinate_order
        if not set(ROOT_COORDINATES).issubset(order):
            raise ValueError(
                "OpenSim force adapter requires the named root coordinates"
            )
        self._actuated = tuple(
            i for i, name in enumerate(order) if name not in ROOT_COORDINATES
        )

    # -- BaseEngineForceAdapter -------------------------------------------
    @property
    def engine_type(self) -> EngineType:
        return EngineType.OPENSIM

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self._parity.coordinate_order

    @property
    def nv(self) -> int:
        return self._parity.nv

    @property
    def actuated_indices(self) -> tuple[int, ...]:
        return self._actuated

    @property
    def contact_names(self) -> tuple[str, ...]:
        return self._parity.contact_names

    @property
    def n_contact_spheres(self) -> int:
        return len(self._parity.contact_names)

    @property
    def model_hash(self) -> str:
        """SHA-256 of the spec document the OpenSim model was built from."""
        return self._hash

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        return self._parity.mass_and_bias(q, v)

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        """Unconstrained ``M a + h`` (``h`` is Simbody ID at zero acceleration)."""
        acceleration = np.asarray(a, dtype=float)
        if acceleration.shape != (self.nv,) or not np.isfinite(acceleration).all():
            raise ValueError(f"a must be a finite vector of shape ({self.nv},)")
        mass, bias = self._parity.mass_and_bias(q, v)
        return mass @ acceleration + bias

    def compute_contact_jacobian(self, q: Array) -> Array:
        return self._parity.contact_jacobian(q)

    def compute_grip_jacobian(self, q: Array) -> Array:
        """Native weld Jacobian in ``CLOSURE_CONVENTION`` (rows ``V_b - V_a``)."""
        return self._parity.closure_jacobian(q)

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """``||weld forward dynamics(q, v, tau) - a_target||_inf`` (one semantics).

        ``tau_effective`` is the applied effort only; the closure reaction is
        solved for, exactly as in the Pinocchio adapter.
        """
        target = np.asarray(a_target, dtype=float)
        if target.shape != (self.nv,) or not np.isfinite(target).all():
            raise ValueError(f"a_target must be a finite vector of shape ({self.nv},)")
        actual = self._parity.weld_acceleration(q, v, tau_effective)
        if not np.isfinite(actual).all():
            raise ValueError("Native forward acceleration is nonfinite")
        return float(np.max(np.abs(actual - target)))

    # -- ClosedLoopSnapshotAdapter ----------------------------------------
    def capability(self, capability: Capability) -> CapabilityLevel:
        """``FULL`` snapshot, ``PARTIAL`` closed-loop derivatives, else ``NONE``.

        The derivatives are NSX-84 central differences
        (``derivative_method=finite_difference``), hence ``PARTIAL`` (NSX-88).
        """
        if capability is Capability.CLOSED_LOOP_SNAPSHOT:
            return CapabilityLevel.FULL
        if capability is Capability.CLOSED_LOOP_DERIVATIVES:
            return CapabilityLevel.PARTIAL
        return CapabilityLevel.NONE

    @property
    def muscle_count(self) -> int:
        """Muscle-tendon actuators of the source model (0 for the spec model)."""
        return self._parity.muscle_count

    def constrained_acceleration_derivatives(
        self,
        q: ArrayLike,
        v: ArrayLike,
        tau: ArrayLike,
        *,
        step: float = 1e-5,
        check_convergence: bool = True,
    ) -> OpenSimClosedLoopDerivatives:
        """Converged FD tangent derivatives of ``(a, lambda)`` (NSX-88, #12717).

        See :func:`opensim_closed_loop_derivatives`; the result flags
        coordinate-only derivatives as insufficient when muscles are present.
        """
        return opensim_closed_loop_derivatives(
            self, q, v, tau, step=step, check_convergence=check_convergence
        )

    @property
    def native_closure_convention(self) -> NativeClosureConvention:
        """Layout of :meth:`compute_grip_jacobian` rows (NSX-02 constant)."""
        return CLOSURE_CONVENTION

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Native ``Jdot v`` (6,) of :meth:`compute_grip_jacobian`."""
        return self._parity.closure_drift(q, v)

    def native_closure_frame(self, q: Array) -> tuple[Array, Array]:
        """WORLD convention: identity rotation, hand-side weld origin as point."""
        _, origin = self._parity.closure_hand_frame(q)
        return np.eye(3), origin

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """Canonical grip frame: the hand-side weld frame in the world."""
        return self._parity.closure_hand_frame(q)

    def club_inertia_split(self) -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]:
        """``(bare club, welded hand solids)`` in the club-side weld frame.

        The spec partitions are expressed in the club body frame; the weld
        frame sits at ``placement_b`` in that body, so the transform is its
        inverse.  Body-fixed (state independent).
        """
        to_frame = np.linalg.inv(self._parity.closure_club_placement)
        partition = club_solids_partition(self._spec)
        return (
            _spatial(partition.bare_club.expressed_in(to_frame)),
            tuple(
                _spatial(hand.expressed_in(to_frame))
                for hand in partition.welded_hands.values()
            ),
        )

    def closure_topology(self) -> LoopTopology:
        """Weld topology: hands close the loop onto the club."""
        order = self.coordinate_order
        actuated = set(self._actuated)
        support = set(self._parity.closure_support_names())
        return LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="hands",
            holding_side="club",
            actuated_indices=self._actuated,
            unactuated_indices=tuple(i for i in range(self.nv) if i not in actuated),
            loop_support_indices=tuple(
                i for i, name in enumerate(order) if name in support
            ),
            coordinate_order=order,
        )


def _spatial(inertia: RigidInertia) -> SpatialInertia:
    return SpatialInertia(
        mass=inertia.mass_kg,
        com=np.asarray(inertia.com_m),
        inertia=inertia.inertia_com_kg_m2,
    )
