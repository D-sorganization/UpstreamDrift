"""Native Drake force equations for the shared allocation pathway (NSX-53).

``DrakeForceAdapter`` is the Drake counterpart of ``PinocchioForceAdapter``:
every equation comes from a real ``MultibodyPlant`` (``CalcMassMatrix``,
``CalcBiasTerm``, ``CalcGravityGeneralizedForces``, ``CalcJacobianSpatialVelocity``
and ``CalcBiasSpatialAcceleration``), never from an analytical stand-in.  It
declares ``Capability.CLOSED_LOOP_SNAPSHOT`` at ``FULL`` so
``snapshot_from_adapter`` can build a canonical closed-loop snapshot from it.

Coordinates are in the spec ``coordinate_order`` (the order of
``FullBodyDrakeModel.names``); Drake's own plant ordering is hidden behind the
model.  Closure rows follow ``CLOSURE_CONVENTION`` of the Drake model module.
"""

from __future__ import annotations

import json
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.full_body_model import (
    CLOSURE_CONVENTION,
    FullBodyDrakeModel,
    solve_weld_acceleration,
)
from src.shared.python.biomechanics.club_inertia_partition import RigidInertia
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

Array: TypeAlias = NDArray[np.float64]


class DrakeForceAdapter:
    """Raw Drake equations in spec coordinate order.

    Contact Jacobians use world XYZ at the sphere centres; the grip Jacobian
    is the native weld Jacobian (``ang;lin``, LOCAL in frame ``a``, ``V_b -
    V_a``).  Every method refreshes its own context state.  Algebraic parity
    does not establish contact replay.  Requires a real ``pydrake`` install.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        self._spec = json.loads(spec_bytes)
        self._model = FullBodyDrakeModel(self._spec)
        self._names = tuple(self._model.coordinate_order)
        if not set(ROOT_COORDINATES).issubset(self._names):
            raise ValueError(
                "Force adapter requires the named unactuated root coordinates"
            )
        self._actuated = tuple(
            i for i, name in enumerate(self._names) if name not in ROOT_COORDINATES
        )
        self._contact_names = tuple(s["name"] for s in self._spec["contact"]["spheres"])

    @property
    def engine_type(self) -> EngineType:
        return EngineType.DRAKE

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self._names

    @property
    def contact_names(self) -> tuple[str, ...]:
        return self._contact_names

    @property
    def nv(self) -> int:
        return len(self._names)

    @property
    def actuated_indices(self) -> tuple[int, ...]:
        return self._actuated

    @property
    def n_contact_spheres(self) -> int:
        return len(self._contact_names)

    @property
    def model_hash(self) -> str:
        return self._model.model_sha256

    def _checked(self, values: Array, name: str) -> Array:
        result = np.asarray(values, dtype=float)
        if result.shape != (self.nv,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector with shape ({self.nv},)")
        return result

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        """``(M(q), h(q, v))`` with ``M a + h = tau`` (Drake plant calls)."""
        return self._model.mass_and_bias(
            self._checked(q, "configuration"), self._checked(v, "velocity")
        )

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        """Unconstrained ``M(q) a + h(q, v)``."""
        acceleration = self._checked(a, "acceleration")
        mass, bias = self.compute_mass_and_bias(q, v)
        return mass @ acceleration + bias

    def compute_contact_jacobian(self, q: Array) -> Array:
        """Sphere-centre translational Jacobians, world XYZ, ``(3 k, nv)``."""
        self._model.set_configuration(self._checked(q, "configuration"))
        return self._model.sphere_jacobians()

    def compute_grip_jacobian(self, q: Array) -> Array:
        """Native weld Jacobian ``(6, nv)`` in ``CLOSURE_CONVENTION`` layout."""
        configuration = self._checked(q, "configuration")
        jacobian, _ = self._model.closure_jacobian_and_drift(
            configuration, np.zeros(self.nv)
        )
        return jacobian

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """``||a* - a_target||_inf`` with ``a*`` the loop-constrained acceleration.

        ``tau_effective`` is the applied generalised force only (actuators plus
        any other external force); the closure reaction is solved for, not
        supplied, exactly as for Pinocchio (#12558).  Raises ``ValueError`` on
        nonfinite input or result.
        """
        effort = self._checked(tau_effective, "effective effort")
        target = self._checked(a_target, "target acceleration")
        mass, bias = self.compute_mass_and_bias(q, v)
        jacobian, drift = self._model.closure_jacobian_and_drift(
            self._checked(q, "configuration"), self._checked(v, "velocity")
        )
        try:
            actual = solve_weld_acceleration(mass, effort - bias, jacobian, drift)
        except FloatingPointError as exc:
            raise ValueError("Native forward acceleration is nonfinite") from exc
        return float(np.max(np.abs(actual - target)))

    # -- closed-loop snapshot capability (NSX-53, builds on NSX-13) ---------

    def capability(self, capability: Capability) -> CapabilityLevel:
        """``CLOSED_LOOP_SNAPSHOT`` is FULL; every other capability is NONE here."""
        if capability is Capability.CLOSED_LOOP_SNAPSHOT:
            return CapabilityLevel.FULL
        return CapabilityLevel.NONE

    @property
    def native_closure_convention(self) -> NativeClosureConvention:
        """Layout of :meth:`compute_grip_jacobian` rows (NSX-02 constant)."""
        return CLOSURE_CONVENTION

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Native ``Jdot v`` (6,) from ``CalcBiasSpatialAcceleration``."""
        _, drift = self._model.closure_jacobian_and_drift(
            self._checked(q, "configuration"), self._checked(v, "velocity")
        )
        return drift

    def native_closure_frame(self, q: Array) -> tuple[Array, Array]:
        """World rotation and origin of the frame the native rows use (LOCAL)."""
        return self._model.closure_frame_pose(self._checked(q, "configuration"))

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """Canonical grip frame ``(R world<-grip, origin)``.

        The grip frame is the hand-side weld frame ``a``, i.e. the native LOCAL
        frame itself; on a closed loop the club-side frame coincides with it.
        """
        return self.native_closure_frame(q)

    def club_inertia_split(self) -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]:
        """``(bare club, welded hand solids)`` in the club-side closure frame."""
        bare, hands = self._model.club_inertia_split()
        return _spatial(bare), tuple(_spatial(h) for h in hands)

    def closure_topology(self) -> LoopTopology:
        """Weld topology: hands close the loop onto the club."""
        actuated = set(self._actuated)
        return LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="hands",
            holding_side="club",
            actuated_indices=self._actuated,
            unactuated_indices=tuple(i for i in range(self.nv) if i not in actuated),
            loop_support_indices=self._model.closure_path_velocity_indices(),
            coordinate_order=self._names,
        )


def _spatial(inertia: RigidInertia) -> SpatialInertia:
    return SpatialInertia(
        mass=inertia.mass_kg,
        com=np.asarray(inertia.com_m),
        inertia=inertia.inertia_com_kg_m2,
    )
