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
from collections.abc import Mapping
from typing import Any, Final, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.drake.python.closed_loop_autodiff import (
    DrakeAutoDiffDerivatives,
)
from src.engines.physics_engines.drake.python.full_body_model import (
    CLOSURE_CONVENTION,
    FullBodyDrakeModel,
    solve_weld_acceleration,
)
from src.engines.physics_engines.mujoco.python.hand_mobility import HandMobility
from src.shared.python.biomechanics.club_inertia_partition import RigidInertia
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.autodiff_derivatives import ModelDifferentials
from src.shared.python.loop_nullspace.capacity import CapacitySubject
from src.shared.python.loop_nullspace.closure_convention import (
    NativeClosureConvention,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosureKind,
    LoopTopology,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.derivative_capability import (
    ClosedLoopDerivatives,
)
from src.shared.python.loop_nullspace.torque_limits import (
    TorqueLimits,
    effective_torque_limits_for_capacity,
    resolve_torque_limits,
    validate_capacity_hand_mode,
)
from src.shared.python.model_appearance.club_assembly import assembly_from_spec
from src.shared.python.model_appearance.grip_hands import grip_hand_frame
from src.shared.python.model_appearance.grip_pose import (
    DEFAULT_GRIP_POSE,
    GripPose,
    Hand,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import EngineType
from src.shared.python.motion_matching.polynomial_actuation import (
    actuated_coordinate_indices,
)
from src.shared.python.spatial_algebra.operational_space import (
    CapacityEllipsoid,
    acceleration_capacity,
    force_capacity,
    force_ellipsoid,
    mobility_matrix,
    velocity_ellipsoid,
)

Array: TypeAlias = NDArray[np.float64]

HANDS: Final = ("lead", "trail")
MODES: Final = ("constrained", "unconstrained")


_FULL = frozenset({Capability.CLOSED_LOOP_SNAPSHOT, Capability.CLOSED_LOOP_DERIVATIVES})


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
        self._actuated = actuated_coordinate_indices(self._names)
        self._contact_names = tuple(s["name"] for s in self._spec["contact"]["spheres"])
        self._derivatives: DrakeAutoDiffDerivatives | None = None

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
        """Snapshot and AutoDiff derivatives are FULL; everything else is NONE."""
        return CapabilityLevel.FULL if capability in _FULL else CapabilityLevel.NONE

    def closed_loop_differentials(self, q: Array, v: Array) -> ModelDifferentials:
        """AutoDiffXd differentials of the native ``M, h, J, Jdot v`` (NSX-86)."""
        return self._model.closed_loop_differentials(
            self._checked(q, "configuration"), self._checked(v, "velocity")
        )

    def constrained_acceleration_derivatives(
        self, q: Array, v: Array, tau: Array
    ) -> ClosedLoopDerivatives:
        """Exact tangent-bundle derivatives of ``(a, lambda)`` by AutoDiff (NSX-86).

        ``tau`` is the generalised applied load in spec order.  Result: method
        ``AUTODIFF``, closure tangent basis, canonical closure convention (R8).
        Raises ``ValueError`` on bad input (including a non-tangent ``v``),
        ``ClosureRankChangeError`` at a rank change and
        ``DerivativeCapabilityUnavailable`` for a component AutoDiff cannot
        differentiate; it never falls back to finite differences.
        """
        if self._derivatives is None:
            self._derivatives = DrakeAutoDiffDerivatives(self)
        return self._derivatives.constrained_acceleration_derivatives(q, v, tau)

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

    def hand_mobility_source(
        self,
        pose: GripPose = DEFAULT_GRIP_POSE,
        torque_limits: TorqueLimits | CapacitySubject | Mapping[str, Any] | None = None,
    ) -> DrakeCapabilitySource:
        """Construct a DrakeCapabilitySource sharing this adapter's spec (#14048)."""
        return DrakeCapabilitySource(
            json.dumps(self._spec).encode("utf-8"), pose, torque_limits
        )


class DrakeCapabilitySource:
    """Drake interface capability and mobility source on the v2 golfer (#14048)."""

    def __init__(
        self,
        spec_bytes: bytes,
        pose: GripPose = DEFAULT_GRIP_POSE,
        torque_limits: TorqueLimits | CapacitySubject | Mapping[str, Any] | None = None,
    ) -> None:
        if isinstance(spec_bytes, bytes):
            self._spec: dict[str, Any] = json.loads(spec_bytes.decode("utf-8"))
            raw_bytes = spec_bytes
        elif isinstance(spec_bytes, str):
            self._spec = json.loads(spec_bytes)
            raw_bytes = spec_bytes.encode("utf-8")
        else:
            self._spec = dict(spec_bytes)
            raw_bytes = json.dumps(self._spec).encode("utf-8")

        self._adapter = DrakeForceAdapter(raw_bytes)
        self._model = self._adapter._model
        self.plant = self._model.plant
        self.context = self._model.context
        self.coordinate_order: tuple[str, ...] = self._adapter.coordinate_order

        club = assembly_from_spec(self._spec)
        if club is None:
            raise ValueError("the specification has no club for the grip points")
        self._grip_points: dict[str, np.ndarray] = {
            "lead": np.asarray(
                grip_hand_frame(club, Hand.LEAD, pose)[:3, 3], dtype=float
            ),
            "trail": np.asarray(
                grip_hand_frame(club, Hand.TRAIL, pose)[:3, 3], dtype=float
            ),
        }
        self.topology: str = (
            "rigid 6-D closure on Drake (explicit KKT weld closure); both grip points "
            "are taken on the club body (welded lead hand topology: lead hand is welded "
            "to the club in v2 golfer, so lead and trail share one rigid body)"
        )
        self._torque_limits = resolve_torque_limits(
            torque_limits, self._spec, self.coordinate_order
        )

    def grip_point_world(
        self, hand: str, coordinates: Mapping[str, float] | Array
    ) -> Array:
        """3-vector: position of ``hand``'s grip frame origin in world [m]."""
        if hand not in HANDS:
            raise ValueError(f"hand must be one of {HANDS}, got {hand!r}")
        q_vec = self._model._named_vector(coordinates)
        self.plant.SetPositions(self.context, self._model.to_plant_positions(q_vec))
        grip_in_club = self._grip_points[hand]
        _, frame_b = self._model._closure
        club_pose = self.plant.EvalBodyPoseInWorld(self.context, frame_b.body())
        return np.asarray(
            club_pose.translation() + club_pose.rotation().matrix() @ grip_in_club,
            dtype=float,
        )

    def _task_matrices(
        self, hand: str, q: Array, constrained: bool
    ) -> tuple[Array, Array, Array | None]:
        q_vec = self._model._named_vector(q)
        self.plant.SetPositions(self.context, self._model.to_plant_positions(q_vec))
        grip_in_club = self._grip_points[hand]
        _, frame_b = self._model._closure
        club_frame = frame_b.body().body_frame()

        jac_plant = self.plant.CalcJacobianTranslationalVelocity(
            self.context,
            self._model._wrt_v,
            club_frame,
            grip_in_club,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        jac = np.asarray(self._model.named_columns(jac_plant), dtype=float)
        mass, _ = self._adapter.compute_mass_and_bias(q_vec, np.zeros(self._adapter.nv))
        closure = (
            np.asarray(self._adapter.compute_grip_jacobian(q_vec), dtype=float)
            if constrained
            else None
        )
        return jac, mass, closure

    def _capacity_task(
        self,
        q: Array,
        hand: str,
        mode: str,
        torque_limits: TorqueLimits | None,
        label: str,
    ) -> tuple[Array, Array, Array | None, TorqueLimits]:
        validate_capacity_hand_mode(hand, mode, hands=HANDS, modes=MODES)
        limits = effective_torque_limits_for_capacity(
            torque_limits, self._torque_limits, label=label
        )
        J, M, Jc = self._task_matrices(hand, q, mode == "constrained")
        return J, M, Jc, limits

    def acceleration_capacity(
        self,
        q: Array,
        hand: str,
        *,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> CapacityEllipsoid:
        """Acceleration-capacity ellipsoid at ``hand``'s grip point [m/s^2]."""
        J, M, Jc, limits = self._capacity_task(
            q, hand, mode, torque_limits, "acceleration"
        )
        return acceleration_capacity(J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc)

    def force_capacity(
        self,
        q: Array,
        hand: str,
        *,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> CapacityEllipsoid:
        """Dynamically consistent force-capacity ellipsoid at ``hand``'s grip point [N]."""
        J, M, Jc, limits = self._capacity_task(q, hand, mode, torque_limits, "force")
        return force_capacity(J, M, limits.B, limits.tau_lo, limits.tau_hi, Jc)

    def hand_mobility(
        self,
        q: Array,
        *,
        mode: str = "constrained",
        torque_limits: TorqueLimits | None = None,
    ) -> dict[str, HandMobility]:
        """Lead and trail 3x3 mobility and capability at state ``q`` (``coordinate_order``)."""
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        limits = torque_limits or self._torque_limits
        provenance = {
            "engine": "drake",
            "coordinates": self.coordinate_order,
            "frame": "world",
            "mode": mode,
            "point": "grip_hand_frame origin on the club",
            "welded_lead_hand": True,
            "caption": (
                "Lead hand is welded to the club in v2 golfer topology; "
                "lead and trail share rigid body kinematics on the club."
            ),
            "torque_limit_source": limits.source_id if limits is not None else None,
            "assumed_slots": limits.assumed_slots if limits is not None else 0,
            "verified_slots": limits.verified_slots if limits is not None else 0,
        }
        out: dict[str, HandMobility] = {}
        for hand in HANDS:
            point = self.grip_point_world(hand, q)
            J, M, Jc = self._task_matrices(hand, q, mode == "constrained")
            w = mobility_matrix(J, M, Jc)

            acc_cap: CapacityEllipsoid | None = None
            f_cap: CapacityEllipsoid | None = None
            if limits is not None:
                acc_cap = self.acceleration_capacity(
                    q, hand, mode=mode, torque_limits=limits
                )
                f_cap = self.force_capacity(q, hand, mode=mode, torque_limits=limits)

            v_ell = velocity_ellipsoid(w)
            f_ell = force_ellipsoid(w)
            m_eff = f_ell.radii**2
            c_val = (
                float(m_eff.max() / m_eff.min()) if m_eff.min() > 0.0 else float("inf")
            )

            out[hand] = HandMobility(
                hand=hand,
                centre=(float(point[0]), float(point[1]), float(point[2])),
                mobility=w,
                mobility_axes=v_ell.axes,
                mobility_eigenvalues=np.maximum(v_ell.radii**2, 0.0),
                effective_mass_kg=m_eff,
                condition=c_val,
                singular=np.asarray(f_ell.singular, dtype=bool),
                provenance=provenance,
                acceleration_capacity=acc_cap,
                force_capacity=f_cap,
            )
        return out


def _spatial(inertia: RigidInertia) -> SpatialInertia:
    return SpatialInertia(
        mass=inertia.mass_kg,
        com=np.asarray(inertia.com_m),
        inertia=inertia.inertia_com_kg_m2,
    )
