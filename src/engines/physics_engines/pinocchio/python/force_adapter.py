"""Native Pinocchio force equations for the shared allocation pathway."""

from __future__ import annotations

import json
from typing import TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.closed_loop_derivatives import (
    PinocchioAnalyticDerivatives,
)
from src.engines.physics_engines.pinocchio.python.marker_kinematics import CoordinateMap
from src.engines.physics_engines.pinocchio.python.native_model import (
    CLOSURE_CONVENTION,
    FullBodyPinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.pinocchio_force_torque import (
    PinocchioForceTorqueSource,
)
from src.shared.python.biomechanics.club_inertia_partition import (
    RigidInertia,
    club_solids_partition,
    newton_euler_keywords,
)
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
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
from src.shared.python.loop_nullspace.evidence import FootContact
from src.shared.python.loop_nullspace.ground_contact import ContactCandidate
from src.shared.python.motion_matching.multi_engine_torque_allocator import EngineType
from src.shared.python.motion_matching.polynomial_actuation import (
    actuated_coordinate_indices,
)

Array: TypeAlias = NDArray[np.float64]


class PinocchioForceAdapter:
    """Raw equations in native scalar-coordinate order, not document order.

    Contact forces/Jacobians use world XYZ at sphere centers. Grip wrenches
    use the constraint LOCAL frame, linear then angular. Every method refreshes
    its state independently. Algebraic parity does not establish contact replay.
    Only the current full-body scalar-joint schema is supported; reject a
    different configuration manifold instead of guessing its coordinate map.
    """

    def __init__(self, spec_bytes: bytes) -> None:
        self._plant = FullBodyPinocchioModel(json.loads(spec_bytes))
        self._pin = self._plant._pin
        self._model = self._plant.model
        self._data = self._model.createData()
        mapping = CoordinateMap.from_plant(self._plant)
        if (
            self._model.nq != self._model.nv
            or mapping.n != self._model.nv
            or sorted(mapping.v_index.tolist()) != list(range(mapping.n))
            or not np.array_equal(mapping.q_index, mapping.v_index)
        ):
            raise ValueError(
                "Force adapter requires a complete scalar q/v coordinate map"
            )
        self._names = tuple(mapping.names[i] for i in np.argsort(mapping.v_index))
        self._actuated = actuated_coordinate_indices(self._names)
        self._contact_names = tuple(s.name for s in self._plant.contact_spheres)
        self._contact_ids = tuple(
            self._model.getFrameId(name) for name in self._contact_names
        )
        self._spec = json.loads(spec_bytes)
        self._bare_club: RigidInertia | None = None

    @property
    def engine_type(self) -> EngineType:
        return EngineType.PINOCCHIO

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
        return len(self._contact_ids)

    @property
    def model_hash(self) -> str:
        return getattr(self._plant, "model_sha256", "")

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        mass = np.asarray(
            self._pin.crba(self._model, self._data, configuration), dtype=float
        ).copy()
        bias = np.asarray(
            self._pin.nonLinearEffects(
                self._model, self._data, configuration, velocity
            ),
            dtype=float,
        ).copy()
        return mass, bias

    def _checked(self, values: Array, name: str) -> Array:
        result = np.asarray(values, dtype=float)
        if result.shape != (self.nv,) or not np.isfinite(result).all():
            raise ValueError(f"{name} must be a finite vector with shape ({self.nv},)")
        return result

    def compute_inverse_dynamics(self, q: Array, v: Array, a: Array) -> Array:
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        acceleration = self._checked(a, "acceleration")
        return np.asarray(
            self._pin.rnea(
                self._model, self._data, configuration, velocity, acceleration
            ),
            dtype=float,
        ).copy()

    # -- inertial parameters (NSX-62, #12587) --------------------------------

    @property
    def n_inertial_parameters(self) -> int:
        """Ten dynamic parameters per moving joint's body."""
        return 10 * (self._model.njoints - 1)

    def inertial_parameters(self) -> Array:
        """Body inertias as a flat vector, ten per joint (Pinocchio layout).

        ``[m, h(3), Ixx, Ixy, Iyy, Ixz, Iyz, Izz]`` per body, inertia about the
        joint-frame origin (``Inertia.toDynamicParameters``); a copy.
        """
        return np.concatenate(
            [
                np.asarray(self._model.inertias[j].toDynamicParameters(), dtype=float)
                for j in range(1, self._model.njoints)
            ]
        )

    def set_inertial_parameters(self, parameters: Array) -> None:
        """Replace the body inertias by ``parameters`` (see the getter's layout).

        Only bodies whose parameters differ are rebuilt; a zero-mass body must
        be all zeros.  Raises ``ValueError`` on a wrong size or nonfinite entry.
        The contact spheres, closure and club split do not depend on inertias.
        """
        values = np.asarray(parameters, dtype=float)
        if (
            values.shape != (self.n_inertial_parameters,)
            or not np.isfinite(values).all()
        ):
            raise ValueError(
                f"parameters must be {self.n_inertial_parameters} finite values"
            )
        inertia_type = self._pin.Inertia
        current = self.inertial_parameters().reshape(-1, 10)
        for index, (new, old) in enumerate(
            zip(values.reshape(-1, 10), current, strict=True)
        ):
            if np.array_equal(new, old):
                continue
            if new[0] == 0.0:
                if np.any(new != 0.0):
                    raise ValueError(f"massless body {index} has nonzero parameters")
                self._model.inertias[index + 1] = inertia_type.Zero()
            else:
                self._model.inertias[index + 1] = inertia_type.FromDynamicParameters(
                    new
                )

    @property
    def club_body_index(self) -> int:
        """Index of the club carrier among the bodies of :meth:`inertial_parameters`."""
        return self._bare_club_in_joint_frame()[0] - 1

    def torque_regressor(self, q: Array, v: Array, a: Array) -> Array:
        """Joint-torque regressor ``Y`` with ``Y pi = rnea(q, v, a)`` (nv, n_pi)."""
        return np.asarray(
            self._pin.computeJointTorqueRegressor(
                self._model,
                self._data,
                self._checked(q, "configuration"),
                self._checked(v, "velocity"),
                self._checked(a, "acceleration"),
            ),
            dtype=float,
        ).copy()

    def compute_contact_jacobian(self, q: Array) -> Array:
        configuration = self._checked(q, "configuration")
        self._pin.computeJointJacobians(self._model, self._data, configuration)
        self._pin.updateFramePlacements(self._model, self._data)
        result = np.empty((3 * self.n_contact_spheres, self.nv))
        ref_frame = self._pin.ReferenceFrame
        for i, frame in enumerate(self._contact_ids):
            jac = self._pin.getFrameJacobian(
                self._model,
                self._data,
                frame,
                ref_frame.LOCAL_WORLD_ALIGNED,
            )
            result[3 * i : 3 * i + 3] = jac[:3]
        return result

    def foot_contact_candidates(self, q: Array) -> tuple[ContactCandidate, ...]:
        """Contact spheres as candidates, in ``contact_names`` order (NSX-60).

        Each candidate sits at the sphere centre (where
        :meth:`compute_contact_jacobian` evaluates its rows), with the ground
        normal and the spec's static friction coefficient (a declared
        hypothesis, ``H-FRICTION-CONE``).  ``gap_m`` is the signed distance
        from the sphere surface to the ground plane (negative when the compliant
        contact law penetrates).
        """
        configuration = self._checked(q, "configuration")
        self._pin.forwardKinematics(self._model, self._data, configuration)
        self._pin.updateFramePlacements(self._model, self._data)
        ground = self._plant.ground
        normal = np.asarray(ground.normal, dtype=float)
        mu = float(self._spec["contact"]["parameters"]["static_friction"])
        out = []
        for sphere, frame in zip(
            self._plant.contact_spheres, self._contact_ids, strict=True
        ):
            center = np.asarray(self._data.oMf[frame].translation, dtype=float).copy()
            gap = float(normal @ center) - ground.height_m - sphere.radius_m
            out.append(
                ContactCandidate(FootContact(sphere.name, center, normal, mu), gap)
            )
        return tuple(out)

    def compute_grip_jacobian(self, q: Array) -> Array:
        configuration = self._checked(q, "configuration")
        coordinates = dict(zip(self._names, configuration.tolist(), strict=True))
        return self._plant.closure_force_jacobian(coordinates).jacobian.copy()

    def grip_analysis_from_allocation(
        self, q: Array, lambda_grip: Array
    ) -> GripAnalysis:
        """Net grip analysis of one allocated ``lambda_grip`` (GCV-8, #11714).

        ``lambda_grip`` is the allocator's 6-D closure wrench (linear then
        angular, constraint LOCAL frame, ``J.T @ lambda`` the load on the
        human).  It is rotated to the world and reported at the closure frame
        origin through :meth:`PinocchioForceTorqueSource.grip_from_allocation`:
        ``split_method="allocation"``, per-hand wrenches unavailable.
        """
        configuration = self._checked(q, "configuration")
        wrench = np.asarray(lambda_grip, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError(
                f"lambda_grip must be a finite 6-vector, got shape {wrench.shape}"
            )
        coordinates = dict(zip(self._names, configuration.tolist(), strict=True))
        rotation, origin = self._plant.closure_frame_pose(coordinates)
        return PinocchioForceTorqueSource.grip_from_allocation(
            wrench,
            point_m=origin,
            ordering="force_torque",
            load_on="human",
            rotation_world_from_frame=rotation,
        )

    def _bare_club_in_joint_frame(self) -> tuple[int, RigidInertia]:
        """``(joint index, bare-club inertia in that joint frame)``, cached."""
        body_b = self._spec["closure"]["body_b"]
        joint, pose = self._plant.body_placement(body_b)
        if self._bare_club is None:
            bare = club_solids_partition(self._spec).bare_club
            self._bare_club = bare.expressed_in(pose.homogeneous)
        return joint, self._bare_club

    def club_newton_euler(self, q: Array, v: Array, a: Array) -> dict:
        """Bare-club Newton-Euler keywords at ``(q, v, a)`` (NSX-01, #12550).

        Same keywords as ``holding_hand_wrench`` takes, built from head, shaft
        and grip only; the hand solids welded into the club body are excluded.
        ``a`` is the constrained generalised acceleration in native order.
        """
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        acceleration = self._checked(a, "acceleration")
        pin = self._pin
        pin.forwardKinematics(
            self._model, self._data, configuration, velocity, acceleration
        )
        joint, inertia = self._bare_club_in_joint_frame()
        gravity = self._model.gravity
        world_aligned = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        placement = self._data.oMi[joint]
        motion = pin.getVelocity(self._model, self._data, joint, world_aligned)
        spatial_acc = pin.getAcceleration(self._model, self._data, joint, world_aligned)
        classical = pin.getClassicalAcceleration(
            self._model, self._data, joint, world_aligned
        )
        return newton_euler_keywords(
            inertia,
            rot_world_from_body=placement.rotation,
            origin_world=placement.translation,
            omega=motion.angular,
            alpha=spatial_acc.angular,
            accel_origin=classical.linear,
            gravity=gravity.linear,
        )

    def bare_club_net_wrench(self, q: Array, v: Array, a: Array) -> tuple[Array, Array]:
        """Net hand-on-club ``(force, torque about CoM)`` of the bare club.

        ``F = m (a_c - g)`` and ``T = I alpha + omega x (I omega)`` in world
        axes: the invariant (R4) the hands must supply together.
        """
        ne = self.club_newton_euler(q, v, a)
        inertia = np.asarray(ne["inertia_world_kg_m2"])
        omega = np.asarray(ne["angular_velocity_rad_s"])
        force = ne["mass_kg"] * (
            np.asarray(ne["com_acceleration_m_s2"]) - np.asarray(ne["gravity_m_s2"])
        )
        torque = inertia @ np.asarray(ne["angular_acceleration_rad_s2"]) + np.cross(
            omega, inertia @ omega
        )
        return force, torque

    def grip_analyses_from_allocation(
        self, q_traj: Array, lambda_grip: Array
    ) -> list[GripAnalysis]:
        """:meth:`grip_analysis_from_allocation` for each frame of a trajectory."""
        q_arr = np.asarray(q_traj, dtype=float)
        lam_arr = np.asarray(lambda_grip, dtype=float)
        if q_arr.ndim != 2 or lam_arr.shape != (q_arr.shape[0], 6):
            raise ValueError(
                "frame count mismatch: q_traj must be (n, nv) and lambda_grip "
                f"(n, 6), got {q_arr.shape} and {lam_arr.shape}"
            )
        return [
            self.grip_analysis_from_allocation(q_k, lam_k)
            for q_k, lam_k in zip(q_arr, lam_arr, strict=True)
        ]

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """``||constraintDynamics(q, v, tau) - a_target||_inf`` (one semantics).

        ``tau_effective`` is the *applied* joint effort only: actuator torques
        plus any other external generalised force (ground reaction, root
        residual).  The closure reaction ``J_c^T lambda`` is NOT part of it;
        Pinocchio's constrained forward dynamics supplies its own multiplier,
        so ``a* = constraintDynamics(q, v, tau)`` is the acceleration the loop
        really produces and the target is compared with it (#12558).  A target
        built with an inconsistent multiplier therefore differs from ``a*``.
        Raises ``ValueError`` on nonfinite input or result.
        """
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        effort = self._checked(tau_effective, "effective effort")
        target = self._checked(a_target, "target acceleration")
        try:
            actual = self._plant.constrained_acceleration(
                configuration, velocity, effort
            )
        except FloatingPointError as exc:
            raise ValueError("Native forward acceleration is nonfinite") from exc
        return float(np.max(np.abs(actual - target)))

    # -- closed-loop snapshot capability (NSX-13, #12558) -------------------

    def capability(self, capability: Capability) -> CapabilityLevel:
        """Level for ``capability``; only adapter-boundary ones are declared.

        ``CLOSED_LOOP_SNAPSHOT`` is FULL: the adapter provides the closure
        Jacobian, its drift, the wrench frame, the club inertia split and the
        native convention.  ``CLOSED_LOOP_DERIVATIVES`` is FULL (NSX-85): the
        derivatives are Pinocchio's analytic ones, not finite differences.
        Anything else answers ``NONE`` here.
        """
        if capability in (
            Capability.CLOSED_LOOP_SNAPSHOT,
            Capability.CLOSED_LOOP_DERIVATIVES,
        ):
            return CapabilityLevel.FULL
        return CapabilityLevel.NONE

    @property
    def native_plant(self) -> FullBodyPinocchioModel:
        """The native full-body plant behind this adapter (read access only)."""
        return self._plant

    def constrained_acceleration_derivatives(
        self, q: Array, v: Array, tau: Array
    ) -> ClosedLoopDerivatives:
        """Analytic tangent-bundle derivatives of ``(a, lambda)`` (NSX-85).

        ``q, v, tau`` are native-order vectors (this adapter's
        ``coordinate_order``); ``v`` must satisfy ``J_c v = 0``.  Wraps
        ``computeConstraintDynamicsDerivatives``; the result is tagged
        ``derivative_method=analytic`` and expressed in the canonical closure
        convention on the closure tangent basis of NSX-84.
        """
        return PinocchioAnalyticDerivatives(self).constrained_acceleration_derivatives(
            q, v, tau
        )

    @property
    def native_closure_convention(self) -> NativeClosureConvention:
        """Layout of :meth:`compute_grip_jacobian` rows (NSX-02 constant)."""
        return CLOSURE_CONVENTION

    def _named_state(self, values: Array, name: str) -> dict[str, float]:
        return dict(zip(self._names, self._checked(values, name).tolist(), strict=True))

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Native ``Jdot v`` (6,) of :meth:`compute_grip_jacobian` (NSX-13)."""
        return np.array(
            self._plant.closure_velocity_drift(
                self._named_state(q, "configuration"), self._named_state(v, "velocity")
            )
        )

    def native_closure_frame(self, q: Array) -> tuple[Array, Array]:
        """World rotation and origin of the frame the native rows use (LOCAL)."""
        return self._plant.closure_frame_pose(self._named_state(q, "configuration"))

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """Canonical grip frame ``(R world<-grip, origin)`` (NSX-13).

        For Pinocchio the grip frame is the hand-side weld frame, i.e. the
        native LOCAL frame itself, so native to canonical conversion reduces to
        the identity.  The club-side frame coincides with it on a closed loop.
        """
        return self.native_closure_frame(q)

    def club_inertia_split(self) -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]:
        """``(bare club, welded hand solids)`` in the club-side closure frame.

        Body-fixed (state independent).  The hand solids are welded into the
        club body in the specification (NSX-01) and belong to the hand side.
        """
        joint, bare_in_joint = self._bare_club_in_joint_frame()
        constraint = self._plant.constraints[0]
        if int(constraint.joint2_id) != joint:
            raise ValueError("closure club-side joint differs from the club body joint")
        to_frame = np.linalg.inv(np.asarray(constraint.joint2_placement.homogeneous))
        _, body_pose = self._plant.body_placement(self._spec["closure"]["body_b"])
        hands = club_solids_partition(self._spec).welded_hands
        in_frame = to_frame @ np.asarray(body_pose.homogeneous)
        return (
            _spatial(bare_in_joint.expressed_in(to_frame)),
            tuple(_spatial(h.expressed_in(in_frame)) for h in hands.values()),
        )

    def closure_topology(self) -> LoopTopology:
        """Weld topology: hands close the loop onto the club (NSX-13).

        ``loop_support_indices`` are the coordinates on the tree path between
        the two constrained joints (common ancestors cancel in a relative
        twist), in this adapter's native order.
        """
        actuated = set(self._actuated)
        return LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="hands",
            holding_side="club",
            actuated_indices=self._actuated,
            unactuated_indices=tuple(i for i in range(self.nv) if i not in actuated),
            loop_support_indices=self._plant.closure_path_velocity_indices(),
            coordinate_order=self._names,
        )


def _spatial(inertia: RigidInertia) -> SpatialInertia:
    return SpatialInertia(
        mass=inertia.mass_kg,
        com=np.asarray(inertia.com_m),
        inertia=inertia.inertia_com_kg_m2,
    )
