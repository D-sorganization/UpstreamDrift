"""Native Pinocchio force equations for the shared allocation pathway."""

from __future__ import annotations

import json
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.pinocchio.python.marker_kinematics import CoordinateMap
from src.engines.physics_engines.pinocchio.python.native_model import (
    FullBodyPinocchioModel,
)
from src.shared.python.biomechanics.grip_extraction import (
    DEFAULT_GRIP_INTERFACE,
    GRIP_INTERFACES,
    ClubSolidsPartition,
    allocation_grip_analysis,
    bare_club_kinematics_from_body_frame,
    closing_side_from_closure,
    closure_and_club_analysis,
)
from src.shared.python.biomechanics.club_solids_partition import (
    BARE_CLUB_SOLID_NAMES,
    CLUB_BODY_SUFFIX,
    club_solids_partition,
)
from src.shared.python.biomechanics.grip_extraction import allocation_grip_analysis
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    normalize_capability,
)
from src.shared.python.loop_nullspace.closure_convention import (
    PINOCCHIO_CLOSURE_CONVENTION,
    NativeClosureConvention,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    FootContact,
    SpatialInertia,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import EngineType
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

Array: TypeAlias = NDArray[np.float64]


def _ensure_club_mass_metadata(spec: dict[str, Any]) -> dict[str, Any]:
    """Ensure spec contains club.total_mass_kg by summing bare-club solids if absent."""
    if isinstance(spec.get("club"), dict) and "total_mass_kg" in spec["club"]:
        return spec
    bare_mass = 0.0
    for body in spec.get("bodies", ()):
        if str(body.get("name", "")).endswith(CLUB_BODY_SUFFIX):
            for solid in body.get("solids", ()):
                leaf = str(solid.get("name", "")).rsplit("/", 1)[-1]
                if leaf in BARE_CLUB_SOLID_NAMES:
                    bare_mass += float(solid["mass_kg"])
    if bare_mass <= 0.0:
        return spec
    return {**spec, "club": {"total_mass_kg": bare_mass}}


def _kkt_constrained_acceleration(
    mass: Array, bias: Array, effort: Array, j_c: Array, jdot_v: Array
) -> Array:
    """Solve KKT constrained forward acceleration M a + h = effort + J_c^T lambda, J_c a + Jdot_v = 0."""
    m_inv = np.linalg.inv(mass)
    delassus = j_c @ m_inv @ j_c.T
    rhs_free = m_inv @ (effort - bias)
    lam = np.linalg.lstsq(delassus, -(j_c @ rhs_free + jdot_v), rcond=None)[0]
    return rhs_free + m_inv @ (j_c.T @ lam)


class PinocchioForceAdapter:
    """Raw equations in native scalar-coordinate order, not document order.

    Contact forces/Jacobians use world XYZ at sphere centers. Grip wrenches
    use the constraint LOCAL frame, linear then angular. Every method refreshes
    its state independently. Algebraic parity does not establish contact replay.
    Only the current full-body scalar-joint schema is supported; reject a
    different configuration manifold instead of guessing its coordinate map.
    """

    NATIVE_CLOSURE_CONVENTION = PINOCCHIO_CLOSURE_CONVENTION

    def __init__(self, spec_bytes: bytes) -> None:
        spec = json.loads(spec_bytes)
        self._plant: Any = FullBodyPinocchioModel(spec)
        self._pin: Any = self._plant._pin
        self._model: Any = self._plant.model
        self._data: Any = self._model.createData()
        self.closing_hand_side: str = closing_side_from_closure(spec["closure"])
        self._club_body_name: str = str(spec["closure"]["body_b"])
        self._gravity = np.asarray(spec["gravity_m_s2"], dtype=float)
        self.club_partition: ClubSolidsPartition | None = (
            club_solids_partition(spec) if "club" in spec else None
        )
        mapping = CoordinateMap.from_plant(self._plant)
        v_idx = mapping.v_index
        if (
            self._model.nq != self._model.nv
            or mapping.n != self._model.nv
            or sorted(v_idx.tolist()) != list(range(mapping.n))
            or not np.array_equal(mapping.q_index, v_idx)
        ):
            raise ValueError(
                "Force adapter requires a complete scalar q/v coordinate map"
            )
        self._names = tuple(mapping.names[i] for i in np.argsort(v_idx))
        if not set(ROOT_COORDINATES).issubset(self._names):
            raise ValueError(
                "Force adapter requires the named unactuated root coordinates"
            )
        self._actuated = tuple(
            i for i, name in enumerate(self._names) if name not in ROOT_COORDINATES
        )
        self._contact_names = tuple(s.name for s in self._plant.contact_spheres)
        self._contact_ids = tuple(
            self._model.getFrameId(name) for name in self._contact_names
        )
        self._oracle_snapshot: ClosedLoopSnapshot | None = None
        self.topology: Any = None
        self.model_provenance: Any = None

    @classmethod
    def from_spatial_oracle(
        cls,
        *,
        floating_base: bool = False,
        include_feet: bool = False,
    ) -> PinocchioForceAdapter:
        """Build a PinocchioForceAdapter backed by the NSX-12 spatial two-arm + club oracle."""
        from src.shared.python.loop_nullspace.fixtures import (
            spatial_two_arm_club_snapshot,
        )

        snap, _ = spatial_two_arm_club_snapshot(
            floating_base=floating_base, include_feet=include_feet
        )
        instance = cls.__new__(cls)
        instance._plant = None
        instance._pin = None
        instance._model = None
        instance._data = None
        instance._names = snap.topology.coordinate_order
        instance._actuated = snap.topology.actuated_indices
        instance._contact_names = tuple(fc.foot_id for fc in snap.foot_contacts)
        instance._contact_ids = tuple(range(len(snap.foot_contacts)))
        instance._oracle_snapshot = snap
        instance.topology = snap.topology
        instance.model_provenance = snap.model_provenance
        return instance

    @property
    def engine_type(self) -> EngineType:
        return EngineType.PINOCCHIO

    @property
    def native_closure_convention(self) -> NativeClosureConvention:
        return getattr(
            self, "_native_closure_convention", self.NATIVE_CLOSURE_CONVENTION
        )

    def closed_loop_snapshot_capability(self) -> CapabilityLevel:
        return CapabilityLevel.FULL

    def capability(self, cap: Capability | str) -> CapabilityLevel:
        if normalize_capability(cap) is Capability.CLOSED_LOOP_SNAPSHOT:
            return self.closed_loop_snapshot_capability()
        return CapabilityLevel.NONE

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
        snap = getattr(self, "_oracle_snapshot", None)
        if snap is not None:
            return snap.model_provenance.spec_sha256
        return getattr(self._plant, "model_sha256", "")

    def compute_mass_and_bias(self, q: Array, v: Array) -> tuple[Array, Array]:
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return snap.M.copy(), snap.h.copy()
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
        if getattr(self, "_oracle_snapshot", None) is not None:
            mass, bias = self.compute_mass_and_bias(configuration, velocity)
            return mass @ acceleration + bias
        return np.asarray(
            self._pin.rnea(
                self._model, self._data, configuration, velocity, acceleration
            ),
            dtype=float,
        ).copy()

    def compute_contact_jacobian(self, q: Array) -> Array:
        configuration = self._checked(q, "configuration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            if snap.J_g is None:
                return np.zeros((0, self.nv), dtype=float)
            return snap.J_g.copy()
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

    def foot_contacts_snapshot(
        self, q: Array
    ) -> tuple[Array | None, tuple[FootContact, ...]]:
        """Return (J_g, foot_contacts) at state q for ClosedLoopSnapshot construction."""
        configuration = self._checked(q, "configuration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return (
                None if snap.J_g is None else snap.J_g.copy(),
                snap.foot_contacts,
            )
        if self.n_contact_spheres == 0:
            return None, ()
        j_g = self.compute_contact_jacobian(configuration)
        spheres = getattr(self._plant, "contact_spheres", ())
        ground = getattr(self._plant, "ground", None)
        normal = (
            np.asarray(ground.normal, dtype=float)
            if ground is not None
            else np.array([0.0, 0.0, 1.0], dtype=float)
        )
        params = getattr(self._plant, "contact_parameters", None)
        mu = float(getattr(params, "mu_dynamic", 0.8))
        contacts: list[FootContact] = []
        for idx, sphere in enumerate(spheres):
            fid = self._contact_ids[idx]
            point_w = np.asarray(self._data.oMf[fid].translation, dtype=float).copy()
            contacts.append(
                FootContact(
                    foot_id=str(sphere.name),
                    point_world_m=point_w,
                    normal_world=normal,
                    jacobian=j_g[3 * idx : 3 * (idx + 1), :],
                    friction_coeff=mu,
                )
            )
        return j_g, tuple(contacts)

    def compute_grip_jacobian(self, q: Array) -> Array:
        configuration = self._checked(q, "configuration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return snap.J_c.copy()
        coordinates = dict(zip(self._names, configuration.tolist(), strict=True))
        closure_jac = self._plant.closure_force_jacobian(coordinates)
        return closure_jac.jacobian.copy()

    def club_newton_euler(
        self,
        q: Array,
        v: Array,
        a: Array,
        *,
        interface: str = DEFAULT_GRIP_INTERFACE,
    ) -> dict[str, Any]:
        """Club kinematics for Newton-Euler grip extraction (NSX-01, #12550)."""
        if interface not in GRIP_INTERFACES:
            raise ValueError(
                f"interface must be one of {GRIP_INTERFACES}, got {interface!r}"
            )
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        acceleration = self._checked(a, "acceleration")
        self._pin.forwardKinematics(
            self._model, self._data, configuration, velocity, acceleration
        )
        joint_id, body_pose = self._plant.body_placement(self._club_body_name)
        o_mb = self._data.oMi[joint_id] * body_pose
        origin_b = np.asarray(o_mb.translation, dtype=float)
        rot_wb = np.asarray(o_mb.rotation, dtype=float)
        v_local = body_pose.actInv(self._data.v[joint_id])
        a_local = body_pose.actInv(self._data.a[joint_id])
        v_origin = rot_wb @ np.asarray(v_local.linear, dtype=float)
        omega = rot_wb @ np.asarray(v_local.angular, dtype=float)
        alpha = rot_wb @ np.asarray(a_local.angular, dtype=float)
        a_origin = rot_wb @ np.asarray(a_local.linear, dtype=float) + np.cross(
            omega, v_origin
        )
        if self.club_partition is not None:
            inertia = (
                self.club_partition.bare_club_inertia
                if interface == DEFAULT_GRIP_INTERFACE
                else self.club_partition.composite_inertia
            )
            return bare_club_kinematics_from_body_frame(
                inertia,
                body_position_world_m=origin_b,
                body_rotation_world=rot_wb,
                body_linear_velocity_world_m_s=v_origin,
                body_linear_acceleration_world_m_s2=a_origin,
                body_angular_velocity_world_rad_s=omega,
                body_angular_acceleration_world_rad_s2=alpha,
                gravity_m_s2=self._gravity,
            )
        joint_inertia = self._model.inertias[joint_id]
        inertia_b = body_pose.actInv(joint_inertia)
        com_b = np.asarray(inertia_b.lever, dtype=float)
        r = rot_wb @ com_b
        return {
            "mass_kg": float(inertia_b.mass),
            "gravity_m_s2": self._gravity.copy(),
            "com_m": origin_b + r,
            "com_acceleration_m_s2": (
                a_origin + np.cross(alpha, r) + np.cross(omega, np.cross(omega, r))
            ),
            "inertia_world_kg_m2": (
                rot_wb @ np.asarray(inertia_b.inertia, dtype=float) @ rot_wb.T
            ),
            "angular_velocity_rad_s": omega,
            "angular_acceleration_rad_s2": alpha,
        }

    def compute_closure_drift(self, q: Array, v: Array) -> Array:
        """Return the 6-D native closure acceleration drift Jdot(q, v) v."""
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return snap.Jdot_v.copy()
        j_nat = self.compute_grip_jacobian(configuration)
        zero_tau = np.zeros(self.nv, dtype=float)
        a_zero = np.asarray(
            self._pin.constraintDynamics(
                self._model,
                self._plant.data,
                configuration,
                velocity,
                zero_tau,
                self._plant.constraints,
                self._plant.constraint_data,
            ),
            dtype=float,
        )
        drift = -(j_nat @ a_zero)
        if drift.shape != (6,) or not np.isfinite(drift).all():
            raise ValueError("Native closure drift is nonfinite")
        return drift

    def closure_wrench_frame(self, q: Array) -> tuple[Array, Array]:
        """Return (R_WG, origin_W) of the weld constraint frame at state q."""
        configuration = self._checked(q, "configuration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return snap.grip_frame.rotation.copy(), snap.grip_frame.origin.copy()
        coordinates = dict(zip(self._names, configuration.tolist(), strict=True))
        rotation, origin = self._plant.closure_frame_pose(coordinates)
        return (
            np.asarray(rotation, dtype=float).copy(),
            np.asarray(origin, dtype=float).copy(),
        )

    def club_inertia_split(
        self,
    ) -> tuple[SpatialInertia, tuple[SpatialInertia, ...]]:
        """Partition the club body into bare-club inertia and welded-hand inertias."""
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return snap.bare_club, snap.welded_hand_inertias
        spec = _ensure_club_mass_metadata(dict(self._plant.specification))
        partition = club_solids_partition(spec)
        bare = SpatialInertia(
            mass_kg=partition.bare_club_inertia.mass_kg,
            com_m=np.asarray(partition.bare_club_inertia.com_body_m, dtype=float),
            inertia_com_kg_m2=partition.bare_club_inertia.inertia_com_body_kg_m2,
            solid_names=("Clubhead", "Rigid Shaft", "Grip"),
        )
        hands = tuple(
            SpatialInertia(
                mass_kg=h.mass_kg,
                com_m=np.asarray(h.com_body_m, dtype=float),
                inertia_com_kg_m2=h.inertia_com_body_kg_m2,
                solid_names=(h.name.rsplit("/", 1)[-1],),
            )
            for h in partition.welded_hand_inertias
        )
        return bare, hands

    def closure_pose_residual(self, q: Array) -> Array:
        """Evaluate the 6-D closure pose error for project_to_closure and admission."""
        configuration = self._checked(q, "configuration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            snap = self._oracle_snapshot
            assert snap is not None
            return np.asarray(snap.J_c @ (configuration - snap.q), dtype=float)
        self._pin.forwardKinematics(self._model, self._plant.data, configuration)
        self._plant._refresh_constraint_data()
        c1mc2 = self._plant.constraint_data[0].c1Mc2
        return -np.asarray(self._pin.log6(c1mc2).vector, dtype=float).copy()

    def closure_rate_matrix(self, q: Array) -> Array:
        """Evaluate the 6xN closure velocity matrix for project_to_closure."""
        return self.compute_grip_jacobian(q)

    def grip_analysis_from_allocation(
        self,
        q: Array,
        lambda_grip: Array,
        *,
        v: Array | None = None,
        a: Array | None = None,
        interface: str = DEFAULT_GRIP_INTERFACE,
    ) -> GripAnalysis:
        """Grip analysis of one allocated ``lambda_grip`` (GCV-8, #11714, #12550).

        When ``v`` and ``a`` are omitted, returns the net 6-D allocation wrench
        at the closure frame origin. When ``v`` and ``a`` are provided, uses
        club Newton-Euler (with the bare-club partition by default) to resolve
        both the closing and holding hand wrenches.
        """
        if (v is None) != (a is None):
            raise ValueError(
                "Both v and a must be provided together for club Newton-Euler grip split"
            )
        configuration = self._checked(q, "configuration")
        wrench = np.asarray(lambda_grip, dtype=float)
        if wrench.shape != (6,) or not np.isfinite(wrench).all():
            raise ValueError(
                f"lambda_grip must be a finite 6-vector, got shape {wrench.shape}"
            )
        coordinates = dict(zip(self._names, configuration.tolist(), strict=True))
        rotation, origin = self._plant.closure_frame_pose(coordinates)
        if v is None or a is None:
            return allocation_grip_analysis(
                wrench,
                point_m=origin,
                ordering="force_torque",
                load_on="human",
                rotation_world_from_frame=rotation,
                interface=interface,
            )
        club_kin = self.club_newton_euler(q, v, a, interface=interface)
        joint_id, _ = self._plant.body_placement(self._club_body_name)
        o_mi = self._data.oMi[joint_id]
        holding_point = np.asarray(o_mi.translation, dtype=float)
        return closure_and_club_analysis(
            closing_side=self.closing_hand_side,
            closing_point_m=origin,
            closing_force_n=-rotation @ wrench[:3],
            closing_torque_nm=-rotation @ wrench[3:],
            holding_point_m=holding_point,
            club=club_kin,
            split_method="allocation",
            interface=interface,
            metadata={
                "engine": "pinocchio",
                "load_on": "human",
                "club_mass_kg": float(club_kin["mass_kg"]),
            },
        )

    def grip_analyses_from_allocation(
        self,
        q_traj: Array,
        lambda_grip: Array,
        *,
        interface: str = DEFAULT_GRIP_INTERFACE,
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
            self.grip_analysis_from_allocation(q_k, lam_k, interface=interface)
            for q_k, lam_k in zip(q_arr, lam_arr, strict=True)
        ]

    def verify_acceleration_parity(
        self, q: Array, v: Array, tau_effective: Array, a_target: Array
    ) -> float:
        """Verify forward acceleration parity using constrained KKT dynamics (constraintDynamics)."""
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        effort = self._checked(tau_effective, "effective effort")
        target = self._checked(a_target, "target acceleration")
        if getattr(self, "_oracle_snapshot", None) is not None:
            mass, bias = self.compute_mass_and_bias(configuration, velocity)
            j_c = self.compute_grip_jacobian(configuration)
            jdot_v = self.compute_closure_drift(configuration, velocity)
            actual = _kkt_constrained_acceleration(mass, bias, effort, j_c, jdot_v)
        elif getattr(self._plant, "constraints", None) and hasattr(
            self._pin, "constraintDynamics"
        ):
            actual = np.asarray(
                self._pin.constraintDynamics(
                    self._model,
                    self._plant.data,
                    configuration,
                    velocity,
                    effort,
                    self._plant.constraints,
                    self._plant.constraint_data,
                ),
                dtype=float,
            )
        else:
            actual = np.asarray(
                self._pin.aba(self._model, self._data, configuration, velocity, effort),
                dtype=float,
            )
        if not np.isfinite(actual).all():
            raise ValueError("Native forward acceleration is nonfinite")
        return float(np.max(np.abs(actual - target)))
