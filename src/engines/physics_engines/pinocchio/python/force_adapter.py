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
    club_solids_partition,
)
from src.shared.python.biomechanics.grip_wrench import GripAnalysis
from src.shared.python.motion_matching.multi_engine_torque_allocator import EngineType
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

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
        spec = json.loads(spec_bytes)
        self._plant = FullBodyPinocchioModel(spec)
        self._pin = self._plant._pin
        self._model = self._plant.model
        self._data = self._model.createData()
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

    def compute_grip_jacobian(self, q: Array) -> Array:
        configuration = self._checked(q, "configuration")
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
        configuration = self._checked(q, "configuration")
        velocity = self._checked(v, "velocity")
        effort = self._checked(tau_effective, "effective effort")
        target = self._checked(a_target, "target acceleration")
        actual = self._pin.aba(self._model, self._data, configuration, velocity, effort)
        if not np.isfinite(actual).all():
            raise ValueError("Native forward acceleration is nonfinite")
        return float(np.max(np.abs(actual - target)))
