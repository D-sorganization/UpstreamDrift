"""Native-derived Pinocchio tree with the original right-hand weld constraint.

The input is a portable native_spec geometry export. Actuator routing, damping,
limits and time integration require separate qualification before swing fitting.
"""

import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.full_body_spec import (
    FULL_BODY_SCHEMA_VERSION,
    ContactSphere,
    upper_body_slice,
)
from src.shared.python.motion_matching.marker_projection import project_markers

# The principal SE(3) logarithm changes branch at a rotation of pi. Keep
# derivative claims outside a small explicit numerical margin of that cut.
_WELD_LOG_BRANCH_MARGIN_RAD = 1e-7


def _require_weld_log_chart(
    pose_error: NDArray[np.float64], margin_rad: float = _WELD_LOG_BRANCH_MARGIN_RAD
) -> None:
    if np.linalg.norm(pose_error[3:]) >= np.pi - margin_rad:
        raise ValueError("Weld log derivative is undefined near the rotation-pi branch")


class NativeAccelerationDerivatives(NamedTuple):
    """Detached derivatives in the stated native scalar-coordinate order."""

    names: tuple[str, ...]
    dq: NDArray[np.float64]
    dv: NDArray[np.float64]
    deffort: NDArray[np.float64]


class NativeMarkerDerivatives(NamedTuple):
    """World positions and marker-by-xyz-by-coordinate tree derivatives."""

    names: tuple[str, ...]
    positions_m: NDArray[np.float64]
    dposition_dq: NDArray[np.float64]


class NativeContactEffortDerivatives(NamedTuple):
    """Contact-force derivatives in explicit scalar-coordinate order."""

    names: tuple[str, ...]
    dq: NDArray[np.float64]
    dv: NDArray[np.float64]
    differentiable: bool


class NativeClosurePositionLinearization(NamedTuple):
    """Weld pose residual and Jacobian in an explicit native coordinate order."""

    names: tuple[str, ...]
    position: NDArray[np.float64]
    jacobian: NDArray[np.float64]


class NativeClosureForceJacobian(NamedTuple):
    """Velocity constraint and its wrench dual in the constraint LOCAL frame.

    Rows are linear then angular; columns follow names. A wrench in this
    constraint frame maps to generalized effort as jacobian.T @ wrench.
    This is distinct from differentiating a finite SE(3) pose-error logarithm.
    """

    names: tuple[str, ...]
    jacobian: NDArray[np.float64]


class NativeClosureTrajectoryResiduals(NamedTuple):
    """Pose, velocity, and acceleration weld residuals at one native state."""

    position: NDArray[np.float64]
    rate: NDArray[np.float64]
    acceleration: NDArray[np.float64]


class NativeClosureTrajectoryLinearization(NamedTuple):
    """Detached partial derivatives of all weld levels at one native state.

    Rows concatenate pose, rate, and acceleration residuals in that order.
    Position and rate derivatives use explicitly configured centered differences;
    acceleration's partial with respect to supplied acceleration is exact.
    """

    names: tuple[str, ...]
    residual: NDArray[np.float64]
    dq: NDArray[np.float64]
    dv: NDArray[np.float64]
    da: NDArray[np.float64]


def depth_first_joints(
    joints: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Keep each subtree contiguous for Pinocchio's compact joint indexing.

    Preserve native identities while rejecting duplicate/disconnected bodies.
    A merely topological, breadth-first order is insufficient for this backend.
    """
    children: dict[str, list[Mapping[str, Any]]] = {}
    body_names = {"world"}
    for joint in joints:
        child = joint["child"]
        if child in body_names:
            raise ValueError("Native joint tree has duplicate children or a cycle")
        body_names.add(child)
        children.setdefault(joint["parent"], []).append(joint)
    pending = list(reversed(children.get("world", [])))
    ordered = []
    while pending:
        joint = pending.pop()
        ordered.append(joint)
        pending.extend(reversed(children.get(joint["child"], [])))
    if len(ordered) != len(joints):
        raise ValueError("Native joint tree is disconnected or cyclic")
    return ordered


class NativePinocchioModel:
    """Own per-instance Pinocchio data and preserve native primitive coordinates."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        # The optional C++ runtime exposes APIs absent from local partial stubs.
        # Keep that dynamic boundary explicit; public arrays/maps remain typed.
        pin: Any = import_module("pinocchio")

        if specification.get("schema_version") != 1:
            raise ValueError("Unsupported native geometry schema")
        self._pin = pin
        self._build_tree(specification)
        self._initialize_closure(specification["closure"])

    def _build_tree(self, specification: Mapping[str, Any]) -> None:
        pin = self._pin
        self.model = pin.Model()
        gravity = self.model.gravity
        gravity.linear[:] = specification["gravity_m_s2"]
        self._coordinates: dict[str, int] = {}
        self._velocity_indices: dict[str, int] = {}
        self._bodies: dict[str, tuple[int, Any]] = {"world": (0, pin.SE3.Identity())}
        coordinate_inventory: set[str] = set()
        for joint_spec in depth_first_joints(specification["joints"]):
            parent, parent_pose = self._bodies[joint_spec["parent"]]
            placement = parent_pose * self._transform(joint_spec["parent_to_base"])
            parent, names = self._add_joint_primitives(parent, placement, joint_spec)
            if coordinate_inventory.intersection(names):
                raise ValueError("Duplicate native coordinate")
            coordinate_inventory.update(names)
            child = joint_spec["child"]
            if child in self._bodies:
                raise ValueError("Native body has multiple tree parents")
            self._bodies[child] = (
                parent,
                self._transform(joint_spec["child_to_follower"]).inverse(),
            )
        if coordinate_inventory != set(specification["coordinate_order"]):
            raise ValueError("Native coordinate inventory was not preserved")
        for body in specification["bodies"]:
            joint, body_pose = self._bodies[body["name"]]
            for solid in body["solids"]:
                inertia = pin.Inertia(
                    solid["mass_kg"],
                    np.asarray(solid["com_m"]),
                    np.asarray(solid["inertia_com_kg_m2"]),
                )
                self.model.appendBodyToJoint(
                    joint, inertia, body_pose * self._transform(solid["placement"])
                )
        self._frames: dict[str, int] = {}
        for frame in specification["frames"]:
            joint, body_pose = self._bodies[frame["body"]]
            placement = body_pose * self._transform(frame["placement"])
            if frame["name"] in self._frames:
                raise ValueError("Duplicate native marker-reference frame")
            self._frames[frame["name"]] = self.model.addFrame(
                pin.Frame(frame["name"], joint, placement, pin.FrameType.OP_FRAME)
            )

    def _add_joint_primitives(
        self, parent: int, placement: Any, joint_spec: Mapping[str, Any]
    ) -> tuple[int, set[str]]:
        """Construct scalar primitives; variants reuse bodies, frames and weld."""
        pin = self._pin
        factories = {
            "Px": pin.JointModelPX,
            "Py": pin.JointModelPY,
            "Pz": pin.JointModelPZ,
            "Rx": pin.JointModelRX,
            "Ry": pin.JointModelRY,
            "Rz": pin.JointModelRZ,
        }
        names = set()
        for primitive in joint_spec["primitives"]:
            name = primitive["coordinate"]
            if name in self._coordinates or primitive["primitive"] not in factories:
                raise ValueError("Duplicate or unsupported native coordinate")
            parent = self.model.addJoint(
                parent, factories[primitive["primitive"]](), placement, name
            )
            self._coordinates[name] = self.model.joints[parent].idx_q
            self._velocity_indices[name] = self.model.joints[parent].idx_v
            placement = pin.SE3.Identity()
            names.add(name)
        return parent, names

    def _initialize_closure(self, closure: Mapping[str, Any]) -> None:
        """Attach the native weld using this model's body-frame placements."""
        pin = self._pin
        joint_a, pose_a = self._bodies[closure["body_a"]]
        joint_b, pose_b = self._bodies[closure["body_b"]]
        constraint = pin.RigidConstraintModel(
            pin.ContactType.CONTACT_6D,
            self.model,
            joint_a,
            pose_a * self._transform(closure["placement_a"]),
            joint_b,
            pose_b * self._transform(closure["placement_b"]),
            pin.ReferenceFrame.LOCAL,
        )
        self.constraints = [constraint]
        self.constraint_data = [constraint.createData()]
        self.data = self.model.createData()
        try:
            pin.initConstraintDynamics(
                self.model, self.data, self.constraints, self.constraint_data
            )
        except (TypeError, Exception):
            if hasattr(pin, "StdVec_RigidConstraintModel"):
                c_vec = pin.StdVec_RigidConstraintModel()
                c_vec.append(constraint)
                pin.initConstraintDynamics(self.model, self.data, c_vec)
            else:
                pin.initConstraintDynamics(self.model, self.data, self.constraints)

    def _transform(self, value: Any) -> Any:
        matrix = np.asarray(value, dtype=float)
        if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
            raise ValueError("Invalid native frame transform")
        return self._pin.SE3(matrix[:3, :3], matrix[:3, 3])

    def configuration(self, coordinates: Mapping[str, float]) -> NDArray[np.float64]:
        if set(coordinates) != set(self._coordinates):
            raise ValueError("Provide exactly the native coordinate inventory")
        q = self._pin.neutral(self.model)
        for name, index in self._coordinates.items():
            q[index] = coordinates[name]
        if not np.all(np.isfinite(q)):
            raise ValueError("Native coordinates must be finite")
        return q

    def frame_poses(
        self, coordinates: Mapping[str, float]
    ) -> dict[str, NDArray[np.float64]]:
        q = self.configuration(coordinates)
        self._pin.forwardKinematics(self.model, self.data, q)
        self._pin.updateFramePlacements(self.model, self.data)
        return {
            name: self.data.oMf[index].homogeneous.copy()
            for name, index in self._frames.items()
        }

    def closure_errors(self) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Copy weld pose/rate residuals from the most recent acceleration call.

        Call accelerations at the desired state first; frame_poses alone does
        not refresh constrained dynamics data. Missing/nonfinite data fails.
        """
        if len(self.constraint_data) != 1:
            raise ValueError("Expected exactly one native weld closure")
        contact = self.constraint_data[0]
        pose = np.array(contact.contact_placement_error.vector, dtype=float, copy=True)
        velocity = np.array(
            contact.contact_velocity_error.vector, dtype=float, copy=True
        )
        if (
            pose.shape != (6,)
            or velocity.shape != (6,)
            or not np.isfinite(np.concatenate((pose, velocity))).all()
        ):
            raise ValueError("Invalid native closure residuals")
        return pose, velocity

    def closure_reaction_wrench(
        self, frame: str = "world"
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Return (force_N, torque_Nm, point_m) for the active weld constraint.

        Extracts the 6D spatial contact wrench from constraint_data[0].contact_force.
        If frame == "world", transports the contact force from the local weld constraint
        frame to the world frame using the weld's world placement.
        """
        if len(self.constraint_data) != 1:
            raise ValueError("Expected exactly one native weld closure")
        contact = self.constraint_data[0]
        contact_force = getattr(contact, "contact_force", None)
        if contact_force is None:
            raise ValueError("Constraint data does not contain contact_force")
        constraint = self.constraints[0]
        joint_id = getattr(
            constraint, "joint1_id", getattr(constraint, "joint1Id", None)
        )
        joint_placement = getattr(
            constraint, "joint1_placement", getattr(constraint, "joint1Placement", None)
        )

        if frame == "world" and joint_id is not None and joint_placement is not None:
            weld_world_pose = self.data.oMi[joint_id] * joint_placement
            if hasattr(contact_force, "linear") and hasattr(contact_force, "angular"):
                world_force = weld_world_pose.act(contact_force)
                f = np.asarray(world_force.linear, dtype=float).copy()
                tau = np.asarray(world_force.angular, dtype=float).copy()
            else:
                raw_vec = np.asarray(contact_force, dtype=float).reshape(-1)
                rot = weld_world_pose.rotation
                f = (rot @ raw_vec[:3]).copy()
                tau = (rot @ raw_vec[3:]).copy()
            point = np.asarray(weld_world_pose.translation, dtype=float).copy()
        else:
            if hasattr(contact_force, "linear") and hasattr(contact_force, "angular"):
                f = np.asarray(contact_force.linear, dtype=float).copy()
                tau = np.asarray(contact_force.angular, dtype=float).copy()
            else:
                raw_vec = np.asarray(contact_force, dtype=float).reshape(-1)
                f = raw_vec[:3].copy()
                tau = raw_vec[3:].copy()
            point = np.zeros(3)

        f.flags.writeable = False
        tau.flags.writeable = False
        point.flags.writeable = False
        return f, tau, point

    def closure_residuals(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float] | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Refresh and return weld pose/rate residuals at one native state.

        The constrained-dynamics backend owns the constraint-data refresh, so
        this intentionally invokes it with explicit zero primitive efforts.
        It is a diagnostic/kinematic-trajectory oracle, never an inverse
        dynamics substitute and never a state correction.  Omitted rates mean
        an explicit zero-rate static probe; supplied rates retain their stated
        values to support trajectory checks.
        """
        names = set(self._velocity_indices)
        position = dict(coordinates)
        velocity = {name: 0.0 for name in names} if rates is None else dict(rates)
        if (
            set(position) != names
            or set(velocity) != names
            or not np.isfinite(tuple(position.values())).all()
            or not np.isfinite(tuple(velocity.values())).all()
        ):
            raise ValueError("Provide exactly native coordinate and rate inventories")
        efforts = {name: 0.0 for name in names}
        self.accelerations(position, velocity, efforts)
        return self.closure_errors()

    def _refresh_constraint_data(self) -> None:
        """Refresh constraint placement data from current joint placements."""
        for cm, cd in zip(self.constraints, self.constraint_data, strict=True):
            if hasattr(cm, "calc"):
                cm.calc(self.model, self.data, cd)
            else:
                oMc1 = self.data.oMi[cm.joint1_id] * cm.joint1_placement
                oMc2 = self.data.oMi[cm.joint2_id] * cm.joint2_placement
                cd.c1Mc2 = oMc1.inverse() * oMc2
                cd.oMc1 = oMc1
                if hasattr(cd, "oMc2"):
                    cd.oMc2 = oMc2

    def _constraints_jacobian(
        self, names: tuple[str, ...]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raw = np.asarray(
            self._pin.getConstraintsJacobian(
                self.model, self.data, self.constraints, self.constraint_data
            ),
            dtype=float,
        )
        indices = [self._velocity_indices[name] for name in names]
        return raw, np.asarray(raw[:, indices], dtype=float)

    def closure_position_linearization(
        self, coordinates: Mapping[str, float]
    ) -> NativeClosurePositionLinearization:
        """Refresh and return the weld pose residual and its native q Jacobian.

        The constraint backend refreshes its data through the zero-rate,
        zero-effort diagnostic probe. The returned Jacobian is kinematic only;
        it is suitable for a local node chart, never for inverse dynamics.
        Differentiate the finite residual ``-log6(c1Mc2)`` with the left
        log Jacobian; the constraint velocity Jacobian alone is valid only
        at zero pose error. The result owns finite, read-only storage.
        Rotations within 1e-7 rad of the principal-log branch cut are rejected.
        """
        names = tuple(coordinates)
        position, _ = self.closure_residuals(coordinates)
        _require_weld_log_chart(position)
        raw, jac_slice = self._constraints_jacobian(names)
        contact = self.constraint_data[0]
        inverse_placement = contact.c1Mc2.inverse()
        jacobian = np.asarray(self._pin.Jlog6(inverse_placement) @ jac_slice)
        if (
            position.shape != (6,)
            or raw.shape != (6, self.model.nv)
            or jacobian.shape != (6, len(names))
            or not np.isfinite(position).all()
            or not np.isfinite(jacobian).all()
        ):
            raise ValueError("Invalid native weld position linearization")
        position.setflags(write=False)
        jacobian.setflags(write=False)
        return NativeClosurePositionLinearization(names, position, jacobian)

    def closure_force_jacobian(
        self, coordinates: Mapping[str, float]
    ) -> NativeClosureForceJacobian:
        """Refresh the native velocity constraint without a dynamics solve.

        Preconditions: exactly the finite native coordinate inventory.
        Postconditions: owned finite read-only 6-by-n matrix in caller order.
        Use its transpose for inverse-dynamics grip-force allocation. Position
        fitting must continue to use closure_position_linearization instead.
        """
        q = self.configuration(coordinates)
        names = tuple(coordinates)
        self._pin.computeJointJacobians(self.model, self.data, q)
        self._refresh_constraint_data()
        _, jacobian = self._constraints_jacobian(names)
        jacobian = np.array(jacobian, dtype=float, copy=True)
        if jacobian.shape != (6, len(names)) or not np.isfinite(jacobian).all():
            raise ValueError("Invalid native weld force Jacobian")
        jacobian.setflags(write=False)
        return NativeClosureForceJacobian(names, jacobian)

    def closure_trajectory_residuals(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
    ) -> NativeClosureTrajectoryResiduals:
        """Evaluate all weld levels without treating the loop as inverse dynamics.

        The zero-effort constrained forward acceleration supplies `a0`, so the
        acceleration residual is J*(a-a0), equivalent to J*a+gamma when
        J*a0+gamma=0. This preserves the loop reaction rather than assuming it
        vanishes. It is a kinematic/dynamic residual oracle, never a fitter.
        """
        names = tuple(coordinates)
        if (
            set(rates) != set(names)
            or set(accelerations) != set(names)
            or not np.isfinite(tuple(rates.values())).all()
            or not np.isfinite(tuple(accelerations.values())).all()
        ):
            raise ValueError(
                "Provide finite native coordinate, rate, and acceleration inventories"
            )
        zero_efforts = {name: 0.0 for name in names}
        drift_values = self.accelerations(coordinates, rates, zero_efforts)
        position, _ = self.closure_errors()
        raw, jacobian = self._constraints_jacobian(names)
        rate_vector = np.asarray([rates[name] for name in names], dtype=float)
        acceleration_vector = np.asarray(
            [accelerations[name] for name in names], dtype=float
        )
        drift = np.asarray([drift_values[name] for name in names], dtype=float)
        rate = jacobian @ rate_vector
        acceleration = jacobian @ (acceleration_vector - drift)
        if (
            raw.ndim != 2
            or jacobian.shape != (position.size, len(names))
            or not np.isfinite(position).all()
            or not np.isfinite(rate).all()
            or not np.isfinite(acceleration).all()
        ):
            raise ValueError("Invalid native weld trajectory residuals")
        for value in (position, rate, acceleration):
            value.setflags(write=False)
        return NativeClosureTrajectoryResiduals(position, rate, acceleration)

    def closure_trajectory_linearization(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        accelerations: Mapping[str, float],
        *,
        finite_difference_step: float,
    ) -> NativeClosureTrajectoryLinearization:
        """Differentiate the weld oracle at one state without inverse dynamics.

        The finite differences are local to one 27-coordinate node, which
        avoids treating a full spline trajectory as a black-box function. The
        returned acceleration partial is the exact constraint Jacobian because
        the oracle defines that residual as ``J * (a - a0)``. The final base
        probe restores the backend data to the reported state.
        The pose must stay away from the log branch by at least twice the
        scalar-coordinate difference step (and the fixed numerical margin).
        """
        names = tuple(coordinates)
        if (
            not np.isfinite(finite_difference_step)
            or finite_difference_step <= 0.0
            or set(rates) != set(names)
            or set(accelerations) != set(names)
        ):
            raise ValueError(
                "Expected finite positive step and matching state inventories"
            )

        def flatten(value: NativeClosureTrajectoryResiduals) -> NDArray[np.float64]:
            return np.concatenate((value.position, value.rate, value.acceleration))

        position = dict(coordinates)
        velocity = dict(rates)
        acceleration = dict(accelerations)
        base = flatten(
            self.closure_trajectory_residuals(position, velocity, acceleration)
        )
        _require_weld_log_chart(
            base[:6], max(_WELD_LOG_BRANCH_MARGIN_RAD, 2.0 * finite_difference_step)
        )
        derivative_shape = (base.size, len(names))
        dq = np.empty(derivative_shape, dtype=float)
        dv = np.empty(derivative_shape, dtype=float)
        for index, name in enumerate(names):
            plus_position = dict(position)
            minus_position = dict(position)
            plus_position[name] += finite_difference_step
            minus_position[name] -= finite_difference_step
            dq[:, index] = (
                flatten(
                    self.closure_trajectory_residuals(
                        plus_position, velocity, acceleration
                    )
                )
                - flatten(
                    self.closure_trajectory_residuals(
                        minus_position, velocity, acceleration
                    )
                )
            ) / (2.0 * finite_difference_step)
            plus_velocity = dict(velocity)
            minus_velocity = dict(velocity)
            plus_velocity[name] += finite_difference_step
            minus_velocity[name] -= finite_difference_step
            dv[:, index] = (
                flatten(
                    self.closure_trajectory_residuals(
                        position, plus_velocity, acceleration
                    )
                )
                - flatten(
                    self.closure_trajectory_residuals(
                        position, minus_velocity, acceleration
                    )
                )
            ) / (2.0 * finite_difference_step)
        self.closure_trajectory_residuals(position, velocity, acceleration)
        raw, velocity_jacobian = self._constraints_jacobian(names)
        da = np.zeros(derivative_shape, dtype=float)
        da[-velocity_jacobian.shape[0] :, :] = velocity_jacobian
        if (
            raw.shape != (6, self.model.nv)
            or velocity_jacobian.shape != (6, len(names))
            or not np.isfinite(base).all()
            or not np.isfinite(dq).all()
            or not np.isfinite(dv).all()
            or not np.isfinite(da).all()
        ):
            raise ValueError("Invalid native weld trajectory linearization")
        base.setflags(write=False)
        dq.setflags(write=False)
        dv.setflags(write=False)
        da.setflags(write=False)
        return NativeClosureTrajectoryLinearization(names, base, dq, dv, da)

    def marker_derivatives(
        self,
        coordinates: Mapping[str, float],
        bodies: Sequence[str],
        offsets: ArrayLike,
    ) -> NativeMarkerDerivatives:
        """Differentiate fixed markers in native scalar-coordinate order.

        These are partial derivatives on the tree configuration. The trajectory
        sensitivity must enforce the weld; no independent marker projection or
        closure correction is applied here. Pinocchio's aligned frame Jacobian
        orders linear rows before angular rows and acts at the frame origin.
        """
        frames = self.frame_poses(coordinates)
        positions = project_markers(frames, bodies, offsets)
        local = np.asarray(offsets, dtype=float)
        names = tuple(coordinates)
        columns = [self._velocity_indices[name] for name in names]
        self._pin.computeJointJacobians(
            self.model, self.data, self.configuration(coordinates)
        )
        derivatives = []
        for body, offset in zip(bodies, local, strict=True):
            pin = self._pin
            raw = np.asarray(
                pin.getFrameJacobian(
                    self.model,
                    self.data,
                    self._frames[body],
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
                ),
                dtype=float,
            )
            if raw.shape != (6, self.model.nv) or not np.isfinite(raw).all():
                raise ValueError("Invalid native frame Jacobian")
            lever = frames[body][:3, :3] @ offset
            point = raw[:3] + np.cross(raw[3:].T, lever).T
            derivatives.append(point[:, columns])
        jacobian = np.asarray(derivatives)
        if not np.isfinite(jacobian).all() or not np.isfinite(positions).all():
            raise ValueError("Native marker linearization is nonfinite")
        positions.setflags(write=False)
        jacobian.setflags(write=False)
        return NativeMarkerDerivatives(names, positions, jacobian)

    def _velocity_vector(self, values: Mapping[str, float]) -> NDArray[np.float64]:
        if set(values) != set(self._velocity_indices):
            raise ValueError("Provide exactly the native primitive inventory")
        vector = np.zeros(self.model.nv)
        for name, index in self._velocity_indices.items():
            vector[index] = values[name]
        if not np.all(np.isfinite(vector)):
            raise ValueError("Native rates and primitive efforts must be finite")
        return vector

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Constrained forward acceleration from primitive forces/torques.

        These are joint-conjugate efforts, not yet mapped from the Simscape
        model's upstream polynomial input signals. No feedback is added.
        """
        q = self.configuration(coordinates)
        v = self._velocity_vector(rates)
        tau = self._velocity_vector(primitive_efforts)
        acceleration = self._pin.constraintDynamics(
            self.model, self.data, q, v, tau, self.constraints, self.constraint_data
        )
        return self._acceleration_dict(acceleration)

    def _acceleration_dict(self, acceleration: Any) -> dict[str, float]:
        if not np.all(np.isfinite(acceleration)):
            raise FloatingPointError(
                "Native constrained dynamics produced nonfinite acceleration"
            )
        return {
            name: float(acceleration[index])
            for name, index in self._velocity_indices.items()
        }

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> NativeAccelerationDerivatives:
        """Refresh constrained dynamics, then map Pinocchio derivatives.

        Efforts are primitive-conjugate, before the upstream force-frame map.
        These local derivatives require numerical qualification before use in
        a trajectory sensitivity solver; this method does not integrate them.
        """
        self.accelerations(coordinates, rates, primitive_efforts)
        raw = self._pin.computeConstraintDynamicsDerivatives(
            self.model, self.data, self.constraints, self.constraint_data
        )
        names = tuple(coordinates)
        indices = [self._velocity_indices[name] for name in names]
        matrices = []
        if len(raw) < 3:
            raise ValueError("Native acceleration derivatives are missing")
        for value in raw[:3]:
            matrices.append(self._ordered_derivative(value, indices))
        return NativeAccelerationDerivatives(names, *matrices)

    def _ordered_derivative(
        self, value: Any, indices: list[int]
    ) -> NDArray[np.float64]:
        matrix = np.asarray(value, dtype=float)
        if (
            matrix.shape != (self.model.nv, self.model.nv)
            or not np.isfinite(matrix).all()
        ):
            raise ValueError("Invalid native acceleration derivatives")
        owned = matrix[np.ix_(indices, indices)].copy()
        owned.setflags(write=False)
        return owned


class FullBodyPinocchioModel(NativePinocchioModel):
    """Full-body Pinocchio model with lower limbs and shared ground contact forces."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        pin: Any = import_module("pinocchio")

        if specification.get("schema_version") != FULL_BODY_SCHEMA_VERSION:
            raise ValueError("Unsupported full-body schema")
        self.specification = specification
        self._pin = pin
        self._build_tree(specification)
        self._initialize_contact(specification)
        self._initialize_closure(specification["closure"])
        from .contact_derivatives import PinocchioContactDifferentiator

        frames = {
            self._contact_frames[sphere.name]: sphere.radius_m
            for sphere in self.contact_spheres
        }
        self._contact_differentiator = PinocchioContactDifferentiator(
            pin, self.model, frames
        )

    def _initialize_contact(self, specification: Mapping[str, Any]) -> None:
        pin = self._pin
        contact_spec = specification["contact"]
        self.contact_parameters = ContactParameters(**contact_spec["parameters"])
        g_vec = np.asarray(specification["gravity_m_s2"], dtype=float)
        g_norm = float(
            math.sqrt(np.dot(g_vec, g_vec))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~2.5x faster than np.linalg.norm
        if g_norm <= 0:
            raise ValueError("Gravity must be a nonzero vector")
        unit_g = -g_vec / g_norm
        normal = (float(unit_g[0]), float(unit_g[1]), float(unit_g[2]))
        height = float(contact_spec["ground"]["height_m"] or 0.0)
        self.ground = GroundPlane(normal=normal, height_m=height)
        self.contact_spheres = tuple(
            ContactSphere(
                s["name"], s["body"], tuple(s["position_m"]), float(s["radius_m"])
            )
            for s in contact_spec["spheres"]
        )
        self._contact_frames: dict[str, int] = {}
        for sphere in self.contact_spheres:
            joint, body_pose = self._bodies[sphere.body]
            placement = body_pose * pin.SE3(
                np.eye(3), np.asarray(sphere.position_m, dtype=float)
            )
            self._contact_frames[sphere.name] = self.model.addFrame(
                pin.Frame(sphere.name, joint, placement, pin.FrameType.OP_FRAME)
            )

    def upper_body_model(self) -> NativePinocchioModel:
        """Construct the qualified upper-body model from this spec's upper-body slice."""
        return NativePinocchioModel(upper_body_slice(self.specification))

    def mass_matrix(self, coordinates: Mapping[str, float]) -> NDArray[np.float64]:
        """Compute the joint-space mass matrix at the given configuration."""
        q = self.configuration(coordinates)
        matrix = np.asarray(
            self._pin.crba(self.model, self.data, q), dtype=float
        ).copy()
        matrix.setflags(write=False)
        return matrix

    def contact_forces(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> dict[str, ContactSample]:
        """Evaluate shared contact law forces for each foot contact sphere."""
        q = self.configuration(coordinates)
        v = self._velocity_vector(rates)
        pin = self._pin
        pin.forwardKinematics(self.model, self.data, q, v)
        pin.updateFramePlacements(self.model, self.data)
        ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        samples = {}
        for sphere in self.contact_spheres:
            fid = self._contact_frames[sphere.name]
            center = self.data.oMf[fid].translation.copy()
            frame_vel = pin.getFrameVelocity(
                self.model,
                self.data,
                fid,
                ref_frame,
            )
            velocity = frame_vel.linear.copy()
            samples[sphere.name] = sphere_ground_contact(
                center, velocity, sphere.radius_m, self.ground, self.contact_parameters
            )
        return samples

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        """Return the canonical 41-coordinate order of the full-body model."""
        return tuple(self.specification["coordinate_order"])

    @property
    def ground_plane(self) -> GroundPlane:
        """Ground plane alias for simulation interface consistency."""
        return self.ground

    @ground_plane.setter
    def ground_plane(self, value: GroundPlane) -> None:
        self.ground = value

    def evaluate_contact_samples(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> dict[str, ContactSample]:
        """Evaluate shared contact law forces for each foot contact sphere."""
        return self.contact_forces(coordinates, rates)

    def contact_effort_derivatives(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> NativeContactEffortDerivatives:
        """Return contact partials without disturbing the constrained solver.

        Matrices are owned, read-only, and ordered by ``coordinates``.
        ``differentiable`` is false at activation/clipping boundaries; the
        shared law's documented inactive branch is selected at those kinks.
        """
        result = self._contact_differentiator.evaluate(
            self.configuration(coordinates),
            self._velocity_vector(rates),
            self.ground,
            self.contact_parameters,
        )
        names = tuple(coordinates)
        indices = [self._velocity_indices[name] for name in names]
        return NativeContactEffortDerivatives(
            names,
            self._ordered_derivative(result.dq, indices),
            self._ordered_derivative(result.dv, indices),
            result.differentiable,
        )

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> NativeAccelerationDerivatives:
        """Differentiate the actual contact-aware constrained acceleration.

        Include both Jacobian and force state dependence in generalized
        contact effort. At contact kinks these are selected branch partials,
        not classical derivatives; query ``contact_effort_derivatives`` for
        branch validity. The return layout preserves the native API.
        """
        base = super().acceleration_derivatives(coordinates, rates, primitive_efforts)
        contact = self.contact_effort_derivatives(coordinates, rates)
        dq = base.dq + base.deffort @ contact.dq
        dv = base.dv + base.deffort @ contact.dv
        for matrix in (dq, dv):
            if not np.isfinite(matrix).all():
                raise ValueError("Full-body acceleration derivatives must be finite")
            matrix.setflags(write=False)
        return NativeAccelerationDerivatives(base.names, dq, dv, base.deffort)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Forward accelerations combining joint efforts, contact forces, and weld closure."""
        q = self.configuration(coordinates)
        v = self._velocity_vector(rates)
        tau = self._velocity_vector(primitive_efforts)
        samples = self.contact_forces(coordinates, rates)
        self._pin.computeJointJacobians(self.model, self.data, q)
        tau_contact = np.zeros(self.model.nv, dtype=float)
        for sphere in self.contact_spheres:
            sample = samples[sphere.name]
            if sample.penetration_m > 0.0:
                f_world = sample.normal_force_n + sample.friction_force_n
                fid = self._contact_frames[sphere.name]
                pin = self._pin
                ref_frame = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                j_raw = np.asarray(
                    pin.getFrameJacobian(
                        self.model,
                        self.data,
                        fid,
                        ref_frame,
                    ),
                    dtype=float,
                )
                tau_contact += j_raw[:3, :].T @ f_world

        acceleration = self._pin.constraintDynamics(
            self.model,
            self.data,
            q,
            v,
            tau + tau_contact,
            self.constraints,
            self.constraint_data,
        )
        return self._acceleration_dict(acceleration)


def build_full_body_pinocchio_model(
    specification: Mapping[str, Any],
) -> FullBodyPinocchioModel:
    """Construct a FullBodyPinocchioModel from a full-body specification."""
    return FullBodyPinocchioModel(specification)
