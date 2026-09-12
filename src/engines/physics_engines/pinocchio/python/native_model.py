"""Native-derived Pinocchio tree with the original right-hand weld constraint.

The input is a portable native_spec geometry export. Actuator routing, damping,
limits and time integration require separate qualification before swing fitting.
"""

from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.shared.python.motion_matching.marker_projection import project_markers


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


class NativeClosurePositionLinearization(NamedTuple):
    """Weld pose residual and Jacobian in an explicit native coordinate order."""

    names: tuple[str, ...]
    position: NDArray[np.float64]
    jacobian: NDArray[np.float64]


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
        self.model = pin.Model()
        self.model.gravity.linear[:] = specification["gravity_m_s2"]
        self._coordinates: dict[str, int] = {}
        self._velocity_indices: dict[str, int] = {}
        self._bodies: dict[str, tuple[int, Any]] = {"world": (0, pin.SE3.Identity())}
        factories = {
            "Px": pin.JointModelPX,
            "Py": pin.JointModelPY,
            "Pz": pin.JointModelPZ,
            "Rx": pin.JointModelRX,
            "Ry": pin.JointModelRY,
            "Rz": pin.JointModelRZ,
        }
        for joint_spec in depth_first_joints(specification["joints"]):
            parent, parent_pose = self._bodies[joint_spec["parent"]]
            placement = parent_pose * self._transform(joint_spec["parent_to_base"])
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
            child = joint_spec["child"]
            if child in self._bodies:
                raise ValueError("Native body has multiple tree parents")
            self._bodies[child] = (
                parent,
                self._transform(joint_spec["child_to_follower"]).inverse(),
            )
        if set(self._coordinates) != set(specification["coordinate_order"]):
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
        self._initialize_closure(specification["closure"])

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
        pin.initConstraintDynamics(
            self.model, self.data, self.constraints, self.constraint_data
        )

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

    def closure_position_linearization(
        self, coordinates: Mapping[str, float]
    ) -> NativeClosurePositionLinearization:
        """Refresh and return the weld pose residual and its native q Jacobian.

        The constraint backend refreshes its data through the zero-rate,
        zero-effort diagnostic probe. The returned Jacobian is kinematic only;
        it is suitable for a local node chart, never for inverse dynamics.
        """
        names = tuple(coordinates)
        position, _ = self.closure_residuals(coordinates)
        raw = np.asarray(
            self._pin.getConstraintsJacobian(
                self.model, self.data, self.constraints, self.constraint_data
            ),
            dtype=float,
        )
        indices = [self._velocity_indices[name] for name in names]
        jacobian = raw[:, indices].copy()
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
            raw = np.asarray(
                self._pin.getFrameJacobian(
                    self.model,
                    self.data,
                    self._frames[body],
                    self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
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
            matrix = np.asarray(value, dtype=float)
            if (
                matrix.shape != (self.model.nv, self.model.nv)
                or not np.isfinite(matrix).all()
            ):
                raise ValueError("Invalid native acceleration derivatives")
            owned = matrix[np.ix_(indices, indices)].copy()
            owned.setflags(write=False)
            matrices.append(owned)
        return NativeAccelerationDerivatives(names, *matrices)
