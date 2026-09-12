"""Native-derived Pinocchio tree with the original right-hand weld constraint.

The input is a portable native_spec geometry export. Actuator routing, damping,
limits and time integration require separate qualification before swing fitting.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import NDArray


class NativePinocchioModel:
    """Own per-instance Pinocchio data and preserve native primitive coordinates."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        import pinocchio as pin

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
        for joint in specification["joints"]:
            parent, parent_pose = self._bodies[joint["parent"]]
            placement = parent_pose * self._transform(joint["parent_to_base"])
            for primitive in joint["primitives"]:
                name = primitive["coordinate"]
                if name in self._coordinates or primitive["primitive"] not in factories:
                    raise ValueError("Duplicate or unsupported native coordinate")
                parent = self.model.addJoint(
                    parent, factories[primitive["primitive"]](), placement, name
                )
                self._coordinates[name] = self.model.joints[parent].idx_q
                self._velocity_indices[name] = self.model.joints[parent].idx_v
                placement = pin.SE3.Identity()
            child = joint["child"]
            if child in self._bodies:
                raise ValueError("Native body has multiple tree parents")
            self._bodies[child] = (
                parent,
                self._transform(joint["child_to_follower"]).inverse(),
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
        self._frames = {}
        for frame in specification["frames"]:
            joint, body_pose = self._bodies[frame["body"]]
            placement = body_pose * self._transform(frame["placement"])
            if frame["name"] in self._frames:
                raise ValueError("Duplicate native marker-reference frame")
            self._frames[frame["name"]] = self.model.addFrame(
                pin.Frame(frame["name"], joint, placement, pin.FrameType.OP_FRAME)
            )
        closure = specification["closure"]
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
