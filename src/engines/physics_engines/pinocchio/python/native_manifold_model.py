"""Explicit spherical-coordinate variant of the native golf mechanism.

Only massless, unattached rotational primitives inside one native joint collapse.
The scalar reference builder owns all solids, placements, frames and the weld.
Pinocchio spherical q is xyzw and its angular tangent is follower/body expressed.
Native actuator polynomials must be evaluated in native coordinates then mapped
at the current configuration; this mapping adds no tracking controller.
"""

from collections.abc import Mapping
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
    NativeManifoldState,
    RotationState,
)
from src.shared.python.pose_interchange.se3 import quat_to_matrix

Array = NDArray[np.float64]


class _ManifoldTree(NativePinocchioModel):
    """Construction-only subclass; scalar replay APIs are never exposed."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        self.adapter = NativeJointStateAdapter(specification)
        self.groups = {group.name: group for group in self.adapter.groups}
        self.spherical: dict[str, tuple[int, int]] = {}
        super().__init__(specification)

    def _add_joint_primitives(
        self, parent: int, placement: Any, joint_spec: Mapping[str, Any]
    ) -> tuple[int, set[str]]:
        group = self.groups.get(joint_spec["name"])
        if group is None:
            return super()._add_joint_primitives(parent, placement, joint_spec)
        prefix = dict(joint_spec, primitives=joint_spec["primitives"][:-3])
        parent, names = super()._add_joint_primitives(parent, placement, prefix)
        if prefix["primitives"]:
            placement = self._pin.SE3.Identity()
        parent = self.model.addJoint(
            parent, self._pin.JointModelSpherical(), placement, group.name
        )
        joint = self.model.joints[parent]
        self.spherical[group.name] = (joint.idx_q, joint.idx_v)
        return parent, names | set(group.coordinates)


class NativeManifoldPinocchioModel:
    """Native physical inventory with explicit nq/nv and body-tangent APIs.

    This is an experimental representation, not accepted Simscape parity.
    Scalar-only derivative/replay methods are deliberately not inherited.
    Native inverse maps reject singular charts instead of adding pseudoinverses.
    """

    representation = "native-pinocchio-spherical-xyzw-body-v1"

    def __init__(self, specification: Mapping[str, Any]) -> None:
        self._tree = _ManifoldTree(specification)
        self.adapter = self._tree.adapter
        self.pin = self._tree._pin
        self.model = self._tree.model
        self.data = self._tree.data
        self.constraints = self._tree.constraints
        self.constraint_data = self._tree.constraint_data
        self.specification_sha256 = self.adapter.specification_sha256

    def _configuration(self, value: ArrayLike) -> Array:
        q = np.asarray(value, dtype=float)
        if q.shape != (self.model.nq,) or not np.isfinite(q).all():
            raise ValueError("Invalid manifold configuration")
        for index, _ in self._tree.spherical.values():
            if abs(np.linalg.norm(q[index : index + 4]) - 1.0) > 1e-10:
                raise ValueError("Manifold quaternion must be unit length")
        return q

    def _tangent(self, value: ArrayLike) -> Array:
        v = np.asarray(value, dtype=float)
        if v.shape != (self.model.nv,) or not np.isfinite(v).all():
            raise ValueError("Invalid manifold tangent")
        return v

    def native_state(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> tuple[Array, Array, Array]:
        """Map named SI native state to q, body-tangent v and conjugate effort."""
        state = self.adapter.export(
            coordinates, rates, dict.fromkeys(coordinates, 0.0), primitive_efforts
        )
        q = self.pin.neutral(self.model)
        v, tau = np.zeros(self.model.nv), np.zeros(self.model.nv)
        for name, values in state.scalars.items():
            q[self._tree._coordinates[name]] = values[0]
            index = self._tree._velocity_indices[name]
            v[index], tau[index] = values[1], values[3]
        for name, rotation in state.rotations.items():
            qi, vi = self._tree.spherical[name]
            quaternion = np.asarray(rotation.quaternion_wxyz)
            q[qi : qi + 4] = quaternion[[1, 2, 3, 0]]
            inverse_rotation = quat_to_matrix(quaternion).T
            v[vi : vi + 3] = inverse_rotation @ rotation.omega_parent_rad_s
            tau[vi : vi + 3] = inverse_rotation @ rotation.moment_parent_nm
        return q, v, tau

    def _native_state(
        self, q: Array, v: Array, acceleration: Array
    ) -> NativeManifoldState:
        rotations = {}
        for name, (qi, vi) in self._tree.spherical.items():
            quaternion = q[qi : qi + 4][[3, 0, 1, 2]]
            rotation = quat_to_matrix(quaternion)
            rotations[name] = RotationState(
                tuple(quaternion),
                tuple(rotation @ v[vi : vi + 3]),
                tuple(rotation @ acceleration[vi : vi + 3]),
                (0.0, 0.0, 0.0),
            )
        scalars = {
            name: (
                float(q[qi]),
                float(v[self._tree._velocity_indices[name]]),
                float(acceleration[self._tree._velocity_indices[name]]),
                0.0,
            )
            for name, qi in self._tree._coordinates.items()
        }
        return NativeManifoldState(self.specification_sha256, rotations, scalars)

    def native_coordinates(
        self,
        configuration: ArrayLike,
        velocity: ArrayLike,
        reference_coordinates: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Restore native rates on the reference middle-angle gimbal branch.

        Outer-axis winding is nearest the reference. Choosing a different middle
        branch solely for angle proximity changes the native actuator map.
        """
        q, v = self._configuration(configuration), self._tangent(velocity)
        native_q, native_v, _, _ = self.adapter.restore(
            self._native_state(q, v, np.zeros(self.model.nv)),
            reference_coordinates,
            preserve_middle_branch=True,
        )
        return native_q, native_v

    def integrate(self, configuration: ArrayLike, increment: ArrayLike) -> Array:
        """Retract a body-tangent increment with Pinocchio's manifold operation."""
        return np.asarray(
            self.pin.integrate(
                self.model, self._configuration(configuration), self._tangent(increment)
            ),
            dtype=float,
        ).copy()

    def difference(self, first: ArrayLike, second: ArrayLike) -> Array:
        """Return the local body-tangent displacement, including quaternion sign."""
        return np.asarray(
            self.pin.difference(
                self.model, self._configuration(first), self._configuration(second)
            ),
            dtype=float,
        ).copy()

    def difference_rate(
        self, anchor: ArrayLike, configuration: ArrayLike, velocity: ArrayLike
    ) -> Array:
        """Derivative of difference(anchor,q) for q's body-tangent velocity.

        The anchor is held fixed. This is the inverse local retraction Jacobian,
        not an identity map except in Euclidean coordinates or at the anchor.
        """
        jacobian = np.asarray(
            self.pin.dDifference(
                self.model,
                self._configuration(anchor),
                self._configuration(configuration),
                self.pin.ARG1,
            ),
            dtype=float,
        )
        if jacobian.shape != (self.model.nv, self.model.nv):
            raise ValueError("Invalid manifold difference Jacobian")
        return self._tangent(jacobian @ self._tangent(velocity)).copy()

    def closure_errors(self) -> tuple[Array, Array]:
        """Detached weld pose/rate errors from the most recent dynamics call."""
        return self._tree.closure_errors()

    def frame_poses(self, configuration: ArrayLike) -> dict[str, Array]:
        """Detached world transforms of every native marker-reference frame."""
        self.pin.forwardKinematics(
            self.model, self.data, self._configuration(configuration)
        )
        self.pin.updateFramePlacements(self.model, self.data)
        return {
            name: self.data.oMf[index].homogeneous.copy()
            for name, index in self._tree._frames.items()
        }

    def frame_velocities(
        self, configuration: ArrayLike, velocity: ArrayLike
    ) -> dict[str, Array]:
        """World-aligned, frame-origin twists in Pinocchio linear/angular order."""
        self.pin.forwardKinematics(
            self.model,
            self.data,
            self._configuration(configuration),
            self._tangent(velocity),
        )
        return {
            name: self.pin.getFrameVelocity(
                self.model,
                self.data,
                index,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ).vector.copy()
            for name, index in self._tree._frames.items()
        }

    def acceleration(
        self, configuration: ArrayLike, velocity: ArrayLike, effort: ArrayLike
    ) -> Array:
        """Constrained forward body-tangent acceleration with the original weld."""
        result = np.asarray(
            self.pin.constraintDynamics(
                self.model,
                self.data,
                self._configuration(configuration),
                self._tangent(velocity),
                self._tangent(effort),
                self.constraints,
                self.constraint_data,
            ),
            dtype=float,
        ).copy()
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite manifold acceleration")
        return result

    def acceleration_from_native_efforts(
        self,
        configuration: ArrayLike,
        velocity: ArrayLike,
        primitive_efforts: Mapping[str, float],
        reference_coordinates: Mapping[str, float],
    ) -> Array:
        """Apply native actuator efforts at the CURRENT alternate configuration.

        The reference selects winding/branch only; it is not a tracking target.
        Evaluate the native polynomial at time t, then call this at every solver
        evaluation. Never freeze the transformed spherical moments at a seed.
        """
        q, v = self._configuration(configuration), self._tangent(velocity)
        native_q, native_v = self.native_coordinates(q, v, reference_coordinates)
        _, _, effort = self.native_state(native_q, native_v, primitive_efforts)
        return self.acceleration(q, v, effort)

    def native_accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Map forward result to native qdd, including the convective term."""
        q, v, tau = self.native_state(coordinates, rates, primitive_efforts)
        a = self.acceleration(q, v, tau)
        return self.adapter.restore(
            self._native_state(q, v, a), coordinates, preserve_middle_branch=True
        )[2]
