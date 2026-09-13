"""Native URDF Drake tree with explicit continuous rigid-weld dynamics.

This is a custom constrained execution path using Drake mass, bias and exact
kinematic Jacobians, not Drake's discrete SAP constraint simulator. Polynomial
input-to-primitive effort routing is supplied by the shared native replay layer.
"""

from collections.abc import Mapping
from importlib import import_module
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.shared.python.motion_matching.native_urdf_contract import (
    validate_native_urdf_bundle,
)

Array = NDArray[np.float64]


def solve_weld_acceleration(
    mass: Array, force: Array, jacobian: Array, bias: Array
) -> Array:
    """Solve M a - J.T lambda = f, J a + bias = 0 without relaxation."""
    mass, force, jacobian, bias = (
        np.asarray(value, dtype=float) for value in (mass, force, jacobian, bias)
    )
    n = force.size
    if (
        force.shape != (n,)
        or mass.shape != (n, n)
        or jacobian.ndim != 2
        or jacobian.shape[1] != n
        or bias.shape != (jacobian.shape[0],)
        or any(not np.isfinite(value).all() for value in (mass, force, jacobian, bias))
    ):
        raise ValueError("Invalid constrained dynamics dimensions or values")
    k = bias.size
    matrix = np.block([[mass, -jacobian.T], [jacobian, np.zeros((k, k))]])
    result = np.linalg.solve(matrix, np.concatenate((force, -bias)))[:n]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite constrained acceleration")
    return result.copy()


class NativeDrakeModel:
    """Own a Drake plant/context; preserve every native scalar coordinate."""

    def __init__(
        self, urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
    ) -> None:
        meta = validate_native_urdf_bundle(urdf_bytes, sidecar_bytes, model_bytes)
        api: Any = import_module("pydrake.all")
        self._api = api
        self.plant = api.MultibodyPlant(time_step=0.0)
        instance = api.Parser(self.plant).AddModelsFromString(
            urdf_bytes.decode("utf-8"), "urdf"
        )[0]
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                meta["body_links"]["world"], instance
            ).body_frame(),
        )
        self.plant.mutable_gravity_field().set_gravity_vector(meta["gravity_m_s2"])
        self.names = tuple(meta["coordinate_order"])
        self._joints = {
            name: self.plant.GetJointByName(name, instance) for name in self.names
        }
        for joint in self._joints.values():
            if joint.num_positions() != 1 or joint.num_velocities() != 1:
                raise ValueError("Native primitive must remain scalar")
            joint.set_position_limits(np.array([-np.inf]), np.array([np.inf]))
            joint.set_velocity_limits(np.array([-np.inf]), np.array([np.inf]))
        closure = meta["closure"]
        self._closure = []
        for suffix in ("a", "b"):
            body = self.plant.GetBodyByName(
                meta["body_links"][closure[f"body_{suffix}"]], instance
            )
            frame = self.plant.AddFrame(
                api.FixedOffsetFrame(
                    f"native_closure_{suffix}",
                    body.body_frame(),
                    api.RigidTransform(
                        np.asarray(closure[f"placement_{suffix}"], dtype=float)
                    ),
                )
            )
            self._closure.append(frame)
        self._frames = {
            name: self.plant.GetBodyByName(link, instance).body_frame()
            for name, link in meta["frame_links"].items()
        }
        self.plant.Finalize()
        if self.plant.num_positions() != len(
            self.names
        ) or self.plant.num_velocities() != len(self.names):
            raise ValueError("Native coordinate inventory differs after Drake parsing")
        self.context = self.plant.CreateDefaultContext()
        self._q_indices = [self._joints[name].position_start() for name in self.names]
        self._v_indices = [self._joints[name].velocity_start() for name in self.names]
        self._last_closure: tuple[Array, Array] | None = None

    def _vector(self, values: Mapping[str, float], indices: list[int]) -> Array:
        if set(values) != set(self.names):
            raise ValueError("Provide exactly the native coordinate inventory")
        result = np.empty(len(self.names))
        result[indices] = [values[name] for name in self.names]
        if not np.isfinite(result).all():
            raise ValueError("Native state and effort values must be finite")
        return result

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, Array]:
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        return {
            name: self.plant.CalcRelativeTransform(
                self.context, self.plant.world_frame(), frame
            )
            .GetAsMatrix4()
            .copy()
            for name, frame in self._frames.items()
        }

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        velocity = self._vector(rates, self._v_indices)
        self.plant.SetVelocities(self.context, velocity)
        a, b = self._closure
        jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            self._api.JacobianWrtVariable.kV,
            b,
            np.zeros(3),
            a,
            a,
        )
        bias = self.plant.CalcBiasSpatialAcceleration(
            self.context,
            self._api.JacobianWrtVariable.kV,
            b,
            np.zeros(3),
            a,
            a,
        ).get_coeffs()
        force = (
            self._vector(primitive_efforts, self._v_indices)
            + self.plant.CalcGravityGeneralizedForces(self.context)
            - self.plant.CalcBiasTerm(self.context)
        )
        acceleration = solve_weld_acceleration(
            self.plant.CalcMassMatrix(self.context),
            force,
            jacobian,
            bias,
        )
        pose = self.plant.CalcRelativeTransform(self.context, a, b)
        self._last_closure = (
            np.concatenate(
                (
                    pose.translation(),
                    Rotation.from_matrix(pose.rotation().matrix()).as_rotvec(),
                )
            ),
            np.concatenate(((jacobian @ velocity)[3:], (jacobian @ velocity)[:3])),
        )
        return {
            name: float(acceleration[index])
            for name, index in zip(self.names, self._v_indices, strict=True)
        }

    def closure_errors(self) -> tuple[Array, Array]:
        """Copy last acceleration's closure pose/rate; never silently recompute."""
        if self._last_closure is None:
            raise ValueError("Call accelerations before requesting closure diagnostics")
        return self._last_closure[0].copy(), self._last_closure[1].copy()
