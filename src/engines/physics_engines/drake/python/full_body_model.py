"""Full-body Drake model adapter with shared rigid-ground contact and rigid closure.

Applies the shared Hunt-Crossley and regularized Coulomb contact law (FB-2) via
spatial contact wrenches on calcaneus bodies and enforces the six-dimensional
dual-grip weld closure via an explicit rigid solve using Drake MultibodyPlant.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from importlib import import_module
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.drake.python.full_body_urdf_contract import (
    validate_full_body_urdf_bundle,
)
from src.shared.python.motion_matching.contact_law import (
    ContactSample,
    contact_setup_from_spec,
    sphere_ground_contact,
)

Array: TypeAlias = NDArray[np.float64]


def _solve_rigid_kkt_acceleration(
    mass: Array, force: Array, jacobian: Array, bias: Array
) -> Array:
    """Solve M a - J.T lambda = force, J a + bias = 0 without relaxation."""
    m_arr, f_arr, j_arr, b_arr = (
        np.asarray(val, dtype=float) for val in (mass, force, jacobian, bias)
    )
    n = f_arr.size
    if (
        f_arr.shape != (n,)
        or m_arr.shape != (n, n)
        or j_arr.ndim != 2
        or j_arr.shape[1] != n
        or b_arr.shape != (j_arr.shape[0],)
    ):
        raise ValueError("Incompatible dimensions for KKT solve")

    k = j_arr.shape[0]
    kkt_matrix = np.block([[m_arr, -j_arr.T], [j_arr, np.zeros((k, k))]])
    result = np.linalg.solve(kkt_matrix, np.concatenate((f_arr, -b_arr)))[:n]
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite constrained acceleration")
    return result.copy()


class NativeDrakeFullBodyModel:
    """Full-body Drake adapter combining closed-loop kinematics and shared contact."""

    def __init__(
        self, urdf_bytes: bytes, sidecar_bytes: bytes, model_bytes: bytes
    ) -> None:
        meta = validate_full_body_urdf_bundle(urdf_bytes, sidecar_bytes, model_bytes)
        spec = json.loads(model_bytes)
        api: Any = import_module("pydrake.all")
        self._api = api
        self._wrt_v = api.JacobianWrtVariable.kV

        self.plant = api.MultibodyPlant(time_step=0.0)
        self.instance = api.Parser(self.plant).AddModelsFromString(
            urdf_bytes.decode("utf-8"), "urdf"
        )[0]
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                meta["body_links"]["world"], self.instance
            ).body_frame(),
        )
        self.plant.mutable_gravity_field().set_gravity_vector(meta["gravity_m_s2"])

        self.names = tuple(meta["coordinate_order"])
        self._joints = {
            name: self.plant.GetJointByName(name, self.instance) for name in self.names
        }
        for joint in self._joints.values():
            if joint.num_positions() != 1 or joint.num_velocities() != 1:
                raise ValueError(
                    f"Primitive joint {joint.name()} must remain scalar, got {joint.num_positions()} pos, {joint.num_velocities()} vel"
                )
            joint.set_position_limits(np.array([-np.inf]), np.array([np.inf]))
            joint.set_velocity_limits(np.array([-np.inf]), np.array([np.inf]))

        self._setup_frames(meta, api)
        self.plant.Finalize()

        if self.plant.num_positions() != len(
            self.names
        ) or self.plant.num_velocities() != len(self.names):
            raise ValueError("Coordinate inventory differs after Drake parsing")

        self.context = self.plant.CreateDefaultContext()
        self._q_indices = [self._joints[name].position_start() for name in self.names]
        self._v_indices = [self._joints[name].velocity_start() for name in self.names]
        self._last_closure: tuple[Array, Array] | None = None

        self.contact_parameters, self.ground_plane = contact_setup_from_spec(spec)
        self._contact_spheres = {
            s["name"]: {
                "body_name": meta["contact_links"][s["name"]],
                "radius": float(s["radius_m"]),
            }
            for s in spec["contact"]["spheres"]
        }

    def _setup_frames(self, meta: Mapping[str, Any], api: Any) -> None:
        closure = meta["closure"]
        self._closure = []
        for suffix in ("a", "b"):
            body = self.plant.GetBodyByName(
                meta["body_links"][closure[f"body_{suffix}"]], self.instance
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
            name: self.plant.GetBodyByName(link, self.instance).body_frame()
            for name, link in meta["frame_links"].items()
        }

    def _vector(self, values: Mapping[str, float], indices: list[int]) -> Array:
        if set(values) != set(self.names):
            raise ValueError("Provide exactly the full-body coordinate inventory")
        result = np.empty(len(self.names))
        result[indices] = [values[name] for name in self.names]
        if not np.isfinite(result).all():
            raise ValueError("State and effort values must be finite")
        return result

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, Array]:
        """Compute 4x4 forward kinematics poses for all defined frame sites."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        world = self.plant.world_frame()
        return {
            name: self.plant.CalcRelativeTransform(self.context, world, frame)
            .GetAsMatrix4()
            .copy()
            for name, frame in self._frames.items()
        }

    def _translation_jacobian(self, frame: Any) -> Array:
        world = self.plant.world_frame()
        return self.plant.CalcJacobianTranslationalVelocity(
            self.context,
            self._wrt_v,
            frame,
            np.zeros(3),
            world,
            world,
        )

    def evaluate_contact_samples(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> dict[str, ContactSample]:
        """Evaluate shared Hunt-Crossley and regularized Coulomb model at state."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        self.plant.SetVelocities(self.context, self._vector(rates, self._v_indices))

        world = self.plant.world_frame()
        samples: dict[str, ContactSample] = {}

        for s_name, s_info in self._contact_spheres.items():
            frame = self.plant.GetBodyByName(
                s_info["body_name"], self.instance
            ).body_frame()
            pos = self.plant.CalcPointsPositions(
                self.context, frame, np.zeros((3, 1)), world
            ).ravel()
            jac_pos = self._translation_jacobian(frame)
            v_vec = self.plant.GetVelocities(self.context)
            vel = jac_pos @ v_vec

            sample = sphere_ground_contact(
                pos, vel, s_info["radius"], self.ground_plane, self.contact_parameters
            )
            samples[s_name] = sample

        return samples

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Solve constrained dynamics with applied contact forces and dual-grip weld."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        velocity = self._vector(rates, self._v_indices)
        self.plant.SetVelocities(self.context, velocity)

        samples = self.evaluate_contact_samples(coordinates, rates)
        tau_contact = np.zeros(len(self.names))

        for s_name, sample in samples.items():
            f_contact = np.asarray(sample.normal_force_n) + np.asarray(
                sample.friction_force_n
            )
            if np.linalg.norm(f_contact) <= 0.0:
                continue
            frame = self.plant.GetBodyByName(
                self._contact_spheres[s_name]["body_name"], self.instance
            ).body_frame()
            jac_pos = self._translation_jacobian(frame)
            tau_contact += jac_pos.T @ f_contact

        a, b = self._closure
        jac = self.plant.CalcJacobianSpatialVelocity(
            self.context, self._wrt_v, b, np.zeros(3), a, a
        )
        drift = self.plant.CalcBiasSpatialAcceleration(
            self.context, self._wrt_v, b, np.zeros(3), a, a
        ).get_coeffs()

        force = (
            self._vector(primitive_efforts, self._v_indices)
            + tau_contact
            + self.plant.CalcGravityGeneralizedForces(self.context)
            - self.plant.CalcBiasTerm(self.context)
        )
        acc_arr = _solve_rigid_kkt_acceleration(
            self.plant.CalcMassMatrix(self.context),
            force,
            jac,
            drift,
        )

        pose = self.plant.CalcRelativeTransform(self.context, a, b)
        self._last_closure = (
            np.concatenate(
                (
                    pose.translation(),
                    Rotation.from_matrix(pose.rotation().matrix()).as_rotvec(),
                )
            ),
            np.concatenate(((jac @ velocity)[3:], (jac @ velocity)[:3])),
        )
        return {
            name: float(acc_arr[idx])
            for name, idx in zip(self.names, self._v_indices, strict=True)
        }

    def closure_errors(self) -> tuple[Array, Array]:
        """Copy last acceleration's closure pose/rate residuals."""
        if self._last_closure is None:
            raise ValueError("Call accelerations before requesting closure diagnostics")
        return self._last_closure[0].copy(), self._last_closure[1].copy()
