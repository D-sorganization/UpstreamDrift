"""Drake full-body model adapter with shared rigid-ground contact and rigid closure.

Applies the shared Hunt-Crossley and regularized Coulomb contact law (FB-2) via
externally applied spatial forces and generalized contact torques on the calcaneus
bodies (calcn_r, calcn_l), and enforces the six-dimensional dual-grip weld closure
via an explicit continuous KKT solve.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from importlib import import_module
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.drake.python.full_body_urdf import (
    export_full_body_urdf,
)
from src.shared.python.contracts import precondition
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.full_body_spec import (
    FULL_BODY_SCHEMA_VERSION,
    upper_body_slice,
)

Array: TypeAlias = NDArray[np.float64]


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


class FullBodyDrakeModel:
    """Full-body Drake adapter combining 41-DOF tree, weld closure, and FB-2 contact."""

    def __init__(self, specification: Mapping[str, Any]) -> None:
        if specification.get("schema_version") != FULL_BODY_SCHEMA_VERSION:
            raise ValueError("Unsupported full-body schema: expected full-body-v1")
        self.specification = specification
        spec_bytes = json.dumps(specification).encode("utf-8")
        self.xml, self.metadata = export_full_body_urdf(spec_bytes)
        self.model_sha256 = self.metadata["model_sha256"]

        api: Any = import_module("pydrake.all")
        if type(api).__module__ == "unittest.mock" or not hasattr(
            api, "MultibodyPlant"
        ):
            raise ImportError("Real pydrake installation required (found mock)")
        self._api = api
        self._wrt_v = api.JacobianWrtVariable.kV
        self.plant = api.MultibodyPlant(time_step=0.0)
        self._instance = api.Parser(self.plant).AddModelsFromString(self.xml, "urdf")[0]
        self.plant.WeldFrames(
            self.plant.world_frame(),
            self.plant.GetBodyByName(
                self.metadata["body_links"]["world"], self._instance
            ).body_frame(),
        )
        self.plant.mutable_gravity_field().set_gravity_vector(
            specification["gravity_m_s2"]
        )

        self.names = tuple(self.metadata["coordinate_order"])
        self._joints = {
            name: self.plant.GetJointByName(name, self._instance) for name in self.names
        }
        for joint in self._joints.values():
            if joint.num_positions() != 1 or joint.num_velocities() != 1:
                raise ValueError("All coordinates must be scalar 1-DOF joints")
            joint.set_position_limits(np.array([-np.inf]), np.array([np.inf]))
            joint.set_velocity_limits(np.array([-np.inf]), np.array([np.inf]))

        self._init_closure_and_frames(specification, api)
        self._init_contact(specification)

        self.plant.Finalize()
        if self.plant.num_positions() != len(
            self.names
        ) or self.plant.num_velocities() != len(self.names):
            raise ValueError("Coordinate inventory differs after Drake parsing")

        self.context = self.plant.CreateDefaultContext()
        self._q_indices = [self._joints[name].position_start() for name in self.names]
        self._v_indices = [self._joints[name].velocity_start() for name in self.names]
        self._last_closure: tuple[Array, Array] | None = None
        self.gravity = np.asarray(specification["gravity_m_s2"], dtype=float)
        self.mass_kg = float(self.plant.CalcTotalMass(self.context))
        self.upper_body_coordinates: int = int(
            specification.get("upper_body_counts", {}).get("coordinates", 0)
        )

    def _init_closure_and_frames(self, spec: Mapping[str, Any], api: Any) -> None:
        closure = spec["closure"]
        self._closure = []
        for suffix in ("a", "b"):
            body = self.plant.GetBodyByName(
                self.metadata["body_links"][closure[f"body_{suffix}"]], self._instance
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
            frame["name"]: self.plant.GetFrameByName(
                self.metadata["frame_links"][frame["name"]], self._instance
            )
            for frame in spec["frames"]
        }

    def _init_contact(self, spec: Mapping[str, Any]) -> None:
        c_spec = spec["contact"]
        self.contact_parameters = ContactParameters(**c_spec["parameters"])
        grav = np.asarray(spec["gravity_m_s2"], dtype=float)
        g_mag = float(
            math.sqrt(np.dot(grav, grav))
        )  # ⚡ Bolt: math.sqrt(np.dot) is ~2.5x faster than np.linalg.norm
        if g_mag <= 0.0:
            raise ValueError("Drake ground contact requires nonzero gravity vector")
        up_vec = (-grav / g_mag).tolist()
        h_m = float(c_spec["ground"].get("height_m") or 0.0)
        self.ground_plane = GroundPlane(
            normal=(float(up_vec[0]), float(up_vec[1]), float(up_vec[2])),
            height_m=h_m,
        )

        self._spheres: dict[str, dict[str, Any]] = {}
        for sphere in c_spec["spheres"]:
            s_name = sphere["name"]
            meta_s = self.metadata["contact_spheres"][s_name]
            frame = self.plant.GetFrameByName(meta_s["link"], self._instance)
            body = self.plant.GetBodyByName(
                self.metadata["body_links"][sphere["body"]], self._instance
            )
            self._spheres[s_name] = {
                "frame": frame,
                "body": body,
                "radius_m": float(sphere["radius_m"]),
                "position_m": np.asarray(sphere["position_m"], dtype=float),
            }

    def upper_body_model(self) -> FullBodyDrakeModel:
        """Construct an adapter for the upper-body slice."""
        upper_spec = upper_body_slice(self.specification)
        full_spec_like = dict(self.specification)
        full_spec_like["bodies"] = upper_spec["bodies"]
        full_spec_like["joints"] = upper_spec["joints"]
        full_spec_like["coordinate_order"] = upper_spec["coordinate_order"]
        full_spec_like["frames"] = upper_spec["frames"]
        full_spec_like["closure"] = upper_spec["closure"]
        full_spec_like["contact"] = {
            **self.specification["contact"],
            "spheres": [],
        }
        return FullBodyDrakeModel(full_spec_like)

    def _vector(self, values: Mapping[str, float], indices: list[int]) -> Array:
        if set(values) != set(self.names):
            raise ValueError("Provide exactly the model coordinate inventory")
        result = np.empty(len(self.names))
        result[indices] = [values[name] for name in self.names]
        if not np.isfinite(result).all():
            raise ValueError("Coordinates and rates must be finite")
        return result

    def frame_poses(self, coordinates: Mapping[str, float]) -> dict[str, Array]:
        """Compute 4x4 forward kinematics poses for all defined frames."""
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

    @precondition(
        lambda self, coordinates, attachments: (
            coordinates is not None and attachments is not None
        ),
        "coordinates and attachments must be provided",
    )
    def marker_positions(
        self,
        coordinates: Mapping[str, float] | Array,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
    ) -> Array:
        """Compute 3D world positions of attached markers.

        Args:
            coordinates: Joint position values by coordinate name, or array in coordinate_order.
            attachments: Map of marker label to (body_or_frame_name, (ox, oy, oz)).

        Returns:
            Array of shape (N, 3) marker positions in world frame.
        """
        if isinstance(coordinates, Mapping):
            q_vec = self._vector(coordinates, self._q_indices)
        else:
            q_vec = np.asarray(coordinates, dtype=float)
            if q_vec.size != len(self.names):
                raise ValueError(f"Expected {len(self.names)} coordinates")
        self.plant.SetPositions(self.context, q_vec)
        world_frame = self.plant.world_frame()
        positions: list[Array] = []
        for body_name, offset in attachments.values():
            offset_arr = np.asarray(offset, dtype=float)
            if body_name in self._frames:
                frame = self._frames[body_name]
                x_wf = self.plant.CalcRelativeTransform(
                    self.context, world_frame, frame
                )
                pos = x_wf.rotation().matrix() @ offset_arr + x_wf.translation()
            elif body_name in self.metadata["body_links"]:
                link_name = self.metadata["body_links"][body_name]
                body_obj = self.plant.GetBodyByName(link_name, self._instance)
                x_wb = self.plant.EvalBodyPoseInWorld(self.context, body_obj)
                pos = x_wb.rotation().matrix() @ offset_arr + x_wb.translation()
            else:
                raise ValueError(
                    f"Body or frame '{body_name}' not found in Drake model"
                )
            positions.append(pos)
        if not positions:
            return np.empty((0, 3), dtype=float)
        return np.asarray(positions, dtype=float)

    def get_sphere_kinematics(
        self,
        sphere_name: str,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> tuple[Array, Array, float]:
        """Compute sphere center world position, linear velocity, and radius."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        self.plant.SetVelocities(self.context, self._vector(rates, self._v_indices))
        info = self._spheres[sphere_name]
        frame = info["frame"]
        pos = self.plant.CalcRelativeTransform(
            self.context, self.plant.world_frame(), frame
        ).translation()
        body = info["body"]
        v_wb = self.plant.EvalBodySpatialVelocityInWorld(self.context, body)
        p_wb = self.plant.CalcRelativeTransform(
            self.context, self.plant.world_frame(), body.body_frame()
        ).translation()
        p_bp_w = pos - p_wb
        v_wp = v_wb.Shift(p_bp_w)
        vel = v_wp.translational()
        return pos.copy(), vel.copy(), float(info["radius_m"])

    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> dict[str, ContactSample]:
        """Evaluate shared contact law forces for each foot contact sphere."""
        samples: dict[str, ContactSample] = {}
        for s_name in self._spheres:
            pos, vel, radius = self.get_sphere_kinematics(s_name, coordinates, rates)
            samples[s_name] = sphere_ground_contact(
                pos, vel, radius, self.ground_plane, self.contact_parameters
            )
        return samples

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        """Return the canonical 41-coordinate order of the full-body model."""
        return self.names

    def evaluate_contact_samples(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
    ) -> dict[str, ContactSample]:
        """Evaluate shared contact law forces for each foot contact sphere."""
        return self.contact_forces(coordinates, rates)

    def _evaluate_closure_residuals(
        self, a: Any, b: Any, jacobian: Array, velocity: Array
    ) -> tuple[Array, Array]:
        pose = self.plant.CalcRelativeTransform(self.context, a, b)
        pos_residual = np.concatenate(
            (
                pose.translation(),
                Rotation.from_matrix(pose.rotation().matrix()).as_rotvec(),
            )
        )
        vel_residual = np.concatenate(
            ((jacobian @ velocity)[3:], (jacobian @ velocity)[:3])
        )
        return pos_residual, vel_residual

    def _closure_jacobian(self) -> tuple[Any, Any, Array]:
        a, b = self._closure
        jacobian = self.plant.CalcJacobianSpatialVelocity(
            self.context,
            self._wrt_v,
            b,
            np.zeros(3),
            a,
            a,
        )
        return a, b, jacobian

    def closure_residuals(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> tuple[Array, Array]:
        """Compute closure pose and rate residuals without solving acceleration."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        velocity = self._vector(rates, self._v_indices)
        self.plant.SetVelocities(self.context, velocity)
        a, b, jacobian = self._closure_jacobian()
        return self._evaluate_closure_residuals(a, b, jacobian, velocity)

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> dict[str, float]:
        """Forward accelerations combining joint efforts, contact forces, and weld closure."""
        self.plant.SetPositions(
            self.context, self._vector(coordinates, self._q_indices)
        )
        velocity = self._vector(rates, self._v_indices)
        self.plant.SetVelocities(self.context, velocity)

        samples = self.contact_forces(coordinates, rates)
        tau_contact = self._accumulate_contact_torques(samples)
        a, b, jacobian, bias = self._closure_bias_and_jacobian()

        force = (
            self._vector(primitive_efforts, self._v_indices)
            + tau_contact
            + self.plant.CalcGravityGeneralizedForces(self.context)
            - self.plant.CalcBiasTerm(self.context)
        )
        acceleration = solve_weld_acceleration(
            self.plant.CalcMassMatrix(self.context),
            force,
            jacobian,
            bias,
        )

        self._last_closure = self._evaluate_closure_residuals(a, b, jacobian, velocity)
        return {
            name: float(acceleration[index])
            for name, index in zip(self.names, self._v_indices, strict=True)
        }

    def closure_errors(self) -> tuple[Array, Array]:
        """Return detached pose/rate residuals from the last acceleration call."""
        if self._last_closure is None:
            raise ValueError("Call accelerations before requesting closure diagnostics")
        return self._last_closure[0].copy(), self._last_closure[1].copy()

    def _coerce_state(
        self,
        coordinates: Mapping[str, float] | Array,
        rates: Mapping[str, float] | Array,
    ) -> tuple[Array, dict[str, float], Array, dict[str, float]]:
        if isinstance(coordinates, Mapping):
            q_vec = self._vector(coordinates, self._q_indices)
            coords_dict = dict(coordinates)
        else:
            q_vec = np.asarray(coordinates, dtype=float)
            coords_dict = dict(zip(self.names, q_vec, strict=True))

        if isinstance(rates, Mapping):
            v_vec = self._vector(rates, self._v_indices)
            rates_dict = dict(rates)
        else:
            v_vec = np.asarray(rates, dtype=float)
            rates_dict = dict(zip(self.names, v_vec, strict=True))
        return q_vec, coords_dict, v_vec, rates_dict

    def _sphere_jacobian_translational(self, frame: Any) -> Array:
        return self.plant.CalcJacobianTranslationalVelocity(
            self.context,
            self._wrt_v,
            frame,
            np.zeros(3),
            self.plant.world_frame(),
            self.plant.world_frame(),
        )

    def _accumulate_contact_torques(
        self, samples: Mapping[str, ContactSample]
    ) -> Array:
        tau_contact = np.zeros(self.plant.num_velocities())
        for s_name, sample in samples.items():
            f_contact = sample.normal_force_n + sample.friction_force_n
            if np.linalg.norm(f_contact) <= 0.0:
                continue
            frame = self._spheres[s_name]["frame"]
            j_trans = self._sphere_jacobian_translational(frame)
            tau_contact += j_trans.T @ f_contact
        return tau_contact

    def _closure_bias_and_jacobian(self) -> tuple[Any, Any, Array, Array]:
        a, b, jacobian = self._closure_jacobian()
        bias = self.plant.CalcBiasSpatialAcceleration(
            self.context,
            self._wrt_v,
            b,
            np.zeros(3),
            a,
            a,
        ).get_coeffs()
        return a, b, jacobian, bias

    @precondition(
        lambda self, coordinates, rates, primitive_efforts, dt: dt > 0.0,
        "dt must be positive",
    )
    def step(
        self,
        coordinates: Mapping[str, float] | Array,
        rates: Mapping[str, float] | Array,
        primitive_efforts: Mapping[str, float] | Array,
        dt: float,
    ) -> tuple[Array, Array]:
        """Advance plant state [q, v] by dt using symplectic Euler integration.

        Accelerations account for joint efforts, shared contact forces, and weld closure.
        """
        q_vec, coords_dict, v_vec, rates_dict = self._coerce_state(coordinates, rates)

        if isinstance(primitive_efforts, Mapping):
            efforts_dict = dict(primitive_efforts)
        else:
            efforts_dict = dict(
                zip(self.names, np.asarray(primitive_efforts, dtype=float), strict=True)
            )

        acc_dict = self.accelerations(coords_dict, rates_dict, efforts_dict)
        qdd = np.array([acc_dict[c] for c in self.names], dtype=float)
        next_v = v_vec + qdd * dt
        next_q = q_vec + next_v * dt
        return next_q, next_v

    def centre_of_mass(self, q: Array) -> tuple[Array, Array]:
        """Whole-body centre of mass and its Jacobian (spec coordinate order)."""
        q_vec = np.asarray(q, dtype=float)
        q_raw = np.empty(len(self.names))
        q_raw[self._q_indices] = q_vec
        self.plant.SetPositions(self.context, q_raw)
        com = np.asarray(
            self.plant.CalcCenterOfMassPositionInWorld(self.context), dtype=float
        )
        jac_raw = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(
            self.context,
            self._wrt_v,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        jac_ordered = jac_raw[:, self._v_indices]
        return com.copy(), jac_ordered.copy()

    def affine_dynamics(
        self,
        coordinates: Mapping[str, float] | Array,
        rates: Mapping[str, float] | Array,
    ) -> tuple[Array, Array]:
        """Return (A, b) with a = A @ tau_actuated + b in canonical coordinate order."""
        q_vec, coords_dict, v_vec, rates_dict = self._coerce_state(coordinates, rates)

        self.plant.SetPositions(self.context, q_vec)
        self.plant.SetVelocities(self.context, v_vec)

        samples = self.contact_forces(coords_dict, rates_dict)
        tau_contact = self._accumulate_contact_torques(samples)
        _, _, jacobian, bias = self._closure_bias_and_jacobian()

        force = (
            tau_contact
            + self.plant.CalcGravityGeneralizedForces(self.context)
            - self.plant.CalcBiasTerm(self.context)
        )
        mass = self.plant.CalcMassMatrix(self.context)
        k = bias.size
        matrix = np.block([[mass, -jacobian.T], [jacobian, np.zeros((k, k))]])

        actuated_indices = self._v_indices[6:]
        actuated_count = len(actuated_indices)
        rhs = np.zeros((mass.shape[0] + k, 1 + actuated_count))
        rhs[: mass.shape[0], 0] = force
        rhs[mass.shape[0] :, 0] = -bias
        rhs[actuated_indices, 1:] = np.eye(actuated_count)

        sol_raw = np.linalg.solve(matrix, rhs)[: mass.shape[0]]
        sol_ordered = sol_raw[self._v_indices]
        return sol_ordered[:, 1:], sol_ordered[:, 0]

    def sphere_jacobians(self) -> Array:
        """Jacobian rows of all contact sphere centers in canonical coordinate order."""
        rows = [
            self._sphere_jacobian_translational(self._spheres[s]["frame"])[
                :, self._v_indices
            ]
            for s in self._spheres
        ]
        return np.concatenate(rows, axis=0)

    def _compute_frame_momentum(
        self, simulator: Any, q: Array, v: Array
    ) -> tuple[Array, Array, Array]:
        """Whole-body CoM position, linear momentum, and angular momentum."""
        q_vec = np.asarray(q, dtype=float)
        v_vec = np.asarray(v, dtype=float)
        q_raw = np.empty(len(self.names))
        q_raw[self._q_indices] = q_vec
        v_raw = np.empty(len(self.names))
        v_raw[self._v_indices] = v_vec
        self.plant.SetPositions(self.context, q_raw)
        self.plant.SetVelocities(self.context, v_raw)

        com = np.asarray(
            self.plant.CalcCenterOfMassPositionInWorld(self.context), dtype=float
        )
        sm = self.plant.CalcSpatialMomentumInWorldAboutPoint(self.context, com)
        return (
            com.copy(),
            np.asarray(sm.translational(), dtype=float),
            np.asarray(sm.rotational(), dtype=float),
        )
