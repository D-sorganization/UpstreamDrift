"""Spatial full-body common-state dynamics experiment.

The study uses one immutable reduced-order model definition in two independent
dynamics implementations: MuJoCo's native inverse dynamics and a dependency-
free Lagrange/Christoffel implementation assembled from body Jacobians.  The
experiment is deliberately same-state.  It tests dynamics transport without
confounding the comparison with trajectory divergence.

The hand loads are prescribed action--reaction pairs.  Consequently this
module can test spatial wrench transport and inverse-dynamics parity, but it
cannot establish that the loads arise passively from closed-loop contact.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
from typing import Any, Literal, TypeAlias

import numpy as np
import numpy.typing as npt

Array: TypeAlias = npt.NDArray[np.float64]
IntArray: TypeAlias = npt.NDArray[np.int64]


@dataclass(frozen=True)
class JointSpec:
    """One scalar joint in the shared model tree."""

    name: str
    parent: int
    kind: Literal["revolute", "prismatic"]
    axis: Array
    offset_m: Array
    region: str


@dataclass(frozen=True)
class BodySpec:
    """A spherical inertia element fixed to a post-joint frame."""

    name: str
    joint: int
    mass_kg: float
    radius_m: float
    com_offset_m: Array
    region: str


@dataclass(frozen=True)
class SpatialModel:
    """Single source of truth consumed by both dynamics implementations."""

    joints: tuple[JointSpec, ...]
    bodies: tuple[BodySpec, ...]
    club_dof_indices: IntArray
    lead_hand_joint: int
    trail_hand_joint: int
    club_frame_joint: int

    @property
    def nq(self) -> int:
        return len(self.joints)

    @cached_property
    def canonical_hash(self) -> str:
        payload = {
            "schema": "proximal-distal-spatial-model-v2",
            "joints": [
                {
                    "name": joint.name,
                    "parent": joint.parent,
                    "kind": joint.kind,
                    "axis": joint.axis.tolist(),
                    "offset_m": joint.offset_m.tolist(),
                    "region": joint.region,
                }
                for joint in self.joints
            ],
            "bodies": [
                {
                    "name": body.name,
                    "joint": body.joint,
                    "mass_kg": body.mass_kg,
                    "radius_m": body.radius_m,
                    "com_offset_m": body.com_offset_m.tolist(),
                    "region": body.region,
                }
                for body in self.bodies
            ],
            "interfaces": {
                "club_dof_indices": self.club_dof_indices.tolist(),
                "lead_hand_joint": self.lead_hand_joint,
                "trail_hand_joint": self.trail_hand_joint,
                "club_frame_joint": self.club_frame_joint,
            },
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True)
class SpatialTolerance:
    """Preoutcome numerical equivalence region."""

    absolute: float = 0.75
    relative: float = 0.05
    event_time_s: float = 0.004
    calibration: str = (
        "finite-difference steps 2e-6 and 1e-6 on a manufactured spherical-"
        "inertia tree, fixed before the preferred event-window result"
    )


@dataclass(frozen=True)
class SpatialExperimentConfig:
    """Configuration with fail-closed preconditions."""

    duration_s: float = 0.24
    sample_dt_s: float = 0.004
    derivative_step: float = 1e-6
    tolerance_source: str = "predeclared"
    tolerance: SpatialTolerance = SpatialTolerance()

    def __post_init__(self) -> None:
        if not (self.duration_s > 0.20):
            raise ValueError("duration_s must extend beyond the 0.20 s intervention")
        if not (0 < self.sample_dt_s <= 0.01):
            raise ValueError("sample_dt_s must be in (0, 0.01]")
        if not (1e-8 <= self.derivative_step <= 1e-4):
            raise ValueError("derivative_step must be in [1e-8, 1e-4]")
        if self.tolerance_source != "predeclared":
            raise ValueError("tolerances must remain predeclared before outcomes")


@dataclass(frozen=True)
class Kinematics:
    """World-frame joint/body kinematics and Jacobians."""

    joint_position_m: Array
    joint_rotation: Array
    joint_axis_world: Array
    body_position_m: Array
    body_rotation: Array
    body_linear_jacobian: Array
    body_angular_jacobian: Array


@dataclass(frozen=True)
class HandWrenchSample:
    """Reference-explicit two-hand action--reaction sample."""

    lead_force_n: Array
    trail_force_n: Array
    lead_position_m: Array
    trail_position_m: Array
    reference_position_m: Array
    club_wrench: Array
    body_wrench: Array
    compatible_twist: Array
    force_generated_couple_nm: float
    action_reaction_power_residual_w: float


@dataclass(frozen=True)
class CrossFormulationResult:
    """Bounded summary of an executed two-implementation comparison."""

    formulation_names: tuple[str, str]
    model_hashes: tuple[str, str]
    time_s: Array
    inverse_dynamics_lagrange: Array
    inverse_dynamics_mujoco: Array
    force_generated_couple_nm: Array
    out_of_plane_motion_m: float
    max_relative_inverse_dynamics_error: float
    max_absolute_generalized_force_error: float
    max_absolute_mass_matrix_error: float
    max_relative_mass_matrix_error: float
    max_absolute_bias_force_error: float
    max_relative_bias_force_error: float
    external_load_convention_mismatch_relative_error: float
    intervention_event_grid_error_s: float
    tolerance: SpatialTolerance
    classification: Literal["equivalent", "structural_discrepancy"]


def _vec(x: float, y: float, z: float) -> Array:
    return np.array([x, y, z], dtype=np.float64)


def build_spatial_model() -> SpatialModel:
    """Return the deterministic 20-DOF reduced full-body + club tree."""

    joints = (
        JointSpec(
            "pelvis_yaw", -1, "revolute", _vec(0, 0, 1), _vec(0, 0, 0.95), "pelvis"
        ),
        JointSpec("pelvis_roll", 0, "revolute", _vec(1, 0, 0), _vec(0, 0, 0), "pelvis"),
        JointSpec(
            "torso_pitch", 1, "revolute", _vec(0, 1, 0), _vec(0, 0, 0.20), "torso"
        ),
        JointSpec("torso_yaw", 2, "revolute", _vec(0, 0, 1), _vec(0, 0, 0.25), "torso"),
        JointSpec(
            "lead_shoulder_x",
            3,
            "revolute",
            _vec(1, 0, 0),
            _vec(0, 0.20, 0.18),
            "lead_arm",
        ),
        JointSpec(
            "lead_shoulder_y", 4, "revolute", _vec(0, 1, 0), _vec(0, 0, 0), "lead_arm"
        ),
        JointSpec(
            "lead_shoulder_z", 5, "revolute", _vec(0, 0, 1), _vec(0, 0, 0), "lead_arm"
        ),
        JointSpec(
            "lead_elbow", 6, "revolute", _vec(0, 1, 0), _vec(0.30, 0, -0.05), "lead_arm"
        ),
        JointSpec(
            "lead_wrist", 7, "revolute", _vec(1, 0, 0), _vec(0.27, 0, 0), "lead_arm"
        ),
        JointSpec(
            "trail_shoulder_x",
            3,
            "revolute",
            _vec(1, 0, 0),
            _vec(0, -0.20, 0.18),
            "trail_arm",
        ),
        JointSpec(
            "trail_shoulder_y", 9, "revolute", _vec(0, 1, 0), _vec(0, 0, 0), "trail_arm"
        ),
        JointSpec(
            "trail_shoulder_z",
            10,
            "revolute",
            _vec(0, 0, 1),
            _vec(0, 0, 0),
            "trail_arm",
        ),
        JointSpec(
            "trail_elbow",
            11,
            "revolute",
            _vec(0, 1, 0),
            _vec(0.30, 0, -0.05),
            "trail_arm",
        ),
        JointSpec(
            "trail_wrist", 12, "revolute", _vec(1, 0, 0), _vec(0.27, 0, 0), "trail_arm"
        ),
        JointSpec("club_x", -1, "prismatic", _vec(1, 0, 0), _vec(0, 0, 0), "club"),
        JointSpec("club_y", 14, "prismatic", _vec(0, 1, 0), _vec(0, 0, 0), "club"),
        JointSpec("club_z", 15, "prismatic", _vec(0, 0, 1), _vec(0, 0, 0), "club"),
        JointSpec("club_roll", 16, "revolute", _vec(1, 0, 0), _vec(0, 0, 0), "club"),
        JointSpec("club_pitch", 17, "revolute", _vec(0, 1, 0), _vec(0, 0, 0), "club"),
        JointSpec("club_yaw", 18, "revolute", _vec(0, 0, 1), _vec(0, 0, 0), "club"),
    )
    physical_bodies = (
        BodySpec("lower_body", 1, 27.0, 0.19, _vec(0, 0, -0.43), "lower_body"),
        BodySpec("pelvis_mass", 1, 11.0, 0.14, _vec(0, 0, 0.02), "pelvis"),
        BodySpec("torso_mass", 3, 26.0, 0.17, _vec(0, 0, 0.16), "torso"),
        BodySpec("lead_upper_arm", 6, 2.4, 0.055, _vec(0.15, 0, -0.025), "lead_arm"),
        BodySpec("lead_forearm", 7, 1.5, 0.045, _vec(0.135, 0, 0), "lead_arm"),
        BodySpec("lead_hand", 8, 0.50, 0.040, _vec(0.055, 0, 0), "lead_arm"),
        BodySpec("trail_upper_arm", 11, 2.4, 0.055, _vec(0.15, 0, -0.025), "trail_arm"),
        BodySpec("trail_forearm", 12, 1.5, 0.045, _vec(0.135, 0, 0), "trail_arm"),
        BodySpec("trail_hand", 13, 0.50, 0.040, _vec(0.055, 0, 0), "trail_arm"),
        BodySpec("club_grip_mass", 19, 0.10, 0.020, _vec(0, 0, 0), "club"),
        BodySpec("club_shaft_mass", 19, 0.16, 0.018, _vec(0, 0, -0.55), "club"),
        BodySpec("clubhead_mass", 19, 0.20, 0.045, _vec(0, 0, -1.08), "club"),
    )
    # MuJoCo requires every moving body, including zero-length gimbal carriers,
    # to own a positive inertia.  The same traceable carrier masses are included
    # in the Lagrange model so this numerical regularization is not engine-only.
    carriers = tuple(
        BodySpec(
            f"joint_carrier_{index}",
            index,
            1.0e-4,
            1.0e-3,
            _vec(0, 0, 0),
            joint.region,
        )
        for index, joint in enumerate(joints)
    )
    bodies = (*physical_bodies, *carriers)
    model = SpatialModel(
        joints=joints,
        bodies=bodies,
        club_dof_indices=np.arange(14, 20, dtype=np.int64),
        lead_hand_joint=8,
        trail_hand_joint=13,
        club_frame_joint=19,
    )
    assert model.nq == 20
    return model


def _rotation(axis: Array, angle: float) -> Array:
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)


def _ancestors(model: SpatialModel, joint_index: int) -> list[int]:
    result: list[int] = []
    cursor = joint_index
    while cursor >= 0:
        result.append(cursor)
        cursor = model.joints[cursor].parent
    result.reverse()
    return result


def forward_kinematics(model: SpatialModel, q: Array) -> Kinematics:
    """Compute world frames and body COM Jacobians."""

    q = np.asarray(q, dtype=np.float64)
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError(f"q must be finite with shape ({model.nq},)")
    positions = np.zeros((model.nq, 3))
    rotations = np.zeros((model.nq, 3, 3))
    axes_world = np.zeros((model.nq, 3))
    for index, joint in enumerate(model.joints):
        parent_rotation = np.eye(3) if joint.parent < 0 else rotations[joint.parent]
        parent_position = np.zeros(3) if joint.parent < 0 else positions[joint.parent]
        origin = parent_position + parent_rotation @ joint.offset_m
        axis_world = parent_rotation @ joint.axis
        axes_world[index] = axis_world
        if joint.kind == "prismatic":
            positions[index] = origin + axis_world * q[index]
            rotations[index] = parent_rotation
        else:
            positions[index] = origin
            rotations[index] = parent_rotation @ _rotation(joint.axis, q[index])

    n_bodies = len(model.bodies)
    body_position = np.zeros((n_bodies, 3))
    body_rotation = np.zeros((n_bodies, 3, 3))
    jv = np.zeros((n_bodies, 3, model.nq))
    jw = np.zeros((n_bodies, 3, model.nq))
    for body_index, body in enumerate(model.bodies):
        rotation = rotations[body.joint]
        com = positions[body.joint] + rotation @ body.com_offset_m
        body_position[body_index] = com
        body_rotation[body_index] = rotation
        for joint_index in _ancestors(model, body.joint):
            joint = model.joints[joint_index]
            if joint.kind == "prismatic":
                jv[body_index, :, joint_index] = axes_world[joint_index]
            else:
                axis = axes_world[joint_index]
                jw[body_index, :, joint_index] = axis
                jv[body_index, :, joint_index] = np.cross(
                    axis, com - positions[joint_index]
                )
    return Kinematics(
        positions, rotations, axes_world, body_position, body_rotation, jv, jw
    )


def mass_matrix(model: SpatialModel, q: Array) -> Array:
    """Assemble the generalized mass matrix from COM Jacobians."""

    kin = forward_kinematics(model, q)
    matrix = np.zeros((model.nq, model.nq))
    for index, body in enumerate(model.bodies):
        inertia_scalar = 0.4 * body.mass_kg * body.radius_m**2
        inertia_world = inertia_scalar * np.eye(3)
        linear = kin.body_linear_jacobian[index]
        angular = kin.body_angular_jacobian[index]
        matrix += body.mass_kg * linear.T @ linear + angular.T @ inertia_world @ angular
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    if eigenvalues[0] <= 1e-10:
        raise RuntimeError(
            f"mass matrix is not positive definite: {eigenvalues[0]:.3e}"
        )
    return matrix


def bias_forces(model: SpatialModel, q: Array, qd: Array, step: float = 1e-6) -> Array:
    """Return Coriolis/centrifugal plus gravity terms by finite differentiation."""

    q = np.asarray(q, dtype=np.float64)
    qd = np.asarray(qd, dtype=np.float64)
    if q.shape != (model.nq,) or qd.shape != (model.nq,):
        raise ValueError("q and qd must match the model dimension")
    derivatives = np.empty((model.nq, model.nq, model.nq))
    for k in range(model.nq):
        delta = np.zeros(model.nq)
        delta[k] = step
        derivatives[k] = (
            mass_matrix(model, q + delta) - mass_matrix(model, q - delta)
        ) / (2.0 * step)
    mass_dot = np.tensordot(qd, derivatives, axes=(0, 0))
    kinetic_gradient = np.einsum("j,k,ijk->i", qd, qd, derivatives)
    coriolis = mass_dot @ qd - 0.5 * kinetic_gradient
    kin = forward_kinematics(model, q)
    gravity_gradient = np.zeros(model.nq)
    for index, body in enumerate(model.bodies):
        gravity_gradient += (
            body.mass_kg * 9.80665 * kin.body_linear_jacobian[index, 2, :]
        )
    return coriolis + gravity_gradient


def prescribed_state(model: SpatialModel, time_s: float) -> tuple[Array, Array, Array]:
    """Return a deterministic nonplanar full-body and club state."""

    if not np.isfinite(time_s) or time_s < 0:
        raise ValueError("time_s must be finite and nonnegative")
    index: Array = np.arange(model.nq, dtype=np.float64)
    amplitude = 0.06 + 0.012 * (index % 5)
    frequency = 5.0 + 0.12 * index
    phase = 0.31 * index
    offset = np.zeros(model.nq)
    offset[14:17] = np.array([0.58, 0.0, 1.18])
    offset[17:20] = np.array([0.20, -0.75, 0.15])
    amplitude[14:17] = np.array([0.025, 0.035, 0.045])
    amplitude[17:20] = np.array([0.12, 0.18, 0.15])
    argument = frequency * time_s + phase
    q = offset + amplitude * np.sin(argument)
    qd = amplitude * frequency * np.cos(argument)
    qdd = -(amplitude * frequency**2) * np.sin(argument)
    return q, qd, qdd


def _point_jacobians(
    model: SpatialModel, kin: Kinematics, joint_index: int, local_point: Array
) -> tuple[Array, Array, Array]:
    point = (
        kin.joint_position_m[joint_index]
        + kin.joint_rotation[joint_index] @ local_point
    )
    jv = np.zeros((3, model.nq))
    jw = np.zeros((3, model.nq))
    for ancestor in _ancestors(model, joint_index):
        joint = model.joints[ancestor]
        if joint.kind == "prismatic":
            jv[:, ancestor] = kin.joint_axis_world[ancestor]
        else:
            axis = kin.joint_axis_world[ancestor]
            jw[:, ancestor] = axis
            jv[:, ancestor] = np.cross(axis, point - kin.joint_position_m[ancestor])
    return point, jv, jw


def evaluate_hand_wrenches(
    model: SpatialModel,
    time_s: float,
    *,
    coincident_hands: bool,
    reverse_geometry: bool = False,
) -> HandWrenchSample:
    """Evaluate the registered two-hand force-couple intervention."""

    q, qd, _ = prescribed_state(model, time_s)
    kin = forward_kinematics(model, q)
    separation = 0.0 if coincident_hands else 0.18
    geometry_sign = -1.0 if reverse_geometry else 1.0
    lead_local = _vec(0, geometry_sign * separation / 2.0, -0.03)
    trail_local = _vec(0, -geometry_sign * separation / 2.0, -0.03)
    lead_position, lead_jv, lead_jw = _point_jacobians(
        model, kin, model.club_frame_joint, lead_local
    )
    trail_position, trail_jv, trail_jw = _point_jacobians(
        model, kin, model.club_frame_joint, trail_local
    )
    reference = 0.5 * (lead_position + trail_position)
    club_rotation = kin.joint_rotation[model.club_frame_joint]
    ramp_coordinate = float(np.clip((time_s - 0.17) / 0.03, 0.0, 1.0))
    activation = ramp_coordinate**2 * (3.0 - 2.0 * ramp_coordinate)
    envelope = activation * math.exp(-(((time_s - 0.215) / 0.050) ** 2))
    common_local = _vec(4.0, 1.5, 3.0) * envelope
    differential_local = _vec(24.0, 1.8, 3.5) * envelope
    lead_force = club_rotation @ (common_local + differential_local)
    trail_force = club_rotation @ (common_local - differential_local)
    moment = np.cross(lead_position - reference, lead_force) + np.cross(
        trail_position - reference, trail_force
    )
    total_force = lead_force + trail_force
    club_wrench = np.concatenate([total_force, moment])
    body_wrench = -club_wrench
    point_velocity = 0.5 * (lead_jv + trail_jv) @ qd
    angular_velocity = 0.5 * (lead_jw + trail_jw) @ qd
    compatible_twist = np.concatenate([point_velocity, angular_velocity])
    power_club = float(club_wrench @ compatible_twist)
    power_body = float(body_wrench @ compatible_twist)
    couple_axis = club_rotation[:, 2]
    force_couple = float(moment @ couple_axis)
    if coincident_hands:
        force_couple = 0.0
    return HandWrenchSample(
        lead_force_n=lead_force,
        trail_force_n=trail_force,
        lead_position_m=lead_position,
        trail_position_m=trail_position,
        reference_position_m=reference,
        club_wrench=club_wrench,
        body_wrench=body_wrench,
        compatible_twist=compatible_twist,
        force_generated_couple_nm=force_couple,
        action_reaction_power_residual_w=power_club + power_body,
    )


def generalized_hand_load(model: SpatialModel, time_s: float) -> Array:
    """Map paired contact forces to the shared generalized coordinates."""

    q, _, _ = prescribed_state(model, time_s)
    kin = forward_kinematics(model, q)
    sample = evaluate_hand_wrenches(model, time_s, coincident_hands=False)
    lead_club, lead_club_jv, _ = _point_jacobians(
        model, kin, model.club_frame_joint, _vec(0, 0.09, -0.03)
    )
    trail_club, trail_club_jv, _ = _point_jacobians(
        model, kin, model.club_frame_joint, _vec(0, -0.09, -0.03)
    )
    lead_hand, lead_hand_jv, _ = _point_jacobians(
        model, kin, model.lead_hand_joint, _vec(0.055, 0, 0)
    )
    trail_hand, trail_hand_jv, _ = _point_jacobians(
        model, kin, model.trail_hand_joint, _vec(0.055, 0, 0)
    )
    del lead_club, trail_club, lead_hand, trail_hand
    return (
        lead_club_jv.T @ sample.lead_force_n
        + trail_club_jv.T @ sample.trail_force_n
        - lead_hand_jv.T @ sample.lead_force_n
        - trail_hand_jv.T @ sample.trail_force_n
    )


def generalized_hand_load_power_residual(model: SpatialModel, time_s: float) -> float:
    """Audit the generalized load against independent point-force power."""

    q, qd, _ = prescribed_state(model, time_s)
    kin = forward_kinematics(model, q)
    sample = evaluate_hand_wrenches(model, time_s, coincident_hands=False)
    point_specs = (
        (model.club_frame_joint, _vec(0, 0.09, -0.03), sample.lead_force_n),
        (model.club_frame_joint, _vec(0, -0.09, -0.03), sample.trail_force_n),
        (model.lead_hand_joint, _vec(0.055, 0, 0), -sample.lead_force_n),
        (model.trail_hand_joint, _vec(0.055, 0, 0), -sample.trail_force_n),
    )
    point_power = 0.0
    for joint_index, local_point, force in point_specs:
        _, linear_jacobian, _ = _point_jacobians(model, kin, joint_index, local_point)
        point_power += float(force @ (linear_jacobian @ qd))
    generalized_power = float(generalized_hand_load(model, time_s) @ qd)
    return generalized_power - point_power


def wrench_reference_power_residual(model: SpatialModel, time_s: float) -> float:
    """Audit wrench/twist power invariance under a reference-point shift."""

    sample = evaluate_hand_wrenches(model, time_s, coincident_hands=False)
    force = sample.club_wrench[:3]
    moment_o = sample.club_wrench[3:]
    velocity_o = sample.compatible_twist[:3]
    angular_velocity = sample.compatible_twist[3:]
    displacement_op = _vec(0.07, -0.04, 0.03)
    # r_P = r_O + displacement_op.  Transport both wrench and twist to P.
    moment_p = moment_o - np.cross(displacement_op, force)
    velocity_p = velocity_o + np.cross(angular_velocity, displacement_op)
    power_o = float(force @ velocity_o + moment_o @ angular_velocity)
    power_p = float(force @ velocity_p + moment_p @ angular_velocity)
    return power_p - power_o


def lagrange_inverse_dynamics(
    model: SpatialModel,
    q: Array,
    qd: Array,
    qdd: Array,
    external_load: Array,
    step: float,
) -> Array:
    """Compute required generalized action with the independent formulation."""

    return mass_matrix(model, q) @ qdd + bias_forces(model, q, qd, step) - external_load


def _mujoco_xml(model: SpatialModel) -> str:
    """Generate an MJCF tree from the same immutable model specification."""

    children: dict[int, list[int]] = {-1: []}
    for index, joint in enumerate(model.joints):
        children.setdefault(joint.parent, []).append(index)
        children.setdefault(index, [])
    bodies_by_joint: dict[int, list[BodySpec]] = {}
    for body in model.bodies:
        bodies_by_joint.setdefault(body.joint, []).append(body)

    def fmt(vector: Array) -> str:
        return " ".join(f"{value:.12g}" for value in vector)

    def emit(index: int, indent: str) -> list[str]:
        joint = model.joints[index]
        joint_type = "hinge" if joint.kind == "revolute" else "slide"
        lines = [
            f'{indent}<body name="joint_{index}_{joint.name}" pos="{fmt(joint.offset_m)}">'
        ]
        lines.append(
            f'{indent}  <joint name="{joint.name}" type="{joint_type}" axis="{fmt(joint.axis)}" damping="0"/>'
        )
        for body in bodies_by_joint.get(index, []):
            if body.name.startswith("joint_carrier_"):
                lines.append(
                    f'{indent}  <geom name="geom_{body.name}" type="sphere" size="{body.radius_m:.12g}" mass="{body.mass_kg:.12g}" contype="0" conaffinity="0"/>'
                )
                continue
            lines.append(
                f'{indent}  <body name="mass_{body.name}" pos="{fmt(body.com_offset_m)}">'
            )
            lines.append(
                f'{indent}    <geom name="geom_{body.name}" type="sphere" size="{body.radius_m:.12g}" mass="{body.mass_kg:.12g}" contype="0" conaffinity="0"/>'
            )
            lines.append(f"{indent}  </body>")
        for child in children[index]:
            lines.extend(emit(child, indent + "  "))
        lines.append(f"{indent}</body>")
        return lines

    world_lines: list[str] = []
    for root in children[-1]:
        world_lines.extend(emit(root, "    "))
    return "\n".join(
        [
            '<mujoco model="proximal_distal_spatial_common_state">',
            '  <compiler angle="radian" inertiafromgeom="true"/>',
            '  <option gravity="0 0 -9.80665" timestep="0.001" integrator="RK4"/>',
            "  <worldbody>",
            *world_lines,
            "  </worldbody>",
            "</mujoco>",
        ]
    )


@lru_cache(maxsize=4)
def _compiled_mujoco_model(model_hash: str, xml: str) -> Any:
    """Compile and cache an immutable MuJoCo model by canonical input hash."""

    import mujoco

    compiled = mujoco.MjModel.from_xml_string(xml)
    if not model_hash or len(model_hash) != 64:
        raise ValueError("model_hash must be a SHA-256 digest")
    return compiled


def mujoco_inverse_dynamics(
    model: SpatialModel, q: Array, qd: Array, qdd: Array, external_load: Array
) -> Array:
    """Evaluate MuJoCo inverse dynamics for the canonical model/state."""

    import mujoco

    mj_model = _compiled_mujoco_model(model.canonical_hash, _mujoco_xml(model))
    if mj_model.nq != model.nq or mj_model.nv != model.nq:
        raise RuntimeError(
            f"MuJoCo dimensions {(mj_model.nq, mj_model.nv)} do not match {model.nq}"
        )
    data = mujoco.MjData(mj_model)
    data.qpos[:] = q
    data.qvel[:] = qd
    data.qacc[:] = qdd
    mujoco.mj_inverse(mj_model, data)
    # ``mj_inverse`` returns inertial + bias generalized force and does not
    # subtract ``qfrc_applied``.  Apply the common external-load convention
    # explicitly so both implementations report actuator action required at the same
    # state: tau_required = M qdd + h - Q_external.
    result = np.asarray(data.qfrc_inverse, dtype=np.float64).copy() - external_load
    if result.shape != (model.nq,) or not np.all(np.isfinite(result)):
        raise RuntimeError("MuJoCo inverse dynamics returned invalid output")
    return result


def mujoco_mass_matrix_and_bias(
    model: SpatialModel, q: Array, qd: Array
) -> tuple[Array, Array]:
    """Return MuJoCo's native full mass matrix and bias-force vector."""

    import mujoco

    mj_model = _compiled_mujoco_model(model.canonical_hash, _mujoco_xml(model))
    data = mujoco.MjData(mj_model)
    data.qpos[:] = q
    data.qvel[:] = qd
    mujoco.mj_forward(mj_model, data)
    matrix = np.empty((model.nq, model.nq), dtype=np.float64)
    mujoco.mj_fullM(mj_model, matrix, data.qM)
    bias = np.asarray(data.qfrc_bias, dtype=np.float64).copy()
    if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(bias)):
        raise RuntimeError("MuJoCo mass/bias audit returned invalid output")
    return matrix, bias


def run_cross_formulation_experiment(
    config: SpatialExperimentConfig = SpatialExperimentConfig(),
) -> CrossFormulationResult:
    """Execute the event-aligned common-state comparison."""

    model = build_spatial_model()
    n_steps = int(round(config.duration_s / config.sample_dt_s))
    time = np.linspace(0.0, config.duration_s, n_steps + 1)
    lagrange = np.empty((time.size, model.nq))
    mujoco_result = np.empty_like(lagrange)
    couples = np.empty(time.size)
    lateral_positions = np.empty(time.size)
    mass_absolute = np.empty(time.size)
    mass_relative = np.empty(time.size)
    bias_absolute = np.empty(time.size)
    bias_relative = np.empty(time.size)
    convention_mismatch = np.empty(time.size)
    for index, sample_time in enumerate(time):
        q, qd, qdd = prescribed_state(model, float(sample_time))
        external = generalized_hand_load(model, float(sample_time))
        lagrange[index] = lagrange_inverse_dynamics(
            model, q, qd, qdd, external, config.derivative_step
        )
        mujoco_result[index] = mujoco_inverse_dynamics(model, q, qd, qdd, external)
        convention_mismatch[index] = np.max(
            np.abs((mujoco_result[index] + external) - lagrange[index])
        ) / max(np.max(np.abs(lagrange[index])), config.tolerance.absolute)
        analytical_mass = mass_matrix(model, q)
        analytical_bias = bias_forces(model, q, qd, config.derivative_step)
        native_mass, native_bias = mujoco_mass_matrix_and_bias(model, q, qd)
        mass_absolute[index] = np.max(np.abs(native_mass - analytical_mass))
        mass_relative[index] = mass_absolute[index] / max(
            np.max(np.abs(analytical_mass)), config.tolerance.absolute
        )
        bias_absolute[index] = np.max(np.abs(native_bias - analytical_bias))
        bias_relative[index] = bias_absolute[index] / max(
            np.max(np.abs(analytical_bias)), config.tolerance.absolute
        )
        sample = evaluate_hand_wrenches(
            model, float(sample_time), coincident_hands=False
        )
        couples[index] = sample.force_generated_couple_nm
        lateral_positions[index] = q[15]
    error = mujoco_result - lagrange
    absolute = float(np.max(np.abs(error)))
    scale = max(float(np.max(np.abs(lagrange))), config.tolerance.absolute)
    relative = absolute / scale
    negative_indices_l = np.flatnonzero(couples < -1e-10)
    # Both formulations receive the same registered input and time grid.  This
    # is an alignment integrity check, not an independently detected event.
    negative_indices_m = np.flatnonzero(couples < -1e-10)
    event_error = (
        abs(float(time[negative_indices_l[0]] - time[negative_indices_m[0]]))
        if negative_indices_l.size and negative_indices_m.size
        else math.inf
    )
    equivalent = (
        absolute <= config.tolerance.absolute
        and relative <= config.tolerance.relative
        and event_error <= config.tolerance.event_time_s
    )
    return CrossFormulationResult(
        formulation_names=("lagrange_christoffel", "mujoco_native_inverse_dynamics"),
        model_hashes=(model.canonical_hash, model.canonical_hash),
        time_s=time,
        inverse_dynamics_lagrange=lagrange,
        inverse_dynamics_mujoco=mujoco_result,
        force_generated_couple_nm=couples,
        out_of_plane_motion_m=float(np.ptp(lateral_positions)),
        max_relative_inverse_dynamics_error=relative,
        max_absolute_generalized_force_error=absolute,
        max_absolute_mass_matrix_error=float(np.max(mass_absolute)),
        max_relative_mass_matrix_error=float(np.max(mass_relative)),
        max_absolute_bias_force_error=float(np.max(bias_absolute)),
        max_relative_bias_force_error=float(np.max(bias_relative)),
        external_load_convention_mismatch_relative_error=float(
            np.max(convention_mismatch)
        ),
        intervention_event_grid_error_s=event_error,
        tolerance=config.tolerance,
        classification="equivalent" if equivalent else "structural_discrepancy",
    )


__all__ = [
    "CrossFormulationResult",
    "HandWrenchSample",
    "SpatialExperimentConfig",
    "SpatialModel",
    "SpatialTolerance",
    "bias_forces",
    "build_spatial_model",
    "evaluate_hand_wrenches",
    "forward_kinematics",
    "generalized_hand_load",
    "generalized_hand_load_power_residual",
    "lagrange_inverse_dynamics",
    "mass_matrix",
    "mujoco_inverse_dynamics",
    "mujoco_mass_matrix_and_bias",
    "prescribed_state",
    "run_cross_formulation_experiment",
    "wrench_reference_power_residual",
]
