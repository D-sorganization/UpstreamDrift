"""Native forward-dynamics adapters for the spatial contact experiment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    CanonicalSpatialState,
    SpatialContactParameters,
    canonical_spatial_state_digest,
    default_spatial_state,
    rotation_matrix_from_quaternion,
)


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class EngineIdentity:
    """Auditable native engine identity."""

    library: str
    version: str
    native_forward_dynamics: bool
    solver: str
    coordinate_count: int
    velocity_count: int


@dataclass(frozen=True)
class AppliedSpatialForces:
    """World-frame forces applied during one forward-dynamics evaluation."""

    hand_forces: FloatArray
    club_points: FloatArray
    club_forces: FloatArray

    def __post_init__(self) -> None:
        for name in ("hand_forces", "club_points", "club_forces"):
            value = np.asarray(getattr(self, name), dtype=float)
            if value.shape != (2, 3) or np.any(~np.isfinite(value)):
                raise ValueError(f"{name} must have finite shape (2, 3)")


@runtime_checkable
class SpatialForwardAdapter(Protocol):
    """Small segregated interface needed by the common experiment runner."""

    engine_identity: EngineIdentity
    initial_state_digest: str
    model_digest: str

    def canonical_state(self) -> CanonicalSpatialState:
        """Return the current achieved state in the shared convention."""

    def step(self, applied: AppliedSpatialForces, time_step: float) -> None:
        """Evaluate native forward dynamics and advance one step."""

    def native_mechanical_energy(self) -> float:
        """Return engine-native rigid-body kinetic plus potential energy."""


class MuJoCoSpatialForwardAdapter:
    """MuJoCo forward-dynamics realization of the common reduced model."""

    def __init__(
        self,
        params: SpatialContactParameters,
        initial_state: CanonicalSpatialState | None = None,
    ) -> None:
        try:
            import mujoco
        except ImportError as exc:  # pragma: no cover - optional dependency gate
            raise RuntimeError("MuJoCo is required for this adapter") from exc
        if not hasattr(mujoco, "MjModel") or not hasattr(mujoco, "mj_forward"):
            raise RuntimeError("the imported mujoco module is not the native engine")
        self._mujoco = mujoco
        self._params = params
        self.model_digest = params.model_digest()
        self._initial_state = initial_state or default_spatial_state(params)
        self.initial_state_digest = canonical_spatial_state_digest(self._initial_state)
        self._model = mujoco.MjModel.from_xml_string(_mujoco_xml(params))
        self._data = mujoco.MjData(self._model)
        self._lead_body = self._model.body("lead_hand").id
        self._trail_body = self._model.body("trail_hand").id
        self._club_body = self._model.body("club").id
        self._initialize_state(self._initial_state)
        self.engine_identity = EngineIdentity(
            library="mujoco",
            version=str(mujoco.__version__),
            native_forward_dynamics=True,
            solver="mj_forward continuous-time acceleration with shared semi-implicit step",
            coordinate_count=int(self._model.nq),
            velocity_count=int(self._model.nv),
        )

    def _initialize_state(self, state: CanonicalSpatialState) -> None:
        self._data.qpos[:3] = state.hand_positions[0]
        self._data.qpos[3:6] = state.hand_positions[1]
        self._data.qpos[6:9] = state.club_position
        self._data.qpos[9:13] = state.club_quaternion_wxyz
        self._data.qvel[:3] = state.hand_velocities[0]
        self._data.qvel[3:6] = state.hand_velocities[1]
        self._data.qvel[6:9] = state.club_linear_velocity
        rotation = rotation_matrix_from_quaternion(state.club_quaternion_wxyz)
        self._data.qvel[9:12] = rotation.T @ state.club_angular_velocity
        self._mujoco.mj_forward(self._model, self._data)

    def _body_velocity(self, body_id: int) -> tuple[FloatArray, FloatArray]:
        value = np.zeros(6, dtype=float)
        self._mujoco.mj_objectVelocity(
            self._model,
            self._data,
            self._mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            value,
            0,
        )
        return value[3:].copy(), value[:3].copy()

    def canonical_state(self) -> CanonicalSpatialState:
        self._mujoco.mj_forward(self._model, self._data)
        lead_velocity, _ = self._body_velocity(self._lead_body)
        trail_velocity, _ = self._body_velocity(self._trail_body)
        club_linear, club_angular = self._body_velocity(self._club_body)
        return CanonicalSpatialState(
            hand_positions=np.vstack(
                [
                    self._data.xpos[self._lead_body],
                    self._data.xpos[self._trail_body],
                ]
            ).copy(),
            hand_velocities=np.vstack([lead_velocity, trail_velocity]),
            club_position=self._data.xpos[self._club_body].copy(),
            club_quaternion_wxyz=self._data.xquat[self._club_body].copy(),
            club_linear_velocity=club_linear,
            club_angular_velocity=club_angular,
        )

    def step(self, applied: AppliedSpatialForces, time_step: float) -> None:
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive")
        self._data.qfrc_applied[:] = 0.0
        for index, body_id in enumerate((self._lead_body, self._trail_body)):
            point = self._data.xpos[body_id]
            self._mujoco.mj_applyFT(
                self._model,
                self._data,
                applied.hand_forces[index],
                np.zeros(3),
                point,
                body_id,
                self._data.qfrc_applied,
            )
        for point, force in zip(applied.club_points, applied.club_forces, strict=True):
            self._mujoco.mj_applyFT(
                self._model,
                self._data,
                force,
                np.zeros(3),
                point,
                self._club_body,
                self._data.qfrc_applied,
            )
        self._mujoco.mj_forward(self._model, self._data)
        self._data.qvel[:] += time_step * self._data.qacc
        self._mujoco.mj_integratePos(
            self._model, self._data.qpos, self._data.qvel, time_step
        )
        self._mujoco.mj_normalizeQuat(self._model, self._data.qpos)
        if np.any(~np.isfinite(self._data.qpos)) or np.any(
            ~np.isfinite(self._data.qvel)
        ):
            raise RuntimeError("MuJoCo forward integration produced invalid state")

    def native_mechanical_energy(self) -> float:
        self._mujoco.mj_forward(self._model, self._data)
        self._mujoco.mj_energyPos(self._model, self._data)
        self._mujoco.mj_energyVel(self._model, self._data)
        return float(np.sum(self._data.energy))


class PinocchioSpatialForwardAdapter:
    """Pinocchio ABA realization of the common reduced model."""

    def __init__(
        self,
        params: SpatialContactParameters,
        initial_state: CanonicalSpatialState | None = None,
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - optional dependency gate
            raise RuntimeError("Pinocchio is required for this adapter") from exc
        required = ("Model", "JointModelTranslation", "JointModelFreeFlyer", "aba")
        version = getattr(pin, "__version__", None)
        version_major = int(version.split(".")[0]) if isinstance(version, str) else 0
        if version_major < 2 or any(not hasattr(pin, name) for name in required):
            raise RuntimeError(
                "the imported pinocchio module is not the robotics engine"
            )
        self._pin = pin
        self._params = params
        self.model_digest = params.model_digest()
        self._initial_state = initial_state or default_spatial_state(params)
        self.initial_state_digest = canonical_spatial_state_digest(self._initial_state)
        self._model, self._lead_joint, self._trail_joint, self._club_joint = (
            _build_pinocchio_model(pin, params)
        )
        self._data = self._model.createData()
        self._q = pin.neutral(self._model)
        self._v = np.zeros(self._model.nv)
        self._initialize_state(self._initial_state)
        self.engine_identity = EngineIdentity(
            library="pinocchio",
            version=str(pin.__version__),
            native_forward_dynamics=True,
            solver="Pinocchio articulated-body algorithm with shared semi-implicit step",
            coordinate_count=int(self._model.nq),
            velocity_count=int(self._model.nv),
        )

    def _initialize_state(self, state: CanonicalSpatialState) -> None:
        for hand_index, (joint_id, position) in enumerate(
            (
                (self._lead_joint, state.hand_positions[0]),
                (self._trail_joint, state.hand_positions[1]),
            )
        ):
            joint = self._model.joints[joint_id]
            self._q[joint.idx_q : joint.idx_q + 3] = position
            self._v[joint.idx_v : joint.idx_v + 3] = state.hand_velocities[hand_index]
        club = self._model.joints[self._club_joint]
        self._q[club.idx_q : club.idx_q + 3] = state.club_position
        self._q[club.idx_q + 3 : club.idx_q + 7] = np.roll(
            state.club_quaternion_wxyz, -1
        )
        rotation = rotation_matrix_from_quaternion(state.club_quaternion_wxyz)
        self._v[club.idx_v : club.idx_v + 3] = rotation.T @ state.club_linear_velocity
        self._v[club.idx_v + 3 : club.idx_v + 6] = (
            rotation.T @ state.club_angular_velocity
        )
        self._forward_kinematics()

    def _forward_kinematics(self) -> None:
        self._pin.forwardKinematics(self._model, self._data, self._q, self._v)
        self._pin.updateFramePlacements(self._model, self._data)

    def _joint_motion(self, joint_id: int) -> tuple[FloatArray, FloatArray]:
        motion = self._pin.getVelocity(
            self._model,
            self._data,
            joint_id,
            self._pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return np.asarray(motion.linear).copy(), np.asarray(motion.angular).copy()

    def canonical_state(self) -> CanonicalSpatialState:
        self._forward_kinematics()
        lead_linear, _ = self._joint_motion(self._lead_joint)
        trail_linear, _ = self._joint_motion(self._trail_joint)
        club_linear, club_angular = self._joint_motion(self._club_joint)
        placement = self._data.oMi[self._club_joint]
        quaternion_xyzw = np.asarray(self._pin.Quaternion(placement.rotation).coeffs())
        quaternion_wxyz = np.roll(quaternion_xyzw, 1)
        return CanonicalSpatialState(
            hand_positions=np.vstack(
                [
                    self._data.oMi[self._lead_joint].translation,
                    self._data.oMi[self._trail_joint].translation,
                ]
            ).copy(),
            hand_velocities=np.vstack([lead_linear, trail_linear]),
            club_position=np.asarray(placement.translation).copy(),
            club_quaternion_wxyz=quaternion_wxyz,
            club_linear_velocity=club_linear,
            club_angular_velocity=club_angular,
        )

    def _external_forces(self, applied: AppliedSpatialForces) -> list[Any]:
        pin = self._pin
        external = [pin.Force.Zero() for _ in range(self._model.njoints)]
        for index, joint_id in enumerate((self._lead_joint, self._trail_joint)):
            rotation = self._data.oMi[joint_id].rotation
            local_force = rotation.T @ applied.hand_forces[index]
            external[joint_id] = pin.Force(local_force, np.zeros(3))
        placement = self._data.oMi[self._club_joint]
        world_force = np.sum(applied.club_forces, axis=0)
        world_moment = np.sum(
            np.cross(applied.club_points - placement.translation, applied.club_forces),
            axis=0,
        )
        external[self._club_joint] = pin.Force(
            placement.rotation.T @ world_force,
            placement.rotation.T @ world_moment,
        )
        return external

    def step(self, applied: AppliedSpatialForces, time_step: float) -> None:
        if not np.isfinite(time_step) or time_step <= 0.0:
            raise ValueError("time_step must be finite and positive")
        self._forward_kinematics()
        acceleration = self._pin.aba(
            self._model,
            self._data,
            self._q,
            self._v,
            np.zeros(self._model.nv),
            self._external_forces(applied),
        )
        self._v = self._v + time_step * np.asarray(acceleration)
        self._q = self._pin.integrate(self._model, self._q, time_step * self._v)
        if np.any(~np.isfinite(self._q)) or np.any(~np.isfinite(self._v)):
            raise RuntimeError("Pinocchio forward integration produced invalid state")

    def native_mechanical_energy(self) -> float:
        kinetic = self._pin.computeKineticEnergy(
            self._model, self._data, self._q, self._v
        )
        potential = self._pin.computePotentialEnergy(self._model, self._data, self._q)
        return float(kinetic + potential)


def _mujoco_xml(params: SpatialContactParameters) -> str:
    gravity = " ".join(f"{value:.17g}" for value in params.gravity)
    hand_inertia = " ".join([f"{params.hand_inertia:.17g}"] * 3)
    club_inertia = " ".join(f"{value:.17g}" for value in params.club_inertia)
    return f"""
<mujoco model="spatial_two_hand_forward_contact">
  <compiler angle="radian" inertiafromgeom="false"/>
  <option gravity="{gravity}" timestep="{params.time_step:.17g}"/>
  <worldbody>
    <body name="lead_hand">
      <joint name="lead_x" type="slide" axis="1 0 0"/>
      <joint name="lead_y" type="slide" axis="0 1 0"/>
      <joint name="lead_z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="{params.hand_mass:.17g}" diaginertia="{hand_inertia}"/>
    </body>
    <body name="trail_hand">
      <joint name="trail_x" type="slide" axis="1 0 0"/>
      <joint name="trail_y" type="slide" axis="0 1 0"/>
      <joint name="trail_z" type="slide" axis="0 0 1"/>
      <inertial pos="0 0 0" mass="{params.hand_mass:.17g}" diaginertia="{hand_inertia}"/>
    </body>
    <body name="club">
      <freejoint name="club_free"/>
      <inertial pos="0 0 0" mass="{params.club_mass:.17g}" diaginertia="{club_inertia}"/>
    </body>
  </worldbody>
</mujoco>
"""


def _build_pinocchio_model(
    pin: Any, params: SpatialContactParameters
) -> tuple[Any, int, int, int]:
    model = pin.Model()
    model.gravity.linear = np.asarray(params.gravity, dtype=float)
    identity = pin.SE3.Identity()
    hand_inertia = pin.Inertia(
        params.hand_mass,
        np.zeros(3),
        np.eye(3) * params.hand_inertia,
    )
    lead = model.addJoint(0, pin.JointModelTranslation(), identity, "lead_hand")
    model.appendBodyToJoint(lead, hand_inertia, identity)
    trail = model.addJoint(0, pin.JointModelTranslation(), identity, "trail_hand")
    model.appendBodyToJoint(trail, hand_inertia, identity)
    club = model.addJoint(0, pin.JointModelFreeFlyer(), identity, "club")
    model.appendBodyToJoint(
        club,
        pin.Inertia(
            params.club_mass,
            np.zeros(3),
            np.diag(np.asarray(params.club_inertia, dtype=float)),
        ),
        identity,
    )
    return model, lead, trail, club


def make_spatial_forward_adapter(
    engine: str,
    params: SpatialContactParameters,
    initial_state: CanonicalSpatialState | None = None,
) -> SpatialForwardAdapter:
    """Construct one actual engine adapter; reject unknown names and stubs."""

    if engine == "mujoco":
        return MuJoCoSpatialForwardAdapter(params, initial_state)
    if engine == "pinocchio":
        return PinocchioSpatialForwardAdapter(params, initial_state)
    raise ValueError(f"unsupported spatial forward engine: {engine}")


__all__ = [
    "AppliedSpatialForces",
    "EngineIdentity",
    "MuJoCoSpatialForwardAdapter",
    "PinocchioSpatialForwardAdapter",
    "SpatialForwardAdapter",
    "make_spatial_forward_adapter",
]
