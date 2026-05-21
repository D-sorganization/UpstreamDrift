from typing import Any

import mujoco
import numpy as np

from .hip_rotation import InclinedPlaneHipRotationTarget


class LowerBodySimulator:
    """
    Simulates the lower body model using MuJoCo.
    """

    def __init__(self, xml_string: str) -> None:
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

        # Identify indices for sensors and joints
        self.joint_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            for i in range(self.model.njnt)
        ]
        self.actuator_names = [
            mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            for i in range(self.model.nu)
        ]

        self._cache_indices()
        self._polynomial_drivers: dict[int, np.ndarray] = {}
        self.hip_rotation_target: InclinedPlaneHipRotationTarget | None = None

        # Inclined-plane pelvis driver state.
        self._pelvis_driver_target: InclinedPlaneHipRotationTarget | None = None
        self._pelvis_driver_origin: np.ndarray | None = None
        self._pelvis_driver_gains: dict[str, float] = {
            "position_kp": 500.0,
            "position_kd": 50.0,
            "orientation_kp": 200.0,
            "orientation_kd": 20.0,
        }

        # Stability control target (rest pose)
        self.qpos_target: np.ndarray | None = None
        self.kp_stability = 0.0
        self.kd_stability = 0.0

        # Simulation history for scrubbing (QPOS, QVEL, TIME)
        self.history: list[dict[str, Any]] = []
        self.max_history_length = 5000

        mujoco.mj_forward(self.model, self.data)

    def _current_hip_rotation_target_diagnostics(
        self, time_sec: float
    ) -> dict[str, float] | None:
        """Return the configured hip rotation target diagnostics for a sample time."""
        if self.hip_rotation_target is None:
            return None

        return {
            "rotation_deg": self.hip_rotation_target.rotation_degrees_at(time_sec),
            "incline_deg": self.hip_rotation_target.incline_degrees,
        }

    def setup_initial_pose(
        self,
        hip_anterior_tilt: float = 20.0,
        knee_flexion: float = 30.0,
        foot_angle: float = 20.0,
    ) -> None:
        """Set the model to the requested initial pose and compute stability targets.

        All angles are in degrees. The hip and knee angles are applied directly;
        the ankle angles are then solved by a closed-form 2-DOF IK so both feet
        land flat on the ground (world Z-axis of each foot body == (0, 0, 1))
        regardless of the hip/knee configuration. If the required ankle angles
        exceed the ±60° joint limits declared in builder.py the pose is
        infeasible and :class:`ValueError` is raised identifying the offending
        axis.

        Args:
            hip_anterior_tilt: Anterior pelvic tilt applied to both hip_y axes.
            knee_flexion: Bilateral knee flexion.
            foot_angle: External foot rotation applied via hip_z (mirrored per side).

        Raises:
            TypeError: If any argument is not a real number.
            ValueError: If any argument is outside its physiological range:
                -90 <= hip_anterior_tilt <= 90, 0 <= knee_flexion <= 150,
                -90 <= foot_angle <= 90; or if the resulting pose requires an
                ankle angle outside ±60°.
        """
        for name, value in (
            ("hip_anterior_tilt", hip_anterior_tilt),
            ("knee_flexion", knee_flexion),
            ("foot_angle", foot_angle),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
        if not -90.0 <= hip_anterior_tilt <= 90.0:
            raise ValueError(
                f"hip_anterior_tilt must be in [-90, 90], got {hip_anterior_tilt}"
            )
        if not 0.0 <= knee_flexion <= 150.0:
            raise ValueError(f"knee_flexion must be in [0, 150], got {knee_flexion}")
        if not -90.0 <= foot_angle <= 90.0:
            raise ValueError(f"foot_angle must be in [-90, 90], got {foot_angle}")

        self.data.qpos[self.jnt_qpos_idx["r_hip_y"]] = np.radians(hip_anterior_tilt)
        self.data.qpos[self.jnt_qpos_idx["l_hip_y"]] = np.radians(hip_anterior_tilt)

        self.data.qpos[self.jnt_qpos_idx["r_knee"]] = np.radians(knee_flexion)
        self.data.qpos[self.jnt_qpos_idx["l_knee"]] = np.radians(knee_flexion)

        self.data.qpos[self.jnt_qpos_idx["r_hip_z"]] = np.radians(-foot_angle)
        self.data.qpos[self.jnt_qpos_idx["l_hip_z"]] = np.radians(foot_angle)

        # Solve per-side ankle angles so each foot's world Z axis is (0, 0, 1).
        self.data.qpos[self.jnt_qpos_idx["r_ankle_x"]] = 0.0
        self.data.qpos[self.jnt_qpos_idx["r_ankle_y"]] = 0.0
        self.data.qpos[self.jnt_qpos_idx["l_ankle_x"]] = 0.0
        self.data.qpos[self.jnt_qpos_idx["l_ankle_y"]] = 0.0
        mujoco.mj_kinematics(self.model, self.data)

        for side in ("r", "l"):
            qx, qy = self._solve_flat_foot_ankle(side)
            self.data.qpos[self.jnt_qpos_idx[f"{side}_ankle_x"]] = qx
            self.data.qpos[self.jnt_qpos_idx[f"{side}_ankle_y"]] = qy
        mujoco.mj_kinematics(self.model, self.data)

        # Drop the pelvis until feet touch the ground.
        # The foot site is at local z=-0.04 in the foot ellipsoid, so when the
        # site's global Z equals 0.04 the bottom of the ellipsoid touches z=0.
        foot_z = self.data.site_xpos[self.site_ids["r_foot_center"]][2]

        # Free joint first 3 qpos are root x, y, z
        self.data.qpos[2] -= foot_z - 0.04

        mujoco.mj_forward(self.model, self.data)

        # Save as the stability target
        self.qpos_target = self.data.qpos.copy()

    def _solve_flat_foot_ankle(self, side: str) -> tuple[float, float]:
        """Return the (ankle_x, ankle_y) radians that put the foot's world Z at +Z.

        Closed-form 2-DOF decomposition. With the hip and knee joints already
        set and ankles zeroed, ``data.xmat[calf]`` is the calf's world rotation.
        The foot body's world rotation is::

            R_world_foot = R_world_calf @ R_x(qx) @ R_y(qy)

        so setting ``R_world_foot @ (0,0,1) = (0,0,1)`` gives::

            R_x(qx) @ R_y(qy) @ (0,0,1) = R_world_calf.T @ (0,0,1)

        The right-hand side is the calf-local view of world Z, which equals
        the third row of ``data.xmat[calf]``. Expanding the left-hand side::

            R_x(qx) @ R_y(qy) @ (0,0,1) = (sy, -sx*cy, cx*cy)

        with ``sx=sin(qx), cx=cos(qx), sy=sin(qy), cy=cos(qy)``. Solve for
        ``qy = arcsin(sy)`` and then ``qx = atan2(-(-sx*cy), cx*cy)``.

        Raises ``ValueError`` if either solved angle exceeds the ±60° joint
        limit declared in ``builder.py``, identifying the axis and overshoot.
        """
        calf_mat = self.data.xmat[self.body_ids[f"{side}_calf"]].reshape(3, 3)
        target_local_z = calf_mat[2, :]  # world Z expressed in calf-local frame
        sy = float(np.clip(target_local_z[0], -1.0, 1.0))
        qy = float(np.arcsin(sy))
        cy = float(np.cos(qy))
        if abs(cy) < 1e-9:
            qx = 0.0
        else:
            qx = float(np.arctan2(-float(target_local_z[1]), float(target_local_z[2])))

        limit_rad = np.radians(60.0)
        if abs(qy) > limit_rad + 1e-9:
            raise ValueError(
                f"{side}_ankle_y required {np.degrees(qy):.1f}° exceeds ±60° limit; "
                "reduce knee_flexion or hip_anterior_tilt"
            )
        if abs(qx) > limit_rad + 1e-9:
            raise ValueError(
                f"{side}_ankle_x required {np.degrees(qx):.1f}° exceeds ±60° limit; "
                "reduce foot_angle"
            )
        return qx, qy

    def _cache_indices(self) -> None:
        """Pre-compute MuJoCo ids so hot paths don't reach into name lookups.

        Everything the simulator needs to reference by id is looked up exactly
        once, here. `compute_diagnostics`, `analyze_induced_acceleration`,
        `step`, and `inverse_kinematics` all read from these caches instead
        of calling ``mj_name2id`` repeatedly on every frame.
        """
        self.jnt_qpos_idx: dict[str, int] = {}
        self.jnt_dof_idx: dict[str, int] = {}
        self.act_ids: dict[str, int] = {}
        self.site_ids: dict[str, int] = {}
        self.geom_ids: dict[str, int] = {}

        for name in self.joint_names:
            if name is None:
                continue
            jnt_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.jnt_qpos_idx[name] = int(self.model.jnt_qposadr[jnt_id])
            self.jnt_dof_idx[name] = int(self.model.jnt_dofadr[jnt_id])

        for i in range(self.model.nu):
            act_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            if act_name is not None:
                self.act_ids[act_name] = i

        for site in ("r_foot_center", "l_foot_center"):
            self.site_ids[site] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_SITE, site
            )

        for geom in ("floor", "r_foot_geom", "l_foot_geom"):
            self.geom_ids[geom] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_GEOM, geom
            )

        self.body_ids: dict[str, int] = {}
        for body in ("pelvis", "r_calf", "l_calf", "r_foot", "l_foot"):
            self.body_ids[body] = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, body
            )
        self.pelvis_body_id = self.body_ids["pelvis"]

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def set_joint_polynomial(self, joint_name: str, coeffs: list[float]) -> None:
        """Drive the named joint's actuator with a polynomial ``np.polyval``.

        ``joint_name`` may be either the bare joint name (``r_hip_x``) or the
        full actuator name (``act_r_hip_x``). Raises ``ValueError`` if no
        actuator exists for the joint.
        """
        act_name = joint_name if joint_name.startswith("act_") else f"act_{joint_name}"
        act_id = self.act_ids.get(act_name, -1)
        if act_id == -1:
            raise ValueError(f"No actuator found for {joint_name}")
        self._polynomial_drivers[act_id] = np.array(coeffs)

    def configure_hip_rotation_target(
        self,
        duration_sec: float,
        *,
        backswing_degrees: float = 45.0,
        counterclockwise_degrees: float = 90.0,
        incline_degrees: float = 12.0,
        sample_count: int = 181,
    ) -> InclinedPlaneHipRotationTarget:
        """Configure the deterministic inclined-plane golf hip rotation target."""
        self.hip_rotation_target = InclinedPlaneHipRotationTarget(
            duration_sec=duration_sec,
            backswing_degrees=backswing_degrees,
            counterclockwise_degrees=counterclockwise_degrees,
            incline_degrees=incline_degrees,
            sample_count=sample_count,
        )
        return self.hip_rotation_target

    def set_pelvis_inclined_rotation(
        self,
        target: InclinedPlaneHipRotationTarget,
        *,
        position_kp: float = 500.0,
        position_kd: float = 50.0,
        orientation_kp: float = 200.0,
        orientation_kd: float = 20.0,
    ) -> None:
        """Configure an inclined-plane driver that actuates the pelvis free joint.

        Each call to :meth:`step` will apply a PD tracking force/torque to the
        pelvis body via ``data.xfrc_applied``, driving the free joint towards
        ``target.target_quaternion_at(t)`` and the lateral position towards
        ``origin + target.lateral_shift_at(t) * Y``. The current pelvis x,y,z
        position is captured as the origin at the moment of the call.

        Args:
            target: The inclined-plane rotation target to drive.
            position_kp: Proportional gain on pelvis linear position error (N/m).
            position_kd: Derivative gain on pelvis linear velocity (Ns/m).
            orientation_kp: Proportional gain on pelvis rotation error (Nm/rad).
            orientation_kd: Derivative gain on pelvis angular velocity (Nms/rad).

        Raises:
            TypeError: If ``target`` is not an ``InclinedPlaneHipRotationTarget``
                or any gain is not a real number.
            ValueError: If any gain is negative.
        """
        if not isinstance(target, InclinedPlaneHipRotationTarget):
            raise TypeError(
                "target must be an InclinedPlaneHipRotationTarget, got "
                f"{type(target).__name__}"
            )
        for name, value in (
            ("position_kp", position_kp),
            ("position_kd", position_kd),
            ("orientation_kp", orientation_kp),
            ("orientation_kd", orientation_kd),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    f"{name} must be a real number, got {type(value).__name__}"
                )
            if value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value}")

        self._pelvis_driver_target = target
        self._pelvis_driver_origin = self.data.xpos[self.pelvis_body_id].copy()
        self._pelvis_driver_gains = {
            "position_kp": float(position_kp),
            "position_kd": float(position_kd),
            "orientation_kp": float(orientation_kp),
            "orientation_kd": float(orientation_kd),
        }

    def clear_pelvis_inclined_rotation(self) -> None:
        """Remove any configured inclined-plane pelvis driver and zero forces."""
        self._pelvis_driver_target = None
        self._pelvis_driver_origin = None
        if self.data.xfrc_applied.shape[0] > self.pelvis_body_id:
            self.data.xfrc_applied[self.pelvis_body_id] = 0.0

    def _apply_pelvis_inclined_driver(self) -> None:
        """Apply the PD tracking wrench for the configured pelvis driver."""
        target = self._pelvis_driver_target
        origin = self._pelvis_driver_origin
        if target is None or origin is None:
            return

        t = float(self.data.time)
        gains = self._pelvis_driver_gains

        # Linear tracking: origin + lateral_shift_at(t) * +Y.
        desired_pos = origin.copy()
        desired_pos[1] += target.lateral_shift_at(t)
        cur_pos = self.data.xpos[self.pelvis_body_id]
        lin_err = desired_pos - cur_pos
        lin_vel = self.data.qvel[0:3]
        lin_force = gains["position_kp"] * lin_err - gains["position_kd"] * lin_vel

        # Angular tracking: body quat -> target quat -> error quat -> axis-angle.
        target_quat = target.target_quaternion_at(t)
        cur_quat = self.data.xquat[self.pelvis_body_id].copy()

        # err_quat = target * conj(cur)
        cur_conj = np.array([cur_quat[0], -cur_quat[1], -cur_quat[2], -cur_quat[3]])
        err_quat = np.zeros(4)
        mujoco.mju_mulQuat(err_quat, target_quat, cur_conj)

        # Ensure shortest-arc (positive w).
        if err_quat[0] < 0.0:
            err_quat = -err_quat

        # Axis-angle: angle = 2 * acos(w); axis = vec / sin(angle/2).
        w = float(np.clip(err_quat[0], -1.0, 1.0))
        vec = err_quat[1:4]
        vec_norm = float(np.linalg.norm(vec))
        if vec_norm > 1e-9:
            angle = 2.0 * float(np.arccos(w))
            if angle > np.pi:
                angle -= 2.0 * np.pi
            ang_err = (vec / vec_norm) * angle
        else:
            ang_err = np.zeros(3)
        ang_vel = self.data.qvel[3:6]
        torque = gains["orientation_kp"] * ang_err - gains["orientation_kd"] * ang_vel

        self.data.xfrc_applied[self.pelvis_body_id, 0:3] = lin_force
        self.data.xfrc_applied[self.pelvis_body_id, 3:6] = torque

    def apply_hip_rotation_target(
        self, time_sec: float | None = None
    ) -> dict[str, float]:
        """Apply the configured hip target to both hip sockets without per-side duplication."""
        if self.hip_rotation_target is None:
            raise ValueError("No hip rotation target configured")

        sample_time = self.data.time if time_sec is None else time_sec
        rotation_deg = self.hip_rotation_target.rotation_degrees_at(sample_time)
        incline_deg = self.hip_rotation_target.incline_degrees

        for side in ("r", "l"):
            self.data.qpos[self.jnt_qpos_idx[f"{side}_hip_z"]] = np.radians(
                rotation_deg
            )
            self.data.qpos[self.jnt_qpos_idx[f"{side}_hip_x"]] = np.radians(incline_deg)

        mujoco.mj_forward(self.model, self.data)
        return {"rotation_deg": rotation_deg, "incline_deg": incline_deg}

    def compute_zero_torque_counterfactual(self) -> dict[str, np.ndarray]:
        """
        Compute Zero Torque Counterfactual (ZTCF).
        Calculates the accelerations or forces that would occur if no torques were applied,
        but maintaining the current kinematic state.
        In MuJoCo, this is essentially inverse dynamics with zero target acceleration (to find gravity/Coriolis),
        or forward dynamics with zero control to find resulting acceleration.
        We return both for analytical purposes.
        """
        # Save state
        qfrc_applied_orig = self.data.qfrc_applied.copy()
        ctrl_orig = self.data.ctrl.copy()

        # Set controls to zero
        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0

        # Forward dynamics to find induced acceleration without torques
        mujoco.mj_forward(self.model, self.data)
        qacc_zero_torque = self.data.qacc.copy()

        # Inverse dynamics to find forces required to maintain ZERO acceleration
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        qfrc_inverse = self.data.qfrc_inverse.copy()

        # Restore state
        self.data.ctrl[:] = ctrl_orig
        self.data.qfrc_applied[:] = qfrc_applied_orig
        mujoco.mj_forward(self.model, self.data)

        return {"qacc_ztcf": qacc_zero_torque, "qfrc_to_hold_state": qfrc_inverse}

    def compute_pelvis_kinematics(self) -> dict[str, float]:
        """Return world-frame pelvis position and Z-Y-X (yaw/pitch/roll) angles.

        Position components are in metres; roll, pitch and yaw are returned in
        degrees. Rotation is extracted from the body world rotation matrix
        (``data.xmat``), which handles both the free joint and any future
        kinematic chain above the pelvis correctly.

        Returns:
            Dict with keys ``x_forward``, ``y_lateral``, ``z_vertical``,
            ``roll``, ``pitch``, ``yaw``.
        """
        pos = self.data.xpos[self.pelvis_body_id]
        mat = self.data.xmat[self.pelvis_body_id].reshape(3, 3)

        # Z-Y-X (yaw-pitch-roll) Tait-Bryan extraction.
        sy = float(np.sqrt(mat[0, 0] * mat[0, 0] + mat[1, 0] * mat[1, 0]))
        if sy > 1e-6:
            roll = float(np.arctan2(mat[2, 1], mat[2, 2]))
            pitch = float(np.arctan2(-mat[2, 0], sy))
            yaw = float(np.arctan2(mat[1, 0], mat[0, 0]))
        else:
            # Gimbal lock: pitch = ±90°; collapse yaw into roll.
            roll = float(np.arctan2(-mat[1, 2], mat[1, 1]))
            pitch = float(np.arctan2(-mat[2, 0], sy))
            yaw = 0.0

        return {
            "x_forward": float(pos[0]),
            "y_lateral": float(pos[1]),
            "z_vertical": float(pos[2]),
            "roll": float(np.degrees(roll)),
            "pitch": float(np.degrees(pitch)),
            "yaw": float(np.degrees(yaw)),
        }

    def compute_diagnostics(self) -> dict[str, str | float | bool | dict[str, Any]]:
        """Return a telemetry snapshot: pose, tracking error, torques, GRF."""
        mujoco.mj_kinematics(self.model, self.data)
        pelvis_pos = self.data.xpos[self.pelvis_body_id]
        div = bool(np.any(np.isnan(self.data.qpos)) or np.any(np.isnan(self.data.qvel)))
        history_len = len(self.history)

        if div:
            max_err = 0.0
            active_torques = 0.0
            joint_torques: dict[str, float] = {}
            grf = {"right_z": 0.0, "left_z": 0.0}
        else:
            max_err = self._collect_tracking_error()
            active_torques, joint_torques = self._collect_joint_torques()
            grf = self._collect_ground_reaction_forces()

        r_knee_qpos_adr = self.jnt_qpos_idx["r_knee"]

        diagnostics: dict[str, str | float | bool | dict[str, Any]] = {
            "time_sec": float(self.data.time),
            "pelvis_z_m": float(pelvis_pos[2]) if not div else float("nan"),
            "is_diverged": div,
            "max_tracking_err_deg": (
                float(np.degrees(max_err)) if not div else float("nan")
            ),
            "total_applied_torque_nm": (
                float(active_torques) if not div else float("nan")
            ),
            "r_knee_deg": (
                float(np.degrees(self.data.qpos[r_knee_qpos_adr]))
                if not div
                else float("nan")
            ),
            "history_frames": history_len,
            "grf": grf,
            "joint_torques": joint_torques,
        }
        hip_rotation_target = self._current_hip_rotation_target_diagnostics(
            self.data.time
        )
        if hip_rotation_target is not None:
            diagnostics["hip_rotation_target"] = hip_rotation_target
        return diagnostics

    def _collect_tracking_error(self) -> float:
        """Return the largest per-joint qpos error against the stability target."""
        if self.qpos_target is None:
            return 0.0
        max_err = 0.0
        for q_idx in self.jnt_qpos_idx.values():
            err = abs(float(self.data.qpos[q_idx] - self.qpos_target[q_idx]))
            if err > max_err:
                max_err = err
        return max_err

    def _collect_joint_torques(self) -> tuple[float, dict[str, float]]:
        """Return (sum |torque|, per-actuator torque dict) from data.ctrl."""
        active = 0.0
        torques: dict[str, float] = {}
        for joint_name in self.jnt_qpos_idx:
            act_id = self.act_ids.get(f"act_{joint_name}")
            if act_id is None:
                continue
            trq = float(self.data.ctrl[act_id])
            active += abs(trq)
            torques[joint_name] = trq
        return active, torques

    def _collect_ground_reaction_forces(self) -> dict[str, float]:
        """Return vertical GRF on each foot accumulated over current contacts."""
        grf = {"right_z": 0.0, "left_z": 0.0}
        floor = self.geom_ids.get("floor", -1)
        r_foot = self.geom_ids.get("r_foot_geom", -1)
        l_foot = self.geom_ids.get("l_foot_geom", -1)
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            touches_floor = contact.geom1 == floor or contact.geom2 == floor
            if not touches_floor:
                continue
            if contact.geom1 == r_foot or contact.geom2 == r_foot:
                grf["right_z"] += float(self.data.efc_force[contact.efc_address])
            elif contact.geom1 == l_foot or contact.geom2 == l_foot:
                grf["left_z"] += float(self.data.efc_force[contact.efc_address])
        return grf

    def analyze_induced_acceleration(
        self, actuator_name: str, torque_value: float = 1.0
    ) -> dict[str, float]:
        """
        Perform Induced Acceleration Analysis (IAA) at the current state.
        This isolates the instantaneous acceleration effect of a specific torque
        applied to a specific actuator on the root pelvis body.

        Returns the induced linear and angular acceleration of the pelvis.
        """
        act_id = self.act_ids.get(actuator_name, -1)
        if act_id == -1:
            raise ValueError(f"No actuator found for {actuator_name}")

        # 1. Baseline acceleration (zero torque)
        ctrl_orig = self.data.ctrl.copy()
        qfrc_applied_orig = self.data.qfrc_applied.copy()

        self.data.ctrl[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        # Free joint accelerations are the first 6 elements of qacc (3 linear, 3 angular)
        qacc_base = self.data.qacc[:6].copy()

        # 2. Perturbed acceleration
        self.data.ctrl[act_id] = torque_value
        mujoco.mj_forward(self.model, self.data)
        qacc_pert = self.data.qacc[:6].copy()

        # 3. Induced acceleration
        induced_acc = qacc_pert - qacc_base

        # Restore
        self.data.ctrl[:] = ctrl_orig
        self.data.qfrc_applied[:] = qfrc_applied_orig
        mujoco.mj_forward(self.model, self.data)

        return {
            "forward_accel": induced_acc[0],
            "lateral_accel": induced_acc[1],
            "vertical_accel": induced_acc[2],
            "roll_accel": induced_acc[3],
            "pitch_accel": induced_acc[4],
            "yaw_accel": induced_acc[5],
        }

    def inverse_kinematics(
        self,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        max_iters: int = 100,
        tol: float = 1e-4,
    ) -> bool:
        """Solve IK to move the pelvis to a target pose using Damped Least Squares.

        The pelvis free joint is snapped to ``target_pos``/``target_quat``, then
        leg joints are optimised so the foot sites meet their nominal ground
        positions. On success the new qpos becomes the new stability target.
        On failure the prior state is restored and the target is left unchanged.

        Args:
            target_pos: World-frame target position (3,) for the pelvis.
            target_quat: World-frame target quaternion (4,) as (w, x, y, z).
            max_iters: Maximum DLS iterations.
            tol: Convergence tolerance on the stacked foot error norm.

        Returns:
            True if converged within ``max_iters``, False otherwise.

        Raises:
            TypeError: If targets are not array-like.
            ValueError: If target shapes are wrong or max_iters / tol invalid.
        """
        target_pos_arr = np.asarray(target_pos, dtype=float)
        target_quat_arr = np.asarray(target_quat, dtype=float)
        if target_pos_arr.shape != (3,):
            raise ValueError(
                f"target_pos must have shape (3,), got {target_pos_arr.shape}"
            )
        if target_quat_arr.shape != (4,):
            raise ValueError(
                f"target_quat must have shape (4,), got {target_quat_arr.shape}"
            )
        if max_iters <= 0:
            raise ValueError(f"max_iters must be positive, got {max_iters}")
        if tol <= 0.0:
            raise ValueError(f"tol must be positive, got {tol}")

        # Snapshot state so we can restore on failure.
        qpos_saved = self.data.qpos.copy()
        qvel_saved = self.data.qvel.copy()

        # Free joint QPOS layout is [x, y, z, qw, qx, qy, qz].
        self.data.qpos[0:3] = target_pos_arr
        self.data.qpos[3:7] = target_quat_arr

        r_foot_id = self.site_ids["r_foot_center"]
        l_foot_id = self.site_ids["l_foot_center"]

        # Nominal foot site targets (match the rest-pose builder placement).
        r_target = np.array([0.05, -0.15, -0.03])
        l_target = np.array([0.05, 0.15, -0.03])

        jac_r = np.zeros((3, self.model.nv))
        jac_l = np.zeros((3, self.model.nv))

        alpha = 0.1  # DLS damping.

        for _ in range(max_iters):
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)

            r_err = r_target - self.data.site_xpos[r_foot_id]
            l_err = l_target - self.data.site_xpos[l_foot_id]
            err = np.concatenate((r_err, l_err))

            if np.linalg.norm(err) < tol:
                # Success: lock the new configuration as the stability target.
                self.qpos_target = self.data.qpos.copy()
                return True

            mujoco.mj_jacSite(self.model, self.data, jac_r, None, r_foot_id)
            mujoco.mj_jacSite(self.model, self.data, jac_l, None, l_foot_id)
            J = np.vstack((jac_r, jac_l))
            # Zero the root DOFs (first 6 in qvel); the free root is pinned.
            J[:, 0:6] = 0.0

            J_T = J.T
            dq = J_T @ np.linalg.inv(J @ J_T + alpha * np.eye(6)) @ err
            mujoco.mj_integratePos(self.model, self.data.qpos, dq, 1.0)

        # Failure: restore the snapshot and leave qpos_target untouched.
        self.data.qpos[:] = qpos_saved
        self.data.qvel[:] = qvel_saved
        mujoco.mj_forward(self.model, self.data)
        return False

    def step(self) -> None:
        """Advance the simulation by one timestep, applying polynomial controls and basic stability."""
        t = self.data.time
        if self.hip_rotation_target is not None:
            self.apply_hip_rotation_target(t)

        # Apply the inclined-plane pelvis driver before mj_step so its wrench
        # is integrated along with every other external force this tick.
        if self._pelvis_driver_target is not None:
            self._apply_pelvis_inclined_driver()

        # PD stability loop to hold the target posture. Everything here reads
        # from pre-cached indices; no mj_name2id calls in the hot path.
        if self.qpos_target is not None:
            for joint_name, q_idx in self.jnt_qpos_idx.items():
                act_id = self.act_ids.get(f"act_{joint_name}")
                if act_id is None or act_id in self._polynomial_drivers:
                    continue
                v_idx = self.jnt_dof_idx[joint_name]
                q_err = self.data.qpos[q_idx] - self.qpos_target[q_idx]
                v_err = self.data.qvel[v_idx]
                self.data.ctrl[act_id] = (
                    -self.kp_stability * q_err - self.kd_stability * v_err
                )

        # Overlay polynomials on top
        for act_id, coeffs in self._polynomial_drivers.items():
            ctrl_val = np.polyval(coeffs, t)
            self.data.ctrl[act_id] = ctrl_val

        mujoco.mj_step(self.model, self.data)

        # Record state
        self.history.append(
            {
                "time": self.data.time,
                "qpos": self.data.qpos.copy(),
                "qvel": self.data.qvel.copy(),
                "ctrl": self.data.ctrl.copy(),
                "hip_rotation_target": self._current_hip_rotation_target_diagnostics(
                    self.data.time
                ),
            }
        )

        if len(self.history) > self.max_history_length:
            self.history.pop(0)

    def restore_frame(self, index: int) -> None:
        """Restores the simulator completely to a cached history frame state."""
        if not self.history or index < 0 or index >= len(self.history):
            return

        frame = self.history[index]
        self.data.time = frame["time"]
        self.data.qpos[:] = frame["qpos"]
        self.data.qvel[:] = frame["qvel"]
        self.data.ctrl[:] = frame["ctrl"]
        mujoco.mj_forward(self.model, self.data)

    def get_history_diagnostics(self, index: int) -> dict[str, Any]:
        """Return playback diagnostics for a cached history frame."""
        if not self.history or index < 0 or index >= len(self.history):
            raise IndexError("History frame index out of range")

        frame = self.history[index]
        diagnostics: dict[str, Any] = {
            "time_sec": float(frame["time"]),
            "hip_rotation_target": frame["hip_rotation_target"],
        }
        return diagnostics

    def clear_history(self) -> None:
        """Empties execution memory logs."""
        self.history.clear()
