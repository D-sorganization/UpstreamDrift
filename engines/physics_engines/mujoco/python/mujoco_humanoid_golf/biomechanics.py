"""Biomechanical analysis module for golf swing simulations.

This module provides comprehensive force, torque, and kinematic analysis
for golf swing models. It extracts data from MuJoCo simulations and computes
derived biomechanical quantities.
"""

from __future__ import annotations

import logging
from typing import cast

import mujoco
import numpy as np
from shared.python.biomechanics_data import BiomechanicalData


class BiomechanicalAnalyzer:
    """Analyzes MuJoCo simulation data for biomechanical insights.

    This class extracts forces, torques, kinematics, and energetics
    from a MuJoCo model and data structure.

    Attributes:
        _prev_club_vel: Previous club velocity for acceleration calculation.
            Initialized to None and updated during state extraction.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Initialize analyzer with MuJoCo model and data.

        Args:
            model: MuJoCo model structure
            data: MuJoCo data structure (will be read, not modified)
        """
        self.model = model
        self.data = data

        # Previous club velocity for acceleration calculation
        self._prev_club_vel: np.ndarray | None = None

        # Find important body IDs
        self.club_head_id = self._find_body_id("club_head")
        self.left_foot_id = self._find_body_id("left_foot")
        self.right_foot_id = self._find_body_id("right_foot")

        # Cache for previous velocities (for acceleration computation)
        self.prev_qvel: np.ndarray | None = None
        self.prev_time: float = 0.0

        # Pre-allocate Jacobian buffers to avoid re-allocation every frame
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Check if MuJoCo supports shaped arrays (optimization)
        # MuJoCo 3.3+ may require reshaped arrays, while older versions might work
        # with both or require flat arrays. We determine the capability once.
        try:
            # Try with shaped arrays on body 0 (world) which always exists
            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, 0)
            self._use_shaped_jac = True
            self._jacp_flat = None
            self._jacr_flat = None
        except TypeError:
            # Fallback to flat arrays
            logging.getLogger(__name__).debug(
                "MuJoCo version requires flat array format for mj_jacBody"
            )
            self._use_shaped_jac = False
            self._jacp_flat = np.zeros(3 * self.model.nv)
            self._jacr_flat = np.zeros(3 * self.model.nv)

        # Import locally to avoid circular import at module level
        from engines.physics_engines.mujoco.python.mujoco_humanoid_golf.rigid_body_dynamics.induced_acceleration import (  # noqa: E501
            MuJoCoInducedAccelerationAnalyzer,
        )

        self.induced_analyzer = MuJoCoInducedAccelerationAnalyzer(model, data)

    def _find_body_id(self, name_pattern: str) -> int | None:
        """Find body ID by name pattern (case-insensitive, partial match)."""
        for i in range(self.model.nbody):
            body_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, i)
            if body_name and name_pattern.lower() in body_name.lower():
                return i
        return None

    def _find_geom_id(self, name_pattern: str) -> int | None:
        """Find geom ID by name pattern (case-insensitive, partial match)."""
        for i in range(self.model.ngeom):
            geom_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if geom_name and name_pattern.lower() in geom_name.lower():
                return i
        return None

    def compute_joint_accelerations(self) -> np.ndarray:
        """Compute joint accelerations using finite differences.

        Returns:
            Array of joint accelerations [rad/s^2 or m/s^2]
        """
        if self.prev_qvel is None:
            # First call - return zeros
            self.prev_qvel = self.data.qvel.copy()
            self.prev_time = self.data.time
            return np.zeros_like(self.data.qvel)

        dt = self.data.time - self.prev_time
        if dt <= 0:
            return np.zeros_like(self.data.qvel)

        qacc = (self.data.qvel - self.prev_qvel) / dt

        # Update cache
        self.prev_qvel = self.data.qvel.copy()
        self.prev_time = self.data.time

        # PERF-004: qacc is already a numpy array, no need to copy
        return cast(np.ndarray, qacc.astype(np.float64, copy=False))

    def compute_induced_acceleration(self, source_name: str) -> np.ndarray:
        """Compute induced acceleration for a specific source.

        Legacy method, delegates to new analyzer.

        Args:
            source_name: "gravity" or "actuator"

        Returns:
            Acceleration array
        """
        comps = self.induced_analyzer.compute_components()
        if source_name == "gravity":
            return cast(np.ndarray, comps["gravity"])
        elif source_name == "actuator":
            return cast(np.ndarray, comps["control"])
        return np.zeros(self.model.nv)

    def compute_counterfactuals(self) -> dict[str, np.ndarray]:
        """Compute instantaneous counterfactuals."""
        # Delegate to a simple implementation or use what we had
        # The logic was simple enough to keep or we can move it to Analyzer
        # For now, keep as is to minimize diff, but use the same logic as before.
        # ZTCF: Acceleration if tau=0.
        # M*a + C + g = 0  => M*a = -(C+g)

        qvel_save = self.data.qvel.copy()
        qacc_save = self.data.qacc.copy()

        try:
            # 1. Compute C+g (Inverse Dynamics with a=0)
            self.data.qacc[:] = 0
            res = np.zeros(self.model.nv)
            mujoco.mj_rne(self.model, self.data, 1, res)
            c_plus_g = res.copy()

            # Solve M * a_ztcf = -(C+g)
            rhs = -c_plus_g
            rhs_row = rhs.reshape(1, self.model.nv)
            x_row = np.zeros((1, self.model.nv))
            mujoco.mj_solveM(self.model, self.data, x_row, rhs_row)
            a_ztcf = x_row.flatten()

            # ZVCF: Torque/Forces if v=0.
            self.data.qvel[:] = 0
            self.data.qacc[:] = 0
            res = np.zeros(self.model.nv)
            mujoco.mj_rne(self.model, self.data, 1, res)
            tau_zvcf = res.copy()  # This is g(q)

        finally:
            # Restore
            self.data.qvel[:] = qvel_save
            self.data.qacc[:] = qacc_save

        return {
            "ztcf_accel": cast(np.ndarray, a_ztcf),
            "zvcf_torque": cast(np.ndarray, tau_zvcf),
        }

    def get_club_head_data(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float]:
        """Get club head position, velocity, and speed.

        Returns:
            Tuple of (position [3], velocity [3], speed [m/s])
            Returns (None, None, 0.0) if club head not found
        """
        if self.club_head_id is None:
            return None, None, 0.0

        # Get position
        pos = self.data.xpos[self.club_head_id].copy()

        # Get velocity (compute from Jacobian)
        if self._use_shaped_jac:
            mujoco.mj_jacBody(
                self.model, self.data, self._jacp, self._jacr, self.club_head_id
            )
            jacp = self._jacp
        else:
            mujoco.mj_jacBody(
                self.model,
                self.data,
                self._jacp_flat,
                self._jacr_flat,
                self.club_head_id,
            )
            if self._jacp_flat is not None:
                jacp = self._jacp_flat.reshape(3, self.model.nv)
            else:
                jacp = np.zeros((3, self.model.nv))

        vel = jacp @ self.data.qvel
        speed = float(np.linalg.norm(vel))

        return pos, vel, speed

    def get_ground_reaction_forces(
        self,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        """Get ground reaction forces for left and right feet.

        Returns:
            Tuple of (left_foot_force [3], right_foot_force [3])
            Returns (None, None) if feet not found or no contacts
        """
        left_force = None
        right_force = None

        # Sum up contact forces for each foot
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get the geom IDs involved in this contact
            geom1 = contact.geom1
            geom2 = contact.geom2

            # Check which body each geom belongs to
            body1 = self.model.geom_bodyid[geom1]
            body2 = self.model.geom_bodyid[geom2]

            # Compute contact force
            c_array = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, i, c_array)
            contact_force = c_array[:3]  # First 3 components are force

            # Assign to appropriate foot
            if self.left_foot_id is not None and (self.left_foot_id in (body1, body2)):
                if left_force is None:
                    left_force = contact_force.copy()
                else:
                    left_force += contact_force

            if self.right_foot_id is not None and (
                self.right_foot_id in (body1, body2)
            ):
                if right_force is None:
                    right_force = contact_force.copy()
                else:
                    right_force += contact_force

        return left_force, right_force

    def get_center_of_mass(self) -> tuple[np.ndarray, np.ndarray]:
        """Get center of mass position and velocity.

        Returns:
            Tuple of (position [3], velocity [3])
        """
        com_pos = self.data.subtree_com[0].copy()  # World body COM
        mujoco.mj_subtreeVel(self.model, self.data)
        com_vel = self.data.subtree_linvel[0].copy()
        return com_pos, com_vel

    def compute_energies(self) -> tuple[float, float, float]:
        """Compute kinetic, potential, and total energy.

        Returns:
            Tuple of (kinetic_energy [J], potential_energy [J], total_energy [J])
        """
        ke = float(self.data.energy[0])  # Kinetic energy
        pe = float(self.data.energy[1])  # Potential energy
        total = ke + pe
        return ke, pe, total

    def get_actuator_powers(self) -> np.ndarray:
        """Compute mechanical power for each actuator.

        Returns:
            Array of actuator powers [W]
        """
        powers = np.zeros(self.model.nu)
        for i in range(self.model.nu):
            joint_id = self.model.actuator_trnid[i, 0]
            actuator_force = self.data.actuator_force[i]
            if joint_id >= 0 and joint_id < self.model.nv:
                joint_velocity = self.data.qvel[joint_id]
                powers[i] = actuator_force * joint_velocity
        return powers

    def extract_full_state(
        self,
        selected_actuator_name: str | None = None,
        compute_advanced_metrics: bool = True,
    ) -> BiomechanicalData:
        """Extract complete biomechanical state at current time.

        Args:
            selected_actuator_name: Optional name of actuator.
            compute_advanced_metrics: Whether to compute expensive metrics.

        Returns:
            BiomechanicalData object
        """
        # Compute derived quantities
        qacc = self.compute_joint_accelerations()
        club_pos, club_vel, club_speed = self.get_club_head_data()
        left_grf, right_grf = self.get_ground_reaction_forces()
        com_pos, com_vel = self.get_center_of_mass()
        ke, pe, te = self.compute_energies()
        powers = self.get_actuator_powers()

        club_acc = None
        if club_vel is not None and self._prev_club_vel is not None:
            dt = self.data.time - self.prev_time
            if dt > 0:
                club_acc = (club_vel - self._prev_club_vel) / dt
        if club_vel is not None:
            self._prev_club_vel = club_vel.copy()

        induced = {}
        club_induced = None
        counterfactuals = {}

        if compute_advanced_metrics:
            # Use the optimized analyzer
            comps = self.induced_analyzer.compute_components()
            induced = {
                "gravity": comps["gravity"],
                "actuator": comps["control"],
                "velocity": comps["velocity"],
            }

            # Compute specific induced acceleration if requested
            if selected_actuator_name:
                act_id = mujoco.mj_name2id(
                    self.model,
                    mujoco.mjtObj.mjOBJ_ACTUATOR,
                    selected_actuator_name,
                )
                if act_id != -1:
                    tau_spec = np.zeros(self.model.nu)
                    tau_spec[act_id] = 1.0
                    # Compute components with specific torque
                    # The 'control' component corresponds to M^-1 * tau_spec
                    spec_comps = self.induced_analyzer.compute_components(
                        tau_app=tau_spec
                    )
                    induced["selected_actuator"] = spec_comps["control"]

            # Compute task space induced accelerations for club head
            if self.club_head_id is not None:
                # Use actual body name to ensure correct lookup
                club_name = mujoco.mj_id2name(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, self.club_head_id
                )
                if club_name:
                    # Pass computed components to avoid re-computation
                    club_induced = self.induced_analyzer.compute_task_space_components(
                        club_name, qdd_comps=comps
                    )

            counterfactuals = self.compute_counterfactuals()

        return BiomechanicalData(
            time=float(self.data.time),
            joint_positions=self.data.qpos.copy(),
            joint_velocities=self.data.qvel.copy(),
            joint_accelerations=qacc,
            joint_torques=self.data.ctrl.copy(),
            joint_forces=self.data.qfrc_constraint.copy(),
            actuator_forces=self.data.actuator_force.copy(),
            actuator_powers=powers,
            club_head_position=club_pos,
            club_head_velocity=club_vel,
            club_head_acceleration=club_acc,
            club_head_speed=club_speed,
            left_foot_force=left_grf,
            right_foot_force=right_grf,
            kinetic_energy=ke,
            potential_energy=pe,
            total_energy=te,
            com_position=com_pos,
            com_velocity=com_vel,
            induced_accelerations=induced,
            club_induced_accelerations=club_induced,
            counterfactuals=counterfactuals,
        )


class SwingRecorder:
    """Records time-series biomechanical data during a golf swing."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.frames: list[BiomechanicalData] = []
        self.is_recording = False

    def start_recording(self) -> None:
        self.is_recording = True
        self.frames = []

    def stop_recording(self) -> None:
        self.is_recording = False

    def record_frame(self, data: BiomechanicalData) -> None:
        if self.is_recording:
            self.frames.append(data)

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        if not self.frames:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        times = np.array([f.time for f in self.frames])
        values = [getattr(f, field_name) for f in self.frames]

        if all(v is None for v in values):
            return times, np.array([], dtype=np.float64)

        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            return times, np.array([], dtype=np.float64)

        times = times[valid_indices]
        values = [values[i] for i in valid_indices]

        try:
            values_array = np.array(values)
        except (ValueError, TypeError):
            return times, values

        return times, cast(np.ndarray, values_array)

    def get_induced_acceleration_series(
        self, source_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self.frames:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        times = []
        values = []

        for f in self.frames:
            val = f.induced_accelerations.get(source_name)
            if val is not None:
                times.append(f.time)
                values.append(val)

        if not values:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        return np.array(times, dtype=np.float64), cast(np.ndarray, np.array(values))

    def get_num_frames(self) -> int:
        """Get number of recorded frames."""
        return len(self.frames)

    def get_duration(self) -> float:
        """Get duration of recording in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def export_to_dict(self) -> dict:
        """Export all recorded data to a dictionary for JSON/CSV export.

        Returns:
            Dictionary with time series for all fields
        """
        if not self.frames:
            return {}

        export_data = {}

        # Scalar fields
        scalar_fields = [
            "time",
            "club_head_speed",
            "kinetic_energy",
            "potential_energy",
            "total_energy",
        ]

        for field_name in scalar_fields:
            _, values = self.get_time_series(field_name)
            if len(values) > 0:
                if isinstance(values, np.ndarray):
                    export_data[field_name] = values.tolist()
                else:
                    export_data[field_name] = list(values)

        # Vector/array fields
        array_fields = [
            "joint_positions",
            "joint_velocities",
            "joint_accelerations",
            "joint_torques",
            "joint_forces",
            "actuator_forces",
            "actuator_powers",
        ]

        for field_name in array_fields:
            _times, values = self.get_time_series(field_name)
            if len(values) > 0 and isinstance(values, np.ndarray):
                if values.ndim == 1:
                    export_data[field_name] = values.tolist()
                elif values.ndim == 2:
                    # Export each component separately
                    for i in range(values.shape[1]):
                        export_data[f"{field_name}_{i}"] = values[:, i].tolist()

        # Special 3D fields
        for field_name in [
            "club_head_position",
            "club_head_velocity",
            "com_position",
            "com_velocity",
        ]:
            _times, values = self.get_time_series(field_name)
            if len(values) > 0 and isinstance(values, np.ndarray) and values.ndim == 2:
                export_data[f"{field_name}_x"] = values[:, 0].tolist()
                export_data[f"{field_name}_y"] = values[:, 1].tolist()
                export_data[f"{field_name}_z"] = values[:, 2].tolist()

        # Export induced
        if self.frames and self.frames[0].induced_accelerations:
            all_keys: set[str] = set()
            for f in self.frames:
                all_keys.update(f.induced_accelerations.keys())

            for key in all_keys:
                _, vals = self.get_induced_acceleration_series(key)
                if len(vals) > 0:
                    for i in range(vals.shape[1]):
                        export_data[f"induced_acc_{key}_{i}"] = vals[:, i].tolist()

        # Export club induced
        if self.frames and self.frames[0].club_induced_accelerations:
            all_keys = set()
            for f in self.frames:
                if f.club_induced_accelerations:
                    all_keys.update(f.club_induced_accelerations.keys())

            for key in all_keys:
                _, vals = self.get_club_induced_acceleration_series(key)
                if len(vals) > 0:
                    for i in range(vals.shape[1]):
                        export_data[f"club_acc_{key}_{i}"] = vals[:, i].tolist()

        # Export counterfactuals
        if self.frames and self.frames[0].counterfactuals:
            all_keys = set()
            for f in self.frames:
                all_keys.update(f.counterfactuals.keys())

            for key in all_keys:
                _, vals = self.get_counterfactual_series(key)
                if len(vals) > 0 and isinstance(vals, np.ndarray):
                    if vals.ndim == 1:
                        export_data[f"cf_{key}"] = vals.tolist()
                    elif vals.ndim == 2:
                        for i in range(vals.shape[1]):
                            export_data[f"cf_{key}_{i}"] = vals[:, i].tolist()

        return export_data

    def get_club_induced_acceleration_series(
        self, component_name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract club head induced acceleration components.

        Args:
            component_name: 'gravity', 'velocity', 'control', 'constraint', 'total'
        """
        if not self.frames:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        times = []
        values = []

        for f in self.frames:
            if (
                f.club_induced_accelerations
                and component_name in f.club_induced_accelerations
            ):
                times.append(f.time)
                values.append(f.club_induced_accelerations[component_name])

        if not values:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        return np.array(times, dtype=np.float64), cast(np.ndarray, np.array(values))

    def get_counterfactual_series(self, cf_name: str) -> tuple[np.ndarray, np.ndarray]:
        if not self.frames:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        times = []
        values = []
        for f in self.frames:
            val = f.counterfactuals.get(cf_name)
            if val is not None:
                times.append(f.time)
                values.append(val)

        if not values:
            return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

        return np.array(times, dtype=np.float64), cast(np.ndarray, np.array(values))
