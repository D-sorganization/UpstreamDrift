"""Biomechanical analysis module for golf swing simulations.

This module provides comprehensive force, torque, and kinematic analysis
for golf swing models. It extracts data from MuJoCo simulations and computes
derived biomechanical quantities.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np


@dataclass
class BiomechanicalData:
    """Container for biomechanical measurements at a single time point.

    All arrays are indexed by joint/actuator number unless otherwise noted.
    """

    # Time
    time: float = 0.0

    # Joint kinematics
    joint_positions: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.array([]))
    joint_accelerations: np.ndarray = field(default_factory=lambda: np.array([]))

    # Joint kinetics
    joint_torques: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )  # Applied torques
    joint_forces: np.ndarray = field(
        default_factory=lambda: np.array([]),
    )  # Constraint forces

    # Actuator data
    actuator_forces: np.ndarray = field(default_factory=lambda: np.array([]))
    actuator_powers: np.ndarray = field(default_factory=lambda: np.array([]))

    # Club head data (if available)
    club_head_position: np.ndarray | None = None
    club_head_velocity: np.ndarray | None = None
    club_head_acceleration: np.ndarray | None = None
    club_head_speed: float = 0.0

    # Ground reaction forces (if legs present)
    left_foot_force: np.ndarray | None = None
    right_foot_force: np.ndarray | None = None

    # Total energy
    kinetic_energy: float = 0.0
    potential_energy: float = 0.0
    total_energy: float = 0.0

    # Center of mass
    com_position: np.ndarray | None = None
    com_velocity: np.ndarray | None = None


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

        return qacc

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
        jacp = np.zeros(3 * self.model.nv)
        jacr = np.zeros(3 * self.model.nv)
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.club_head_id)
        jacp = jacp.reshape(3, self.model.nv)

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
        # Use MuJoCo's built-in COM computation
        # Note: subtree_com[0] is the world body COM, which may legitimately be [0,0,0]
        # We always use MuJoCo's computed values as they are authoritative
        com_pos = self.data.subtree_com[0].copy()  # World body COM

        # Compute COM velocity manually using Jacobians\
        # (subtree_linvel doesn't exist in MuJoCo)
        # Compute COM velocity from body velocities
        total_mass = 0.0
        com_vel = np.zeros(3)

        for i in range(self.model.nbody):
            if i == 0:  # Skip world body
                continue
            body_mass = self.model.body_mass[i]

            # Get body velocity using Jacobian
            # MuJoCo 3.3+ may require reshaped arrays
            try:
                jacp = np.zeros((3, self.model.nv))
                jacr = np.zeros((3, self.model.nv))
                mujoco.mj_jacBodyCom(self.model, self.data, jacp, jacr, i)
            except TypeError:
                # Fallback to flat array approach
                jacp_flat = np.zeros(3 * self.model.nv)
                jacr_flat = np.zeros(3 * self.model.nv)
                mujoco.mj_jacBodyCom(self.model, self.data, jacp_flat, jacr_flat, i)
                jacp = jacp_flat.reshape(3, self.model.nv)
            body_vel = jacp @ self.data.qvel

            com_vel += body_mass * body_vel
            total_mass += body_mass

        if total_mass > 0:
            com_vel /= total_mass

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

        Power = torque * angular_velocity (or force * linear_velocity)

        Returns:
            Array of actuator powers [W]
        """
        powers = np.zeros(self.model.nu)

        for i in range(self.model.nu):
            # Get the joint this actuator acts on
            joint_id = self.model.actuator_trnid[i, 0]

            # Get actuator force/torque
            actuator_force = self.data.actuator_force[i]

            # Get joint velocity
            if joint_id >= 0 and joint_id < self.model.nv:
                joint_velocity = self.data.qvel[joint_id]
                powers[i] = actuator_force * joint_velocity

        return powers

    def extract_full_state(self) -> BiomechanicalData:
        """Extract complete biomechanical state at current time.

        Returns:
            BiomechanicalData object with all available measurements
        """
        # Compute derived quantities
        qacc = self.compute_joint_accelerations()
        club_pos, club_vel, club_speed = self.get_club_head_data()
        left_grf, right_grf = self.get_ground_reaction_forces()
        com_pos, com_vel = self.get_center_of_mass()
        ke, pe, te = self.compute_energies()
        powers = self.get_actuator_powers()

        # Club head acceleration (finite difference)
        club_acc = None
        if club_vel is not None and self._prev_club_vel is not None:
            dt = self.data.time - self.prev_time
            if dt > 0:
                club_acc = (club_vel - self._prev_club_vel) / dt
        if club_vel is not None:
            self._prev_club_vel = club_vel.copy()

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
        )


class SwingRecorder:
    """Records time-series biomechanical data during a golf swing.

    This class accumulates BiomechanicalData snapshots over time
    for later analysis and visualization.
    """

    def __init__(self) -> None:
        """Initialize empty recorder."""
        self.reset()

    def reset(self) -> None:
        """Clear all recorded data."""
        self.frames: list[BiomechanicalData] = []
        self.is_recording = False

    def start_recording(self) -> None:
        """Start recording data."""
        self.is_recording = True
        self.frames = []

    def stop_recording(self) -> None:
        """Stop recording data."""
        self.is_recording = False

    def record_frame(self, data: BiomechanicalData) -> None:
        """Add a frame of data to the recording.

        Args:
            data: BiomechanicalData snapshot to record
        """
        if self.is_recording:
            self.frames.append(data)

    def get_time_series(self, field_name: str) -> tuple[np.ndarray, np.ndarray | list]:
        """Extract time series for a specific field.

        Args:
            field_name: Name of the field in BiomechanicalData

        Returns:
            Tuple of (times, values) where values may be 1D or 2D array or list
        """
        if not self.frames:
            return np.array([]), np.array([])

        times = np.array([f.time for f in self.frames])
        values = [getattr(f, field_name) for f in self.frames]

        # Handle None values
        if all(v is None for v in values):
            return times, np.array([])

        # Filter out None values
        valid_indices = [i for i, v in enumerate(values) if v is not None]
        if not valid_indices:
            return times, np.array([])

        times = times[valid_indices]
        values = [values[i] for i in valid_indices]

        # Stack into array
        try:
            values_array = np.array(values)
        except (ValueError, TypeError):
            # Handle ragged arrays or mixed types
            return times, values

        return times, values_array

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
            times, values = self.get_time_series(field_name)
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

        return export_data
