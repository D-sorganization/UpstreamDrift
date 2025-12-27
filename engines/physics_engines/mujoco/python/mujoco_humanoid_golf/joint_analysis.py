"""Analysis tools for joint constraint forces, torque transmission, and universal
joints.

This module provides tools for analyzing:
- Constraint forces in universal and gimbal joints
- Torque transmission through multi-DOF joints
- Torque wobble in universal joints (Cardan angles)
- Joint coupling effects

Author: MuJoCo Golf Swing Project
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np


class UniversalJointAnalyzer:
    """Analyze universal joint behavior including torque wobble and constraints."""

    def __init__(self, model: mj.MjModel, data: mj.MjData) -> None:
        """Initialize the analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    def get_joint_forces(self, joint_name: str) -> np.ndarray:
        """Get constraint forces for a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Array of constraint forces (size depends on joint DOF)
        """
        joint_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id == -1:
            msg = f"Joint '{joint_name}' not found"
            raise ValueError(msg)

        # Get joint DOF address (not qpos address!)
        # qfrc_constraint is indexed by DOF (nv), not qpos (nq)
        jnt_dofadr = self.model.jnt_dofadr[joint_id]
        jnt_type = self.model.jnt_type[joint_id]

        # Extract constraint forces from qfrc_constraint using DOF indices
        if jnt_type in (mj.mjtJoint.mjJNT_HINGE, mj.mjtJoint.mjJNT_SLIDE):
            # Hinge and slide joints have 1 DOF
            return np.array([self.data.qfrc_constraint[jnt_dofadr]], dtype=np.float64)
        if jnt_type == mj.mjtJoint.mjJNT_BALL:
            # Ball joint has 3 DOF
            return np.array(self.data.qfrc_constraint[jnt_dofadr : jnt_dofadr + 3], dtype=np.float64)
        if jnt_type == mj.mjtJoint.mjJNT_FREE:
            # Free joint has 6 DOF (3 translation + 3 rotation)
            return np.array(self.data.qfrc_constraint[jnt_dofadr : jnt_dofadr + 6], dtype=np.float64)
        return np.array([0.0], dtype=np.float64)

    def get_universal_joint_angles(
        self,
        joint1_name: str,
        joint2_name: str,
    ) -> tuple[float, float]:
        """Get the two angles of a universal joint (implemented as 2 hinges).

        Args:
            joint1_name: Name of first hinge
            joint2_name: Name of second hinge

        Returns:
            Tuple of (angle1, angle2) in radians
        """
        joint1_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint1_name)
        joint2_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint2_name)

        if joint1_id == -1 or joint2_id == -1:
            msg = "One or both joints not found"
            raise ValueError(msg)

        qpos1_addr = self.model.jnt_qposadr[joint1_id]
        qpos2_addr = self.model.jnt_qposadr[joint2_id]

        return self.data.qpos[qpos1_addr], self.data.qpos[qpos2_addr]

    def calculate_torque_wobble(self, input_angle: float, joint_angle: float) -> float:
        """Calculate torque wobble (variation) due to universal joint geometry.

        Universal joints exhibit torque wobble when the two shafts are at an angle.
        The output shaft speed varies sinusoidally even if input speed is constant.

        Args:
            input_angle: Rotation angle of input shaft (radians)
            joint_angle: Angle between the two shafts (radians)

        Returns:
            Angular velocity ratio (output/input)
        """
        # Classic universal joint velocity relationship
        # ω_out/ω_in = cos(β) / (1 - sin²(β)sin²(θ))
        # where β is the joint angle and θ is the input rotation
        if abs(joint_angle) < 1e-6:
            return 1.0

        sin_beta = np.sin(joint_angle)
        sin_theta = np.sin(input_angle)
        cos_beta = np.cos(joint_angle)

        denominator = 1.0 - (sin_beta**2) * (sin_theta**2)
        if abs(denominator) < 1e-9:
            return float(np.inf)

        return float(cos_beta / denominator)

    def analyze_torque_transmission(
        self,
        input_joint: str,
        output_joint: str,
        num_cycles: int = 2,
    ) -> dict[str, np.ndarray | float]:
        """Analyze torque transmission through a universal joint over full rotations.

        Args:
            input_joint: Name of input joint
            output_joint: Name of output joint
            num_cycles: Number of complete rotations to analyze

        Returns:
            Dictionary with analysis results
        """
        # Get joint angles
        input_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, input_joint)
        output_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, output_joint)

        if input_id == -1 or output_id == -1:
            msg = "One or both joints not found"
            raise ValueError(msg)

        self.model.jnt_qposadr[input_id]
        self.model.jnt_qposadr[output_id]

        # Sample over full rotations
        angles = np.linspace(0, 2 * np.pi * num_cycles, num_cycles * 360)
        velocity_ratios = []
        torque_ratios = []

        for angle in angles:
            # Get current joint angles
            angle1, angle2 = self.get_universal_joint_angles(input_joint, output_joint)

            # Calculate joint bend angle (angle between shafts)
            joint_angle = np.sqrt(angle1**2 + angle2**2)

            # Calculate velocity and torque ratios
            vel_ratio = self.calculate_torque_wobble(angle, joint_angle)
            torque_ratio = 1.0 / vel_ratio if abs(vel_ratio) > 1e-9 else 0.0

            velocity_ratios.append(vel_ratio)
            torque_ratios.append(torque_ratio)

        return {
            "angles": angles,
            "velocity_ratios": np.array(velocity_ratios),
            "torque_ratios": np.array(torque_ratios),
            "wobble_amplitude": float(np.std(velocity_ratios)),
            "mean_velocity_ratio": float(np.mean(velocity_ratios)),
        }


class GimbalJointAnalyzer:
    """Analyze gimbal joint behavior and singularities."""

    def __init__(self, model: mj.MjModel, data: mj.MjData) -> None:
        """Initialize the analyzer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
        """
        self.model = model
        self.data = data

    def get_gimbal_angles(
        self,
        joint_x: str,
        joint_y: str,
        joint_z: str,
    ) -> tuple[float, float, float]:
        """Get the three Euler angles from a gimbal joint.

        Args:
            joint_x: Name of X-axis rotation joint
            joint_y: Name of Y-axis rotation joint
            joint_z: Name of Z-axis rotation joint

        Returns:
            Tuple of (angle_x, angle_y, angle_z) in radians
        """
        x_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_x)
        y_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_y)
        z_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_JOINT, joint_z)

        if x_id == -1 or y_id == -1 or z_id == -1:
            msg = "One or more gimbal joints not found"
            raise ValueError(msg)

        angle_x = self.data.qpos[self.model.jnt_qposadr[x_id]]
        angle_y = self.data.qpos[self.model.jnt_qposadr[y_id]]
        angle_z = self.data.qpos[self.model.jnt_qposadr[z_id]]

        return angle_x, angle_y, angle_z

    def check_gimbal_lock(
        self,
        joint_x: str,
        joint_y: str,
        joint_z: str,
        threshold: float = 0.087,  # 5 degrees
    ) -> tuple[bool, float]:
        """Check if gimbal is near gimbal lock singularity.

        Gimbal lock occurs when the middle ring is rotated ±90 degrees.

        Args:
            joint_x: Name of X-axis rotation joint
            joint_y: Name of Y-axis rotation joint (middle ring)
            joint_z: Name of Z-axis rotation joint
            threshold: Threshold in radians for detecting near-gimbal-lock

        Returns:
            Tuple of (is_near_lock, distance_to_lock)
        """
        _, angle_y, _ = self.get_gimbal_angles(joint_x, joint_y, joint_z)

        # Distance to nearest ±90 degree position
        distance = min(abs(angle_y - np.pi / 2), abs(angle_y + np.pi / 2))

        is_near_lock = distance < threshold

        return is_near_lock, distance


def plot_torque_wobble(
    analysis_results: dict[str, np.ndarray | float],
    save_path: str | None = None,
) -> None:
    """Plot torque wobble analysis results.

    Args:
        analysis_results: Results from UniversalJointAnalyzer.analyze_torque_trans
        save_path: Optional path to save the plot
    """
    _fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))  # type: ignore[misc]

    angles_deg = np.degrees(analysis_results["angles"])

    # Velocity ratio plot
    ax1.plot(angles_deg, analysis_results["velocity_ratios"], "b-", linewidth=2)
    ax1.axhline(y=1.0, color="r", linestyle="--", label="Ideal (no wobble)")
    ax1.set_xlabel("Input Rotation Angle (degrees)", fontsize=12)
    ax1.set_ylabel("Velocity Ratio (ω_out/ω_in)", fontsize=12)
    ax1.set_title("Universal Joint Velocity Wobble", fontsize=14, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Torque ratio plot
    ax2.plot(angles_deg, analysis_results["torque_ratios"], "g-", linewidth=2)
    ax2.axhline(y=1.0, color="r", linestyle="--", label="Ideal (no wobble)")
    ax2.set_xlabel("Input Rotation Angle (degrees)", fontsize=12)
    ax2.set_ylabel("Torque Ratio (τ_out/τ_in)", fontsize=12)
    ax2.set_title("Universal Joint Torque Transmission", fontsize=14, fontweight="bold")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def analyze_constraint_forces_over_time(
    model: mj.MjModel,
    data: mj.MjData,
    joint_names: list[str],
    duration: float = 2.0,
    timestep: float | None = None,
) -> dict[str, np.ndarray]:
    """Record constraint forces for specified joints over a simulation duration.

    Args:
        model: MuJoCo model
        data: MuJoCo data
        joint_names: List of joint names to monitor
        duration: Simulation duration in seconds
        timestep: Timestep (if None, uses model timestep)

    Returns:
        Dictionary mapping joint names to force time series
    """
    if timestep is None:
        timestep = model.opt.timestep

    num_steps = int(duration / timestep)
    analyzer = UniversalJointAnalyzer(model, data)

    # Initialize storage
    force_history: dict[str, list[float]] = {name: [] for name in joint_names}
    time_history: list[float] = []

    # Run simulation and record forces
    for _step in range(num_steps):
        mj.mj_step(model, data)

        for joint_name in joint_names:
            try:
                force_raw = analyzer.get_joint_forces(joint_name)
                # Convert to scalar if it's an array
                if isinstance(force_raw, np.ndarray):
                    force_scalar: float = (
                        float(np.linalg.norm(force_raw)) if force_raw.size > 0 else 0.0
                    )
                else:
                    force_scalar = float(force_raw)
                force_history[joint_name].append(force_scalar)
            except ValueError:
                force_history[joint_name].append(0.0)

        time_history.append(data.time)

    # Convert to numpy arrays
    result = {
        "time": np.array(time_history),
    }
    for name, forces in force_history.items():
        result[name] = np.array(forces)

    return result


def plot_constraint_forces(
    force_data: dict[str, np.ndarray],
    joint_names: list[str],
    save_path: str | None = None,
) -> None:
    """Plot constraint forces over time for multiple joints.

    Args:
        force_data: Dictionary from analyze_constraint_forces_over_time
        joint_names: List of joint names to plot
        save_path: Optional path to save the plot
    """
    num_joints = len(joint_names)
    _fig, axes = plt.subplots(num_joints, 1, figsize=(12, 4 * num_joints))

    # Ensure axes is always a list for type checking
    axes_list: list[plt.Axes] = (
        [axes] if num_joints == 1 else axes  # type: ignore[assignment,list-item]
    )

    time = force_data["time"]

    for _idx, (ax, joint_name) in enumerate(zip(axes_list, joint_names, strict=False)):
        forces = force_data[joint_name]

        if forces.ndim == 1:
            ax.plot(time, forces, "b-", linewidth=2, label="Constraint Force")
        else:
            # Multiple force components
            for i in range(forces.shape[1]):
                ax.plot(time, forces[:, i], linewidth=2, label=f"Component {i+1}")

        ax.set_xlabel("Time (s)", fontsize=12)
        ax.set_ylabel("Force/Torque (N or N⋅m)", fontsize=12)
        ax.set_title(f"Constraint Forces: {joint_name}", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
