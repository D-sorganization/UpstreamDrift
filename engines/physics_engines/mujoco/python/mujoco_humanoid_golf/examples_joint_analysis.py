"""Examples for analyzing universal and gimbal joints in golf swing models.

This module demonstrates:
1. Analyzing universal joint constraint forces
2. Measuring torque wobble in universal joints
3. Testing gimbal joint behavior
4. Comparing rigid vs flexible club shafts

Run examples with:
    python -m mujoco_humanoid_golf.examples_joint_analysis
"""

import os

import matplotlib.pyplot as plt
import mujoco as mj
import numpy as np
from mujoco_humanoid_golf.joint_analysis import (
    GimbalJointAnalyzer,
    UniversalJointAnalyzer,
    plot_torque_wobble,
)
from mujoco_humanoid_golf.models import (
    CLUB_CONFIGS,
    GIMBAL_JOINT_DEMO_XML,
    TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML,
    generate_flexible_club_xml,
    generate_rigid_club_xml,
)


def example_universal_joint_wobble() -> None:
    """Example: Analyze torque wobble in universal joint."""

    # Load two-link model with universal joint
    model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
    data = mj.MjData(model)

    # Initialize analyzer
    analyzer = UniversalJointAnalyzer(model, data)

    # Set the universal joint to a bent position (30 degrees on each axis)
    wrist_u1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "wrist_universal_1")
    wrist_u2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "wrist_universal_2")

    wrist_u1_addr = model.jnt_qposadr[wrist_u1_id]
    wrist_u2_addr = model.jnt_qposadr[wrist_u2_id]

    # Set initial angles
    data.qpos[wrist_u1_addr] = np.radians(30)
    data.qpos[wrist_u2_addr] = np.radians(30)

    # Analyze torque transmission
    results = analyzer.analyze_torque_transmission(
        "wrist_universal_1",
        "wrist_universal_2",
        num_cycles=2,
    )

    # Plot results
    plot_torque_wobble(results, save_path="output/universal_joint_wobble.png")


def example_constraint_forces() -> None:
    """Example: Analyze constraint forces in universal joint during motion."""

    # Load model
    model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
    data = mj.MjData(model)

    # Apply sinusoidal torque to shoulder and wrist joints
    shoulder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
    wrist_u1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "wrist_u1_motor")
    wrist_u2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "wrist_u2_motor")

    # Simulate and record forces
    duration = 2.0
    num_steps = int(duration / model.opt.timestep)
    time_history = []
    shoulder_force = []
    wrist_u1_force = []
    wrist_u2_force = []

    for _step in range(num_steps):
        t = data.time

        # Apply sinusoidal control (simulating swing)
        data.ctrl[shoulder_id] = 50 * np.sin(2 * np.pi * 0.5 * t)  # 0.5 Hz
        data.ctrl[wrist_u1_id] = 20 * np.sin(2 * np.pi * 0.5 * t + np.pi / 4)
        data.ctrl[wrist_u2_id] = 20 * np.cos(2 * np.pi * 0.5 * t + np.pi / 4)

        mj.mj_step(model, data)

        # Record forces
        time_history.append(t)
        shoulder_joint_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "shoulder")
        wrist_u1_joint_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_JOINT,
            "wrist_universal_1",
        )
        wrist_u2_joint_id = mj.mj_name2id(
            model,
            mj.mjtObj.mjOBJ_JOINT,
            "wrist_universal_2",
        )

        # Use DOF addresses (jnt_dofadr), not qpos addresses (jnt_qposadr)
        shoulder_force.append(data.qfrc_constraint[model.jnt_dofadr[shoulder_joint_id]])
        wrist_u1_force.append(data.qfrc_constraint[model.jnt_dofadr[wrist_u1_joint_id]])
        wrist_u2_force.append(data.qfrc_constraint[model.jnt_dofadr[wrist_u2_joint_id]])

    # Plot results
    _fig, axes = plt.subplots(3, 1, figsize=(12, 10))  # type: ignore[misc]

    time_array = np.array(time_history)

    axes[0].plot(time_array, shoulder_force, "b-", linewidth=2)  # type: ignore[index]
    axes[0].set_ylabel("Constraint Torque (N⋅m)", fontsize=11)  # type: ignore[index]
    axes[0].set_title(  # type: ignore[index]
        "Shoulder Joint Constraint Forces",
        fontsize=13,
        fontweight="bold",
    )
    axes[0].grid(True, alpha=0.3)  # type: ignore[index]

    axes[1].plot(time_array, wrist_u1_force, "g-", linewidth=2)  # type: ignore[index]
    axes[1].set_ylabel("Constraint Torque (N⋅m)", fontsize=11)  # type: ignore[index]
    axes[1].set_title("Wrist Universal Joint - Axis 1", fontsize=13, fontweight="bold")  # type: ignore[index]
    axes[1].grid(True, alpha=0.3)  # type: ignore[index]

    axes[2].plot(time_array, wrist_u2_force, "r-", linewidth=2)  # type: ignore[index]
    axes[2].set_xlabel("Time (s)", fontsize=12)  # type: ignore[index]
    axes[2].set_ylabel("Constraint Torque (N⋅m)", fontsize=11)  # type: ignore[index]
    axes[2].set_title("Wrist Universal Joint - Axis 2", fontsize=13, fontweight="bold")  # type: ignore[index]
    axes[2].grid(True, alpha=0.3)  # type: ignore[index]

    plt.tight_layout()
    plt.savefig("output/constraint_forces.png", dpi=300, bbox_inches="tight")
    plt.show()


def example_gimbal_joint() -> None:
    """Example: Demonstrate gimbal joint behavior and check for gimbal lock."""

    # Load gimbal model
    model = mj.MjModel.from_xml_string(GIMBAL_JOINT_DEMO_XML)
    data = mj.MjData(model)

    # Initialize analyzer
    analyzer = GimbalJointAnalyzer(model, data)

    # Test various gimbal configurations
    test_configs = [
        (0, 0, 0, "Neutral position"),
        (np.pi / 4, np.pi / 4, np.pi / 4, "45° on all axes"),
        (0, np.pi / 2, 0, "90° middle ring (GIMBAL LOCK)"),
        (0, np.pi / 2 - 0.1, 0, "Near gimbal lock"),
    ]

    for x_angle, y_angle, z_angle, _description in test_configs:
        # Set gimbal angles
        gimbal_x_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "gimbal_x")
        gimbal_y_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "gimbal_y")
        gimbal_z_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "gimbal_z")

        data.qpos[model.jnt_qposadr[gimbal_x_id]] = x_angle
        data.qpos[model.jnt_qposadr[gimbal_y_id]] = y_angle
        data.qpos[model.jnt_qposadr[gimbal_z_id]] = z_angle

        # Check for gimbal lock
        _is_locked, _distance = analyzer.check_gimbal_lock(
            "gimbal_x",
            "gimbal_y",
            "gimbal_z",
        )


def example_flexible_vs_rigid_club() -> None:
    """Example: Compare flexible vs rigid club shaft behavior."""

    # Generate flexible club XML
    # Note: Return value is intentionally discarded in this example
    _ = generate_flexible_club_xml("driver", num_segments=3)

    # Generate rigid club XML
    generate_rigid_club_xml("driver")

    # Show shaft parameters
    CLUB_CONFIGS["driver"]


def example_interactive_universal_joint() -> None:
    """Interactive visualization of universal joint model."""

    # Load model
    model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
    data = mj.MjData(model)

    # Apply some initial motion
    shoulder_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "shoulder_motor")
    wrist_u1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "wrist_u1_motor")

    def controller(model, data) -> None:
        """Simple sinusoidal controller for demonstration."""
        t = data.time
        data.ctrl[shoulder_id] = 30 * np.sin(2 * np.pi * 0.3 * t)
        data.ctrl[wrist_u1_id] = 15 * np.sin(2 * np.pi * 0.6 * t)

    # Launch passive viewer
    with mj.viewer.launch_passive(model, data) as viewer_inst:
        viewer_inst.cam.distance = 3.0
        viewer_inst.cam.azimuth = 135
        viewer_inst.cam.elevation = -20

        # Run simulation
        for _ in range(10000):  # 10 seconds at 1ms timestep
            controller(model, data)
            mj.mj_step(model, data)
            viewer_inst.sync()


def main() -> None:
    """Run all examples."""
    os.makedirs("output", exist_ok=True)

    # Run examples
    example_universal_joint_wobble()
    example_constraint_forces()
    example_gimbal_joint()
    example_flexible_vs_rigid_club()

    # Optional interactive example (comment out if not needed)
    # example_interactive_universal_joint()


if __name__ == "__main__":
    main()
