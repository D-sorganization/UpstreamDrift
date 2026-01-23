"""Examples demonstrating motion capture integration and force analysis.

This module provides comprehensive examples for the complete workflow:
1. Loading motion capture data
2. Retargeting to model
3. Computing kinematic forces (Coriolis, centrifugal)
4. Inverse dynamics analysis
5. Force decomposition
6. Swing comparison

These examples show how to analyze real player swings captured with
motion capture systems.
"""

import traceback

import mujoco
import numpy as np

from .inverse_dynamics import (
    InverseDynamicsAnalyzer,
    InverseDynamicsSolver,
    export_inverse_dynamics_to_csv,
)
from .kinematic_forces import KinematicForceAnalyzer, export_kinematic_forces_to_csv
from .models import ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
from .motion_capture import (
    MarkerSet,
    MotionCaptureFrame,
    MotionCaptureProcessor,
    MotionCaptureSequence,
    MotionRetargeting,
)


def example_1_load_motion_capture() -> None:
    """Example 1: Load motion capture data from file.

    Demonstrates loading mocap data from CSV and JSON formats.
    """

    # Generate synthetic mocap data for demonstration

    times = np.linspace(0, 2.0, 240)  # 120 Hz for 2 seconds
    num_markers = 5

    frames = []
    for t in times:
        # Simulate some marker motion
        markers = {}
        for i in range(num_markers):
            # Sinusoidal motion
            x = 0.5 * np.sin(2 * np.pi * t) + i * 0.1
            y = 0.3 * np.cos(2 * np.pi * t)
            z = 1.0 + 0.2 * np.sin(4 * np.pi * t)

            markers[f"MARKER_{i}"] = np.array([x, y, z])

        frame = MotionCaptureFrame(time=t, marker_positions=markers)
        frames.append(frame)

    mocap_sequence = MotionCaptureSequence(
        frames=frames,
        frame_rate=120.0,
        marker_names=[f"MARKER_{i}" for i in range(num_markers)],
    )

    # Get trajectory for one marker
    marker_name = "MARKER_0"
    _times_marker, _positions = mocap_sequence.get_marker_trajectory(marker_name)

    return mocap_sequence  # type: ignore[return-value]


def example_2_motion_retargeting() -> None:
    """Example 2: Retarget motion capture to MuJoCo model.

    Shows how to map marker positions to joint angles using IK.
    """

    # Load model
    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    # Get marker set (simplified for example)
    marker_set = MarkerSet.golf_swing_marker_set()

    # Create synthetic mocap data with fewer frames for speed
    times = np.linspace(0, 1.0, 30)  # Fewer frames for demo
    frames = []

    for t in times:
        markers = {}
        # Simplified: just create a few key markers
        markers["CLUB_HEAD"] = np.array([1.0 + 0.5 * np.sin(2 * np.pi * t), 0, 0.5])
        markers["RSHO"] = np.array([0, -0.2, 1.4])
        markers["LSHO"] = np.array([0, 0.2, 1.4])

        frame = MotionCaptureFrame(time=t, marker_positions=markers)
        frames.append(frame)

    mocap_seq = MotionCaptureSequence(
        frames=frames,
        frame_rate=30.0,
        marker_names=list(frames[0].marker_positions.keys()),
    )

    # Initialize retargeting
    retargeting = MotionRetargeting(model, data, marker_set)

    # Retarget (simplified - using available markers)
    times_ret, joint_traj, success_flags = retargeting.retarget_sequence(
        mocap_seq,
        use_markers=list(frames[0].marker_positions.keys()),
        ik_iterations=20,
    )

    100.0 * sum(success_flags) / len(success_flags)

    return times_ret, joint_traj  # type: ignore[return-value]


def example_3_compute_kinematic_forces() -> None:
    """Example 3: Compute Coriolis and centrifugal forces from kinematics.

    This is the KEY example for analyzing captured motion.
    Shows how to compute motion-dependent forces without full inverse dynamics.
    """

    # Load model
    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    # Create analyzer
    analyzer = KinematicForceAnalyzer(model, data)

    # Generate synthetic trajectory (or use retargeted mocap)
    times = np.linspace(0, 1.0, 60)
    positions = np.zeros((len(times), model.nv))
    velocities = np.zeros((len(times), model.nv))
    accelerations = np.zeros((len(times), model.nv))

    for i, t in enumerate(times):
        # Simulate swing motion
        positions[i, 0] = 1.0 * np.sin(np.pi * t)  # Spine rotation
        velocities[i, 0] = np.pi * 1.0 * np.cos(np.pi * t)
        accelerations[i, 0] = -(np.pi**2) * 1.0 * np.sin(np.pi * t)

        if model.nv > 5:
            positions[i, 1] = 0.5 * np.sin(2 * np.pi * t)  # Shoulder
            velocities[i, 1] = np.pi * 1.0 * np.cos(2 * np.pi * t)
            accelerations[i, 1] = -2 * np.pi**2 * 0.5 * np.sin(2 * np.pi * t)

    # Analyze forces
    force_data_list = analyzer.analyze_trajectory(
        times,
        positions,
        velocities,
        accelerations,
    )

    # Extract statistics
    max(np.max(np.abs(fd.coriolis_forces)) for fd in force_data_list)
    max(
        np.max(np.abs(fd.centrifugal_forces))
        for fd in force_data_list
        if fd.centrifugal_forces is not None
    )
    max(abs(fd.coriolis_power) for fd in force_data_list)

    # Analyze club head forces
    if force_data_list[0].club_head_apparent_force is not None:
        max(
            float(np.linalg.norm(fd.club_head_apparent_force))
            for fd in force_data_list
            if fd.club_head_apparent_force is not None
        )

    # Power analysis
    np.mean([fd.coriolis_power for fd in force_data_list])
    np.mean([fd.centrifugal_power for fd in force_data_list])

    export_kinematic_forces_to_csv(force_data_list, "kinematic_forces.csv")

    return force_data_list  # type: ignore[return-value]


def example_4_inverse_dynamics() -> None:
    """Example 4: Full inverse dynamics analysis.

    Compute required joint torques for a given motion.
    This provides complete force decomposition.
    """

    # Load model
    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    # Create solver
    solver = InverseDynamicsSolver(model, data)

    # Generate trajectory
    times = np.linspace(0, 1.0, 60)
    positions = np.zeros((len(times), model.nv))
    velocities = np.zeros((len(times), model.nv))
    accelerations = np.zeros((len(times), model.nv))

    for i, t in enumerate(times):
        positions[i, 0] = 1.5 * np.sin(np.pi * t)
        velocities[i, 0] = 1.5 * np.pi * np.cos(np.pi * t)
        accelerations[i, 0] = -1.5 * np.pi**2 * np.sin(np.pi * t)

    # Solve inverse dynamics
    id_results = solver.solve_inverse_dynamics_trajectory(
        times,
        positions,
        velocities,
        accelerations,
    )

    # Statistics
    max(np.max(np.abs(r.joint_torques)) for r in id_results)
    max(
        np.max(np.abs(r.inertial_torques))
        for r in id_results
        if r.inertial_torques is not None
    )
    max(
        np.max(np.abs(r.coriolis_torques))
        for r in id_results
        if r.coriolis_torques is not None
    )
    max(
        np.max(np.abs(r.gravity_torques))
        for r in id_results
        if r.gravity_torques is not None
    )

    # Decomposition percentages
    np.mean([np.linalg.norm(r.joint_torques) for r in id_results])
    np.mean(
        [
            np.linalg.norm(r.inertial_torques)
            for r in id_results
            if r.inertial_torques is not None
        ],
    )
    np.mean(
        [
            np.linalg.norm(r.coriolis_torques)
            for r in id_results
            if r.coriolis_torques is not None
        ],
    )
    np.mean(
        [
            np.linalg.norm(r.gravity_torques)
            for r in id_results
            if r.gravity_torques is not None
        ],
    )

    export_inverse_dynamics_to_csv(times, id_results, "inverse_dynamics.csv")

    return id_results  # type: ignore[return-value]


def example_5_complete_analysis_pipeline() -> None:
    """Example 5: Complete analysis from mocap to forces.

    This is the full workflow you would use for analyzing a real
    player's swing captured with motion capture.
    """

    # Step 1: Load mocap (simulated)

    times_mocap = np.linspace(0, 1.5, 180)  # 120 Hz
    frames = []

    for t in times_mocap:
        markers = {
            "CLUB_HEAD": np.array(
                [1.0 + 0.8 * np.sin(np.pi * t), 0, 0.5 + 0.3 * np.sin(2 * np.pi * t)],
            ),
            "RSHO": np.array([0, -0.2, 1.4]),
            "LSHO": np.array([0, 0.2, 1.4]),
        }
        frame = MotionCaptureFrame(time=t, marker_positions=markers)
        frames.append(frame)

    mocap_seq = MotionCaptureSequence(
        frames=frames,
        frame_rate=120.0,
        marker_names=list(markers.keys()),
    )

    # Step 2: Retarget

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    marker_set = MarkerSet.golf_swing_marker_set()
    retargeting = MotionRetargeting(model, data, marker_set)

    times_ret, joint_traj, success = retargeting.retarget_sequence(
        mocap_seq,
        use_markers=list(markers.keys()),
        ik_iterations=15,
    )

    100.0 * sum(success) / len(success)

    # Step 3: Filter and compute derivatives

    processor = MotionCaptureProcessor()

    # Filter
    joint_traj_filtered = processor.filter_trajectory(
        times_ret,
        joint_traj,
        cutoff_frequency=10.0,
        sampling_rate=120.0,
    )

    # Compute velocities and accelerations
    velocities = processor.compute_velocities(
        times_ret,
        joint_traj_filtered,
        method="spline",
    )
    accelerations = processor.compute_accelerations(
        times_ret,
        velocities,
        method="spline",
    )

    # Step 4: Kinematic forces

    kin_analyzer = KinematicForceAnalyzer(model, data)
    force_data_list = kin_analyzer.analyze_trajectory(
        times_ret,
        joint_traj_filtered,
        velocities,
        accelerations,
    )

    max(abs(fd.coriolis_power) for fd in force_data_list)

    # Step 5: Inverse dynamics

    id_solver = InverseDynamicsSolver(model, data)
    id_results = id_solver.solve_inverse_dynamics_trajectory(
        times_ret,
        joint_traj_filtered,
        velocities,
        accelerations,
    )

    max(np.max(np.abs(r.joint_torques)) for r in id_results)

    # Step 6: Results

    # Kinetic energy
    avg_rot_ke = np.mean([fd.rotational_kinetic_energy for fd in force_data_list])
    avg_trans_ke = np.mean([fd.translational_kinetic_energy for fd in force_data_list])
    avg_rot_ke + avg_trans_ke

    return {  # type: ignore[return-value]
        "mocap": mocap_seq,
        "kinematics": (times_ret, joint_traj_filtered, velocities, accelerations),
        "forces": force_data_list,
        "torques": id_results,
    }


def example_6_swing_comparison() -> None:
    """Example 6: Compare two different swings.

    Shows how to quantitatively compare swings from different players
    or the same player at different times.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    analyzer = InverseDynamicsAnalyzer(model, data)

    # Generate two different swing trajectories

    times = np.linspace(0, 1.0, 60)

    # Swing 1: Slower, smoother
    pos1 = np.zeros((len(times), model.nv))
    vel1 = np.zeros((len(times), model.nv))
    acc1 = np.zeros((len(times), model.nv))

    for i, t in enumerate(times):
        pos1[i, 0] = 1.2 * np.sin(np.pi * t)
        vel1[i, 0] = 1.2 * np.pi * np.cos(np.pi * t)
        acc1[i, 0] = -1.2 * np.pi**2 * np.sin(np.pi * t)

    # Swing 2: Faster, more aggressive
    pos2 = np.zeros((len(times), model.nv))
    vel2 = np.zeros((len(times), model.nv))
    acc2 = np.zeros((len(times), model.nv))

    for i, t in enumerate(times):
        pos2[i, 0] = 1.5 * np.sin(1.2 * np.pi * t)
        vel2[i, 0] = 1.5 * 1.2 * np.pi * np.cos(1.2 * np.pi * t)
        acc2[i, 0] = -1.5 * (1.2 * np.pi) ** 2 * np.sin(1.2 * np.pi * t)

    # Analyze both
    analysis1 = analyzer.analyze_captured_motion(times, pos1, vel1, acc1)

    analysis2 = analyzer.analyze_captured_motion(times, pos2, vel2, acc2)

    # Compare

    stats1 = analysis1["statistics"]
    stats2 = analysis2["statistics"]

    diff_power = stats2["peak_coriolis_power"] - stats1["peak_coriolis_power"]

    diff_torque = stats2["max_joint_torque"] - stats1["max_joint_torque"]

    # Energy comparison
    avg_ke1 = np.mean(
        [
            fd.rotational_kinetic_energy + fd.translational_kinetic_energy
            for fd in analysis1["kinematic_forces"]
        ],
    )
    avg_ke2 = np.mean(
        [
            fd.rotational_kinetic_energy + fd.translational_kinetic_energy
            for fd in analysis2["kinematic_forces"]
        ],
    )

    diff_ke = avg_ke2 - avg_ke1

    # Note: Differences can be used for comparison analysis
    _ = diff_power > 0  # Check if power increased
    _ = diff_torque > 0  # Check if torque increased
    _ = diff_ke > 0  # Check if kinetic energy increased

    # Significant differences could indicate meaningful changes
    _ = diff_power > 0.5 * stats1["peak_coriolis_power"]
    _ = diff_torque > 0.3 * stats1["max_joint_torque"]

    return analysis1, analysis2  # type: ignore[return-value]


def run_all_examples() -> None:
    """Run all motion capture examples."""

    examples = [
        ("Load Motion Capture", example_1_load_motion_capture),
        ("Motion Retargeting", example_2_motion_retargeting),
        ("Kinematic Forces", example_3_compute_kinematic_forces),
        ("Inverse Dynamics", example_4_inverse_dynamics),
        ("Complete Pipeline", example_5_complete_analysis_pipeline),
        ("Swing Comparison", example_6_swing_comparison),
    ]

    for _i, (_name, example_func) in enumerate(examples, 1):
        try:
            example_func()
        except Exception:
            traceback.print_exc()


if __name__ == "__main__":
    run_all_examples()
