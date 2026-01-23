"""Examples demonstrating advanced robotics features.

This module provides comprehensive examples of using the advanced
robotics capabilities for golf swing analysis.
"""

import traceback

import mujoco
import numpy as np

from .advanced_control import (
    AdvancedController,
    ControlMode,
    HybridControlMask,
    ImpedanceParameters,
)
from .advanced_kinematics import AdvancedKinematicsAnalyzer
from .models import ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
from .motion_optimization import (
    MotionPrimitiveLibrary,
    OptimizationConstraints,
    OptimizationObjectives,
    SwingOptimizer,
)

# Minimum degrees of freedom required for full swing configuration
MIN_DOF_FOR_CONFIG = 10


def example_1_constraint_jacobian_analysis() -> None:
    """Example 1: Analyze constraint Jacobian for two-handed grip.

    This demonstrates analysis of the closed-chain constraint created
    by both hands on the club (a parallel mechanism).
    """

    # Load model
    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    # Initialize analyzer
    analyzer = AdvancedKinematicsAnalyzer(model, data)

    # Set a swing configuration
    if model.nv >= MIN_DOF_FOR_CONFIG:
        data.qpos[0] = 0.5  # Spine rotation
        data.qpos[1] = 1.0  # Left shoulder
        data.qpos[5] = 0.8  # Right shoulder

    mujoco.mj_forward(model, data)

    # Compute constraint Jacobian
    constraint_data = analyzer.compute_constraint_jacobian()

    # Note: constraint_data can be used for further analysis if needed
    _ = constraint_data.nullspace_dimension  # Acknowledge the data is available


def example_2_manipulability_analysis() -> None:
    """Example 2: Compute manipulability and detect singularities.

    This shows how to analyze the manipulability ellipsoid and
    detect singular configurations.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    analyzer = AdvancedKinematicsAnalyzer(model, data)

    # Analyze at different configurations
    rng = np.random.default_rng()
    configurations = [
        ("Address", np.zeros(model.nv)),
        ("Backswing", rng.standard_normal(model.nv) * 0.5),
        ("Impact", rng.standard_normal(model.nv) * 0.3),
    ]

    for _config_name, qpos in configurations:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)

        if analyzer.club_head_id is not None:
            # Compute Jacobian
            jacp, _jacr = analyzer.compute_body_jacobian(analyzer.club_head_id)

            # Compute manipulability
            metrics = analyzer.compute_manipulability(jacp)

            # Note: Check for singularities in the metrics
            _ = metrics.is_near_singularity  # Available for analysis


def example_3_inverse_kinematics() -> None:
    """Example 3: Solve inverse kinematics to reach target position.

    This demonstrates the IK solver with nullspace optimization.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    analyzer = AdvancedKinematicsAnalyzer(model, data)

    if analyzer.club_head_id is None:
        return

    # Target position (slightly forward and to the side)
    current_pos = data.xpos[analyzer.club_head_id].copy()
    target_pos = current_pos + np.array([0.3, -0.2, 0.1])

    # Solve IK
    q_solution, success, _iterations = analyzer.solve_inverse_kinematics(
        target_body_id=analyzer.club_head_id,
        target_position=target_pos,
        q_init=data.qpos.copy(),
    )

    if success:
        # Verify solution
        data.qpos[:] = q_solution
        mujoco.mj_forward(model, data)
        achieved_pos = data.xpos[analyzer.club_head_id].copy()
        np.linalg.norm(achieved_pos - target_pos)


def example_4_impedance_control() -> None:
    """Example 4: Implement impedance control.

    This shows compliant control useful for robotic golf swing.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    controller = AdvancedController(model, data)

    # Configure impedance parameters
    # High stiffness for rigid control, low for compliant
    stiffness = np.ones(model.nv) * 50.0  # Medium stiffness
    damping = 2.0 * np.sqrt(stiffness)  # Critical damping

    params = ImpedanceParameters(stiffness=stiffness, damping=damping)

    controller.set_impedance_parameters(params)
    controller.set_control_mode(ControlMode.IMPEDANCE)

    # Simulate with impedance control
    target_position = np.zeros(model.nv)
    target_position[0] = 0.5  # Rotate spine

    # Simulate a few steps
    num_steps = 100
    for _step in range(num_steps):
        # Compute control
        tau = controller.compute_control(
            target_position=target_position,
            target_velocity=np.zeros(model.nv),
        )

        data.ctrl[:] = tau
        mujoco.mj_step(model, data)


def example_5_hybrid_force_position_control() -> None:
    """Example 5: Hybrid force-position control.

    This demonstrates simultaneous force and position control.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    controller = AdvancedController(model, data)

    # Configure hybrid control
    # Force control on first 2 DOFs, position control on rest
    force_mask = np.zeros(model.nv, dtype=bool)
    force_mask[:2] = True  # Force control for spine

    mask = HybridControlMask(force_mask=force_mask)
    controller.set_hybrid_mask(mask)
    controller.set_control_mode(ControlMode.HYBRID)

    # Set targets
    target_force = np.zeros(model.nv)
    target_force[:2] = 50.0  # 50 Nm torque on spine

    target_position = np.zeros(model.nv)
    target_position[2:] = 0.3  # Position control for arms


def example_6_trajectory_optimization() -> None:
    """Example 6: Optimize golf swing trajectory.

    This demonstrates trajectory optimization for maximum club speed.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    # Configure optimization
    objectives = OptimizationObjectives(
        maximize_club_speed=True,
        minimize_jerk=True,
        weight_speed=10.0,
        weight_jerk=0.5,
    )

    constraints = OptimizationConstraints(
        joint_position_limits=True,
        joint_velocity_limits=True,
        joint_torque_limits=True,
    )

    optimizer = SwingOptimizer(model, data, objectives, constraints)
    optimizer.num_knot_points = 5  # Coarse for example (use 10-20 for real)

    # Optimize
    result = optimizer.optimize_swing_for_speed(target_speed=45.0)

    # Note: result.success indicates optimization success
    _ = result.success  # Available for checking


def example_7_motion_primitive_library() -> None:
    """Example 7: Create and use motion primitive library.

    This shows how to build reusable motion libraries.
    """

    # Create library
    library = MotionPrimitiveLibrary()

    # Add some example primitives (normally these would be optimized)

    # Address position
    address = np.zeros((10, 28))
    library.add_primitive(
        "address",
        address,
        metadata={"description": "Standard address position", "duration": 0.5},
    )

    # Backswing
    rng = np.random.default_rng()
    backswing = rng.standard_normal((10, 28)) * 0.5
    library.add_primitive(
        "backswing",
        backswing,
        metadata={"description": "Full backswing", "duration": 0.7},
    )

    # Downswing
    downswing = rng.standard_normal((10, 28)) * 0.3
    library.add_primitive(
        "downswing",
        downswing,
        metadata={"description": "Powerful downswing", "duration": 0.3},
    )

    # Note: library.metadata contains primitive metadata for reference
    _ = list(library.metadata.items())  # Metadata available for inspection

    # Blend primitives
    blended = library.blend_primitives(
        ["backswing", "downswing"],
        weights=np.array([0.6, 0.4]),
    )

    # Note: blended contains the blended trajectory if successful
    _ = blended  # Available for use


def example_8_singularity_analysis() -> None:
    """Example 8: Analyze workspace for singularities.

    This shows comprehensive singularity analysis.
    """

    model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
    data = mujoco.MjData(model)

    analyzer = AdvancedKinematicsAnalyzer(model, data)

    if analyzer.club_head_id is None:
        return

    # Analyze singularities
    _singular_configs, condition_numbers = analyzer.analyze_singularities(
        body_id=analyzer.club_head_id,
        num_samples=50,
    )

    # Find most singular configuration
    if condition_numbers:
        np.argmax(condition_numbers)


def run_all_examples() -> None:
    """Run all examples."""
    examples = [
        example_1_constraint_jacobian_analysis,
        example_2_manipulability_analysis,
        example_3_inverse_kinematics,
        example_4_impedance_control,
        example_5_hybrid_force_position_control,
        example_6_trajectory_optimization,
        example_7_motion_primitive_library,
        example_8_singularity_analysis,
    ]

    for _i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception:  # noqa: BLE001 - Example runner should continue on any error
            traceback.print_exc()


if __name__ == "__main__":
    # Run all examples
    run_all_examples()
