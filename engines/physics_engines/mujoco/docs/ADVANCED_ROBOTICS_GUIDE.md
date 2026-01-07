# Advanced Robotics Features Guide

**Version 2.0** - Professional-Grade Parallel Mechanism Analysis and Control

---

## Table of Contents

1. [Overview](#overview)
2. [Advanced Kinematics](#advanced-kinematics)
3. [Control Schemes](#control-schemes)
4. [Motion Optimization](#motion-optimization)
5. [Quick Start Examples](#quick-start-examples)
6. [API Reference](#api-reference)
7. [Applications](#applications)

---

## Overview

This guide covers the professional-grade robotics features added to the MuJoCo Golf Swing Analysis Suite. These features bring **leading-edge parallel mechanism robotics capabilities** to golf swing analysis.

### Key Features

✅ **Constraint Jacobian Analysis** - Analyze closed-chain systems (two-handed grip is a parallel mechanism)
✅ **Manipulability Analysis** - Singularity detection and avoidance
✅ **Inverse Kinematics** - Professional IK solver with nullspace optimization
✅ **Multiple Control Schemes** - Impedance, admittance, hybrid force-position control
✅ **Trajectory Optimization** - Generate optimal swing motions
✅ **Motion Planning** - Synthesize complex multi-phase movements

### Why This Matters

The golf swing is fundamentally a **parallel mechanism** - both hands on the club create a closed kinematic chain. Professional robotics analysis requires:

1. **Constraint Analysis**: Understanding how the two-handed grip constrains motion
2. **Singularity Avoidance**: Identifying configurations where control becomes difficult
3. **Optimal Control**: Generating efficient, high-performance swings
4. **Multi-Objective Optimization**: Balancing speed, accuracy, and biomechanical safety

---

## Advanced Kinematics

### Module: `advanced_kinematics.py`

This module provides state-of-the-art kinematics analysis tools.

### 1. Constraint Jacobian Analysis

The two-handed grip creates a **closed-chain constraint**. The constraint Jacobian describes how this constraint affects motion.

#### Theory

For a parallel mechanism with constraints **Φ(q) = 0**, the constraint Jacobian is:

```
J_c = ∂Φ/∂q
```

The **nullspace** of J_c represents internal motions that satisfy the constraint.

#### Code Example

```python
from mujoco_golf_pendulum import advanced_kinematics
import mujoco

# Load model
model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
data = mujoco.MjData(model)

# Create analyzer
analyzer = advanced_kinematics.AdvancedKinematicsAnalyzer(model, data)

# Compute constraint Jacobian
constraint_data = analyzer.compute_constraint_jacobian()

print(f"Constraint Jacobian shape: {constraint_data.constraint_jacobian.shape}")
print(f"Nullspace dimension: {constraint_data.nullspace_dimension}")
print(f"Rank: {constraint_data.rank}")
```

#### Applications

- **Parallel Mechanism Analysis**: Understand how two-handed grip constrains motion
- **Redundancy Resolution**: Use nullspace for secondary objectives
- **Constraint-Aware Control**: Design controllers that respect closed-chain constraints

---

### 2. Manipulability and Singularity Analysis

**Manipulability** measures how easily the end-effector can move in different directions. **Singularities** are configurations where manipulability drops to zero.

#### Metrics

1. **Yoshikawa's Manipulability Index**: `w = √det(JJ^T)`
2. **Condition Number**: `κ = σ_max / σ_min` (σ = singular values)
3. **Minimum Singular Value**: Indicates proximity to singularity

#### Code Example

```python
# Compute Jacobian for club head
jacp, jacr = analyzer.compute_body_jacobian(analyzer.club_head_id)

# Compute manipulability
metrics = analyzer.compute_manipulability(jacp)

print(f"Manipulability index: {metrics.manipulability_index}")
print(f"Condition number: {metrics.condition_number}")
print(f"Near singularity: {metrics.is_near_singularity}")
```

#### Interpreting Results

- **High manipulability (> 0.1)**: Good dexterity in all directions
- **Low manipulability (< 0.01)**: Limited motion capability
- **Condition number > 30**: Near singular (avoid this configuration)
- **Condition number < 10**: Well-conditioned (good for control)

---

### 3. Inverse Kinematics

Professional IK solver using **Damped Least-Squares** with nullspace optimization.

#### Features

✅ Singularity-robust (damped pseudo-inverse)
✅ Nullspace projection for redundancy resolution
✅ Joint limit handling
✅ Iterative refinement with convergence checking

#### Code Example

```python
# Target position for club head
target_position = np.array([2.0, 0.0, 0.5])  # 2m forward, 0.5m up

# Solve IK
q_solution, success, iterations = analyzer.solve_inverse_kinematics(
    target_body_id=analyzer.club_head_id,
    target_position=target_position,
    q_init=data.qpos.copy(),
    nullspace_objective=None  # Optional: preferred joint configuration
)

if success:
    print(f"IK converged in {iterations} iterations")
    data.qpos[:] = q_solution
    mujoco.mj_forward(model, data)
```

#### Applications

- **Motion Planning**: Generate feasible paths to target positions
- **Motion Capture Retargeting**: Map recorded motions to model
- **Trajectory Optimization**: Find joint configurations for desired end-effector poses

---

## Control Schemes

### Module: `advanced_control.py`

This module implements **multiple professional control strategies** used in industrial robotics.

### Control Modes

| Mode | Description | Best For |
|------|-------------|----------|
| **Torque** | Direct torque control | Precise low-level control |
| **Impedance** | Position-based compliant control | Safe interaction, natural motion |
| **Admittance** | Force-based compliant control | Force-sensitive tasks |
| **Hybrid** | Combined force-position control | Simultaneous force and position objectives |
| **Computed Torque** | Model-based feedforward | High-precision tracking |
| **Task Space** | Cartesian-space control | End-effector control |

---

### 1. Impedance Control

Creates a **virtual spring-damper** system:

```
τ = K(q_d - q) + D(q̇_d - q̇) + g(q)
```

#### Theory

Impedance control makes the system behave like a mass-spring-damper:
- **High stiffness**: Rigid position tracking
- **Low stiffness**: Compliant, soft interaction
- **Critical damping**: No oscillations

#### Code Example

```python
from mujoco_golf_pendulum.advanced_control import (
    AdvancedController,
    ControlMode,
    ImpedanceParameters
)

# Create controller
controller = AdvancedController(model, data)

# Configure impedance parameters
params = ImpedanceParameters(
    stiffness=np.ones(model.nv) * 100.0,  # 100 Nm/rad
    damping=np.ones(model.nv) * 20.0      # 20 Nms/rad (critical damping)
)

controller.set_impedance_parameters(params)
controller.set_control_mode(ControlMode.IMPEDANCE)

# Compute control
target_position = np.zeros(model.nv)
target_position[0] = 0.5  # Desired spine rotation

tau = controller.compute_control(
    target_position=target_position,
    target_velocity=np.zeros(model.nv)
)

data.ctrl[:] = tau
```

#### Tuning Guidelines

- **For rigid tracking**: K = 200 Nm/rad, D = 40 Nms/rad
- **For compliant interaction**: K = 50 Nm/rad, D = 10 Nms/rad
- **For very soft**: K = 10 Nm/rad, D = 5 Nms/rad
- **Critical damping**: D = 2√(K·I) where I = inertia

---

### 2. Admittance Control

**Dual of impedance control** - modifies position based on force error:

```
Δq̈ = M^{-1}(F_d - F)
```

#### When to Use

- When you have **good force sensing**
- For **force-guided tasks** (e.g., maintaining grip pressure)
- When you want the system to **move away from obstacles**

#### Code Example

```python
controller.set_control_mode(ControlMode.ADMITTANCE)

# Desired force
target_force = np.zeros(model.nv)
target_force[0] = 50.0  # 50 Nm on spine

tau = controller.compute_control(target_force=target_force)
```

---

### 3. Hybrid Force-Position Control

**Simultaneously** control force in some directions and position in others.

#### Theory

Selection matrices split control:

```
τ = S_p τ_position + S_f τ_force
```

where S_p and S_f are complementary selection matrices.

#### Code Example

```python
from mujoco_golf_pendulum.advanced_control import HybridControlMask

# Force control on first 2 DOFs, position control on rest
force_mask = np.zeros(model.nv, dtype=bool)
force_mask[:2] = True  # Force control for spine

mask = HybridControlMask(force_mask=force_mask)
controller.set_hybrid_mask(mask)
controller.set_control_mode(ControlMode.HYBRID)

# Set both targets
target_position = np.zeros(model.nv)
target_position[2:] = 0.3  # Position control for arms

target_force = np.zeros(model.nv)
target_force[:2] = 50.0  # Force control for spine

tau = controller.compute_control(
    target_position=target_position,
    target_force=target_force
)
```

#### Applications

- **Maintain grip force** while moving club
- **Balance control** (force control for feet, position for upper body)
- **Contact tasks** (force control in contact directions)

---

### 4. Computed Torque Control

**Model-based feedforward** control using inverse dynamics:

```
τ = M(q)q̈_d + C(q,q̇)q̇ + g(q)
```

#### Benefits

✅ **Linearizes** the nonlinear system
✅ **Compensates** for dynamics
✅ **High precision** tracking

#### Code Example

```python
controller.set_control_mode(ControlMode.COMPUTED_TORQUE)

tau = controller.compute_control(
    target_position=target_position,
    target_velocity=target_velocity
)
```

---

### 5. Operational Space Control

**Task-space control** that accounts for configuration-dependent inertia:

```
F = Λ(q)(ẍ_d + K_d ė + K_p e) + μ(q,q̇) + p(q)
```

where Λ is the task-space inertia matrix.

#### Code Example

```python
# Control club head in Cartesian space
target_position = np.array([2.0, 0.0, 0.5])
target_velocity = np.array([0.0, 0.0, 0.0])
target_acceleration = np.array([0.0, 0.0, 0.0])

tau = controller.compute_operational_space_control(
    target_position=target_position,
    target_velocity=target_velocity,
    target_acceleration=target_acceleration,
    body_id=controller.club_head_id
)
```

---

## Motion Optimization

### Module: `motion_optimization.py`

Generate **optimal golf swing trajectories** using advanced optimization techniques.

### Optimization Framework

The trajectory optimization problem:

```
minimize    J(q(t), u(t))
subject to  q̇ = f(q, u)           (dynamics)
            q_min ≤ q ≤ q_max       (joint limits)
            u_min ≤ u ≤ u_max       (torque limits)
            |q̇| ≤ v_max            (velocity limits)
```

### Objectives

1. **Maximize Club Speed**: Maximize club head velocity at impact
2. **Minimize Energy**: Reduce total energy expenditure
3. **Minimize Jerk**: Ensure smooth motion
4. **Minimize Torque**: Reduce actuator effort
5. **Accuracy**: Hit specific target position

---

### Code Example: Optimize for Speed

```python
from mujoco_golf_pendulum.motion_optimization import (
    SwingOptimizer,
    OptimizationObjectives,
    OptimizationConstraints
)

# Configure objectives
objectives = OptimizationObjectives(
    maximize_club_speed=True,
    minimize_jerk=True,
    weight_speed=10.0,
    weight_jerk=0.5
)

# Configure constraints
constraints = OptimizationConstraints(
    joint_position_limits=True,
    joint_velocity_limits=True,
    joint_torque_limits=True
)

# Create optimizer
optimizer = SwingOptimizer(model, data, objectives, constraints)
optimizer.num_knot_points = 10
optimizer.swing_duration = 1.5  # seconds

# Optimize
result = optimizer.optimize_swing_for_speed(target_speed=45.0)  # m/s

if result.success:
    print(f"Peak club speed: {result.peak_club_speed:.2f} m/s")
    print(f"                 ({result.peak_club_speed * 2.237:.1f} mph)")

    # Use optimal trajectory
    optimal_trajectory = result.optimal_trajectory
    optimal_controls = result.optimal_controls
```

---

### Multi-Objective Optimization

Balance multiple objectives with weights:

```python
objectives = OptimizationObjectives(
    maximize_club_speed=True,
    minimize_energy=True,
    minimize_jerk=True,
    target_ball_position=np.array([200.0, 0.0, 0.0]),  # 200m target
    weight_speed=10.0,
    weight_energy=1.0,
    weight_jerk=0.5,
    weight_accuracy=5.0
)
```

---

### Generate Swing Library

Create multiple swing variants:

```python
# Generate 10 swings with different speeds
swings = optimizer.generate_library_of_swings(
    num_swings=10,
    variation="speed"  # or "accuracy", "style"
)

for i, swing in enumerate(swings):
    print(f"Swing {i+1}: {swing.peak_club_speed:.1f} m/s "
          f"({swing.peak_club_speed * 2.237:.0f} mph)")
```

---

### Motion Primitive Library

Store and reuse optimized motions:

```python
from mujoco_golf_pendulum.motion_optimization import MotionPrimitiveLibrary

# Create library
library = MotionPrimitiveLibrary()

# Add primitives (from optimization)
library.add_primitive(
    "power_swing",
    result.optimal_trajectory,
    metadata={"speed": result.peak_club_speed, "energy": result.objective_value}
)

# Blend primitives
blended = library.blend_primitives(
    ["power_swing", "control_swing"],
    weights=np.array([0.7, 0.3])
)

# Save/load library
library.save_library("my_swing_library.npz")
library.load_library("my_swing_library.npz")
```

---

## Quick Start Examples

### Example 1: Analyze Parallel Mechanism Constraints

```python
from mujoco_golf_pendulum import advanced_kinematics
from mujoco_golf_pendulum.models import ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
import mujoco

model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
data = mujoco.MjData(model)

analyzer = advanced_kinematics.AdvancedKinematicsAnalyzer(model, data)

# Analyze closed-chain constraints
constraint_data = analyzer.compute_constraint_jacobian()

print(f"Nullspace dimension: {constraint_data.nullspace_dimension}")
print(f"System has {constraint_data.nullspace_dimension} internal DOFs")
```

---

### Example 2: Implement Impedance Control

```python
from mujoco_golf_pendulum.advanced_control import *

controller = AdvancedController(model, data)

params = ImpedanceParameters(
    stiffness=np.ones(model.nv) * 100.0,
    damping=np.ones(model.nv) * 20.0
)

controller.set_impedance_parameters(params)
controller.set_control_mode(ControlMode.IMPEDANCE)

# Simulation loop
for step in range(1000):
    tau = controller.compute_control(
        target_position=target_pos,
        target_velocity=np.zeros(model.nv)
    )

    data.ctrl[:] = tau
    mujoco.mj_step(model, data)
```

---

### Example 3: Optimize Golf Swing

```python
from mujoco_golf_pendulum.motion_optimization import *

optimizer = SwingOptimizer(model, data)

# Optimize for maximum speed
result = optimizer.optimize_swing_for_speed(target_speed=45.0)

print(f"Optimized swing: {result.peak_club_speed * 2.237:.1f} mph")

# Simulate optimal trajectory
for step, q in enumerate(result.optimal_trajectory):
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    # ... render ...
```

---

## API Reference

### AdvancedKinematicsAnalyzer

**Methods:**

| Method | Description | Returns |
|--------|-------------|---------|
| `compute_body_jacobian(body_id)` | Compute Jacobian for body | `(jacp, jacr)` |
| `compute_constraint_jacobian()` | Analyze closed-chain constraints | `ConstraintJacobianData` |
| `compute_manipulability(jacobian)` | Compute manipulability metrics | `ManipulabilityMetrics` |
| `solve_inverse_kinematics(...)` | Solve IK to target position | `(q_solution, success, iters)` |
| `analyze_singularities(...)` | Find singular configurations | `(configs, condition_numbers)` |
| `compute_manipulability_ellipsoid(...)` | Compute visualization ellipsoid | `(center, radii, axes)` |

---

### AdvancedController

**Control Modes:**

- `ControlMode.TORQUE` - Direct torque
- `ControlMode.IMPEDANCE` - Impedance control
- `ControlMode.ADMITTANCE` - Admittance control
- `ControlMode.HYBRID` - Hybrid force-position
- `ControlMode.COMPUTED_TORQUE` - Computed torque
- `ControlMode.TASK_SPACE` - Task-space control

**Methods:**

| Method | Description |
|--------|-------------|
| `set_control_mode(mode)` | Set control mode |
| `set_impedance_parameters(params)` | Configure impedance |
| `set_hybrid_mask(mask)` | Configure hybrid control |
| `compute_control(...)` | Compute control torques |
| `compute_operational_space_control(...)` | OSC control |

---

### SwingOptimizer

**Methods:**

| Method | Description |
|--------|-------------|
| `optimize_trajectory()` | General trajectory optimization |
| `optimize_swing_for_speed(target_speed)` | Optimize for club speed |
| `optimize_swing_for_accuracy(target_pos)` | Optimize for accuracy |
| `generate_library_of_swings(...)` | Generate swing variations |

---

## Applications

### 1. Biomechanical Analysis

- **Optimal Swing Synthesis**: Generate biomechanically optimal swings
- **Injury Prevention**: Analyze joint torques and stress
- **Performance Enhancement**: Identify efficiency improvements

### 2. Golf Training

- **Personalized Coaching**: Generate optimal swings for individual physiology
- **Swing Comparison**: Compare student swings to optimized references
- **Progress Tracking**: Monitor improvement over time

### 3. Club Design

- **Shaft Optimization**: Analyze club-body interaction
- **Custom Fitting**: Optimize club parameters for player
- **Performance Prediction**: Simulate different equipment

### 4. Robotics Research

- **Parallel Mechanism Analysis**: Study closed-chain systems
- **Control Algorithm Development**: Test new control strategies
- **Motion Planning**: Develop trajectory planning algorithms

### 5. Virtual Coaching

- **Real-Time Feedback**: Provide immediate swing corrections
- **Motion Capture Integration**: Analyze recorded swings
- **Simulation Training**: Practice in virtual environment

---

## Performance Considerations

### Optimization Speed

- **Direct optimization**: 1-5 minutes for 10 knot points
- **Sampling-based**: Faster but less optimal
- **GPU acceleration**: 10-100x speedup (future)

### Real-Time Control

- **Impedance/Admittance**: <1ms compute time
- **Computed Torque**: ~1ms compute time
- **Operational Space**: ~2ms compute time

### Recommended Settings

- **Prototyping**: 5-10 knot points, SLSQP optimizer
- **Production**: 15-20 knot points, differential evolution
- **Real-time**: Pre-computed primitives, runtime blending

---

## References

### Robotics

1. **Craig, J.J.** (2005). *Introduction to Robotics: Mechanics and Control*. Pearson.
2. **Siciliano, B., et al.** (2009). *Robotics: Modelling, Planning and Control*. Springer.
3. **Khatib, O.** (1987). "A unified approach for motion and force control of robot manipulators." *IEEE Journal on Robotics and Automation*.

### Golf Biomechanics

1. **Zheng, N., et al.** (2008). "Kinematic analysis of golf swing." *Journal of Sports Science*.
2. **McNitt-Gray, J.L., et al.** (2013). "Biomechanics of the golf swing." *Sports Biomechanics*.

### Optimization

1. **Betts, J.T.** (2010). *Practical Methods for Optimal Control Using Nonlinear Programming*. SIAM.
2. **Kelly, M.** (2017). "An introduction to trajectory optimization." *SIAM Review*.

---

## Support and Contributing

For questions, issues, or contributions:

- **GitHub Issues**: https://github.com/yourusername/MuJoCo_Golf_Swing_Model/issues
- **Documentation**: See `docs/` directory
- **Examples**: See `examples_advanced_robotics.py`

---

**Version 2.0.0** | © 2024 | MIT License
