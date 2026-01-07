

# Motion Capture Integration and Force Analysis Guide

**Version 2.1** - Professional Motion Capture Workflow for Golf Swing Analysis

---

## Table of Contents

1. [Overview](#overview)
2. [Motion Capture Integration](#motion-capture-integration)
3. [Kinematic-Dependent Forces](#kinematic-dependent-forces)
4. [Inverse Dynamics](#inverse-dynamics)
5. [Complete Workflow](#complete-workflow)
6. [API Reference](#api-reference)
7. [Use Cases](#use-cases)

---

## Overview

This guide covers the complete workflow for analyzing real golf swings captured with motion capture systems. The key innovation is the ability to compute **motion-dependent forces** (Coriolis, centrifugal) from kinematics alone, **without requiring full inverse dynamics**.

### Why This Matters

For parallel mechanisms like the golf swing (two hands on club), computing full inverse dynamics is challenging due to closed-chain constraints. However, many critical forces can be computed directly from kinematics:

✅ **Coriolis Forces** - Velocity coupling between joints
✅ **Centrifugal Forces** - Rotation-induced outward forces
✅ **Gravitational Forces** - Configuration-dependent weight
✅ **Kinetic Energy Distribution** - Rotational vs translational

These forces are crucial for understanding swing dynamics and can be computed even when full inverse dynamics is not feasible.

### Key Capabilities

- **Motion Capture Loading**: CSV, JSON formats (BVH with library)
- **Motion Retargeting**: IK-based mapping to model
- **Kinematic Force Analysis**: Coriolis, centrifugal, gravity
- **Inverse Dynamics**: Full and partial solutions
- **Swing Comparison**: Quantitative analysis
- **Force Decomposition**: Break down into components

---

## Motion Capture Integration

### 1. Loading Motion Capture Data

#### Supported Formats

##### CSV Format
```csv
time, LSHO_x, LSHO_y, LSHO_z, RSHO_x, RSHO_y, RSHO_z, ...
0.000, 0.0, 0.2, 1.4, 0.0, -0.2, 1.4, ...
0.008, 0.01, 0.21, 1.41, 0.01, -0.19, 1.41, ...
```

##### JSON Format
```json
{
  "frame_rate": 120.0,
  "marker_names": ["LSHO", "RSHO", "CLUB_HEAD", ...],
  "frames": [
    {
      "time": 0.0,
      "markers": {
        "LSHO": [0.0, 0.2, 1.4],
        "RSHO": [0.0, -0.2, 1.4],
        ...
      }
    },
    ...
  ]
}
```

#### Loading Code

```python
from mujoco_golf_pendulum.motion_capture import MotionCaptureLoader

# Load from CSV
mocap_seq = MotionCaptureLoader.load_csv(
    'player_swing.csv',
    frame_rate=120.0
)

# Load from JSON
mocap_seq = MotionCaptureLoader.load_json('player_swing.json')

print(f"Loaded {mocap_seq.num_frames} frames")
print(f"Duration: {mocap_seq.duration:.2f} seconds")
print(f"Markers: {mocap_seq.marker_names}")
```

---

### 2. Motion Retargeting

Motion retargeting maps marker positions to joint angles using inverse kinematics.

#### Marker Set Configuration

```python
from mujoco_golf_pendulum.motion_capture import MarkerSet

# Use standard golf marker set
marker_set = MarkerSet.golf_swing_marker_set()

print(f"Markers defined: {len(marker_set.markers)}")
# Includes: HEAD, C7, LSHO, RSHO, LELB, RELB, CLUB_HEAD, etc.
```

#### Retargeting Process

```python
from mujoco_golf_pendulum.motion_capture import MotionRetargeting
import mujoco

# Load model
model = mujoco.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
data = mujoco.MjData(model)

# Create retargeting system
retargeting = MotionRetargeting(model, data, marker_set)

# Retarget sequence
times, joint_trajectories, success_flags = retargeting.retarget_sequence(
    mocap_seq,
    use_markers=['LSHO', 'RSHO', 'CLUB_HEAD', ...],  # Key markers
    ik_iterations=50  # IK solver iterations
)

success_rate = 100.0 * sum(success_flags) / len(success_flags)
print(f"Success rate: {success_rate:.1f}%")
print(f"Joint trajectories: {joint_trajectories.shape}")  # [N x nv]
```

#### IK Solver Details

The retargeting uses a multi-target IK solver:
- **Algorithm**: Damped Least-Squares (singularity-robust)
- **Objective**: Minimize error to all marker positions
- **Constraints**: Joint limits automatically enforced
- **Iteration**: Typically converges in 20-50 iterations

---

### 3. Kinematic Processing

#### Filtering Noisy Data

```python
from mujoco_golf_pendulum.motion_capture import MotionCaptureProcessor

processor = MotionCaptureProcessor()

# Low-pass filter (removes high-frequency noise)
filtered_trajectory = processor.filter_trajectory(
    times,
    joint_trajectories,
    cutoff_frequency=10.0,  # Hz
    sampling_rate=120.0     # Hz
)
```

#### Computing Derivatives

```python
# Velocities
velocities = processor.compute_velocities(
    times,
    filtered_trajectory,
    method="spline"  # or "finite_difference"
)

# Accelerations
accelerations = processor.compute_accelerations(
    times,
    velocities,
    method="spline"
)
```

#### Time Normalization

```python
# Normalize to 0-100% of swing
normalized_times, normalized_traj = processor.time_normalize(
    times,
    filtered_trajectory,
    num_samples=101  # 0%, 1%, ..., 100%
)
```

---

## Kinematic-Dependent Forces

### Overview

These forces can be computed from kinematics (position, velocity, acceleration) **without needing full inverse dynamics**. This is the KEY capability for analyzing parallel mechanisms.

### Force Types

#### 1. Coriolis Forces

Arise from velocity coupling between joints:

```
F_coriolis = C(q,q̇)q̇
```

**Physical Meaning**: When one joint moves, it induces forces on other joints due to the changing reference frame.

#### 2. Centrifugal Forces

Rotation-induced outward forces:

```
F_centrifugal = ω²r
```

**Physical Meaning**: Spinning creates outward force proportional to rotation speed squared.

#### 3. Gravitational Forces

Configuration-dependent weight:

```
F_gravity = g(q)
```

**Physical Meaning**: Gravity acts differently at each joint depending on body orientations.

---

### Computing Kinematic Forces

```python
from mujoco_golf_pendulum.kinematic_forces import KinematicForceAnalyzer

# Create analyzer
analyzer = KinematicForceAnalyzer(model, data)

# Analyze entire trajectory
force_data_list = analyzer.analyze_trajectory(
    times,
    positions,
    velocities,
    accelerations
)

# Each frame contains:
for force_data in force_data_list:
    print(f"Time: {force_data.time:.3f}s")
    print(f"  Coriolis forces: {force_data.coriolis_forces}")
    print(f"  Centrifugal forces: {force_data.centrifugal_forces}")
    print(f"  Gravity forces: {force_data.gravity_forces}")
    print(f"  Coriolis power: {force_data.coriolis_power:.2f} W")
```

---

### Single Frame Analysis

```python
# Compute for a single configuration
qpos = positions[50]  # Frame 50
qvel = velocities[50]

# Coriolis forces
coriolis = analyzer.compute_coriolis_forces(qpos, qvel)

# Decompose into centrifugal and coupling
centrifugal, coupling = analyzer.decompose_coriolis_forces(qpos, qvel)

print(f"Coriolis forces: {coriolis}")
print(f"Centrifugal component: {centrifugal}")
print(f"Velocity coupling: {coupling}")
```

---

### Club Head Apparent Forces

Forces experienced at the club head in the rotating reference frame:

```python
qacc = accelerations[50]

club_coriolis, club_centrifugal, club_apparent = \
    analyzer.compute_club_head_apparent_forces(qpos, qvel, qacc)

print(f"Club head Coriolis force: {club_coriolis} N")
print(f"Club head centrifugal force: {club_centrifugal} N")
print(f"Total apparent force: {club_apparent} N")
```

---

### Power Analysis

```python
# Power contributions from different force types
power_dict = analyzer.compute_kinematic_power(qpos, qvel)

print(f"Coriolis power: {power_dict['coriolis_power']:.2f} W")
print(f"Centrifugal power: {power_dict['centrifugal_power']:.2f} W")
print(f"Gravity power: {power_dict['gravity_power']:.2f} W")
```

**Key Insight**: Coriolis power should ideally be zero (conservative force). Non-zero values indicate numerical errors or non-conservative effects.

---

### Kinetic Energy Decomposition

```python
ke_dict = analyzer.compute_kinetic_energy_components(qpos, qvel)

print(f"Rotational KE: {ke_dict['rotational']:.2f} J")
print(f"Translational KE: {ke_dict['translational']:.2f} J")
print(f"Total KE: {ke_dict['total']:.2f} J")

# Percentage
rot_percentage = 100 * ke_dict['rotational'] / ke_dict['total']
print(f"Rotational: {rot_percentage:.1f}%")
```

---

### Export Results

```python
from mujoco_golf_pendulum.kinematic_forces import export_kinematic_forces_to_csv

export_kinematic_forces_to_csv(
    force_data_list,
    'swing_forces.csv'
)
```

CSV includes:
- Time
- Coriolis forces (per joint)
- Centrifugal forces (per joint)
- Gravity forces (per joint)
- Club head forces (3D)
- Power contributions
- Kinetic energy

---

## Inverse Dynamics

### Overview

Inverse dynamics computes the joint torques required to produce a given motion:

```
M(q)q̈ + C(q,q̇)q̇ + g(q) = τ
```

Where:
- **M(q)**: Mass matrix (configuration-dependent)
- **C(q,q̇)q̇**: Coriolis and centrifugal forces
- **g(q)**: Gravitational forces
- **τ**: Joint torques (what we solve for)

---

### Computing Required Torques

```python
from mujoco_golf_pendulum.inverse_dynamics import InverseDynamicsSolver

# Create solver
solver = InverseDynamicsSolver(model, data)

# Solve for entire trajectory
id_results = solver.solve_inverse_dynamics_trajectory(
    times,
    positions,
    velocities,
    accelerations
)

# Each result contains:
for result in id_results:
    print(f"Joint torques: {result.joint_torques}")
    print(f"  Inertial (Ma): {result.inertial_torques}")
    print(f"  Coriolis (Cq̇): {result.coriolis_torques}")
    print(f"  Gravity (g): {result.gravity_torques}")
```

---

### Force Decomposition

Break down total torques into components:

```python
decomposition = solver.decompose_forces(qpos, qvel, qacc)

print("Force Decomposition:")
print(f"  Total: {decomposition.total}")
print(f"  Inertial: {decomposition.inertial}")
print(f"  Coriolis: {decomposition.coriolis}")
print(f"  Centrifugal: {decomposition.centrifugal}")
print(f"  Gravity: {decomposition.gravity}")
```

---

### End-Effector Forces

Map joint torques to task-space forces at club head:

```python
ee_force = solver.compute_end_effector_forces(
    qpos, qvel, qacc,
    body_id=club_head_id
)

print(f"Club head force: {ee_force} N")
```

---

### Validation

Verify that computed torques produce desired acceleration:

```python
validation = solver.validate_solution(
    qpos, qvel, qacc,
    computed_torques=id_results[0].joint_torques
)

print(f"Acceleration error: {validation['acceleration_error']:.6f} m/s²")
print(f"Relative error: {validation['relative_error']:.2%}")
print(f"Max torque: {validation['max_torque']:.2f} Nm")
```

---

### Partial Inverse Dynamics

For parallel mechanisms with constraints:

```python
# Specify which joints are constrained
constrained_joints = [5, 6]  # Example: wrist joints

result = solver.compute_partial_inverse_dynamics(
    qpos, qvel, qacc,
    constrained_joints=constrained_joints
)

print(f"Actuated joint torques: {result.joint_torques}")
print(f"Constraint forces: {result.constraint_forces}")
```

---

### Export Results

```python
from mujoco_golf_pendulum.inverse_dynamics import export_inverse_dynamics_to_csv

export_inverse_dynamics_to_csv(
    times,
    id_results,
    'swing_torques.csv'
)
```

---

## Complete Workflow

### Recommended Pipeline

```python
from mujoco_golf_pendulum.inverse_dynamics import InverseDynamicsAnalyzer

# Initialize analyzer (combines kinematic and inverse dynamics)
analyzer = InverseDynamicsAnalyzer(model, data)

# Complete analysis
results = analyzer.analyze_captured_motion(
    times,
    positions,
    velocities,
    accelerations
)

# Results contain:
kinematic_forces = results['kinematic_forces']
inverse_dynamics = results['inverse_dynamics']
statistics = results['statistics']

print(f"Peak Coriolis power: {statistics['peak_coriolis_power']:.2f} W")
print(f"Max joint torque: {statistics['max_joint_torque']:.2f} Nm")
```

---

### Step-by-Step Example

```python
# 1. Load motion capture
from mujoco_golf_pendulum.motion_capture import MotionCaptureLoader, MotionRetargeting

mocap_seq = MotionCaptureLoader.load_csv('player_swing.csv')

# 2. Retarget to model
model = mujoco.MjModel.from_xml_string(MODEL_XML)
data = mujoco.MjData(model)
marker_set = MarkerSet.golf_swing_marker_set()

retargeting = MotionRetargeting(model, data, marker_set)
times, joint_traj, success = retargeting.retarget_sequence(mocap_seq)

# 3. Filter and process
from mujoco_golf_pendulum.motion_capture import MotionCaptureProcessor

processor = MotionCaptureProcessor()
filtered_traj = processor.filter_trajectory(times, joint_traj)
velocities = processor.compute_velocities(times, filtered_traj, method="spline")
accelerations = processor.compute_accelerations(times, velocities, method="spline")

# 4. Analyze kinematic forces
from mujoco_golf_pendulum.kinematic_forces import KinematicForceAnalyzer

kin_analyzer = KinematicForceAnalyzer(model, data)
force_data = kin_analyzer.analyze_trajectory(
    times, filtered_traj, velocities, accelerations
)

# 5. Compute inverse dynamics
from mujoco_golf_pendulum.inverse_dynamics import InverseDynamicsSolver

id_solver = InverseDynamicsSolver(model, data)
id_results = id_solver.solve_inverse_dynamics_trajectory(
    times, filtered_traj, velocities, accelerations
)

# 6. Analyze results
max_coriolis_power = max(abs(fd.coriolis_power) for fd in force_data)
max_torque = max(np.max(np.abs(r.joint_torques)) for r in id_results)

print(f"Peak Coriolis power: {max_coriolis_power:.2f} W")
print(f"Peak torque: {max_torque:.2f} Nm")

# 7. Export
from mujoco_golf_pendulum.kinematic_forces import export_kinematic_forces_to_csv
from mujoco_golf_pendulum.inverse_dynamics import export_inverse_dynamics_to_csv

export_kinematic_forces_to_csv(force_data, 'forces.csv')
export_inverse_dynamics_to_csv(times, id_results, 'torques.csv')
```

---

## API Reference

### MotionCaptureLoader

```python
# Load CSV
mocap_seq = MotionCaptureLoader.load_csv(
    filepath='data.csv',
    frame_rate=120.0,
    marker_names=None  # Auto-detect from header
)

# Load JSON
mocap_seq = MotionCaptureLoader.load_json('data.json')
```

---

### MotionRetargeting

```python
retargeting = MotionRetargeting(model, data, marker_set)

times, joint_traj, success = retargeting.retarget_sequence(
    mocap_sequence,
    use_markers=['LSHO', 'RSHO', ...],
    ik_iterations=50
)

# Check marker errors
errors = retargeting.compute_marker_errors(frame, q)
```

---

### KinematicForceAnalyzer

```python
analyzer = KinematicForceAnalyzer(model, data)

# Full trajectory
force_data_list = analyzer.analyze_trajectory(times, pos, vel, acc)

# Single frame
coriolis = analyzer.compute_coriolis_forces(qpos, qvel)
gravity = analyzer.compute_gravity_forces(qpos)
centrifugal, coupling = analyzer.decompose_coriolis_forces(qpos, qvel)

# Club head forces
club_coriolis, club_centrifugal, club_apparent = \
    analyzer.compute_club_head_apparent_forces(qpos, qvel, qacc)

# Analysis
power_dict = analyzer.compute_kinematic_power(qpos, qvel)
ke_dict = analyzer.compute_kinetic_energy_components(qpos, qvel)
```

---

### InverseDynamicsSolver

```python
solver = InverseDynamicsSolver(model, data)

# Single frame
result = solver.compute_required_torques(qpos, qvel, qacc)

# Trajectory
id_results = solver.solve_inverse_dynamics_trajectory(times, pos, vel, acc)

# Decomposition
decomp = solver.decompose_forces(qpos, qvel, qacc)

# End-effector forces
ee_force = solver.compute_end_effector_forces(qpos, qvel, qacc, body_id)

# Validation
validation = solver.validate_solution(qpos, qvel, qacc, torques)
```

---

## Use Cases

### 1. Biomechanical Analysis

Understand the forces at play in a golf swing:

```python
# Identify dominant forces
for i, fd in enumerate(force_data_list):
    coriolis_norm = np.linalg.norm(fd.coriolis_forces)
    gravity_norm = np.linalg.norm(fd.gravity_forces)

    if coriolis_norm > 2 * gravity_norm:
        print(f"Frame {i}: Coriolis-dominated (dynamic coupling high)")
```

---

### 2. Swing Optimization

Compare different swing techniques:

```python
analyzer = InverseDynamicsAnalyzer(model, data)

analysis_1 = analyzer.analyze_captured_motion(times1, pos1, vel1, acc1)
analysis_2 = analyzer.analyze_captured_motion(times2, pos2, vel2, acc2)

comparison = analyzer.compare_swings(analysis_1, analysis_2)

print(f"Coriolis power difference: {comparison['coriolis_power_diff']:.2f} W")
print(f"Torque difference: {comparison['torque_diff']:.2f} Nm")
```

---

### 3. Player Assessment

Evaluate player's biomechanical efficiency:

```python
# Compute efficiency metrics
total_ke = sum(fd.rotational_kinetic_energy + fd.translational_kinetic_energy
               for fd in force_data_list)
avg_ke = total_ke / len(force_data_list)

peak_power = max(abs(fd.coriolis_power) for fd in force_data_list)

efficiency_score = avg_ke / peak_power  # Higher is better

print(f"Biomechanical efficiency: {efficiency_score:.2f}")
```

---

### 4. Injury Prevention

Identify excessive joint loading:

```python
# Check for dangerous torques
SAFE_TORQUE_LIMIT = 100.0  # Nm

for i, result in enumerate(id_results):
    max_torque = np.max(np.abs(result.joint_torques))

    if max_torque > SAFE_TORQUE_LIMIT:
        print(f"⚠️  Frame {i}: Excessive torque {max_torque:.1f} Nm")
        print(f"   Consider modifying technique to reduce loading")
```

---

### 5. Club Selection

Analyze how different clubs affect forces:

```python
# Compare driver vs iron swings
driver_forces = analyze_swing('driver_swing.csv')
iron_forces = analyze_swing('iron_swing.csv')

driver_peak = max(np.max(np.abs(fd.coriolis_forces)) for fd in driver_forces)
iron_peak = max(np.max(np.abs(fd.coriolis_forces)) for fd in iron_forces)

print(f"Driver peak Coriolis: {driver_peak:.2f} Nm")
print(f"Iron peak Coriolis: {iron_peak:.2f} Nm")
```

---

## Performance Tips

### Optimization

1. **Use spline derivatives** for smoother velocity/acceleration:
   ```python
   velocities = processor.compute_velocities(times, pos, method="spline")
   ```

2. **Filter before analysis** to reduce noise:
   ```python
   filtered = processor.filter_trajectory(times, pos, cutoff_frequency=10.0)
   ```

3. **Reduce IK iterations** for faster retargeting:
   ```python
   times, traj, success = retargeting.retarget_sequence(mocap, ik_iterations=20)
   ```

---

### Recommended Settings

| Task | Setting | Value |
|------|---------|-------|
| **Filter cutoff** | Motion capture | 10-15 Hz |
| **IK iterations** | Retargeting | 30-50 |
| **Derivative method** | Kinematics | "spline" |
| **Frame rate** | Mocap | 120+ Hz |

---

## Troubleshooting

### Common Issues

#### 1. IK Convergence Failure

**Problem**: Low success rate (<80%) in retargeting

**Solution**:
- Increase IK iterations
- Use more markers
- Check marker visibility
- Calibrate marker offsets

```python
# Validate marker visibility
from mujoco_golf_pendulum.motion_capture import MotionCaptureValidator

validator = MotionCaptureValidator()
visibility = validator.check_marker_visibility(mocap_seq, 'CLUB_HEAD')
print(f"Visibility: {visibility['visibility_percentage']:.1f}%")
```

---

#### 2. Noisy Force Profiles

**Problem**: Forces have high-frequency oscillations

**Solution**:
- Lower filter cutoff frequency
- Use spline derivatives
- Increase mocap frame rate

```python
# More aggressive filtering
filtered = processor.filter_trajectory(times, pos, cutoff_frequency=6.0)
```

---

#### 3. Large Coriolis Power

**Problem**: Coriolis power not near zero

**Solution**:
- Check numerical derivatives
- Verify mass matrix
- Look for constraint violations

```python
# Validate mass matrix
M = analyzer.compute_mass_matrix(qpos)
condition_number = np.linalg.cond(M)
print(f"Mass matrix condition: {condition_number:.2e}")
```

---

## References

### Biomechanics

1. **Zheng, N., et al.** (2008). "Kinematic analysis of swing sports."
2. **Nesbit, S.M.** (2005). "A three dimensional kinematic and kinetic study of the golf swing."

### Robotics

1. **Featherstone, R.** (2008). *Rigid Body Dynamics Algorithms*.
2. **Murray, R.M., et al.** (1994). *A Mathematical Introduction to Robotic Manipulation*.

### Motion Capture

1. **Cappozzo, A., et al.** (1995). "Position and orientation in space of bones."
2. **Winter, D.A.** (2009). *Biomechanics and Motor Control of Human Movement*.

---

**Version 2.1.0** | © 2024 | MIT License

For questions or issues, see the project GitHub repository.
