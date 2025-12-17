# Universal and Gimbal Joint Models for Golf Swing Simulation

This guide covers the new models featuring universal joints, gimbal joints, and flexible club shafts for advanced golf swing biomechanics research.

## Table of Contents

1. [Overview](#overview)
2. [Models](#models)
3. [Joint Types](#joint-types)
4. [Golf Club Configurations](#golf-club-configurations)
5. [Analysis Tools](#analysis-tools)
6. [Examples](#examples)
7. [Theory](#theory)

## Overview

The enhanced golf swing models include:

- **Universal joints** at the wrists for realistic torque transmission
- **Gimbal joints** at the shoulders for full 3-DOF rotation
- **Flexible beam models** for golf club shafts
- **Realistic club geometries** with proper inertial properties
- **Constraint force analysis** tools for studying joint mechanics

## Models

### 1. Two-Link Inclined Plane Model (`TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML`)

A simplified model featuring:
- Base mounted on 15° inclined plane
- Single shoulder hinge joint
- **Universal joint at wrist** (2 perpendicular hinges)
- Sensors for constraint forces and torques

**Key Features:**
- Ideal for studying universal joint torque wobble
- Inclined plane simulates swing plane geometry
- Torque sensors at shoulder and wrist

**Joint Configuration:**
```
shoulder (hinge) → upper_arm → wrist_universal_1 (hinge, Y-axis)
                                     ↓
                              wrist_universal_2 (hinge, X-axis)
                                     ↓
                                   club
```

**Usage:**
```python
from mujoco_golf_pendulum.models import TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML
import mujoco as mj

model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
data = mj.MjData(model)
```

### 2. Gimbal Joint Demonstration Model (`GIMBAL_JOINT_DEMO_XML`)

A pure gimbal joint demonstration with three nested rings:

**Structure:**
- Outer ring: Z-axis rotation
- Middle ring: Y-axis rotation (perpendicular to outer)
- Inner ring: X-axis rotation (perpendicular to middle)
- Central payload platform

**Key Features:**
- Demonstrates gimbal lock phenomenon
- Full 3-DOF orientation control
- Visual representation of gimbal mechanics

**Applications:**
- Understanding shoulder joint kinematics
- Testing gimbal lock detection
- Studying Euler angle singularities

**Usage:**
```python
from mujoco_golf_pendulum.models import GIMBAL_JOINT_DEMO_XML
import mujoco as mj

model = mj.MjModel.from_xml_string(GIMBAL_JOINT_DEMO_XML)
data = mj.MjData(model)

# Check for gimbal lock
from mujoco_golf_pendulum.joint_analysis import GimbalJointAnalyzer
analyzer = GimbalJointAnalyzer(model, data)
is_locked, distance = analyzer.check_gimbal_lock("gimbal_x", "gimbal_y", "gimbal_z")
```

### 3. Advanced Biomechanical Model (`ADVANCED_BIOMECHANICAL_GOLF_SWING_XML`)

Full-body golf swing model with anatomically accurate joint types:

**Joint Configuration:**

| Body Part | Joint Type | DOF | Implementation |
|-----------|------------|-----|----------------|
| Shoulders | Gimbal | 3 | flexion, abduction, rotation |
| Scapula | Universal | 2 | elevation, protraction |
| Elbows | Revolute | 1 | flexion/extension |
| Wrists | Universal | 2 | flexion, deviation |
| Spine | Universal + Revolute | 3 | lateral, sagittal, rotation |
| Ankles | Universal | 2 | plantar, inversion |
| Knees | Revolute | 1 | flexion/extension |

**Key Features:**
- Flexible 3-segment club shaft
- Realistic anthropometric parameters
- Constraint forces at all joints
- Optional rigid or flexible club

**Club Attachment:**
- Right hand holds club via flexible shaft
- Left hand connects via weld constraint
- Configurable shaft stiffness and segments

## Joint Types

### Universal Joint

A **universal joint** (U-joint) allows rotation about two perpendicular axes. Implemented in MuJoCo as two sequential hinge joints.

**Characteristics:**
- 2 degrees of freedom
- Exhibits torque wobble when bent
- Used in automotive driveshafts, wrists

**Torque Wobble:**
The velocity ratio between input and output shafts varies as:

```
ω_out/ω_in = cos(β) / (1 - sin²(β)sin²(θ))
```

where:
- `β` = angle between shafts
- `θ` = rotation angle
- Wobble increases with larger `β`

**Implementation:**
```xml
<body name="wrist">
  <joint name="wrist_u1" type="hinge" axis="0 1 0" range="-1.57 1.57"/>
  <joint name="wrist_u2" type="hinge" axis="1 0 0" range="-1.57 1.57"/>
  <!-- child bodies -->
</body>
```

### Gimbal Joint

A **gimbal joint** provides 3 DOF rotation via three nested rings rotating about perpendicular axes.

**Characteristics:**
- 3 degrees of freedom
- Can reach any orientation (except gimbal lock)
- Avoids torque wobble
- Used in gyroscopes, camera mounts, shoulder joints

**Gimbal Lock:**
Occurs when middle ring is at ±90°, causing loss of one DOF.

**Implementation:**
```xml
<body name="gimbal_outer">
  <joint name="gimbal_z" type="hinge" axis="0 0 1"/>
  <body name="gimbal_middle">
    <joint name="gimbal_y" type="hinge" axis="0 1 0"/>
    <body name="gimbal_inner">
      <joint name="gimbal_x" type="hinge" axis="1 0 0"/>
      <!-- payload -->
    </body>
  </body>
</body>
```

### Comparison

| Feature | Universal Joint | Gimbal Joint |
|---------|----------------|--------------|
| DOF | 2 | 3 |
| Torque Wobble | Yes | No |
| Singularities | None | Gimbal lock at ±90° |
| Complexity | Simple | Complex |
| Golf Swing Use | Wrists, scapula | Shoulders |

## Golf Club Configurations

### Available Clubs

Three realistic club configurations are provided:

#### Driver
- **Length:** 116 cm
- **Head mass:** 198 g
- **Loft:** 10°
- **Flex:** Soft (stiffness: 180/150/120 N⋅m/rad)
- **Use:** Maximum distance off the tee

#### 7-Iron
- **Length:** 95 cm
- **Head mass:** 253 g
- **Loft:** 32°
- **Flex:** Medium (stiffness: 220/200/180 N⋅m/rad)
- **Use:** Approach shots, 150-170 yards

#### Wedge
- **Length:** 90 cm
- **Head mass:** 288 g
- **Loft:** 55°
- **Flex:** Stiff (stiffness: 240/220/200 N⋅m/rad)
- **Use:** Short game, high trajectory

### Shaft Options

#### Rigid Shaft
```python
from mujoco_golf_pendulum.models import generate_rigid_club_xml

club_xml = generate_rigid_club_xml("driver")
# Single rigid body from grip to head
```

#### Flexible Shaft (Beam Model)
```python
from mujoco_golf_pendulum.models import generate_flexible_club_xml

# 3-segment flexible shaft (default)
club_xml = generate_flexible_club_xml("driver", num_segments=3)

# 5-segment for higher fidelity
club_xml = generate_flexible_club_xml("driver", num_segments=5)
```

**Flexible Shaft Parameters:**
- Each segment has independent stiffness
- Damping decreases toward club head
- Stiffness values from real club measurements

### Custom Club Configuration

Modify `CLUB_CONFIGS` dictionary in `models.py`:

```python
CLUB_CONFIGS["custom_club"] = {
    "grip_length": 0.28,
    "grip_radius": 0.0145,
    "grip_mass": 0.050,
    "shaft_length": 1.10,
    "shaft_radius": 0.0062,
    "shaft_mass": 0.065,
    "head_mass": 0.198,
    "head_size": [0.062, 0.048, 0.038],
    "total_length": 1.16,
    "club_loft": 0.17,
    "flex_stiffness": [200, 175, 150],
}
```

## Analysis Tools

### Universal Joint Analysis

```python
from mujoco_golf_pendulum.joint_analysis import UniversalJointAnalyzer

analyzer = UniversalJointAnalyzer(model, data)

# Get joint forces
force = analyzer.get_joint_forces("wrist_universal_1")

# Analyze torque wobble
results = analyzer.analyze_torque_transmission(
    "wrist_universal_1",
    "wrist_universal_2",
    num_cycles=2
)

print(f"Wobble amplitude: {results['wobble_amplitude']:.4f}")
```

### Gimbal Joint Analysis

```python
from mujoco_golf_pendulum.joint_analysis import GimbalJointAnalyzer

analyzer = GimbalJointAnalyzer(model, data)

# Get gimbal angles
x, y, z = analyzer.get_gimbal_angles("gimbal_x", "gimbal_y", "gimbal_z")

# Check for gimbal lock
is_locked, distance = analyzer.check_gimbal_lock(
    "gimbal_x", "gimbal_y", "gimbal_z",
    threshold=0.087  # 5 degrees
)
```

### Constraint Force Recording

```python
from mujoco_golf_pendulum.joint_analysis import (
    analyze_constraint_forces_over_time,
    plot_constraint_forces
)

# Record forces over 2-second simulation
force_data = analyze_constraint_forces_over_time(
    model, data,
    joint_names=["shoulder", "wrist_universal_1", "wrist_universal_2"],
    duration=2.0
)

# Plot results
plot_constraint_forces(force_data, joint_names=["shoulder", "wrist_universal_1"])
```

## Examples

### Example 1: Analyze Universal Joint Wobble

```python
import numpy as np
import mujoco as mj
from mujoco_golf_pendulum.models import TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML
from mujoco_golf_pendulum.joint_analysis import UniversalJointAnalyzer, plot_torque_wobble

# Load model
model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
data = mj.MjData(model)

# Set wrist to bent position
wrist_u1_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "wrist_universal_1")
wrist_u2_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, "wrist_universal_2")
data.qpos[model.jnt_qposadr[wrist_u1_id]] = np.radians(30)
data.qpos[model.jnt_qposadr[wrist_u2_id]] = np.radians(30)

# Analyze
analyzer = UniversalJointAnalyzer(model, data)
results = analyzer.analyze_torque_transmission("wrist_universal_1", "wrist_universal_2")

# Visualize
plot_torque_wobble(results)
```

### Example 2: Compare Flexible vs Rigid Clubs

```python
from mujoco_golf_pendulum.models import generate_flexible_club_xml, generate_rigid_club_xml

# Generate both versions
rigid_club = generate_rigid_club_xml("driver")
flex_club = generate_flexible_club_xml("driver", num_segments=3)

# Create models and compare shaft deflection during swing
# (see examples_joint_analysis.py for complete implementation)
```

### Example 3: Full-Body Swing with Force Analysis

```python
from mujoco_golf_pendulum.models import ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
from mujoco_golf_pendulum.joint_analysis import analyze_constraint_forces_over_time

model = mj.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
data = mj.MjData(model)

# Simulate swing and record forces at all joints
joints_to_monitor = [
    "left_shoulder_flexion",
    "left_wrist_flexion",
    "left_wrist_deviation",
    "spine_rotation"
]

force_data = analyze_constraint_forces_over_time(
    model, data,
    joint_names=joints_to_monitor,
    duration=1.5  # Full swing duration
)
```

## Theory

### Universal Joint Mechanics

**Velocity Relationship:**

For a universal joint with input shaft rotating at constant velocity ω₁:

```
ω₂/ω₁ = cos(β) / (1 - sin²(β)sin²(θ))
```

**Torque Relationship:**

By conservation of energy (assuming no friction):
```
τ₂/τ₁ = 1 / (ω₂/ω₁)
```

**Maximum Wobble:**

Occurs at θ = 90°, 270°:
```
(ω₂/ω₁)_max = 1 / cos(β)
(ω₂/ω₁)_min = cos(β)
```

**Operating Limits:**

- Maximum recommended angle: β < 25° (sustained)
- Critical angle for vibration: β > 30°
- Gimbal lock equivalent: β → 90°

### Constraint Forces

MuJoCo computes constraint forces to satisfy joint constraints. Access via:

```python
constraint_force = data.qfrc_constraint[joint_qpos_address]
```

**Interpretation:**
- Hinge joint: scalar constraint torque
- Universal joint: 2D constraint torque vector
- Gimbal joint: 3D constraint torque vector

### Flexible Beam Model

The flexible shaft uses **serially-connected rigid segments** with torsion springs:

```
Segment 1 (stiff) ⇿ Segment 2 (medium) ⇿ Segment 3 (flexible) → Head
```

**Stiffness Distribution:**
- Upper shaft: Stiffer (180-240 N⋅m/rad)
- Middle shaft: Medium (150-220 N⋅m/rad)
- Lower shaft: More flexible (120-200 N⋅m/rad)

**Equation of Motion (per segment):**
```
I·α = τ_applied - k·θ - b·ω
```

where:
- I = moment of inertia
- k = torsional stiffness
- b = damping coefficient
- θ = angular displacement

## Tips and Best Practices

### Model Selection

- **Simple analysis:** Use `TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML`
- **Joint testing:** Use `GIMBAL_JOINT_DEMO_XML`
- **Full biomechanics:** Use `ADVANCED_BIOMECHANICAL_GOLF_SWING_XML`

### Performance

- Flexible shafts increase computation time
- Use 3 segments for good balance of accuracy/speed
- Use 5 segments only for detailed shaft dynamics studies

### Stability

- Universal joints can become unstable at extreme angles
- Monitor gimbal lock warnings
- Use damping to prevent oscillations in flexible shafts

### Analysis Workflow

1. **Run simulation** with desired control inputs
2. **Record constraint forces** using analysis tools
3. **Analyze torque wobble** for universal joints
4. **Check gimbal lock** for shoulder joints
5. **Visualize results** with plotting functions

## References

- Universal Joint Mechanics: Norton, "Design of Machinery", McGraw-Hill
- Gimbal Systems: Wertz, "Spacecraft Attitude Determination and Control"
- Golf Biomechanics: Nesbit & Serrano, "Work and Power Analysis of the Golf Swing"
- MuJoCo Documentation: https://mujoco.readthedocs.io/

## Further Reading

- `examples_joint_analysis.py` - Complete working examples
- `joint_analysis.py` - Analysis tool implementation
- `models.py` - Model definitions and club configurations
