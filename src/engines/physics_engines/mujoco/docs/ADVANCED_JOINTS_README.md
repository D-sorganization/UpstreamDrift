# Advanced Joint Models - Quick Start Guide

This document provides a quick overview of the new universal joint, gimbal joint, and flexible club models added to the MuJoCo Golf Swing project.

## 🎯 What's New

### 1. **Universal Joint Models**
   - Two-link inclined plane swing model with universal joint at wrist
   - Realistic torque transmission and wobble effects
   - Constraint force visualization

### 2. **Gimbal Joint Models**
   - 3-DOF gimbal demonstration model
   - Gimbal lock detection
   - Full spherical range of motion

### 3. **Enhanced Full-Body Model**
   - Shoulders: Gimbal joints (3 DOF)
   - Scapula: Universal joints (2 DOF)
   - Elbows: Revolute joints (1 DOF)
   - Wrists: Universal joints (2 DOF)
   - Spine: Universal + Revolute (3 DOF total)

### 4. **Flexible Club Shafts**
   - Multi-segment beam model (1-5 segments)
   - Realistic stiffness profiles
   - Three club types: Driver, 7-Iron, Wedge

### 5. **Analysis Tools**
   - Constraint force recording and plotting
   - Torque wobble analysis
   - Gimbal lock detection
   - Joint coupling visualization

## 🚀 Quick Start

### Installation

No additional installation required if you already have the project set up. The new modules are:

```
python/mujoco_golf_pendulum/
├── models.py                      # ← Enhanced with new models
├── joint_analysis.py              # ← New analysis tools
└── examples_joint_analysis.py     # ← New examples
```

### Running Examples

```bash
# Run all joint analysis examples
cd python
python -m mujoco_golf_pendulum.examples_joint_analysis

# This will generate:
# - Universal joint torque wobble plots
# - Constraint force time series
# - Gimbal joint demonstrations
# - Club configuration examples
```

## 📊 Example Usage

### Analyze Universal Joint Wobble

```python
import mujoco as mj
from mujoco_golf_pendulum.models import TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML
from mujoco_golf_pendulum.joint_analysis import UniversalJointAnalyzer, plot_torque_wobble

# Load model
model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
data = mj.MjData(model)

# Analyze torque wobble
analyzer = UniversalJointAnalyzer(model, data)
results = analyzer.analyze_torque_transmission(
    "wrist_universal_1",
    "wrist_universal_2",
    num_cycles=2
)

# Plot results
plot_torque_wobble(results, save_path="wobble_analysis.png")

print(f"Wobble amplitude: {results['wobble_amplitude']:.4f}")
```

### Generate Flexible Club

```python
from mujoco_golf_pendulum.models import generate_flexible_club_xml, CLUB_CONFIGS

# Generate 3-segment flexible driver
driver_xml = generate_flexible_club_xml("driver", num_segments=3)

# Check parameters
driver_config = CLUB_CONFIGS["driver"]
print(f"Shaft stiffness: {driver_config['flex_stiffness']}")  # [180, 150, 120]
print(f"Head mass: {driver_config['head_mass']*1000:.1f}g")  # 198.0g
```

### Check Gimbal Lock

```python
from mujoco_golf_pendulum.models import GIMBAL_JOINT_DEMO_XML
from mujoco_golf_pendulum.joint_analysis import GimbalJointAnalyzer

model = mj.MjModel.from_xml_string(GIMBAL_JOINT_DEMO_XML)
data = mj.MjData(model)

analyzer = GimbalJointAnalyzer(model, data)
is_locked, distance = analyzer.check_gimbal_lock(
    "gimbal_x", "gimbal_y", "gimbal_z",
    threshold=0.087  # 5 degrees
)

if is_locked:
    print("⚠️  Gimbal lock detected!")
else:
    print(f"✓ Safe - {np.degrees(distance):.1f}° from gimbal lock")
```

## 📚 Available Models

| Model | XML Constant | Use Case |
|-------|-------------|----------|
| Two-Link Universal | `TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML` | Universal joint study |
| Gimbal Demo | `GIMBAL_JOINT_DEMO_XML` | Gimbal mechanics |
| Advanced Full-Body | `ADVANCED_BIOMECHANICAL_GOLF_SWING_XML` | Complete biomechanics |

## 🏌️ Golf Club Configurations

| Club Type | Length | Head Mass | Loft | Stiffness |
|-----------|--------|-----------|------|-----------|
| Driver | 116 cm | 198 g | 10° | Soft |
| 7-Iron | 95 cm | 253 g | 32° | Medium |
| Wedge | 90 cm | 288 g | 55° | Stiff |

All clubs include realistic:
- Grip dimensions and mass
- Shaft radius and mass distribution
- Head geometry and inertia
- Loft angles

## 🔧 Flexible Shaft Options

```python
# Rigid shaft (fastest simulation)
rigid_club = generate_rigid_club_xml("driver")

# Flexible shaft - 3 segments (recommended)
flex_club_3 = generate_flexible_club_xml("driver", num_segments=3)

# Flexible shaft - 5 segments (high fidelity)
flex_club_5 = generate_flexible_club_xml("driver", num_segments=5)
```

**Segment Distribution:**
- **Upper:** Stiffest, near hands
- **Middle:** Medium stiffness
- **Lower:** Most flexible, near head

## 📈 Analysis Functions

### Constraint Forces

```python
from mujoco_golf_pendulum.joint_analysis import (
    analyze_constraint_forces_over_time,
    plot_constraint_forces
)

force_data = analyze_constraint_forces_over_time(
    model, data,
    joint_names=["shoulder", "wrist_universal_1"],
    duration=2.0
)

plot_constraint_forces(force_data, ["shoulder"])
```

### Torque Transmission

```python
analyzer = UniversalJointAnalyzer(model, data)

# Analyze over 2 complete rotations
results = analyzer.analyze_torque_transmission(
    input_joint="wrist_universal_1",
    output_joint="wrist_universal_2",
    num_cycles=2
)

# Results include:
# - velocity_ratios: Output/input velocity over rotation
# - torque_ratios: Output/input torque
# - wobble_amplitude: Standard deviation of velocity ratio
```

## 🎓 Key Concepts

### Universal Joint

- **DOF:** 2 (two perpendicular hinges)
- **Wobble:** Velocity varies sinusoidally even with constant input
- **Max angle:** ~25° recommended for sustained operation
- **Golf application:** Wrist joints, scapula

**Torque Wobble Formula:**
```
ω_out/ω_in = cos(β) / (1 - sin²(β)sin²(θ))
```
where β = joint angle, θ = rotation angle

### Gimbal Joint

- **DOF:** 3 (three perpendicular hinges in nested rings)
- **No wobble:** Smooth torque transmission
- **Singularity:** Gimbal lock at ±90° middle ring
- **Golf application:** Shoulder joints

### Flexible Shaft

- **Model:** Serial rigid segments with torsion springs
- **Realism:** Based on actual club measurements
- **Computation:** Moderate increase in solve time
- **Use when:** Studying shaft lag, release dynamics

## 📖 Documentation

- **Detailed Guide:** `docs/JOINT_MODELS_GUIDE.md`
- **Examples:** `python/mujoco_golf_pendulum/examples_joint_analysis.py`
- **API Reference:** Docstrings in `joint_analysis.py`

## 🎯 Common Use Cases

### 1. Study Wrist Action in Golf Swing

Use the two-link universal joint model to isolate wrist mechanics:

```python
model = mj.MjModel.from_xml_string(TWO_LINK_INCLINED_PLANE_UNIVERSAL_XML)
# Apply swing torques and measure constraint forces
```

### 2. Analyze Shoulder ROM and Limits

Use gimbal joint model to test full range of motion:

```python
model = mj.MjModel.from_xml_string(GIMBAL_JOINT_DEMO_XML)
# Sweep through Euler angles and detect gimbal lock
```

### 3. Compare Shaft Flexibility Effects

Generate multiple models with different shaft configurations:

```python
models = {
    "rigid": generate_rigid_club_xml("driver"),
    "flex_3": generate_flexible_club_xml("driver", num_segments=3),
    "flex_5": generate_flexible_club_xml("driver", num_segments=5),
}
# Compare clubhead speed and lag angles
```

### 4. Measure Joint Loading

Record constraint forces during full swing:

```python
model = mj.MjModel.from_xml_string(ADVANCED_BIOMECHANICAL_GOLF_SWING_XML)
force_data = analyze_constraint_forces_over_time(
    model, data,
    joint_names=["left_wrist_flexion", "left_wrist_deviation"],
    duration=1.5
)
# Identify peak loads and injury risk
```

## ⚠️ Important Notes

### Stability

- Universal joints can oscillate at extreme angles (>45°)
- Add damping if instabilities occur
- Use smaller timesteps for flexible shafts

### Performance

- Flexible shafts: ~2-3x slower than rigid
- 3 segments: Good balance
- 5 segments: Use only when necessary

### Gimbal Lock

- Occurs at ±90° middle ring rotation
- Check with `GimbalJointAnalyzer.check_gimbal_lock()`
- Avoid prolonged operation near singularity

## 🔬 Research Applications

1. **Universal Joint Mechanics**
   - Torque wobble quantification
   - Constraint force analysis
   - Optimal joint angles

2. **Biomechanical Studies**
   - Joint loading during swing
   - Range of motion analysis
   - Injury prevention

3. **Equipment Design**
   - Shaft flexibility optimization
   - Club head speed vs. control
   - Grip pressure effects

4. **Motor Control**
   - Optimal control with joint constraints
   - Trajectory planning with singularities
   - Force transmission efficiency

## 📞 Support

For issues or questions:
- Check `docs/JOINT_MODELS_GUIDE.md` for detailed documentation
- Review `examples_joint_analysis.py` for working code
- Open an issue on GitHub

## 📄 License

Same license as main project (see LICENSE file)

---

**Happy Simulating! 🏌️⛳**
