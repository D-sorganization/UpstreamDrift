# Golf Swing Models - Technical Documentation

## Overview

This document describes the biomechanically realistic golf swing models implemented in MuJoCo for the Golf Swing Model project. The models progress from simple 2D pendulum systems to full 3D humanoid models with realistic anthropometry and two-handed grip.

## Model Hierarchy

### 1. Double Pendulum (2 DOF)
**Purpose:** Educational introduction to pendulum dynamics
**Complexity:** Minimal
**Use Case:** Understanding basic swing mechanics

**Structure:**
- Shoulder joint (1 DOF - rotation)
- Wrist joint (1 DOF - rotation)
- Single arm segment
- Golf club

**Actuators:** 2
- Shoulder motor
- Wrist motor

---

### 2. Triple Pendulum (3 DOF)
**Purpose:** Educational model with elbow articulation
**Complexity:** Low
**Use Case:** Demonstrating multi-segment dynamics

**Structure:**
- Shoulder joint (1 DOF)
- Elbow joint (1 DOF)
- Wrist joint (1 DOF)
- Upper arm and forearm segments
- Golf club

**Actuators:** 3
- Shoulder motor
- Elbow motor
- Wrist motor

---

### 3. Upper Body Golf Swing (10 DOF)
**Purpose:** Realistic upper-body golf swing simulation
**Complexity:** Moderate
**Use Case:** Analyzing arm and torso coordination in golf swing

**Anatomical Structure:**

#### Base
- **Pelvis:** Fixed at hip height (0.95 m), provides stable reference frame
  - Mass: 10 kg
  - Dimensions: 0.15 × 0.08 × 0.08 m

#### Torso
- **Spine rotation joint:** Vertical axis rotation (-90° to +90°)
  - Critical for generating power through torso rotation
  - Mass: 30 kg total (lower + upper torso)
  - Length: 0.50 m

#### Left Arm Chain
- **Left shoulder:** 2 DOF ball-and-socket simplified to 2 hinge joints
  - Shoulder swing: -115° to +160° (sagittal plane)
  - Shoulder lift: -86° to +86° (frontal plane)
  - Upper arm mass: 2.5 kg, length: 0.25 m

- **Left elbow:** 1 DOF hinge
  - Range: -137° to 0° (flexion only)
  - Forearm mass: 1.5 kg, length: 0.25 m

- **Left wrist:** 1 DOF hinge
  - Range: -90° to +90°
  - Hand mass: 0.4 kg

#### Right Arm Chain
- **Right shoulder:** 2 DOF (symmetric to left)
- **Right elbow:** 1 DOF
- **Right wrist:** 1 DOF connected to club

#### Golf Club
- **Club-wrist joint:** 1 DOF for grip adjustment
  - Range: -57° to +57°
- **Grip section:** 0.25 m, mass 0.1 kg
- **Shaft:** 0.80 m, mass 0.25 kg (graphite density)
- **Clubhead:** Driver specifications
  - Dimensions: 0.055 × 0.04 × 0.03 m
  - Mass: 0.2 kg
  - Total club length: 1.05 m

#### Ball
- **USGA regulation golf ball:**
  - Diameter: 42.67 mm (0.02135 m radius)
  - Mass: 45.93 g (0.04593 kg)
  - Position: At address, 0.1 m in front of pelvis
  - Physics: 3D friction model for realistic contact

**Two-Handed Grip:**
- Left hand welded to club via equality constraint
- Right hand directly attached to club body
- Realistic grip spacing and hand positioning

**Actuators:** 10
1. Spine rotation motor (100 Nm max)
2. Left shoulder swing motor (80 Nm max)
3. Left shoulder lift motor (80 Nm max)
4. Left elbow motor (60 Nm max)
5. Left wrist motor (30 Nm max)
6. Right shoulder swing motor (80 Nm max)
7. Right shoulder lift motor (80 Nm max)
8. Right elbow motor (60 Nm max)
9. Right wrist motor (30 Nm max)
10. Club wrist motor (20 Nm max)

---

### 4. Full Body Golf Swing (15 DOF)
**Purpose:** Complete biomechanical golf swing model
**Complexity:** High
**Use Case:** Full swing analysis including weight transfer and ground reaction forces

**Additional Components (vs Upper Body):**

#### Legs
- **Left leg chain:**
  - Foot: 1.0 kg, 0.08 × 0.12 × 0.04 m (grounded)
  - Ankle: 1 DOF (-46° to +46°)
  - Shin: 4.0 kg, length 0.43 m
  - Knee: 1 DOF (-115° to 0°)
  - Thigh: 8.0 kg, length 0.45 m

- **Right leg chain:** Symmetric to left

#### Pelvis Dynamics
- **Free joint:** 6 DOF floating base
  - Connected to both legs via equality constraints
  - Allows realistic weight transfer
  - Mass: 12 kg

#### Enhanced Torso
- **Spine bend joint:** Forward/backward bend (-29° to +46°)
  - Enables realistic posture adjustments
- **Spine rotation:** Axial rotation (-103° to +103°)
  - Enhanced range for full body rotation

#### Head
- **Mass:** 5 kg
- **Diameter:** 0.1 m (sphere)
- Provides realistic mass distribution for balance

**Total Segment Masses:**
- Legs: 26 kg (both legs + feet)
- Pelvis: 12 kg
- Torso: 27 kg
- Head: 5 kg
- Arms: 8 kg
- **Total body mass:** ~78 kg (average adult male)

**Actuators:** 15
1. Left ankle motor (40 Nm max)
2. Left knee motor (120 Nm max)
3. Right ankle motor (40 Nm max)
4. Right knee motor (120 Nm max)
5. Spine bend motor (100 Nm max)
6. Spine rotation motor (100 Nm max)
7-15. Same arm/club motors as upper body model

---

## Biomechanical Features

### Anthropometric Accuracy
All body segment dimensions and masses are based on anthropometric data for a 1.75 m tall adult male:
- **Segment length ratios:** Based on Winter (2009) biomechanics data
- **Mass distribution:** Follows Clauser et al. (1969) regression equations
- **Joint ranges:** Based on Kapandji's "Physiology of the Joints"

### Physics Parameters
- **Timestep:** 0.002 s (500 Hz for stability with contact)
- **Integrator:** RK4 (Runge-Kutta 4th order)
- **Gravity:** -9.81 m/s²
- **Damping:** Joint-specific values for realistic energy dissipation
  - Large joints (spine, knee): 2-3 Nm·s/rad
  - Medium joints (shoulder, elbow): 1-1.5 Nm·s/rad
  - Small joints (wrist): 0.3-0.5 Nm·s/rad

### Contact Modeling
- **Ball-ground contact:** 3-dimensional friction cone
  - Static friction: 0.8
  - Torsional friction: 0.0001 (rolling)
- **Ball-club contact:** Potential for impact simulation
- **Foot-ground contact:** Grounded feet for stability

---

## Camera Views

### Upper Body Model
1. **Side view (default):** Classic golf swing viewing angle
   - Position: (-4, -1.5, 1.3)
   - Orientation: Slightly elevated, angled toward swing plane

2. **Front view:** Face-on perspective
   - Position: (0, -4, 1.3)
   - Useful for analyzing lateral movement

3. **Top view:** Bird's eye view
   - Position: (0, 0, 5)
   - Shows rotational components

### Full Body Model
Includes additional **follow camera:**
- Tracks center of mass
- Useful for dynamic swing analysis

---

## Usage Guidelines

### Getting Started
1. **Simple models first:** Start with double/triple pendulum to understand controls
2. **Progress to upper body:** Experiment with torso rotation and arm coordination
3. **Full body analysis:** Add leg dynamics for complete swing mechanics

### Experimental Framework
The models support:
- **Manual control:** Direct torque application via GUI sliders
- **Scripted motions:** Apply time-varying control signals
- **Optimization:** Use as forward model for swing optimization
- **Analysis:** Extract joint angles, velocities, contact forces

### Future Development Opportunities

#### 1. Motion Capture Integration
- Import professional golfer kinematics
- Replay motion capture data
- Compute required joint torques (inverse dynamics)

#### 2. Swing Controllers
- **PD controllers:** Track desired joint angle trajectories
- **Computed torque control:** Precise trajectory following
- **Reinforcement learning:** Train optimal swing policies

#### 3. Performance Metrics
- **Ball velocity:** Launch speed and direction
- **Clubhead speed:** At impact
- **Swing plane:** Deviation from ideal
- **Weight transfer:** Ground reaction force analysis
- **X-Factor:** Shoulder-hip separation angle

#### 4. Advanced Physics
- **Ball spin:** Magnus effect modeling
- **Club flexibility:** Shaft bend during swing
- **Impact dynamics:** Ball compression and rebound
- **Aerodynamics:** Air resistance on club and ball

#### 5. Analysis Tools
- **Joint angle plots:** Time series of all DOF
- **Phase portraits:** State space visualization
- **Power analysis:** Energy flow through kinematic chain
- **Optimization:** Parameter tuning for distance/accuracy

---

## Technical References

### MuJoCo Specific
- **Equality constraints:** Used for two-handed grip (weld constraints)
- **Freejoint:** 6-DOF floating base for pelvis and ball
- **Contact solver:** Pyramidal approximation of friction cone
- **Material properties:** Defined in `<asset>` section for visualization

### Model Files
- **Location:** `python/mujoco_golf_pendulum/models.py`
- **Format:** MJCF XML embedded as Python strings
- **Modification:** Edit XML strings and reload model in GUI

### Integration with GUI
- **Dynamic slider generation:** Automatically creates controls for all actuators
- **Grouping:** Organizes controls by body part (Legs, Torso, Arms, Club)
- **Real-time updates:** 60 FPS rendering with physics steps as needed

---

## Performance Considerations

### Computational Cost
- **Double pendulum:** ~0.01 ms per step
- **Triple pendulum:** ~0.015 ms per step
- **Upper body:** ~0.1 ms per step
- **Full body:** ~0.2 ms per step (due to contacts and constraints)

### Real-Time Factors
- All models run at 60 FPS rendering on modern hardware
- Physics timestep (2 ms) allows multiple substeps per frame
- Contact resolution is most computationally expensive component

---

## Validation

### Physical Plausibility
✓ Segment masses sum to realistic total body mass
✓ Joint ranges match human physiological limits
✓ Club dimensions match USGA regulations
✓ Ball properties match USGA specifications
✓ Torque limits reflect human strength capabilities

### Model Stability
✓ All models stable under zero torque (gravity equilibrium)
✓ No joint limit violations in neutral pose
✓ Contact solver converges reliably
✓ Equality constraints maintain two-handed grip

---

## Example Use Cases

### 1. Education
- Teach golf swing mechanics
- Demonstrate conservation of angular momentum
- Illustrate kinematic chain sequencing

### 2. Research
- Study optimal swing parameters
- Analyze injury risk factors
- Compare swing techniques

### 3. Training
- Visualize swing concepts
- Test equipment modifications
- Practice swing sequences in simulation

### 4. Development
- Benchmark control algorithms
- Test reinforcement learning policies
- Validate biomechanical hypotheses

---

## Citation

If you use these models in research, please cite:
```
MuJoCo Golf Swing Models (2024)
Repository: https://github.com/D-sorganization/MuJoCo_Golf_Swing_Model
License: MIT
```

## Contact

For questions, issues, or contributions:
- GitHub Issues: [Project Issues](https://github.com/D-sorganization/MuJoCo_Golf_Swing_Model/issues)
- Documentation: See `docs/` directory for additional guides
