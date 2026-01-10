# MyoSuite Golf - Getting Started Guide

Welcome to MyoSuite Golf Simulation! This guide will help you get started with the MyoSuite physics engine for muscle-driven golf swing biomechanics.

## Prerequisites

### MyoSuite Installation

MyoSuite provides muscle-actuated models built on MuJoCo. To install:

```bash
pip install myosuite
```

**Note:** MyoSuite requires MuJoCo. Install both:

```bash
pip install mujoco myosuite
```

### Alternative Engines

Don't have MyoSuite? The Golf Modeling Suite supports multiple engines:

- **MuJoCo**: Core physics engine (`pip install mujoco`)
- **OpenSim**: Detailed musculoskeletal models
- **Pinocchio**: Analytical kinematics

---

## Quick Start

### Step 1: Launch MyoSuite GUI

From the main Golf Suite Launcher:

1. Click on the "MyoSim Suite" tile
2. The GUI will launch with available MyoSuite environments

### Step 2: Select an Environment

MyoSuite provides pre-built environments:

- `myoElbowPose1D6MRandom-v0` - Elbow model with 6 muscles
- `myoHandReachFixed-v0` - Hand reaching task
- Custom golf-specific environments

### Step 3: Run Simulation

1. Select your environment
2. Set muscle activations or let the model run
3. View real-time muscle forces and joint trajectories

---

## MyoSuite vs OpenSim Comparison

| Feature                 | MyoSuite         | OpenSim              |
| ----------------------- | ---------------- | -------------------- |
| **Physics Engine**      | MuJoCo           | SimTK                |
| **Muscle Model**        | MuJoCo actuators | Hill-type analytical |
| **Speed**               | Very fast (GPU)  | Moderate             |
| **Activation Dynamics** | Numerical        | Analytical ODE       |
| **RL Training**         | Native support   | Requires wrapper     |

Both engines are cross-validated for consistent results (±20% muscle forces).

---

## Muscle Analysis Features

### Available Analysis

- **Muscle Activations**: 0-1 normalized activation levels
- **Muscle Forces**: Computed from MuJoCo actuator_force
- **Moment Arms**: Finite difference (dL/dq method)
- **Induced Acceleration**: Per-muscle contribution via M⁻¹τ
- **Grip Modeling**: Hand muscle forces for club interface

### Example Code

```python
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine
)
from engines.physics_engines.myosuite.python.muscle_analysis import (
    MyoSuiteMuscleAnalyzer
)

# Create environment
engine = MyoSuitePhysicsEngine("myoElbowPose1D6MRandom-v0")

# Analyze muscles
analyzer = engine.get_muscle_analyzer()
report = analyzer.analyze_all()

print(f"Muscle names: {report.muscle_names}")
print(f"Total grip force: {report.total_grip_force_N} N")
```

---

## Key Concepts

### Muscle Actuators in MuJoCo

MyoSuite uses MuJoCo's muscle actuator type (`dyntype == mjDYN_MUSCLE`):

```xml
<actuator>
  <muscle name="biceps" joint="elbow"
          scale="500"
          lengthrange="0.1 0.3"/>
</actuator>
```

### Activation-Force-Torque Pipeline

1. **Activation** (0-1): Neural signal to muscle
2. **Force** (N): Computed from F-L-V properties
3. **Moment Arm** (m): Computed via finite differences
4. **Torque** (N·m): Force × Moment Arm
5. **Acceleration** (rad/s²): M⁻¹ × Torque

---

## Drift-Control Decomposition

MyoSuite implements Section F requirements:

```python
# Compute passive dynamics (zero muscle activation)
a_drift = engine.compute_drift_acceleration()

# Compute muscle-driven acceleration
a_control = engine.compute_control_acceleration(muscle_activations)

# Verify superposition
assert np.allclose(a_drift + a_control, a_total)
```

---

## Grip Force Analysis

MyoSuite identifies hand muscles automatically:

```python
grip_model = MyoSuiteGripModel(sim)
grip_analysis = grip_model.analyze_grip()

print(f"Total grip force: {grip_analysis['total_grip_force_N']} N")
print(f"Within MVC range: {grip_analysis['within_mvc_range']}")  # 200-800 N
```

---

## Troubleshooting

### "MyoSuite Not Installed"

```bash
pip install myosuite
```

### "MuJoCo DLL Error"

1. Ensure MuJoCo 3.x is installed
2. Check environment variables
3. Restart Python interpreter

### "No Muscle Actuators Found"

Your model may not define muscle actuators. Check the MJCF file for:

```xml
<actuator>
  <muscle ... />
</actuator>
```

---

## Cross-Validation with OpenSim

For research-grade analysis, validate results across engines:

```python
# Compare muscle forces between engines
opensim_forces = opensim_analyzer.get_muscle_forces()
myosuite_forces = myosuite_analyzer.get_muscle_forces()

# Should agree within ±20% (different muscle models)
np.allclose(opensim_forces, myosuite_forces, rtol=0.2)
```

---

## References

- [MyoSuite Documentation](https://myosuite.readthedocs.io/)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [Golf Modeling Suite Wiki](../../docs/README.md)
- [Section K: Muscle Requirements](../../docs/assessments/project_design_guidelines.qmd)

---

## Getting Help

For assistance:

1. Check the `docs/` folder for detailed documentation
2. Review `MYOSUITE_COMPLETE_SUMMARY.md` for implementation details
3. Open an issue on the repository

---

_Last Updated: January 2026_
