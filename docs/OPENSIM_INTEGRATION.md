# OpenSim Integration Architecture

This document describes how OpenSim is integrated into the Golf Modeling Suite, its capabilities, and how to use it through the application.

---

## Overview

OpenSim is an open-source software platform for modeling, simulating, and analyzing the musculoskeletal system. The Golf Modeling Suite integrates OpenSim as a physics engine backend, enabling:

1. **Musculoskeletal simulation** with Hill-type muscle models
2. **Biomechanical analysis** of golf swing mechanics
3. **Muscle force estimation** from motion data
4. **Cross-validation** with other physics engines (MuJoCo, Drake, Pinocchio)

---

## Architecture

### API Integration Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                    Golf Modeling Suite                        │
├──────────────────────────────────────────────────────────────┤
│  GUI Layer (PyQt6)                                            │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │ opensim_gui.py  │  │ golf_launcher   │                    │
│  │ - Load Model    │  │ - Engine Select │                    │
│  │ - Run Simulation│  │ - Status Badge  │                    │
│  │ - View Results  │  │ - Help System   │                    │
│  └────────┬────────┘  └─────────────────┘                    │
├───────────┼──────────────────────────────────────────────────┤
│  Engine Layer                                                 │
│  ┌────────▼────────┐                                         │
│  │ PhysicsEngine   │  ◄─── Abstract Interface                │
│  │ (Protocol)      │                                         │
│  └────────┬────────┘                                         │
│           │                                                   │
│  ┌────────▼───────────────────────┐                          │
│  │ OpenSimPhysicsEngine           │  engines/.../python/     │
│  │ ├─ load_from_path(osim_file)   │  opensim_physics_engine.py│
│  │ ├─ step(dt)                    │                          │
│  │ ├─ get_state() -> (q, v)       │                          │
│  │ ├─ set_control(u)              │                          │
│  │ ├─ compute_mass_matrix()       │                          │
│  │ ├─ compute_inverse_dynamics()  │                          │
│  │ ├─ compute_drift_acceleration()│                          │
│  │ ├─ get_muscle_analyzer()       │                          │
│  │ └─ create_grip_model()         │                          │
│  └────────┬───────────────────────┘                          │
├───────────┼──────────────────────────────────────────────────┤
│  OpenSim Python Bindings (opensim package)                   │
│  ┌────────▼────────┐                                         │
│  │ opensim.Model   │  ◄─── Official OpenSim API              │
│  │ opensim.State   │                                         │
│  │ opensim.Manager │                                         │
│  │ opensim.InverseDynamicsSolver                             │
│  └─────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

### File Organization

```
engines/physics_engines/opensim/
├── __init__.py
└── python/
    ├── opensim_gui.py              # PyQt6 GUI for OpenSim
    ├── opensim_physics_engine.py   # PhysicsEngine implementation
    └── opensim_golf/
        ├── __init__.py
        ├── core.py                 # GolfSwingModel class
        └── muscle_analysis.py      # Muscle analysis tools

shared/models/opensim/
├── examples/                       # Custom examples and scripts
│   ├── README.md                   # Model documentation
│   ├── pendulum_1dof.osim          # Validation model
│   └── build_simple_arm_model.py   # Example script
│
└── opensim-models/                 # Official OpenSim models (submodule)
    ├── Models/                     # 18 model directories
    │   ├── Arm26/arm26.osim
    │   ├── Pendulum/double_pendulum.osim
    │   ├── Gait2392_Simbody/gait2392.osim
    │   ├── Rajagopal/Rajagopal2015.osim
    │   └── ... (14 more models)
    ├── Geometry/                   # VTP visualization files
    ├── Tutorials/                  # Official tutorials
    └── Pipelines/                  # IK/ID/CMC examples
```

**Note:** The `opensim-models/` directory is a Git submodule pointing to
`https://github.com/opensim-org/opensim-models`. Update with:

```bash
git submodule update --remote shared/models/opensim/opensim-models
```

---

## Key Classes

### 1. OpenSimPhysicsEngine

The main interface to OpenSim. Implements the `PhysicsEngine` protocol.

```python
from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine
)

# Create engine
engine = OpenSimPhysicsEngine()

# Load model
engine.load_from_path("path/to/model.osim")

# Get state (q = positions, v = velocities)
q, v = engine.get_state()

# Set controls (muscle activations or joint torques)
engine.set_control(np.array([0.5, 0.3, ...]))

# Step simulation
engine.step(dt=0.001)

# Get mass matrix
M = engine.compute_mass_matrix()

# Compute inverse dynamics
tau = engine.compute_inverse_dynamics(qacc)
```

### 2. GolfSwingModel (GUI Layer)

Higher-level wrapper for golf-specific simulations.

```python
from engines.physics_engines.opensim.python.opensim_golf.core import (
    GolfSwingModel
)

# Load model (requires valid .osim file)
model = GolfSwingModel("path/to/golf_model.osim")

# Run simulation (returns SimulationResult)
result = model.run_simulation()

# Access results
print(f"Time points: {len(result.time)}")
print(f"Muscle forces shape: {result.muscle_forces.shape}")
print(f"Joint torques shape: {result.joint_torques.shape}")
```

### 3. Muscle Analysis

OpenSim's key advantage: Hill-type muscle models.

```python
# Get muscle analyzer
analyzer = engine.get_muscle_analyzer()

if analyzer:
    # Analyze all muscles
    report = analyzer.analyze_all()

    print(f"Muscle forces: {report.muscle_forces}")
    print(f"Moment arms: {report.moment_arms}")
    print(f"Muscle powers: {report.muscle_powers}")

    # Induced acceleration analysis
    contributions = engine.compute_muscle_induced_accelerations()
    for muscle, accel in contributions.items():
        print(f"{muscle}: {accel} rad/s²")
```

---

## OpenSim API Concepts

### Model Structure

An OpenSim model (`.osim` file) contains:

| Component       | Description                                    |
| --------------- | ---------------------------------------------- |
| **Bodies**      | Rigid segments (bones, links)                  |
| **Joints**      | Connections between bodies (hinge, ball, etc.) |
| **Muscles**     | Hill-type actuators with F-L-V properties      |
| **Markers**     | Virtual points for motion capture              |
| **Constraints** | Kinematic relationships                        |
| **Controllers** | Feedback control systems                       |

### State Vector

OpenSim uses a split state representation:

```python
# Q: Generalized coordinates (positions)
#    - Joint angles (rad)
#    - Joint translations (m)
q = state.getQ()  # Vector of size nq

# U: Generalized speeds (velocities)
#    - Joint angular velocities (rad/s)
#    - Joint translational velocities (m/s)
u = state.getU()  # Vector of size nu

# Note: Usually nq == nu, but not always (e.g., quaternion joints)
```

### Realization Stages

OpenSim uses lazy evaluation with "realization stages":

```python
# Each stage computes progressively more quantities
model.realizeTime(state)      # Stage 0: Time only
model.realizePosition(state)  # Stage 1: Positions, lengths
model.realizeVelocity(state)  # Stage 2: Velocities
model.realizeDynamics(state)  # Stage 3: Forces, accelerations
```

### Muscle Model (Hill-Type)

OpenSim implements the Hill muscle model with:

1. **Activation dynamics**: Neural signal → muscle activation
2. **Force-length curve**: Optimal fiber length
3. **Force-velocity curve**: Concentric/eccentric differences
4. **Tendon compliance**: Series elastic element

```
F_muscle = a(t) × F_max × f_FL(l) × f_FV(v) + F_passive(l)

Where:
- a(t): Time-varying activation [0-1]
- F_max: Maximum isometric force
- f_FL: Force-length relationship
- f_FV: Force-velocity relationship
- F_passive: Passive fiber force
```

---

## How to Use Through the Application

### GUI Workflow

1. **Launch OpenSim GUI**
   - From main launcher, click "OpenSim Golf" tile
   - If no model loaded, shows "Getting Started" interface

2. **Load a Model**
   - Click "Load Model" button
   - Select a `.osim` file
   - Sample models in `shared/models/opensim/examples/`

3. **Run Simulation**
   - Click "Run Simulation"
   - View muscle forces, joint angles, trajectories
   - Export data for analysis

4. **Get Help**
   - Click "Help / Guide" for documentation
   - See `launchers/assets/opensim_getting_started.md`

### Programmatic Workflow

```python
from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine
)
import numpy as np

# 1. Setup
engine = OpenSimPhysicsEngine()
engine.load_from_path("shared/models/opensim/examples/arm26.osim")

# 2. Initialize
engine.reset()

# 3. Define muscle activations (26 muscles for arm26 model)
activations = np.zeros(26)
activations[0] = 0.5  # Activate first muscle at 50%

# 4. Simulate
dt = 0.001  # 1ms timestep
for i in range(1000):  # 1 second
    engine.set_control(activations)
    engine.step(dt)

    # Get current state
    q, v = engine.get_state()
    t = engine.get_time()

    if i % 100 == 0:
        print(f"t={t:.3f}s, elbow_angle={np.degrees(q[0]):.1f}°")

# 5. Analysis
M = engine.compute_mass_matrix()
print(f"Mass matrix shape: {M.shape}")

analyzer = engine.get_muscle_analyzer()
if analyzer:
    report = analyzer.analyze_all()
    print(f"Total muscle force: {np.sum(report.muscle_forces)} N")
```

---

## Available Features (Implementation Status)

| Feature              | Status      | Method                             |
| -------------------- | ----------- | ---------------------------------- |
| Load .osim model     | ✅ Complete | `load_from_path()`                 |
| Get/set state        | ✅ Complete | `get_state()`, `set_state()`       |
| Step simulation      | ✅ Complete | `step(dt)`                         |
| Mass matrix          | ✅ Complete | `compute_mass_matrix()`            |
| Bias forces (C+G)    | ✅ Complete | `compute_bias_forces()`            |
| Gravity forces       | ✅ Complete | `compute_gravity_forces()`         |
| Inverse dynamics     | ✅ Complete | `compute_inverse_dynamics()`       |
| Drift-control decomp | ✅ Complete | `compute_drift_acceleration()`     |
| Muscle analyzer      | ✅ Complete | `get_muscle_analyzer()`            |
| Grip modeling        | ✅ Complete | `create_grip_model()`              |
| Jacobians            | ⚠️ Partial  | `compute_jacobian()`               |
| ZTCF/ZVCF            | ❌ Pending  | `compute_ztcf()`, `compute_zvcf()` |
| Full simulation GUI  | ⚠️ Pending  | `run_simulation()`                 |

---

## Installation

### Option 1: Conda (Recommended)

```bash
conda install -c opensim-org opensim
```

### Option 2: pip (if available)

```bash
pip install opensim
```

### Option 3: From Source

1. Download OpenSim from https://opensim.stanford.edu/
2. Install following platform-specific instructions
3. Configure Python bindings

### Verify Installation

```python
import opensim
print(opensim.__version__)

# Load a test model
model = opensim.Model()
print(f"Default model: {model.getName()}")
```

---

## Sample Models

### Arm26 (Beginner)

Simple upper arm model with 26 muscles.

- **Muscles**: Biceps, Triceps, Deltoids, etc.
- **Joints**: Shoulder, elbow
- **Use case**: Learning OpenSim API

### Gait2392 (Intermediate)

Full lower body model for gait analysis.

- **Muscles**: 92 muscles
- **Joints**: Hip, knee, ankle
- **Use case**: Walking/running analysis

### Golf Swing Model (Domain-Specific)

Custom model for golf biomechanics.

- **Muscles**: Upper body + grip muscles
- **Joints**: Shoulder, elbow, wrist, trunk
- **Use case**: Golf swing optimization

---

## Cross-Validation with Other Engines

OpenSim results should be validated against MuJoCo/Drake/Pinocchio:

```python
from shared.python.cross_engine_validator import CrossEngineValidator

validator = CrossEngineValidator()

# Compare inverse dynamics torques
result = validator.compare_states(
    "OpenSim", opensim_tau,
    "MuJoCo", mujoco_tau,
    metric="torque"
)

if result.passed:
    print("✅ Engines agree")
else:
    print(f"❌ Deviation: {result.max_deviation:.2e}")
```

**Expected Agreement:**

- Torques: ±1e-3 N·m (or <10% RMS)
- Positions: ±1e-6 m (after short simulations)
- Note: Muscle forces are OpenSim-specific (no equivalent in rigid-body engines)

---

## References

- [OpenSim Official Documentation](https://opensim.stanford.edu/)
- [OpenSim API Guide](https://opensim-org.github.io/)
- [OpenSim Models Repository](https://github.com/opensim-org/opensim-models)
- Golf Modeling Suite: `docs/assessments/project_design_guidelines.qmd` Section J

---

_Last Updated: January 2026_
