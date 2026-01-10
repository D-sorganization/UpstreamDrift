# MyoSuite Integration Architecture

This document describes how MyoSuite is integrated into the Golf Modeling Suite, its capabilities, and how to use it through the application.

---

## Overview

MyoSuite is a physics-based environment for musculoskeletal control using reinforcement learning. It provides:

1. **Muscle-driven biomechanical models** simulated with MuJoCo
2. **OpenAI Gym-compatible environments** for training RL policies
3. **Pre-built tasks** for manipulation, locomotion, and sport motions
4. **Cross-validation** with other physics engines (MuJoCo, OpenSim)

---

## Architecture

### Integration Pattern

```
┌──────────────────────────────────────────────────────────────┐
│                    Golf Modeling Suite                        │
├──────────────────────────────────────────────────────────────┤
│  GUI Layer (PyQt6)                                            │
│  ┌─────────────────┐  ┌─────────────────┐                    │
│  │ myosuite_gui    │  │ golf_launcher   │                    │
│  │ - Env Select    │  │ - Engine Select │                    │
│  │ - RL Training   │  │ - Status Badge  │                    │
│  │ - Visualization │  │ - Help System   │                    │
│  └────────┬────────┘  └─────────────────┘                    │
├───────────┼──────────────────────────────────────────────────┤
│  Adapter Layer                                                │
│  ┌────────▼────────────────────────────┐                     │
│  │ MuscleDrivenEnv                      │ myosuite_adapter.py │
│  │ ├─ Gym-compatible interface          │                     │
│  │ ├─ Observation: q, v, activations    │                     │
│  │ ├─ Action: muscle excitations [0-1]  │                     │
│  │ └─ Reward: task-specific             │                     │
│  └────────┬────────────────────────────┘                     │
├───────────┼──────────────────────────────────────────────────┤
│  Engine Layer                                                 │
│  ┌────────▼───────────────────────┐                          │
│  │ MyoSuitePhysicsEngine          │  myosuite_physics_engine │
│  │ ├─ load_from_path(env_id)      │                          │
│  │ ├─ step(dt)                    │                          │
│  │ ├─ get_state() -> (q, v)       │                          │
│  │ ├─ set_muscle_activations()    │                          │
│  │ ├─ compute_drift_acceleration()│                          │
│  │ └─ get_muscle_analyzer()       │                          │
│  └────────┬───────────────────────┘                          │
├───────────┼──────────────────────────────────────────────────┤
│  MyoSuite Package (myosuite)                                 │
│  ┌────────▼────────┐                                         │
│  │ gym.make()      │  ◄─── OpenAI Gym API                    │
│  │ env.reset()     │                                         │
│  │ env.step()      │                                         │
│  │ MuJoCo backend  │  ◄─── Physics simulation                │
│  └─────────────────┘                                         │
└──────────────────────────────────────────────────────────────┘
```

### File Organization

```
engines/physics_engines/myosuite/
└── python/
    └── myosuite_physics_engine.py   # PhysicsEngine implementation

shared/python/
└── myosuite_adapter.py              # RL environment adapter

shared/models/myosuite/
├── examples/                        # Custom examples and scripts
│   ├── README.md                    # Documentation
│   └── train_elbow_policy.py        # Example training script
│
└── myo_sim/                         # Official MyoSim models (submodule)
    ├── arm/                         # Upper arm models
    ├── elbow/                       # Elbow joint models
    ├── hand/                        # Hand/finger models
    ├── leg/                         # Lower limb models
    ├── body/                        # Full body models
    └── meshes/                      # Visualization assets
```

---

## Key Classes

### 1. MyoSuitePhysicsEngine

The main interface to MyoSuite environments. Implements the `PhysicsEngine` protocol.

```python
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine
)

# Create engine
engine = MyoSuitePhysicsEngine()

# Load environment by ID (not file path!)
engine.load_from_path("myoElbowPose1D6MRandom-v0")

# Get state (q = positions, v = velocities)
q, v = engine.get_state()

# Set muscle activations (6 muscles for elbow)
activations = {"BIClong": 0.5, "TRIlong": 0.2}
engine.set_muscle_activations(activations)

# Step simulation
engine.step(dt=0.002)

# Get drift acceleration (passive dynamics)
drift = engine.compute_drift_acceleration()
```

### 2. MuscleDrivenEnv (RL Adapter)

Gym-compatible environment for training neural control policies.

```python
from shared.python.myosuite_adapter import MuscleDrivenEnv, train_muscle_policy

# Create environment with muscle system
env = MuscleDrivenEnv(muscle_system, task="tracking", dt=0.001)

# Reset environment
obs = env.reset()

# Run episode
for _ in range(1000):
    action = policy.predict(obs)  # Neural network action
    obs, reward, done, info = env.step(action)
    if done:
        break

# Train a policy using Stable-Baselines3
policy = train_muscle_policy(env, total_timesteps=100000)
```

### 3. Direct MyoSuite Usage

For advanced users who want direct access to MyoSuite environments:

```python
import myosuite
from myosuite.utils import gym

# Create environment
env = gym.make("myoElbowPose1D6MRandom-v0")
env.reset()

# Run simulation with random actions
for _ in range(1000):
    env.mj_render()  # Visualize
    action = env.action_space.sample()  # Random muscle activations
    obs, reward, done, info = env.step(action)

env.close()
```

---

## Available Environments

MyoSuite includes many pre-built environments. Key ones for golf:

### Elbow Environments

| Environment ID                 | Description        | Muscles |
| ------------------------------ | ------------------ | ------- |
| `myoElbowPose1D6MRandom-v0`    | Elbow pose control | 6       |
| `myoElbowPose1D6MExoRandom-v0` | With exoskeleton   | 6       |

### Hand/Finger Environments

| Environment ID            | Description       | Muscles |
| ------------------------- | ----------------- | ------- |
| `myoHandPose100Random-v0` | Hand pose control | 39      |
| `myoHandReach-v0`         | Reaching task     | 39      |
| `myoHandKey-v0`           | Key turning       | 39      |

### Full Arm Environments

| Environment ID   | Description      | Muscles  |
| ---------------- | ---------------- | -------- |
| `myoArmPose-v0`  | Arm pose control | Multiple |
| `myoArmReach-v0` | Reaching task    | Multiple |

### Locomotion

| Environment ID  | Description | Muscles    |
| --------------- | ----------- | ---------- |
| `myoLegWalk-v0` | Walking     | Lower body |
| `myoLegRun-v0`  | Running     | Lower body |

---

## MyoSim Models (Submodule)

The `myo_sim/` directory contains MuJoCo-based musculoskeletal models:

```
myo_sim/
├── arm/          # Shoulder + elbow (9 files)
│   ├── myoarm.xml
│   └── ...
├── elbow/        # Elbow joint (10 files)
│   ├── myoelbow.xml
│   └── ...
├── hand/         # Full hand (5 files)
│   └── myohand.xml
├── finger/       # Individual fingers (4 files)
├── leg/          # Hip + knee + ankle (7 files)
├── body/         # Full body (5 files)
├── head/         # Head/neck (4 files)
└── meshes/       # 215 visualization files
```

### Loading MyoSim Models Directly

```python
import mujoco

# Load arm model
model = mujoco.MjModel.from_xml_path(
    "shared/models/myosuite/myo_sim/arm/myoarm.xml"
)
data = mujoco.MjData(model)

# Run simulation
mujoco.mj_step(model, data)
```

---

## How to Use Through the Application

### GUI Workflow

1. **Launch MyoSuite GUI**
   - From main launcher, click "MyoSuite" tile
   - If not installed, shows setup instructions

2. **Select Environment**
   - Choose from dropdown (e.g., "myoElbowPose1D6MRandom-v0")
   - Click "Load Environment"

3. **Run Simulation or Training**
   - Click "Visualize" for demo with random actions
   - Click "Train Policy" to start RL training
   - Monitor training progress in real-time

4. **Get Help**
   - Click "Help / Guide" for documentation
   - See `launchers/assets/myosuite_getting_started.md`

### Programmatic Workflow

```python
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine
)
import numpy as np

# 1. Setup
engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")

# 2. Initialize
engine.reset()

# 3. Get muscle names
muscle_names = engine.get_muscle_names()
print(f"Muscles: {muscle_names}")  # ['BIClong', 'BICshort', 'TRIlong', ...]

# 4. Simulate with muscle control
for i in range(1000):
    # Set muscle activations (simulating biceps contraction)
    activations = {
        "BIClong": 0.5 * (1 + np.sin(i * 0.01)),
        "TRIlong": 0.2,
    }
    engine.set_muscle_activations(activations)
    engine.step(dt=0.002)

    # Get current state
    q, v = engine.get_state()
    t = engine.get_time()

    if i % 100 == 0:
        print(f"t={t:.3f}s, elbow_angle={np.degrees(q[0]):.1f}°")

# 5. Analysis
analyzer = engine.get_muscle_analyzer()
if analyzer:
    forces = analyzer.get_muscle_forces()
    print(f"Muscle forces: {forces}")
```

---

## Available Features (Implementation Status)

| Feature                | Status      | Method                         |
| ---------------------- | ----------- | ------------------------------ |
| Load environment       | ✅ Complete | `load_from_path(env_id)`       |
| Get/set state          | ✅ Complete | `get_state()`, `set_state()`   |
| Step simulation        | ✅ Complete | `step(dt)`                     |
| Mass matrix            | ✅ Complete | `compute_mass_matrix()`        |
| Inverse dynamics       | ✅ Complete | `compute_inverse_dynamics()`   |
| Drift-control decomp   | ✅ Complete | `compute_drift_acceleration()` |
| Set muscle activations | ✅ Complete | `set_muscle_activations()`     |
| Muscle analyzer        | ✅ Complete | `get_muscle_analyzer()`        |
| Grip modeling          | ✅ Complete | `create_grip_model()`          |
| RL training            | ✅ Complete | `train_muscle_policy()`        |
| Direct MuJoCo access   | ✅ Complete | `engine.model`                 |

---

## Installation

### Option 1: pip (Recommended)

```bash
pip install -U myosuite
```

### Option 2: conda + pip

```bash
conda create --name myosuite python=3.10
conda activate myosuite
pip install myosuite
```

### Verify Installation

```bash
# Test installation
python -m myosuite.tests.test_myo

# List all environments
python -c "from myosuite.utils import gym; [print(e) for e in gym.envs.registry.keys() if 'myo' in e]"

# Visualize an environment
python -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0
```

---

## RL Training Example

For golf-related training (e.g., grip optimization):

```python
from stable_baselines3 import SAC
from myosuite.utils import gym

# Create environment
env = gym.make("myoHandPose100Random-v0")

# Create RL agent
model = SAC("MlpPolicy", env, verbose=1)

# Train
model.learn(total_timesteps=500000)

# Save policy
model.save("golf_grip_policy")

# Load and test
model = SAC.load("golf_grip_policy")
obs = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.mj_render()
    if done:
        obs = env.reset()
```

---

## Cross-Validation with OpenSim

For studies requiring both MyoSuite (RL training) and OpenSim (muscle analysis):

```python
from shared.python.cross_engine_validator import CrossEngineValidator
from engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine
)
from engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine
)

# Load comparable models
myosuite_engine = MyoSuitePhysicsEngine()
myosuite_engine.load_from_path("myoElbowPose1D6MRandom-v0")

opensim_engine = OpenSimPhysicsEngine()
opensim_engine.load_from_path("opensim-models/Models/Arm26/arm26.osim")

# Compare inverse dynamics
validator = CrossEngineValidator()
result = validator.compare_states(
    "MyoSuite", myosuite_tau,
    "OpenSim", opensim_tau,
    metric="torque"
)

print(f"Passed: {result.passed}, Severity: {result.severity}")
```

---

## References

- [MyoSuite Documentation](https://myosuite.readthedocs.io/)
- [MyoSuite GitHub](https://github.com/MyoHub/myosuite)
- [MyoSim Models](https://github.com/MyoHub/myo_sim)
- [Paper: MyoSuite (arXiv:2205.13600)](https://arxiv.org/abs/2205.13600)
- Golf Modeling Suite: `docs/assessments/project_design_guidelines.qmd` Section K

---

_Last Updated: January 2026_
