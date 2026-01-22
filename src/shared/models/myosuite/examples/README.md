# MyoSuite Sample Models and Environments

This directory contains MyoSuite environments and MyoSim models for muscle-driven simulation.

---

## Quick Start

1. Install MyoSuite: `pip install myosuite`
2. Test installation: `python -m myosuite.tests.test_myo`
3. Launch Golf Suite → Select "MyoSuite"

---

## Official MyoSim Models (Submodule)

The `myo_sim/` directory contains the **official MyoSim musculoskeletal models** from MyoHub.

### Available Body Regions

| Directory   | Description             | Models        |
| ----------- | ----------------------- | ------------- |
| **arm/**    | Shoulder + Elbow        | myoarm.xml    |
| **elbow/**  | Elbow joint only        | myoelbow.xml  |
| **hand/**   | Full hand               | myohand.xml   |
| **finger/** | Individual fingers      | myofinger.xml |
| **leg/**    | Lower limb              | myoleg.xml    |
| **body/**   | Full body               | mybody.xml    |
| **head/**   | Head/neck               | myohead.xml   |
| **torso/**  | Trunk                   | myotorso.xml  |
| **meshes/** | 215 visualization files | .stl, .obj    |

### Model Example

```python
import mujoco

# Load elbow model directly
model = mujoco.MjModel.from_xml_path(
    "shared/models/myosuite/myo_sim/elbow/myoelbow.xml"
)
data = mujoco.MjData(model)

# Simulate
for _ in range(1000):
    mujoco.mj_step(model, data)
    print(f"Joint angle: {data.qpos[0]:.3f}")
```

---

## Pre-Built Gym Environments

MyoSuite includes many ready-to-use environments:

### Recommended for Golf

| Environment ID              | Use Case          | Muscles  |
| --------------------------- | ----------------- | -------- |
| `myoElbowPose1D6MRandom-v0` | Elbow control     | 6        |
| `myoHandPose100Random-v0`   | Grip control      | 39       |
| `myoArmReach-v0`            | Reaching motion   | Multiple |
| `myoHandKey-v0`             | Fine manipulation | 39       |

### Usage Example

```python
from myosuite.utils import gym

# Create environment
env = gym.make("myoElbowPose1D6MRandom-v0")
obs = env.reset()

# Run with random actions
for _ in range(1000):
    env.mj_render()  # Visualize
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()

env.close()
```

---

## RL Training Example

Train a neural network policy to control muscles:

```python
from stable_baselines3 import SAC
from myosuite.utils import gym

# Create environment
env = gym.make("myoElbowPose1D6MRandom-v0")

# Create and train agent
model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Save trained policy
model.save("elbow_control_policy")
```

---

## Custom Examples

The `examples/` directory contains Golf Modeling Suite-specific scripts:

| File                    | Description                      |
| ----------------------- | -------------------------------- |
| `train_elbow_policy.py` | Train elbow controller for swing |
| `train_grip_policy.py`  | Train grip controller            |
| `visualize_muscles.py`  | Muscle visualization demo        |

---

## Cross-Validation

MyoSuite models can be validated against OpenSim:

```python
from shared.python.cross_engine_validator import CrossEngineValidator

validator = CrossEngineValidator()
result = validator.compare_states(
    "MyoSuite", myosuite_torques,
    "OpenSim", opensim_torques,
    metric="torque"
)

if result.passed:
    print("✅ Engines agree within tolerance")
else:
    print(f"⚠️ Deviation: {result.severity}")
```

---

## Installation

### MyoSuite Package

```bash
pip install -U myosuite

# Verify
python -m myosuite.tests.test_myo
```

### RL Dependencies (for training)

```bash
pip install stable-baselines3
```

---

## Troubleshooting

### "gym not found"

```bash
pip install gymnasium  # or pip install gym==0.26
```

### Visualization issues on macOS

```bash
mjpython -m myosuite.utils.examine_env --env_name myoElbowPose1D6MRandom-v0
```

### Missing MyoSkeleton

```bash
python -m myosuite_init
```

---

## Resources

- [MyoSuite Documentation](https://myosuite.readthedocs.io/)
- [MyoSuite GitHub](https://github.com/MyoHub/myosuite)
- [MyoSim Models](https://github.com/MyoHub/myo_sim)
- [ICRA2024 Tutorial](https://colab.research.google.com/drive/1KGqZgSYgKXF-vaYC33GR9llDsIW9Rp-qY)

---

_For more details, see `/docs/MYOSUITE_INTEGRATION.md`_
