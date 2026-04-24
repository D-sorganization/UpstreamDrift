# Choose Your Engine: A User Guide

This guide helps you select the right physics engine for your UpstreamDrift simulation task.
For a quick hands-on demo, see [`examples/choose_engine_demo.py`](../../examples/choose_engine_demo.py).

---

## Quick Decision Flowchart

```
Do you need Hill-type muscle models?
  YES → MuJoCo + MyoSuite
  NO  ↓
Do you need trajectory optimisation (e.g. golf swing)?
  YES → Drake
  NO  ↓
Do you need fast kinematics / algorithmic differentiation?
  YES → Pinocchio
  NO  ↓
Do you need clinical biomechanics / OpenSim .osim models?
  YES → OpenSim
  NO  → MuJoCo  (safe general default)
```

---

## Engine Profiles

### MuJoCo

**Best for:** Contact-heavy simulations, muscle-driven models, general-purpose robotics.

| Property       | Value                    |
| -------------- | ------------------------ |
| Model format   | MJCF XML, URDF           |
| Muscle support | Via MyoSuite (Hill-type) |
| Contact model  | Soft contact (mature)    |
| Speed          | Fast                     |
| Installation   | `pip install mujoco`     |

```python
from src.shared.python.engine_manager import EngineManager
manager = EngineManager()
engine = manager.load_engine("mujoco")
```

### Pinocchio

**Best for:** Fast rigid-body kinematics, algorithmic differentiation, lightweight footprint.

| Property       | Value                                                   |
| -------------- | ------------------------------------------------------- |
| Model format   | URDF                                                    |
| Muscle support | None                                                    |
| Contact model  | Basic                                                   |
| Speed          | Very fast (~20 % faster than MuJoCo on pure kinematics) |
| Installation   | `conda install pinocchio -c conda-forge`                |

```python
engine = manager.load_engine("pinocchio")
```

### Drake

**Best for:** Trajectory optimisation, motion planning, contact planning with Trajopt.

| Property       | Value               |
| -------------- | ------------------- |
| Model format   | URDF, SDF           |
| Muscle support | Manual only         |
| Contact model  | Good                |
| Speed          | Medium              |
| Installation   | `pip install drake` |

```python
engine = manager.load_engine("drake")
```

### OpenSim

**Best for:** Clinical biomechanics, existing `.osim` model libraries, joint reaction analysis.

| Property       | Value                                                           |
| -------------- | --------------------------------------------------------------- |
| Model format   | .osim                                                           |
| Muscle support | Excellent (built-in)                                            |
| Contact model  | Basic                                                           |
| Speed          | Slow                                                            |
| Installation   | See [opensim-core](https://github.com/opensim-org/opensim-core) |

```python
engine = manager.load_engine("opensim")
```

---

## Feature Comparison Matrix

| Feature                     | MuJoCo       | Pinocchio | Drake     | OpenSim    |
| --------------------------- | ------------ | --------- | --------- | ---------- |
| Forward dynamics            | Yes          | Yes       | Yes       | Yes        |
| Inverse dynamics            | Yes          | Yes       | Yes       | Limited    |
| Hill-type muscles           | Via MyoSuite | No        | No        | Yes        |
| Trajectory optimisation     | Basic        | Crocoddyl | Excellent | No         |
| URDF loading                | Yes          | Yes       | Yes       | No (.osim) |
| Algorithmic differentiation | No           | Yes       | Yes       | No         |
| GPU support                 | Limited      | No        | No        | No         |

---

## Humanoid Simulation Scenarios

### Scenario 1: Walking / running gait

Recommended: **MuJoCo** or **Pinocchio**.
MuJoCo for contact fidelity; Pinocchio if you need fast Jacobian sweeps for an outer
optimisation loop.

### Scenario 2: Golf swing optimisation

Recommended: **Drake** (Trajopt) or **MuJoCo** (quick prototyping).

### Scenario 3: Injury-risk analysis

Recommended: **OpenSim** (joint reaction forces from .osim models) or
**MuJoCo + MyoSuite** (muscle-driven fatigue modelling).

### Scenario 4: RL policy training

Recommended: **MuJoCo** (fastest step time) or **MuJoCo + MyoSuite**
(physiologically realistic action space).

---

## Switching Engines at Runtime

The `EngineManager` provides a unified interface so you can swap engines without
changing your simulation code:

```python
from src.shared.python.engine_manager import EngineManager

manager = EngineManager()

# Load your preferred engine by name
for engine_name in ["mujoco", "pinocchio", "drake"]:
    try:
        engine = manager.load_engine(engine_name)
        engine.load_from_path("models/humanoid.urdf")
        engine.reset()
        for _ in range(100):
            engine.step()
        print(f"{engine_name}: OK")
    except ImportError:
        print(f"{engine_name}: not installed, skipping")
```

See `examples/choose_engine_demo.py` for a runnable version of this pattern.

---

## Further Reading

- [`docs/engines/engine_capabilities.md`](../engines/engine_capabilities.md) — detailed capability tables
- [`docs/engines/engine_selection_guide.md`](../engines/engine_selection_guide.md) — concise decision matrix
- [`docs/tutorials/content/03_engine_comparison.md`](content/03_engine_comparison.md) — cross-engine validation tutorial
- [`docs/adr/0002-physics-engine-plugin-architecture.md`](../adr/0002-physics-engine-plugin-architecture.md) — architecture decision record
