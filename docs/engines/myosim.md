# MyoSim Engine (MyoSuite)

## Overview

`MyoSim` is this suite's engine id for the **MyoSuite** integration —
`EngineType.MYOSIM` (`"myosim"`) resolves to
`src/engines/physics_engines/myosuite/`. MyoSuite provides musculoskeletal
environments built on MuJoCo, with Hill-type muscle actuators and Gym-compatible
reinforcement-learning interfaces.

There is no separate half-sarcomere MyoSim solver in this repository; the engine
id and the MyoSuite implementation are the same thing.

## Key Features in Suite

- **Hill-type muscle actuation**: muscle-driven forward dynamics via MuJoCo.
- **Muscle analysis**: activation, fibre length, and force introspection
  (`muscle_analysis.py`, `_muscle_interface.py`).
- **Drift-control decomposition**: `_drift_control.py` implements
  `compute_drift_acceleration` for this engine.
- **RL environments**: Gym-compatible environments for muscle-activation
  optimisation.

## Integration

The implementation is `MyoSuitePhysicsEngine` in
`src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py`, which
derives from `BasePhysicsEngine` and therefore satisfies the `PhysicsEngine`
protocol. It is loaded by `load_myosim_engine()` in `src/engines/loaders.py`.

Requires the optional extra: `pip install "upstream-drift[biomechanics]"`
(or `pip install myosuite>=2.0.0`).

## Usage

Located in `src/engines/physics_engines/myosuite/`.

### Python Access

```python
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)

engine = MyoSuitePhysicsEngine()
engine.reset()
engine.step(0.001)
```

### GUI

The launcher prefers the embedded path (`default_launch: tab` for the
`myosim_suite` entry in `src/config/models.yaml`). To drive the dashboard
outside the launcher, run from the repository root:

```bash
python -m src.engines.physics_engines.myosuite.python
```
