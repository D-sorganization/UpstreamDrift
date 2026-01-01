# OpenSim Engine

## Overview
OpenSim is a freely available software package for musculoskeletal modeling and simulation. It enables users to build models of the musculoskeletal system and create dynamic simulations of movement.

## Key Features in Suite
- **Inverse Kinematics**: Computes generalized coordinates that best match experimental marker data.
- **Inverse Dynamics**: Determines the generalized forces (e.g., net moments) that generate a given motion.
- **Static Optimization**: Estimates muscle activations and forces.
- **Muscle-Actuated Simulation**: Forward dynamics simulations driven by muscle actuators.

## Integration
The OpenSim implementation adheres to the `PhysicsEngine` protocol, allowing interchangeable usage within the Golf Modeling Suite.

## Usage
Located in `engines/physics_engines/opensim/`.

### Python Access
```python
from engines.physics_engines.opensim.python.opensim_physics_engine import OpenSimPhysicsEngine

engine = OpenSimPhysicsEngine()
engine.load_from_path("path/to/model.osim")
engine.reset()
engine.step(0.01)
```

### GUI
Launch the OpenSim-specific diagnostic GUI:
```bash
python -m engines.physics_engines.opensim.python.opensim_gui
```
