# MyoSim Engine

## Overview
MyoSim is a specialized solver for simulating muscle physiology at the half-sarcomere level. It models ca2+ activation, cross-bridge cycling, and force generation using system-of-ODEs approaches.

## Key Features in Suite
- **Half-Sarcomere Modeling**: Detailed kinetics of actin-myosin interactions.
- **Calcium Dynamics**: Models activation dependence on calcium concentration history.
- **Tension Generation**: Predicts force production under varying length and velocity conditions.

## Integration
The MyoSim implementation adheres to the `PhysicsEngine` protocol. It is particularly useful for detailed bio-fidelity in muscle force estimation, often used in conjunction with larger multibody models.

## Usage
Located in `engines/physics_engines/myosim/`.

### Python Access
```python
from engines.physics_engines.myosim.python.myosim_physics_engine import MyoSimPhysicsEngine

engine = MyoSimPhysicsEngine()
engine.load_from_string(config_json) # Loads muscle parameters
engine.reset()
engine.step(0.001)
```

### GUI
Launch the MyoSim-specific diagnostic GUI:
```bash
python -m engines.physics_engines.myosim.python.myosim_gui
```
