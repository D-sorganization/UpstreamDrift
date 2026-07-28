# Pinocchio Engine

## Overview

Pinocchio is a library for fast rigid body dynamics algorithms based on the Featherstone arithmetic. It is particularly efficient for computing kinematics and dynamics derivatives.

## Key Features in Suite

- Extremely fast recursive algorithms (RNEA, ABA, CRBA).
- Python bindings for rapid prototyping.
- Integration with other robotics tools.

## Usage

Located in `src/engines/physics_engines/pinocchio/`.

The engine adapter is `PinocchioPhysicsEngine` in
`src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`.
Launch the Pinocchio dashboard from the repository root:

```bash
python -m src.engines.physics_engines.pinocchio.python
```
