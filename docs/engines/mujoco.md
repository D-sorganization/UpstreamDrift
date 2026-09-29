# MuJoCo Engine

## Overview

MuJoCo (Multi-Joint dynamics with Contact) is a physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, and other areas where fast and accurate simulation is needed.

## Key Features in Suite

- High-fidelity contact modeling.
- Support for complex humanoid golf models.
- Fast simulation speed suitable for optimization loops.

## Usage

Located in `src/engines/physics_engines/mujoco/`.

The engine adapter is `MuJoCoPhysicsEngine` in
`src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py`.
Launch the humanoid golf GUI from the repository root:

```bash
python -m src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf
```
