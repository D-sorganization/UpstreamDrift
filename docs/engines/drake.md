# Drake Engine

rake is a C++ toolbox for model-based design and verification of robotics, developed by the Toyota Research Institute. It provides high-accuracy dynamics and advanced system analysis capabilities.

## ✨ Key Features
- **Rigid Body Dynamics**: Uses `MultibodyPlant` for continuous physics.
- **System Framework**: Diagram-based architecture for connecting controllers, plants, and visualizers.
- **MeshCat**: Web-based 3D visualization.
- **Trajectory Optimization**: Powerful solvers for motion planning.

## 📁 Directory Structure

```
engines/physics_engines/drake/
├── assets/          # SDF/URDF model files
├── python/          # Python wrappers
└── tests/           # Engine-specific tests
```

## Integration Details

The Drake engine is loaded via `pydrake`. The suite initializes a `DiagramBuilder` enclosing the physics plant and visualizer.

### Coordinate Systems
Drake uses specific conventions for frames:
- **World Frame**: Z-up.
- **Body Frames**: Defined in URDF/SDF.

### Energy Conservation
Drake's integrators (e.g., `SemiExplicitEuler`, `RungeKutta3`) are tested for energy conservation in the Validation Suite.
