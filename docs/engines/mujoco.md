# MuJoCo Engine

The MuJoCo (Multi-Joint dynamics with Contact) engine is the primary physics backend for high-fidelity biomechanical simulation in this suite.

## ✨ Key Features
- **Musculoskeletal Modeling**: Supports a full-body model with 290 muscles.
- **Contact Dynamics**: Accurate club-ball impact simulation.
- **Inverse Dynamics**: Calculate muscle forces from motion capture data.
- **Interactive Visualization**: Real-time rendering with OpenGL.

## 📁 Directory Structure

```
engines/physics_engines/mujoco/
├── assets/          # XML model files and meshes
├── python/          # Python wrappers and logic
└── tests/           # Engine-specific tests
```

## 🔧 Configuration

MuJoCo models are defined in `.xml` (MJCF) format. The main entry point is `golf_swing.xml`.

parameters can be tuned in `shared/python/physics_parameters.py`.
