# Pinocchio Engine

Pinocchio is a fast and flexible library for rigid multi-body dynamics algorithms. It is particularly optimized for robotics control and trajectory optimization.

## ✨ Key Features
- **High Performance**: Optimized C++ core with Python bindings.
- **Analytical Derivatives**: Efficient computation of gradients for optimization.
- **Versatile Dynamics**: Supports Forward/Inverse Dynamics, ABA, RNEA, and CRBA.
- **Geometry**: Collision detection and distance computation via HPP-FCL.

## 📁 Directory Structure

```
engines/physics_engines/pinocchio/
├── models/          # URDF model files
├── python/          # Python wrappers
└── tests/           # Engine-specific tests
```

## Integration Details

The suite uses Pinocchio for rapid dynamics calculations, which is ideal for:
- **Optimization Loops**: Where thousands of iterations are needed.
- **Control**: Real-time feedback controllers.

### Algorithms Used
- **ABA (Articulated Body Algorithm)**: For forward dynamics ($q, v, \tau \to \dot{v}$).
- **RNEA (Recursive Newton-Euler Algorithm)**: For inverse dynamics ($q, v, \dot{v} \to \tau$).

## Usage

Pinocchio models are typically loaded from URDF. The suite manages the `pinocchio.Model` and `pinocchio.Data` structures automatically.
