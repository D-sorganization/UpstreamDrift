# Rigid Body Dynamics and Screw Theory Implementation

This document describes the implementation of Featherstone's rigid body dynamics algorithms and screw theory representations in the MuJoCo Golf Swing Model project.

## Overview

This implementation provides:

1. **Spatial Algebra**: Featherstone's 6D vector algebra for rigid body dynamics
2. **Rigid Body Dynamics**: RNEA, CRBA, and ABA algorithms
3. **Screw Theory**: Geometric representations using twists, wrenches, and exponential maps

Both MATLAB and Python implementations are provided, with identical APIs and comprehensive test suites.

## Contents

- [Spatial Algebra](#spatial-algebra)
- [Rigid Body Dynamics Algorithms](#rigid-body-dynamics-algorithms)
- [Screw Theory](#screw-theory)
- [Usage Examples](#usage-examples)
- [Testing](#testing)
- [References](#references)

## Spatial Algebra

Spatial algebra uses 6D vectors to represent motion and forces in 3D space. A spatial vector combines angular and linear components:

- **Motion vector**: `[ω; v]` (angular velocity; linear velocity)
- **Force vector**: `[n; f]` (moment; force)

### Key Functions

#### MATLAB (`matlab/spatial_v2/`)

```matlab
% Spatial cross products
X = crm(v)           % Motion cross product operator
X = crf(v)           % Force cross product operator (dual)
result = spatial_cross(v, u, 'motion')

% Spatial transformations (Plücker transforms)
X = xrot(E)          % Pure rotation
X = xlt(r)           % Pure translation
X = xtrans(E, r)     % General transform
X_inv = inv_xtrans(E, r)

% Spatial inertia
I = mcI(mass, com, I_com)  % Construct from mass, COM, inertia
I_A = transform_spatial_inertia(I_B, X)

% Joint kinematics
[Xj, S] = jcalc(jtype, q)  % Joint transform and motion subspace
```

#### Python (`python/mujoco_golf_pendulum/spatial_algebra/`)

```python
from mujoco_golf_pendulum.spatial_algebra import (
    crm, crf, spatial_cross,
    xrot, xlt, xtrans, inv_xtrans,
    mcI, transform_spatial_inertia,
    jcalc
)

# Spatial cross products
X = crm(v)
X = crf(v)
result = spatial_cross(v, u, 'motion')

# Spatial transformations
X = xrot(E)
X = xlt(r)
X = xtrans(E, r)

# Spatial inertia
I = mcI(mass, com, I_com)
I_A = transform_spatial_inertia(I_B, X)

# Joint kinematics
Xj, S = jcalc('Rz', q)
```

### Supported Joint Types

- `'Rx'`, `'Ry'`, `'Rz'`: Revolute joints about x, y, z axes
- `'Px'`, `'Py'`, `'Pz'`: Prismatic joints along x, y, z axes

## Rigid Body Dynamics Algorithms

Three fundamental O(n) algorithms from Featherstone's book:

### 1. RNEA (Recursive Newton-Euler Algorithm)

**Purpose**: Inverse dynamics - compute joint torques from motion

**Time Complexity**: O(n) where n is the number of bodies

```matlab
% MATLAB
tau = rnea(model, q, qd, qdd, f_ext);
```

```python
# Python
tau = rnea(model, q, qd, qdd, f_ext)
```

**Algorithm**:
1. Forward pass: Compute velocities and accelerations
2. Backward pass: Compute forces and project to joint torques

### 2. CRBA (Composite Rigid Body Algorithm)

**Purpose**: Compute the joint-space mass matrix H(q)

**Time Complexity**: O(n²)

```matlab
% MATLAB
H = crba(model, q);
```

```python
# Python
H = crba(model, q)
```

**Properties of H**:
- Symmetric: H = Hᵀ
- Positive definite
- Configuration-dependent

### 3. ABA (Articulated Body Algorithm)

**Purpose**: Forward dynamics - compute joint accelerations from torques

**Time Complexity**: O(n)

```matlab
% MATLAB
qdd = aba(model, q, qd, tau, f_ext);
```

```python
# Python
qdd = aba(model, q, qd, tau, f_ext)
```

**Algorithm**:
1. Forward pass: Kinematics
2. Backward pass: Compute articulated-body inertias
3. Forward pass: Compute accelerations

### Model Structure

Both MATLAB and Python use the same model dictionary/struct:

```matlab
% MATLAB
model.NB = 2;                    % Number of bodies
model.parent = [0, 1];           % Parent indices (0 = no parent)
model.jtype = {'Rz', 'Rz'};      % Joint types
model.Xtree = ...;               % Joint transforms (6x6xNB)
model.I = ...;                   % Spatial inertias (6x6xNB)
model.gravity = [0;0;0;0;0;-9.81];  % Gravity vector
```

```python
# Python
model = {
    'NB': 2,
    'parent': np.array([-1, 0]),  # -1 = no parent in Python
    'jtype': ['Rz', 'Rz'],
    'Xtree': [...],               # List of 6x6 arrays
    'I': [...],                   # List of 6x6 arrays
    'gravity': np.array([0, 0, 0, 0, 0, -9.81])
}
```

## Screw Theory

Screw theory provides a geometric framework for representing rigid body motions.

### Key Concepts

1. **Twist**: Instantaneous velocity `[ω; v]`
2. **Wrench**: Force and moment `[n; f]`
3. **Screw axis**: Axis of rotation/translation
4. **Exponential map**: Convert screw motion to transformation
5. **Adjoint**: Transform twists between frames

### MATLAB Functions (`matlab/screw_theory/`)

```matlab
% Twist and wrench conversions
V = twist_to_spatial(omega, v, point)
F = wrench_to_spatial(moment, force, point)

% Screw axes
S = screw_axis(axis, point, pitch)
T = screw_to_transform(axis, point, pitch, theta)

% Exponential and logarithmic maps
T = exponential_map(S, theta)
[S, theta] = logarithmic_map(T)

% Adjoint transformation
Ad = adjoint_transform(T)
```

### Python Functions (`python/mujoco_golf_pendulum/screw_theory/`)

```python
from mujoco_golf_pendulum.screw_theory import (
    twist_to_spatial, wrench_to_spatial,
    screw_axis, screw_to_transform,
    exponential_map, logarithmic_map,
    adjoint_transform
)

# Same API as MATLAB
V = twist_to_spatial(omega, v, point)
S = screw_axis(axis, point, pitch)
T = exponential_map(S, theta)
S, theta = logarithmic_map(T)
Ad = adjoint_transform(T)
```

### Screw Motion Parameters

- **Pure rotation**: pitch = 0
- **Pure translation**: pitch = inf
- **General screw**: pitch = h (distance per radian)

## Usage Examples

### MATLAB

Run the example scripts:

```bash
cd matlab/examples
matlab -nodisplay -r "example_2link_robot"
matlab -nodisplay -r "example_screw_theory"
```

### Python

Run the example scripts:

```bash
cd python/examples
python example_featherstone_algorithms.py
python example_screw_theory.py
```

### Quick Example: 2-Link Robot Dynamics

```python
import numpy as np
from mujoco_golf_pendulum.spatial_algebra import mcI, xlt
from mujoco_golf_pendulum.rigid_body_dynamics import rnea, crba, aba

# Create model (see full example for details)
model = create_2link_model()

# Configuration
q = np.array([0.5, -0.3])      # Joint angles
qd = np.array([0.1, 0.2])      # Joint velocities
qdd = np.array([0.5, -0.2])    # Joint accelerations

# Inverse dynamics (RNEA)
tau = rnea(model, q, qd, qdd)
print(f"Required torques: {tau}")

# Mass matrix (CRBA)
H = crba(model, q)
print(f"Mass matrix:\n{H}")

# Forward dynamics (ABA)
tau_applied = np.array([1.0, 0.5])
qdd_result = aba(model, q, qd, tau_applied)
print(f"Accelerations: {qdd_result}")
```

## Testing

### MATLAB Tests

Run all tests:

```bash
cd matlab/tests
matlab -nodisplay -r "test_spatial_algebra; test_screw_theory; test_rigid_body_dynamics"
```

### Python Tests

Run tests with pytest:

```bash
cd python
pytest tests/test_spatial_algebra.py -v
pytest tests/test_screw_theory.py -v
pytest tests/test_rigid_body_dynamics.py -v

# Or run all tests
pytest tests/ -v
```

### Test Coverage

- **Spatial algebra**: Cross products, transformations, inertia
- **RBD algorithms**: RNEA, CRBA, ABA correctness and consistency
- **Screw theory**: Exponential/logarithmic maps, adjoint properties
- **Edge cases**: Single-body systems, special configurations

## Implementation Details

### Differences from MuJoCo

The existing MuJoCo integration in `python/mujoco_golf_pendulum/` uses MuJoCo's internal dynamics engine. This new implementation provides:

1. **Educational value**: Explicit algorithms for learning
2. **Research flexibility**: Modify algorithms for custom dynamics
3. **Validation**: Cross-check against MuJoCo results
4. **Portability**: Pure NumPy/MATLAB, no MuJoCo dependency

### Numerical Precision

- All algorithms maintain symmetry of mass matrices
- Tolerance: ~1e-10 for most operations
- Articulated-body inertias protected against singularities

### Performance

- RNEA and ABA: O(n) time complexity
- CRBA: O(n²) time complexity
- Suitable for real-time control of systems with <100 bodies

## References

### Books

1. **Featherstone, R. (2008)**. *Rigid Body Dynamics Algorithms*.
   Cambridge University Press.
   - The definitive reference for spatial algebra and RBD algorithms
   - Chapters 2, 5, 6, 7 cover the implemented algorithms

2. **Murray, R. M., Li, Z., & Sastry, S. S. (1994)**. *A Mathematical Introduction to Robotic Manipulation*.
   CRC Press.
   - Screw theory foundations
   - Exponential coordinates

3. **Lynch, K. M., & Park, F. C. (2017)**. *Modern Robotics: Mechanics, Planning, and Control*.
   Cambridge University Press.
   - Modern treatment of screw theory
   - Excellent examples and exercises

### Papers

- Featherstone, R. (1983). "The calculation of robot dynamics using articulated-body inertias." *IJRR*, 2(1), 13-30.
- Park, F. C. (1995). "Distance metrics on the rigid-body motions with applications to mechanism design." *ASME J. Mech. Design*, 117(1), 48-54.

### Online Resources

- [Featherstone's website](http://royfeatherstone.org/spatial/) - Spatial vector algebra resources
- [Modern Robotics textbook](http://modernrobotics.org/) - Free online book and code library

## Future Enhancements

Potential extensions:

1. **Multi-DOF joints**: Spherical, universal joints
2. **Floating base**: For humanoid robots
3. **Constraints**: Closed kinematic chains, contact
4. **Hybrid dynamics**: Impacts and switching
5. **Visualization**: Animate robot motion

## License and Citation

This implementation follows the same license as the parent project. When using these algorithms, please cite:

```bibtex
@book{featherstone2008rigid,
  title={Rigid Body Dynamics Algorithms},
  author={Featherstone, Roy},
  year={2008},
  publisher={Springer}
}
```

## Contact

For questions or issues with this implementation, please open an issue on the project repository.
