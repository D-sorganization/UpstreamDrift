# URDF Import/Export and Pinocchio Interface Guide

## Overview

This guide covers the new URDF import/export functionality and Pinocchio interface for advanced dynamics algorithms.

## URDF Import/Export

### Features

The URDF import/export module (`urdf_io.py`) enables:
- **Export MuJoCo models to URDF format** for sharing with ROS, Pinocchio, Drake, and other robotics frameworks
- **Import URDF models into MuJoCo** for using models from other sources
- **Automatic conversion** of joint types, geometries, and inertial properties
- **Preservation of visual and collision geometries**

### Exporting MuJoCo Models to URDF

```python
import mujoco
from mujoco_golf_pendulum.models import DOUBLE_PENDULUM_XML
from mujoco_golf_pendulum.urdf_io import export_model_to_urdf

# Load MuJoCo model
model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)

# Export to URDF
urdf_xml = export_model_to_urdf(
    model,
    "double_pendulum.urdf",
    model_name="double_pendulum",
    include_visual=True,
    include_collision=True,
)
```

### Importing URDF Models to MuJoCo

```python
from mujoco_golf_pendulum.urdf_io import import_urdf_to_mujoco
import mujoco

# Import URDF
mujoco_xml = import_urdf_to_mujoco("robot.urdf", model_name="imported_robot")

# Load in MuJoCo
model = mujoco.MjModel.from_xml_string(mujoco_xml)
data = mujoco.MjData(model)
```

### Supported Features

**Joint Types:**
- Revolute (hinge) ↔ `mjJNT_HINGE`
- Prismatic (slide) ↔ `mjJNT_SLIDE`
- Fixed joints (handled as constraints in MuJoCo)
- Continuous joints (treated as unlimited revolute)

**Geometry Types:**
- Box
- Sphere
- Cylinder
- Capsule (approximated as cylinder in URDF)
- Mesh (with scale support)

**Properties:**
- Inertial properties (mass, inertia matrix, center of mass)
- Visual geometries (with materials/colors)
- Collision geometries
- Joint limits and axes

### Limitations

- **Complex constraints**: MuJoCo equality constraints (welds, connects) are not fully represented in URDF
- **Actuators**: URDF transmission elements are not converted
- **Materials**: Complex material properties may be simplified
- **Meshes**: Mesh file paths need to be adjusted manually for URDF package structure

## Pinocchio Interface

### Overview

The Pinocchio interface (`pinocchio_interface.py`) provides a bridge between MuJoCo and Pinocchio, enabling:

- **Fast inverse dynamics** (RNEA) - compute required torques
- **Forward dynamics** (ABA) - compute accelerations from torques
- **Mass matrix computation** (CRBA) - analytical mass matrix
- **Coriolis and gravity terms** - explicit dynamics decomposition
- **Analytical Jacobians** - end-effector Jacobians
- **Dynamics derivatives** - ∂f/∂q, ∂f/∂v, ∂f/∂τ for optimization

### Installation

Pinocchio is an optional dependency. Install with:

```bash
pip install pin
```

Or using conda:

```bash
conda install -c conda-forge pinocchio
```

### Basic Usage

```python
import mujoco
from mujoco_golf_pendulum.models import DOUBLE_PENDULUM_XML
from mujoco_golf_pendulum.pinocchio_interface import PinocchioWrapper

# Load MuJoCo model
model = mujoco.MjModel.from_xml_string(DOUBLE_PENDULUM_XML)
data = mujoco.MjData(model)

# Create Pinocchio wrapper
wrapper = PinocchioWrapper(model, data)

# Compute inverse dynamics (required torques for desired accelerations)
q = data.qpos.copy()
v = data.qvel.copy()
a = np.array([0.0, 0.0])  # Desired accelerations
torques = wrapper.compute_inverse_dynamics(q, v, a)

# Compute forward dynamics (accelerations from torques)
tau = np.array([1.0, -0.5])  # Applied torques
a = wrapper.compute_forward_dynamics(q, v, tau)

# Compute mass matrix
M = wrapper.compute_mass_matrix(q)

# Compute Coriolis matrix
C = wrapper.compute_coriolis_matrix(q, v)

# Compute gravity vector
g = wrapper.compute_gravity_vector(q)
```

### Advanced Features

#### End-Effector Jacobians

```python
# Compute Jacobian for end-effector frame
J = wrapper.compute_end_effector_jacobian("club_head", q, local=True)

# J is [6 x nv] matrix: [linear_velocity; angular_velocity] = J @ qvel
```

#### Dynamics Derivatives

```python
# Compute analytical derivatives for optimization
df_dq, df_dv, df_dtau, df_du = wrapper.compute_dynamics_derivatives(q, v, tau)

# These are useful for:
# - Gradient-based trajectory optimization
# - Model-predictive control
# - Reinforcement learning (policy gradients)
```

#### Energy Computation

```python
# Compute kinetic and potential energy
KE = wrapper.compute_kinetic_energy(q, v)
PE = wrapper.compute_potential_energy(q)
total_energy = KE + PE
```

### Workflow: MuJoCo + Pinocchio

The recommended workflow combines both tools:

1. **MuJoCo for simulation**: Use MuJoCo for time-stepping, contacts, constraints
2. **Pinocchio for analysis**: Use Pinocchio for fast dynamics computations

```python
import mujoco
import numpy as np
from mujoco_golf_pendulum.pinocchio_interface import PinocchioWrapper

# Setup
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)
wrapper = PinocchioWrapper(model, data)

# Simulation loop
for step in range(1000):
    # Get current state
    q = data.qpos.copy()
    v = data.qvel.copy()
    
    # Use Pinocchio for fast inverse dynamics
    desired_a = compute_desired_acceleration(q, v)  # Your control law
    tau = wrapper.compute_inverse_dynamics(q, v, desired_a)
    
    # Apply torques in MuJoCo
    data.ctrl[:] = tau
    
    # Step simulation (MuJoCo handles contacts, constraints, etc.)
    mujoco.mj_step(model, data)
    
    # Optional: Use Pinocchio for analysis
    if step % 10 == 0:
        M = wrapper.compute_mass_matrix(q)
        J = wrapper.compute_end_effector_jacobian("club_head", q)
        # ... analysis ...
```

### State Synchronization

The wrapper provides methods to sync state between MuJoCo and Pinocchio:

```python
# Sync from MuJoCo to Pinocchio (for analysis)
wrapper.sync_mujoco_to_pinocchio()

# Sync from Pinocchio to MuJoCo (after optimization)
wrapper.sync_pinocchio_to_mujoco()
```

### Model Conversion

The Pinocchio wrapper automatically converts the MuJoCo model to Pinocchio format. It uses URDF as an intermediate format:

1. MuJoCo model → URDF (via `urdf_io`)
2. URDF → Pinocchio model

This ensures compatibility, though direct MJCF parsing (when available in Pinocchio) would be faster.

## Use Cases

### 1. Trajectory Optimization

```python
# Use Pinocchio for fast dynamics in optimization
def objective(trajectory):
    total_torque = 0.0
    for q, v, a in trajectory:
        tau = wrapper.compute_inverse_dynamics(q, v, a)
        total_torque += np.sum(tau**2)
    return total_torque

# Optimize trajectory
optimized_trajectory = optimize(objective, ...)
```

### 2. Model-Predictive Control

```python
# Use derivatives for MPC
df_dq, df_dv, df_dtau, _ = wrapper.compute_dynamics_derivatives(q, v, tau)

# Linearize dynamics: δa = df_dq @ δq + df_dv @ δv + df_dtau @ δτ
# Use in MPC formulation
```

### 3. Force Analysis

```python
# Compute required torques for a motion
q_traj, v_traj, a_traj = load_motion_capture_data()

torques = []
for q, v, a in zip(q_traj, v_traj, a_traj):
    tau = wrapper.compute_inverse_dynamics(q, v, a)
    torques.append(tau)
```

### 4. Model Sharing

```python
# Export model for sharing
export_model_to_urdf(model, "golf_swing_model.urdf")

# Others can import in ROS, Pinocchio, Drake, etc.
```

## Performance Notes

- **Pinocchio is fast**: RNEA/ABA are O(n) algorithms, much faster than numerical differentiation
- **MuJoCo handles contacts**: Use MuJoCo for simulation with contacts, constraints
- **Best of both worlds**: Use Pinocchio for analysis, MuJoCo for simulation

## Troubleshooting

### Pinocchio Not Found

If you get `ImportError: Pinocchio is required but not installed`:

```bash
pip install pin
```

### Model Conversion Issues

If model conversion fails:
1. Check that all bodies have valid inertial properties
2. Ensure joint types are supported
3. Verify geometry types are compatible

### State Synchronization

If states don't match:
- Check quaternion conventions (MuJoCo: w,x,y,z vs Pinocchio: x,y,z,w)
- Verify joint ordering matches between models
- Use `sync_mujoco_to_pinocchio()` before analysis

## References

- [Pinocchio Documentation](https://stack-of-tasks.github.io/pinocchio/)
- [URDF Specification](http://wiki.ros.org/urdf/XML)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)

