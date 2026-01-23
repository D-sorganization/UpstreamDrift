# Pinocchio Golf Model - Fast Rigid Body Dynamics

Pinocchio integration provides high-performance rigid body dynamics algorithms, enabling efficient computation of kinematics, dynamics, and derivatives for golf swing analysis and control.

## Overview

Pinocchio is a C++ library with Python bindings for efficient rigid body dynamics algorithms. This integration brings Pinocchio's computational speed to golf swing biomechanics:

- **High-Performance Dynamics**: 10-100× faster than symbolic approaches
- **Analytical Derivatives**: Efficient computation of Jacobians and Hessians
- **Constrained Systems**: Loop closures and contact constraints
- **Inverse Kinematics**: PINK task-space IK solver
- **Algorithms Collection**: RNEA, ABA, CRBA, and more
- **Cross-Validation**: Compare with MuJoCo and Drake results

## Features

### Rigid Body Dynamics Algorithms
- **RNEA** (Recursive Newton-Euler): Inverse dynamics O(n)
- **ABA** (Articulated Body Algorithm): Forward dynamics O(n)
- **CRBA** (Composite Rigid Body Algorithm): Mass matrix O(n²)
- **Centroidal Dynamics**: COM kinematics and momentum
- **Energy Computation**: Kinetic and potential energy

### Kinematics & Jacobians
- Forward kinematics for all bodies
- Geometric and analytical Jacobians
- Jacobian derivatives (velocity product)
- Frame transformations
- Spatial velocities and accelerations

### Constrained Dynamics
- Loop closure constraints (two-handed grip)
- Contact constraints
- Constraint Jacobians
- Constrained forward dynamics

### Inverse Kinematics (PINK)
- Task-space goals (position, orientation)
- Multiple simultaneous tasks with priorities
- Joint limit avoidance
- Collision avoidance
- Real-time capable

### Visualization
- **MeshCat**: Browser-based 3D visualization
- **Gepetto-Viewer**: Desktop Qt-based viewer
- URDF/MJCF model loading
- Real-time playback

## Installation

### Prerequisites

```bash
# Install Pinocchio
conda install -c conda-forge pinocchio

# Or using pip
pip install pin

# Install PINK for inverse kinematics
pip install pink

# Install visualization
pip install meshcat
```

### Verification

```python
import pinocchio as pin
import pink
print(f"Pinocchio version: {pin.__version__}")
print(f"PINK version: {pink.__version__}")
```

## Quick Start

### Basic Usage

```bash
# Launch Pinocchio GUI
python engines/physics_engines/pinocchio/python/golf_swing_launcher.py

# Or use unified launcher
python launchers/golf_launcher.py --engine pinocchio
```

### Python API

```python
from pinocchio_physics_engine import PinocchioPhysicsEngine
import pinocchio as pin

# Initialize engine
engine = PinocchioPhysicsEngine(urdf_file="golfer.urdf")

# Set configuration
q = engine.get_neutral_configuration()
v = np.zeros(engine.model.nv)

# Compute forward kinematics
pin.forwardKinematics(engine.model, engine.data, q, v)

# Get club head position
club_frame_id = engine.model.getFrameId("club_head")
club_pos = engine.data.oMf[club_frame_id].translation

# Compute Jacobian
J = pin.computeFrameJacobian(
    engine.model, engine.data, q,
    club_frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

# Compute inverse dynamics
tau = pin.rnea(engine.model, engine.data, q, v, a)
```

## Dynamics Algorithms

### Inverse Dynamics (RNEA)

```python
import pinocchio as pin

# Compute required torques for given motion
tau = pin.rnea(model, data, q, v, a)

# With external forces
f_ext = pin.StdVec_Force()
f_ext.append(pin.Force.Zero())  # For each joint
tau = pin.rnea(model, data, q, v, a, f_ext)
```

### Forward Dynamics (ABA)

```python
# Compute acceleration from applied torques
a = pin.aba(model, data, q, v, tau)

# Simulate forward
dt = 0.001
for t in np.arange(0, 2.0, dt):
    a = pin.aba(model, data, q, v, tau)
    v += a * dt
    q = pin.integrate(model, q, v * dt)
```

### Mass Matrix (CRBA)

```python
# Compute mass/inertia matrix
M = pin.crba(model, data, q)

# Compute Coriolis matrix
C = pin.computeCoriolisMatrix(model, data, q, v)

# Compute gravity vector
g = pin.computeGeneralizedGravity(model, data, q)

# Equations of motion: M(q)·a + C(q,v)·v + g(q) = tau
```

### Centroidal Dynamics

```python
# Compute COM position and velocity
pin.centerOfMass(model, data, q, v)
com_pos = data.com[0]
com_vel = data.vcom[0]

# Compute centroidal momentum
pin.computeCentroidalMomentum(model, data, q, v)
h = data.hg  # Spatial momentum

# Centroidal momentum matrix
Ag = pin.computeCentroidalMomentumTimeVariation(model, data, q, v, a)
```

## Jacobians and Derivatives

### Compute Jacobians

```python
# Frame Jacobian (body-fixed or world-aligned)
J = pin.computeFrameJacobian(
    model, data, q, frame_id,
    reference_frame=pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)

# Joint Jacobian
J_joint = pin.getJointJacobian(model, data, joint_id, pin.ReferenceFrame.WORLD)

# COM Jacobian
Jcom = pin.jacobianCenterOfMass(model, data, q)
```

### Jacobian Derivatives

```python
# Jacobian time derivative
dJ = pin.computeJointJacobiansTimeVariation(model, data, q, v)

# Frame Jacobian time derivative
dJ_frame = pin.getFrameJacobianTimeVariation(
    model, data, frame_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
)
```

## Inverse Kinematics with PINK

### Task-Space IK

```python
from pink import Configuration, Task
from pink.tasks import FrameTask, PostureTask

# Create configuration
config = Configuration(model, data, q_init)

# Define tasks
club_task = FrameTask(
    frame_name="club_head",
    position_cost=1.0,
    orientation_cost=0.1
)
club_task.set_target_from_se3(target_SE3)

posture_task = PostureTask(
    cost=1e-3  # Low weight for secondary objective
)
posture_task.set_target(q_nominal)

# Solve IK
tasks = [club_task, posture_task]
q_result = config.solve_ik(tasks, dt=0.1, max_iters=100)
```

### Multi-Task IK with Priorities

```python
from pink.tasks import FrameTask, JointCouplingTask

# Primary task: Club head position
primary_task = FrameTask("club_head", position_cost=1.0)
primary_task.set_target_position(target_pos)

# Secondary task: Elbow configuration
secondary_task = FrameTask("elbow", position_cost=0.1)

# Joint limits task (lowest priority)
limits_task = PostureTask(cost=1e-4)

# Solve with task hierarchy
q_result = config.solve_ik_with_hierarchy(
    [primary_task, secondary_task, limits_task],
    dt=0.1
)
```

## Constrained Dynamics

### Loop Closure Constraints

```python
# Two-handed grip as loop closure
import pinocchio as pin

# Define constraint
constraint = pin.RigidConstraintModel(
    type=pin.ContactType.CONTACT_6D,
    joint1_id=left_hand_id,
    joint2_id=right_hand_id,
    joint1_placement=left_hand_placement,
    joint2_placement=right_hand_placement
)

# Create constrained model
constrained_model = pin.createConstrainedModel(model, [constraint])

# Solve constrained forward dynamics
pin.constrainedForwardDynamics(
    constrained_model, constrained_data, q, v, tau
)
```

## Visualization

### MeshCat Visualization

```python
from pinocchio.visualize import MeshcatVisualizer

# Create visualizer
viz = MeshcatVisualizer(model, collision_model, visual_model)
viz.initViewer(open=True)
viz.loadViewerModel()

# Display configuration
viz.display(q)

# Animate trajectory
for q_t in trajectory:
    viz.display(q_t)
    time.sleep(0.01)
```

### Gepetto Viewer

```python
from pinocchio.visualize import GepettoVisualizer

viz = GepettoVisualizer(model, collision_model, visual_model)
viz.initViewer()
viz.loadViewerModel()
viz.display(q)
```

## Performance

Pinocchio is highly optimized for computational efficiency:

| Algorithm | DOF=10 | DOF=30 | DOF=100 |
|-----------|--------|--------|---------|
| RNEA | 2 µs | 5 µs | 15 µs |
| ABA | 5 µs | 15 µs | 50 µs |
| CRBA | 8 µs | 30 µs | 150 µs |
| Jacobian | 3 µs | 8 µs | 25 µs |

**Real-time capable** for models up to ~100 DOF

## Best Practices

### When to Use Pinocchio

✅ **Use Pinocchio for**:
- High-performance dynamics computation
- Analytical derivatives needed
- Complex kinematic chains
- Inverse kinematics with multiple tasks
- Centroidal dynamics analysis
- Cross-validation of dynamics

❌ **Prefer MuJoCo for**:
- Contact-rich simulation
- Muscle-level biomechanics
- Interactive visualization
- Unknown contact sequences

### Optimization with Pinocchio

Pinocchio is ideal for gradient-based optimization:

```python
# Efficient gradient computation
def objective(q, v):
    pin.forwardKinematics(model, data, q, v)
    club_pos = data.oMf[club_frame_id].translation
    return -club_pos[0]  # Maximize forward distance

def gradient(q, v):
    J = pin.computeFrameJacobian(model, data, q, club_frame_id, ...)
    return -J[0, :]  # Gradient of objective

# Use with scipy.optimize, CasADi, etc.
```

## Available Models

### Golf Swing Models

1. **Simple Pendulum (2 DOF)** - Educational
2. **Upper Body (10 DOF)** - Realistic swing
3. **Full Body (23 DOF)** - Complete body
4. **Advanced (28 DOF)** - Research-grade

All models in URDF format compatible with Pinocchio.

## Integration with Other Engines

### Cross-Validation

```python
from pinocchio_physics_engine import PinocchioPhysicsEngine
from mujoco_physics_engine import MuJoCoPhysicsEngine
from cross_engine_validator import CrossEngineValidator

# Initialize engines
pin_engine = PinocchioPhysicsEngine("golfer.urdf")
mj_engine = MuJoCoPhysicsEngine("golfer.xml")

# Validate inverse dynamics
validator = CrossEngineValidator()
results = validator.compare_inverse_dynamics(
    engines=[pin_engine, mj_engine],
    trajectory=test_trajectory
)

print(f"RMS torque difference: {results.rms_difference}")
```

## Examples

### Example 1: Swing Analysis

```python
import pinocchio as pin
from pinocchio_physics_engine import PinocchioPhysicsEngine

# Load model
engine = PinocchioPhysicsEngine("golfer.urdf")

# Analyze swing trajectory
for q, v, a in swing_trajectory:
    # Compute inverse dynamics
    tau = pin.rnea(engine.model, engine.data, q, v, a)

    # Compute club head velocity
    pin.forwardKinematics(engine.model, engine.data, q, v)
    J = pin.computeFrameJacobian(...)
    club_vel = J @ v

    # Analyze power
    power = tau.T @ v
```

### Example 2: Optimal Trajectory

```python
from pink import Configuration
import numpy as np

# Define waypoints
waypoints = [q_start, q_mid, q_impact, q_finish]
times = [0.0, 0.5, 1.0, 1.5]

# Interpolate with IK
trajectory = []
for t in np.linspace(times[0], times[-1], 100):
    # Interpolate target
    target = interpolate_waypoints(waypoints, times, t)

    # Solve IK
    task.set_target(target)
    q_t = config.solve_ik([task], dt=0.01)
    trajectory.append(q_t)
```

## Documentation

- **[Pinocchio Official Docs](https://stack-of-tasks.github.io/pinocchio/)**
- **[PINK Documentation](https://github.com/stephane-caron/pink)**
- **[Pinocchio Tutorials](https://github.com/stack-of-tasks/pinocchio/tree/master/examples)**
- **[Engine Selection Guide](../../../docs/engine_selection_guide.md)**

## Citation

If you use Pinocchio in your research, please cite:

```bibtex
@inproceedings{carpentier2019pinocchio,
  title={The Pinocchio C++ library: A fast and flexible implementation of rigid body dynamics algorithms and their analytical derivatives},
  author={Carpentier, Justin and Saurel, Guilhem and Buondonno, Gabriele and Mirabel, Joseph and Lamiraux, Florent and Stasse, Olivier and Mansard, Nicolas},
  booktitle={2019 IEEE/SICE International Symposium on System Integration (SII)},
  pages={614--619},
  year={2019},
  organization={IEEE}
}
```

## License

Pinocchio is licensed under the BSD-2-Clause License. See the [Pinocchio repository](https://github.com/stack-of-tasks/pinocchio) for details.

## Support

For Pinocchio-specific questions:
- [Pinocchio GitHub Issues](https://github.com/stack-of-tasks/pinocchio/issues)
- [Pinocchio Discussions](https://github.com/stack-of-tasks/pinocchio/discussions)

For Golf Modeling Suite integration:
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
