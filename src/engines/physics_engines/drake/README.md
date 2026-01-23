# Drake Golf Swing Model - Model-Based Design and Control

Drake integration provides advanced trajectory optimization, contact modeling, and system analysis tools for golf swing biomechanics research and control design.

## Overview

Drake is a C++-based toolbox for model-based design and verification of robotics systems. This integration brings Drake's powerful capabilities to golf swing analysis:

- **Trajectory Optimization**: Generate optimal swing motions
- **Contact Modeling**: Sophisticated ground and ball contact dynamics
- **System Analysis**: Linearization, reachability, and stability tools
- **Visualization**: Real-time 3D rendering with MeshCat
- **Control Design**: Model predictive control and optimal control
- **Cross-Validation**: Compare with MuJoCo and Pinocchio results

## Features

### Trajectory Optimization
- Direct collocation methods (Hermite-Simpson, trapez oidal)
- Nonlinear programming (SNOPT, IPOPT)
- Multi-phase optimization
- Constraints on joint limits, torques, and contact forces
- Cost functions for speed, accuracy, smoothness, energy

### Contact Dynamics
- Compliant contact models
- Hybrid mode automaton (stance/flight transitions)
- Ground reaction forces
- Ball contact dynamics
- Multi-contact scenarios

### System Analysis
- Linearization around trajectories
- Finite-time LQR (time-varying)
- Reachability analysis
- Lyapunov analysis for stability

### Visualization
- MeshCat browser-based 3D viewer
- Real-time playback of optimized trajectories
- Contact force visualization
- Multiple camera views

## Installation

### Prerequisites

Drake requires:
- Python 3.10-3.11 (Drake not yet compatible with 3.12+)
- macOS, Ubuntu 20.04/22.04, or Windows with WSL2

### Install Drake

```bash
# Using pip (recommended)
pip install drake

# Or using conda
conda install -c conda-forge drake

# Verify installation
python -c "import pydrake; print(pydrake.getDrakePath())"
```

### Install Drake Golf Suite

```bash
# From suite root
pip install -r requirements.txt

# Or install Drake dependencies separately
cd engines/physics_engines/drake/python
pip install -r requirements.txt
```

## Quick Start

### Run Drake Golf Simulation

```bash
# Launch Drake GUI
python engines/physics_engines/drake/python/golf_swing_launcher.py

# Or use unified launcher
python launchers/golf_launcher.py --engine drake
```

### Basic Python Usage

```python
from drake_physics_engine import DrakePhysicsEngine

# Initialize Drake engine
engine = DrakePhysicsEngine(model_name="upper_body_golfer")

# Set initial configuration
q0 = engine.get_nominal_configuration()
engine.set_state(q0, v0=np.zeros_like(q0))

# Simulate forward dynamics
for t in np.arange(0, 2.0, 0.01):
    tau = compute_control(engine.get_state())
    engine.step(tau, dt=0.01)

    # Get club head position
    club_pos = engine.get_club_head_position()
    club_vel = engine.get_club_head_velocity()
```

## Trajectory Optimization

### Optimize Golf Swing

```python
from drake_trajectory_optimization import SwingOptimizer

# Create optimizer
optimizer = SwingOptimizer(
    model_file="golfer_urdf.urdf",
    timesteps=50,
    min_time=0.5,
    max_time=2.0
)

# Define objective: maximize club head speed
optimizer.add_objective_club_head_speed(weight=1.0)

# Add constraints
optimizer.add_joint_limit_constraints()
optimizer.add_torque_limit_constraints(max_torque=100.0)  # N⋅m
optimizer.add_initial_configuration_constraint(q_init)
optimizer.add_contact_force_constraints()

# Solve optimization
result = optimizer.solve()

if result.is_success():
    trajectory = result.get_trajectory()
    club_speed = result.get_max_club_speed()
    print(f"Optimized club speed: {club_speed:.1f} m/s")

    # Visualize result
    optimizer.visualize(trajectory)
else:
    print(f"Optimization failed: {result.get_solver_details()}")
```

### Multi-Objective Optimization

```python
# Optimize for multiple objectives
optimizer.add_objective_club_head_speed(weight=1.0)
optimizer.add_objective_smoothness(weight=0.1)
optimizer.add_objective_energy_efficiency(weight=0.05)
optimizer.add_objective_accuracy(target_direction=[1, 0, 0], weight=0.5)

# Solve
result = optimizer.solve()
pareto_front = optimizer.get_pareto_front()
```

## Contact Modeling

### Ground Contact

```python
from drake_contact import GroundContactModel

# Add ground contact
contact = GroundContactModel(
    stiffness=1e6,  # N/m
    dissipation=10.0,  # s/m
    friction_coeff=0.8
)

engine.add_ground_contact(
    body_name="left_foot",
    contact_points=foot_contact_points,
    contact_model=contact
)

# Simulate with contact
for t in time_steps:
    engine.step(tau, dt)
    grf = engine.get_ground_reaction_forces()
```

### Ball Contact

```python
from drake_contact import BallContactModel

# Add ball contact
ball_contact = BallContactModel(
    ball_mass=0.0459,  # kg (USGA regulation)
    ball_radius=0.0214,  # m
    restitution=0.78,
    friction=0.2
)

engine.add_ball_contact(
    club_face_body="club_head",
    ball_initial_position=[0.5, 0, 0.1]
)

# Simulate impact
result = engine.simulate_ball_strike(
    club_trajectory=optimized_trajectory,
    record_ball_flight=True
)

ball_velocity = result.get_ball_launch_velocity()
ball_spin = result.get_ball_spin_rate()
```

## System Analysis

### Linearization

```python
from drake_analysis import LinearSystemAnalyzer

# Linearize around trajectory
analyzer = LinearSystemAnalyzer(engine)
linear_systems = analyzer.linearize_trajectory(
    trajectory=reference_trajectory,
    num_samples=50
)

# Analyze stability
for t, sys in zip(time_samples, linear_systems):
    eigenvalues = sys.get_eigenvalues()
    controllable = sys.is_controllable()
    observable = sys.is_observable()
```

### LQR Control Design

```python
from drake.systems.controllers import FiniteHorizonLinearQuadraticRegulator

# Design time-varying LQR
lqr_result = FiniteHorizonLinearQuadraticRegulator(
    system=plant,
    context=context,
    t0=0.0,
    tf=swing_duration,
    Q=state_cost_matrix,
    R=control_cost_matrix,
    Qf=terminal_cost_matrix
)

# Extract controller
controller = lqr_result.get_controller()
```

## Visualization

### MeshCat Visualization

```python
from drake_visualization import MeshCatVisualizer

# Create visualizer
viz = MeshCatVisualizer(engine)

# Visualize trajectory
viz.start_recording()
for t in time:
    engine.set_state(trajectory.value(t))
    viz.publish()
    time.sleep(0.01)
viz.stop_recording()

# Save animation
viz.save_animation("golf_swing.html")
```

### Real-Time Simulation Visualization

```python
# Run simulation with live visualization
viz = MeshCatVisualizer(engine)
viz.start_realtime()

for t in np.arange(0, 2.0, 0.01):
    tau = controller.compute_control(engine.get_state(), t)
    engine.step(tau, dt=0.01)
    viz.publish(engine.get_state())

viz.stop()
```

## Available Models

### Golf Swing Models

1. **Simple Pendulum (2 DOF)**
   - Shoulder + wrist
   - Fast optimization

2. **Upper Body (10 DOF)**
   - Torso + arms + club
   - Realistic swing mechanics

3. **Full Body (15 DOF)**
   - Legs + torso + arms
   - Weight transfer analysis

4. **Advanced (28 DOF)**
   - Scapulae + flexible shaft
   - Research-grade

All models available in URDF format compatible with Drake.

## Performance

Drake provides excellent performance for optimization and analysis:

| Operation | Time | Notes |
|-----------|------|-------|
| Forward simulation | 10-50× realtime | Depends on model complexity |
| Trajectory optimization | 1-60 seconds | 50-100 timesteps |
| Linearization | <1 second | Per sample point |
| LQR synthesis | <1 second | Per time sample |

## Best Practices

### When to Use Drake

✅ **Use Drake for**:
- Trajectory optimization
- Optimal control design
- System analysis and verification
- Contact-rich scenarios with known contact sequence
- Model predictive control

❌ **Prefer MuJoCo for**:
- Real-time interactive simulation
- Muscle-level biomechanics
- Large-scale parameter sweeps
- Unknown contact sequences

### Optimization Tips

1. **Start Simple**: Begin with coarse timesteps, refine iteratively
2. **Warm Start**: Use simple trajectory as initial guess
3. **Scale Properly**: Normalize states and controls
4. **Check Gradients**: Use `check_constraints()` to verify
5. **Adjust Solver**: Try different NLP solvers (SNOPT usually best)

## Examples

### Example 1: Optimize for Speed

```python
# Complete optimization script
optimizer = SwingOptimizer(model="upper_body_golfer")
optimizer.add_objective_club_head_speed(weight=1.0)
optimizer.add_joint_limit_constraints()
optimizer.add_initial_posture_constraint(q_init)
optimizer.add_torque_limit_constraints(max_torque=150.0)

result = optimizer.solve()
trajectory = result.get_trajectory()

# Analyze result
max_speed = result.get_max_club_speed()
energy = result.get_total_energy()
smoothness = result.get_smoothness_metric()

print(f"Club speed: {max_speed:.1f} m/s")
print(f"Energy: {energy:.1f} J")
print(f"Smoothness: {smoothness:.3f}")
```

### Example 2: MPC Controller

```python
from drake.systems.controllers import LinearModelPredictiveController

# Create MPC
mpc = LinearModelPredictiveController(
    system=plant,
    context=context,
    Q=Q_matrix,
    R=R_matrix,
    horizon=20,  # timesteps
    dt=0.05  # seconds
)

# Run MPC-controlled simulation
for t in time:
    current_state = engine.get_state()
    reference = trajectory.value(t)
    tau = mpc.compute_control(current_state, reference)
    engine.step(tau, dt=0.05)
```

## API Reference

### DrakePhysicsEngine

```python
class DrakePhysicsEngine:
    def __init__(self, model_file: str, dt: float = 0.001)
    def set_state(self, q: np.ndarray, v: np.ndarray)
    def get_state(self) -> Tuple[np.ndarray, np.ndarray]
    def step(self, tau: np.ndarray, dt: float) -> None
    def get_club_head_position(self) -> np.ndarray
    def get_club_head_velocity(self) -> np.ndarray
    def add_ground_contact(...) -> None
    def get_ground_reaction_forces(self) -> np.ndarray
```

### SwingOptimizer

```python
class SwingOptimizer:
    def add_objective_club_head_speed(self, weight: float)
    def add_objective_smoothness(self, weight: float)
    def add_objective_energy_efficiency(self, weight: float)
    def add_joint_limit_constraints(self)
    def add_torque_limit_constraints(self, max_torque: float)
    def solve(self) -> OptimizationResult
```

## Troubleshooting

### Common Issues

**Issue**: Drake fails to import
- **Solution**: Ensure Python 3.10-3.11, not 3.12+

**Issue**: Optimization doesn't converge
- **Solution**: Adjust initial guess, scale variables, try different solver

**Issue**: Contact forces are unstable
- **Solution**: Increase contact stiffness, reduce timestep

## Documentation

- **[Drake Official Documentation](https://drake.mit.edu/)**
- **[Drake Python API](https://drake.mit.edu/pydrake/)**
- **[Trajectory Optimization Guide](https://drake.mit.edu/doxygen_cxx/group__planning__trajectory__optimization.html)**
- **[Contact Modeling](https://drake.mit.edu/doxygen_cxx/group__multibody__contact.html)**
- **[Engine Selection Guide](../../../docs/engine_selection_guide.md)**

## Citation

If you use Drake in your research, please cite:

```bibtex
@misc{drake,
  author = {Russ Tedrake and the Drake Development Team},
  title = {Drake: Model-based design and verification for robotics},
  year = {2019},
  url = {https://drake.mit.edu}
}
```

## License

Drake is licensed under the BSD 3-Clause License. See the [Drake repository](https://github.com/RobotLocomotion/drake) for details.

## Support

For Drake-specific questions:
- [Drake GitHub Discussions](https://github.com/RobotLocomotion/drake/discussions)
- [Drake Slack Channel](https://drake.mit.edu/getting_help.html)

For Golf Modeling Suite integration:
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
