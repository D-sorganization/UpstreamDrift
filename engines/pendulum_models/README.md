# Pendulum Models - Educational Golf Swing Simplified Models

Simplified pendulum models for educational purposes and rapid conceptual analysis of golf swing dynamics. These models reduce the complexity of the golf swing to its fundamental mechanical principles.

## Overview

Pendulum models represent the golf swing as a system of connected rigid links (pendulums), providing:

- **Educational Value**: Teach fundamental biomechanics and dynamics principles
- **Rapid Prototyping**: Quick testing of concepts before full implementation
- **Analytical Solutions**: Some configurations allow closed-form solutions
- **Computational Efficiency**: Extremely fast simulation (1000s per second)
- **Intuitive Understanding**: Visual and mathematical clarity

**These models are ideal for**: Education, teaching, initial research exploration, and developing intuition about swing mechanics.

## Available Pendulum Models

### 1. Single Pendulum (1 DOF)
**Description**: Simplest model - single rigid link representing the entire arm-club system

**Components**:
- One rigid link (arm + club combined)
- One revolute joint (shoulder)
- Fixed base (torso)

**Applications**:
- Teaching basic pendulum dynamics
- Understanding energy conservation
- Demonstrating phase space
- Introduction to nonlinear dynamics

### 2. Double Pendulum (2 DOF)
**Description**: Two-segment model representing arm and club

**Components**:
- Upper arm segment
- Forearm + club segment
- Two revolute joints (shoulder, wrist)

**Applications**:
- Chaotic dynamics demonstration
- Energy transfer between segments
- Timing and coordination
- Sensitivity to initial conditions

**Note**: Double pendulum exhibits chaotic behavior - small changes in initial conditions lead to dramatically different outcomes.

### 3. Triple Pendulum (3 DOF)
**Description**: Three-segment model with shoulder, elbow, and wrist

**Components**:
- Upper arm
- Forearm
- Club
- Three revolute joints (shoulder, elbow, wrist)

**Applications**:
- Multi-segment coordination
- Sequential energy transfer
- Kinetic chain analysis
- Optimal timing strategies

### 4. Inverted Pendulum (1-2 DOF)
**Description**: Upside-down pendulum models (unstable equilibrium)

**Applications**:
- Balance and stability analysis
- Control system design
- Postural control during swing
- Dynamic stability

## Implementation Platforms

### Python Implementation
**Location**: `python/`

```bash
cd pendulum_models/python
python -m pendulum_golf_swing
```

**Features**:
- NumPy/SciPy integration
- Matplotlib visualization
- Animation capabilities
- Integration with Python physics engines

### MATLAB Implementation
**Location**: `matlab/`

```matlab
cd pendulum_models/matlab
run_pendulum_model
```

**Features**:
- Symbolic Math Toolbox support
- Simulink models
- Optimization Toolbox integration
- Easy GUI development

### JavaScript Implementation
**Location**: `javascript/`

**Features**:
- Browser-based visualization
- Interactive web demos
- Real-time animation
- Educational web apps

### Arduino Implementation
**Location**: `arduino/`

**Features**:
- Physical pendulum control
- Sensor integration
- Real-time embedded systems
- Hands-on learning

## Quick Start

### Python Example

```python
import numpy as np
from pendulum_golf import DoublePendulum

# Create double pendulum
pendulum = DoublePendulum(
    L1=0.3,  # Upper arm length (m)
    L2=0.5,  # Forearm + club length (m)
    m1=2.0,  # Upper arm mass (kg)
    m2=1.5   # Forearm + club mass (kg)
)

# Set initial conditions
theta1_0 = np.pi / 4  # 45 degrees
theta2_0 = 0.0
omega1_0 = 0.0
omega2_0 = 0.0

# Simulate
t, states = pendulum.simulate(
    t_span=(0, 2),
    y0=[theta1_0, theta2_0, omega1_0, omega2_0],
    dt=0.001
)

# Visualize
pendulum.animate(t, states)
pendulum.plot_energy(t, states)
```

### MATLAB Example

```matlab
% Create double pendulum
params.L1 = 0.3;  % m
params.L2 = 0.5;  % m
params.m1 = 2.0;  % kg
params.m2 = 1.5;  % kg

% Initial conditions
y0 = [pi/4, 0, 0, 0];  % [theta1, theta2, omega1, omega2]

% Simulate
[t, y] = simulate_double_pendulum(params, y0, [0 2]);

% Visualize
animate_pendulum(t, y, params);
plot_phase_space(y);
```

## Mathematical Formulation

### Double Pendulum Equations of Motion

Using Lagrangian mechanics:

**Lagrangian**: L = T - V (kinetic - potential energy)

**Kinetic Energy**:
```
T = (1/2) * m1 * v1² + (1/2) * m2 * v2²
  + (1/2) * I1 * ω1² + (1/2) * I2 * ω2²
```

**Potential Energy**:
```
V = m1 * g * h1 + m2 * g * h2
```

**Equations of Motion** (highly nonlinear):
```
(m1 + m2) * L1² * θ1'' + m2 * L1 * L2 * θ2'' * cos(θ1 - θ2)
  + m2 * L1 * L2 * θ2'² * sin(θ1 - θ2)
  + (m1 + m2) * g * L1 * sin(θ1) = 0

m2 * L2² * θ2'' + m2 * L1 * L2 * θ1'' * cos(θ1 - θ2)
  - m2 * L1 * L2 * θ1'² * sin(θ1 - θ2)
  + m2 * g * L2 * sin(θ2) = 0
```

Where:
- θ₁, θ₂: Joint angles
- θ₁', θ₂': Angular velocities
- θ₁'', θ₂'': Angular accelerations
- L₁, L₂: Link lengths
- m₁, m₂: Link masses
- g: Gravitational acceleration

## Analysis Capabilities

### Energy Analysis
```python
# Compute energy
KE = pendulum.kinetic_energy(states)
PE = pendulum.potential_energy(states)
TE = KE + PE

# Plot energy conservation
plot_energy_conservation(t, KE, PE, TE)
```

### Phase Space Analysis
```python
# Phase portrait
plot_phase_space(theta1, omega1)
plot_phase_space(theta2, omega2)

# Poincaré section (for chaotic systems)
poincare_section(theta1, theta2, omega1, omega2)
```

### Sensitivity Analysis
```python
# Study sensitivity to initial conditions
initial_conditions = generate_ic_variations(theta0, epsilon=0.01)
for ic in initial_conditions:
    trajectory = pendulum.simulate(ic)
    trajectories.append(trajectory)

plot_trajectory_divergence(trajectories)
```

## Educational Applications

### Classroom Demonstrations

1. **Energy Conservation**
   - Show total energy remains constant
   - Visualize kinetic/potential exchange
   - Demonstrate energy loss with damping

2. **Chaos Theory**
   - Double pendulum chaotic behavior
   - Butterfly effect visualization
   - Lyapunov exponents

3. **Optimization**
   - Find initial conditions for maximum speed
   - Optimal timing of joint actuation
   - Trade-offs between speed and accuracy

4. **Control Systems**
   - Stabilize inverted pendulum
   - Track desired trajectories
   - PID controller design

### Student Projects

**Beginner Projects**:
- Measure period vs. amplitude
- Compare simulation to analytical solution
- Energy conservation verification

**Intermediate Projects**:
- Implement double pendulum from scratch
- Add damping and compare to experiments
- Optimize launch angle for distance

**Advanced Projects**:
- Chaotic dynamics analysis
- Control system design
- Comparison with full biomechanical models

## Comparison with Full Models

| Aspect | Pendulum Models | Full Biomechanical |
|--------|----------------|-------------------|
| DOF | 1-3 | 10-28 |
| Computation | Milliseconds | Seconds |
| Accuracy | Conceptual | High fidelity |
| Muscle Modeling | No | Yes (MuJoCo+MyoSuite) |
| Contact Dynamics | Simplified | Realistic |
| Use Case | Education | Research |
| Learning Curve | Easy | Steep |

**When to use pendulum models**:
- Teaching and learning
- Quick concept validation
- Developing intuition
- Computational efficiency needed

**When to use full models**:
- Research publications
- Detailed biomechanics
- Muscle-level analysis
- Realistic simulation

## Integration with Full Suite

### Export Pendulum Results
```python
# Simulate pendulum
results = pendulum.simulate()

# Export for comparison
pendulum.export_csv('pendulum_trajectory.csv')

# Compare with MuJoCo
from mujoco_golf import compare_with_pendulum
comparison = compare_with_pendulum('pendulum_trajectory.csv')
```

### Use as Initial Guess
```python
# Get rough trajectory from pendulum
pendulum_trajectory = simple_pendulum.optimize()

# Use as initial guess for full optimization
drake_trajectory = drake_optimizer.optimize(
    initial_guess=pendulum_trajectory.upsample()
)
```

## Visualization

### Animation
```python
# Real-time animation
pendulum.animate_realtime(states)

# Save animation
pendulum.animate(states, save_as='swing.gif')
pendulum.animate(states, save_as='swing.mp4')
```

### Plotting
```python
# Kinematics
plot_joint_angles(t, theta1, theta2)
plot_angular_velocities(t, omega1, omega2)

# Dynamics
plot_endpoint_trajectory(x_end, y_end)
plot_angular_momentum(t, L_angular)

# Phase space
plot_phase_portrait(theta, omega)
```

## Performance

Pendulum models are extremely fast:

| Model | Python | MATLAB | JavaScript |
|-------|--------|--------|------------|
| Single | 10,000× RT | 5,000× RT | 1,000× RT |
| Double | 5,000× RT | 2,000× RT | 500× RT |
| Triple | 2,000× RT | 1,000× RT | 200× RT |

RT = Real-time

**Typical use case**: 10,000 simulations in < 1 second

## Examples

### Example 1: Energy Conservation
```python
# examples/energy_conservation.py
pendulum = SinglePendulum(L=1.0, m=1.0)
t, states = pendulum.simulate(y0=[np.pi/2, 0], t_span=(0, 10))
pendulum.verify_energy_conservation(t, states)
```

### Example 2: Chaotic Behavior
```python
# examples/chaos_demo.py
pendulum = DoublePendulum()

# Two very similar initial conditions
ic1 = [np.pi/2, 0, 0, 0]
ic2 = [np.pi/2 + 0.001, 0, 0, 0]  # 0.001 rad difference

# Simulate both
traj1 = pendulum.simulate(ic1, t_span=(0, 20))
traj2 = pendulum.simulate(ic2, t_span=(0, 20))

# Show divergence
plot_trajectory_comparison(traj1, traj2)
```

### Example 3: Optimization
```python
# examples/optimize_swing.py
def objective(ic):
    """Maximize endpoint speed"""
    t, states = pendulum.simulate(ic)
    v_end = pendulum.endpoint_velocity(states[-1])
    return -np.linalg.norm(v_end)  # Minimize negative = maximize

# Optimize
from scipy.optimize import minimize
result = minimize(objective, x0=[0, 0, 0, 0], bounds=bounds)
optimal_ic = result.x
```

## Documentation

- [Python API](python/README.md) - Python implementation details
- [MATLAB Guide](matlab/README.md) - MATLAB implementation
- [JavaScript Demo](javascript/README.md) - Browser-based demos
- [Arduino Setup](arduino/README.md) - Physical pendulum control
- [Educational Resources](docs/README.md) - Teaching materials

## Related Documentation

- [Engine Selection Guide](../../docs/engine_selection_guide.md) - Compare with full models
- [MuJoCo README](../physics_engines/mujoco/README.md) - Full biomechanical models
- [Simscape Models](../Simscape_Multibody_Models/README.md) - MATLAB multibody models

## License

MIT License - See [LICENSE](LICENSE)

## Support

For questions about pendulum models:
- Review example scripts
- Check platform-specific READMEs (python/, matlab/, etc.)
- See [Golf Modeling Suite Issues](https://github.com/D-sorganization/Golf_Modeling_Suite/issues)
