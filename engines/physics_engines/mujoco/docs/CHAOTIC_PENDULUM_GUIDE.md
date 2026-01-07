# Chaotic Driven Pendulum - Control Demonstration Model

## Overview

The chaotic driven pendulum is a simple yet powerful model for demonstrating various control principles and nonlinear dynamics phenomena. This model consists of a single pendulum with a bob attached to a horizontally-driven base, creating a system that can exhibit rich chaotic behavior depending on the forcing parameters.

## Model Description

### Physical Configuration

- **Pendulum Length**: 0.8 m
- **Bob Mass**: 1.0 kg
- **Rod Mass**: 0.05 kg (distributed along the rod)
- **Base Drive**: Horizontal slider (x-axis)
- **Damping**:
  - Pendulum hinge: 0.05 N·m·s/rad
  - Base slider: 0.1 N·s/m

### Degrees of Freedom

The system has **2 DOF**:
1. **Base Drive (x)**: Horizontal position of the pendulum's pivot point
2. **Pendulum Angle (θ)**: Angle of the pendulum from vertical (downward equilibrium)

### Actuators

1. **Base Drive Motor**: Controls horizontal oscillation of the base
   - Gear ratio: 10
   - Control range: -20 to 20 N
   - Purpose: Provides parametric/external forcing to induce complex dynamics

2. **Pendulum Motor**: Direct torque control on the pendulum
   - Gear ratio: 5
   - Control range: -10 to 10 N·m
   - Purpose: Enables active control experiments (stabilization, tracking, etc.)

## Physics and Dynamics

### Equations of Motion

The driven pendulum system is governed by:

```
θ̈ = -(g/L)sin(θ) - (b/mL²)θ̇ + (1/L)ẍ_base·cos(θ) + τ/(mL²)
```

Where:
- `θ` = pendulum angle from vertical
- `g` = gravitational acceleration (9.81 m/s²)
- `L` = pendulum length (0.8 m)
- `m` = bob mass (1.0 kg)
- `b` = damping coefficient
- `ẍ_base` = horizontal acceleration of the base
- `τ` = control torque

### Key Phenomena

This system can exhibit:

1. **Regular Periodic Motion**: At low forcing amplitudes and frequencies
2. **Resonance**: When forcing frequency matches natural frequency (~1.75 Hz for this pendulum)
3. **Subharmonic Resonance**: Response at integer fractions of forcing frequency
4. **Chaotic Motion**: Sensitive dependence on initial conditions
5. **Bifurcations**: Transitions between different dynamic regimes
6. **Strange Attractors**: Fractal structures in phase space

## Control Applications

### 1. Stabilization Control

**Objective**: Stabilize the pendulum at the unstable upright position (θ = π)

**Control Strategies**:
- **LQR (Linear Quadratic Regulator)**: Linearize around upright position
- **Energy-based Control**: Swing-up followed by stabilization
- **Sliding Mode Control**: Robust to disturbances and uncertainties

**Use the pendulum motor** actuator to apply stabilizing torques.

### 2. Trajectory Tracking

**Objective**: Make the pendulum follow a desired time-varying trajectory

**Examples**:
- Sinusoidal tracking
- Square wave response
- Polynomial trajectories

### 3. Vibration Isolation

**Objective**: Minimize pendulum motion despite base excitation

**Strategy**: Use pendulum motor to counteract the effects of base drive forcing.

### 4. Resonance Studies

**Objective**: Understand and control resonant behavior

**Experiment**:
1. Apply sinusoidal forcing via base drive: `x_base(t) = A·sin(2πf·t)`
2. Sweep forcing frequency `f` from 0.5 to 3.0 Hz
3. Observe amplitude response and identify resonance peaks
4. Test control strategies to dampen resonance

### 5. Chaos Suppression

**Objective**: Stabilize chaotic motion to periodic orbits

**Control Methods**:
- **OGY Method** (Ott-Grebogi-Yorke): Small perturbations to stabilize unstable periodic orbits
- **Time-Delayed Feedback**: Pyragas control
- **Adaptive Control**: Learn and adapt to chaotic dynamics

## Usage Examples

### Example 1: Free Oscillation with Damping

**Setup**:
- Base Drive Motor: 0 N (no forcing)
- Pendulum Motor: 0 N·m (no control)
- Initial angle: 30° from vertical

**Expected Behavior**: Pendulum oscillates and gradually decays to vertical equilibrium due to damping.

### Example 2: Driven Oscillation Near Resonance

**Setup**:
- Base Drive Motor: Apply sinusoidal forcing at 1.75 Hz (near natural frequency)
  - Control signal: `F(t) = 15·sin(2π·1.75·t)` N
- Pendulum Motor: 0 N·m
- Initial angle: Small perturbation (~5°)

**Expected Behavior**: Large amplitude oscillations due to resonance.

### Example 3: Swing-Up Control

**Setup**:
- Implement energy-based swing-up controller
- Switch to LQR stabilization near upright position

**Pseudocode**:
```python
# Calculate total energy
E = 0.5*m*L²*θ̇² + m*g*L*(1 - cos(θ))
E_target = 2*m*g*L  # Energy at upright position

if abs(θ - π) > 0.3:  # Swing-up phase
    τ = k_swingup * θ̇ * cos(θ) * (E - E_target)
else:  # Stabilization phase (LQR)
    τ = -K @ [θ - π, θ̇]
```

### Example 4: Exploring Chaos

**Setup**:
- Base Drive Motor: Strong forcing at moderate frequency
  - Example: `F(t) = 20·sin(2π·2.0·t)` N
- Pendulum Motor: 0 N·m
- Initial angle: Try multiple initial conditions

**Expected Behavior**:
- Chaotic motion with sensitive dependence on initial conditions
- Phase plot shows strange attractor
- Poincaré map reveals fractal structure

## Visualization and Analysis

### Recommended Plots

1. **Time Series**:
   - Pendulum angle θ(t)
   - Angular velocity θ̇(t)
   - Base position x(t)

2. **Phase Portrait**:
   - Plot θ̇ vs θ to visualize system trajectories
   - Look for limit cycles, attractors, or chaotic behavior

3. **Poincaré Map**:
   - Sample phase space at fixed intervals of forcing period
   - Useful for identifying periodic orbits and chaos

4. **Frequency Response**:
   - Sweep forcing frequency and measure response amplitude
   - Identify resonance peaks and bifurcation points

5. **Energy Plot**:
   - Total mechanical energy vs time
   - Useful for energy-based control design

## Comparison to Golf Swing Models

| Feature | Chaotic Pendulum | Golf Swing Models |
|---------|-----------------|-------------------|
| **Primary Purpose** | Control demonstration, education | Biomechanical analysis |
| **DOF** | 2 | 2-28 |
| **Complexity** | Minimal (single pendulum) | High (multi-body, anthropometric) |
| **Dynamics** | Highly nonlinear, chaotic | Complex but constrained |
| **Control Focus** | Stabilization, chaos, resonance | Trajectory optimization, coordination |
| **Applications** | Research, teaching, algorithm testing | Sports science, motion analysis |

## Implementation in Code

### Python Example (Sinusoidal Forcing)

```python
import numpy as np
import mujoco

# Load the chaotic pendulum model
xml = CHAOTIC_PENDULUM_XML
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)

# Simulation parameters
dt = 0.001  # timestep
t_max = 30.0  # simulation duration
n_steps = int(t_max / dt)

# Forcing parameters
forcing_freq = 1.75  # Hz (near resonance)
forcing_amplitude = 15  # Newtons

# Storage for results
time_history = []
angle_history = []
velocity_history = []

# Simulation loop
for step in range(n_steps):
    t = step * dt

    # Apply sinusoidal forcing to base drive
    base_force = forcing_amplitude * np.sin(2 * np.pi * forcing_freq * t)
    data.ctrl[0] = base_force  # Base drive motor

    # No pendulum control (for free response)
    data.ctrl[1] = 0.0  # Pendulum motor

    # Step simulation
    mujoco.mj_step(model, data)

    # Record data
    if step % 100 == 0:  # Subsample for efficiency
        time_history.append(t)
        angle_history.append(data.qpos[1])  # Pendulum angle
        velocity_history.append(data.qvel[1])  # Angular velocity

# Plot results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(time_history, angle_history)
plt.ylabel('Angle (rad)')
plt.grid(True)
plt.title('Driven Pendulum Response')

plt.subplot(3, 1, 2)
plt.plot(time_history, velocity_history)
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(angle_history, velocity_history)
plt.xlabel('Angle (rad)')
plt.ylabel('Angular Velocity (rad/s)')
plt.grid(True)
plt.title('Phase Portrait')

plt.tight_layout()
plt.show()
```

## References and Further Reading

1. **Nonlinear Dynamics and Chaos**: Steven Strogatz - Classic textbook on nonlinear systems
2. **Chaos in Dynamical Systems**: Edward Ott - Advanced treatment of chaotic systems
3. **Feedback Control of Dynamic Systems**: Franklin, Powell, Emami-Naeini - Control theory fundamentals
4. **The Duffing Equation**: Ivana Kovacic, Michael J. Brennan - Similar driven oscillator system

## Tips for Experimentation

1. **Start Simple**: Begin with free oscillation (no forcing) to understand basic dynamics
2. **Explore Parameter Space**: Vary forcing frequency and amplitude systematically
3. **Use Phase Plots**: Visualize behavior in phase space (θ, θ̇) for better insight
4. **Try Different Controllers**: Implement PID, LQR, or nonlinear control schemes
5. **Document Initial Conditions**: Chaotic systems are sensitive - record all parameters
6. **Long Simulations**: Run for 50+ forcing periods to observe long-term behavior

## Troubleshooting

**Problem**: Pendulum doesn't exhibit chaotic behavior
- **Solution**: Increase forcing amplitude or adjust frequency. Try forcing near 2× natural frequency.

**Problem**: Controller makes system unstable
- **Solution**: Reduce control gains. Check for phase lag in control loop.

**Problem**: Simulation appears frozen
- **Solution**: Check if pendulum is at stable equilibrium (θ = 0) with no forcing.

**Problem**: Results not reproducible
- **Solution**: This is actually expected for chaotic systems! Document random seed and initial conditions carefully.

---

**Model File**: `python/mujoco_golf_pendulum/models.py:7-70`
**Version**: Added in v2.2.0
**Contact**: For questions or contributions, please file an issue on GitHub
