# Chapter 10: OpenSim Physics Engine

## 10.1 Overview

The OpenSim engine provides integration with the OpenSim musculoskeletal modeling platform,
the de facto standard in biomechanics research. Within UpstreamDrift, OpenSim serves as the
**biomechanical validation engine** -- its Hill-type muscle models, established model library,
and clinical validation history make it the gold standard for verifying that simulation
results are physiologically plausible.

### Source Files

| File                                                                   | Purpose                                      |
| ---------------------------------------------------------------------- | -------------------------------------------- |
| `src/engines/physics_engines/opensim/python/opensim_physics_engine.py` | Full `PhysicsEngine` protocol implementation |
| `src/engines/physics_engines/opensim/python/muscle_analysis.py`        | Muscle analysis and grip modeling            |
| `src/engines/physics_engines/opensim/python/opensim_gui.py`            | Diagnostic GUI                               |

### Dependencies

```
pip install opensim      # OpenSim Python bindings (>= 4.4)
pip install numpy
```

Availability is checked via `src/shared/python/engine_availability.OPENSIM_AVAILABLE`.

---

## 10.2 Engine Initialization and Model Loading

### 10.2.1 Loading `.osim` Models

OpenSim uses its own XML-based model format (`.osim`):

```python
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)

engine = OpenSimPhysicsEngine()
engine.load_from_path("/path/to/model.osim")

print(f"Model: {engine.model_name}")
print(f"Coordinates: {engine._model.getNumCoordinates()}")
print(f"Speeds: {engine._model.getNumSpeeds()}")
print(f"Controls: {engine._model.getNumControls()}")
```

The loading sequence:

1. `opensim.Model(path)` -- Parse the `.osim` XML file
2. `model.initSystem()` -- Build the Simbody multibody tree and initialize the `State`
3. `opensim.Manager(model)` -- Create an integration manager for time stepping

### 10.2.2 Loading From String

For dynamically generated models, loading from an XML string is supported via a temporary
file:

```python
osim_xml = """<?xml version="1.0" encoding="UTF-8"?>
<OpenSimDocument Version="40000">
  <Model name="simple_pendulum">
    ...
  </Model>
</OpenSimDocument>"""

engine.load_from_string(osim_xml, extension="osim")
```

The temporary file is cleaned up automatically after loading.

### 10.2.3 State Management

OpenSim separates configuration ($Q$, generalized coordinates) from speeds ($U$,
generalized velocities):

```python
# Get state
q, v = engine.get_state()
# q: Generalized coordinates (n_q,) -- joint angles, translations
# v: Generalized speeds (n_u,) -- joint angular velocities, translational velocities

# Set state
engine.set_state(q_desired, v_desired)
# Internally calls model.realizeVelocity(state) after setting
```

---

## 10.3 Simulation and Integration

### 10.3.1 Time Stepping

OpenSim uses a variable-step integrator (Runge-Kutta-Merson by default):

```python
engine.step(dt=0.01)
# Internally:
#   manager.setInitialTime(current_time)
#   manager.setFinalTime(current_time + dt)
#   manager.integrate(current_time + dt)
```

### 10.3.2 Forward Computation

```python
engine.forward()
# Calls: model.realizeDynamics(state)
# Updates all derived quantities (forces, accelerations) without advancing time
```

### 10.3.3 Reset

```python
engine.reset()
# Re-initializes state to defaults
# Equilibrates muscles (finds static equilibrium of muscle states)
# Resets integration manager
```

The muscle equilibration step is critical: `model.equilibrateMuscles(state)` solves
for the fiber lengths that satisfy the force-length-velocity equilibrium at the current
activation levels. Without this, initial muscle transients can destabilize the simulation.

---

## 10.4 Dynamics Computations

### 10.4.1 Mass Matrix

OpenSim computes the mass matrix via the Simbody `MatterSubsystem`:

```python
M = engine.compute_mass_matrix()
# Internally:
#   matter = model.getMatterSubsystem()
#   model.realizePosition(state)
#   matter.calcM(state, m_mat)
#   # Convert opensim.Matrix to numpy
```

### 10.4.2 Bias Forces

```python
bias = engine.compute_bias_forces()
# Uses inverse dynamics with zero acceleration:
# bias = ID(q, v, 0) = C(q,v)*v + g(q)
```

### 10.4.3 Gravity Forces

OpenSim isolates gravity by temporarily zeroing velocities:

```python
g = engine.compute_gravity_forces()
# Temporarily sets v=0, computes bias (which becomes pure gravity),
# then restores original velocities
```

### 10.4.4 Inverse Dynamics

```python
qacc = np.array([...])  # Desired accelerations
tau = engine.compute_inverse_dynamics(qacc)
# Uses: opensim.InverseDynamicsSolver(model).solve(state, udot)
```

### 10.4.5 Jacobian Computation

OpenSim computes Jacobians via finite differences, since the Simbody API does not expose
direct Jacobian computation as easily as Pinocchio:

```python
jac = engine.compute_jacobian("hand_r")
# Returns dict with 'linear', 'angular', 'spatial'
```

The implementation uses a perturbation $\epsilon = \sqrt{\epsilon_{mach}} \approx 1.49 \times 10^{-8}$
(optimal for first-order finite differences balancing truncation and round-off errors,
see Nocedal & Wright, _Numerical Optimization_, Chapter 8):

$$J_{ij} \approx \frac{p(q + \epsilon\,e_j) - p(q)}{\epsilon}$$

For the angular Jacobian, the rotation difference is computed via the Rodrigues formula:

$$\theta = \arccos\left(\frac{\text{tr}(R_0^T R_1) - 1}{2}\right), \quad \hat{n} = \frac{1}{2\sin\theta}\begin{pmatrix} R_{32} - R_{23} \\ R_{13} - R_{31} \\ R_{21} - R_{12} \end{pmatrix}$$

---

## 10.5 Drift-Control Decomposition (Section F)

### 10.5.1 Drift Acceleration

```python
a_drift = engine.compute_drift_acceleration()
# Drift = -M^{-1} * bias
# (bias includes Coriolis + gravity with zero muscle forces)
```

### 10.5.2 Control Acceleration

```python
a_control = engine.compute_control_acceleration(tau)
# Control = M^{-1} * tau
```

---

## 10.6 Counterfactual Experiments (Section G)

### 10.6.1 ZTCF -- Zero-Torque Counterfactual

```python
a_ztcf = engine.compute_ztcf(q, v)
```

Implementation:

1. Save current state and controls
2. Set desired state $(q, v)$
3. Set zero control (all muscle activations = 0)
4. Realize dynamics and extract accelerations from `state.getUDot()`
5. Restore original state and controls

### 10.6.2 ZVCF -- Zero-Velocity Counterfactual

```python
a_zvcf = engine.compute_zvcf(q)
```

Implementation:

1. Save current state
2. Set state with desired $q$ and $v = 0$
3. Controls are preserved
4. Realize dynamics and extract accelerations
5. Restore original state

---

## 10.7 Muscle Model Integration (Section J)

### 10.7.1 OpenSimMuscleAnalyzer

The `OpenSimMuscleAnalyzer` class provides comprehensive muscle analysis using OpenSim's
Hill-type muscle models.

```python
analyzer = engine.get_muscle_analyzer()
# Returns OpenSimMuscleAnalyzer(model, state) or None
```

**Muscle forces** (Hill-type model output):

$$F_m = a \cdot F_{max} \cdot f_L(l_m) \cdot f_V(v_m) + F_{PE}(l_m)$$

```python
forces = analyzer.get_muscle_forces()
# Returns: {"biceps_long": 245.3, "triceps": 120.1, ...}
# Units: Newtons [N]
```

**Activation levels**:

```python
activations = analyzer.get_activation_levels()
# Returns: {"biceps_long": 0.45, "triceps": 0.12, ...}
# Range: [0, 1]

# Set activations
analyzer.set_activation_levels({"biceps_long": 0.8, "triceps": 0.2})
```

**Moment arms** -- the perpendicular distance from the muscle line of action to the
joint center of rotation:

$$r_i = \frac{\partial L_m}{\partial q_i}$$

where $L_m$ is the musculotendon length and $q_i$ is the generalized coordinate.

```python
moment_arms = analyzer.get_moment_arms()
# Returns nested dict: {"biceps_long": {"elbow_flexion": 0.035, ...}, ...}
# Units: meters [m]

# For a specific coordinate only:
moment_arms_elbow = analyzer.get_moment_arms(coordinate_name="elbow_flexion")
```

**Muscle-generated joint torques**:

$$\tau_j^{(m)} = F_m \cdot r_{mj}$$

```python
muscle_torques = analyzer.compute_muscle_joint_torques()
# Returns: {"biceps_long": np.array([0, 0, 8.6, 0, ...]), ...}
# Units: N*m
```

**Muscle-induced accelerations**:

$$\ddot{q}^{(m)} = M(q)^{-1} \, \tau^{(m)}$$

```python
induced = analyzer.compute_muscle_induced_accelerations()
# Returns: {"biceps_long": np.array([...]), ...}
# Units: rad/s^2
```

Also available directly on the engine:

```python
induced = engine.compute_muscle_induced_accelerations()
```

### 10.7.2 Comprehensive Analysis

```python
analysis = engine.analyze_muscle_contributions()
# Returns MuscleAnalysis dataclass:
#   analysis.muscle_forces       -- dict[str, float]
#   analysis.moment_arms         -- dict[str, dict[str, float]]
#   analysis.activation_levels   -- dict[str, float]
#   analysis.muscle_lengths      -- dict[str, float]
#   analysis.total_muscle_torque -- np.ndarray
```

---

## 10.8 Grip Modeling (Section J1)

The `OpenSimGripModel` class models the hand-club interface using wrapping geometry:

```python
grip_model = engine.create_grip_model()
# Returns OpenSimGripModel(model) or None
```

### 10.8.1 Cylindrical Wrapping

Muscles route around cylindrical wrapping surfaces representing the grip:

```python
grip_model.add_cylindrical_wrap(
    muscle_name="flexor_digitorum",
    grip_body_name="club_grip",
    radius=0.015,    # Shaft radius + hand thickness [m]
    length=0.25,     # Grip length [m]
    location=(0.0, 0.0, 0.0),  # In grip body frame [m]
)
```

### 10.8.2 Grip Force Analysis

```python
grip_analysis = grip_model.analyze_grip_forces(state, analyzer)
# Returns dict:
#   "total_grip_force_N": 125.0,
#   "n_grip_muscles": 8,
#   "grip_muscles": ["flexor_digitorum", ...],
#   "within_physiological_range": True  # 50-200 N per hand
```

Validation range per hand: $50 \leq F_{grip} \leq 200$ N (constants
`MIN_PHYSIOLOGICAL_GRIP_N` and `MAX_PHYSIOLOGICAL_GRIP_N`).

---

## 10.9 Musculoskeletal Model Validation Workflows

### 10.9.1 Validation Against Clinical Data

The recommended validation workflow:

1. **Load a standard OpenSim model** (e.g., `gait2392.osim`, `arm26.osim`)
2. **Apply experimental motion data** (`.mot` files for coordinates, `.sto` for muscle data)
3. **Run inverse dynamics** to compute required joint torques
4. **Compare** joint torques with published electromyography (EMG) data
5. **Verify** muscle activation patterns match known biomechanical literature

```python
# Example validation pipeline
engine = OpenSimPhysicsEngine()
engine.load_from_path("opensim-models/Arm26/arm26.osim")

# Set a known configuration from experimental data
q_experimental = np.array([0.5, 0.8, ...])  # From motion capture
v_experimental = np.array([1.2, -0.5, ...])
engine.set_state(q_experimental, v_experimental)

# Compute inverse dynamics
a_measured = np.array([...])  # From marker-based acceleration
tau = engine.compute_inverse_dynamics(a_measured)

# Compare with EMG-predicted torques
analyzer = engine.get_muscle_analyzer()
muscle_torques = analyzer.compute_muscle_joint_torques()
total_muscle_tau = sum(muscle_torques.values())

# Check consistency
residual = np.linalg.norm(tau - total_muscle_tau)
print(f"Residual torque: {residual:.2f} N*m")
```

### 10.9.2 Force-Length-Velocity Validation

The Hill-type muscle model produces characteristic curves that should match published data:

**Force-Length**: The active force-length relationship is bell-shaped, centered at the
optimal fiber length $l_{opt}$:

$$f_L(l) = e^{-\left(\frac{l - l_{opt}}{\sigma}\right)^2}$$

**Force-Velocity**: The force-velocity relationship follows the Hill equation:

$$f_V(v) = \begin{cases} \frac{v_{max} - v}{v_{max} + K\,v} & v \leq 0 \text{ (shortening)} \\ N + (N-1)\frac{v_{max} + v}{7.56\,K\,v - v_{max}} & v > 0 \text{ (lengthening)} \end{cases}$$

### 10.9.3 OpenSim File Formats

| Extension | Description            | Usage                                          |
| --------- | ---------------------- | ---------------------------------------------- |
| `.osim`   | Model definition (XML) | Musculoskeletal geometry, muscles, joints      |
| `.mot`    | Motion file            | Joint coordinates over time                    |
| `.sto`    | Storage file           | General time-series data (forces, activations) |
| `.trc`    | Marker trajectory      | 3D marker positions from motion capture        |
| `.xml`    | Setup files            | Tool configuration (IK, ID, SO settings)       |

---

## 10.10 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.opensim.python.opensim_physics_engine import (
    OpenSimPhysicsEngine,
)

# 1. Load model
engine = OpenSimPhysicsEngine()
engine.load_from_path("path/to/golf_swing_model.osim")
engine.reset()

# 2. Set controls (muscle excitations)
n_controls = engine._model.getNumControls()
excitations = np.full(n_controls, 0.1)  # Low baseline activation
engine.set_control(excitations)

# 3. Simulate
dt = 0.01
for i in range(200):  # 2 seconds
    engine.step(dt)
    q, v = engine.get_state()
    t = engine.get_time()

    # Every 0.5s, print state
    if i % 50 == 0:
        M = engine.compute_mass_matrix()
        print(f"t={t:.2f}s, |q|={np.linalg.norm(q):.3f}, cond(M)={np.linalg.cond(M):.1f}")

# 4. Muscle analysis at final configuration
analysis = engine.analyze_muscle_contributions()
if analysis:
    print(f"Number of muscles analyzed: {len(analysis.muscle_forces)}")
    print(f"Total muscle torque norm: {np.linalg.norm(analysis.total_muscle_torque):.2f} N*m")

    # Top 5 muscles by force
    sorted_muscles = sorted(analysis.muscle_forces.items(), key=lambda x: x[1], reverse=True)
    for name, force in sorted_muscles[:5]:
        print(f"  {name}: {force:.1f} N (activation: {analysis.activation_levels[name]:.2f})")

# 5. Counterfactual analysis
a_ztcf = engine.compute_ztcf(q, v)
a_zvcf = engine.compute_zvcf(q)
print(f"ZTCF acceleration norm: {np.linalg.norm(a_ztcf):.3f} rad/s^2")
print(f"ZVCF acceleration norm: {np.linalg.norm(a_zvcf):.3f} rad/s^2")
```

---
