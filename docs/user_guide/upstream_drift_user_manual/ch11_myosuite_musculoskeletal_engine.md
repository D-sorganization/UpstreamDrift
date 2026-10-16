# Chapter 11: MyoSuite Musculoskeletal Engine

## 11.1 Overview

The MyoSuite engine wraps the MyoSuite framework (Caggiano et al., 2022) to provide
MuJoCo-based musculoskeletal simulation within the UpstreamDrift `PhysicsEngine` protocol.
MyoSuite combines anatomically detailed MuJoCo muscle models with OpenAI Gym-compatible
reinforcement learning interfaces, making it ideal for:

1. **Muscle-driven simulation** with realistic Hill-type muscle dynamics
2. **Neural control policy training** via reinforcement learning (RL)
3. **Cross-validation** against OpenSim for biomechanical fidelity

### Source Files

| File                                                                     | Purpose                                    |
| ------------------------------------------------------------------------ | ------------------------------------------ |
| `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py` | `PhysicsEngine` protocol wrapper           |
| `src/engines/physics_engines/myosuite/python/muscle_analysis.py`         | Muscle analysis, grip modeling             |
| `src/shared/python/myosuite_adapter.py`                                  | RL environment adapter (`MuscleDrivenEnv`) |

### Dependencies

```
pip install myosuite     # MyoSuite (includes MuJoCo)
pip install gym          # OpenAI Gym
pip install mujoco       # MuJoCo Python bindings
pip install stable-baselines3  # Optional: for RL policy training
```

---

## 11.2 Model Configurations

MyoSuite provides pre-built musculoskeletal models of varying complexity. The models are
MuJoCo XML files with Hill-type muscle actuators:

### 11.2.1 MyoUpperBody

| Property               | Value                             |
| ---------------------- | --------------------------------- |
| Degrees of Freedom     | 19 DOF                            |
| Actuators              | 20 muscle actuators               |
| Coverage               | Bilateral arms + torso            |
| Typical Environment ID | `myoArmPose-v0`, `myoArmReach-v0` |

Suitable for: upper body swing mechanics, arm coordination analysis, shoulder-elbow coupling.

### 11.2.2 MyoBody Full

| Property           | Value                           |
| ------------------ | ------------------------------- |
| Degrees of Freedom | 52 DOF                          |
| Muscles            | 290 muscles                     |
| Coverage           | Complete body (head to feet)    |
| Typical Usage      | Full-body golf swing simulation |

This is the most anatomically complete model. Due to the high muscle count, simulation
is computationally intensive. Recommended for validation studies rather than real-time
applications.

### 11.2.3 MyoArm Simple

| Property               | Value                       |
| ---------------------- | --------------------------- |
| Degrees of Freedom     | 14 DOF                      |
| Actuators              | 14 actuators                |
| Coverage               | Simplified bilateral arms   |
| Typical Environment ID | `myoElbowPose1D6MRandom-v0` |

Suitable for: rapid prototyping, RL training, elbow/wrist analysis, grip studies.

### 11.2.4 Available MyoSuite Environments

| Category | Environment ID                 | Muscles    | Description        |
| -------- | ------------------------------ | ---------- | ------------------ |
| Elbow    | `myoElbowPose1D6MRandom-v0`    | 6          | Elbow pose control |
| Elbow    | `myoElbowPose1D6MExoRandom-v0` | 6          | With exoskeleton   |
| Hand     | `myoHandPose100Random-v0`      | 39         | Hand pose control  |
| Hand     | `myoHandReach-v0`              | 39         | Reaching task      |
| Hand     | `myoHandKey-v0`                | 39         | Key turning        |
| Arm      | `myoArmPose-v0`                | Multiple   | Arm pose control   |
| Leg      | `myoLegWalk-v0`                | Lower body | Walking            |
| Leg      | `myoLegRun-v0`                 | Lower body | Running            |

---

## 11.3 Engine Initialization

### 11.3.1 Loading Environments

Unlike other engines that load model files, MyoSuite loads Gym environment IDs:

```python
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)

engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")
# Internally: gym.make(env_id) + env.reset()

print(f"Model: {engine.model_name}")        # "myoElbowPose1D6MRandom-v0"
print(f"Timestep: {engine._dt:.4f}s")       # From sim.model.opt.timestep
print(f"Initialized: {engine.is_initialized}")  # True
```

The wrapper accesses the underlying MuJoCo simulation:

```python
# Direct MuJoCo model access
mj_model = engine.model   # MuJoCo MjModel object (or None)
mj_data = engine.sim.data  # MuJoCo MjData object
```

### 11.3.2 State Management

```python
# Get state (qpos, qvel from underlying MuJoCo sim)
q, v = engine.get_state()

# Set state
engine.set_state(q_desired, v_desired)
# Calls sim.forward() to update derived quantities

# Set control (muscle activations via ctrl buffer)
u = np.array([0.5, 0.3, 0.1, ...])
engine.set_control(u)
```

### 11.3.3 Simulation Stepping

```python
engine.step(dt=0.002)
# If dt is provided, temporarily overrides sim.model.opt.timestep
# Uses sim.step() directly (bypasses Gym wrapper for low-level control)
```

---

## 11.4 Hill-Type Muscle Model

MyoSuite implements the Hill-type muscle model via MuJoCo's built-in muscle actuator dynamics.
The total muscle force is:

$$F_m = a \cdot F_{max} \cdot f_L(l_m) \cdot f_V(v_m) + F_{PE}(l_m)$$

where:

- $a \in [0, 1]$ is the muscle activation level
- $F_{max}$ is the maximum isometric force [N]
- $f_L(l_m)$ is the normalized force-length relationship
- $f_V(v_m)$ is the normalized force-velocity relationship
- $F_{PE}(l_m)$ is the passive elastic element force

### 11.4.1 Force-Length Relationship

The active force-length curve is a Gaussian centered at the optimal fiber length $l_{opt}$:

$$f_L(l) = e^{-\left(\frac{l - l_{opt}}{\sigma}\right)^2}$$

where $\sigma$ controls the width of the bell curve. At $l = l_{opt}$, the muscle can
produce maximum force ($f_L = 1$). As the muscle shortens or lengthens beyond this optimum,
force capacity decreases.

### 11.4.2 Force-Velocity Relationship

The force-velocity relationship distinguishes shortening (concentric) from lengthening
(eccentric) contractions:

$$f_V(v) = \begin{cases} \dfrac{v_{max} - v}{v_{max} + K\,v} & v \leq 0 \text{ (shortening)} \\[10pt] N + (N-1)\dfrac{v_{max} + v}{7.56\,K\,v - v_{max}} & v > 0 \text{ (lengthening)} \end{cases}$$

where:

- $v_{max}$ is the maximum shortening velocity (typically $10 \cdot l_{opt}$ per second)
- $K$ is the curvature parameter (typically 0.25)
- $N$ is the eccentric force enhancement factor (typically 1.5)
- $v < 0$ denotes shortening; $v > 0$ denotes lengthening

Key properties:

- At $v = 0$ (isometric): $f_V = 1$
- At $v = -v_{max}$ (maximum shortening): $f_V = 0$ (zero force)
- During eccentric contraction: $f_V > 1$ (force exceeds isometric maximum)

### 11.4.3 Parallel Elastic Element

The passive elastic element produces force only when the muscle is stretched beyond its
slack length $l_{slack}$:

$$F_{PE}(l) = k_{PE} \cdot \max(0,\; l - l_{slack})^2$$

where $k_{PE}$ is the passive element stiffness. This models the connective tissue
(tendons, fascia) that resists excessive stretch.

### 11.4.4 Activation Dynamics

Muscle activation follows first-order dynamics governed by the neural excitation signal $u(t)$:

$$\dot{a}(t) = \frac{u(t) - a(t)}{\tau(u, a)}$$

where the time constant $\tau$ depends on the direction of change:

$$\tau = \begin{cases} \tau_{act} & \text{if } u > a \text{ (activation, fast: 10-50 ms)} \\ \tau_{deact} & \text{if } u \leq a \text{ (deactivation, slow: 40-110 ms)} \end{cases}$$

The asymmetry between activation and deactivation time constants reflects the physiology:
calcium release (activation) is faster than calcium reuptake (deactivation).

**Usage via the RL adapter**:

```python
from src.shared.python.myosuite_adapter import MuscleDrivenEnv

env = MuscleDrivenEnv(muscle_system, task="tracking", dt=0.001)
obs = env.reset()

# Neural excitation -> activation dynamics -> muscle force
action = np.array([0.8, 0.2, 0.5])  # Excitations [0,1]
obs, reward, done, info = env.step(action)
# info["activations"] shows the filtered activation levels
```

---

## 11.5 Muscle Analysis Module

The `MyoSuiteMuscleAnalyzer` class provides detailed muscle analysis via the underlying
MuJoCo simulation.

### 11.5.1 Initialization

```python
analyzer = engine.get_muscle_analyzer()
# Returns MyoSuiteMuscleAnalyzer(sim) or None

print(f"Muscles found: {len(analyzer.muscle_names)}")
print(f"Muscle names: {analyzer.muscle_names}")
```

Muscle identification: The analyzer identifies muscle actuators by checking
`model.actuator_dyntype[i] == 2` (MuJoCo's `mjDYN_MUSCLE` constant). If the dyntype
attribute is unavailable, all actuators are treated as muscles (appropriate for MyoSuite
models).

### 11.5.2 Muscle State Queries

```python
# Current activation levels [0, 1]
activations = analyzer.get_muscle_activations()
# Sources: data.act[] if available, falls back to data.ctrl[]

# Current muscle forces [N]
forces = analyzer.get_muscle_forces()
# Sources: data.actuator_force[]

# Current muscle lengths [m]
lengths = analyzer.get_muscle_lengths()
# Sources: data.actuator_length[]

# Current contraction velocities [m/s] (negative = shortening)
velocities = analyzer.get_muscle_velocities()
# Sources: data.actuator_velocity[]
```

All queries return arrays indexed by muscle order in `analyzer.muscle_names`.

### 11.5.3 Setting Muscle Activations by Name

```python
engine.set_muscle_activations({
    "BIClong": 0.7,
    "BICshort": 0.5,
    "TRIlong": 0.1,
    "TRIlat": 0.1,
})
# Internally maps names to actuator indices, clips to [0, 1]
```

### 11.5.4 Complete Muscle State Snapshot

```python
state = engine.get_muscle_state()
# Returns MyoSuiteMuscleState dataclass:
#   state.muscle_names   -- list[str]
#   state.activations    -- np.ndarray, shape (n_muscles,)
#   state.forces         -- np.ndarray, shape (n_muscles,)
#   state.lengths        -- np.ndarray, shape (n_muscles,)
#   state.velocities     -- np.ndarray, shape (n_muscles,)
```

---

## 11.6 Moment Arm Analysis

Moment arms $r$ quantify the mechanical advantage of each muscle about each joint.
The MyoSuite analyzer computes moment arms via finite differences:

$$r_{mj} = -\frac{\partial L_m}{\partial q_j} \approx -\frac{L_m(q + \delta e_j) - L_m(q)}{\delta}$$

where $\delta = 10^{-6}$ rad (or m) is the perturbation size.

The negative sign arises because muscles typically shorten (decrease $L_m$) when the joint
flexes (increases $q_j$).

```python
moment_arms = analyzer.compute_moment_arms()
# Returns: {"BIClong": np.array([0.0, 0.035, 0.0, ...]), ...}
# Units: meters [m]
# Shape of each array: (nv,) -- one value per DOF
```

---

## 11.7 Muscle-Generated Joint Torques

Each muscle generates a torque about each joint equal to the product of its force and
moment arm:

$$\tau_j^{(m)} = F_m \cdot r_{mj}$$

```python
muscle_torques = analyzer.compute_muscle_joint_torques()
# Returns: {"BIClong": np.array([0, 8.6, 0, ...]), ...}

# Total muscle torque across all muscles
total_tau = np.zeros(engine.sim.model.nv)
for tau in muscle_torques.values():
    total_tau += tau
```

---

## 11.8 Muscle-Induced Accelerations

Each muscle's contribution to joint acceleration is:

$$\ddot{q}^{(m)} = M(q)^{-1} \, \tau^{(m)}$$

```python
induced = analyzer.compute_muscle_induced_accelerations()
# Returns: {"BIClong": np.array([...]), ...}
# Units: rad/s^2

# Verify: sum should approximate total acceleration from muscles
total_induced = sum(induced.values())
```

Also accessible directly via the engine:

```python
induced = engine.compute_muscle_induced_accelerations()
```

---

## 11.9 Muscle Co-Contraction Analysis

Co-contraction occurs when antagonist muscles are simultaneously active, increasing joint
stiffness at the expense of metabolic efficiency. The co-contraction index for an
agonist-antagonist pair is:

$$CI = \frac{2 \cdot \min(a_{agonist}, a_{antagonist})}{a_{agonist} + a_{antagonist}}$$

where $CI = 0$ means pure agonist activation and $CI = 1$ means equal co-contraction.

```python
activations = analyzer.get_muscle_activations()
muscle_names = analyzer.muscle_names

# Example: biceps vs triceps co-contraction
bic_idx = muscle_names.index("BIClong")
tri_idx = muscle_names.index("TRIlong")

a_bic = activations[bic_idx]
a_tri = activations[tri_idx]

if (a_bic + a_tri) > 0:
    ci = 2.0 * min(a_bic, a_tri) / (a_bic + a_tri)
else:
    ci = 0.0

print(f"Co-contraction index: {ci:.3f}")
```

---

## 11.10 Power Analysis

Joint power quantifies the rate of energy generation or absorption at each joint:

$$P = \tau \cdot \dot{q}$$

where $P > 0$ indicates power generation (concentric) and $P < 0$ indicates power
absorption (eccentric).

Total work over an interval $[0, T]$:

$$W = \int_0^T P \, dt = \int_0^T \tau(t) \cdot \dot{q}(t) \, dt$$

The metabolic activation cost accounts for the energy spent maintaining muscle activation
beyond mechanical work:

$$P_{metabolic} = F \cdot v + \alpha \cdot a^2 \cdot |F \cdot v|$$

where $\alpha = 0.25$ is the metabolic cost constant.

```python
activation_power = analyzer.compute_activation_power()
# Returns: {"BIClong": -1.23, "TRIlong": 0.45, ...}
# Units: Watts [W]
# Includes mechanical power + metabolic activation cost
```

---

## 11.11 Muscle Recruitment Optimization

The standard muscle recruitment problem minimizes the sum of squared activations subject
to producing the required joint torques:

$$\min_{a} \sum_{i=1}^{n_m} a_i^2$$
$$\text{subject to:} \quad \tau = R \cdot F(a)$$

where $R$ is the moment arm matrix and $F(a)$ is the vector of muscle forces as a function
of activations. Additional constraints include:

- $0 \leq a_i \leq 1$ (activation bounds)
- $F_i = a_i \cdot F_{max,i} \cdot f_L(l_i) \cdot f_V(v_i)$ (Hill model)

This is a nonlinear optimization problem because $F$ depends nonlinearly on $a$ through
the force-length and force-velocity relationships. In practice, for a given configuration
and velocity, $f_L$ and $f_V$ are constants, and the problem reduces to a quadratic program.

---

## 11.12 Comprehensive Analysis

```python
analysis = analyzer.analyze_all()
# Returns MyoSuiteMuscleAnalysis dataclass:
#   analysis.muscle_state          -- MyoSuiteMuscleState
#   analysis.moment_arms           -- dict[str, np.ndarray]
#   analysis.joint_torques         -- dict[str, np.ndarray]
#   analysis.induced_accelerations -- dict[str, np.ndarray]
#   analysis.total_muscle_torque   -- np.ndarray
#   analysis.activation_power      -- dict[str, float]
```

---

## 11.13 Grip Modeling (Section K1)

The `MyoSuiteGripModel` provides activation-driven grip force analysis:

```python
grip_model = engine.create_grip_model()
# Returns MyoSuiteGripModel(sim, analyzer) or None
```

### 11.13.1 Grip Muscle Identification

```python
grip_muscles = grip_model.get_grip_muscles()
# Filters by keywords: flexor, extensor, grip, hand, finger, thumb, etc.
```

### 11.13.2 Grip Force Computation

```python
total_grip = grip_model.compute_total_grip_force()
# Sum of forces from grip-related muscles [N]
```

### 11.13.3 Comprehensive Grip Analysis

```python
grip_info = grip_model.analyze_grip()
# Returns dict:
#   "total_grip_force_N": 350.0,
#   "mean_grip_activation": 0.65,
#   "n_grip_muscles": 12,
#   "grip_muscles": ["flexor_digitorum", ...],
#   "within_mvc_range": True,     # 200-800 N
#   "individual_forces": {...},
#   "individual_activations": {...}
```

Validation range (MVC): $200 \leq F_{grip} \leq 800$ N (constants `MIN_MVC_GRIP_N`
and `MAX_MVC_GRIP_N`).

---

## 11.14 Drift-Control Decomposition and Counterfactuals

### 11.14.1 Drift Acceleration

```python
a_drift = engine.compute_drift_acceleration()
# Sets all muscle activations to zero, computes forward dynamics,
# extracts qacc, then restores original activations
```

### 11.14.2 Control Acceleration

```python
a_control = engine.compute_control_acceleration(tau)
# M^{-1} * tau using MuJoCo mass matrix
```

### 11.14.3 ZTCF

```python
a_ztcf = engine.compute_ztcf(q, v)
# Saves state/ctrl, sets state, zeros ctrl, sim.forward(), reads qacc, restores
```

### 11.14.4 ZVCF

```python
a_zvcf = engine.compute_zvcf(q)
# Saves state, sets q with v=0, sim.forward(), reads qacc, restores
```

---

## 11.15 RL Adapter: MuscleDrivenEnv

The `MuscleDrivenEnv` class bridges the UpstreamDrift muscle models with standard RL
frameworks (Stable-Baselines3, RLlib):

```python
from src.shared.python.myosuite_adapter import MuscleDrivenEnv, train_muscle_policy

# Create environment
env = MuscleDrivenEnv(muscle_system, task="tracking", dt=0.001)

# Observation space: [q, v, a_1, ..., a_n]
obs = env.reset()

# Action space: neural excitations [0, 1] per muscle
action = np.array([0.5, 0.3, 0.1])
obs, reward, done, info = env.step(action)

# Train a policy
policy = train_muscle_policy(env, total_timesteps=100000)
# Uses SAC (Soft Actor-Critic) from stable-baselines3
```

### Reward Functions

| Task       | Reward | Description         |
| ---------- | ------ | ------------------- | --------------- | ------------------------- |
| `tracking` | $-     | q - q\_{target}     | $               | Track time-varying target |
| `reach`    | $-     | q - q\_{target}     | - 0.01 \cdot t$ | Reach target quickly      |
| `swing`    | Custom | Golf swing specific |

---

## 11.16 Cross-Validation With OpenSim

```python
from src.shared.python.cross_engine_validator import CrossEngineValidator

myosuite_engine = MyoSuitePhysicsEngine()
myosuite_engine.load_from_path("myoElbowPose1D6MRandom-v0")

opensim_engine = OpenSimPhysicsEngine()
opensim_engine.load_from_path("arm26.osim")

validator = CrossEngineValidator()
result = validator.compare_states(
    "MyoSuite", myosuite_tau,
    "OpenSim", opensim_tau,
    metric="torque"
)
print(f"Passed: {result.passed}, Severity: {result.severity}")
```

---

## 11.17 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)

# 1. Initialize
engine = MyoSuitePhysicsEngine()
engine.load_from_path("myoElbowPose1D6MRandom-v0")
engine.reset()

# 2. Get muscle information
muscle_names = engine.get_muscle_names()
print(f"Muscles: {muscle_names}")

# 3. Simulate with varying activations
trajectory_q = []
trajectory_forces = []

for i in range(1000):
    # Sinusoidal biceps activation
    activations = {
        "BIClong": 0.5 * (1 + np.sin(i * 0.01)),
        "BICshort": 0.4 * (1 + np.sin(i * 0.01)),
        "TRIlong": 0.1,
        "TRIlat": 0.1,
    }
    engine.set_muscle_activations(activations)
    engine.step(dt=0.002)

    q, v = engine.get_state()
    trajectory_q.append(q.copy())

    # Periodic muscle analysis
    if i % 100 == 0:
        state = engine.get_muscle_state()
        if state:
            trajectory_forces.append(state.forces.copy())
            print(f"t={engine.get_time():.3f}s, "
                  f"elbow={np.degrees(q[0]):.1f} deg, "
                  f"peak_force={np.max(state.forces):.1f} N")

# 4. Full analysis at final state
analysis = engine.analyze_muscle_contributions()
if analysis:
    print(f"\nFinal analysis:")
    print(f"  Total muscle torque: {np.linalg.norm(analysis.total_muscle_torque):.2f} N*m")

    for name, power in analysis.activation_power.items():
        print(f"  {name}: power={power:.2f} W")

# 5. Drift-control decomposition
a_drift = engine.compute_drift_acceleration()
print(f"\nDrift acceleration norm: {np.linalg.norm(a_drift):.3f} rad/s^2")
```

---
