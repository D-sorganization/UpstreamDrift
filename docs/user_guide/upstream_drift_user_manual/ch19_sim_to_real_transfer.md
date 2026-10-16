# Chapter 19: Sim-to-Real Transfer

The `learning/sim2real/` module provides tools for bridging the gap between
simulated training environments and real robot hardware. It includes domain
randomization for training robust policies and system identification for
calibrating simulation parameters.

Source files referenced in this chapter:

- `src/learning/sim2real/domain_randomization.py`
- `src/learning/sim2real/system_identification.py`

---

## 19.1 Domain Randomization

Domain randomization is a technique that trains policies in simulation with
randomly varied physical parameters. The intuition is that if the policy
performs well across a wide distribution of simulated dynamics, the real-world
dynamics are likely to fall within that distribution, enabling zero-shot
transfer.

### 19.1.1 DomainRandomizationConfig

The `DomainRandomizationConfig` dataclass defines the randomization ranges for
each physical parameter. Ranges are specified as `(min, max)` tuples,
interpreted as multipliers relative to nominal values for most parameters.

**Dynamics parameters:**

| Field                  | Default         | Description                    |
| ---------------------- | --------------- | ------------------------------ |
| `mass_range`           | `(0.8, 1.2)`    | Link mass scale factor.        |
| `friction_range`       | `(0.5, 1.5)`    | Friction coefficient scale.    |
| `damping_range`        | `(0.8, 1.2)`    | Joint damping scale.           |
| `motor_strength_range` | `(0.9, 1.1)`    | Motor strength scale.          |
| `inertia_range`        | `(0.9, 1.1)`    | Link inertia scale.            |
| `center_of_mass_range` | `(-0.01, 0.01)` | CoM position offset in meters. |

**Delays and noise:**

| Field                     | Default  | Description                     |
| ------------------------- | -------- | ------------------------------- |
| `action_delay_range`      | `(0, 3)` | Action delay in timesteps.      |
| `observation_delay_range` | `(0, 2)` | Observation delay in timesteps. |
| `observation_noise_std`   | `0.01`   | Gaussian noise on observations. |
| `action_noise_std`        | `0.01`   | Gaussian noise on actions.      |

**Environment parameters:**

| Field                  | Default       | Description                 |
| ---------------------- | ------------- | --------------------------- |
| `gravity_range`        | `(9.5, 10.1)` | Gravity magnitude in m/s^2. |
| `floor_friction_range` | `(0.5, 1.5)`  | Floor friction scale.       |

**Actuator parameters:**

| Field                 | Default         | Description                       |
| --------------------- | --------------- | --------------------------------- |
| `joint_offset_range`  | `(-0.02, 0.02)` | Joint position offset in radians. |
| `actuator_gain_range` | `(0.95, 1.05)`  | Actuator gain scale.              |

**Enable flags.** Each category of randomization can be enabled or disabled
independently:

| Flag                 | Default | Controls                      |
| -------------------- | ------- | ----------------------------- |
| `randomize_mass`     | `True`  | Mass randomization.           |
| `randomize_friction` | `True`  | Friction randomization.       |
| `randomize_damping`  | `True`  | Damping randomization.        |
| `randomize_motor`    | `True`  | Motor strength randomization. |
| `randomize_gravity`  | `True`  | Gravity randomization.        |
| `randomize_delays`   | `True`  | Delay randomization.          |
| `randomize_noise`    | `True`  | Noise randomization.          |

### 19.1.2 The DomainRandomizer Class

`DomainRandomizer` applies randomization to a physics engine and manages the
delay and noise pipelines.

**Construction.**

```python
from src.learning.sim2real.domain_randomization import (
    DomainRandomizer,
    DomainRandomizationConfig,
)

config = DomainRandomizationConfig(
    mass_range=(0.7, 1.3),
    friction_range=(0.3, 2.0),
    randomize_delays=True,
)
randomizer = DomainRandomizer(engine, config)
```

During construction, the randomizer:

1. Stores the engine reference.
2. Calls `_store_nominal_params()` to read and cache the engine's current
   parameter values (masses, damping, friction, motor strength, gravity,
   actuator gains). These nominal values serve as the baseline for
   multiplicative randomization.
3. Initializes empty delay buffers and a random number generator.

**The `randomize` method.** This is the main entry point, typically called at
the start of each episode:

```python
randomization_values = randomizer.randomize(seed=42)
```

The method applies the following randomizations in order:

1. **Mass.** A single scale factor is sampled uniformly from `mass_range` and
   applied to all link masses:

$$
m_i^{new} = m_i^{nom} \cdot \text{Uniform}(m_{lo}, m_{hi})
$$

2. **Friction.** A scale factor is sampled and applied to all friction
   coefficients.

3. **Damping.** A scale factor is sampled and applied to all joint damping
   values.

4. **Motor strength.** A scale factor is sampled and applied to all motor
   strength values.

5. **Gravity.** The gravity magnitude is sampled uniformly from `gravity_range`
   and set as the z-component (assuming z-up convention):

$$
g^{new} = [0, 0, -\text{Uniform}(g_{lo}, g_{hi})]
$$

6. **Delays.** Integer delays are sampled for both actions and observations:

$$
k_{action} \sim U\{0, 1, \ldots, k_{max}\}
$$

$$
k_{obs} \sim U\{0, 1, \ldots, k_{max}^{obs}\}
$$

Delay buffers are cleared when new delays are sampled.

The method returns a dictionary recording the sampled values, which is useful
for logging and debugging.

**The `reset_to_nominal` method.** Restores all engine parameters to their
original values and clears delay buffers. Call this before `randomize` to
ensure a clean starting state.

### 19.1.3 Action Delay Buffer

Action delays simulate the communication latency between the control computer
and the robot actuators. The `apply_action_with_delay` method implements a
queue-based delay:

$$
a_{applied}(t) = a_{commanded}(t - k)
$$

where $k \sim U[0, k_{max}]$ is the sampled delay in timesteps.

The implementation maintains a FIFO buffer. At each timestep:

1. The current action is appended to the buffer.
2. If the buffer has accumulated more entries than the delay, the oldest entry
   is popped and returned.
3. During the initial fill period (when the buffer has fewer entries than the
   delay), a zero action is returned.

After the delayed action is retrieved, action noise is applied if enabled:

$$
a_{noisy} = a_{delayed} + \mathcal{N}(0, \sigma_a^2 I)
$$

### 19.1.4 Observation Delay

Observation delays simulate sensor processing latency. The
`get_observation_with_delay` method works analogously to the action delay:

$$
o_{received}(t) = o_{true}(t - k_{obs})
$$

During the initial fill period, the first observation in the buffer is returned
(rather than zeros), since returning a zero observation would be unrealistic.
After retrieval, observation noise is added:

$$
o_{noisy} = o_{delayed} + \mathcal{N}(0, \sigma_o^2 I)
$$

### 19.1.5 Batch Sampling for Parallel Environments

The `sample_randomization_batch` method generates a batch of randomization
configurations for use with vectorized environments:

```python
configs = randomizer.sample_randomization_batch(batch_size=64)
# configs is a list of 64 dictionaries, each with different randomization values
```

For each sample, the method calls `randomize` with a unique seed, records the
configuration, and then calls `reset_to_nominal` to restore the engine state
before sampling the next configuration. This allows parallel environments to
each apply a different configuration from the batch.

### 19.1.6 Integration With RL Training

A typical training loop with domain randomization:

```python
randomizer = DomainRandomizer(engine, config)

for episode in range(n_episodes):
    # Randomize at episode start
    randomizer.randomize()
    obs, info = env.reset()

    done = False
    while not done:
        action = policy(obs)

        # Apply action with delay and noise
        delayed_action = randomizer.apply_action_with_delay(action)
        obs, reward, terminated, truncated, info = env.step(delayed_action)

        # Apply observation delay and noise
        obs = randomizer.get_observation_with_delay(obs)

        done = terminated or truncated

    # Reset to nominal before next randomization
    randomizer.reset_to_nominal()
```

---

## 19.2 System Identification

While domain randomization trains policies that are robust to parameter
uncertainty, system identification takes the complementary approach of
calibrating the simulation to match the real robot as closely as possible.

### 19.2.1 IdentificationResult

The `IdentificationResult` dataclass holds the output of the identification
process:

| Field               | Type             | Description                     |
| ------------------- | ---------------- | ------------------------------- | ---------------------------- |
| `identified_params` | `dict[str, float | NDArray]`                       | Identified parameter values. |
| `residual_error`    | `float`          | Final optimization residual.    |
| `iterations`        | `int`            | Number of iterations performed. |
| `converged`         | `bool`           | Whether optimization converged. |

### 19.2.2 The SystemIdentifier Class

`SystemIdentifier` finds simulation parameters that minimize the discrepancy
between simulated and real trajectories.

**Construction.**

```python
from src.learning.sim2real.system_identification import SystemIdentifier

identifier = SystemIdentifier(
    model=engine,
    param_bounds={
        "mass_scale": (0.5, 2.0),
        "friction_scale": (0.2, 3.0),
        "damping_scale": (0.5, 2.0),
        "motor_scale": (0.5, 1.5),
        "com_offset_x": (-0.05, 0.05),
        "com_offset_y": (-0.05, 0.05),
        "com_offset_z": (-0.05, 0.05),
    },
)
```

If `param_bounds` is not provided, sensible defaults are used. The constructor
also caches the nominal engine parameters.

**Default parameter bounds:**

| Parameter        | Lower | Upper | Description                 |
| ---------------- | ----- | ----- | --------------------------- |
| `mass_scale`     | 0.5   | 2.0   | Overall mass multiplier.    |
| `friction_scale` | 0.2   | 3.0   | Friction coefficient scale. |
| `damping_scale`  | 0.5   | 2.0   | Joint damping scale.        |
| `motor_scale`    | 0.5   | 1.5   | Motor strength scale.       |
| `com_offset_x`   | -0.05 | 0.05  | CoM x-offset in meters.     |
| `com_offset_y`   | -0.05 | 0.05  | CoM y-offset in meters.     |
| `com_offset_z`   | -0.05 | 0.05  | CoM z-offset in meters.     |

### 19.2.3 Identification Objective

The identification problem is formulated as:

$$
\min_\theta \frac{1}{N} \sum_{i=1}^{N} \text{MSE}\bigl(\text{sim}(\theta, s_0^{(i)}, a^{(i)}),\; x_{real}^{(i)}\bigr)
$$

where:

- $\theta$ is the parameter vector being optimized.
- $N$ is the number of real-robot trajectories.
- $\text{sim}(\theta, s_0, a)$ simulates a trajectory from initial state $s_0$
  under action sequence $a$ with parameters $\theta$.
- $x_{real}^{(i)}$ is the recorded real-robot state trajectory.
- MSE is the mean squared error between the simulated and real trajectories.

The MSE is computed over the full state vector $[q, \dot{q}]$ (positions and
velocities concatenated):

$$
\text{MSE} = \frac{1}{T \cdot 2n_q} \sum_{t=0}^{T} \|x_{sim}^{(t)} - x_{real}^{(t)}\|^2
$$

An optional weight vector can emphasize certain state components (e.g.,
weighting positions more than velocities).

### 19.2.4 Optimization Algorithm

The implementation uses coordinate descent with numerical perturbations. This
gradient-free approach is chosen for robustness with non-smooth simulation
dynamics.

**Algorithm:**

```
Initialize: theta = [1, 1, ..., 1] (nominal scales)
best_error = objective(theta)

For iteration = 1 to max_iterations:
    improved = False
    For each parameter i:
        For delta in [+0.1, -0.1, +0.05, -0.05, +0.01, -0.01]:
            theta_test = theta.copy()
            theta_test[i] = clip(theta_test[i] + delta, lower[i], upper[i])
            error = objective(theta_test)
            If error < best_error - tolerance:
                best_error = error
                theta = theta_test
                improved = True
    If not improved:
        converged = True
        Break
```

For each parameter, six perturbation magnitudes are tried:
$\{+0.1, -0.1, +0.05, -0.05, +0.01, -0.01\}$. The coarse perturbations
($\pm 0.1$) allow rapid initial progress, while the fine perturbations
($\pm 0.01$) refine the solution near the optimum.

The algorithm converges when no parameter perturbation improves the objective
by more than `tolerance` (default $10^{-6}$).

### 19.2.5 Trajectory Simulation

The `_simulate_trajectory` method runs a forward simulation from an initial
state under a given action sequence:

1. Set initial joint positions $q_0$ and velocities $v_0$ in the engine.
2. For each timestep:
   a. Apply the torque action to the engine.
   b. Step the simulation by $\Delta t$.
   c. Record the resulting state $[q, \dot{q}]$.
3. Return the full state trajectory as a `(T+1, 2n_q)` array.

The timestep $\Delta t$ is computed from the demonstration timestamps:

$$
\Delta t = \frac{1}{T-1} \sum_{t=1}^{T-1} (t_{k+1} - t_k)
$$

### 19.2.6 Reality Gap Metrics

The `compute_reality_gap` method provides a comprehensive set of metrics
quantifying the simulation-to-reality discrepancy:

| Metric                   | Formula                                                |
| ------------------------ | ------------------------------------------------------ | ---------------------------------------------- | --- |
| `total_mse`              | $\frac{1}{T \cdot 2n_q} \sum \|x_{sim} - x_{real}\|^2$ |
| `position_mse`           | MSE over position components only.                     |
| `velocity_mse`           | MSE over velocity components only.                     |
| `max_position_error`     | $\max\_{t,j}                                           | q*{sim}^{(t,j)} - q*{real}^{(t,j)}             | $   |
| `max_velocity_error`     | $\max\_{t,j}                                           | \dot{q}_{sim}^{(t,j)} - \dot{q}_{real}^{(t,j)} | $   |
| `mean_position_error`    | Mean absolute position error.                          |
| `mean_velocity_error`    | Mean absolute velocity error.                          |
| `trajectory_length`      | Number of timesteps compared.                          |
| `joint_{j}_position_mse` | Per-joint position MSE.                                |
| `joint_{j}_velocity_mse` | Per-joint velocity MSE.                                |

### 19.2.7 Validation

The `validate_identification` method evaluates the identified parameters on
held-out test trajectories. This is essential for checking that the identified
parameters generalize and are not overfit to the training trajectories.

```python
# Split trajectories
train_trajs = trajectories[:8]
test_trajs = trajectories[8:]

# Identify
result = identifier.identify_from_trajectories(train_trajs)

# Validate
val_metrics = identifier.validate_identification(
    test_trajectories=test_trajs,
    identified_params=result.identified_params,
)
print(f"Validation error: {val_metrics['mean_error']:.6f}")
print(f"Std error: {val_metrics['std_error']:.6f}")
print(f"Max error: {val_metrics['max_error']:.6f}")
```

The validation metrics include `mean_error`, `std_error`, `max_error`,
`min_error`, and `n_trajectories`.

### 19.2.8 Combined Workflow

A recommended workflow combining system identification with domain
randomization:

1. Collect real-robot trajectories (10-20 short trajectories).
2. Run system identification to find the best parameter vector $\theta^*$.
3. Set the nominal simulation parameters to $\theta^*$.
4. Apply domain randomization centered around $\theta^*$, using narrower ranges
   than the defaults since the nominal parameters are now closer to reality.
5. Train the RL policy with domain randomization.

---
