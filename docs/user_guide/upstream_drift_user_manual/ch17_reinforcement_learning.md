# UpstreamDrift User Manual -- Part 5: Machine Learning and Deployment

---

# Chapter 17: Reinforcement Learning

The `learning/rl/` module provides a complete reinforcement learning framework for
training robot control policies in simulation. It builds on the Gymnasium API standard
and integrates tightly with the UpstreamDrift physics engines through the
`PhysicsEngineProtocol` interface. This chapter covers the configuration system,
the base environment class, and the concrete humanoid and manipulation environments.

Source files referenced in this chapter:

- `src/learning/rl/configs.py`
- `src/learning/rl/base_env.py`
- `src/learning/rl/humanoid_envs.py`
- `src/learning/rl/manipulation_envs.py`

---

## 17.1 Configuration System

The RL module uses a layered dataclass-based configuration system defined in
`src/learning/rl/configs.py`. Every aspect of the environment -- observations,
actions, rewards, and task parameters -- is governed by its own configuration
object. This design allows users to compose environments from independent,
reusable configuration blocks.

### 17.1.1 ActionMode Enum

The `ActionMode` enum selects the interface between the policy output and the
physics engine. It has four members:

| Member      | Value         | Description                                        |
| ----------- | ------------- | -------------------------------------------------- |
| `TORQUE`    | `"torque"`    | Direct joint torque commands in Nm.                |
| `POSITION`  | `"position"`  | Target joint positions in radians.                 |
| `VELOCITY`  | `"velocity"`  | Target joint velocities in rad/s.                  |
| `IMPEDANCE` | `"impedance"` | Position targets with stiffness and damping gains. |

The action mode determines how the raw policy output is interpreted by the
`_apply_action` method of each environment subclass. Torque mode provides the
most direct control but requires the policy to learn inverse dynamics implicitly.
Position mode is the most forgiving because low-level joint PD controllers track
the target, but it hides dynamics from the policy. Impedance mode is a hybrid
that exposes compliance parameters.

Usage:

```python
from src.learning.rl.configs import ActionMode

mode = ActionMode.TORQUE
print(mode.value)  # "torque"
```

### 17.1.2 TaskType Enum

The `TaskType` enum categorizes environments by the high-level task objective:

| Member         | Value            | Typical environments                   |
| -------------- | ---------------- | -------------------------------------- |
| `LOCOMOTION`   | `"locomotion"`   | Walking, running, stair climbing       |
| `MANIPULATION` | `"manipulation"` | Pick-and-place, assembly, tool use     |
| `BALANCE`      | `"balance"`      | Standing, balancing under perturbation |
| `TRACKING`     | `"tracking"`     | End-effector trajectory tracking       |
| `REACHING`     | `"reaching"`     | Reaching to target positions           |

The task type is stored in `TaskConfig` and is available to reward functions and
termination conditions. It can also be used by curriculum learning schedulers to
organize training progressions.

### 17.1.3 ObservationConfig

`ObservationConfig` is a dataclass that specifies which sensor channels to
include in the observation vector and what noise to apply to each channel.

**Fields and defaults:**

| Field                    | Type    | Default | Description                               |
| ------------------------ | ------- | ------- | ----------------------------------------- |
| `include_joint_pos`      | `bool`  | `True`  | Include joint position readings.          |
| `include_joint_vel`      | `bool`  | `True`  | Include joint velocity readings.          |
| `include_joint_torque`   | `bool`  | `False` | Include measured joint torques.           |
| `include_ee_pos`         | `bool`  | `False` | Include end-effector Cartesian position.  |
| `include_ee_vel`         | `bool`  | `False` | Include end-effector Cartesian velocity.  |
| `include_contact_forces` | `bool`  | `False` | Include ground reaction / contact forces. |
| `include_imu`            | `bool`  | `False` | Include IMU accelerometer + gyroscope.    |
| `include_privileged`     | `bool`  | `False` | Include privileged info (teacher only).   |
| `position_noise_std`     | `float` | `0.0`   | Gaussian noise on position readings.      |
| `velocity_noise_std`     | `float` | `0.0`   | Gaussian noise on velocity readings.      |
| `history_length`         | `int`   | `1`     | Number of stacked observation frames.     |

The `include_privileged` flag is intended for asymmetric actor-critic training.
When set to `True`, the observation includes ground-truth information (such as
terrain height maps or exact object poses) that would not be available on real
hardware. A teacher policy is trained with privileged information, then a student
policy is distilled from it using only the non-privileged observations.

**Computing the observation dimension.** The `get_obs_dim` method calculates the
total flat observation size:

```python
def get_obs_dim(self, n_joints: int, n_ee: int = 0) -> int:
```

The formula accumulates dimensions channel by channel:

$$
d_{obs} = \Bigl(\underbrace{n_q}_{\text{pos}} + \underbrace{n_q}_{\text{vel}} + \underbrace{n_q}_{\text{torque}} + \underbrace{3 n_{ee}}_{\text{ee pos}} + \underbrace{6 n_{ee}}_{\text{ee vel}} + \underbrace{6}_{\text{IMU}}\Bigr) \times H
$$

where each term is included only if its corresponding flag is `True`, $n_q$ is
`n_joints`, $n_{ee}$ is the number of end-effectors, the IMU contributes 6
dimensions (3 accelerometer + 3 gyroscope), and $H$ is `history_length`. Note
that end-effector velocity is 6-dimensional (linear + angular), while
end-effector position is 3-dimensional (Cartesian xyz only).

Contact forces are excluded from the `get_obs_dim` calculation because their
dimensionality varies by robot model. Environments that include contact forces
override `_build_observation_space` to account for them.

Example:

```python
from src.learning.rl.configs import ObservationConfig

obs_config = ObservationConfig(
    include_joint_pos=True,
    include_joint_vel=True,
    include_imu=True,
    position_noise_std=0.005,
    velocity_noise_std=0.01,
    history_length=3,
)
dim = obs_config.get_obs_dim(n_joints=21, n_ee=2)
# dim = (21 + 21 + 6) * 3 = 144
```

### 17.1.4 ActionConfig

`ActionConfig` governs the action space and the processing pipeline that maps raw
policy outputs to actuator commands.

**Fields and defaults:**

| Field             | Type         | Default             | Description                               |
| ----------------- | ------------ | ------------------- | ----------------------------------------- |
| `mode`            | `ActionMode` | `ActionMode.TORQUE` | Control mode for the action space.        |
| `action_scale`    | `float`      | `1.0`               | Multiplicative scaling for actions.       |
| `action_clip`     | `float`      | `1.0`               | Symmetric clipping bound for raw actions. |
| `smoothing_alpha` | `float`      | `0.0`               | Exponential smoothing coefficient.        |

**The `process_action` pipeline.** Every raw action from the policy passes through
a three-stage pipeline implemented in `process_action`:

```python
def process_action(
    self,
    action: NDArray[np.floating],
    prev_action: NDArray[np.floating] | None,
) -> NDArray[np.floating]:
```

The three stages are:

1. **Clipping.** The raw action is clamped to $[-c, c]$ where $c$ is
   `action_clip`:

$$
a^{clip}_i = \text{clip}(a^{raw}_i,\; -c,\; c)
$$

2. **Scaling.** The clipped action is multiplied by `action_scale`:

$$
a^{scaled}_i = s \cdot a^{clip}_i
$$

where $s$ is `action_scale`. This is useful when the policy outputs are
normalized to $[-1, 1]$ and must be mapped to physical torque ranges.

3. **Exponential smoothing.** If `smoothing_alpha` $\alpha > 0$ and a previous
   action is available, the output is a weighted blend of the previous and
   current actions:

$$
a_t = \alpha \, a_{t-1} + (1 - \alpha) \, a_t^{raw}
$$

where $a_{t-1}$ is the previous processed action and $a_t^{raw}$ is the
current scaled action. When $\alpha = 0$ there is no smoothing. Higher values
of $\alpha$ produce smoother but more sluggish motion, which can be beneficial
for sim-to-real transfer by filtering out high-frequency oscillations.

Example:

```python
from src.learning.rl.configs import ActionConfig, ActionMode

action_config = ActionConfig(
    mode=ActionMode.POSITION,
    action_scale=0.5,
    action_clip=1.0,
    smoothing_alpha=0.2,
)

import numpy as np
raw = np.array([1.5, -0.3, 0.8])
prev = np.array([0.1, 0.2, -0.1])
processed = action_config.process_action(raw, prev)
# Step 1: clip to [-1, 1] -> [1.0, -0.3, 0.8]
# Step 2: scale by 0.5 -> [0.5, -0.15, 0.4]
# Step 3: smooth: 0.2 * prev + 0.8 * scaled
```

### 17.1.5 RewardConfig

`RewardConfig` specifies the weighting of reward components. The total reward at
each timestep is assembled by the environment's `_compute_reward` method using
these weights.

**Fields and defaults:**

| Field                       | Type    | Default  | Description                           |
| --------------------------- | ------- | -------- | ------------------------------------- |
| `task_reward_weight`        | `float` | `1.0`    | Weight for the task-specific reward.  |
| `energy_penalty_weight`     | `float` | `0.001`  | Weight for the energy penalty.        |
| `smoothness_penalty_weight` | `float` | `0.0001` | Weight for the smoothness penalty.    |
| `contact_penalty_weight`    | `float` | `0.0`    | Weight for undesired contacts.        |
| `use_potential_shaping`     | `bool`  | `False`  | Enable potential-based shaping.       |
| `alive_bonus`               | `float` | `0.0`    | Constant bonus per non-terminal step. |

**Energy penalty.** The `compute_energy_penalty` method penalizes high torques to
encourage efficient motions:

$$
R_{energy} = -w_e \sum_{i=1}^{n} \tau_i^2
$$

where $w_e$ is `energy_penalty_weight` and $\tau_i$ is the torque applied to
joint $i$. The method computes `float(np.sum(torques**2)) * self.energy_penalty_weight`.
This quadratic penalty strongly discourages large torques while allowing moderate
ones, which tends to produce smoother, more energy-efficient gaits.

**Smoothness penalty.** The `compute_smoothness_penalty` method penalizes abrupt
changes in the action signal:

$$
R_{smooth} = -w_s \sum_{i=1}^{n} (a_i^t - a_i^{t-1})^2
$$

where $w_s$ is `smoothness_penalty_weight`, $a_i^t$ is the current action for
joint $i$, and $a_i^{t-1}$ is the previous action. When `prev_action` is `None`
(at the start of an episode), the penalty is zero. This penalty complements the
exponential smoothing in `ActionConfig` -- smoothing operates on the action
signal directly, while the smoothness penalty provides a gradient signal to the
policy.

**Alive bonus.** The constant `alive_bonus` is added at every non-terminal
timestep. This encourages the policy to avoid early termination conditions (such
as falling). A typical value for locomotion tasks is 0.5 to 1.0.

**Potential shaping.** When `use_potential_shaping` is `True`, the environment
adds a potential-based shaping term:

$$
\Delta\Phi = \gamma \, \Phi(s') - \Phi(s)
$$

where $\Phi(s)$ is a potential function defined by the environment subclass.
Potential shaping preserves the optimal policy under certain conditions
(Ng et al., 1999) while providing denser learning signal. For the walking
environment, $\Phi(s) = x_{forward}$, so the shaping reward is proportional to
the forward displacement per step.

Example:

```python
from src.learning.rl.configs import RewardConfig

reward_config = RewardConfig(
    task_reward_weight=2.0,
    energy_penalty_weight=0.005,
    smoothness_penalty_weight=0.001,
    alive_bonus=0.5,
    use_potential_shaping=True,
)
```

### 17.1.6 TaskConfig

`TaskConfig` defines the task objective and episode parameters.

**Fields and defaults:**

| Field                | Type                   | Default               | Description                      |
| -------------------- | ---------------------- | --------------------- | -------------------------------- | ------------------------------- |
| `task_type`          | `TaskType`             | `TaskType.LOCOMOTION` | Task category.                   |
| `target_velocity`    | `NDArray[np.floating]` | `[1.0, 0.0, 0.0]`     | Target velocity (locomotion).    |
| `target_position`    | `NDArray[np.floating]  | None`                 | `None`                           | Target position (manipulation). |
| `target_orientation` | `NDArray[np.floating]  | None`                 | `None`                           | Target orientation quaternion.  |
| `max_episode_steps`  | `int`                  | `1000`                | Maximum timesteps per episode.   |
| `early_termination`  | `bool`                 | `True`                | Terminate on failure conditions. |
| `success_threshold`  | `float`                | `0.05`                | Error threshold for success.     |

The `is_success(error)` method returns `True` when the task error falls below
`success_threshold`, providing a binary success metric for evaluation.

---

## 17.2 Gymnasium Environment Base

The `RoboticsGymEnv` class in `src/learning/rl/base_env.py` is the abstract base
for all RL environments. It follows the Gymnasium `Env` interface but does not
inherit from `gymnasium.Env` directly -- instead it implements the same API to
avoid a hard dependency on Gymnasium at import time. If Gymnasium is not
installed, importing the module raises a clear error message at instantiation.

### 17.2.1 Constructor

```python
class RoboticsGymEnv:
    def __init__(
        self,
        engine: PhysicsEngineProtocol,
        model_path: str | None = None,
        task_config: TaskConfig | None = None,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
```

**Parameters:**

- `engine` -- A physics engine implementing `PhysicsEngineProtocol`. This can be
  any of the UpstreamDrift backends (MuJoCo, PyBullet, Drake, etc.).
- `model_path` -- Optional path to a robot model file (URDF, MJCF). If provided
  and the engine has a `load_from_path` method, the model is loaded
  automatically.
- `task_config`, `obs_config`, `action_config`, `reward_config` -- Configuration
  objects. If `None`, default configurations are used.
- `render_mode` -- Either `"human"` (on-screen display) or `"rgb_array"` (return
  pixel buffer). Matches the Gymnasium render mode convention.

During construction the class:

1. Loads the model into the engine if `model_path` is provided.
2. Instantiates default configs for any that are `None`.
3. Queries the engine for `n_joints` (from `engine.n_q`), `n_actuators` (from
   `engine.n_v`), and `n_end_effectors` (default 1, overridden by subclasses).
4. Builds the observation and action spaces as `gymnasium.spaces.Box` objects.
5. Initializes episode state variables: step counter, previous action, previous
   potential, and the random number generator.

### 17.2.2 Observation and Action Spaces

The observation space is a `Box` of shape `(obs_dim,)` with bounds
$(-\infty, +\infty)$ and `dtype=np.float32`. The observation dimension is
computed from `obs_config.get_obs_dim(n_joints, n_ee)`.

The action space is a `Box` of shape `(n_actuators,)` with bounds
$[-c, c]$ where $c$ is `action_config.action_clip`, and `dtype=np.float32`.

### 17.2.3 The `step` Method

```python
def step(
    self, action: NDArray[np.floating]
) -> tuple[NDArray[np.floating], float, bool, bool, dict[str, Any]]:
```

The `step` method follows the standard Gymnasium 5-tuple return signature and
executes the following sequence:

1. **Process action.** The raw action passes through
   `action_config.process_action(action, self._prev_action)` for clipping,
   scaling, and smoothing.

2. **Apply action.** Calls the abstract `_apply_action(processed_action)` which
   the subclass implements to write commands to the physics engine.

3. **Step simulation.** Calls `_step_simulation()` to advance the physics by one
   control timestep (which may correspond to multiple physics substeps).

4. **Get observation.** Calls `_get_observation()` to read sensor data from the
   engine and assemble the observation vector.

5. **Compute reward.** Calls `_compute_reward(processed_action)` to compute the
   scalar reward.

6. **Check termination.** Calls `_check_termination()` for failure-based
   termination (`terminated`), and compares `_step_count` against
   `task_config.max_episode_steps` for time-based truncation (`truncated`).

7. **Build info dict.** Calls `_get_info()` for environment-specific metrics.

8. **Update state.** Increments `_step_count` and stores the processed action as
   `_prev_action`.

### 17.2.4 The `reset` Method

```python
def reset(
    self,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[NDArray[np.floating], dict[str, Any]]:
```

The `reset` method:

1. Seeds or initializes the random number generator via
   `np.random.default_rng(seed)`.
2. Calls `_reset_simulation(options)` to reset the physics state.
3. Resets `_step_count` to 0, `_prev_action` to `None`, and
   `_prev_potential` to `_compute_potential()`.
4. Returns the initial observation and info dict.

### 17.2.5 Abstract Methods

Subclasses must implement six abstract methods:

| Method               | Signature                    | Purpose                               |
| -------------------- | ---------------------------- | ------------------------------------- | --------------------------------------- |
| `_apply_action`      | `(action: NDArray) -> None`  | Write commands to the physics engine. |
| `_step_simulation`   | `() -> None`                 | Advance the simulation forward.       |
| `_get_observation`   | `() -> NDArray`              | Read and assemble the observation.    |
| `_compute_reward`    | `(action: NDArray) -> float` | Compute the scalar reward.            |
| `_check_termination` | `() -> bool`                 | Check for early termination.          |
| `_reset_simulation`  | `(options: dict              | None) -> None`                        | Reset the physics to the initial state. |

Additionally, two optional hook methods can be overridden:

- `_compute_potential() -> float` -- Returns the potential function value for
  reward shaping. Default returns 0.0.
- `_get_info() -> dict[str, Any]` -- Returns additional info. Default returns
  `{"step_count": self._step_count}`.

### 17.2.6 Rendering

The `render()` method dispatches to `_render_frame()` (returns an RGB `uint8`
array of shape `(480, 640, 3)`) or `_render_human()` (on-screen display) based
on `render_mode`. The `close()` method calls `engine.close()` if available.

---

## 17.3 Humanoid Environments

The humanoid environments in `src/learning/rl/humanoid_envs.py` provide
locomotion and balance tasks for full-body humanoid robots. Both environments
inherit from `RoboticsGymEnv` and implement all six abstract methods.

### 17.3.1 HumanoidWalkEnv

`HumanoidWalkEnv` trains a humanoid to walk at a commanded forward velocity.

**Constructor parameters (beyond the base class):**

| Parameter         | Type    | Default | Description                  |
| ----------------- | ------- | ------- | ---------------------------- |
| `target_velocity` | `float` | `1.0`   | Target forward speed in m/s. |

The constructor creates a `TaskConfig` with `task_type=TaskType.LOCOMOTION`,
`target_velocity=[target_velocity, 0.0, 0.0]`, and `max_episode_steps=1000`.
Two termination thresholds are defined:

- `_base_height_threshold = 0.3` -- Minimum base height in meters before
  the episode terminates (the robot has fallen).
- `_base_tilt_threshold = 0.8` -- Minimum vertical component of the torso's
  up-vector. This corresponds to a maximum tilt angle of approximately
  $\cos^{-1}(0.8) \approx 37^{\circ}$.

**Observation assembly.** The `_get_observation` method concatenates the enabled
sensor channels in order: joint positions, joint velocities, joint torques, IMU
data, and contact forces. Gaussian noise is added to positions and velocities
according to `obs_config.position_noise_std` and `obs_config.velocity_noise_std`:

$$
\tilde{q} = q + \mathcal{N}(0, \sigma_q^2 I)
$$

$$
\tilde{\dot{q}} = \dot{q} + \mathcal{N}(0, \sigma_v^2 I)
$$

**Reward function.** The total reward at each step is:

$$
R = w_{task} \exp\bigl(-\|v - v_{target}\|\bigr) + R_{alive} + R_{energy} + R_{smooth} + \Delta\Phi
$$

where:

- $v$ is the current base velocity (first two components, i.e., forward and
  lateral), and $v_{target}$ is the target velocity.
- $w_{task}$ is `reward_config.task_reward_weight`.
- $R_{alive}$ is `reward_config.alive_bonus`.
- $R_{energy} = -w_e \sum_i \tau_i^2$ as described in Section 17.1.5.
- $R_{smooth} = -w_s \sum_i (a_i^t - a_i^{t-1})^2$ as described in Section 17.1.5.
- $\Delta\Phi = \gamma \Phi(s') - \Phi(s)$ with $\Phi(s) = x_{forward}$, the
  x-coordinate of the base position. This is only added when
  `reward_config.use_potential_shaping` is `True`.

The exponential velocity reward $\exp(-\|v - v_{target}\|)$ is maximized at 1.0
when the robot matches the target velocity exactly, and decays smoothly toward 0
as the error increases. This is preferable to a linear penalty because it is
bounded and provides strong signal near the optimum.

**Termination conditions.** The episode terminates early if
`task_config.early_termination` is `True` and either:

1. The base height drops below 0.3 m (the robot has fallen):

$$
z_{base} < 0.3
$$

2. The base tilts excessively. The vertical component of the up-vector is
   computed from the orientation quaternion as:

$$
up_z = 1 - 2(q_y^2 + q_z^2)
$$

and the episode terminates if $up_z < 0.8$.

**Reset with perturbation.** The `_reset_simulation` method resets the engine and
optionally applies a small random perturbation to the initial joint positions.
The `options` dictionary supports:

- `randomize_init` (bool, default `True`) -- Whether to perturb the initial state.
- `init_noise_scale` (float, default `0.01`) -- Standard deviation of the
  Gaussian perturbation added to joint positions.

**Info dictionary.** The returned info includes:

- `forward_velocity` -- Current forward speed.
- `lateral_velocity` -- Current lateral speed.
- `base_height` -- Current base height.
- `forward_distance` -- Total forward distance traveled.

### 17.3.2 HumanoidStandEnv

`HumanoidStandEnv` trains a humanoid to maintain a stable standing posture,
optionally under external perturbations.

**Constructor parameters (beyond the base class):**

| Parameter            | Type    | Default | Description                      |
| -------------------- | ------- | ------- | -------------------------------- |
| `perturbation_force` | `float` | `0.0`   | Maximum perturbation force in N. |

The constructor creates a `TaskConfig` with `task_type=TaskType.BALANCE`,
zero target velocity, and `max_episode_steps=500`. The target standing height
is `_target_height = 0.9` meters.

**External perturbations.** When `perturbation_force > 0`, the `_step_simulation`
method applies random horizontal forces to the torso with a 1% probability per
timestep. The force is sampled uniformly from
$[-F_{max}, F_{max}]$ in the x and y components, with zero vertical component:

```python
force = rng.uniform(-perturbation_force, perturbation_force, size=3)
force[2] = 0  # No vertical force
engine.apply_external_force("torso", force)
```

**Reward function.** The standing reward combines four components:

$$
R = w_h \, e^{-10\,|z - 0.9|} + w_u \, \frac{up_z + 1}{2} + w_s \, e^{-\|v\|} + R_{alive} + R_{energy}
$$

where:

- $w_h$ is `reward_config.task_reward_weight`, and $z$ is the base height.
  The exponential term $e^{-10|z - 0.9|}$ rewards maintaining the target height
  of 0.9 m with a sharp peak.

- $w_u = 0.5$ (hardcoded) weights the upright reward. The quantity
  $up_z = 1 - 2(q_y^2 + q_z^2)$ is the vertical component of the orientation,
  normalized to $[0, 1]$ by $(up_z + 1)/2$.

- $w_s = 0.3$ (hardcoded) weights the stillness reward. The term
  $e^{-\|v\|}$ rewards low base velocity, encouraging the robot to stand in
  place rather than drift.

- $R_{alive}$ and $R_{energy}$ are as previously defined.

**Termination.** The episode terminates if the base height drops below 0.3 m.

**Info dictionary.** Returns `base_height` and `upright_score` (the raw $up_z$
value before normalization).

---

## 17.4 Manipulation Environments

The manipulation environments in `src/learning/rl/manipulation_envs.py` provide
object interaction tasks for robotic arms.

### 17.4.1 ManipulationPickPlaceEnv

`ManipulationPickPlaceEnv` implements a three-phase pick-and-place task where the
robot must reach for an object, grasp it, and place it at a target location.

**Constructor parameters (beyond the base class):**

| Parameter            | Type     | Default | Description       |
| -------------------- | -------- | ------- | ----------------- | -------------------------- |
| `object_initial_pos` | `NDArray | None`   | `[0.5, 0.0, 0.1]` | Initial object position.   |
| `target_pos`         | `NDArray | None`   | `[0.5, 0.3, 0.1]` | Target placement position. |

The environment uses `TaskType.MANIPULATION`, `max_episode_steps=500`,
`early_termination=False`, and `success_threshold=0.05`.

**Grasp state machine.** The task proceeds through three logical phases tracked
by the boolean `_object_grasped`:

1. **Reaching phase** (`_object_grasped = False`). The robot must move its
   end-effector toward the object. The grasp threshold is 0.05 m.

2. **Grasping transition.** When the end-effector is within `_grasp_threshold`
   distance of the object AND the gripper is closed (gripper state < 0.1), the
   `_update_grasp_state` method sets `_object_grasped = True`. The object then
   follows the end-effector position.

3. **Placing phase** (`_object_grasped = True`). The robot must transport the
   object to the target location. Success is declared when the object-to-target
   distance is less than `_place_threshold` (0.05 m).

**Observation space.** The observation extends the base observation (joint
positions, velocities, end-effector state) with:

- Object position (3D) -- either the engine's reported position or the
  end-effector position if grasped.
- Target position (3D).
- Grasp state (1D binary).

The total observation dimension is `base_dim + 7`.

**Reward function.** The reward is phase-dependent:

- **Reaching phase:**

$$
R_{reach} = e^{-5 \, d_{hand \to obj}} \cdot w_{task} \cdot 0.5
$$

where $d_{hand \to obj} = \|p_{ee} - p_{obj}\|$ is the Euclidean distance
from the end-effector to the object.

- **Placing phase:**

$$
R_{place} = e^{-5 \, d_{obj \to target}} \cdot w_{task}
$$

where $d_{obj \to target} = \|p_{obj} - p_{target}\|$ is the distance from
the (grasped) object to the target.

- **Success bonus:** A reward of $+10$ is awarded when
  $d_{obj \to target} < 0.05$.

Both phases also include energy and smoothness penalties.

**Termination.** The episode terminates (success) when the grasped object is
within `_place_threshold` of the target. There is no failure termination since
`early_termination=False` in the task config.

**Reset options.** The `options` dictionary supports `randomize_positions`
(bool), which adds uniform noise in $[-0.1, 0.1]$ to both the object and target
positions for curriculum-style training.

### 17.4.2 DualArmManipulationEnv

`DualArmManipulationEnv` implements a bimanual coordination task where two robot
arms must cooperate to lift a heavy object.

**Constructor parameters (beyond the base class):**

| Parameter     | Type    | Default | Description               |
| ------------- | ------- | ------- | ------------------------- |
| `object_mass` | `float` | `5.0`   | Mass of the object in kg. |

This environment overrides `_get_n_end_effectors` to return 2 and doubles the
action space to accommodate both arms. The action vector is split in half:
`action[:n]` controls the left arm and `action[n:]` controls the right arm.

**Bimanual grasp state.** Three boolean flags track the task state:

- `_left_grasped` -- Left gripper has grasped the object (distance < 0.08 m and
  gripper closed).
- `_right_grasped` -- Right gripper has grasped the object.
- `_object_lifted` -- Both arms are grasping simultaneously. This is the only
  condition under which the object can be lifted, reflecting the requirement
  that a single arm cannot support the heavy object.

**Observation.** The observation concatenates:

1. Left arm joint positions and velocities.
2. Right arm joint positions and velocities.
3. Object position (3D).
4. Target position (3D).
5. Grasp states (3D): `[left_grasped, right_grasped, object_lifted]`.

**Reward function.** The reward structure encourages coordination:

- **Before lifting** ($\text{object\_lifted} = \text{False}$):

$$
R_{reach} = e^{-3(d_{left} + d_{right})} \cdot w_{task} \cdot 0.3
$$

where $d_{left}$ and $d_{right}$ are the distances from each end-effector to
the object. Additionally, a bonus of $+1.0$ is awarded for each arm that
achieves a grasp.

- **During lifting** ($\text{object\_lifted} = \text{True}$):

$$
R_{lift} = e^{-3 \, d_{obj \to target}} \cdot w_{task}
$$

A large bonus of $+20$ is awarded when $d_{obj \to target} < 0.1$.

- **Coordination penalty:** The velocity difference between the two
  end-effectors is penalized to encourage synchronized motion:

$$
R_{coord} = -0.1 \, \|v_{left} - v_{right}\|
$$

This term prevents one arm from jerking the object away from the other.

- Energy penalty is also applied.

**Termination.** The episode terminates if the object has been lifted
(`_object_lifted = True`) but then drops below $z = 0.05$ m, indicating the
arms lost their grip.

---
