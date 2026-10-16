# Chapter 15: Locomotion

The locomotion module (`src/robotics/locomotion/`) implements the gait management
pipeline for bipedal robots: gait parameterization, finite-state-machine
transitions, footstep planning, and balance analysis via the Zero Moment Point.

---

## 15.1 Gait Types and Parameters

### 15.1.1 GaitType Enumeration

The `GaitType` enum (`src/robotics/locomotion/gait_types.py`) enumerates the
supported gait patterns:

| Member   | Description                                              |
| -------- | -------------------------------------------------------- |
| `STAND`  | Static standing with both feet on the ground.            |
| `WALK`   | Normal walking gait with a double-support phase.         |
| `TROT`   | Trot gait using diagonal leg pairs (quadruped or biped). |
| `RUN`    | Running gait with a flight phase (no ground contact).    |
| `CRAWL`  | Slow, maximally stable crawl.                            |
| `BOUND`  | Bounding gait (synchronous front/rear pairs).            |
| `GALLOP` | Galloping with asymmetric leg phasing.                   |

### 15.1.2 GaitPhase Enumeration

`GaitPhase` describes the phase within a single gait cycle:

| Member           | Description                         |
| ---------------- | ----------------------------------- |
| `DOUBLE_SUPPORT` | Both feet are on the ground.        |
| `LEFT_SUPPORT`   | Only the left foot is in contact.   |
| `RIGHT_SUPPORT`  | Only the right foot is in contact.  |
| `FLIGHT`         | Both feet are in the air (running). |
| `LEFT_SWING`     | Left leg is in swing phase.         |
| `RIGHT_SWING`    | Right leg is in swing phase.        |

### 15.1.3 LegState Enumeration

`LegState` tracks the state of an individual leg through its cycle:

| Member          | Description                        |
| --------------- | ---------------------------------- |
| `STANCE`        | Foot on ground, supporting weight. |
| `SWING`         | Foot in air, moving forward.       |
| `EARLY_CONTACT` | Just touched down.                 |
| `LATE_CONTACT`  | About to lift off.                 |
| `LOADING`       | Weight shifting onto this leg.     |
| `UNLOADING`     | Weight shifting off this leg.      |

### 15.1.4 GaitParameters Dataclass

The `GaitParameters` dataclass bundles all parameters that define a gait:

| Field                  | Type       | Default  | Description                                         |
| ---------------------- | ---------- | -------- | --------------------------------------------------- |
| `gait_type`            | `GaitType` | `WALK`   | Selected gait pattern.                              |
| `step_length`          | `float`    | `0.3`    | Forward step length in meters.                      |
| `step_width`           | `float`    | `0.2`    | Lateral step width in meters.                       |
| `step_height`          | `float`    | `0.05`   | Maximum foot lift height in meters.                 |
| `step_duration`        | `float`    | `0.5`    | Total duration of one step in seconds.              |
| `double_support_ratio` | `float`    | `0.2`    | Fraction of step spent in double support.           |
| `swing_height_profile` | `str`      | `"sine"` | Height profile during swing (`"sine"` or `"trap"`). |
| `com_height`           | `float`    | `0.9`    | Target center-of-mass height in meters.             |
| `max_foot_velocity`    | `float`    | `1.0`    | Maximum foot velocity in m/s.                       |
| `settling_time`        | `float`    | `0.1`    | Time to settle after a step in seconds.             |

**Construction-time validation:**

- `step_length >= 0`
- `step_width >= 0`
- `step_height >= 0`
- `step_duration > 0`
- `0 <= double_support_ratio <= 1`
- `com_height > 0`

**Derived properties:**

- **Swing duration:**

$$
t_{\text{swing}} = t_{\text{step}} \times (1 - r_{\text{ds}})
$$

where $t_{\text{step}}$ is `step_duration` and $r_{\text{ds}}$ is
`double_support_ratio`.

- **Double-support duration:**

$$
t_{\text{ds}} = t_{\text{step}} \times r_{\text{ds}}
$$

- **Step frequency:**

$$
f_{\text{step}} = \frac{1}{t_{\text{step}}}
$$

**Factory functions** provide pre-configured parameter sets:

- `create_walk_parameters(step_length=0.3, step_duration=0.5, com_height=0.9)` --
  Standard walking with 20% double support.
- `create_run_parameters(step_length=0.6, step_duration=0.3, com_height=0.85)` --
  Running with zero double support ratio (flight phase).
- `create_stand_parameters(step_width=0.2, com_height=0.9)` -- Standing with
  100% double support, zero step length and height.

---

## 15.2 Gait State Machine

### 15.2.1 Events

The `GaitEvent` enum defines the events that can trigger state transitions:

| Event             | Description                              |
| ----------------- | ---------------------------------------- |
| `STEP_COMPLETE`   | The current step has finished.           |
| `FOOT_CONTACT`    | A foot has touched the ground.           |
| `FOOT_LIFTOFF`    | A foot has left the ground.              |
| `BALANCE_LOST`    | The balance criterion has been violated. |
| `STOP_REQUESTED`  | The user has requested a stop.           |
| `START_REQUESTED` | The user has requested to start walking. |
| `SPEED_CHANGE`    | The velocity command has changed.        |
| `EMERGENCY_STOP`  | An emergency stop has been triggered.    |

### 15.2.2 GaitState

The `GaitState` dataclass holds the complete instantaneous state of the gait
controller:

| Field              | Type           | Default                                |
| ------------------ | -------------- | -------------------------------------- |
| `gait_type`        | `GaitType`     | `STAND`                                |
| `phase`            | `GaitPhase`    | `DOUBLE_SUPPORT`                       |
| `support_state`    | `SupportState` | `DOUBLE_SUPPORT_CENTERED`              |
| `phase_time`       | `float`        | `0.0` -- time elapsed in current phase |
| `cycle_time`       | `float`        | `0.0` -- time elapsed in current cycle |
| `step_count`       | `int`          | `0` -- total steps taken               |
| `is_walking`       | `bool`         | `False`                                |
| `stance_foot`      | `str`          | `"both"`                               |
| `next_stance_foot` | `str`          | `"left"`                               |

### 15.2.3 The GaitStateMachine Class

The FSM manages phase transitions for bipedal walking. The state diagram for
walking is:

```
DOUBLE_SUPPORT --> RIGHT_SWING --> DOUBLE_SUPPORT --> LEFT_SWING --> ...
```

**Initialization:**

```python
gait = GaitStateMachine(parameters=GaitParameters())
```

**Starting and stopping:**

- `start_walking()` -- transitions from standing to walking. Starts in
  `DOUBLE_SUPPORT` with `DOUBLE_SUPPORT_LEFT_LEADING` support state, preparing
  to lift the right foot. Invokes the `gait_change` callbacks.

- `stop_walking()` -- transitions back to standing. Only takes effect during
  `DOUBLE_SUPPORT` phase to avoid stopping mid-swing.

- `emergency_stop()` -- immediately forces the state to `DOUBLE_SUPPORT` /
  `STAND` regardless of current phase.

**Updating:**

```python
state = gait.update(dt=0.001)
```

Each call advances `phase_time` and `cycle_time` by `dt`. If the accumulated
phase time exceeds the current phase duration, a transition is triggered. Excess
time carries over to the new phase. A safety limit of 20 iterations per update
prevents infinite transition loops.

**Phase transitions:**

| From             | To               | Condition                     | Side effects                                    |
| ---------------- | ---------------- | ----------------------------- | ----------------------------------------------- |
| `DOUBLE_SUPPORT` | `RIGHT_SWING`    | `next_stance_foot == "left"`  | `stance_foot = "left"`                          |
| `DOUBLE_SUPPORT` | `LEFT_SWING`     | `next_stance_foot == "right"` | `stance_foot = "right"`                         |
| `RIGHT_SWING`    | `DOUBLE_SUPPORT` | Phase timer expires           | `step_count += 1`, `next_stance_foot = "left"`  |
| `LEFT_SWING`     | `DOUBLE_SUPPORT` | Phase timer expires           | `step_count += 1`, `next_stance_foot = "right"` |

Phase durations are computed from the gait parameters:

- `DOUBLE_SUPPORT` duration: `step_duration * double_support_ratio`
- Swing duration: `step_duration * (1 - double_support_ratio)`

**Callbacks:**

Three callback channels are supported: `"phase_change"`, `"step_complete"`, and
`"gait_change"`. Register with:

```python
gait.register_callback("step_complete", lambda state, event: print(f"Step {state.step_count}"))
```

Callbacks receive the current `GaitState` and the triggering `GaitEvent`.
Exceptions in callbacks are silently caught to prevent breaking the state
machine.

**External events:**

`handle_event(event)` processes external sensor events:

- `FOOT_CONTACT` -- forces transition from swing to double support (early
  touchdown).
- `FOOT_LIFTOFF` -- forces transition from double support to swing (early
  liftoff).

**Phase progress:**

`phase_progress` returns a float in $[0, 1]$ indicating how far through the
current phase the system has progressed:

$$
p = \min\left(1, \frac{t_{\text{phase}}}{T_{\text{phase}}}\right)
$$

**Foot trajectory phase:**

`get_foot_trajectory_phase(foot)` returns $p \in [0, 1]$ for trajectory
interpolation. During swing, it equals `phase_progress`; during stance, it
returns $1.0$.

---

## 15.3 Footstep Planning

### 15.3.1 Footstep and FootstepPlan Data Structures

A `Footstep` represents a single foot placement:

| Field         | Type                    | Description                          |
| ------------- | ----------------------- | ------------------------------------ |
| `position`    | `NDArray[float64]` (3,) | World-frame position of foot center. |
| `orientation` | `NDArray[float64]` (4,) | Quaternion $[w, x, y, z]$.           |
| `foot`        | `str`                   | `"left"` or `"right"`.               |
| `step_index`  | `int`                   | Sequential index in plan.            |
| `is_support`  | `bool`                  | Whether this is a support position.  |
| `timing`      | `float`                 | Planned start time of this step.     |
| `duration`    | `float`                 | Duration allocated for this step.    |

The `yaw` property extracts the heading angle from the quaternion using the
standard atan2 decomposition:

$$
\psi = \arctan2\bigl(2(w z + x y),\; 1 - 2(y^2 + z^2)\bigr)
$$

A `FootstepPlan` is an iterable container of `Footstep` objects with:

- `n_steps` -- total number of steps.
- `total_duration` -- sum of all step durations.
- `get_footsteps_for_foot(foot)` -- filter by foot.
- `get_footstep_at_time(t)` -- find the active footstep at time $t$.

### 15.3.2 The FootstepPlanner Class

```python
planner = FootstepPlanner(
    parameters=GaitParameters(),
    max_step_length=0.5,     # kinematic limit [m]
    max_step_width=0.4,      # kinematic limit [m]
    max_step_rotation=0.5,   # kinematic limit [rad]
)
```

**Yaw-to-quaternion conversion.** The internal `_yaw_to_quat` method converts a
yaw angle $\psi$ to a quaternion representing pure rotation about the $z$-axis:

$$
\mathbf{q} = \bigl[\cos(\psi/2),\; 0,\; 0,\; \sin(\psi/2)\bigr]
$$

**Angle normalization.** The `_normalize_angle` method maps an angle to the
range $[-\pi, \pi]$.

### 15.3.3 Planning to a Goal

```python
plan = planner.plan_to_goal(
    start=np.zeros(3),
    goal=np.array([2.0, 0.0, 0.0]),
    start_yaw=0.0,
    goal_yaw=None,      # defaults to direction toward goal
    start_foot="left",
)
```

The algorithm:

1. Compute the direction vector and distance in the $xy$-plane.
2. If distance $< 10^{-3}$ m, return an empty plan (already at goal).
3. If `goal_yaw` is `None`, set it to $\arctan2(\Delta y, \Delta x)$.
4. Determine the number of steps:
   $n = \max\bigl(1, \lceil d / \min(l_{\text{step}}, l_{\max})\rceil\bigr)$.
5. For each step $i$, interpolate position along the straight path at
   fraction $t = (i+1)/n$ and orientation by linear interpolation of yaw.
6. Add lateral offsets to place left and right feet at
   $\pm w_{\text{nom}}/2$ perpendicular to the walking direction.
7. Assign timing: each step starts at $i \cdot t_{\text{step}}$.

### 15.3.4 Planning From Velocity

```python
plan = planner.plan_from_velocity(
    current_position=np.zeros(3),
    current_yaw=0.0,
    velocity_command=np.array([0.5, 0.0, 0.1]),  # [vx, vy, omega]
    n_steps=4,
    start_foot="left",
)
```

The velocity command $[v_x, v_y, \omega]$ is integrated over one step duration to
compute the displacement:

$$
\Delta x = v_x \cdot \Delta t, \qquad
\Delta y = v_y \cdot \Delta t, \qquad
\Delta\psi = \omega \cdot \Delta t
$$

Each displacement is clamped to the kinematic limits:

$$
|\Delta x| \leq l_{\max}, \qquad
|\Delta y| \leq w_{\max}, \qquad
|\Delta\psi| \leq \psi_{\max}
$$

The yaw rotation is applied in two halves (before and after position update) for
smoother curvature. The displacement is then transformed to the world frame:

$$
\Delta x_w = \cos\psi \cdot \Delta x - \sin\psi \cdot \Delta y
$$

$$
\Delta y_w = \sin\psi \cdot \Delta x + \cos\psi \cdot \Delta y
$$

Lateral foot offsets are computed relative to the current heading:

- Left foot: offset $= (-\sin\psi, \cos\psi) \cdot w_{\text{nom}}/2$
- Right foot: offset $= (\sin\psi, -\cos\psi) \cdot w_{\text{nom}}/2$

### 15.3.5 Planning an In-Place Turn

```python
plan = planner.plan_in_place_turn(
    current_position=np.zeros(3),
    current_yaw=0.0,
    target_yaw=np.pi / 2,
    start_foot="left",
)
```

The total rotation $\Delta\psi = \text{normalize}(\psi_{\text{target}} - \psi_{\text{current}})$
is divided into steps of at most `max_step_rotation`:

$$
n = \max\bigl(1, \lceil|\Delta\psi| / \psi_{\max}\rceil\bigr),
\qquad
\delta\psi = \Delta\psi / n
$$

Each step rotates the feet around the current position by $\delta\psi$, with
alternating left/right placement.

---

## 15.4 Zero Moment Point

### 15.4.1 ZMP Theory

The Zero Moment Point is the point on the ground plane where the net moment of
the inertial forces and gravity has zero horizontal components. For a bipedal
robot to remain dynamically balanced, the ZMP must stay within the support
polygon.

### 15.4.2 ZMP Computation

The `ZMPComputer` class (`src/robotics/locomotion/zmp_computer.py`) computes the
ZMP from the robot's centroidal dynamics.

The ZMP position on a flat ground plane at height $z_g$ is:

$$
x_{\text{ZMP}} = x_{\text{CoM}} -
  \frac{z_{\text{CoM}} - z_g}{\ddot{z}_{\text{CoM}} + g}
  \left(\ddot{x}_{\text{CoM}} + \frac{\dot{L}_y}{m}\right)
$$

$$
y_{\text{ZMP}} = y_{\text{CoM}} -
  \frac{z_{\text{CoM}} - z_g}{\ddot{z}_{\text{CoM}} + g}
  \left(\ddot{y}_{\text{CoM}} - \frac{\dot{L}_x}{m}\right)
$$

where:

- $(x_{\text{CoM}}, y_{\text{CoM}}, z_{\text{CoM}})$ is the center of mass position.
- $(\ddot{x}_{\text{CoM}}, \ddot{y}_{\text{CoM}}, \ddot{z}_{\text{CoM}})$ is the
  CoM acceleration.
- $g = 9.81 \text{ m/s}^2$ is gravitational acceleration.
- $m$ is the total robot mass.
- $\dot{L}_x, \dot{L}_y$ are the rates of change of angular momentum about the
  $x$ and $y$ axes at the CoM.

The denominator $\ddot{z}_{\text{CoM}} + g$ represents the total vertical
acceleration. When this is near zero (free fall), the ZMP is undefined and the
method returns an invalid result with `is_valid=False`.

**Implementation.** The `compute_zmp` method accepts optional overrides for all
state quantities. If not provided:

- `com_position` is queried from the engine via `get_com_position()` (requires
  `HumanoidCapable`).
- `com_acceleration` defaults to zero (quasi-static assumption).
- `angular_momentum_rate` defaults to zero.

**Free-fall detection.** If $|\ddot{z}_{\text{CoM}} + g| < 10^{-6}$, the robot
is in free fall. The returned `ZMPResult` has `is_valid=False` and
`total_normal_force=0`.

### 15.4.3 ZMPResult

The `ZMPResult` dataclass contains:

| Field                | Type                    | Description                                       |
| -------------------- | ----------------------- | ------------------------------------------------- |
| `zmp_position`       | `NDArray[float64]` (3,) | ZMP on ground plane.                              |
| `cop_position`       | `NDArray[float64]` (3,) | Center of Pressure (same as ZMP for flat ground). |
| `is_valid`           | `bool`                  | Whether ZMP is within support polygon.            |
| `support_margin`     | `float`                 | Signed distance to boundary (negative = outside). |
| `total_normal_force` | `float`                 | Total vertical ground reaction force.             |
| `ground_height`      | `float`                 | Height of ground plane.                           |

### 15.4.4 Capture Point and Divergent Component of Motion

The **Capture Point** (also called the Instantaneous Capture Point or Divergent
Component of Motion, DCM) is the point on the ground where the robot should
place its foot to come to a complete stop from its current state:

$$
\mathbf{p}_{\text{CP}} = \mathbf{p}_{\text{CoM}} + \frac{\dot{\mathbf{p}}_{\text{CoM}}}{\omega_0}
$$

where $\omega_0 = \sqrt{g / z_{\text{CoM}}}$ is the natural frequency of the
Linear Inverted Pendulum Model (LIPM).

The `compute_capture_point` and `compute_dcm` methods both compute this
quantity (they are equivalent for the LIPM). Only the horizontal components
($x, y$) are meaningful; the $z$ component is set to the ground height.

**Physical interpretation.** If the capture point lies outside the support
polygon, the robot cannot stop without taking a step. The distance between the
capture point and the support polygon boundary indicates the urgency of stepping.

```python
zmp_computer = ZMPComputer(engine, ground_height=0.0)
capture = zmp_computer.compute_capture_point()
# capture[0:2] gives the horizontal capture point
```

### 15.4.5 Stability Margin

The `compute_stability_margin(zmp_position, support_polygon)` method returns the
**signed distance** from the ZMP to the nearest edge of the support polygon:

- **Positive:** ZMP is inside the polygon, with the value indicating the
  distance to the closest boundary.
- **Negative:** ZMP is outside the polygon, with the magnitude indicating the
  distance to the closest boundary.

The boundary distance is computed by finding the minimum point-to-segment
distance over all polygon edges. For a polygon with vertices
$\mathbf{v}_0, \mathbf{v}_1, \ldots, \mathbf{v}_{n-1}$, the distance from
point $\mathbf{p}$ to edge $(\mathbf{v}_i, \mathbf{v}_{i+1 \bmod n})$ is:

$$
d_i = \|\mathbf{p} - \text{closest}(\mathbf{p}, \overline{\mathbf{v}_i \mathbf{v}_{i+1}})\|
$$

where the closest point on the segment is:

$$
\text{closest} = \mathbf{v}_i + \text{clamp}\left(\frac{(\mathbf{p} - \mathbf{v}_i) \cdot (\mathbf{v}_{i+1} - \mathbf{v}_i)}{\|\mathbf{v}_{i+1} - \mathbf{v}_i\|^2}, 0, 1\right) (\mathbf{v}_{i+1} - \mathbf{v}_i)
$$

The stability margin is $\min_i d_i$ with sign determined by the point-in-polygon
test.

---
