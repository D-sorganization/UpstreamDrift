# Chapter 20: Deployment Infrastructure

The `deployment/` module provides the complete infrastructure for deploying
trained policies to real robot hardware. It covers real-time control loops,
digital twin synchronization, safety monitoring, and teleoperation.

Source files referenced in this chapter:

- `src/deployment/realtime/controller.py`
- `src/deployment/realtime/state.py`
- `src/deployment/digital_twin/twin.py`
- `src/deployment/digital_twin/estimator.py`
- `src/deployment/safety/monitor.py`
- `src/deployment/safety/collision.py`
- `src/deployment/teleoperation/interface.py`
- `src/deployment/teleoperation/devices.py`

---

## 20.1 Real-Time Control

The real-time control subsystem manages the high-frequency control loop that
interfaces with robot hardware. It is designed around a callback model where
user-defined control functions are executed at a fixed frequency.

### 20.1.1 Communication Protocols

The `CommunicationType` enum defines supported communication protocols:

| Member       | Value          | Description                             |
| ------------ | -------------- | --------------------------------------- |
| `ETHERCAT`   | `"ethercat"`   | Industrial real-time fieldbus.          |
| `ROS2`       | `"ros2"`       | ROS 2 middleware.                       |
| `UDP`        | `"udp"`        | Raw UDP sockets.                        |
| `SIMULATION` | `"simulation"` | Simulated interface (no real hardware). |

The `SIMULATION` protocol is always available and is the default. It allows
testing the full deployment pipeline without hardware.

### 20.1.2 RobotConfig

`RobotConfig` specifies the hardware parameters of the robot:

| Field                | Type                | Default        | Description              |
| -------------------- | ------------------- | -------------- | ------------------------ | ---------------------------- |
| `name`               | `str`               | required       | Robot identifier string. |
| `n_joints`           | `int`               | required       | Number of joints.        |
| `joint_names`        | `list[str]`         | auto-generated | Names for each joint.    |
| `joint_limits_lower` | `NDArray            | None`          | `None`                   | Lower position limits (rad). |
| `joint_limits_upper` | `NDArray            | None`          | `None`                   | Upper position limits (rad). |
| `velocity_limits`    | `NDArray            | None`          | `None`                   | Maximum velocities (rad/s).  |
| `torque_limits`      | `NDArray            | None`          | `None`                   | Maximum torques (Nm).        |
| `communication_type` | `CommunicationType` | `SIMULATION`   | Communication protocol.  |
| `ip_address`         | `str`               | `"127.0.0.1"`  | Robot IP (if networked). |
| `port`               | `int`               | `5000`         | Communication port.      |

If `joint_names` is not provided, they are auto-generated as
`["joint_0", "joint_1", ...]`.

### 20.1.3 ControlMode and ControlCommand

The `ControlMode` enum (in `src/deployment/realtime/state.py`) defines the
hardware-level control modes:

| Member      | Value         | Required fields                            |
| ----------- | ------------- | ------------------------------------------ |
| `POSITION`  | `"position"`  | `position_targets`                         |
| `VELOCITY`  | `"velocity"`  | `velocity_targets`                         |
| `TORQUE`    | `"torque"`    | `torque_commands`                          |
| `IMPEDANCE` | `"impedance"` | `position_targets`, `stiffness`, `damping` |
| `HYBRID`    | `"hybrid"`    | Mixed mode (implementation-specific).      |

The `ControlCommand` dataclass packages a control command to send to the robot:

| Field                | Type          | Description                    |
| -------------------- | ------------- | ------------------------------ | ------------------------------------ |
| `timestamp`          | `float`       | Command timestamp (seconds).   |
| `mode`               | `ControlMode` | Control mode for this command. |
| `position_targets`   | `NDArray      | None`                          | Target joint positions (rad).        |
| `velocity_targets`   | `NDArray      | None`                          | Target joint velocities (rad/s).     |
| `torque_commands`    | `NDArray      | None`                          | Commanded joint torques (Nm).        |
| `feedforward_torque` | `NDArray      | None`                          | Feed-forward torque term (Nm).       |
| `stiffness`          | `NDArray      | None`                          | Joint stiffness (impedance mode).    |
| `damping`            | `NDArray      | None`                          | Joint damping (impedance mode).      |
| `gripper_command`    | `float        | None`                          | Gripper position (0=closed, 1=open). |

**Validation.** The `validate(n_joints)` method checks that the required fields
for the current mode are present and have the correct dimension. It raises
`ValueError` on any inconsistency.

**Factory methods.** Three convenience constructors simplify command creation:

```python
cmd = ControlCommand.position_command(timestamp, positions, feedforward=None)
cmd = ControlCommand.torque_command(timestamp, torques)
cmd = ControlCommand.impedance_command(timestamp, positions, stiffness, damping)
```

### 20.1.4 RobotState

`RobotState` encapsulates all sensor data received from the robot at a single
timestep:

| Field                | Type                | Description                      |
| -------------------- | ------------------- | -------------------------------- | ------------------------------------- |
| `timestamp`          | `float`             | Measurement timestamp (seconds). |
| `joint_positions`    | `NDArray`           | Joint positions (rad).           |
| `joint_velocities`   | `NDArray`           | Joint velocities (rad/s).        |
| `joint_torques`      | `NDArray`           | Measured joint torques (Nm).     |
| `ft_wrenches`        | `dict[str, NDArray] | None`                            | Force/torque sensor readings by name. |
| `imu_data`           | `IMUReading         | None`                            | IMU sensor data.                      |
| `contact_states`     | `list[bool]         | None`                            | Binary contact flags.                 |
| `motor_temperatures` | `NDArray            | None`                            | Motor temperatures (Celsius).         |
| `battery_level`      | `float              | None`                            | Battery level (0.0 to 1.0).           |

The `get_state_vector()` method returns the concatenated state $[q, \dot{q}]$.
The `get_ft_wrench(sensor_name)` method looks up a specific force/torque sensor.

The `IMUReading` dataclass stores inertial measurement unit data:

| Field                 | Shape  | Description                        |
| --------------------- | ------ | ---------------------------------- |
| `timestamp`           | scalar | Measurement time.                  |
| `linear_acceleration` | `(3,)` | Accelerometer [ax, ay, az] m/s^2.  |
| `angular_velocity`    | `(3,)` | Gyroscope [wx, wy, wz] rad/s.      |
| `orientation`         | `(4,)` | Orientation quaternion (optional). |

### 20.1.5 TimingStatistics

`TimingStatistics` records control loop performance:

| Field             | Description                                           |
| ----------------- | ----------------------------------------------------- |
| `mean_cycle_time` | Average time per control cycle (seconds).             |
| `max_cycle_time`  | Worst-case cycle time.                                |
| `min_cycle_time`  | Best-case cycle time.                                 |
| `std_cycle_time`  | Standard deviation of cycle times.                    |
| `jitter`          | Maximum deviation from target period.                 |
| `overruns`        | Number of times the cycle exceeded the target period. |
| `total_cycles`    | Total number of control cycles executed.              |
| `uptime`          | Total running time in seconds.                        |

Jitter is computed as:

$$
\text{jitter} = \max_t |T_{cycle}(t) - T_{target}|
$$

where $T_{target} = 1/f_{control}$ is the target period.

### 20.1.6 The RealTimeController Class

`RealTimeController` orchestrates the control loop.

**Construction.**

```python
from src.deployment.realtime.controller import RealTimeController

controller = RealTimeController(
    control_frequency=1000.0,  # 1 kHz
    communication_type="simulation",
)
```

The controller stores the frequency and computes `dt = 1.0 / control_frequency`.
It initializes thread locks for state and command buffers, ensuring thread-safe
access from the main thread while the control loop runs in a background thread.

**Connection lifecycle.**

```python
config = RobotConfig(name="my_robot", n_joints=7)

# Connect
success = controller.connect(config)  # Returns True on success

# ... use controller ...

# Disconnect (stops loop if running)
controller.disconnect()
```

The `connect` method dispatches to protocol-specific connection handlers
(`_connect_ros2`, `_connect_udp`, `_connect_ethercat`) or succeeds immediately
for the `SIMULATION` protocol.

**Setting the control callback.**

```python
def my_controller(state: RobotState) -> ControlCommand:
    """User-defined control law."""
    # Compute desired torques from state
    error = desired_q - state.joint_positions
    tau = kp * error - kd * state.joint_velocities
    return ControlCommand.torque_command(
        timestamp=state.timestamp,
        torques=tau,
    )

controller.set_control_callback(my_controller)
```

The callback function receives the current `RobotState` and must return a
`ControlCommand`. It is called at `control_frequency` Hz.

**Starting and stopping.**

```python
controller.start()   # Starts background control thread
# ... robot is running ...
controller.stop()    # Stops loop, sends zero torque command
```

The `start` method raises `RuntimeError` if the controller is not connected or
no callback is set. The `stop` method sets a flag, waits for the control thread
to finish (with a 2-second timeout), and sends a zero-torque safety command.

**The control loop.** The `_control_loop` method runs in a daemon thread and
implements a fixed-rate loop:

```
next_cycle_time = now()

while not should_stop:
    cycle_start = now()

    state = read_state()
    command = control_callback(state)
    command.validate(n_joints)
    send_command(command)

    cycle_time = now() - cycle_start
    record_timing(cycle_time)

    if cycle_time > dt:
        overruns += 1

    sleep_until(next_cycle_time + dt)
```

If the sleep time is negative (a deadline was missed), the timing reference is
reset to prevent cascading overruns. Cycle times are recorded for the
`get_timing_stats()` method.

**State and command access.** Thread-safe access to the most recent state and
command is provided by:

```python
state = controller.get_last_state()
command = controller.get_last_command()
state = controller.wait_for_state(timeout=1.0)  # Blocks until new state
```

The `wait_for_state` method polls at 1 kHz until a new state arrives or the
timeout expires.

---

## 20.2 Digital Twin

The digital twin system maintains a synchronized simulation that mirrors the
physical robot's state, enabling prediction, anomaly detection, and contact
estimation.

### 20.2.1 The DigitalTwin Class

```python
from src.deployment.digital_twin.twin import DigitalTwin

twin = DigitalTwin(
    sim_engine=physics_engine,
    real_interface=real_time_controller,
)
```

The constructor takes a physics engine (for simulation) and a real-time
controller (for reading hardware state). It initializes a `StateEstimator` for
sensor fusion.

**Synchronization.** The `synchronize()` method reads the latest hardware state
and updates the simulation to match:

```python
error = twin.synchronize()
```

The synchronization error is computed as:

$$
e_{sync} = \|q_{real} - q_{sim}\| + 0.1 \, \|\dot{q}_{real} - \dot{q}_{sim}\|
$$

The velocity error is weighted by 0.1 because velocities are typically noisier
than positions.

**Prediction.** The `predict` method rolls out the simulation forward under a
given control sequence:

```python
trajectory = twin.predict(
    horizon=0.5,         # 0.5 seconds ahead
    control_sequence=u,  # Shape (n_steps, n_controls)
    dt=0.001,            # 1 kHz simulation
)
# trajectory has shape (n_steps + 1, 2 * n_dof)
```

The method saves the current simulation state, performs the rollout, and then
restores the original state. This allows prediction without disturbing the
twin's synchronized state.

### 20.2.2 Anomaly Detection

The `detect_anomaly` method compares simulation predictions with actual hardware
readings to detect deviations:

$$
\|x_{sim} - x_{real}\| > \epsilon
$$

where $\epsilon$ is the anomaly threshold (default 0.1 rad).

**AnomalyType enum.** Detected anomalies are categorized:

| Type                  | Description                                    |
| --------------------- | ---------------------------------------------- |
| `COLLISION`           | Unexpected contact detected from torque spike. |
| `SLIP`                | Slip detected (contact loss).                  |
| `STUCK`               | Joint is stuck (no motion despite torque).     |
| `MODEL_MISMATCH`      | Simulation-reality discrepancy.                |
| `SENSOR_FAULT`        | Sensor reading is out of range.                |
| `COMMUNICATION_ERROR` | Communication timeout or error.                |
| `JOINT_LIMIT`         | Joint approaching or exceeding limits.         |
| `VELOCITY_LIMIT`      | Velocity exceeding safe limits.                |
| `TORQUE_LIMIT`        | Torque exceeding safe limits.                  |

**AnomalyReport dataclass.**

| Field                | Type          | Description                       |
| -------------------- | ------------- | --------------------------------- | ------------------------------ |
| `timestamp`          | `float`       | Detection time.                   |
| `anomaly_type`       | `AnomalyType` | Category of anomaly.              |
| `severity`           | `float`       | Severity level, 0.0 to 1.0.       |
| `affected_joints`    | `list[int]`   | Indices of affected joints.       |
| `description`        | `str`         | Human-readable description.       |
| `recommended_action` | `str`         | Suggested response.               |
| `confidence`         | `float`       | Detection confidence, 0.0 to 1.0. |
| `raw_data`           | `dict         | None`                             | Raw sensor data for debugging. |

The `detect_anomaly` method checks two conditions:

1. **Position mismatch.** If any per-joint position error exceeds the threshold,
   a `MODEL_MISMATCH` anomaly is reported. The severity is normalized by 0.5 rad
   and capped at 1.0.

2. **Torque spike.** If any per-joint torque discrepancy exceeds 10 Nm, a
   `COLLISION` anomaly is reported with severity 0.8.

**Anomaly history.** All detected anomalies are stored in an internal list.
The `get_anomaly_history(max_age=None)` method retrieves anomalies, optionally
filtering by age. `clear_anomaly_history()` resets the history.

**Threshold tuning.** The `set_anomaly_threshold(threshold)` method adjusts the
detection sensitivity. Lower thresholds increase sensitivity but may produce
more false positives.

### 20.2.3 Contact Estimation

The `get_estimated_contacts` method estimates contact states from sensor data:

- Force/torque sensors: A contact is reported when the force magnitude exceeds
  1.0 N.
- Binary contact sensors: Contacts are reported directly.

The returned list contains dictionaries with contact location, force vector,
torque vector, and magnitude.

### 20.2.4 Virtual Force Estimation

The `compute_virtual_forces` method estimates external forces from the torque
discrepancy between simulation and reality:

$$
\tau_{virtual} = \tau_{real} - \tau_{sim}
$$

In the simplified implementation, the first 6 components of the torque
difference are interpreted as a Cartesian wrench. A full implementation would
use the Jacobian transpose mapping.

### 20.2.5 State Estimator

The `StateEstimator` in `src/deployment/digital_twin/estimator.py` implements
an Extended Kalman Filter (EKF) for fusing noisy sensor data into smooth state
estimates.

**State model.** The estimator tracks a $3n$-dimensional state vector:

$$
x = \begin{bmatrix} q \\ \dot{q} \\ \ddot{q} \end{bmatrix}
$$

consisting of joint positions, velocities, and accelerations for $n$ degrees
of freedom.

**EstimatorConfig:**

| Field                   | Default | Description                            |
| ----------------------- | ------- | -------------------------------------- |
| `process_noise`         | `0.001` | Process noise covariance.              |
| `measurement_noise`     | `0.01`  | Measurement noise covariance.          |
| `use_velocity_filter`   | `True`  | Apply low-pass filter to velocities.   |
| `velocity_filter_alpha` | `0.3`   | Filter coefficient (lower = smoother). |
| `outlier_threshold`     | `3.0`   | Sigma threshold for outlier rejection. |

**Prediction step.** The state transition model assumes constant acceleration:

$$
\hat{x}_{k|k-1} = F \, \hat{x}_{k-1|k-1}
$$

where the state transition matrix is:

$$
F = \begin{bmatrix}
I & I \Delta t & \frac{1}{2} I \Delta t^2 \\
0 & I & I \Delta t \\
0 & 0 & I
\end{bmatrix}
$$

The covariance prediction is:

$$
P_{k|k-1} = F \, P_{k-1|k-1} \, F^T + Q
$$

where $Q = \sigma_p^2 I_{3n}$ is the process noise covariance.

**Measurement model.** The measurement is:

$$
z = \begin{bmatrix} q_{meas} \\ \dot{q}_{meas} \end{bmatrix}
$$

with measurement matrix:

$$
H = \begin{bmatrix} I_n & 0 & 0 \\ 0 & I_n & 0 \end{bmatrix}
$$

**Update step.** The standard Kalman update is:

$$
y = z - H \hat{x}_{k|k-1}
$$

$$
S = H P_{k|k-1} H^T + R
$$

$$
K = P_{k|k-1} H^T S^{-1}
$$

$$
\hat{x}_{k|k} = \hat{x}_{k|k-1} + K y
$$

$$
P_{k|k} = (I - K H) P_{k|k-1}
$$

where $R = \sigma_m^2 I_{2n}$ is the measurement noise covariance.

**Outlier rejection.** Before the update, the `_reject_outliers` method checks
each measurement component against the predicted value. If the residual exceeds
`outlier_threshold` standard deviations (computed from the innovation
covariance), the measurement is replaced with the predicted value:

$$
\text{if } |z_i - \hat{z}_i| > \kappa \, \sigma_i \implies z_i \leftarrow \hat{z}_i
$$

where $\kappa$ is the outlier threshold (default 3.0).

**Velocity filtering.** After the Kalman update, an additional exponential
smoothing filter is applied to the velocity estimate if
`use_velocity_filter=True`:

$$
\hat{\dot{q}}_{filtered} = \alpha \, \hat{\dot{q}}_{KF} + (1 - \alpha) \, \hat{\dot{q}}_{prev}
$$

where $\alpha$ is `velocity_filter_alpha`. This provides additional smoothing
beyond what the Kalman filter achieves.

**API methods:**

```python
estimator = StateEstimator(n_dof=7)

# Update with new measurement
result = estimator.update(robot_state, dt=0.001)
# result = {"position": ..., "velocity": ..., "acceleration": ...}

# Query current estimates
q = estimator.get_position()
qd = estimator.get_velocity()
qdd = estimator.get_acceleration()

# Get uncertainty
pos_std = estimator.get_position_uncertainty()
vel_std = estimator.get_velocity_uncertainty()

# Predict future state
predicted = estimator.predict(dt=0.01, control=qdd_desired)

# Reset
estimator.reset(position=q0, velocity=v0)
```

---

## 20.3 Safety System

The safety system provides real-time monitoring and enforcement of safety
constraints, following guidelines from ISO 10218-1 (industrial robots) and
ISO/TS 15066 (collaborative robots).

### 20.3.1 SafetyLimits

The `SafetyLimits` dataclass defines the safety boundaries:

| Field                    | Default  | Description                               |
| ------------------------ | -------- | ----------------------------------------- |
| `max_joint_velocity`     | required | Per-joint velocity limits (rad/s).        |
| `max_joint_torque`       | required | Per-joint torque limits (Nm).             |
| `max_cartesian_velocity` | `0.25`   | Max EE velocity for collaborative mode.   |
| `max_cartesian_force`    | `150.0`  | Maximum contact force (N).                |
| `workspace_bounds`       | `None`   | Workspace limits [xmin, xmax, ymin, ...]. |
| `forbidden_zones`        | `[]`     | Boxes where the robot must not enter.     |
| `max_contact_force`      | `150.0`  | ISO 10218-1 contact force limit (N).      |
| `max_pressure`           | `110.0`  | ISO/TS 15066 pressure limit (N/cm^2).     |
| `joint_limits_lower`     | `None`   | Per-joint lower position limits (rad).    |
| `joint_limits_upper`     | `None`   | Per-joint upper position limits (rad).    |

The `from_config(robot_config)` class method creates `SafetyLimits` from a
`RobotConfig`, using the robot's declared limits or defaults (2.0 rad/s
velocity, 50 Nm torque).

### 20.3.2 SafetyMonitor

`SafetyMonitor` performs real-time safety checking at the control frequency.

**Construction.**

```python
from src.deployment.safety.monitor import SafetyMonitor, SafetyLimits

monitor = SafetyMonitor(
    robot_config=config,
    limits=SafetyLimits.from_config(config),
)
```

**SafetyStatusLevel enum:**

| Level       | Description                                   |
| ----------- | --------------------------------------------- |
| `OK`        | All checks passed, operation is nominal.      |
| `WARNING`   | Approaching limits, operation can continue.   |
| `ERROR`     | Limits violated, commands should be modified. |
| `EMERGENCY` | Critical violation, emergency stop required.  |

**State checking.** The `check_state(state)` method validates the current robot
state against all safety limits:

1. **Joint velocity limits:**

$$
|\dot{q}_i| \leq \dot{q}_{max,i} \quad \forall i
$$

2. **Joint torque limits:**

$$
|\tau_i| \leq \tau_{max,i} \quad \forall i
$$

3. **Joint position limits (lower):**

$$
q_i \geq q_{min,i} \quad \forall i
$$

4. **Joint position limits (upper):**

$$
q_i \leq q_{max,i} \quad \forall i
$$

5. **Approach warnings.** If any joint is within 0.1 rad of its upper limit,
   a warning is issued.

6. **Emergency stop.** If the emergency stop is active, a violation is reported.

The method returns a `SafetyStatus` with the overall level, a boolean `is_safe`
flag, lists of violations and warnings, and the current `speed_override`.

**Command checking.** The `check_command(command)` method validates a control
command before it is sent to the robot:

- Torque commands are checked against `max_joint_torque`.
- Position targets are checked against joint limits.

**Safe command computation.** The `compute_safe_command(desired, state)` method
modifies a command to ensure it satisfies all safety limits:

1. Apply speed override: scale velocity and torque commands by the speed factor.
2. Clip torque commands to $[-\tau_{max}, \tau_{max}]$.
3. Clip position targets to $[q_{min}, q_{max}]$.

This method is designed to be called in the control loop between the policy
output and the hardware command, providing a safety layer that cannot be
bypassed by the policy.

**Stopping distance.** The `get_stopping_distance(state, body)` method estimates
the minimum distance needed to stop:

$$
d_{stop} = \frac{v_{max}^2}{2 \, a_{max}}
$$

where $v_{max}$ is the maximum joint velocity magnitude and $a_{max} = 2.0$ m/s^2
is a conservative deceleration estimate.

**Speed override.** The `set_speed_override(factor)` method sets a global speed
reduction factor in $[0, 1]$ that scales all velocity and torque commands.

**Human proximity.** When `set_human_nearby(True)` is called, the speed override
is automatically capped at 0.5 (50% speed), per collaborative robot safety
guidelines.

**Emergency stop.** The emergency stop cascade is:

```python
# Detection
monitor.trigger_emergency_stop()
# This sets speed_override to 0.0 and emergency_stop flag to True

# All subsequent check_state() calls will report a violation
# compute_safe_command() will output zero commands due to speed_override = 0.0

# Recovery (manual reset required)
monitor.clear_emergency_stop()
# Restores speed_override to 1.0
```

### 20.3.3 CollisionAvoidance

The `CollisionAvoidance` class in `src/deployment/safety/collision.py`
implements real-time collision checking using artificial potential fields.

**Obstacle types.** The `ObstacleType` enum defines supported obstacle
geometries:

| Type       | Description                        | Dimensions               |
| ---------- | ---------------------------------- | ------------------------ |
| `BOX`      | Axis-aligned box.                  | `[width, depth, height]` |
| `SPHERE`   | Sphere.                            | `[radius]`               |
| `CYLINDER` | Z-axis aligned cylinder.           | `[radius, height]`       |
| `HUMAN`    | Human bounding box (extra safety). | `[width, depth, height]` |
| `DYNAMIC`  | Moving obstacle with velocity.     | (geometry-dependent)     |

**Obstacle dataclass.** Each obstacle has:

- `name` -- Identifier string.
- `obstacle_type` -- Geometry type.
- `position` -- Center position [x, y, z].
- `dimensions` -- Size (type-dependent).
- `velocity` -- For dynamic obstacles.
- `inflation` -- Safety inflation radius (default 0.05 m, 0.3 m for humans).

**Signed distance computation.** The `get_distance(point)` method computes the
signed distance from a query point to the obstacle surface. Negative values
indicate the point is inside the obstacle:

- **Sphere:** $d = \|p - c\| - r - r_{inflate}$
- **Box/Human:** Clamped distance from point to box surface minus inflation.
- **Cylinder:** Combined radial and axial distance.

**Distance gradient.** The `get_gradient(point)` method computes the normalized
gradient of the distance function via central finite differences with step
$\epsilon = 10^{-6}$. The gradient points away from the obstacle.

**HumanState dataclass.** Represents a detected human with position, velocity,
bounding box, optional skeleton joints, detection confidence, and motion flag.
The `to_obstacle()` method converts to an `Obstacle` with `ObstacleType.HUMAN`
and an inflation radius of 0.3 m.

**The CollisionAvoidance class.**

```python
from src.deployment.safety.collision import CollisionAvoidance

collision = CollisionAvoidance(
    robot_model=engine,
    safety_distance=0.1,  # 10 cm minimum clearance
)
```

**Obstacle management:**

```python
collision.add_obstacle(obstacle)
collision.remove_obstacle("obstacle_name")
collision.clear_obstacles()
collision.update_human_position(human_state)
```

**Repulsive potential field.** The `compute_repulsive_field(state)` method
computes artificial potential field repulsive forces in joint space:

For each robot link and each obstacle, if the distance is less than
`_repulsion_distance` (default 0.5 m):

- If inside the obstacle ($d \leq 0$): maximum repulsion force.
- If outside but within range: inverse-square repulsion:

$$
F_{rep} = k_{rep} \left(\frac{1}{d} - \frac{1}{d_{max}}\right) \frac{1}{d^2}
$$

where $k_{rep}$ is `_repulsion_gain` (default 1.0) and $d_{max}$ is
`_repulsion_distance`. The magnitude is capped at `_max_repulsion` (default 10.0).

The Cartesian repulsion force is mapped to joint space and averaged over all
links.

**Path clearance checking.** The `check_path_clearance(trajectory, min_distance)`
method verifies that an entire trajectory maintains sufficient clearance:

```python
is_clear, min_dist = collision.check_path_clearance(
    trajectory=joint_trajectory,  # Shape (n_steps, n_joints)
    min_distance=0.1,
)
```

Returns `(True, min_distance_found)` if all waypoints are clear, or
`(False, min_distance_found)` on the first violation.

**Velocity scaling.** The `get_safe_velocity_scaling(state)` method computes a
scaling factor in $[0, 1]$ based on proximity to obstacles:

- Distance $\leq 0$ (inside obstacle): scale = 0.0
- Distance $< d_{safety}$: scale = $d / d_{safety}$
- Distance $< d_{repulsion}$: scale = $0.5 + 0.5 \cdot \frac{d - d_{safety}}{d_{repulsion} - d_{safety}}$
- Distance $\geq d_{repulsion}$: scale = 1.0

**Minimum distance query.** The `get_minimum_distance(state)` method returns the
closest distance to any obstacle from any robot link.

### 20.3.4 Integrated Safety Pipeline

A complete safety-aware control loop integrates all safety components:

```python
controller = RealTimeController(control_frequency=1000.0)
monitor = SafetyMonitor(robot_config, limits)
collision = CollisionAvoidance(engine, safety_distance=0.1)
twin = DigitalTwin(engine, controller)

def safe_control_callback(state: RobotState) -> ControlCommand:
    # 1. Update state estimator
    estimated = twin._state_estimator.update(state)

    # 2. Check current state safety
    status = monitor.check_state(state)
    if status.level == SafetyStatusLevel.EMERGENCY:
        return ControlCommand.torque_command(state.timestamp, np.zeros(7))

    # 3. Compute desired command from policy
    obs = build_observation(estimated)
    desired_action = policy(obs)
    desired_cmd = ControlCommand.torque_command(state.timestamp, desired_action)

    # 4. Add collision avoidance
    repulsion = collision.compute_repulsive_field(state)
    desired_cmd.torque_commands += repulsion

    # 5. Apply velocity scaling
    vel_scale = collision.get_safe_velocity_scaling(state)
    monitor.set_speed_override(vel_scale)

    # 6. Enforce safety limits
    safe_cmd = monitor.compute_safe_command(desired_cmd, state)

    # 7. Check for anomalies
    anomaly = twin.detect_anomaly()
    if anomaly and anomaly.severity > 0.8:
        monitor.trigger_emergency_stop()

    return safe_cmd
```

---

## 20.4 Teleoperation

The teleoperation subsystem enables remote control of robots through various
input devices, with support for demonstration recording.

### 20.4.1 Input Device Protocol

The `InputDevice` protocol in `src/deployment/teleoperation/devices.py` defines
the interface that all input devices must implement:

| Method                 | Return type       | Description                          |
| ---------------------- | ----------------- | ------------------------------------ |
| `get_pose()`           | `NDArray` (7,)    | Device pose [x,y,z,qw,qx,qy,qz].     |
| `get_twist()`          | `NDArray` (6,)    | Device velocity [vx,vy,vz,wx,wy,wz]. |
| `get_gripper_state()`  | `float`           | Gripper openness (0=closed, 1=open). |
| `set_force_feedback()` | `None`            | Send haptic feedback wrench.         |
| `get_buttons()`        | `dict[str, bool]` | Button press states.                 |

All devices also support `connect()`, `disconnect()`, and `update()` for
lifecycle management.

### 20.4.2 Supported Devices

Four device implementations are provided:

**KeyboardMouseInput.** A fallback device for testing without specialized
hardware. Keyboard keys map to discrete velocity commands:

| Key        | Direction | Velocity |
| ---------- | --------- | -------- |
| `forward`  | +x        | 0.1 m/s  |
| `backward` | -x        | 0.1 m/s  |
| `left`     | +y        | 0.1 m/s  |
| `right`    | -y        | 0.1 m/s  |
| `up`       | +z        | 0.1 m/s  |
| `down`     | -z        | 0.1 m/s  |

Gripper is controlled by `open_gripper` and `close_gripper` keys. Key states
are set externally via `set_key_state(key, pressed)`.

**SpaceMouseInput.** Interface for 3Dconnexion SpaceMouse 6-DOF controllers.
The class exposes the intended contract and checks for optional SpaceMouse
backend packages, but it does not claim a live connection until a hardware
backend is implemented. Use `KeyboardMouseInput` as the software-only fallback
for testing without specialized hardware. Tracked by GitHub issue #7360.

**VRControllerInput.** Interface for VR controllers (Oculus Touch, Vive Wands,
etc.). Supports left or right hand configuration and multiple tracking systems
(SteamVR, Oculus) at the API boundary. It reports unavailable until a real VR
runtime backend is implemented; it does not synthesize live controller pose.
Tracked by GitHub issue #7360.

**HapticDeviceInput.** Interface for force-feedback devices (Phantom Omni,
Force Dimension Sigma.7, etc.). The class defines the desired API and force
limits, but hardware polling and force output remain unavailable until a real
device SDK backend is implemented. Disconnected reads and feedback writes raise
a state error rather than returning frozen identity data. Tracked by GitHub
issue #7360.

### 20.4.3 TeleoperationMode

The `TeleoperationMode` enum defines how device input is mapped to robot
commands:

| Mode        | Description                                       |
| ----------- | ------------------------------------------------- |
| `POSITION`  | Device pose maps to robot end-effector position.  |
| `VELOCITY`  | Device twist maps to robot end-effector velocity. |
| `WRENCH`    | Device twist interpreted as force/torque command. |
| `IMPEDANCE` | Position control with compliance.                 |

### 20.4.4 WorkspaceMapping

The `WorkspaceMapping` dataclass configures the coordinate transformation
between device and robot:

| Field            | Default  | Description                             |
| ---------------- | -------- | --------------------------------------- |
| `leader_frame`   | `eye(4)` | Device reference frame (4x4 transform). |
| `follower_frame` | `eye(4)` | Robot reference frame (4x4 transform).  |
| `position_scale` | `1.0`    | Position scaling factor.                |
| `rotation_scale` | `1.0`    | Rotation scaling factor.                |
| `deadband`       | `0.001`  | Motion deadband (meters).               |
| `rate_limit`     | `0.5`    | Maximum velocity (m/s).                 |

### 20.4.5 The TeleoperationInterface Class

`TeleoperationInterface` is the main class for teleoperation. It bridges input
devices and robot control.

**Construction.**

```python
from src.deployment.teleoperation.interface import TeleoperationInterface
from src.deployment.teleoperation.devices import SpaceMouseInput

device = SpaceMouseInput()
device.connect()

teleop = TeleoperationInterface(
    robot=engine,
    input_device=device,
)
teleop.set_control_mode(TeleoperationMode.POSITION)
```

**Clutch mechanism.** The clutch allows the operator to reposition the input
device without moving the robot:

```python
teleop.engage_clutch()    # Motion enabled
teleop.disengage_clutch() # Motion disabled, robot holds position
```

When the clutch is engaged, the current device pose is stored as the reference.
Subsequent device motion is computed relative to this reference. When the clutch
is disengaged, all commands are zero (the robot holds its current position).

Button 1 engages the clutch; button 2 disengages it.

**The `update` method.** This is the main method called at control frequency:

```python
command = teleop.update()
```

It performs the following sequence:

1. Read device pose, twist, gripper state, and buttons.
2. Handle clutch state from button input.
3. Dispatch to mode-specific command computation.
4. Return the `ControlCommand`.

**Position mode.** Computes position delta from the reference pose, applies
scaling and deadband, adds to current robot EE position, solves inverse
kinematics, and returns joint position targets:

$$
\Delta p = (p_{device} - p_{ref}) \cdot s
$$

$$
p_{target} = p_{ee} + \Delta p
$$

$$
q_{target} = \text{IK}(p_{target})
$$

The reference pose is updated after each command to provide relative (incremental)
control.

**Velocity mode.** Scales the device twist, applies rate limiting, and maps
to joint velocities using the Jacobian pseudo-inverse:

$$
\dot{q}_{target} = J^{\dagger} \cdot (v_{device} \cdot s)
$$

where $J^{\dagger} = J^T(JJ^T)^{-1}$ is the Moore-Penrose pseudo-inverse.

**Wrench mode.** Interprets the device twist as a desired end-effector wrench
(scaled by 10x) and maps to joint torques using the Jacobian transpose:

$$
\tau = J^T \cdot (v_{device} \cdot 10)
$$

**Impedance mode.** Combines position targeting with compliance parameters:
position targets from IK, with stiffness of 100 N/rad and damping of 10 Ns/rad
per joint.

### 20.4.6 Haptic Feedback

The `get_haptic_feedback` method computes force feedback for haptic devices:

```python
wrench = teleop.get_haptic_feedback()
device.set_force_feedback(wrench)
```

If the robot has contact force sensors, the measured forces are scaled down
by 0.1 and returned as a 6D wrench.

### 20.4.7 Demonstration Recording

The teleoperation interface integrates with the imitation learning module for
recording expert demonstrations:

```python
# Start recording
teleop.start_demonstration_recording()

# Operate the robot...
for step in range(1000):
    command = teleop.update()
    controller.send_command(command)
    state = controller.get_last_state()

    # Record state at each step
    teleop.record_state(
        joint_positions=state.joint_positions,
        joint_velocities=state.joint_velocities,
        action=command.torque_commands,
    )

# Stop and retrieve demonstration
demo = teleop.stop_demonstration_recording()
# demo is a Demonstration object ready for the imitation learning pipeline
```

The recording stores timestamps (relative to recording start), joint positions,
joint velocities, and actions. The returned `Demonstration` object has
`source="teleoperation"` and `success=True`.

### 20.4.8 Complete Teleoperation Pipeline

A full teleoperation deployment integrating all subsystems:

```python
from src.deployment.realtime.controller import RealTimeController, RobotConfig
from src.deployment.safety.monitor import SafetyMonitor
from src.deployment.safety.collision import CollisionAvoidance, HumanState
from src.deployment.teleoperation.interface import TeleoperationInterface
from src.deployment.teleoperation.devices import HapticDeviceInput

# Setup
config = RobotConfig(name="franka", n_joints=7)
controller = RealTimeController(control_frequency=1000.0)
controller.connect(config)

monitor = SafetyMonitor(config)
collision = CollisionAvoidance(engine, safety_distance=0.15)

device = HapticDeviceInput(device_name="sigma7")
device.connect()
teleop = TeleoperationInterface(engine, device)

# Control loop callback
def teleop_callback(state: RobotState) -> ControlCommand:
    # Update safety
    safety = monitor.check_state(state)
    if not safety.is_safe:
        return ControlCommand.torque_command(state.timestamp, np.zeros(7))

    # Get teleop command
    cmd = teleop.update()

    # Enforce safety
    safe_cmd = monitor.compute_safe_command(cmd, state)

    # Send haptic feedback
    feedback = teleop.get_haptic_feedback()
    device.set_force_feedback(feedback)

    # Record if active
    if teleop.is_recording:
        teleop.record_state(
            state.joint_positions,
            state.joint_velocities,
            safe_cmd.torque_commands,
        )

    return safe_cmd

controller.set_control_callback(teleop_callback)
controller.start()

# Record a demonstration
teleop.start_demonstration_recording()
input("Press Enter to stop recording...")
demo = teleop.stop_demonstration_recording()

# Save demonstration
from src.learning.imitation.dataset import DemonstrationDataset
dataset = DemonstrationDataset([demo])
dataset.save("teleop_demo.json")

# Cleanup
controller.stop()
controller.disconnect()
device.disconnect()
```

---

## Summary

Chapters 17 through 20 cover the complete machine learning and deployment
pipeline in UpstreamDrift:

- **Chapter 17** describes the reinforcement learning framework, including
  the configuration system (observation, action, reward, and task configs),
  the Gymnasium-compatible base environment, and concrete environments for
  humanoid locomotion/balance and manipulation tasks.

- **Chapter 18** covers imitation learning from demonstrations, including
  the demonstration dataset format, behavior cloning with NumPy-based MLPs,
  DAgger for addressing distribution shift, and GAIL for adversarial imitation.

- **Chapter 19** addresses sim-to-real transfer through domain randomization
  (randomizing masses, friction, delays, noise, and other parameters) and
  system identification (coordinate descent optimization to match simulated
  trajectories to real-robot data).

- **Chapter 20** presents the deployment infrastructure: real-time control
  with hardware interfaces, digital twin synchronization with Kalman filtering,
  comprehensive safety monitoring with collision avoidance, and teleoperation
  with multiple input device types and demonstration recording.

Together, these modules provide a path from initial policy training in
simulation to safe deployment on physical robot hardware.

---

# UpstreamDrift User Manual -- Part 6

# Research Modules, Visualization, REST API, Motion Retargeting, and Tools

---
