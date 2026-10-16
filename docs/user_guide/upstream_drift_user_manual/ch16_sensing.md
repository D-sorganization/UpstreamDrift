# Chapter 16: Sensing

The sensing module (`src/robotics/sensing/`) provides configurable sensor
simulations for realistic noise modeling. This is critical for sim-to-real
transfer: controllers developed in simulation must be robust to the noise
characteristics of physical sensors.

---

## 16.1 Noise Models

The noise model framework (`src/robotics/sensing/noise_models.py`) defines an
abstract `NoiseModel` base class with two methods:

- `apply(signal) -> noisy_signal` -- transforms a clean signal array into a
  noisy one.
- `reset()` -- resets any internal state (e.g., accumulated bias).

All noise models preserve the shape of the input: if `signal.shape == (3,)`,
the output also has shape `(3,)`.

### 16.1.1 GaussianNoise

Additive white Gaussian noise:

$$
y = x + \mathcal{N}(\mu, \sigma^2)
$$

| Parameter | Type          | Default | Description                      |
| --------- | ------------- | ------- | -------------------------------- |
| `std`     | `float`       | `0.01`  | Standard deviation $\sigma$.     |
| `mean`    | `float`       | `0.0`   | Bias $\mu$.                      |
| `seed`    | `int or None` | `None`  | Random seed for reproducibility. |

The noise is drawn independently for each element of the signal array using
NumPy's `default_rng`. Calling `reset()` re-seeds the generator, producing the
same noise sequence again.

```python
from src.robotics.sensing.noise_models import GaussianNoise

noise = GaussianNoise(std=0.1, mean=0.0, seed=42)
clean = np.array([1.0, 2.0, 3.0])
noisy = noise.apply(clean)  # each element perturbed by N(0, 0.01)
```

### 16.1.2 BrownianNoise (Bias Drift)

Models slowly-varying sensor bias that accumulates over time via a random walk.
This is the dominant error source in MEMS gyroscopes and accelerometers over
long time horizons.

**Update equation:**

$$
b_{t+1} = \text{clip}\bigl(b_t + \mathcal{N}(0, \sigma_{\text{drift}}^2),\; [-b_{\max}, b_{\max}]\bigr)
$$

$$
y = x + b_t
$$

| Parameter      | Type          | Default | Description                                     |
| -------------- | ------------- | ------- | ----------------------------------------------- |
| `drift_rate`   | `float`       | `0.001` | Std of drift increment $\sigma_{\text{drift}}$. |
| `initial_bias` | `float`       | `0.0`   | Starting bias value.                            |
| `max_bias`     | `float`       | `1.0`   | Clipping bounds $b_{\max}$.                     |
| `seed`         | `int or None` | `None`  | Random seed.                                    |

The bias is a scalar that applies uniformly to all elements of the signal. The
clipping prevents unbounded growth. The `current_bias` property provides
read access to the current bias state for monitoring.

```python
from src.robotics.sensing.noise_models import BrownianNoise

drift = BrownianNoise(drift_rate=0.0001, max_bias=0.05, seed=42)
# After many calls to drift.apply(...), the bias will random-walk
# within [-0.05, 0.05]
```

### 16.1.3 QuantizationNoise

Models the discrete nature of analog-to-digital conversion:

$$
y = \text{round}\!\left(\frac{x - x_0}{r}\right) \cdot r + x_0
$$

where $r$ is the quantization resolution (least significant bit value) and $x_0$
is a DC offset.

| Parameter    | Type    | Default | Description                 |
| ------------ | ------- | ------- | --------------------------- |
| `resolution` | `float` | `0.001` | Quantization step size $r$. |
| `offset`     | `float` | `0.0`   | Offset before quantization. |

This noise model is stateless -- `reset()` is a no-op.

For a 16-bit ADC with a full-scale range of $R$, the resolution is:

$$
r = \frac{R}{2^{16}} = \frac{R}{65536}
$$

```python
from src.robotics.sensing.noise_models import QuantizationNoise

quant = QuantizationNoise(resolution=0.001)
clean = np.array([1.2345, 2.3456])
quantized = quant.apply(clean)  # [1.235, 2.346] approximately
```

### 16.1.4 BandwidthLimitedNoise

Models the finite bandwidth of physical sensors using a first-order IIR
(infinite impulse response) low-pass filter:

$$
y_t = \alpha x_t + (1 - \alpha) y_{t-1}
$$

where the filter coefficient is derived from the cutoff frequency:

$$
\tau = \frac{1}{2\pi f_c}, \qquad
\alpha = \frac{\Delta t}{\tau + \Delta t}
$$

Here $f_c$ is the cutoff frequency, $\Delta t = 1/f_s$ is the sample period,
and $\tau$ is the time constant.

| Parameter          | Type    | Default  | Description                           |
| ------------------ | ------- | -------- | ------------------------------------- |
| `cutoff_frequency` | `float` | `100.0`  | Filter cutoff frequency $f_c$ [Hz].   |
| `sample_rate`      | `float` | `1000.0` | Sampling rate $f_s$ [Hz].             |
| `order`            | `int`   | `2`      | Filter order (currently first-order). |

The filter state (`_filter_state`) is initialized to the first input signal and
accumulates thereafter. Calling `reset()` clears the state.

**Frequency response.** For a first-order IIR filter, the magnitude response at
frequency $f$ is:

$$
|H(f)| = \frac{\alpha}{\sqrt{1 - 2(1-\alpha)\cos(2\pi f / f_s) + (1-\alpha)^2}}
$$

At the cutoff frequency, the attenuation is approximately $-3$ dB.

```python
from src.robotics.sensing.noise_models import BandwidthLimitedNoise

bw_filter = BandwidthLimitedNoise(cutoff_frequency=50.0, sample_rate=1000.0)
for t in range(1000):
    signal = np.array([np.sin(2 * np.pi * 200 * t / 1000)])  # 200 Hz
    filtered = bw_filter.apply(signal)
    # High-frequency content will be attenuated
```

### 16.1.5 CompositeNoise

Chains multiple noise models in sequence:

$$
y = N_k\bigl(\cdots N_2\bigl(N_1(x)\bigr)\cdots\bigr)
$$

| Parameter | Type               | Default | Description            |
| --------- | ------------------ | ------- | ---------------------- |
| `models`  | `list[NoiseModel]` | `[]`    | Noise models in order. |

The `apply` method iterates through `models` in order, feeding each output into
the next model's input. The `reset` method resets all constituent models.

The `add_model(model)` method appends a noise model to the chain.

```python
from src.robotics.sensing.noise_models import CompositeNoise, GaussianNoise, BrownianNoise

composite = CompositeNoise(models=[
    BrownianNoise(drift_rate=0.0001),
    GaussianNoise(std=0.01),
])
noisy = composite.apply(clean_signal)
```

### 16.1.6 Factory: `create_realistic_sensor_noise()`

`create_realistic_sensor_noise(noise_std, bias_drift_rate, quantization_bits,
signal_range, seed)` creates a `CompositeNoise` with three stages:

1. **BrownianNoise** -- bias drift with rate `bias_drift_rate`.
2. **GaussianNoise** -- white noise with std `noise_std`.
3. **QuantizationNoise** -- resolution $r = R / 2^b$ where $R$ is `signal_range`
   and $b$ is `quantization_bits`.

---

## 16.2 IMU Sensor

The `IMUSensor` class (`src/robotics/sensing/imu_sensor.py`) simulates a 6-axis
inertial measurement unit with 3-axis accelerometer and 3-axis gyroscope.

### 16.2.1 IMUSensorConfig

| Field              | Type                    | Default         | Description                                            |
| ------------------ | ----------------------- | --------------- | ------------------------------------------------------ |
| `sensor_id`        | `str`                   | `"imu"`         | Unique identifier.                                     |
| `accel_range`      | `float`                 | `160.0`         | Max measurable acceleration [m/s^2] (~16g).            |
| `gyro_range`       | `float`                 | `35.0`          | Max measurable angular velocity [rad/s] (~2000 deg/s). |
| `accel_noise_std`  | `float`                 | `0.01`          | Accelerometer white noise std [m/s^2].                 |
| `gyro_noise_std`   | `float`                 | `0.001`         | Gyroscope white noise std [rad/s].                     |
| `accel_bias_drift` | `float`                 | `0.0001`        | Accelerometer bias drift rate [m/s^2/step].            |
| `gyro_bias_drift`  | `float`                 | `0.00001`       | Gyroscope bias drift rate [rad/s/step].                |
| `gravity`          | `NDArray[float64]` (3,) | `[0, 0, -9.81]` | Gravity vector in world frame.                         |
| `cutoff_frequency` | `float`                 | `200.0`         | Sensor bandwidth [Hz].                                 |
| `sample_rate`      | `float`                 | `1000.0`        | Sampling rate [Hz].                                    |
| `seed`             | `int or None`           | `None`          | Random seed.                                           |

### 16.2.2 Noise Pipeline

Each measurement channel (accelerometer and gyroscope) passes through a
composite noise model containing:

1. **BrownianNoise** -- bias drift with `max_bias = noise_std * 10`.
2. **GaussianNoise** -- white noise.

After the noise stage, a `BandwidthLimitedNoise` filter is applied to model the
sensor's finite bandwidth. Finally, the result is clipped to the sensor's
measurement range:

$$
y_{\text{accel}} = \text{clip}\bigl(\text{filter}(\text{noise}(a_{\text{true}})),\; -a_{\max},\; a_{\max}\bigr)
$$

### 16.2.3 Reading the Sensor

```python
reading = imu.read(
    linear_accel=np.array([0.0, 0.0, 9.81]),
    angular_vel=np.array([0.0, 0.0, 0.1]),
    timestamp=0.001,
    include_orientation=True,
)
# reading.linear_acceleration  -- noisy accel (3,)
# reading.angular_velocity     -- noisy gyro (3,)
# reading.orientation          -- quaternion (4,) or None
```

The `read` method:

1. Validates that inputs are 3-vectors.
2. Applies noise to acceleration and angular velocity independently.
3. Applies bandwidth filters.
4. Clips to sensor ranges.
5. If `include_orientation=True` and a previous timestamp exists, integrates
   the gyroscope to update the orientation estimate.

### 16.2.4 Quaternion Integration

The orientation is updated by integrating the angular velocity using first-order
quaternion kinematics. Given angular velocity $\boldsymbol{\omega}$ and timestep
$\Delta t$:

1. Compute the rotation angle $\theta = \|\boldsymbol{\omega}\| \cdot \Delta t$
   and the rotation axis $\hat{\boldsymbol{\omega}} = \boldsymbol{\omega} / \|\boldsymbol{\omega}\|$.

2. Form the incremental rotation quaternion:

$$
\Delta\mathbf{q} = \left[\cos\!\left(\frac{\theta}{2}\right),\;
  \sin\!\left(\frac{\theta}{2}\right) \hat{\boldsymbol{\omega}}\right]
$$

3. Update via quaternion multiplication:

$$
\mathbf{q}_{\text{new}} = \mathbf{q}_{\text{old}} \otimes \Delta\mathbf{q}
$$

4. Normalize to maintain unit length:

$$
\mathbf{q}_{\text{new}} \leftarrow \frac{\mathbf{q}_{\text{new}}}{\|\mathbf{q}_{\text{new}}\|}
$$

The quaternion multiplication $\mathbf{q}_1 \otimes \mathbf{q}_2$ with
$\mathbf{q}_i = [w_i, x_i, y_i, z_i]$ is computed via the Hamilton product:

$$
\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix}
w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \\
w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \\
w_1 y_2 - x_1 z_2 + y_1 w_2 + z_1 x_2 \\
w_1 z_2 + x_1 y_2 - y_1 x_2 + z_1 w_2
\end{bmatrix}
$$

If $\|\boldsymbol{\omega}\| < 10^{-10}$, the integration is skipped to avoid
division by zero.

### 16.2.5 Gravity in Sensor Frame

`get_gravity_in_sensor_frame()` transforms the world-frame gravity vector into
the current sensor frame using the orientation quaternion:

$$
\mathbf{g}' = \mathbf{q}^{-1} \otimes [0, \mathbf{g}] \otimes \mathbf{q}
$$

where $\mathbf{q}^{-1} = [w, -x, -y, -z]$ is the quaternion conjugate (inverse
for unit quaternions). The operation `_rotate_vector_by_quaternion(v, q)` wraps
the 3-vector as a pure quaternion $[0, v_x, v_y, v_z]$, performs the
sandwich product $q \otimes v \otimes q^{-1}$, and extracts the vector part.

This is essential for separating gravitational acceleration from linear
acceleration in the sensor reading:

$$
a_{\text{measured}} = a_{\text{linear}} + \mathbf{g}'
$$

### 16.2.6 Quality Presets

`create_realistic_imu(sensor_id, quality, seed)` provides three quality levels:

| Quality        | `accel_noise_std` | `gyro_noise_std` | `accel_bias_drift` | `gyro_bias_drift` |
| -------------- | ----------------- | ---------------- | ------------------ | ----------------- |
| `"mems"`       | 0.1               | 0.01             | 0.001              | 0.0001            |
| `"industrial"` | 0.01              | 0.001            | 0.0001             | 0.00001           |
| `"tactical"`   | 0.001             | 0.0001           | 0.00001            | 0.000001          |

Each tier represents roughly an order-of-magnitude improvement in all noise
parameters, reflecting the progression from consumer MEMS devices through
industrial units to tactical/navigation-grade sensors.

`create_ideal_imu(sensor_id)` sets all noise parameters to zero, producing a
perfect sensor useful for debugging control algorithms in isolation.

### 16.2.7 Sensor Reset and State Management

- `reset()` -- resets all noise models, filters, orientation (to identity
  quaternion), and timestamp.
- `set_orientation(quaternion)` -- overrides the current orientation estimate.
  The quaternion is normalized after assignment.

---

## 16.3 Force-Torque Sensor

The `ForceTorqueSensor` class (`src/robotics/sensing/force_torque_sensor.py`)
simulates a 6-axis force/torque transducer that measures three force components
and three torque components.

### 16.3.1 ForceTorqueSensorConfig

| Field               | Type          | Default       | Description                  |
| ------------------- | ------------- | ------------- | ---------------------------- |
| `sensor_id`         | `str`         | `"ft_sensor"` | Unique identifier.           |
| `force_range`       | `float`       | `1000.0`      | Max measurable force [N].    |
| `torque_range`      | `float`       | `100.0`       | Max measurable torque [Nm].  |
| `force_noise_std`   | `float`       | `0.1`         | Force noise std [N].         |
| `torque_noise_std`  | `float`       | `0.01`        | Torque noise std [Nm].       |
| `force_bias_drift`  | `float`       | `0.001`       | Force bias drift [N/step].   |
| `torque_bias_drift` | `float`       | `0.0001`      | Torque bias drift [Nm/step]. |
| `cutoff_frequency`  | `float`       | `100.0`       | Sensor bandwidth [Hz].       |
| `sample_rate`       | `float`       | `1000.0`      | Sampling rate [Hz].          |
| `seed`              | `int or None` | `None`        | Random seed.                 |

### 16.3.2 Noise Pipeline

The force and torque channels have independent noise models, each containing:

1. **BrownianNoise** -- bias drift with `max_bias = noise_std * 10`.
2. **GaussianNoise** -- white noise.

After applying noise independently to the force (first 3 components) and torque
(last 3 components), the full 6D wrench is passed through a single
`BandwidthLimitedNoise` filter. A tare offset is then subtracted, and the result
is clipped to the sensor's measurement range:

$$
y_{\text{force}} = \text{clip}\bigl(w_f - \text{tare}_f,\; -F_{\max},\; F_{\max}\bigr)
$$

$$
y_{\text{torque}} = \text{clip}\bigl(w_\tau - \text{tare}_\tau,\; -\tau_{\max},\; \tau_{\max}\bigr)
$$

### 16.3.3 Reading the Sensor

```python
reading = sensor.read(
    true_wrench=np.array([10.0, 0.0, 50.0, 0.0, 0.0, 0.0]),
    timestamp=0.001,
)
# reading.wrench  -- noisy 6D wrench
# reading.force   -- first 3 components
# reading.torque  -- last 3 components
```

The `read_raw` method applies noise but skips the bandwidth filter, useful for
diagnostic purposes.

### 16.3.4 Tare Offset Handling

The `tare(current_wrench=None)` method zeros the sensor by recording the current
reading as a bias offset. Subsequent measurements subtract this offset:

```python
# Zero the sensor under no-load conditions
sensor.tare(current_wrench=np.zeros(6))

# Or tare using the last reading
sensor.tare()  # uses self._last_reading
```

This models the common laboratory practice of "taring" or "zeroing" a
force/torque sensor before use.

### 16.3.5 Contact Location Estimation

`estimate_contact_location(wrench)` uses the measured wrench to estimate the
point of application of a single contact force:

$$
\mathbf{r} = \frac{\mathbf{f} \times \boldsymbol{\tau}}{\|\mathbf{f}\|^2}
$$

**Derivation.** For a single point contact at position $\mathbf{r}$ applying
force $\mathbf{f}$, the resulting torque is:

$$
\boldsymbol{\tau} = \mathbf{r} \times \mathbf{f}
$$

Taking the cross product of $\mathbf{f}$ with both sides:

$$
\mathbf{f} \times \boldsymbol{\tau} = \mathbf{f} \times (\mathbf{r} \times \mathbf{f})
= \mathbf{r} (\mathbf{f} \cdot \mathbf{f}) - \mathbf{f} (\mathbf{f} \cdot \mathbf{r})
= \|\mathbf{f}\|^2 \mathbf{r} - (\mathbf{f} \cdot \mathbf{r}) \mathbf{f}
$$

The formula $\mathbf{r} = (\mathbf{f} \times \boldsymbol{\tau}) / \|\mathbf{f}\|^2$
gives the component of $\mathbf{r}$ perpendicular to $\mathbf{f}$. This is the
unique minimum-distance solution for $\mathbf{r}$ consistent with the measured
wrench.

The method returns `None` if the force magnitude is below $10^{-6}$ N (too small
for reliable estimation).

```python
wrench = np.array([0.0, 0.0, 10.0, 0.5, -0.3, 0.0])
location = sensor.estimate_contact_location(wrench)
# location gives approximate contact point relative to sensor frame
```

### 16.3.6 Quality Presets

`create_realistic_sensor(sensor_id, quality, seed)` provides three quality levels:

| Quality        | `force_noise_std` | `torque_noise_std` | `force_bias_drift` | `torque_bias_drift` |
| -------------- | ----------------- | ------------------ | ------------------ | ------------------- |
| `"research"`   | 0.01              | 0.001              | 0.0001             | 0.00001             |
| `"industrial"` | 0.1               | 0.01               | 0.001              | 0.0001              |
| `"consumer"`   | 1.0               | 0.1                | 0.01               | 0.001               |

Research-grade sensors (e.g., ATI Nano17) achieve sub-Newton force resolution
and are used in precision manipulation studies. Industrial sensors are suitable
for most factory automation. Consumer-grade sensors offer wider measurement
ranges at the cost of increased noise.

`create_ideal_sensor(sensor_id)` sets all noise parameters to zero.

### 16.3.7 Sensor Reset

`reset()` clears all internal state: noise models, bandwidth filter, tare offset,
and last reading cache. This restores the sensor to its initial condition.

---

## Summary of Key Imports

For quick reference, here are the primary imports for each chapter:

**Chapter 13 -- Contact Dynamics:**

```python
from src.robotics.core.types import ContactState, ContactType
from src.robotics.contact.friction_cone import (
    FrictionCone,
    linearize_friction_cone,
    compute_friction_cone_constraint,
    project_to_friction_cone,
)
from src.robotics.contact.contact_manager import ContactManager
from src.robotics.contact.grasp_analysis import (
    compute_grasp_matrix,
    check_force_closure,
    compute_grasp_quality,
    compute_contact_wrench_cone,
    required_contact_forces,
)
```

**Chapter 14 -- Whole-Body Control:**

```python
from src.robotics.core.types import TaskPriority, ControlMode, TaskDescriptor
from src.robotics.control.whole_body.task import (
    Task, TaskType,
    create_com_task, create_posture_task,
    create_ee_task, create_contact_constraint,
    create_joint_limit_task,
)
from src.robotics.control.whole_body.qp_solver import (
    QPProblem, QPSolution, QPSolver,
    ScipyQPSolver, NullspaceQPSolver,
    create_default_solver,
)
from src.robotics.control.whole_body.wbc_controller import (
    WholeBodyController, WBCConfig, WBCSolution,
)
```

**Chapter 15 -- Locomotion:**

```python
from src.robotics.locomotion.gait_types import (
    GaitType, GaitPhase, LegState, SupportState,
    GaitParameters,
    create_walk_parameters, create_run_parameters, create_stand_parameters,
)
from src.robotics.locomotion.gait_state_machine import (
    GaitEvent, GaitState, GaitStateMachine,
)
from src.robotics.locomotion.footstep_planner import (
    Footstep, FootstepPlan, FootstepPlanner,
)
from src.robotics.locomotion.zmp_computer import (
    ZMPResult, ZMPComputer,
)
```

**Chapter 16 -- Sensing:**

```python
from src.robotics.sensing.noise_models import (
    NoiseModel, GaussianNoise, BrownianNoise,
    QuantizationNoise, BandwidthLimitedNoise,
    CompositeNoise,
    create_realistic_sensor_noise,
)
from src.robotics.sensing.imu_sensor import (
    IMUSensorConfig, IMUSensor,
    create_ideal_imu, create_realistic_imu,
)
from src.robotics.sensing.force_torque_sensor import (
    ForceTorqueSensorConfig, ForceTorqueSensor,
    create_ideal_sensor, create_realistic_sensor,
)
```

---
