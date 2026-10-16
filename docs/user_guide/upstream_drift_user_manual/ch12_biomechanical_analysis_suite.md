# Chapter 12: Biomechanical Analysis Suite

## 12.1 Overview

The Biomechanical Analysis Suite is a collection of shared analysis modules that operate
across all physics engines. These modules are implemented as Python mixins (classes designed
to be combined via multiple inheritance) and are located in
`src/shared/python/analysis/`. They compute golf-specific biomechanical metrics from
simulation data regardless of the underlying physics engine.

### Source Files

| File                                              | Purpose                        |
| ------------------------------------------------- | ------------------------------ |
| `src/shared/python/analysis/swing_metrics.py`     | Golf swing kinematics/kinetics |
| `src/shared/python/analysis/energy_metrics.py`    | Energy analysis                |
| `src/shared/python/analysis/grf_metrics.py`       | Ground reaction forces         |
| `src/shared/python/analysis/angular_momentum.py`  | Angular momentum tracking      |
| `src/shared/python/analysis/phase_detection.py`   | Golf swing phase detection     |
| `src/shared/python/analysis/stability_metrics.py` | Balance and stability          |
| `src/shared/python/analysis/dataclasses.py`       | Shared data structures         |

### Architecture

All analysis modules are implemented as mixins that expect certain attributes on the host
class:

```python
class SwingAnalyzer(
    SwingMetricsMixin,
    EnergyMetricsMixin,
    GRFMetricsMixin,
    AngularMomentumMetricsMixin,
    PhaseDetectionMixin,
    StabilityMetricsMixin,
    BasicStatsMixin,
):
    """Complete analysis class combining all mixins."""

    def __init__(self, times, joint_positions, ...):
        self.times = times
        self.joint_positions = joint_positions
        self.club_head_speed = club_head_speed
        self.dt = dt
        # ... other attributes
```

---

## 12.2 Swing Metrics (`swing_metrics.py`)

The `SwingMetricsMixin` computes golf-specific kinematics including club head speed,
X-factor, tempo, and range of motion.

### 12.2.1 Club Head Speed

The club head speed is the magnitude of the linear velocity at the club head frame:

$$v_{club} = \|\dot{p}_{head}\| = \|J_{linear}(q) \cdot \dot{q}\|$$

where $\dot{p}_{head}$ is the 3D velocity of the club head and $J_{linear}$ is the
linear component of the Jacobian at the club head frame.

Peak club head speed typically occurs at or just before impact:

```python
peak = analyzer.find_club_head_speed_peak()
# Returns PeakInfo dataclass:
#   peak.value       -- peak speed [m/s]
#   peak.time        -- time of peak [s]
#   peak.index       -- array index
#   peak.prominence  -- peak prominence [m/s]
#   peak.width       -- peak width [s]
```

### 12.2.2 X-Factor

The X-Factor measures the angular separation between the shoulder plane and hip plane
during the swing:

$$X\text{-Factor}(t) = \theta_{shoulder}(t) - \theta_{hip}(t)$$

where $\theta_{shoulder}$ and $\theta_{hip}$ are the axial rotation angles (in degrees)
of the shoulder girdle and pelvis, respectively.

**API**:

```python
def compute_x_factor(
    self,
    shoulder_joint_idx: int,
    hip_joint_idx: int,
) -> np.ndarray | None:
    """Compute X-Factor (shoulder-hip rotation difference).

    Args:
        shoulder_joint_idx: Index of shoulder/torso rotation joint
        hip_joint_idx: Index of hip rotation joint

    Returns:
        X-Factor time series (degrees) or None if indices invalid
    """
```

**Usage**:

```python
x_factor = analyzer.compute_x_factor(
    shoulder_joint_idx=3,  # Torso rotation
    hip_joint_idx=0,       # Pelvis rotation
)

if x_factor is not None:
    max_x_factor = np.max(x_factor)
    print(f"Peak X-Factor: {max_x_factor:.1f} degrees")
    # Professional golfers: 45-60 degrees at top of backswing
```

### 12.2.3 X-Factor Stretch

The X-Factor velocity (stretch rate) quantifies how rapidly the X-factor increases during
the transition from backswing to downswing:

$$\dot{X}(t) = \frac{d}{dt}\bigl[\theta_{shoulder}(t) - \theta_{hip}(t)\bigr]$$

```python
def compute_x_factor_stretch(
    self,
    shoulder_joint_idx: int,
    hip_joint_idx: int,
) -> tuple[np.ndarray, float] | None:
    """Compute X-Factor velocity (stretch rate) and peak stretch.

    Returns:
        Tuple of (x_factor_velocity_array, peak_stretch_rate) or None
    """
```

**Usage**:

```python
result = analyzer.compute_x_factor_stretch(
    shoulder_joint_idx=3,
    hip_joint_idx=0,
)
if result is not None:
    x_vel, peak_stretch = result
    print(f"Peak X-Factor stretch rate: {peak_stretch:.1f} deg/s")
```

### 12.2.4 Range of Motion

```python
def compute_range_of_motion(self, joint_idx: int) -> tuple[float, float, float]:
    """Compute range of motion for a joint.

    Args:
        joint_idx: Joint index

    Returns:
        (min_angle, max_angle, rom) in degrees
    """
```

**Usage**:

```python
min_ang, max_ang, rom = analyzer.compute_range_of_motion(joint_idx=2)
print(f"Shoulder ROM: {rom:.1f} degrees ({min_ang:.1f} to {max_ang:.1f})")
```

### 12.2.5 Swing Tempo

Tempo is the ratio of backswing to downswing duration:

$$\text{Tempo} = \frac{t_{transition} - t_{address}}{t_{impact} - t_{transition}}$$

Professional golfers typically have a 3:1 tempo ratio.

```python
def compute_tempo(self) -> tuple[float, float, float] | None:
    """Compute swing tempo (backswing:downswing ratio).

    Returns:
        (backswing_duration, downswing_duration, ratio) or None
    """
```

The transition point (top of backswing) is detected as the minimum club head speed
in the first 70% of the pre-impact interval, after Savitzky-Golay smoothing.

**Usage**:

```python
tempo = analyzer.compute_tempo()
if tempo is not None:
    bs_dur, ds_dur, ratio = tempo
    print(f"Backswing: {bs_dur:.3f}s, Downswing: {ds_dur:.3f}s, Ratio: {ratio:.2f}")
```

### 12.2.6 Smash Factor

The smash factor relates ball speed to club head speed, measuring the efficiency of
energy transfer at impact:

$$SF = \frac{v_{ball}}{v_{club}}$$

Typical values: 1.44-1.50 for drivers, decreasing with shorter clubs. A higher smash
factor indicates more efficient energy transfer.

### 12.2.7 Impact Detection

```python
def detect_impact_time(self) -> float | None:
    """Detect ball impact time.

    Uses peak club head speed as proxy for impact.

    Returns:
        Impact time in seconds, or None
    """
```

### 12.2.8 Kinematic Sequence

The kinematic sequence describes the order in which body segments reach peak angular
velocity during the downswing. The optimal sequence is proximal-to-distal:

1. **Hips** -- peak first (initiate downswing)
2. **Torso** -- peak second (transfer energy from lower body)
3. **Arms** -- peak third (amplify through lever system)
4. **Club** -- peak last (maximum speed at impact)

This is stored in the `KinematicSequenceInfo` dataclass:

```python
@dataclass
class KinematicSequenceInfo:
    segment_name: str      # e.g., "hips", "torso", "arms", "club"
    peak_velocity: float   # Peak angular velocity [deg/s]
    peak_time: float       # Time of peak [s]
    peak_index: int        # Array index of peak
    order_index: int       # Actual order (0=first to peak)
```

### 12.2.9 Attack Angle, Dynamic Loft, and Face Angle

These impact parameters characterize the club-ball interaction:

- **Attack angle**: The vertical angle of the club head path at impact.
  Negative = descending blow (irons), positive = ascending (driver).
- **Dynamic loft**: The effective loft of the club face at impact, accounting for
  shaft lean and face rotation.
- **Face angle**: The horizontal orientation of the club face relative to the target
  line at impact. Open = face pointing right of target (for right-handed golfer),
  closed = left.

---

## 12.3 Energy Metrics (`energy_metrics.py`)

The `EnergyMetricsMixin` computes energy-related quantities for the multibody system.

### 12.3.1 Kinetic Energy

The total kinetic energy of the system in generalized coordinates is:

$$T = \frac{1}{2}\,\dot{q}^T\,M(q)\,\dot{q}$$

where $M(q)$ is the joint-space inertia matrix and $\dot{q}$ is the generalized velocity
vector.

### 12.3.2 Potential Energy

The gravitational potential energy is:

$$V = -\sum_{i=1}^{n} m_i\,g^T\,p_i$$

where $m_i$ is the mass of body $i$, $g$ is the gravity vector, and $p_i$ is the position
of the center of mass of body $i$.

### 12.3.3 Total Mechanical Energy

$$E = T + V$$

For a conservative system (no friction, no external forces), $E$ should be constant.
Energy drift indicates either non-conservative effects or numerical integration errors.

### 12.3.4 Energy Transfer Efficiency

The energy transfer efficiency measures how much of the total system energy is
concentrated in the club at impact:

$$\eta = \frac{E_{club}}{E_{total}} \times 100\%$$

**API**:

```python
def compute_energy_metrics(
    self,
    kinetic_energy: np.ndarray,
    potential_energy: np.ndarray,
) -> dict[str, Any]:
    """Compute energy-related metrics.

    Args:
        kinetic_energy: Kinetic energy time series [J]
        potential_energy: Potential energy time series [J]

    Returns:
        Dictionary with keys:
          - "max_kinetic_energy": float [J]
          - "max_potential_energy": float [J]
          - "max_total_energy": float [J]
          - "energy_efficiency": float [%]
          - "energy_variation": float [J] (std of total energy)
          - "energy_drift": float [J] (E_final - E_initial)
    """
```

**Usage**:

```python
# Compute energy time series from simulation data
T_series = np.array([0.5 * v.T @ M @ v for M, v in zip(mass_matrices, velocities)])
V_series = np.array([compute_PE(q) for q in configurations])

metrics = analyzer.compute_energy_metrics(T_series, V_series)
print(f"Peak kinetic energy: {metrics['max_kinetic_energy']:.1f} J")
print(f"Energy efficiency at impact: {metrics['energy_efficiency']:.1f}%")
print(f"Energy drift: {metrics['energy_drift']:.3f} J")
```

### 12.3.5 Segmental Energy Flow

The proximal-to-distal energy flow pattern is a hallmark of efficient golf swings. Energy
generated by large proximal muscles (hips, torso) transfers through the kinematic chain
to the distal segments (arms, club). This is quantified by tracking the kinetic energy
of each segment over time and observing the sequential peaking pattern.

---

## 12.4 Ground Reaction Force Metrics (`grf_metrics.py`)

The `GRFMetricsMixin` computes ground reaction force (GRF) and centre of pressure (CoP)
metrics essential for golf swing balance analysis.

### 12.4.1 Vertical GRF Analysis

The vertical ground reaction force $F_z(t)$ characterizes the golfer's interaction with
the ground. Key metrics:

- **Peak vertical force**: Maximum $F_z$ during the swing (typically 1.0-1.5 body weight
  during the downswing)

The loading rate $\dot{F}_z = dF_z/dt$ is a common force-plate metric, but
`compute_grf_metrics` does not compute it.

### 12.4.2 Shear Forces

The horizontal (shear) GRF components generate the rotational moments needed for
the swing:

$$F_{shear} = \sqrt{F_x^2 + F_y^2}$$

### 12.4.3 Centre of Pressure

The centre of pressure (CoP) is the point on the ground plane $z = z_g$ where the
resultant GRF acts. With contact forces $F_i$ at points $p_i$ (and optional contact
torques $\tau_i$), the resultant is $F = \sum F_i$ and its moment about the world
origin is $M_O = \sum (p_i \times F_i + \tau_i)$. Then, for $F_z > 0$:

$$x_{cop} = \frac{z_g F_x - M_{O,y}}{F_z}, \qquad y_{cop} = \frac{M_{O,x} + z_g F_y}{F_z}$$

The shared ground-reaction core (`src/shared/python/biomechanics/ground_reaction.py`,
GCV-1) evaluates this per foot and for both feet together, and reports the CoP as
unavailable (`None`, never zero) while $F_z$ is below 10 N. `compute_grf_metrics`
below does not compute the CoP; it summarizes a CoP series supplied by the caller.

For a two-foot stance, the CoP moves between the feet during the swing, reflecting
weight transfer patterns.

**API**:

```python
def compute_grf_metrics(self) -> GRFMetrics | None:
    """Compute Ground Reaction Force and Center of Pressure metrics.

    Expects attributes:
        self.cop_position:  (N, 2) or (N, 3) array of CoP positions
        self.ground_forces: (N, 3) array of GRF vectors [N]
        self.dt:            time step [s]

    Returns:
        GRFMetrics dataclass or None if data unavailable
    """
```

The returned `GRFMetrics` dataclass contains:

```python
@dataclass
class GRFMetrics:
    cop_path_length: float       # Total CoP excursion [m]
    cop_max_velocity: float      # Maximum CoP velocity [m/s]
    cop_x_range: float           # Lateral CoP range [m]
    cop_y_range: float           # Anterior-posterior CoP range [m]
    peak_vertical_force: float | None   # Peak Fz [N]
    peak_shear_force: float | None      # Peak shear [N]
```

**Usage**:

```python
grf = analyzer.compute_grf_metrics()
if grf is not None:
    print(f"CoP path length: {grf.cop_path_length:.3f} m")
    print(f"CoP lateral range: {grf.cop_x_range:.3f} m")
    print(f"Peak vertical GRF: {grf.peak_vertical_force:.1f} N")
    print(f"Peak shear GRF: {grf.peak_shear_force:.1f} N")
```

### 12.4.4 Lateral Weight Shift

The CoP x-range (`cop_x_range`) directly quantifies the lateral weight shift during the
swing. A typical golf swing shows:

1. **Address**: CoP centered between feet
2. **Backswing**: CoP shifts toward trail foot
3. **Downswing**: CoP rapidly shifts toward lead foot
4. **Follow-through**: CoP settles over lead foot

### 12.4.5 Free Moment Analysis

The free moment is the vertical torque the ground applies about the CoP. It represents
the rotational resistance the ground provides to the golfer's axial rotation.
`compute_grf_metrics` does not compute it. The shared ground-reaction core (GCV-1)
does, per foot and for both feet together, from the same resultant wrench as the CoP:

$$T_z = M_{O,z} - (x_{cop} F_y - y_{cop} F_x)$$

It is unavailable whenever the CoP is. The net free moment is taken about the net
CoP, so it is in general not the sum of the two feet's free moments: the shear forces
acting at the offset foot CoPs add a vertical moment.

### 12.4.6 Implementation Notes

The CoP path length computation uses optimized array operations:

```python
# Manual slicing (faster than np.diff for this use case)
cop_diff = cop_position[1:] - cop_position[:-1]

# np.hypot is faster than np.linalg.norm for 2D vectors
if cop_diff.shape[1] == 2:
    cop_dist = np.hypot(cop_diff[:, 0], cop_diff[:, 1])
else:
    cop_dist = np.sqrt(np.sum(cop_diff**2, axis=1))

path_length = np.sum(cop_dist)
```

---

## 12.5 Angular Momentum (`angular_momentum.py`)

The `AngularMomentumMetricsMixin` tracks the angular momentum of the system and its
individual segments.

### 12.5.1 Segmental Angular Momentum

The angular momentum of segment $i$ about the system center of mass is:

$$\mathbf{H}_i = I_i\,\omega_i + m_i\,(r_i \times v_i)$$

where:

- $I_i$ is the inertia tensor of segment $i$ about its own center of mass
- $\omega_i$ is the angular velocity of segment $i$
- $m_i$ is the mass of segment $i$
- $r_i$ is the position vector from the system CoM to the segment CoM
- $v_i$ is the velocity of the segment CoM relative to the system CoM

The first term $I_i\,\omega_i$ is the _local_ (spin) angular momentum, and the second
term $m_i\,(r_i \times v_i)$ is the _remote_ (orbital) angular momentum.

### 12.5.2 Total Angular Momentum

The total angular momentum is the sum over all segments:

$$\mathbf{H}_{total} = \sum_{i=1}^{n} \mathbf{H}_i = \sum_{i=1}^{n} \bigl[I_i\,\omega_i + m_i\,(r_i \times v_i)\bigr]$$

### 12.5.3 Conservation Analysis

For a system with no external torques, angular momentum should be conserved:
$\dot{\mathbf{H}}_{total} = \mathbf{0}$. In a golf swing, GRF provides an external
torque, so $\mathbf{H}_{total}$ is not conserved. However, the _rate of change_ of
angular momentum equals the external moment:

$$\dot{\mathbf{H}}_{total} = \mathbf{M}_{ext} = \mathbf{r}_{CoP} \times \mathbf{F}_{GRF}$$

The variability metric quantifies how much angular momentum changes relative to its mean:

$$\text{variability} = \frac{\sigma_{|\mathbf{H}|}}{\mu_{|\mathbf{H}|}}$$

### 12.5.4 Angular Momentum Transfer

During the downswing, angular momentum transfers from proximal to distal segments
(analogous to the kinematic sequence for velocities). This transfer is what creates the
"whip" effect in the golf swing.

**API**:

```python
def compute_angular_momentum_metrics(self) -> AngularMomentumMetrics | None:
    """Compute metrics related to system angular momentum.

    Expects attributes:
        self.angular_momentum: (N, 3) array of angular momentum vectors [kg*m^2/s]
        self.times:            (N,) array of time stamps [s]

    Returns:
        AngularMomentumMetrics dataclass or None if data unavailable
    """
```

The returned dataclass:

```python
@dataclass
class AngularMomentumMetrics:
    peak_magnitude: float     # Peak |H| [kg*m^2/s]
    peak_time: float          # Time of peak [s]
    mean_magnitude: float     # Mean |H| [kg*m^2/s]
    peak_lx: float            # Peak |Hx| [kg*m^2/s]
    peak_ly: float            # Peak |Hy| [kg*m^2/s]
    peak_lz: float            # Peak |Hz| [kg*m^2/s]
    variability: float        # std(|H|) / mean(|H|) (dimensionless)
```

**Usage**:

```python
am_metrics = analyzer.compute_angular_momentum_metrics()
if am_metrics is not None:
    print(f"Peak angular momentum: {am_metrics.peak_magnitude:.2f} kg*m^2/s")
    print(f"  at time: {am_metrics.peak_time:.3f} s")
    print(f"Components: Lx={am_metrics.peak_lx:.2f}, "
          f"Ly={am_metrics.peak_ly:.2f}, Lz={am_metrics.peak_lz:.2f}")
    print(f"Variability: {am_metrics.variability:.3f}")
```

---

## 12.6 Phase Detection (`phase_detection.py`)

The `PhaseDetectionMixin` automatically segments a golf swing into canonical phases using
heuristic algorithms based on club head speed.

### 12.6.1 Golf Swing Phases

A complete golf swing is divided into the following phases:

| Phase              | Description                          | Key Event                      |
| ------------------ | ------------------------------------ | ------------------------------ |
| **Address**        | Setup position before movement       | Start of recording             |
| **Takeaway**       | Initial club movement away from ball | First significant speed        |
| **Backswing**      | Club moves to top of backswing       | Increasing shoulder turn       |
| **Transition**     | Direction change at top              | Speed minimum before downswing |
| **Downswing**      | Acceleration toward impact           | Rapid speed increase           |
| **Impact**         | Club contacts ball                   | Peak club head speed           |
| **Follow-through** | Deceleration after impact            | Speed decreasing               |
| **Finish**         | Final pose                           | Speed below threshold          |

### 12.6.2 Phase Detection Algorithm

The detection algorithm proceeds as follows:

1. **Smooth** the club head speed signal using a Savitzky-Golay filter (window = 11,
   polynomial order = 3) to remove noise while preserving sharp transitions.

2. **Find impact** as the time of peak smoothed speed:
   $$t_{impact} = \arg\max_t \tilde{v}_{club}(t)$$

3. **Find transition** (top of backswing) as the speed minimum in the first 70% of
   the pre-impact interval, avoiding the initial 5 frames of noise:
   $$t_{transition} = \arg\min_{t \in [t_5, 0.7 \cdot t_{impact}]} \tilde{v}_{club}(t)$$

4. **Find takeaway** as the first frame where speed exceeds 10% of the transition speed.

5. **Find finish** as the first post-impact frame where speed drops below 30% of peak.

6. **Assign phase boundaries** based on proportional divisions between key events.

**API**:

```python
def detect_swing_phases(self) -> list[SwingPhase]:
    """Automatically detect swing phases.

    Expects attributes:
        self.club_head_speed: (N,) array of speeds [m/s]
        self.times:           (N,) array of timestamps [s]
        self.duration:        total duration [s]

    Returns:
        List of SwingPhase objects
    """
```

Each `SwingPhase` contains:

```python
@dataclass
class SwingPhase:
    name: str           # Phase name
    start_time: float   # Phase start [s]
    end_time: float     # Phase end [s]
    start_index: int    # Array index of phase start
    end_index: int      # Array index of phase end
    duration: float     # Phase duration [s]
```

**Usage**:

```python
phases = analyzer.detect_swing_phases()
for phase in phases:
    print(f"{phase.name:15s}  {phase.start_time:.3f}s - {phase.end_time:.3f}s  "
          f"({phase.duration:.3f}s)")
```

Example output:

```
Address          0.000s - 0.120s  (0.120s)
Takeaway         0.120s - 0.280s  (0.160s)
Backswing        0.280s - 0.650s  (0.370s)
Transition       0.650s - 0.720s  (0.070s)
Downswing        0.720s - 0.950s  (0.230s)
Impact           0.948s - 0.952s  (0.004s)
Follow-through   0.950s - 1.150s  (0.200s)
Finish           1.150s - 1.500s  (0.350s)
```

### 12.6.3 Phase-Specific Statistics

After detecting phases, per-phase statistics can be computed for any signal:

```python
def compute_phase_statistics(
    self,
    phases: list[SwingPhase],
    data: np.ndarray,
) -> dict[str, SummaryStatistics]:
    """Compute statistics for each phase.

    Args:
        phases: List of detected swing phases
        data: 1D data array (same length as self.times)

    Returns:
        Dictionary mapping phase name to SummaryStatistics
    """
```

**Usage**:

```python
phases = analyzer.detect_swing_phases()

# Speed statistics per phase
speed_stats = analyzer.compute_phase_statistics(phases, analyzer.club_head_speed)
for phase_name, stats in speed_stats.items():
    print(f"{phase_name}: mean={stats.mean:.2f}, max={stats.max:.2f}, "
          f"std={stats.std:.2f} m/s")
```

### 12.6.4 Event Markers

Key discrete events within the swing:

- **Top of backswing**: Maximum X-factor / minimum club head speed before downswing
- **Impact**: Peak club head speed
- **Finish**: Speed drops below threshold

```python
impact_time = analyzer.detect_impact_time()
if impact_time is not None:
    print(f"Impact at t = {impact_time:.3f}s")
```

---

## 12.7 Stability Metrics (`stability_metrics.py`)

The `StabilityMetricsMixin` computes balance and postural stability metrics essential
for golf swing analysis. It computes the CoM-CoP distance and the inclination angle
(12.7.2 and 12.7.5). The base of support, time-to-boundary and dynamic balance index
(12.7.1, 12.7.3 and 12.7.4) are described for context only; they are not computed.

### 12.7.1 Base of Support

The base of support (BoS) is the convex hull of all ground contact points. For a golfer,
this is approximately the rectangle defined by the outer edges of both shoes. Stability
requires the CoM projection to remain within the BoS.

### 12.7.2 CoM-CoP Distance

The horizontal distance between the centre of mass (CoM) and centre of pressure (CoP)
is a primary stability measure:

$$d_{stability} = \|CoM_{xy} - CoP_{xy}\|$$

where the subscript $xy$ denotes the horizontal (ground plane) components.

### 12.7.3 Time-to-Boundary

The time-to-boundary (TTB) estimates how long until the CoM would reach the edge of the
base of support at the current velocity:

$$TTB = \frac{d_{boundary}}{v_{CoM}}$$

where $d_{boundary}$ is the distance from the CoM projection to the nearest BoS boundary
and $v_{CoM}$ is the horizontal CoM velocity. Smaller TTB indicates less stability margin.

### 12.7.4 Dynamic Balance Index

The dynamic balance index combines CoM-CoP distance with CoM velocity to give a single
measure of balance control during dynamic movement.

### 12.7.5 Inclination Angle

The inclination angle $\theta$ measures the lean of the CoM-CoP vector from vertical:

$$\theta = \arccos\left(\frac{v_z}{\|v\|}\right)$$

where $v = CoM - CoP$ is the 3D vector from CoP to CoM, and $v_z$ is its vertical
component.

**API**:

```python
def compute_stability_metrics(self) -> StabilityMetrics | None:
    """Compute postural stability metrics.

    Expects attributes:
        self.cop_position: (N, 2) or (N, 3) array of CoP positions [m]
        self.com_position: (N, 3) array of CoM positions [m]

    Returns:
        StabilityMetrics dataclass or None if data unavailable
    """
```

The returned dataclass:

```python
@dataclass
class StabilityMetrics:
    min_com_cop_distance: float     # Minimum horizontal CoM-CoP distance [m]
    max_com_cop_distance: float     # Maximum horizontal CoM-CoP distance [m]
    mean_com_cop_distance: float    # Mean horizontal CoM-CoP distance [m]
    peak_inclination_angle: float   # Maximum lean angle [degrees]
    mean_inclination_angle: float   # Mean lean angle [degrees]
```

**Usage**:

```python
stability = analyzer.compute_stability_metrics()
if stability is not None:
    print(f"CoM-CoP distance: {stability.mean_com_cop_distance:.3f} m "
          f"(min={stability.min_com_cop_distance:.3f}, "
          f"max={stability.max_com_cop_distance:.3f})")
    print(f"Peak inclination angle: {stability.peak_inclination_angle:.1f} degrees")
    print(f"Mean inclination angle: {stability.mean_inclination_angle:.1f} degrees")
```

### 12.7.6 Implementation Notes

The implementation assumes a Z-up coordinate convention (consistent with MuJoCo). The
CoP is treated as a 2D point on the ground plane ($z=0$); if provided as 3D, only the
first two columns are used for horizontal distance computation.

Optimized operations used:

```python
# np.hypot is faster than np.linalg.norm for 2D vectors
diff = cop_xy - com_xy
dist = np.hypot(diff[:, 0], diff[:, 1])
```

---

## 12.8 Shared Dataclasses

All analysis modules use dataclasses defined in `src/shared/python/analysis/dataclasses.py`
for structured output. The complete set:

| Dataclass                | Module           | Key Fields                                                                            |
| ------------------------ | ---------------- | ------------------------------------------------------------------------------------- |
| `PeakInfo`               | Basic stats      | `value`, `time`, `index`, `prominence`, `width`                                       |
| `SummaryStatistics`      | Basic stats      | `mean`, `median`, `std`, `min`, `max`, `range`, `rms`                                 |
| `SwingPhase`             | Phase detection  | `name`, `start_time`, `end_time`, `duration`                                          |
| `KinematicSequenceInfo`  | Swing metrics    | `segment_name`, `peak_velocity`, `peak_time`, `order_index`                           |
| `GRFMetrics`             | GRF analysis     | `cop_path_length`, `cop_max_velocity`, `peak_vertical_force`                          |
| `AngularMomentumMetrics` | Angular momentum | `peak_magnitude`, `peak_time`, `peak_lx/ly/lz`, `variability`                         |
| `StabilityMetrics`       | Stability        | `min/max/mean_com_cop_distance`, `peak/mean_inclination_angle`                        |
| `CoordinationMetrics`    | Coordination     | `in_phase_pct`, `anti_phase_pct`, `mean_coupling_angle`                               |
| `JointPowerMetrics`      | Power analysis   | `peak_generation`, `peak_absorption`, `net_work`                                      |
| `ImpulseMetrics`         | Impulse          | `net_impulse`, `positive_impulse`, `negative_impulse`                                 |
| `JerkMetrics`            | Smoothness       | `peak_jerk`, `rms_jerk`, `dimensionless_jerk`                                         |
| `JointStiffnessMetrics`  | Stiffness        | `stiffness`, `r_squared`, `hysteresis_area`                                           |
| `SwingProfileMetrics`    | Profile          | `speed_score`, `sequence_score`, `stability_score`, `efficiency_score`, `power_score` |
| `PCAResult`              | PCA              | `components`, `explained_variance`, `projected_data`                                  |
| `RQAMetrics`             | Recurrence       | `recurrence_rate`, `determinism`, `laminarity`                                        |

---

## 12.9 Muscle Synergy Analysis

The `MuscleSynergyAnalyzer` (in `src/shared/python/muscle_analysis.py`) extracts coordinated
muscle activation patterns using Non-negative Matrix Factorization (NMF).

### 12.9.1 Theory

Muscle synergies represent the hypothesis that the CNS simplifies control by activating
groups of muscles together. The activation matrix $V$ (muscles $\times$ time) is
factorized as:

$$V \approx W \cdot H$$

where:

- $W \in \mathbb{R}^{n_{muscles} \times n_{synergies}}$ -- muscle weights (which muscles
  belong to each synergy)
- $H \in \mathbb{R}^{n_{synergies} \times n_{samples}}$ -- temporal activation profiles
  (when each synergy is active)

### 12.9.2 Variance Accounted for (VAF)

The quality of the factorization is measured by:

$$\text{VAF} = 1 - \frac{\sum (V - WH)^2}{\sum V^2}$$

A VAF $\geq 0.90$ (90%) is typically considered a good reconstruction.

### 12.9.3 API

```python
from src.shared.python.muscle_analysis import MuscleSynergyAnalyzer

# activation_data: (n_samples, n_muscles), non-negative
analyzer = MuscleSynergyAnalyzer(activation_data, muscle_names=["bic", "tri", ...])

# Extract specific number of synergies
result = analyzer.extract_synergies(n_synergies=3)
print(f"VAF: {result.vaf:.2%}")
print(f"Weights shape: {result.weights.shape}")      # (n_muscles, 3)
print(f"Activations shape: {result.activations.shape}")  # (3, n_samples)

# Find optimal number automatically
optimal = analyzer.find_optimal_synergies(max_synergies=10, vaf_threshold=0.90)
print(f"Optimal synergies: {optimal.n_synergies} (VAF={optimal.vaf:.2%})")
```

---

## 12.10 Complete Analysis Workflow

```python
import numpy as np

# Assume simulation data has been collected:
#   times:            (N,) timestamps
#   joint_positions:  (N, n_joints) joint angles
#   club_head_speed:  (N,) club head speed
#   cop_position:     (N, 2) centre of pressure
#   com_position:     (N, 3) centre of mass
#   ground_forces:    (N, 3) ground reaction forces
#   angular_momentum: (N, 3) total angular momentum
#   kinetic_energy:   (N,) kinetic energy
#   potential_energy: (N,) potential energy

# 1. Phase detection
phases = analyzer.detect_swing_phases()
print("=== Swing Phases ===")
for phase in phases:
    print(f"  {phase.name:15s}  {phase.start_time:.3f}s - {phase.end_time:.3f}s")

# 2. Swing metrics
impact_time = analyzer.detect_impact_time()
tempo = analyzer.compute_tempo()
x_factor = analyzer.compute_x_factor(shoulder_idx, hip_idx)

print(f"\n=== Swing Metrics ===")
print(f"  Impact time: {impact_time:.3f}s")
if tempo:
    print(f"  Tempo ratio: {tempo[2]:.2f}")
if x_factor is not None:
    print(f"  Peak X-Factor: {np.max(x_factor):.1f} deg")

# 3. Energy analysis
energy = analyzer.compute_energy_metrics(kinetic_energy, potential_energy)
print(f"\n=== Energy ===")
print(f"  Peak KE: {energy['max_kinetic_energy']:.1f} J")
print(f"  Efficiency: {energy['energy_efficiency']:.1f}%")
print(f"  Energy drift: {energy['energy_drift']:.3f} J")

# 4. GRF analysis
grf = analyzer.compute_grf_metrics()
if grf:
    print(f"\n=== Ground Reaction Forces ===")
    print(f"  CoP path length: {grf.cop_path_length:.3f} m")
    print(f"  Peak vertical GRF: {grf.peak_vertical_force:.1f} N")
    print(f"  CoP lateral range: {grf.cop_x_range:.3f} m")

# 5. Angular momentum
am = analyzer.compute_angular_momentum_metrics()
if am:
    print(f"\n=== Angular Momentum ===")
    print(f"  Peak: {am.peak_magnitude:.2f} kg*m^2/s at {am.peak_time:.3f}s")
    print(f"  Variability: {am.variability:.3f}")

# 6. Stability
stability = analyzer.compute_stability_metrics()
if stability:
    print(f"\n=== Stability ===")
    print(f"  Mean CoM-CoP distance: {stability.mean_com_cop_distance:.3f} m")
    print(f"  Peak inclination: {stability.peak_inclination_angle:.1f} deg")

# 7. Phase-specific analysis
speed_by_phase = analyzer.compute_phase_statistics(phases, club_head_speed)
print(f"\n=== Speed by Phase ===")
for phase_name, stats in speed_by_phase.items():
    print(f"  {phase_name:15s}  mean={stats.mean:.1f}  max={stats.max:.1f} m/s")
```

---

## 12.11 Summary of Equations

For reference, the key biomechanical equations used across the analysis suite:

| Quantity                   | Equation                                                       | Units            |
| -------------------------- | -------------------------------------------------------------- | ---------------- |
| Club head speed            | $v_{club} = \|\dot{p}_{head}\|$                                | m/s              |
| X-Factor                   | $X = \theta_{shoulder} - \theta_{hip}$                         | degrees          |
| Smash factor               | $SF = v_{ball} / v_{club}$                                     | dimensionless    |
| Kinetic energy             | $T = \frac{1}{2}\,\dot{q}^T\,M(q)\,\dot{q}$                    | J                |
| Potential energy           | $V = -\sum m_i\,g^T\,p_i$                                      | J                |
| Energy efficiency          | $\eta = E_{club}/E_{total} \times 100\%$                       | %                |
| Centre of pressure         | $\text{CoP} = \frac{\sum F_i \times p_i}{\sum F_i}$            | m                |
| Segmental angular momentum | $\mathbf{H}_i = I_i\,\omega_i + m_i\,(r_i \times v_i)$         | kg$\cdot$m$^2$/s |
| Total angular momentum     | $\mathbf{H}_{total} = \sum \mathbf{H}_i$                       | kg$\cdot$m$^2$/s |
| CoM-CoP stability distance | $d = \|CoM_{xy} - CoP_{xy}\|$                                  | m                |
| Time-to-boundary           | $TTB = d_{boundary} / v_{CoM}$                                 | s                |
| Hill muscle force          | $F_m = a \cdot F_{max} \cdot f_L \cdot f_V + F_{PE}$           | N                |
| Joint power                | $P = \tau \cdot \dot{q}$                                       | W                |
| Total work                 | $W = \int_0^T P\,dt$                                           | J                |
| Activation dynamics        | $\dot{a} = (u - a) / \tau$                                     | 1/s              |
| Muscle recruitment         | $\min \sum a_i^2$ s.t. $\tau = R \cdot F(a)$                   | --               |
| Jacobian                   | $J(q) = \frac{\partial \phi(q)}{\partial q}$                   | varies           |
| Jacobian derivative        | $\dot{J}(q, \dot{q}) = \frac{\partial J}{\partial q}\,\dot{q}$ | varies           |
| Drift acceleration         | $\ddot{q}_{drift} = M^{-1}(-C\dot{q} - g)$                     | rad/s$^2$        |
| Control acceleration       | $\ddot{q}_{control} = M^{-1}\tau$                              | rad/s$^2$        |
| Superposition              | $\ddot{q}_{full} = \ddot{q}_{drift} + \ddot{q}_{control}$      | rad/s$^2$        |

---
