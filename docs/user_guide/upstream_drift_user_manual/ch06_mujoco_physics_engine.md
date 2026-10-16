# Chapter 6: MuJoCo Physics Engine

## 6.1 Overview

The MuJoCo backend is the primary physics engine for the Golf Modeling Suite. It
provides high-fidelity rigid-body simulation with contact, constraints, and
muscle-driven actuation. The implementation is spread across several modules:

```
src/engines/physics_engines/mujoco/
    python/mujoco_humanoid_golf/
        physics_engine.py       -- PhysicsEngine protocol implementation
        models.py               -- MJCF model hierarchy
        advanced_kinematics.py  -- IK, manipulability, singularity analysis
        advanced_control.py     -- Impedance, admittance, computed torque, OSC
        kinematic_forces.py     -- Coriolis, centrifugal, gravitational analysis
        inverse_dynamics.py     -- Full ID solver with force decomposition
        motion_capture.py       -- Motion capture data loading
        motion_optimization.py  -- Swing trajectory optimization
    docker/src/humanoid_golf/
        sim.py                  -- Simulation loop, controllers, visualization
```

---

## 6.2 `MuJoCoPhysicsEngine`

**File:** `physics_engine.py`

```python
class MuJoCoPhysicsEngine(PhysicsEngine):
    def __init__(self) -> None:
        self.model: mujoco.MjModel | None = None
        self.data:  mujoco.MjData  | None = None
        self.xml_path: str | None = None
```

### 6.2.1 Model Loading

Models are loaded via `mujoco.MjModel.from_xml_path()` or
`mujoco.MjModel.from_xml_string()`. Path traversal is mitigated by validating
against `ALLOWED_MODEL_DIRS`:

```python
ALLOWED_MODEL_DIRS = [
    SUITE_ROOT / "engines",
    SUITE_ROOT / "shared" / "resources",
    Path(tempfile.gettempdir()),
]
```

### 6.2.2 Core Dynamics

**Mass Matrix** -- computed via `mujoco.mj_fullM()`:

```python
@precondition(lambda self: self.is_initialized, "Engine must be initialized")
@postcondition(check_finite, "Mass matrix must contain finite values")
def compute_mass_matrix(self) -> np.ndarray:
    nv = self.model.nv
    M = np.zeros((nv, nv), dtype=np.float64)
    mujoco.mj_fullM(self.model, M, self.data.qM)
    return M
```

**Bias Forces** -- read directly from `data.qfrc_bias` (populated by
`mj_forward` / `mj_step`).

**Inverse Dynamics** -- via `mujoco.mj_inverse()`:

```python
self.data.qacc[:] = qacc
mujoco.mj_inverse(self.model, self.data)
return self.data.qfrc_inverse.copy()
```

### 6.2.3 Drift-Control Decomposition

The drift acceleration is computed by zeroing out all control inputs, running
`mj_forward`, and reading back `data.qacc`:

```python
def compute_affine_drift(self) -> np.ndarray:
    saved_ctrl = self.data.ctrl.copy()
    self.data.ctrl[:] = 0
    mujoco.mj_forward(self.model, self.data)
    drift_acc = self.data.qacc.copy()
    self.data.ctrl[:] = saved_ctrl
    mujoco.mj_forward(self.model, self.data)  # Restore
    return drift_acc
```

Control acceleration uses $\ddot{q}_{\text{control}} = M^{-1}\tau$:

```python
def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    M = self.compute_mass_matrix()
    return np.linalg.solve(M, tau)
```

### 6.2.4 Counterfactual Experiments (ZTCF / ZVCF)

Both methods follow a save-mutate-restore pattern to prevent Observer Effect
corruption of simulation state:

```python
def compute_ztcf(self, q, v) -> np.ndarray:
    saved_qpos = self.data.qpos.copy()
    saved_qvel = self.data.qvel.copy()
    saved_ctrl = self.data.ctrl.copy()
    try:
        self.data.qpos[:] = q; self.data.qvel[:] = v
        self.data.ctrl[:] = 0               # Zero torque
        mujoco.mj_forward(self.model, self.data)
        return self.data.qacc.copy()
    finally:
        # Restore original state
        self.data.qpos[:] = saved_qpos
        self.data.qvel[:] = saved_qvel
        self.data.ctrl[:] = saved_ctrl
        mujoco.mj_forward(self.model, self.data)
```

### 6.2.5 Flexible Shaft

The MuJoCo engine supports flexible club shafts using Euler-Bernoulli beam
theory. Natural frequencies are computed analytically from cantilever beam
eigenvalues:

$$\omega_n = \left(\frac{\beta_n}{L}\right)^2 \sqrt{\frac{EI}{\mu}}$$

where $\beta_n L$ takes the standard clamped-free values
$\{1.875, 4.694, 7.855, 10.996, 14.137\}$ and $\mu$ is the linear mass density.
The first three bending modes are retained.

---

## 6.3 Model Hierarchy

**File:** `models.py`

The suite provides a graduated hierarchy of MJCF models, from simple pendulums
to full-body musculoskeletal systems.

### 6.3.1 Pendulum Models

| Model                  | DOF | Description                           |
| ---------------------- | --- | ------------------------------------- |
| `CHAOTIC_PENDULUM_XML` | 1   | Single pendulum with chaotic regime   |
| `DOUBLE_PENDULUM_XML`  | 2   | Shoulder + wrist (minimal golf swing) |
| `TRIPLE_PENDULUM_XML`  | 3   | Shoulder + elbow + wrist              |

The double pendulum serves as the simplest golf-swing analogue, capturing the
essential energy-transfer mechanism from proximal to distal segments.

### 6.3.2 Multi-Segment Golf Models

| Model                                   | DOF | Bodies                      | Key Features                                                        |
| --------------------------------------- | --- | --------------------------- | ------------------------------------------------------------------- |
| `UPPER_BODY_GOLF_SWING_XML`             | 10  | Spine, bilateral arms, club | Spine torsion, bilateral shoulder/elbow/wrist                       |
| `FULL_BODY_GOLF_SWING_XML`              | 15  | Full body with legs         | Lower body (hip, knee, ankle) + upper body                          |
| `ADVANCED_BIOMECHANICAL_GOLF_SWING_XML` | 28  | Full body + scapulae        | 3-DOF spine, scapulae, forearm supination, 3-segment flexible shaft |

The advanced model includes:

- **3-DOF spine** (flexion/extension, lateral bending, axial rotation).
- **Scapular protraction/retraction** on both sides.
- **Forearm supination** for club face control.
- **3-segment flexible club shaft** with torsional spring-damper joints modelling
  shaft flex and kick.

### 6.3.3 MyoSuite Integration

For muscle-driven simulation, the suite supports MyoSuite models:

```python
MYOUPPERBODY_PATH = "myosuite/envs/myo/assets/arm/myoarm.xml"
MYOBODY_PATH = "myosuite/envs/myo/assets/body/myobody.xml"
# 52 DOF, 290 muscle actuators
```

### 6.3.4 Golf Ball Physics

Golf ball parameters follow USGA regulations and are imported from
`shared.python.constants`:

- Mass: `GOLF_BALL_MASS_KG` (0.04593 kg)
- Radius: `GOLF_BALL_RADIUS_M` (0.02135 m)

Ball bodies are defined with appropriate `condim`, `friction`, and `solref`
parameters for realistic impact dynamics.

### 6.3.5 Parametric Model Generation

Two factory functions produce custom club models:

```python
def generate_flexible_club_xml(
    n_segments: int = 3,
    shaft_length: float = 1.0,
    shaft_stiffness: float = 500.0,
    shaft_damping: float = 5.0,
) -> str:
    """Generate MJCF XML for a flexible multi-segment club."""

def generate_rigid_club_xml(
    shaft_length: float = 1.0,
    club_head_mass: float = 0.3,
) -> str:
    """Generate MJCF XML for a rigid single-segment club."""
```

---

## 6.4 Simulation and Controllers

**File:** `docker/src/humanoid_golf/sim.py`

The Docker-hosted simulation provides a complete run loop with configurable
controllers and visualization.

### 6.4.1 Controller Hierarchy

```python
class BaseController:
    def get_action(self, physics) -> np.ndarray:
        return np.zeros(physics.model.nu)

class PDController(BaseController):
    def __init__(self, actuators, target_pose, kp=60.0, kd=6.0): ...

class PolynomialController(BaseController):
    # 6th-order polynomial: u(t) = c_0 + c_1 t + ... + c_6 t^6
    def __init__(self, physics): ...

class LQRController(BaseController):
    # u = -K (x - x_target) with gain matrix K
    def __init__(self, physics, target_pose, actuators, height_scale=1.0): ...
```

**PD Controller.** For each joint in the `target_pose` dictionary:

$$\tau_j = K_p (q_j^d - q_j) - K_d \dot{q}_j$$

with default gains $K_p = 60$, $K_d = 6$.

**Polynomial Controller.** Generates time-varying torque profiles using
sixth-order polynomials. A truncated Taylor series $60t - 20t^3$ approximates
a swing-like sinusoidal torque on the right shoulder.

**LQR Controller.** Computes a linear state-feedback gain matrix $K$ such that
$u = -K(x - x^*)$. A decoupled diagonal PD fallback with $K_p = 100$,
$K_d = 10$ is used for robustness with quaternion-based root configurations.

### 6.4.2 Induced Acceleration Analysis in the Loop

Each simulation step computes induced accelerations and counterfactuals via
the `iaa_helper` module:

```python
iaa = iaa_helper.compute_induced_accelerations(physics)
cf  = iaa_helper.compute_counterfactuals(physics)
```

Data is emitted as JSON packets for real-time analytics consumption.

---

## 6.5 Advanced Kinematics

**File:** `advanced_kinematics.py`

### 6.5.1 `AdvancedKinematicsAnalyzer`

```python
class AdvancedKinematicsAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.singularity_threshold = 30.0
        self.ik_damping = 0.01
        self.ik_max_iterations = 100
        self.ik_tolerance = 1e-4
```

### 6.5.2 Jacobian Computation

Jacobians are computed via `mujoco.mj_jacBody()`. An offset point $r$ in the
body frame adjusts the linear Jacobian:

$$J_p^{\text{offset}} = J_p^{\text{body}} - [\hat{r}]_\times \, J_r^{\text{body}}$$

where $[\hat{r}]_\times$ is the skew-symmetric matrix of the world-frame offset.

### 6.5.3 Manipulability Analysis

Yoshikawa's manipulability index is computed from the Jacobian SVD:

$$w = \sqrt{\det(J J^T)} = \prod_i \sigma_i$$

```python
@dataclass
class ManipulabilityMetrics:
    manipulability_index: float       # w = sqrt(det(J*J^T))
    condition_number: float           # sigma_max / sigma_min
    singular_values: np.ndarray       # [sigma_1, ..., sigma_m]
    is_near_singularity: bool         # True if kappa > threshold
    min_singular_value: float
    max_singular_value: float
```

Singularity is flagged when $\kappa(J) > 30$ or $\sigma_{\min} < 10^{-3}$.

### 6.5.4 Inverse Kinematics (DLS)

The IK solver uses Damped Least-Squares with nullspace projection for
redundancy resolution:

$$\Delta q = J^T (J J^T + \lambda^2 I)^{-1} e + \alpha (I - J^+ J)(q^* - q)$$

where $e$ is the task-space error, $\lambda$ is the damping factor, $q^*$ is the
nullspace objective, and $\alpha$ is the nullspace gain. Joint limits are
enforced by clamping after each iteration.

### 6.5.5 Constraint Jacobian (Closed-Chain Analysis)

For the two-handed golf grip (closed kinematic chain), the constraint Jacobian
is:

$$J_c = J_{\text{right hand}} - J_{\text{left hand}}$$

The nullspace dimension $\dim(\ker J_c)$ determines the effective degrees of freedom
of the constrained system.

### 6.5.6 Task-Space Inertia

$$\Lambda = (J M^{-1} J^T)^{-1}$$

This matrix maps task-space forces to task-space accelerations and is essential
for operational space control.

---

## 6.6 Advanced Control

**File:** `advanced_control.py`

### 6.6.1 Control Modes

```python
class ControlMode(Enum):
    TORQUE         = "torque"
    IMPEDANCE      = "impedance"
    ADMITTANCE     = "admittance"
    HYBRID         = "hybrid"
    COMPUTED_TORQUE = "computed_torque"
    TASK_SPACE     = "task_space"
```

### 6.6.2 Impedance Control

Creates a virtual spring-damper system around the desired trajectory:

$$\tau = K_p(q_d - q) + D(\dot{q}_d - \dot{q}) + g(q)$$

```python
@dataclass
class ImpedanceParameters:
    stiffness: np.ndarray  # K [n x n] or [n]
    damping:   np.ndarray  # D [n x n] or [n]
    inertia:   np.ndarray | None = None  # M (optional)
```

Default parameters: $K_p = 100$ N/m, $D = 20$ Ns/m.

### 6.6.3 Admittance Control

The dual of impedance control, modifying position based on force error:

$$\Delta \ddot{q} = M_d^{-1}(F_d - F_{\text{measured}})$$

where $M_d$ is the desired inertia matrix.

### 6.6.4 Hybrid Force-Position Control

Combines position and force control using selection matrices:

$$\tau = S_p \tau_{\text{position}} + S_f \tau_{\text{force}}$$

```python
@dataclass
class HybridControlMask:
    force_mask: np.ndarray  # Boolean: True = force control, False = position
```

### 6.6.5 Computed Torque Control

Model-based feedforward control using inverse dynamics:

$$\tau = M(q)\ddot{q}_d + C(q, \dot{q})\dot{q} + g(q)$$

where $\ddot{q}_d = K_p(q_d - q) + K_d(\dot{q}_d - \dot{q})$.

The implementation uses `mujoco.mj_mulM()` for efficient sparse
matrix-vector multiplication.

### 6.6.6 Operational Space Control (OSC)

Full operational space controller that accounts for configuration-dependent
inertia:

$$F = \Lambda(q)(\ddot{x}_d + K_d \dot{e} + K_p e) + \mu(q, \dot{q}) + p(q)$$
$$\tau = J^T F + N^T \tau_{\text{posture}}$$

where $\Lambda = (J M^{-1} J^T)^{-1}$ is the task-space inertia and
$N = I - J^T \bar{J}^T$ is the nullspace projector. The dynamically consistent
pseudoinverse is $\bar{J} = M^{-1} J^T \Lambda$.

### 6.6.7 Trajectory Generation

The `TrajectoryGenerator` class produces smooth reference trajectories:

**Minimum-jerk trajectory:**

$$s(\tau) = 10\tau^3 - 15\tau^4 + 6\tau^5, \quad \tau = t/T$$

satisfying boundary conditions $s(0) = 0$, $\dot{s}(0) = 0$, $\ddot{s}(0) = 0$,
$s(1) = 1$, $\dot{s}(1) = 0$, $\ddot{s}(1) = 0$.

---

## 6.7 Kinematic Forces Analysis

**File:** `kinematic_forces.py`

### 6.7.1 `KinematicForceAnalyzer`

Analyses velocity-dependent forces that can be computed from kinematics alone,
without full inverse dynamics.

```python
class KinematicForceAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self._perturb_data = mujoco.MjData(model)  # Scratch copy for thread safety
```

### 6.7.2 Thread Safety via `MjDataContext`

All analysis methods use a dedicated `_perturb_data` copy or the `MjDataContext`
context manager to prevent Observer Effect bugs:

```python
class MjDataContext:
    """Context manager for safe MuJoCo MjData state isolation."""
    def __enter__(self) -> mujoco.MjData:
        self.qpos_backup = self.data.qpos.copy()
        # ... save all state
        return self.data

    def __exit__(self, ...):
        self.data.qpos[:] = self.qpos_backup
        # ... restore all state
        mujoco.mj_forward(self.model, self.data)
```

### 6.7.3 Coriolis Force Computation

The equations of motion are:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau$$

The Coriolis/centrifugal term $C(q, \dot{q})\dot{q}$ is computed using
MuJoCo's Recursive Newton-Euler algorithm (`mj_rne`):

```python
def compute_coriolis_forces_rne(self, qpos, qvel) -> np.ndarray:
    # With qacc=0: mj_rne returns C(q,v)*v + g(q)
    self._perturb_data.qpos[:] = qpos
    self._perturb_data.qvel[:] = qvel
    self._perturb_data.qacc[:] = 0.0
    bias = np.zeros(self.model.nv)
    mujoco.mj_rne(self.model, self._perturb_data, 0, bias)

    # Subtract gravity: with qvel=0, mj_rne returns g(q)
    self._perturb_data.qvel[:] = 0.0
    gravity = np.zeros(self.model.nv)
    mujoco.mj_rne(self.model, self._perturb_data, 0, gravity)

    return bias - gravity  # = C(q,v)*v
```

### 6.7.4 Coriolis-Centrifugal Decomposition

The total Coriolis force is decomposed into centrifugal (diagonal) and
velocity-coupling (off-diagonal) components. For each DOF $i$, the centrifugal
contribution is computed by setting all velocities except $\dot{q}_i$ to zero:

$$f_{\text{centrifugal}} = \sum_i C(q, e_i \dot{q}_i) e_i \dot{q}_i$$

$$f_{\text{coupling}} = f_{\text{total}} - f_{\text{centrifugal}}$$

Note: This decomposition is $O(n^2)$ and the combined `compute_coriolis_forces()`
at $O(n)$ is preferred for most applications.

### 6.7.5 `KinematicForceData`

```python
@dataclass
class KinematicForceData:
    time: float
    coriolis_forces: np.ndarray             # [nv]
    gravity_forces: np.ndarray              # [nv]
    centrifugal_forces: np.ndarray | None   # [nv]
    velocity_coupling_forces: np.ndarray | None
    club_head_coriolis_force: np.ndarray | None   # [3]
    club_head_centrifugal_force: np.ndarray | None
    club_head_apparent_force: np.ndarray | None
    coriolis_power: float
    centrifugal_power: float
    rotational_kinetic_energy: float
    translational_kinetic_energy: float
```

### 6.7.6 Effective Mass

The directional effective mass at a body determines the resistance to
acceleration along a given direction:

$$m_{\text{eff}} = \frac{1}{\hat{d}^T J M^{-1} J^T \hat{d}}$$

where $\hat{d}$ is the unit direction vector. Near kinematic singularities,
$m_{\text{eff}} \to \infty$. Condition number monitoring triggers warnings
when $\kappa(M) > 10^6$.

---

## 6.8 Inverse Dynamics

**File:** `inverse_dynamics.py`

### 6.8.1 `InverseDynamicsSolver`

```python
class InverseDynamicsSolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData): ...
```

**Core method:**

```python
def compute_required_torques(
    self, qpos, qvel, qacc, external_forces=None
) -> InverseDynamicsResult
```

**Result structure:**

```python
@dataclass
class InverseDynamicsResult:
    joint_torques: np.ndarray
    constraint_forces: np.ndarray | None
    inertial_torques: np.ndarray | None     # M(q) * qacc
    coriolis_torques: np.ndarray | None     # C(q,v) * v
    gravity_torques: np.ndarray | None      # g(q)
```

### 6.8.2 Nullspace Projection for Posture

When tracking a primary task (e.g. club head trajectory), secondary posture
objectives are projected into the task nullspace:

$$\tau = \tau_{\text{primary}} + (I - J^+ J) \, K_p (q_d - q)$$

```python
def compute_torques_with_posture(
    self, qpos, qvel, qacc_primary, qpos_desired, kp_posture, primary_body_name
) -> InverseDynamicsResult
```

### 6.8.3 Force Decomposition

```python
@dataclass
class ForceDecomposition:
    inertial: np.ndarray      # M * qacc
    coriolis: np.ndarray      # C(q,v) * v
    gravity: np.ndarray       # g(q)
    external: np.ndarray      # J^T * f_ext
    total: np.ndarray         # Sum
```

### 6.8.4 Trajectory Inverse Dynamics

```python
def solve_inverse_dynamics_trajectory(
    self, times, positions, velocities, accelerations
) -> list[InverseDynamicsResult]
```

---

## 6.9 Motion Capture

**File:** `motion_capture.py`

### 6.9.1 Data Structures

```python
@dataclass
class MotionCaptureFrame:
    time: float
    marker_positions: dict[str, np.ndarray]   # name -> [x, y, z]
    marker_velocities: dict[str, np.ndarray] | None
    body_orientations: dict[str, np.ndarray] | None
    joint_angles: np.ndarray | None

@dataclass
class MotionCaptureSequence:
    frames: list[MotionCaptureFrame]
    frame_rate: float
    marker_names: list[str]
    metadata: dict
```

### 6.9.2 Supported Formats

The `MotionCaptureSequence` class provides loaders for:

- **CSV** -- Simple tabular format with columns for time, marker x/y/z.
- **JSON** -- Structured format with metadata.
- **C3D** -- Industry-standard motion capture format (via `ezc3d` library).

---

## 6.10 Motion Optimization

**File:** `motion_optimization.py`

### 6.10.1 Objective Configuration

```python
@dataclass
class OptimizationObjectives:
    maximize_club_speed: bool = True
    minimize_energy: bool = True
    minimize_jerk: bool = True
    minimize_torque: bool = True
    weight_speed: float = 10.0
    weight_energy: float = 1.0
    weight_jerk: float = 0.5
    weight_torque: float = 0.1
    weight_accuracy: float = 5.0
```

### 6.10.2 `SwingOptimizer`

The `SwingOptimizer` class performs multi-objective trajectory optimisation,
balancing club head speed, energy efficiency, smoothness, and joint torque
minimisation. It uses the MuJoCo forward dynamics to evaluate candidate
trajectories against the physics model.

---

---
