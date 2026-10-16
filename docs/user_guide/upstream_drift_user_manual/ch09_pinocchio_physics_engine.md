# UpstreamDrift User Manual -- Part 3: Physics Engines and Biomechanical Analysis

---

# Chapter 9: Pinocchio Physics Engine

## 9.1 Overview

The Pinocchio engine provides high-performance rigid body dynamics based on Roy Featherstone's
spatial algebra and recursive algorithms. Unlike contact-rich simulators such as MuJoCo,
Pinocchio excels at _analytical_ computations -- Jacobians, dynamics derivatives, mass matrices,
and inverse dynamics -- all with $O(n)$ complexity in the number of joints. Within UpstreamDrift,
Pinocchio serves two primary roles:

1. **Standalone physics engine** via `PinocchioPhysicsEngine`, implementing the full
   `PhysicsEngine` protocol for forward simulation, state management, and drift-control
   decomposition.
2. **Analytical co-processor** via `PinocchioBackend` and `PinocchioWrapper`, providing
   fast Jacobians and dynamics derivatives alongside a MuJoCo-based simulation.

### Source Files

| File                                                                                    | Purpose                                      |
| --------------------------------------------------------------------------------------- | -------------------------------------------- |
| `src/engines/physics_engines/pinocchio/python/pinocchio_physics_engine.py`              | Full `PhysicsEngine` protocol implementation |
| `src/engines/physics_engines/pinocchio/python/dtack/backends/pinocchio_backend.py`      | Standalone backend for URDF-loaded models    |
| `src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/pinocchio_interface.py` | MuJoCo-Pinocchio bridge (`PinocchioWrapper`) |

### Dependencies

```
pip install pin            # Pinocchio (>= 2.6)
pip install numpy          # Required
pip install pink           # Optional: inverse kinematics
```

Availability is checked at import time via `src/shared/python/engine_availability.PINOCCHIO_AVAILABLE`.
When Pinocchio is absent, a `DummyPin` stub prevents `NameError` but raises `ImportError` on any
actual usage.

---

## 9.2 Core Algorithms

Pinocchio implements three foundational $O(n)$ algorithms for an articulated rigid body system
with $n$ joints, configuration vector $q \in \mathbb{R}^{n_q}$, velocity vector
$\dot{q} \in \mathbb{R}^{n_v}$, and applied torque $\tau \in \mathbb{R}^{n_v}$.

### 9.2.1 Recursive Newton-Euler Algorithm (RNEA) -- Inverse Dynamics

Given a desired acceleration $\ddot{q}$, RNEA computes the required joint torques:

$$\tau = \text{RNEA}(q, \dot{q}, \ddot{q}) = M(q)\,\ddot{q} + C(q, \dot{q})\,\dot{q} + g(q)$$

where $M(q)$ is the joint-space inertia matrix, $C(q, \dot{q})$ captures Coriolis and centrifugal
terms, and $g(q)$ is the gravity vector.

**API** (`PinocchioPhysicsEngine`):

```python
def compute_inverse_dynamics(self, qacc: np.ndarray) -> np.ndarray:
    """Compute inverse dynamics tau = ID(q, v, a).

    Args:
        qacc: Desired acceleration vector (n_v,) [rad/s^2 or m/s^2]

    Returns:
        tau: Required generalized forces (n_v,) [N*m or N]
    """
```

**API** (`PinocchioBackend`):

```python
def compute_inverse_dynamics(
    self,
    q: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    a: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute inverse dynamics (RNEA).

    Args:
        q: Joint positions [nq]
        v: Joint velocities [nv]
        a: Joint accelerations [nv]

    Returns:
        Joint torques [nv]
    """
```

**Usage**:

```python
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)

engine = PinocchioPhysicsEngine()
engine.load_from_path("/path/to/humanoid.urdf")

# Compute torques to hold current position (zero acceleration)
qacc_zero = np.zeros(engine.model.nv)
tau_gravity_compensation = engine.compute_inverse_dynamics(qacc_zero)
# tau_gravity_compensation contains the torques needed to resist gravity
```

**Special case -- Bias forces**: Setting $\ddot{q} = 0$ yields the bias force vector
$b(q, \dot{q}) = C(q, \dot{q})\,\dot{q} + g(q)$:

```python
bias = engine.compute_bias_forces()
# Equivalent to: pin.rnea(model, data, q, v, np.zeros(nv))
```

**Special case -- Gravity vector**: Setting $\ddot{q} = 0$ and $\dot{q} = 0$ yields
$g(q)$:

```python
gravity = engine.compute_gravity_forces()
# Uses: pin.computeGeneralizedGravity(model, data, q)
```

### 9.2.2 Articulated Body Algorithm (ABA) -- Forward Dynamics

Given applied torques $\tau$, ABA computes the resulting accelerations:

$$\ddot{q} = \text{ABA}(q, \dot{q}, \tau) = M(q)^{-1}\bigl(\tau - C(q,\dot{q})\,\dot{q} - g(q)\bigr)$$

This is the central algorithm for forward simulation and counterfactual analysis.

**API** (`PinocchioPhysicsEngine.step`):

```python
def step(self, dt: float | None = None) -> None:
    """Advance the simulation by one time step.

    Uses ABA for forward dynamics, then semi-implicit Euler integration:
        a = ABA(q, v, tau)
        v_next = v + a * dt
        q_next = integrate(q, v_next * dt)
    """
```

**API** (`PinocchioBackend`):

```python
def compute_forward_dynamics(
    self,
    q: npt.NDArray[np.float64],
    v: npt.NDArray[np.float64],
    tau: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Compute forward dynamics (ABA).

    Returns:
        Joint accelerations [nv]
    """
```

**Usage** -- Forward simulation loop:

```python
engine = PinocchioPhysicsEngine()
engine.load_from_path("robot.urdf")

# Apply constant torque and simulate
tau = np.array([0.0, 0.0, 5.0, -2.0])  # Example torques
engine.set_control(tau)

for _ in range(1000):
    engine.step(dt=0.001)
    q, v = engine.get_state()
    print(f"t={engine.get_time():.3f}s, q={q}")
```

### 9.2.3 Composite Rigid Body Algorithm (CRBA) -- Mass Matrix

CRBA computes the joint-space inertia matrix $M(q) \in \mathbb{R}^{n_v \times n_v}$:

$$M(q) = \sum_{i=1}^{n} J_i(q)^T \, \mathcal{I}_i \, J_i(q)$$

where $J_i$ is the Jacobian of body $i$ and $\mathcal{I}_i$ is its spatial inertia.

**API**:

```python
def compute_mass_matrix(self) -> np.ndarray:
    """Compute the dense inertia matrix M(q).

    Returns:
        M: (n_v, n_v) symmetric positive definite mass matrix.

    Notes:
        Pinocchio's CRBA fills only the upper triangular part.
        The engine symmetrizes: M = triu(M) + triu(M,1).T
    """
```

**Usage**:

```python
M = engine.compute_mass_matrix()

# Verify symmetry and positive definiteness
assert np.allclose(M, M.T), "Mass matrix must be symmetric"
eigenvalues = np.linalg.eigvalsh(M)
assert np.all(eigenvalues > 0), "Mass matrix must be positive definite"

# Kinetic energy
v = engine.v
T = 0.5 * v.T @ M @ v
```

---

## 9.3 Fast Analytical Jacobians and Derivatives

One of Pinocchio's key strengths is analytical (not finite-difference) computation of Jacobians
and dynamics derivatives. This makes it ideal for trajectory optimization and sensitivity analysis.

### 9.3.1 Frame Jacobians

The geometric Jacobian maps joint velocities to the spatial velocity of a frame:

$$\begin{pmatrix} v \\ \omega \end{pmatrix} = J(q) \, \dot{q}$$

where $J(q) \in \mathbb{R}^{6 \times n_v}$ and $v, \omega$ are the linear and angular velocities
of the frame.

The Jacobian time derivative satisfies:

$$\dot{J}(q, \dot{q}) = \frac{\partial J(q)}{\partial q}\,\dot{q}$$

This is critical for operational-space control where the task-space acceleration is:

$$\ddot{x} = J(q)\,\ddot{q} + \dot{J}(q, \dot{q})\,\dot{q}$$

**API** (`PinocchioPhysicsEngine`):

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    """Compute spatial Jacobian for a specific body.

    Args:
        body_name: Name of the body frame in the model.

    Returns:
        Dictionary with keys:
          - 'linear':  (3, n_v) position Jacobian  [m/rad or m/m]
          - 'angular': (3, n_v) rotation Jacobian   [rad/rad]
          - 'spatial': (6, n_v) combined [angular; linear]
        Returns None if body_name not found.
    """
```

**API** (`PinocchioBackend`):

```python
def compute_frame_jacobian(
    self,
    q: npt.NDArray[np.float64],
    frame_id: int | str,
    reference_frame: int = pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
) -> npt.NDArray[np.float64]:
    """Compute frame Jacobian.

    Args:
        q: Joint positions [nq]
        frame_id: Frame ID (int) or frame name (str)
        reference_frame: Reference frame for Jacobian
            - pin.LOCAL: Body-fixed frame
            - pin.WORLD: World frame
            - pin.LOCAL_WORLD_ALIGNED: Body origin, world orientation (default)

    Returns:
        Jacobian matrix [6 x nv]
    """
```

**Usage** -- Club head velocity computation:

```python
engine.forward()  # Update kinematics

jac = engine.compute_jacobian("club_head")
if jac is not None:
    # Linear velocity of club head
    v_club = jac["linear"] @ engine.v   # (3,) in m/s
    club_speed = np.linalg.norm(v_club)
    print(f"Club head speed: {club_speed:.1f} m/s")

    # Angular velocity of club head
    omega_club = jac["angular"] @ engine.v  # (3,) in rad/s
```

### 9.3.2 Analytical Dynamics Derivatives

Pinocchio can compute the partial derivatives of the forward dynamics equations analytically,
which is essential for trajectory optimization and model-predictive control.

Given forward dynamics $\ddot{q} = f(q, \dot{q}, \tau)$, the derivatives are:

$$\frac{\partial \ddot{q}}{\partial q}, \quad \frac{\partial \ddot{q}}{\partial \dot{q}}, \quad \frac{\partial \ddot{q}}{\partial \tau} = M(q)^{-1}$$

**API** (`PinocchioWrapper` -- MuJoCo bridge):

```python
def compute_dynamics_derivatives(
    self,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
    tau: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute analytical derivatives of dynamics.

    Returns:
        Tuple of (df_dq, df_dv, df_dtau, df_du):
          - df_dq:   (n_v, n_v) partial of acceleration w.r.t. position
          - df_dv:   (n_v, n_v) partial of acceleration w.r.t. velocity
          - df_dtau: (n_v, n_v) partial of acceleration w.r.t. torque = M^{-1}
          - df_du:   (n_v, n_v) partial of acceleration w.r.t. control
    """
```

Under the hood this calls:

```python
pin.computeABADerivatives(self.pin_model, self.pin_data, q, v, tau)
# Results stored in:
#   self.pin_data.ddq_dq   -- partial(ddot{q}) / partial(q)
#   self.pin_data.ddq_dv   -- partial(ddot{q}) / partial(dot{q})
#   self.pin_data.Minv     -- M(q)^{-1} = partial(ddot{q}) / partial(tau)
```

**Usage** -- Linearization for LQR control:

```python
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.pinocchio_interface import (
    PinocchioWrapper,
)

wrapper = PinocchioWrapper(mj_model, mj_data)
wrapper.sync_mujoco_to_pinocchio()

# Get linearization matrices at current state
A_qq, A_qv, B, _ = wrapper.compute_dynamics_derivatives()

# Continuous-time state-space: dx/dt = A x + B u
# where x = [q; v], u = tau
n = wrapper.pin_model.nv
A = np.block([
    [np.zeros((n, n)), np.eye(n)],
    [A_qq,             A_qv     ]
])
# B is already M^{-1}
```

### 9.3.3 Coriolis Matrix

Pinocchio provides direct computation of the Coriolis matrix $C(q, \dot{q})$ satisfying:

$$M(q)\,\ddot{q} + C(q, \dot{q})\,\dot{q} + g(q) = \tau$$

**API** (`PinocchioWrapper`):

```python
def compute_coriolis_matrix(
    self,
    q: np.ndarray | None = None,
    v: np.ndarray | None = None,
) -> np.ndarray:
    """Compute Coriolis matrix [nv x nv].

    Uses: pin.computeCoriolisMatrix(model, data, q, v)
    """
```

---

## 9.4 Drift-Control Decomposition (Section F)

Pinocchio implements the Section F drift-control decomposition, which separates total
acceleration into passive and active components:

$$\ddot{q}_{full} = \underbrace{\ddot{q}_{drift}}_{\text{passive}} + \underbrace{\ddot{q}_{control}}_{\text{active}}$$

### 9.4.1 Drift Acceleration

The drift (passive) acceleration is what would occur with zero applied torques. It arises
from gravity, Coriolis, and centrifugal effects:

$$\ddot{q}_{drift} = M(q)^{-1}\bigl(-C(q, \dot{q})\,\dot{q} - g(q)\bigr) = \text{ABA}(q, \dot{q}, \mathbf{0})$$

```python
def compute_drift_acceleration(self) -> np.ndarray:
    """Compute passive (drift) acceleration with zero control inputs.

    Uses ABA with zero torque: pin.aba(model, data, q, v, zeros(nv))
    """
```

### 9.4.2 Control Acceleration

The control acceleration is the component due solely to applied torques:

$$\ddot{q}_{control} = M(q)^{-1}\,\tau$$

```python
def compute_control_acceleration(self, tau: np.ndarray) -> np.ndarray:
    """Compute control-attributed acceleration.

    Uses: np.linalg.solve(M, tau)
    """
```

### 9.4.3 Superposition Verification

The critical contract is: $\ddot{q}_{drift} + \ddot{q}_{control} = \ddot{q}_{full}$.

```python
# Verify superposition
a_drift = engine.compute_drift_acceleration()
a_control = engine.compute_control_acceleration(engine.tau)
a_full = pin.aba(engine.model, engine.data, engine.q, engine.v, engine.tau)

assert np.allclose(a_drift + a_control, a_full, atol=1e-10), \
    "Superposition violated!"
```

---

## 9.5 Counterfactual Experiments (Section G)

### 9.5.1 Zero-Torque Counterfactual (ZTCF)

ZTCF answers: "What would happen if all actuators turned off right now?"

$$\ddot{q}_{ZTCF} = \text{ABA}(q, \dot{q}, \mathbf{0})$$

```python
def compute_ztcf(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Zero-Torque Counterfactual (ZTCF) - Guideline G1.

    Args:
        q: Joint positions (n_q,) [rad or m]
        v: Joint velocities (n_v,) [rad/s or m/s]

    Returns:
        Acceleration under zero applied torque (n_v,)
    """
```

### 9.5.2 Zero-Velocity Counterfactual (ZVCF)

ZVCF answers: "What acceleration would occur if motion froze instantaneously?"

$$\ddot{q}_{ZVCF} = \text{ABA}(q, \mathbf{0}, \tau)$$

This isolates gravity and control from Coriolis/centrifugal effects. The velocity-dependent
contribution can then be computed as:

$$\Delta \ddot{q}_{velocity} = \ddot{q}_{full} - \ddot{q}_{ZVCF}$$

```python
def compute_zvcf(self, q: np.ndarray) -> np.ndarray:
    """Zero-Velocity Counterfactual (ZVCF) - Guideline G2.

    Args:
        q: Joint positions (n_q,) [rad or m]

    Returns:
        Acceleration with v=0, preserving current control (n_v,)
    """
```

**Usage** -- Downswing analysis:

```python
# During downswing, separate gravity from centrifugal "whip"
q, v = engine.get_state()

a_full = pin.aba(engine.model, engine.data, q, v, engine.tau)
a_zvcf = engine.compute_zvcf(q)   # No Coriolis/centrifugal
a_ztcf = engine.compute_ztcf(q, v)  # No control torques

a_coriolis_centrifugal = a_full - a_zvcf
a_control_effect = a_full - a_ztcf

print(f"Centrifugal contribution: {np.linalg.norm(a_coriolis_centrifugal):.2f} rad/s^2")
print(f"Control contribution:     {np.linalg.norm(a_control_effect):.2f} rad/s^2")
```

---

## 9.6 URDF Model Loading and Joint Configuration

Pinocchio primarily loads models from URDF (Unified Robot Description Format) files.

### 9.6.1 Loading From URDF

```python
engine = PinocchioPhysicsEngine()
engine.load_from_path("/path/to/model.urdf")

# After loading:
print(f"Model: {engine.model_name}")
print(f"Configuration DOF (nq): {engine.model.nq}")
print(f"Velocity DOF (nv): {engine.model.nv}")
print(f"Joints: {engine.get_joint_names()}")
```

The distinction between $n_q$ (configuration) and $n_v$ (velocity) dimensions matters for
models with non-Euclidean joints. For example, a free-floating base uses quaternions
($n_q = 7$) but has 6 velocity DOF ($n_v = 6$).

### 9.6.2 Loading From XML String

```python
engine.load_from_string(urdf_content, extension="urdf")
# Uses: pin.buildModelFromXML(content)
```

### 9.6.3 Neutral Configuration

After loading, the engine initializes to the neutral (zero) configuration:

```python
q0 = pin.neutral(engine.model)  # Neutral configuration
engine.q = q0
engine.v = np.zeros(engine.model.nv)
```

### 9.6.4 Configuration Space Integration

Pinocchio handles Lie group integration correctly for non-Euclidean joints:

```python
# Correct integration on manifold (handles quaternions, SO(3), etc.)
q_next = pin.integrate(model, q, v * dt)
# NOT: q_next = q + v * dt  (wrong for quaternions!)
```

This is automatically used in `PinocchioPhysicsEngine.step()`.

---

## 9.7 MuJoCo-Pinocchio Bridge

The `PinocchioWrapper` class enables using Pinocchio's analytical algorithms alongside
MuJoCo's simulation capabilities. This is the recommended architecture for golf swing
analysis: MuJoCo handles contacts and constraints while Pinocchio provides fast Jacobians
and derivatives.

### 9.7.1 Initialization

```python
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.pinocchio_interface import (
    PinocchioWrapper,
    create_pinocchio_wrapper,
)

# From MuJoCo model
wrapper = create_pinocchio_wrapper(mj_model, mj_data)

# Or directly
wrapper = PinocchioWrapper(mj_model, mj_data, use_mjcf_parser=True)
```

The wrapper converts MuJoCo models to URDF via a temporary file, then loads the URDF in
Pinocchio.

### 9.7.2 Quaternion Convention Conversion

MuJoCo and Pinocchio use different quaternion conventions for free joints:

- MuJoCo: $(w, x, y, z)$
- Pinocchio: $(x, y, z, w)$

The wrapper handles this automatically:

```python
# Internal conversion (transparent to user)
q_pin = wrapper._mujoco_q_to_pinocchio_q(mj_data.qpos)
q_mj  = wrapper._pinocchio_q_to_mujoco_q(q_pin)
```

### 9.7.3 State Synchronization

```python
# After MuJoCo simulation step:
mujoco.mj_step(mj_model, mj_data)

# Sync to Pinocchio for analytical computation
wrapper.sync_mujoco_to_pinocchio()

# Now use Pinocchio's fast algorithms
J = wrapper.compute_end_effector_jacobian("club_head")
M = wrapper.compute_mass_matrix()
KE = wrapper.compute_kinetic_energy()
PE = wrapper.compute_potential_energy()
```

### 9.7.4 Energy Computation

```python
# Kinetic energy: T = 0.5 * v^T M v
KE = wrapper.compute_kinetic_energy()

# Potential energy via Pinocchio's built-in
PE = wrapper.compute_potential_energy()

# Total mechanical energy
E = KE + PE
```

---

## 9.8 PinocchioBackend for Standalone Usage

The `PinocchioBackend` class provides a simpler interface for standalone dynamics
computations without the full `PhysicsEngine` protocol overhead.

```python
from src.engines.physics_engines.pinocchio.python.dtack.backends.pinocchio_backend import (
    PinocchioBackend,
)

backend = PinocchioBackend("/path/to/robot.urdf")

q = np.zeros(backend.model.nq)
v = np.zeros(backend.model.nv)

# Mass matrix
M = backend.compute_mass_matrix(q)

# Bias forces (gravity + Coriolis)
# Uses pin.nle() which is more efficient than rnea with zero acceleration
b = backend.compute_bias_forces(q, v)

# Forward kinematics -- all frame placements
frames = backend.forward_kinematics(q)
for i, frame in enumerate(frames):
    print(f"Frame {i}: position = {frame.translation}")
```

---

## 9.9 PINK Inverse Kinematics Integration

Pinocchio integrates with PINK (Pinocchio INverse Kinematics) for solving IK problems
using the computed Jacobians. Given a target frame placement $X_{target} \in SE(3)$,
IK solves:

$$\Delta q = J(q)^{\dagger} \, \text{Log}(X_{current}^{-1} \, X_{target})$$

where $J^{\dagger}$ is the Moore-Penrose pseudo-inverse and Log is the $SE(3)$ logarithm map.

```python
import pink

# Define IK task
task = pink.FrameTask(
    "club_head",
    position_cost=1.0,
    orientation_cost=0.5,
)

# Set target
task.set_target(target_transform)

# Solve IK
configuration = pink.Configuration(engine.model, engine.data, engine.q)
velocity = pink.solve_ik(configuration, [task], dt=0.01)
engine.q = pin.integrate(engine.model, engine.q, velocity * 0.01)
```

---

## 9.10 Performance Comparison With MuJoCo

| Operation               | Pinocchio           | MuJoCo             | Notes                             |
| ----------------------- | ------------------- | ------------------ | --------------------------------- |
| Forward dynamics (ABA)  | $O(n)$ analytical   | $O(n)$ analytical  | Comparable speed                  |
| Inverse dynamics (RNEA) | $O(n)$ analytical   | `mj_inverse`       | Pinocchio more flexible           |
| Mass matrix (CRBA)      | $O(n^2)$ analytical | `mj_fullM`         | Comparable                        |
| Jacobians               | Analytical, $O(n)$  | Analytical         | Both fast                         |
| Dynamics derivatives    | Analytical, $O(n)$  | Finite differences | **Pinocchio far superior**        |
| Contact handling        | Limited             | Excellent          | Use MuJoCo for contacts           |
| Constraint dynamics     | Via ProxQP solver   | Built-in           | MuJoCo more mature                |
| Lie group integration   | Native              | Not applicable     | Pinocchio handles SO(3) correctly |

**Recommendation**: Use the `PinocchioWrapper` bridge for golf swing analysis. MuJoCo
simulates the swing with contacts, while Pinocchio provides analytical Jacobians and
derivatives for optimization and control design.

---

## 9.11 Complete Workflow Example

```python
import numpy as np
from src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine import (
    PinocchioPhysicsEngine,
)

# 1. Initialize engine
engine = PinocchioPhysicsEngine()
engine.load_from_path("models/humanoid_golfer.urdf")

# 2. Set initial configuration (address position)
q_address = np.array([...])  # Pre-computed address pose
engine.set_state(q_address, np.zeros(engine.model.nv))

# 3. Forward kinematics
engine.forward()
jac = engine.compute_jacobian("club_head")

# 4. Compute dynamics quantities
M = engine.compute_mass_matrix()
g = engine.compute_gravity_forces()
b = engine.compute_bias_forces()

# 5. Simulate with torque control
for step in range(5000):
    # PD control example
    q, v = engine.get_state()
    tau = -100.0 * (q - q_target) - 20.0 * v  # PD gains

    engine.set_control(tau)
    engine.step(dt=0.001)

    # Analyze drift vs control at each step
    a_drift = engine.compute_drift_acceleration()
    a_control = engine.compute_control_acceleration(tau)

# 6. Batched state query (performance optimization)
full_state = engine.get_full_state()
q, v, t, M = full_state["q"], full_state["v"], full_state["t"], full_state["M"]
```

---
