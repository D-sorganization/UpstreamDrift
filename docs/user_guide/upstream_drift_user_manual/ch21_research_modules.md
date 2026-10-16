# Chapter 21: Research Modules

The `src/research/` package contains four experimental subsystems that
extend the core simulation framework into advanced control, differentiable
simulation, deformable body dynamics, and multi-robot coordination.
Each module follows the same `PhysicsEngineProtocol` conventions used by
the main engines so that they compose naturally with the rest of the
codebase.

```
src/research/
    mpc/
        controller.py      # Model Predictive Controller
        specialized.py      # CentroidalMPC, WholeBodyMPC
    differentiable/
        engine.py           # DifferentiableEngine, ContactDifferentiableEngine
    deformable/
        objects.py          # SoftBody, Cable, Cloth
    multi_robot/
        system.py           # MultiRobotSystem, TaskCoordinator
        coordination.py     # FormationController, CooperativeManipulation
```

---

## 21.1 Model Predictive Control (MPC)

### 21.1.1 Overview

Model Predictive Control (MPC) solves an open-loop optimal control problem
at every control step, applies the first element of the resulting control
sequence, and re-solves after the system has advanced by one time step.
This **receding-horizon** strategy provides closed-loop feedback despite
relying on an open-loop optimization internally.

The controller lives in `src/research/mpc/controller.py` and is
implemented by the class `ModelPredictiveController`.

### 21.1.2 Mathematical Formulation

At each time step the controller solves:

$$\min_{u_0, \ldots, u_{N-1}} \sum_{k=0}^{N-1} \ell(x_k, u_k) + V_f(x_N)$$

subject to the constraints:

$$x_{k+1} = f(x_k, u_k), \quad x_0 = x_{\text{current}}$$

$$x_k \in \mathcal{X}, \quad u_k \in \mathcal{U} \quad \forall\, k$$

where:

- $N$ is the **prediction horizon** (attribute `horizon`, default 20).
- $f$ is the discrete-time dynamics model, evaluated by calling
  `model.step(dt)` on the wrapped `PhysicsEngineProtocol`.
- $\ell$ is the **stage cost** (running cost).
- $V_f$ is the **terminal cost** evaluated at the final predicted state.
- $\mathcal{X}$ and $\mathcal{U}$ are the state and control constraint
  sets, represented by the `Constraint` dataclass.

### 21.1.3 Cost Function

The cost function is specified through the `CostFunction` dataclass. It
supports both quadratic and linear terms.

**Stage cost** (running cost evaluated at each step $k$):

$$\ell(x, u) = (x - x_{\text{ref}})^T Q\, (x - x_{\text{ref}}) + q^T (x - x_{\text{ref}}) + (u - u_{\text{ref}})^T R\, (u - u_{\text{ref}}) + r^T (u - u_{\text{ref}})$$

When no linear terms are provided ($q = 0$, $r = 0$), this reduces to the
purely quadratic form:

$$\ell(x, u) = (x - x_{\text{ref}})^T Q\, (x - x_{\text{ref}}) + u^T R\, u$$

**Terminal cost** (evaluated at the final predicted state $x_N$):

$$V_f(x) = (x - x_{\text{ref}})^T P\, (x - x_{\text{ref}}) + p^T (x - x_{\text{ref}})$$

If the terminal cost matrix $P$ is `None`, the terminal cost is zero.

The `CostFunction` fields are:

| Field   | Type                 | Description                |
| ------- | -------------------- | -------------------------- | ----------------------------------------- |
| `Q`     | `NDArray (n_x, n_x)` | State cost weight matrix   |
| `R`     | `NDArray (n_u, n_u)` | Control cost weight matrix |
| `P`     | `NDArray (n_x, n_x)  | None`                      | Terminal cost weight matrix               |
| `q`     | `NDArray (n_x,)      | None`                      | Linear state cost (optional)              |
| `r`     | `NDArray (n_u,)      | None`                      | Linear control cost (optional)            |
| `p`     | `NDArray (n_x,)      | None`                      | Linear terminal cost (optional)           |
| `x_ref` | `NDArray             | None`                      | Reference state (1-D or 2-D trajectory)   |
| `u_ref` | `NDArray             | None`                      | Reference control (1-D or 2-D trajectory) |

When `x_ref` is two-dimensional (shape `(T, n_x)`), the reference at
step $k$ is `x_ref[k]`; if $k$ exceeds the length the last entry is
used. The same convention applies to `u_ref`.

```python
from src.research.mpc.controller import CostFunction
import numpy as np

Q = np.diag([100, 100, 10, 10])   # position-heavy
R = np.eye(2) * 0.01              # cheap control
P = Q * 10                        # stabilising terminal cost
cost = CostFunction(Q=Q, R=R, P=P, x_ref=np.array([1, 0, 0, 0]))
```

### 21.1.4 Constraint Specification

Constraints are expressed in the form:

$$\text{lb} \leq A\, x + B\, u \leq \text{ub}$$

and are managed through the `Constraint` dataclass:

| Field             | Type     | Description                          |
| ----------------- | -------- | ------------------------------------ | -------------------------- |
| `A`               | `NDArray | None`                                | State coefficient matrix   |
| `B`               | `NDArray | None`                                | Control coefficient matrix |
| `lb`              | `NDArray | None`                                | Lower bound vector         |
| `ub`              | `NDArray | None`                                | Upper bound vector         |
| `constraint_type` | `str`    | `"state"`, `"control"`, or `"mixed"` |

Setting only `A` with `lb` and `ub` yields a pure state constraint;
setting only `B` yields a pure control constraint. The controller
accumulates constraints via `add_constraint()` and enforces them during
the forward rollout.

```python
from src.research.mpc.controller import Constraint

# Joint limits: -pi <= q <= pi
A_pos = np.eye(n_x)
ctrl.add_constraint(Constraint(
    A=A_pos, lb=-np.pi * np.ones(n_x), ub=np.pi * np.ones(n_x),
    constraint_type="state",
))

# Torque limits: |u| <= 50
ctrl.add_constraint(Constraint(
    B=np.eye(n_u), lb=-50 * np.ones(n_u), ub=50 * np.ones(n_u),
    constraint_type="control",
))
```

### 21.1.5 Solver: Iterative Linear Quadratic Regulator (iLQR)

The `solve()` method implements the **iLQR** algorithm, an iterative
local-approximation method that alternates between:

1. **Backward pass** -- Linearise dynamics around the current trajectory
   and compute the Q-function approximation. At each step $k$ from $N-1$
   down to 0:

   $$
   A_k = \frac{\partial f}{\partial x}\bigg|_{(x_k, u_k)}, \qquad
     B_k = \frac{\partial f}{\partial u}\bigg|_{(x_k, u_k)}
   $$

   The Jacobians $A_k$ and $B_k$ are obtained via forward finite
   differences with perturbation $\epsilon = 10^{-5}$. The Q-function
   matrices are:

   $$Q_x = \ell_x + A_k^T V_x, \qquad Q_u = \ell_u + B_k^T V_x$$

   $$Q_{xx} = \ell_{xx} + A_k^T V_{xx} A_k, \qquad Q_{uu} = \ell_{uu} + B_k^T V_{xx} B_k$$

   $$Q_{ux} = \ell_{ux} + B_k^T V_{xx} A_k$$

   The optimal feedback gain $K_k$ and feed-forward term $d_k$ are:

   $$K_k = -Q_{uu}^{-1} Q_{ux}, \qquad d_k = -Q_{uu}^{-1} Q_u$$

   A Tikhonov regularisation of $10^{-6} I$ is added to $Q_{uu}$ before
   inversion for numerical stability.

2. **Forward pass** -- Roll out new trajectory using the gains with a
   backtracking line search over parameter $\alpha$:

   $$\delta x_k = x_k^{\text{new}} - x_k^{\text{old}}$$

   $$u_k^{\text{new}} = u_k^{\text{old}} + \alpha\, d_k + K_k\, \delta x_k$$

   The line search tries up to 10 values of $\alpha$, halving on each
   iteration ($\alpha \leftarrow 0.5\, \alpha$), and accepts the $\alpha$
   that yields the lowest total cost.

Convergence is declared when the absolute change in cost between
consecutive iterations drops below the tolerance (default $10^{-6}$).
The maximum number of iterations defaults to 100.

### 21.1.6 Solver Settings

Settings are exposed as private attributes that can be changed before
calling `solve()`:

| Attribute            | Default   | Description                          |
| -------------------- | --------- | ------------------------------------ |
| `_max_iterations`    | 100       | Maximum iLQR iterations              |
| `_tolerance`         | $10^{-6}$ | Convergence tolerance on cost change |
| `_line_search_alpha` | 0.5       | Initial line-search step size        |
| `_line_search_beta`  | 0.5       | Line-search back-off factor          |

### 21.1.7 Result Object

`solve()` returns an `MPCResult` dataclass:

| Field                   | Type                 | Description                            |
| ----------------------- | -------------------- | -------------------------------------- |
| `success`               | `bool`               | Whether iLQR converged                 |
| `optimal_states`        | `NDArray (N+1, n_x)` | Predicted state trajectory             |
| `optimal_controls`      | `NDArray (N, n_u)`   | Optimal control sequence               |
| `cost`                  | `float`              | Total cost of the solution             |
| `solve_time`            | `float`              | Wall-clock solver time in seconds      |
| `iterations`            | `int`                | Number of iLQR iterations completed    |
| `constraint_violations` | `float`              | Maximum constraint violation magnitude |

### 21.1.8 Receding-Horizon Loop

In a closed-loop application only the first control action $u_0^*$ is
applied. The helper method `get_first_control(result)` extracts it:

```python
ctrl = ModelPredictiveController(engine, horizon=20, dt=0.01)
ctrl.set_cost_function(cost)

for step in range(simulation_steps):
    result = ctrl.solve(current_state)
    u0 = ctrl.get_first_control(result)
    engine.set_joint_torques(u0)
    engine.step(dt)
    current_state = get_state(engine)
```

### 21.1.9 Specialised MPC Variants

The file `src/research/mpc/specialized.py` provides two domain-specific
subclasses.

#### 21.1.9.1 CentroidalMPC

`CentroidalMPC` uses a simplified **centroidal dynamics** model for
locomotion planning. The state is nine-dimensional:

$$x = \begin{bmatrix} c_x & c_y & c_z & \dot c_x & \dot c_y & \dot c_z & L_x & L_y & L_z \end{bmatrix}^T$$

where $c$ is the centre-of-mass position and $L$ is the angular momentum.
Controls are the 3-D contact forces at each of $n_c$ contact points,
giving $n_u = 3 n_c$.

The dynamics are:

$$
\ddot c = \frac{1}{m}\sum_{i=1}^{n_c} f_i + g, \qquad
  \dot L = \sum_{i=1}^{n_c} (p_i - c) \times f_i
$$

integrated with forward Euler. Default cost weights emphasise CoM
position tracking ($Q_{\text{pos}} = 100 I_3$) with a small control
regulariser ($R = 0.001 I_{n_u}$).

Key methods:

- `set_mass(mass)` -- Set the robot total mass in kilograms.
- `update_contact_positions(positions)` -- Provide world-frame contact
  point locations.
- `add_friction_cone_constraints()` -- Add linearised Coulomb friction
  constraints:
  $f_z \geq 0$, $|f_x| \leq \mu f_z$, $|f_y| \leq \mu f_z$.
- `set_gait_reference(target_velocity, target_height)` -- Generate a
  reference CoM trajectory at constant velocity and height.

#### 21.1.9.2 WholeBodyMPC

`WholeBodyMPC` uses the full rigid-body dynamics model for manipulation
and whole-body control tasks. The state dimension equals twice the number
of generalised coordinates ($n_x = 2 n_q$, positions and velocities),
and controls are joint torques.

Key methods:

- `set_end_effector_target(ee_name, target_pose)` -- Specify a desired
  7-D pose (position + quaternion) for an end effector.
- `set_joint_targets(target_positions, target_velocities)` -- Provide
  reference joint angles and velocities.
- `add_joint_limit_constraints(lower, upper)` -- Enforce position limits.
- `add_torque_limit_constraints(torque_limits)` -- Enforce torque limits:
  $-\tau_{\max} \leq u \leq \tau_{\max}$.
- `solve_with_ee_tracking(initial_state)` -- Converts end-effector
  targets to joint targets via IK and then calls `solve()`.

---

## 21.2 Differentiable Physics

The differentiable-physics subsystem in
`src/research/differentiable/engine.py` wraps any `PhysicsEngineProtocol`
to enable **gradient-based trajectory optimisation**. Two classes are
provided: `DifferentiableEngine` for smooth dynamics and
`ContactDifferentiableEngine` for contact-rich scenarios.

### 21.2.1 DifferentiableEngine

#### Constructor

```python
DifferentiableEngine(
    engine: PhysicsEngineProtocol,
    backend: str = "numpy",       # "jax", "torch", or "numpy"
)
```

The `backend` parameter selects the automatic-differentiation framework.
When set to `"numpy"` (the default), all gradients and Jacobians are
computed via **numerical finite differences** -- no AD library is
required.

#### AutodiffBackend Enum

```python
class AutodiffBackend(Enum):
    JAX   = "jax"
    TORCH = "torch"
    NUMPY = "numpy"
```

#### Forward Simulation

```python
trajectory = engine.simulate_trajectory(initial_state, controls, dt=0.01)
# trajectory.shape == (T+1, n_x)
```

Rolls out the physics for `T = len(controls)` steps, recording
the full state at each instant. The state is partitioned as
$x = [q, v]$ where $q$ has dimension `n_q` and $v$ has dimension `n_v`.

#### Gradient Computation

```python
gradient = engine.compute_gradient(initial_state, controls, loss_fn, dt)
# gradient.shape == controls.shape == (T, n_u)
```

Computes the gradient of a scalar loss with respect to the control
sequence using forward finite differences:

$$\frac{\partial L}{\partial u_{t,i}} \approx \frac{L(u + \epsilon\, e_{t,i}) - L(u)}{\epsilon}$$

where $\epsilon = 10^{-5}$ and $e_{t,i}$ is the unit vector in the
$(t, i)$ direction of the flattened control array. Each partial
derivative requires one additional full forward simulation, making the
cost $\mathcal{O}(T \cdot n_u)$ simulations per gradient evaluation.

#### Jacobian Computation

```python
A, B = engine.compute_jacobian(state, control, dt)
# A.shape == (n_x, n_x)   -- state Jacobian  df/dx
# B.shape == (n_x, n_u)   -- control Jacobian df/du
```

Returns the linearised dynamics around a single operating point
$(x, u)$, computed column-by-column via finite differences:

$$
A_{:,i} = \frac{f(x + \epsilon\, e_i, u) - f(x, u)}{\epsilon}, \qquad
  B_{:,j} = \frac{f(x, u + \epsilon\, e_j) - f(x, u)}{\epsilon}
$$

These Jacobians are the building blocks for LQR, Kalman filtering, and
sensitivity analysis.

#### Trajectory Optimisation

```python
result = engine.optimize_trajectory(
    initial_state, goal_state,
    horizon=50, dt=0.01,
    method="adam",           # or "sgd"
    max_iterations=100,
    learning_rate=0.01,
)
```

Finds a control sequence that drives the system from `initial_state` to
`goal_state` by minimising the squared state error at the final time
step. The loss function is:

$$L(u) = \| x_T - x_{\text{goal}} \|^2$$

When `method="adam"`, the Adam optimiser is used with the standard update
rules. At iteration $t$ the first and second moment estimates are:

$$m_t = \beta_1\, m_{t-1} + (1 - \beta_1)\, \nabla L$$

$$v_t = \beta_2\, v_{t-1} + (1 - \beta_2)\, (\nabla L)^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$u \leftarrow u - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

with $\beta_1 = 0.9$, $\beta_2 = 0.999$, and $\epsilon = 10^{-8}$.
When `method="sgd"`, a simple gradient-descent step is used:
$u \leftarrow u - \eta\, \nabla L$.

The result is an `OptimizationResult` dataclass:

| Field              | Type                 | Description                    |
| ------------------ | -------------------- | ------------------------------ |
| `success`          | `bool`               | True if final cost < 0.1       |
| `optimal_states`   | `NDArray (T+1, n_x)` | Best trajectory found          |
| `optimal_controls` | `NDArray (T, n_u)`   | Best control sequence          |
| `final_cost`       | `float`              | Loss at the best iterate       |
| `iterations`       | `int`                | Number of iterations performed |
| `gradient_norm`    | `float`              | Gradient norm at termination   |

### 21.2.2 ContactDifferentiableEngine

`ContactDifferentiableEngine` extends `DifferentiableEngine` to handle
the **non-smooth** gradient landscape that arises from contact and
collision dynamics. Three smoothing strategies are available:

#### Randomised Smoothing

In `"randomized"` mode, the gradient is estimated as the average of $K$
perturbed gradient evaluations:

$$\nabla L \approx \frac{1}{K} \sum_{k=1}^{K} \nabla L(u + \xi_k), \qquad \xi_k \sim \mathcal{N}(0, \sigma^2 I)$$

The default uses $K = 10$ samples and the `smoothing_factor` attribute
sets $\sigma$. This is the recommended mode for contact-rich problems.

#### Stochastic Smoothing

In `"stochastic"` mode, a single-sample approximation is used (i.e.,
$K = 1$), which is faster but noisier.

#### Standard Smoothing

In the default `"smoothed"` mode, the gradient is computed without
perturbation, relying on the underlying finite-difference scheme. This
is appropriate when contacts are not dominant.

#### Optimisation Through Contact

```python
result = engine.optimize_through_contact(
    initial_state, goal_state,
    contact_schedule=[False, False, True, True, True, False, ...],
    horizon=50, dt=0.01,
    contact_smoothing_multiplier=5.0,
    contact_penalty_weight=0.1,
)
```

This method implements **phase-aware smoothing**: during time steps where
`contact_schedule[t]` is `True`, the smoothing factor is increased by
`contact_smoothing_multiplier` to broaden the non-smooth contact
landscape.

An additional contact-transition penalty discourages velocity
discontinuities at the boundaries between contact and flight phases:

$$C_{\text{contact}} = w_c \sum_{t \in \mathcal{T}_{\text{transition}}} \| v_{t+1} - v_t \|^2$$

where $\mathcal{T}_{\text{transition}}$ is the set of time steps where
the contact flag changes between consecutive steps, and $w_c$ is
`contact_penalty_weight`.

The total loss minimised is:

$$L_{\text{total}} = \| x_T - x_{\text{goal}} \|^2 + C_{\text{contact}}$$

Internally the method uses the Adam optimiser (same parameters as
`optimize_trajectory`) and runs for up to 100 iterations.

---

## 21.3 Deformable Objects

The deformable objects module (`src/research/deformable/objects.py`)
provides finite-element and mass-spring simulations for soft bodies,
cables, and cloth. All classes inherit from the abstract base class
`DeformableObject`.

### 21.3.1 Material Properties

The `MaterialProperties` dataclass specifies physical material
parameters:

| Field               | Type    | Default | Description                    |
| ------------------- | ------- | ------- | ------------------------------ | ------------------------------------ |
| `youngs_modulus`    | `float` | $10^6$  | Young's modulus $E$ in Pascals |
| `poisson_ratio`     | `float` | 0.3     | Poisson's ratio $\nu$          |
| `density`           | `float` | 1000.0  | Material density in kg/m$^3$   |
| `damping`           | `float` | 0.01    | Viscous damping coefficient    |
| `bending_stiffness` | `float  | None`   | `None`                         | Bending stiffness (for shells/cloth) |
| `shear_stiffness`   | `float  | None`   | `None`                         | Shear stiffness (for cloth)          |

Derived elastic moduli are available as computed properties:

$$G = \frac{E}{2(1+\nu)}, \qquad K = \frac{E}{3(1-2\nu)}$$

where $G$ is the **shear modulus** and $K$ is the **bulk modulus**.

### 21.3.2 DeformableObject Base Class

Every deformable object manages:

- `_mesh` / `_rest_mesh` -- Current and rest-configuration node positions,
  each of shape `(N, 3)`.
- `_velocities` -- Node velocities `(N, 3)`.
- `_external_forces` -- Accumulated external forces `(N, 3)`.
- `_fixed_nodes` -- Set of node indices held at their rest positions
  (Dirichlet boundary conditions).

Public interface:

| Method                                  | Description                                        |
| --------------------------------------- | -------------------------------------------------- |
| `get_node_positions()`                  | Returns copy of current node positions             |
| `get_node_velocities()`                 | Returns copy of current node velocities            |
| `set_node_positions(pos)`               | Directly set node positions                        |
| `apply_external_force(indices, forces)` | Apply forces to specified nodes                    |
| `clear_external_forces()`               | Zero out all applied external forces               |
| `fix_nodes(indices)`                    | Pin nodes to rest positions                        |
| `unfix_nodes(indices)`                  | Release pinned nodes                               |
| `reset()`                               | Restore rest configuration, zero velocities/forces |
| `step(dt)` (abstract)                   | Advance simulation by `dt` seconds                 |
| `compute_internal_forces()` (abstract)  | Compute elastic restoring forces                   |

### 21.3.3 SoftBody (Volumetric FEM)

`SoftBody` implements volumetric soft-body simulation using tetrahedral
finite elements with a **neo-Hookean** constitutive model.

#### Constructor

```python
body = SoftBody(
    mesh=node_positions,       # (N, 3)
    tetrahedra=connectivity,   # (M, 4) integer indices
    material=MaterialProperties(youngs_modulus=1e5, poisson_ratio=0.45),
)
```

#### Constitutive Model

For each tetrahedron $e$ with deformation gradient $F$:

1. Compute the deformation gradient:
   $F = D_s\, B_e$ where $D_s$ is the matrix of deformed edge vectors
   and $B_e$ is the inverse of the rest-configuration edge matrix.

2. Compute the First Piola-Kirchhoff stress (neo-Hookean):

   $$P = \mu\,(F - F^{-T}) + \lambda\, \ln(J)\, F^{-T}$$

   where $J = \det(F)$, $\mu$ is the shear modulus, and $\lambda$ is
   the first Lame parameter:

   $$\lambda = \frac{E\,\nu}{(1 + \nu)(1 - 2\nu)}$$

3. Compute nodal forces from the stress:
   $H = -V_0\, P\, B_e^T$
   where $V_0$ is the rest volume of the tetrahedron, computed as:

   $$V_0 = \frac{|\det([v_1 - v_0,\; v_2 - v_0,\; v_3 - v_0])|}{6}$$

4. Distribute $H$ to the four nodes of the tetrahedron.

#### Time Integration

`step(dt)` uses explicit (forward) Euler integration:

$$v_{n+1} = v_n + \frac{f_{\text{internal}} + f_{\text{external}} - c\, v_n}{m_{\text{node}}}\, \Delta t$$

$$x_{n+1} = x_n + v_{n+1}\, \Delta t$$

Node mass is computed as $m_{\text{node}} = \rho \sum V_0 / N$. Fixed
nodes are reset to their rest positions with zero velocity after each
step.

### 21.3.4 Cable

`Cable` simulates a one-dimensional deformable body (rope, wire, shaft)
using a mass-spring model with bending resistance.

```python
cable = Cable(
    mesh=np.linspace([0,0,0], [1,0,0], 20),  # 20 nodes
    material=MaterialProperties(youngs_modulus=1e4, damping=0.1),
)
cable.fix_nodes([0])   # fix one end
```

**Stretch forces.** Between consecutive nodes $i$ and $i+1$:

$$f_{\text{stretch}} = E \cdot \frac{l - l_0}{l_0} \cdot \hat{d}$$

where $l$ is the current distance, $l_0$ is the rest length, and
$\hat{d}$ is the unit direction from node $i$ to node $i+1$.

**Bending forces.** At each interior node $i$ the angle between adjacent
segments is penalised:

$$f_{\text{bend}} = k_b\,(1 - \cos\theta)\, \hat{n}$$

where $\theta$ is the angle between the two segment vectors and
$k_b$ defaults to $0.1 E$ when `bending_stiffness` is not set.

Gravity ($g = -9.81\,\hat{z}$) is applied automatically.

Useful properties:

- `rest_length` -- Total rest length of the cable.
- `get_length()` -- Current total length.
- `get_tension()` -- Average tension (mean force magnitude across nodes).

### 21.3.5 Cloth

`Cloth` simulates a two-dimensional deformable surface using a
grid-based mass-spring system with three spring types:

| Spring Type | Connectivity         | Stiffness Default |
| ----------- | -------------------- | ----------------- |
| Stretch     | Horizontal, vertical | $E$               |
| Shear       | Diagonal             | $0.5\, E$         |
| Bend        | Skip-one (h, v)      | $0.1\, E$         |

```python
# Create 10x10 cloth
w, h = 10, 10
mesh = np.zeros((w * h, 3))
for y in range(h):
    for x in range(w):
        mesh[y * w + x] = [x * 0.1, y * 0.1, 1.0]

cloth = Cloth(mesh, width=w, height=h, material=MaterialProperties())
cloth.fix_nodes([0, w - 1])   # pin top corners
```

The `attach_to_body(body_id, nodes, positions)` method connects
cloth nodes to a rigid body, setting their positions and pinning them.

---

## 21.4 Multi-Robot Systems

The multi-robot subsystem in `src/research/multi_robot/` provides
infrastructure for managing, coordinating, and controlling teams of
robots within a shared simulation environment.

### 21.4.1 Task Model

Tasks are the atomic unit of work assigned to robots. The `Task`
dataclass carries:

| Field                | Type          | Description                                        |
| -------------------- | ------------- | -------------------------------------------------- | ---------------------------------- |
| `task_id`            | `str`         | Unique identifier                                  |
| `task_type`          | `TaskType`    | MOVE_TO, PICK, PLACE, MANIPULATE, INSPECT, or WAIT |
| `target_position`    | `NDArray (3,) | None`                                              | Spatial target for motion tasks    |
| `target_object`      | `str          | None`                                              | Object identifier for manipulation |
| `priority`           | `int`         | Higher values = higher priority                    |
| `status`             | `TaskStatus`  | PENDING, ASSIGNED, IN_PROGRESS, COMPLETED, FAILED  |
| `assigned_robot`     | `str          | None`                                              | Robot currently assigned           |
| `dependencies`       | `list[str]`   | Task IDs that must complete first                  |
| `estimated_duration` | `float`       | Expected completion time (seconds)                 |

The `is_ready(completed_tasks)` method checks whether all
`dependencies` have been satisfied.

### 21.4.2 TaskCoordinator

`TaskCoordinator` is a centralised task scheduler that:

1. Maintains a pool of tasks indexed by `task_id`.
2. Tracks which tasks have been completed.
3. Manages robot-to-task assignments (one task per robot at a time).

Key methods:

| Method                           | Description                                                               |
| -------------------------------- | ------------------------------------------------------------------------- |
| `add_task(task)`                 | Register a task                                                           |
| `remove_task(task_id)`           | Remove a task from the pool                                               |
| `get_ready_tasks()`              | Return pending tasks with satisfied deps, sorted by priority (descending) |
| `assign_task(task_id, robot_id)` | Bind a task to a robot                                                    |
| `start_task(task_id)`            | Transition ASSIGNED to IN_PROGRESS                                        |
| `complete_task(task_id)`         | Transition to COMPLETED                                                   |
| `fail_task(task_id)`             | Transition to FAILED                                                      |
| `get_robot_task(robot_id)`       | Return the task currently assigned to a robot                             |
| `get_available_robots(all_ids)`  | Return robots not currently busy                                          |

### 21.4.3 MultiRobotSystem

`MultiRobotSystem` is the top-level manager for a team of robots. Each
robot is identified by a string ID and is backed by its own
`PhysicsEngineProtocol` instance.

```python
system = MultiRobotSystem()
system.add_robot("arm_1", engine_1, base_pose=np.array([0,0,0, 1,0,0,0]))
system.add_robot("arm_2", engine_2, base_pose=np.array([2,0,0, 1,0,0,0]))
```

Key methods:

- `step_all(dt)` -- Step all robots synchronously.
- `check_inter_robot_collision(safety_distance=0.2)` -- Bounding-sphere
  collision check returning pairs of robot IDs whose base positions are
  closer than `safety_distance`.
- `allocate_tasks(tasks)` -- Greedy allocation: the highest-priority
  ready task is assigned to the closest available robot.
- `get_system_state()` -- Return a serialisable dictionary of all robot
  states (base poses, joint positions, joint velocities).

### 21.4.4 Formation Control

`FormationController` in `coordination.py` computes velocity commands to
maintain a geometric formation among robots.

#### Formation Presets

Three factory methods create common formations:

| Factory                              | Layout                   | Parameters               |
| ------------------------------------ | ------------------------ | ------------------------ |
| `line_formation(n, spacing)`         | Robots in a line along y | `spacing` (m)            |
| `circle_formation(n, radius)`        | Robots on a circle       | `radius` (m)             |
| `wedge_formation(n, spacing, angle)` | V-formation              | `spacing`, `angle` (rad) |

#### Control Law

Given a leader pose $(p_L, R_L)$ and per-robot desired offsets
$\delta_i$ from the `FormationConfig`, the desired world-frame position
for robot $i$ is:

$$p_i^d = p_L + R_L\, \delta_i$$

A proportional-derivative controller computes the velocity command:

$$v_i = K_p\,(p_i^d - p_i) - K_v\, \dot{p}_i$$

Gains default to $K_p = 2.0$ and $K_v = 1.0$. Use `set_gains()` to
adjust.

The formation error is the sum of Euclidean distances:

$$e = \sum_{i} \| p_i^d - p_i \|$$

available via `get_formation_error()`.

### 21.4.5 Cooperative Manipulation

`CooperativeManipulation` handles the case where multiple robots jointly
manipulate a single object.

#### Grasp Matrix

The grasp matrix $G$ maps contact forces to the resultant wrench on the
object:

$$G^T f = w_{\text{object}}$$

where $G \in \mathbb{R}^{6 \times 3 n_c}$, $f$ is the stacked vector of
contact forces, and $w = [f_x, f_y, f_z, \tau_x, \tau_y, \tau_z]^T$ is
the object wrench. The torque contribution uses the skew-symmetric
cross-product matrix:

$$[r]_\times = \begin{bmatrix} 0 & -r_z & r_y \\ r_z & 0 & -r_x \\ -r_y & r_x & 0 \end{bmatrix}$$

#### Load Sharing

`compute_load_sharing(desired_wrench, object_pose)` distributes forces
among the contacts to achieve the desired object wrench while minimising
the total force magnitude. The minimum-norm solution is:

$$f^* = G^T (G\, G^T + \epsilon I)^{-1}\, w$$

with $\epsilon = 10^{-6}$ for regularisation.

#### Force Closure

`check_force_closure(object_pose, friction_coeff)` checks whether the
grasp can resist arbitrary external wrenches by verifying that the grasp
matrix has rank 6. A quality metric (smallest singular value of $G$) is
also returned.

#### Cooperative Motion Planning

`plan_cooperative_motion(goal_pose, current_pose, dt, duration)` generates
smooth end-effector trajectories for all robots using:

1. **Smoothstep** position interpolation:
   $s(t) = 3t^2 - 2t^3$

2. **SLERP** quaternion interpolation:

   $$q(t) = \frac{\sin((1-t)\theta)}{\sin\theta}\, q_0 + \frac{\sin(t\theta)}{\sin\theta}\, q_1$$

   where $\theta = \arccos(q_0 \cdot q_1)$. When $q_0 \cdot q_1 > 0.9995$,
   linear interpolation is used as a numerically stable fallback.

---
