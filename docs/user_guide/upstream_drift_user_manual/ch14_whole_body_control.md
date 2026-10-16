# Chapter 14: Whole-Body Control

The whole-body control module (`src/robotics/control/whole_body/`) implements a
hierarchical, optimization-based controller that resolves multiple competing
objectives for complex robotic systems. The module comprises three primary
files: task descriptors (`task.py`), QP solvers (`qp_solver.py`), and the
controller itself (`wbc_controller.py`).

---

## 14.1 Task-Based Control

### 14.1.1 Task Types

The `TaskType` enum classifies how a task participates in the optimization:

| Member       | Semantics                                                                                                  |
| ------------ | ---------------------------------------------------------------------------------------------------------- |
| `EQUALITY`   | Hard equality constraint: $J \ddot{\mathbf{q}} = \ddot{\mathbf{x}}_d$. Used for contact constraints.       |
| `INEQUALITY` | Bounded constraint: $\mathbf{lb} \leq J \ddot{\mathbf{q}} \leq \mathbf{ub}$. Used for joint limits.        |
| `SOFT`       | Weighted objective: minimize $\|J \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_d\|_W^2$. Used for tracking tasks. |

### 14.1.2 Task Priority Levels

The `TaskPriority` enum (from `src/robotics/core/types.py`) defines five
priority levels, with lower numeric values indicating higher priority:

| Level             | Value | Typical usage                           |
| ----------------- | ----- | --------------------------------------- |
| `HARD_CONSTRAINT` | 0     | Contact constraints, equation of motion |
| `SAFETY`          | 1     | Joint limit avoidance                   |
| `PRIMARY`         | 2     | CoM tracking, balance                   |
| `SECONDARY`       | 3     | End-effector tracking                   |
| `TERTIARY`        | 4     | Posture regularization                  |

In hierarchical mode, tasks at priority level $k$ are solved in the null space
of all tasks at levels $0$ through $k-1$, ensuring that lower-priority tasks
cannot interfere with higher-priority objectives.

### 14.1.3 The Task Dataclass

```python
@dataclass
class Task:
    name: str
    task_type: TaskType
    priority: int
    jacobian: NDArray[np.float64]     # (task_dim, n_v)
    target: NDArray[np.float64]       # (task_dim,)
    weight: NDArray[np.float64] | None = None
    lower_bound: NDArray[np.float64] | None = None
    upper_bound: NDArray[np.float64] | None = None
    gain_p: float = 100.0
    gain_d: float = 20.0
```

**Validation.** `__post_init__` verifies:

- `jacobian` is 2D with shape `(task_dim, n_v)`.
- `target` has shape `(task_dim,)` consistent with the Jacobian row count.
- `weight` (if provided) has shape `(task_dim,)`.
- Inequality tasks have at least one bound.
- All values are finite (no NaN or Inf).

**Properties:**

- `task_dim`: the number of task-space dimensions (rows of the Jacobian).
- `config_dim`: the number of configuration-space dimensions (columns of the Jacobian).

**`get_weight_matrix()`** returns a `(task_dim, task_dim)` diagonal matrix. If
`weight` is provided, it becomes the diagonal; otherwise, the identity matrix is
returned.

**`compute_error_feedback(position_error, velocity_error)`** computes the desired
task-space acceleration using PD control:

$$
\ddot{\mathbf{x}}_d = \ddot{\mathbf{x}}_{\text{ff}} + K_p \mathbf{e}_p + K_d \mathbf{e}_v
$$

where $\ddot{\mathbf{x}}_{\text{ff}}$ is the feedforward target (stored in
`self.target`), $\mathbf{e}_p$ is the position error, and $\mathbf{e}_v$ is the
velocity error. The gains $K_p$ and $K_d$ are scalar (applied uniformly across
all task dimensions).

### 14.1.4 Task Creation Utilities

Four factory functions create properly configured tasks:

**`create_com_task()`** -- Center-of-mass tracking.

```python
task = create_com_task(
    jacobian_com,           # (3, n_v) CoM Jacobian
    com_current,            # (3,) current CoM position
    com_target,             # (3,) desired CoM position
    com_velocity,           # (3,) current CoM velocity
    com_velocity_target=None,  # (3,) desired CoM velocity (default: zero)
    weight=1.0,
    priority=2,             # PRIMARY
    gain_p=100.0,
    gain_d=20.0,
)
```

The target acceleration is computed as:

$$
\ddot{\mathbf{x}}_{\text{CoM}} = K_p (\mathbf{x}_d - \mathbf{x}) + K_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
$$

**`create_posture_task()`** -- Joint-space posture regulation.

```python
task = create_posture_task(
    n_v,                    # number of velocity DOFs
    q_current,              # current joint positions
    q_target,               # desired joint positions
    v_current,              # current joint velocities
    weight=0.1,
    priority=4,             # TERTIARY
    gain_p=50.0,
    gain_d=10.0,
    mask=None,              # boolean mask for active joints
)
```

The Jacobian is an identity-like selector matrix that picks out the active
joints: `jacobian = np.eye(n_v)[mask]`.

**`create_ee_task()`** -- End-effector tracking.

```python
task = create_ee_task(
    jacobian_ee,            # (6, n_v) or (3, n_v)
    ee_current,             # current EE pose or position
    ee_target,              # desired EE pose or position
    ee_velocity,            # current EE twist or velocity
    ee_velocity_target=None,
    weight=1.0,
    priority=3,             # SECONDARY
    gain_p=100.0,
    gain_d=20.0,
    position_only=False,    # if True, use only position (3 DOF)
)
```

When `position_only=True`, the Jacobian is truncated to its first 3 rows and
only position components of the pose/velocity are used.

**`create_contact_constraint()`** -- Zero-velocity contact constraint.

```python
task = create_contact_constraint(
    jacobian_contact,       # (3, n_v) or (6, n_v)
    contact_velocity,       # should be ~0
    priority=0,             # HARD_CONSTRAINT
)
```

Creates an equality task with target zero acceleration, enforcing
$J_c \ddot{\mathbf{q}} = \mathbf{0}$ (assuming $\dot{J}_c \dot{\mathbf{q}} \approx \mathbf{0}$).

**`create_joint_limit_task()`** -- Repulsive joint-limit avoidance.

Only creates a task when at least one joint is within `margin` radians of its
limit. The repulsive acceleration pushes the joint away from the limit with gain
proportional to the proximity.

---

## 14.2 QP Solver

### 14.2.1 Problem Formulation

The `QPProblem` dataclass encodes a standard quadratic program:

$$
\min_{\mathbf{x}} \; \frac{1}{2} \mathbf{x}^T H \mathbf{x} + \mathbf{g}^T \mathbf{x}
$$

subject to:

$$
A_{\text{eq}} \mathbf{x} = \mathbf{b}_{\text{eq}}
$$

$$
\mathbf{lb} \leq A_{\text{ineq}} \mathbf{x} \leq \mathbf{ub}
$$

$$
\mathbf{x}_{\text{lb}} \leq \mathbf{x} \leq \mathbf{x}_{\text{ub}}
$$

| Field     | Shape                   | Description                  |
| --------- | ----------------------- | ---------------------------- |
| `H`       | `(n, n)`                | Hessian (must be PSD)        |
| `g`       | `(n,)`                  | Linear cost vector           |
| `A_eq`    | `(m_eq, n)` or `None`   | Equality constraint matrix   |
| `b_eq`    | `(m_eq,)` or `None`     | Equality constraint RHS      |
| `A_ineq`  | `(m_ineq, n)` or `None` | Inequality constraint matrix |
| `lb_ineq` | `(m_ineq,)` or `None`   | Inequality lower bounds      |
| `ub_ineq` | `(m_ineq,)` or `None`   | Inequality upper bounds      |
| `x_lb`    | `(n,)` or `None`        | Variable lower bounds        |
| `x_ub`    | `(n,)` or `None`        | Variable upper bounds        |

**Validation:**

- $H$ must be square.
- $\mathbf{g}$ must have the same dimension as $H$.
- If $A_{\text{eq}}$ is provided, $\mathbf{b}_{\text{eq}}$ is also required.
- Column counts of constraint matrices must equal $n$.

### 14.2.2 QPSolution

The `QPSolution` dataclass carries the solver output:

| Field        | Type                | Description                    |
| ------------ | ------------------- | ------------------------------ |
| `success`    | `bool`              | Did the solver converge?       |
| `x`          | `NDArray` or `None` | Optimal solution               |
| `cost`       | `float`             | Optimal cost (`inf` if failed) |
| `iterations` | `int`               | Iteration count                |
| `solve_time` | `float`             | Wall-clock time in seconds     |
| `status`     | `str`               | Solver-specific message        |
| `dual_eq`    | `NDArray` or `None` | Equality dual variables        |
| `dual_ineq`  | `NDArray` or `None` | Inequality dual variables      |

### 14.2.3 Solver Implementations

**Abstract base:** `QPSolver` defines the interface with two methods:
`solve(problem) -> QPSolution` and `is_available() -> bool`.

**ScipyQPSolver** -- uses `scipy.optimize.minimize` with SLSQP or trust-constr:

- Constructs the objective function $f(\mathbf{x}) = \frac{1}{2}\mathbf{x}^T H \mathbf{x} + \mathbf{g}^T \mathbf{x}$ and its gradient $\nabla f = H\mathbf{x} + \mathbf{g}$.
- Equality constraints become `{'type': 'eq', 'fun': lambda x: A_eq @ x - b_eq}`.
- Double-sided inequality $\mathbf{lb} \leq A\mathbf{x} \leq \mathbf{ub}$ is split into two one-sided SciPy inequality constraints for each row: $A_i \mathbf{x} - lb_i \geq 0$ and $ub_i - A_i \mathbf{x} \geq 0$.
- Variable bounds are passed as a `scipy.optimize.Bounds` object.

**NullspaceQPSolver** -- a lightweight solver using the KKT system directly:

For equality-constrained QPs, the KKT conditions are:

$$
\begin{bmatrix}
H + \epsilon I & A_{\text{eq}}^T \\
A_{\text{eq}} & 0
\end{bmatrix}
\begin{bmatrix}
\mathbf{x} \\
\boldsymbol{\lambda}
\end{bmatrix}
=
\begin{bmatrix}
-\mathbf{g} \\
\mathbf{b}_{\text{eq}}
\end{bmatrix}
$$

where $\epsilon$ is the regularization parameter (default $10^{-6}$). This
system is solved directly via `np.linalg.solve`. For unconstrained QPs, the
system reduces to $(H + \epsilon I)\mathbf{x} = -\mathbf{g}$.

The `NullspaceQPSolver` is always available (NumPy-only dependency) and is fast
for small problems but does not handle inequality constraints.

**`create_default_solver()`** returns `ScipyQPSolver` if available, otherwise
`NullspaceQPSolver`.

### 14.2.4 Priority-Based Task Stacking

In weighted mode, all tasks are combined into a single QP. The Hessian and
gradient accumulate contributions from each task:

$$
H \mathrel{+}= J_i^T W_i J_i, \qquad \mathbf{g} \mathrel{+}= -J_i^T W_i \ddot{\mathbf{x}}_{d,i}
$$

This is equivalent to minimizing:

$$
\sum_i \|J_i \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_{d,i}\|_{W_i}^2
$$

### 14.2.5 Nullspace Projection for Hierarchical Control

In hierarchical (HQP) mode, tasks are grouped by priority and solved sequentially.
At each level $k$, the accumulated Jacobians from all higher-priority tasks
define a constraint subspace. The nullspace projector is:

$$
N_k = I - J_{\leq k-1}^+ J_{\leq k-1}
$$

where $J_{\leq k-1}$ is the vertical stack of all Jacobians at priorities
$0, 1, \ldots, k-1$, and $J^+$ denotes the Moore-Penrose pseudoinverse. The
current-level task Jacobian is projected:

$$
J_k^{\text{proj}} = J_k N_k
$$

This ensures that the optimization at level $k$ operates only in the subspace
that does not disturb higher-priority tasks. The general recursion for the
resulting torque at each level is:

$$
\boldsymbol{\tau}_k = \boldsymbol{\tau}_{k-1}
  + N_{k-1}\bigl(\boldsymbol{\tau}_k^* - \boldsymbol{\tau}_{k-1}\bigr)
$$

where $\boldsymbol{\tau}_k^*$ is the optimal torque for level $k$ in isolation.

---

## 14.3 Control Modes

### 14.3.1 ControlMode Enumeration

The `ControlMode` enum defines six operating modes:

| Mode         | Description                                                                       |
| ------------ | --------------------------------------------------------------------------------- |
| `TORQUE`     | Direct torque commands sent to joint actuators. Requires model-based computation. |
| `POSITION`   | Joint position setpoints tracked by low-level PD controllers.                     |
| `VELOCITY`   | Joint velocity commands.                                                          |
| `IMPEDANCE`  | Virtual spring-damper behavior at the end-effector.                               |
| `ADMITTANCE` | External forces modulate a reference trajectory.                                  |
| `HYBRID`     | Selective force/position control along different task-space axes.                 |

### 14.3.2 Impedance Control

Impedance control makes the robot behave as a virtual mass-spring-damper system.
The commanded torque is:

$$
\boldsymbol{\tau} = K_p (\mathbf{x}_d - \mathbf{x})
  + B_d (\dot{\mathbf{x}}_d - \dot{\mathbf{x}})
  + J^T \mathbf{f}_{\text{ff}}
$$

where:

- $K_p$ is the stiffness matrix (proportional gain in task space).
- $B_d$ is the damping matrix (derivative gain in task space).
- $\mathbf{x}_d, \dot{\mathbf{x}}_d$ are the desired position and velocity.
- $\mathbf{x}, \dot{\mathbf{x}}$ are the actual position and velocity.
- $\mathbf{f}_{\text{ff}}$ is a feedforward force term.
- $J$ is the task-space Jacobian.

Impedance control is passive by design: the robot absorbs energy from
interactions rather than injecting it, making it inherently safe for
human-robot collaboration.

### 14.3.3 Admittance Control

Admittance control inverts the impedance relationship. Rather than commanding
torques, it modifies the position reference based on measured external forces:

$$
M_d \ddot{\mathbf{x}} + B_d \dot{\mathbf{x}} + K_d \mathbf{x} = \mathbf{f}_{\text{ext}}
$$

where $M_d$, $B_d$, $K_d$ are the desired inertia, damping, and stiffness
matrices, and $\mathbf{f}_{\text{ext}}$ is the measured external force. The
solution $\mathbf{x}(t)$ is used as the reference for a position controller.

Admittance control is preferred when the robot has stiff low-level position
controllers that cannot be bypassed.

### 14.3.4 Computed Torque Control

Computed torque (also called inverse dynamics control) uses the full robot
dynamics model to achieve precise trajectory tracking:

$$
\boldsymbol{\tau} = M(\mathbf{q})\ddot{\mathbf{q}}_d
  + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}
  + \mathbf{g}(\mathbf{q})
$$

where:

- $M(\mathbf{q})$ is the joint-space inertia matrix.
- $C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}$ is the Coriolis/centrifugal
  force vector.
- $\mathbf{g}(\mathbf{q})$ is the gravity torque vector.
- $\ddot{\mathbf{q}}_d$ is the desired joint acceleration, typically computed
  from PD feedback:
  $\ddot{\mathbf{q}}_d = \ddot{\mathbf{q}}_{\text{ref}} + K_p (\mathbf{q}_d - \mathbf{q}) + K_d (\dot{\mathbf{q}}_d - \dot{\mathbf{q}})$.

When applied perfectly, the closed-loop system reduces to
$\ddot{\mathbf{q}} = \ddot{\mathbf{q}}_d$ (linear decoupled dynamics).

### 14.3.5 Operational Space Control

Operational space control extends computed torque to task space, enabling
Cartesian control of end-effectors without requiring explicit inverse kinematics:

$$
\boldsymbol{\tau} = J^T\bigl[\Lambda(\ddot{\mathbf{x}}_d - \dot{J}\dot{\mathbf{q}})
  + \boldsymbol{\mu} + \mathbf{p}\bigr]
$$

where:

- $\Lambda = (J M^{-1} J^T)^{-1}$ is the operational-space inertia matrix.
- $\boldsymbol{\mu} = \Lambda J M^{-1} C \dot{\mathbf{q}} - \Lambda \dot{J}\dot{\mathbf{q}}$
  is the operational-space Coriolis/centrifugal force.
- $\mathbf{p} = \Lambda J M^{-1} \mathbf{g}$ is the operational-space gravity
  compensation.
- $\dot{J}\dot{\mathbf{q}}$ accounts for the time derivative of the Jacobian.

The operational-space inertia $\Lambda$ provides the mapping between task-space
accelerations and task-space forces. When the Jacobian is not full rank
(kinematic singularity), $\Lambda$ becomes ill-conditioned and regularization
or damped pseudoinverses are needed.

### 14.3.6 The WholeBodyController Class

`WholeBodyController` is the top-level orchestrator. It manages a task list,
builds the QP, and extracts the optimal joint accelerations and torques.

**Initialization:**

```python
wbc = WholeBodyController(
    engine,                 # RoboticsCapable physics engine
    config=WBCConfig(
        dt=0.001,
        regularization=1e-6,
        torque_limits=None,
        velocity_limits=None,
        acceleration_limits=None,
        contact_force_regularization=1e-4,
        use_hierarchical=True,
    ),
    solver=None,            # Uses create_default_solver() if None
)
```

**Task management:**

- `add_task(task)` -- adds a task, sorted by priority. Rejects duplicate names.
- `remove_task(name)` -- removes by name.
- `get_task(name)` -- retrieves by name.
- `clear_tasks()` -- removes all.
- `set_contact_jacobians(jacobians)` -- provides contact Jacobians for the
  equation-of-motion constraint.

**Solving:**

```python
solution = wbc.solve()  # returns WBCSolution
```

The solver builds the following QP:

**Decision variables:** $\mathbf{x} = [\ddot{\mathbf{q}}^T, \mathbf{f}_c^T]^T$
where $\ddot{\mathbf{q}} \in \mathbb{R}^{n_v}$ are joint accelerations and
$\mathbf{f}_c \in \mathbb{R}^{3 n_c}$ are contact forces.

**Objective (weighted mode):**

$$
\min_{\mathbf{x}} \sum_i \|J_i \ddot{\mathbf{q}} - \ddot{\mathbf{x}}_{d,i}\|_{W_i}^2
  + \epsilon_q \|\ddot{\mathbf{q}}\|^2
  + \epsilon_f \|\mathbf{f}_c\|^2
$$

**Equation of motion constraint:**

$$
M(\mathbf{q}) \ddot{\mathbf{q}} + C(\mathbf{q}, \dot{\mathbf{q}})\dot{\mathbf{q}}
  + \mathbf{g}(\mathbf{q})
  = \boldsymbol{\tau} + J_c^T \mathbf{f}_c
$$

Rearranged for the QP: $M \ddot{\mathbf{q}} - J_c^T \mathbf{f}_c = -(\text{bias forces})$.

**Friction cone constraints:** For each contact $i$, four pyramid constraints
plus one non-negative normal force constraint:

$$
f_{t,x} \leq \mu f_n, \quad -f_{t,x} \leq \mu f_n, \quad
f_{t,y} \leq \mu f_n, \quad -f_{t,y} \leq \mu f_n, \quad
f_n \geq 0
$$

**Variable bounds:** acceleration limits, velocity limits (translated to
acceleration via $\ddot{q}_{\max} = (v_{\max} - \dot{q})/\Delta t$), and
contact force bounds ($\pm 10000$ N for numerical stability).

The `WBCSolution` contains:

| Field                 | Description                                                                 |
| --------------------- | --------------------------------------------------------------------------- |
| `joint_accelerations` | Optimal $\ddot{\mathbf{q}}$                                                 |
| `joint_torques`       | $\boldsymbol{\tau} = M\ddot{\mathbf{q}} + \text{bias} - J_c^T \mathbf{f}_c$ |
| `contact_forces`      | Optimal $\mathbf{f}_c$                                                      |
| `task_errors`         | Per-task weighted squared error                                             |

---
