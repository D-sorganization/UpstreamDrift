# Chapter 8: Drake Physics Engine

## 8.1 Overview

The Drake backend provides an alternative physics engine based on MIT's Drake
multibody simulation toolkit. It excels at trajectory optimization, contact
modeling via time-stepping methods, and 3D visualization through MeshCat.

```
src/engines/physics_engines/drake/
    python/
        drake_physics_engine.py    -- PhysicsEngine protocol implementation
        motion_optimization.py     -- Multi-objective trajectory optimization
        swing_plane_integration.py -- Swing plane analysis + MeshCat visualization
```

---

## 8.2 `DrakePhysicsEngine`

**File:** `drake_physics_engine.py`

```python
class DrakePhysicsEngine(PhysicsEngine):
    def __init__(self, time_step: float = DEFAULT_TIME_STEP):
        self.builder = DiagramBuilder()
        self.plant: MultibodyPlant
        self.scene_graph: Any
        result = AddMultibodyPlantSceneGraph(self.builder, time_step)
        self.plant = result[0]
        self.scene_graph = result[1]
        self.diagram: framework.Diagram | None = None
        self.context: framework.Context | None = None
        self.plant_context: framework.Context | None = None
        self._is_finalized = False
        self.simulator: analysis.Simulator | None = None
```

### 8.2.1 Drake Architecture Concepts

**MultibodyPlant** is Drake's representation of a multibody system. It
encapsulates the kinematics and dynamics of rigid and deformable bodies
connected by joints. The plant is constructed incrementally (add models, add
constraints) and then **finalized** -- after which the model topology is
immutable.

**SceneGraph** manages the geometric representation of bodies for collision
detection and visualization.

**DiagramBuilder** wires together systems (plant, scene graph, controllers,
visualizers) into a composite `Diagram`.

**Context** is Drake's mechanism for storing state. Unlike MuJoCo's single
global `MjData`, Drake uses functional-style contexts that allow multiple
independent state snapshots.

### 8.2.2 Initialization Sequence

```python
def _ensure_finalized(self) -> None:
    if not self._is_finalized:
        self.plant.Finalize()
        self.diagram = self.builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyContextFromRoot(self.context)
        self.simulator = analysis.Simulator(self.diagram, self.context)
        self.simulator.Initialize()
        self._is_finalized = True
```

### 8.2.3 Model Loading

Drake's `Parser` supports URDF, SDF, and (experimentally) MJCF formats:

```python
def load_from_path(self, path: str) -> None:
    parser = Parser(self.plant)
    parser.AddModels(path)
    self._ensure_finalized()
```

### 8.2.4 State Access

Drake uses `plant.GetPositions()` / `plant.GetVelocities()` instead of
direct array access:

```python
def get_state(self) -> tuple[np.ndarray, np.ndarray]:
    q = self.plant.GetPositions(self.plant_context)
    v = self.plant.GetVelocities(self.plant_context)
    return q, v

def set_state(self, q, v) -> None:
    self.plant.SetPositions(self.plant_context, q)
    self.plant.SetVelocities(self.plant_context, v)
```

### 8.2.5 Simulation Stepping

Drake's `Simulator.AdvanceTo()` integrates the system forward in time:

```python
def step(self, dt: float | None = None) -> None:
    current_time = self.context.get_time()
    step_size = dt if dt is not None else self.plant.time_step()
    self.simulator.AdvanceTo(current_time + step_size)
```

### 8.2.6 Dynamics Interface

**Mass Matrix:**

```python
def compute_mass_matrix(self) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    return M
```

Drake computes $M(q)$ via the inverse dynamics identity: column $j$ of $M$ is
obtained by evaluating $\text{ID}(q, 0, e_j)$ where $e_j$ is the $j$-th
unit vector.

**Bias Forces:**

```python
def compute_bias_forces(self) -> np.ndarray:
    nv = self.plant.num_velocities()
    vdot_zero = np.zeros(nv)
    forces = self.plant.CalcInverseDynamics(
        self.plant_context, vdot_zero,
        self.plant.MakeMultibodyForces(self.plant))
    return forces
```

This evaluates $\text{ID}(q, v, 0) = C(q, v)v + g(q)$.

**Gravity Forces:**

```python
def compute_gravity_forces(self) -> np.ndarray:
    return self.plant.CalcGravityGeneralizedForces(self.plant_context)
```

**Jacobian:**

Drake's `CalcJacobianSpatialVelocity` returns a $6 \times n_v$ matrix with
angular components in rows 0--2 and linear components in rows 3--5:

```python
def compute_jacobian(self, body_name: str) -> dict[str, np.ndarray] | None:
    body = self.plant.GetBodyByName(body_name)
    J = self.plant.CalcJacobianSpatialVelocity(
        self.plant_context,
        JacobianWrtVariable.kV,
        body.body_frame(),
        np.zeros(3),
        self.plant.world_frame(),
        self.plant.world_frame())
    return {
        "linear":  J[3:, :],
        "angular": J[:3, :],
        "spatial": J,
    }
```

### 8.2.7 Drift-Control Decomposition

The Drake implementation solves $M\ddot{q} = -b$ for drift:

```python
def compute_drift_acceleration(self) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    bias = self.compute_bias_forces()
    a_drift = -np.linalg.solve(M, bias)
    return a_drift
```

And $M\ddot{q} = \tau$ for control:

```python
def compute_control_acceleration(self, tau) -> np.ndarray:
    M = self.plant.CalcMassMatrixViaInverseDynamics(self.plant_context)
    return np.linalg.solve(M, tau)
```

### 8.2.8 Counterfactual Experiments

Both ZTCF and ZVCF follow the same save-mutate-restore pattern as MuJoCo,
using Drake's `SetPositions`/`SetVelocities`/`GetPositions`/`GetVelocities`:

**ZTCF:** Set $\tau = 0$ at state $(q, v)$:

$$\ddot{q}_{\text{ZTCF}} = -M(q)^{-1} b(q, v)$$

**ZVCF:** Set $v = 0$ at configuration $q$:

$$\ddot{q}_{\text{ZVCF}} = M(q)^{-1}(\tau - g(q))$$

### 8.2.9 Contact Forces

The current Drake wrapper returns a placeholder zero vector for ground
reaction forces. Full contact force retrieval requires querying `ContactResults`
from a `Simulator`-managed context:

```python
def compute_contact_forces(self) -> np.ndarray:
    # Placeholder: full GRF requires ContactResults from Simulator
    return np.zeros(3)
```

---

## 8.3 Motion Optimization

**File:** `motion_optimization.py`

### 8.3.1 `DrakeMotionOptimizer`

```python
class DrakeMotionOptimizer:
    def __init__(self):
        self.objectives: list[OptimizationObjective] = []
        self.constraints: list[OptimizationConstraint] = []
```

### 8.3.2 Objective and Constraint Data Structures

```python
@dataclass
class OptimizationObjective:
    name: str
    weight: float
    target_value: float | None = None
    cost_function: Callable[[np.ndarray], float] | None = None

@dataclass
class OptimizationConstraint:
    name: str
    constraint_type: str   # 'equality', 'inequality', 'bounds'
    lower_bound: float | None = None
    upper_bound: float | None = None
    constraint_function: Callable[[np.ndarray], float] | None = None
```

### 8.3.3 SLSQP Trajectory Optimization

The optimizer formulates a nonlinear program:

$$\min_{x} \sum_i w_i \, f_i(x)$$

subject to:

$$g_j(x) \leq 0 \quad \text{(inequality constraints)}$$
$$h_k(x) = 0 \quad \text{(equality constraints)}$$

where $x$ is the flattened trajectory. The solver is SciPy's SLSQP
(Sequential Least Squares Programming):

```python
def optimize_trajectory(
    self,
    initial_trajectory: np.ndarray,    # (N, dim)
    max_iterations: int = 100,
    tolerance: float = 1e-6,
) -> OptimizationResult:
    opt_result = scipy_minimize(
        total_cost, x0,
        method="SLSQP",
        constraints=scipy_constraints,
        options={"maxiter": max_iterations, "ftol": tolerance})
```

### 8.3.4 Standard Golf Objectives

The `setup_standard_golf_objectives()` method registers three default objectives:

1. **Ball speed** ($w = 1.0$): Maximize club head velocity at impact.
2. **Accuracy** ($w = 0.8$): Minimize lateral deviation from target line.
3. **Smoothness** ($w = 0.5$): Minimize second-derivative norm (jerk).

### 8.3.5 Standard Golf Constraints

The `setup_standard_golf_constraints()` method registers:

1. **Joint limits** (inequality): Joint angles must remain within anatomical range.
2. **Impact timing** (equality): Ball contact must occur at a specified time.

### 8.3.6 Result Structure

```python
@dataclass
class OptimizationResult:
    success: bool
    optimal_trajectory: np.ndarray
    optimal_cost: float
    iterations: int
    convergence_message: str
    objective_values: dict[str, float]
    constraint_violations: dict[str, float]
```

### 8.3.7 Convenience Methods

```python
def optimize_for_distance(self, initial_trajectory, target_distance=250.0) -> OptimizationResult
def optimize_for_accuracy(self, initial_trajectory, target_point) -> OptimizationResult
def export_optimization_results(self, result, output_path) -> None
```

---

## 8.4 Swing Plane Integration

**File:** `swing_plane_integration.py`

### 8.4.1 `DrakeSwingPlaneAnalyzer`

```python
class DrakeSwingPlaneAnalyzer:
    def __init__(self):
        self.analyzer = SwingPlaneAnalyzer()  # From shared module
```

### 8.4.2 Trajectory Analysis

```python
def analyze_trajectory(
    self, positions: np.ndarray, timestamps: np.ndarray | None = None
) -> SwingPlaneMetrics:
    """Fit a plane to club head positions (N, 3)."""
```

### 8.4.3 Drake Context Extraction

The analyzer can extract club head positions directly from a Drake simulation
context or log:

```python
def analyze_from_drake_context(
    self, context, plant, club_body_index, num_samples=100
) -> SwingPlaneMetrics:
    """Extract club positions from Drake plant context and analyze."""
```

### 8.4.4 Swing Plane Deviation Cost

The swing plane deviation cost penalises the sum of squared distances from each
trajectory point to the fitted swing plane:

$$C_{\text{plane}} = w \sum_{i=1}^{N} (\mathbf{n} \cdot (\mathbf{p}_i - \mathbf{p}_0))^2$$

where $\mathbf{n}$ is the plane normal, $\mathbf{p}_0$ is a point on the plane,
and $w$ is the constraint weight. This cost is registered with the
`DrakeMotionOptimizer` via:

```python
def integrate_with_optimization(
    self, trajectory_optimizer, swing_plane_constraint_weight=1.0
) -> None:
    def swing_plane_cost(trajectory):
        positions = trajectory[:, :3]
        metrics = self.analyzer.analyze(positions)
        return metrics.rmse ** 2

    trajectory_optimizer.add_objective(
        name="swing_plane_deviation",
        weight=swing_plane_constraint_weight,
        cost_function=swing_plane_cost,
        target_value=0.0)
```

### 8.4.5 MeshCat Visualization

The `visualize_with_meshcat()` method renders three visual elements into Drake's
MeshCat 3D viewer:

1. **Semi-transparent plane** -- the fitted swing plane as a triangulated quad.
2. **Trajectory line** -- club head path as a colored line strip.
3. **Deviation lines** -- (planned) lines from each point to the plane showing
   deviation magnitude.

```python
def visualize_with_meshcat(
    self,
    meshcat_visualizer,       # pydrake.geometry.Meshcat instance
    metrics: SwingPlaneMetrics,
    trajectory_positions: np.ndarray,
) -> None:
    meshcat.SetTriangleMesh(f"{prefix}/plane", mesh, Rgba(0.2, 0.5, 0.8, 0.3))
    meshcat.SetLine(f"{prefix}/trajectory", positions.T, line_width=3.0,
                    rgba=Rgba(1.0, 0.3, 0.1, 1.0))
```

### 8.4.6 Export

```python
def export_for_analysis(
    self,
    metrics: SwingPlaneMetrics,
    trajectory_positions: np.ndarray,
    output_path: str,
) -> None:
    """Export to JSON with normal vector, steepness, RMSE, trajectory."""
```

---

## 8.5 Direct Collocation (Drake Integration Notes)

While the current codebase uses SLSQP via SciPy for trajectory optimization,
Drake natively supports **direct collocation** methods for trajectory
optimization through `pydrake.trajectories` and `pydrake.solvers`.

### 8.5.1 Hermite-Simpson Collocation

In Hermite-Simpson collocation, the trajectory between knot points is
approximated by a cubic Hermite polynomial. The collocation constraint at the
midpoint ensures dynamic feasibility:

$$x_{k+\frac{1}{2}} = \frac{1}{2}(x_k + x_{k+1}) + \frac{h_k}{8}(\dot{x}_k - \dot{x}_{k+1})$$

$$\dot{x}_{k+\frac{1}{2}} = f(x_{k+\frac{1}{2}}, u_{k+\frac{1}{2}})$$

where $h_k$ is the time step between knot points $k$ and $k+1$.

### 8.5.2 Trapezoidal Collocation

The simpler trapezoidal method enforces:

$$x_{k+1} = x_k + \frac{h_k}{2}\bigl(f(x_k, u_k) + f(x_{k+1}, u_{k+1})\bigr)$$

### 8.5.3 Nonlinear Solvers

Drake supports multiple nonlinear programming solvers:

- **SNOPT** -- Sequential quadratic programming solver for large-scale NLPs.
  Preferred for trajectory optimization due to sparse Jacobian exploitation.
- **IPOPT** -- Interior point optimizer. Good for problems with many inequality
  constraints.
- **SLSQP** -- Used in the current implementation via SciPy. Suitable for
  moderate-scale problems.

The solver selection is automatic in Drake based on problem structure.

---

## 8.6 LQR Control in Drake Context

The simulation framework (`sim.py`) implements an LQR-structured controller
that can be used with either MuJoCo or Drake backends:

$$u = -K(x - x^*)$$

where $K \in \mathbb{R}^{n_u \times n_x}$ is the gain matrix and $x^*$ is
the target state. The current implementation uses a decoupled PD gain structure:

$$K_{i, q_j} = K_p, \quad K_{i, \dot{q}_j} = K_d$$

with $K_p = 100$, $K_d = 10$. The full linearisation-based LQR (via Drake's
`LinearQuadraticRegulator`) is available for models without quaternion
singularities.

---

## 8.7 Contact Modeling

Drake provides two contact models:

### 8.7.1 Point Contact (Compliant)

Forces are computed using a spring-damper model at each contact point:

$$f_n = k_n \phi + d_n \dot{\phi}$$

where $\phi$ is the penetration depth, $k_n$ is the contact stiffness, and
$d_n$ is the damping coefficient. This is analogous to MuJoCo's default
solver.

### 8.7.2 Hydroelastic Contact

Drake's hydroelastic contact model computes contact pressures over a surface
patch rather than at a single point, providing smoother and more physically
accurate contact forces. This is particularly relevant for modelling the
club-ball impact where the contact area evolves during the collision.

The pressure field is computed as:

$$p(\mathbf{x}) = E^* \, \phi(\mathbf{x})$$

where $E^*$ is the effective elastic modulus and $\phi(\mathbf{x})$ is the
local interpenetration function.

---

## 8.8 Engine Comparison: MuJoCo vs. Drake

| Feature                 | MuJoCo                               | Drake                                      |
| ----------------------- | ------------------------------------ | ------------------------------------------ |
| Primary formulation     | Maximal coordinates with constraints | Minimal coordinates (MultibodyPlant)       |
| Contact solver          | Convex optimisation (PGS, Newton)    | Time-stepping, hydroelastic                |
| Muscle support          | Native (via actuators or MyoSuite)   | Via external force elements                |
| Trajectory optimisation | Via `motion_optimization.py` + SciPy | Direct collocation (SNOPT/IPOPT/SLSQP)     |
| Visualisation           | `mujoco.viewer`, OpenGL              | MeshCat (WebGL), Drake Visualizer          |
| State management        | Single `MjData` (mutable)            | Functional `Context` (immutable snapshots) |
| MJCF model loading      | Native                               | Experimental                               |
| URDF/SDF loading        | Via conversion                       | Native                                     |
| Flexible shaft          | Modal (Euler-Bernoulli modes)        | Compliant multibody elements               |
| Counterfactuals         | Save-mutate-restore                  | Context-based snapshots                    |
| Induced acceleration    | Built-in analyser                    | Via protocol methods                       |

Both engines satisfy the `PhysicsEngine` protocol, ensuring that all GUI,
analytics, and optimization code works identically regardless of the active
backend.

---
