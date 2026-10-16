# UpstreamDrift User Manual -- Part IV: Robotics Framework

---

# Chapter 13: Contact Dynamics

The contact dynamics module (`src/robotics/contact/`) provides the foundation for
multi-body contact modeling, friction analysis, and grasp quality assessment. All
components adhere to the framework's Design by Contract methodology: immutable
data structures enforce invariants at construction time, preconditions guard every
public entry point, and postconditions verify results before they leave the module
boundary. This chapter covers the four major subsystems -- core contact types,
friction cone analysis, contact management, and grasp analysis -- in full
mathematical and programmatic detail.

---

## 13.1 Core Types and Protocols

### 13.1.1 ContactType Enumeration

The `ContactType` enum (defined in `src/robotics/core/types.py`) classifies the
geometry of contact between two bodies. The available members are:

| Member  | Description                                                                                                 |
| ------- | ----------------------------------------------------------------------------------------------------------- |
| `POINT` | Single-point contact. The simplest model, adequate for finger tips and convex surface interactions.         |
| `LINE`  | Edge or line contact. Occurs when a straight edge contacts a surface.                                       |
| `PATCH` | Surface or patch contact. Models extended contact areas such as the sole of a foot resting flat on a floor. |
| `SOFT`  | Soft or deformable contact. Used when one or both bodies exhibit compliance at the contact interface.       |

Every `ContactType` value is produced by `auto()`, making the underlying integer
representation an implementation detail. Code should compare against the symbolic
members rather than raw integers:

```python
from src.robotics.core.types import ContactType

if contact.contact_type == ContactType.PATCH:
    # Use extended contact model
    ...
```

### 13.1.2 ContactState Frozen Dataclass

`ContactState` is a frozen (immutable) dataclass that captures the complete
instantaneous state of a single contact. Immutability ensures thread safety and
prevents accidental mutation during multi-step computations. The class definition
lives in `src/robotics/core/types.py`.

**Field Reference**

| Field                  | Type                    | Default             | Description                                                     |
| ---------------------- | ----------------------- | ------------------- | --------------------------------------------------------------- |
| `contact_id`           | `int`                   | (required)          | Unique identifier assigned by the ContactManager.               |
| `body_a`               | `str`                   | (required)          | Name of the first body in the contact pair.                     |
| `body_b`               | `str`                   | (required)          | Name of the second body in the contact pair.                    |
| `position`             | `NDArray[float64]` (3,) | (required)          | Contact point in the world frame, measured in meters.           |
| `normal`               | `NDArray[float64]` (3,) | (required)          | Unit contact normal pointing from body B toward body A.         |
| `penetration`          | `float`                 | `0.0`               | Penetration depth in meters. Must be non-negative.              |
| `normal_force`         | `float`                 | `0.0`               | Scalar normal force magnitude in newtons. Must be non-negative. |
| `friction_force`       | `NDArray[float64]` (3,) | `np.zeros(3)`       | Friction (tangential) force vector in newtons.                  |
| `friction_coefficient` | `float`                 | `0.5`               | Coulomb friction coefficient $\mu$. Must be non-negative.       |
| `contact_type`         | `ContactType`           | `ContactType.POINT` | Geometric classification of the contact.                        |
| `is_active`            | `bool`                  | `True`              | Whether the contact is currently active.                        |

**Construction-Time Invariants**

The `__post_init__` method enforces the following invariants immediately after
construction:

1. `position.shape == (3,)` -- the contact point must be a 3-vector.
2. `normal.shape == (3,)` -- the contact normal must be a 3-vector.
3. `friction_force.shape == (3,)` -- the friction force must be a 3-vector.
4. `penetration >= 0` -- negative penetration is physically meaningless.
5. `normal_force >= 0` -- tensile normal forces violate the unilateral contact
   assumption.
6. `friction_coefficient >= 0` -- negative friction is non-physical.
7. The normal vector is automatically normalized: if $\|\mathbf{n}\| > 10^{-10}$
   then the stored value becomes $\hat{\mathbf{n}} = \mathbf{n} / \|\mathbf{n}\|$.

All NumPy array fields are coerced to `float64` via `np.asarray` during
construction, regardless of the dtype passed in by the caller. Because the
dataclass is frozen, these assignments use `object.__setattr__` to bypass the
write protection that `frozen=True` normally imposes.

**Example -- Creating a ContactState**

```python
import numpy as np
from src.robotics.core.types import ContactState, ContactType

contact = ContactState(
    contact_id=0,
    body_a="robot_foot",
    body_b="ground",
    position=np.array([0.1, -0.05, 0.0]),
    normal=np.array([0.0, 0.0, 1.0]),
    penetration=0.001,
    normal_force=490.5,
    friction_force=np.array([12.0, -3.0, 0.0]),
    friction_coefficient=0.6,
    contact_type=ContactType.PATCH,
    is_active=True,
)
```

### 13.1.3 The `get_wrench()` Method

`get_wrench()` assembles the full 6D spatial wrench at the contact point:

$$
\mathbf{w} = \begin{bmatrix} \mathbf{f} \\ \boldsymbol{\tau} \end{bmatrix}
= \begin{bmatrix} f_n \hat{\mathbf{n}} + \mathbf{f}_t \\ \mathbf{0} \end{bmatrix}
$$

The total contact force $\mathbf{f}$ is the sum of the normal component
$f_n \hat{\mathbf{n}}$ and the tangential friction force $\mathbf{f}_t$. Under
the point-contact assumption, the torque at the contact point is zero because
all forces act through a single point. The method returns a length-6 NumPy array
with layout `[fx, fy, fz, tx, ty, tz]`.

```python
wrench = contact.get_wrench()  # shape (6,)
force = wrench[:3]   # net contact force in world frame
torque = wrench[3:]  # zero for point contacts
```

To compute the torque contribution at a different reference point $\mathbf{p}$,
apply the standard cross-product transport:

$$
\boldsymbol{\tau}_p = (\mathbf{r}_c - \mathbf{p}) \times \mathbf{f}
$$

where $\mathbf{r}_c$ is `contact.position`.

### 13.1.4 The `is_sliding()` Method

`is_sliding(tolerance=1e-6)` checks whether the contact is at the Coulomb
friction limit and hence in a sliding (kinetic) state:

$$
\text{sliding} \iff \|\mathbf{f}_t\| \geq \mu \cdot f_n - \epsilon
$$

where $\epsilon$ is the `tolerance` parameter. The method returns `True` when
the friction force magnitude reaches the maximum value allowed by the friction
cone, indicating that the contact surfaces are sliding relative to each other
(or are on the verge of doing so).

```python
if contact.is_sliding():
    print("Contact is at friction limit -- kinetic friction active")
```

### 13.1.5 The `with_force()` Method

Because `ContactState` is frozen, force updates are performed by creating a new
instance with modified values. The `with_force(normal_force, friction_force=None)`
convenience method copies every field except the two force fields:

```python
updated = contact.with_force(
    normal_force=600.0,
    friction_force=np.array([15.0, -4.0, 0.0]),
)
# updated is a new ContactState; the original is unchanged
```

If `friction_force` is `None`, the original friction force is preserved.

### 13.1.6 Supporting Protocols

The `src/robotics/core/protocols.py` module defines runtime-checkable protocols
that physics engines must implement to participate in the contact subsystem.

**RoboticsCapable** -- the minimal protocol. Provides `get_state()`,
`set_state()`, `compute_mass_matrix()`, `compute_bias_forces()`,
`compute_gravity_forces()`, `compute_jacobian()`, and `get_time()`.

**ContactCapable** -- extends the engine with contact-specific queries:

| Method                      | Return Type                    | Description                                                                            |
| --------------------------- | ------------------------------ | -------------------------------------------------------------------------------------- |
| `get_contact_count()`       | `int`                          | Number of active contacts detected by the engine.                                      |
| `get_contact_info(idx)`     | `dict[str, Any]`               | Dictionary with keys `body_a`, `body_b`, `position`, `normal`, `penetration`, `force`. |
| `get_contact_jacobian(idx)` | `NDArray` (3, n_v) or (6, n_v) | Linear or spatial Jacobian at the contact point.                                       |

Both protocols use `@runtime_checkable` so they can be validated at runtime via
`isinstance` checks. The `ContactManager` (Section 13.3) verifies that the
engine satisfies these protocols during initialization.

---

## 13.2 Friction Cone Analysis

The friction cone module (`src/robotics/contact/friction_cone.py`) provides tools
for representing, testing, linearizing, and projecting forces with respect to
Coulomb friction constraints. All functions employ Design by Contract decorators
from `src/shared/python/contracts`.

### 13.2.1 The FrictionCone Frozen Dataclass

```python
@dataclass(frozen=True)
class FrictionCone:
    mu: float                        # Coulomb friction coefficient
    normal: NDArray[np.float64]      # Contact normal (unit vector)
    num_sides: int = 8               # Sides for linearized approximation
```

**Construction-time validation:**

- `mu >= 0` (non-negative friction).
- `num_sides >= 3` (a pyramid needs at least three faces).
- `normal` is coerced to `float64` and normalized to unit length. A zero-length
  normal raises `ValueError`.

The Coulomb friction model constrains the tangential component of the contact
force to lie within a cone:

$$
\|\mathbf{f}_t\| \leq \mu \cdot f_n
$$

where $\mathbf{f}_t$ is the component of force perpendicular to the contact
normal and $f_n = \mathbf{f} \cdot \hat{\mathbf{n}}$ is the component along
the contact normal. This defines a second-order cone (ice-cream cone) in 3D
force space with half-angle $\alpha = \arctan(\mu)$.

### 13.2.2 The `contains()` Method

`contains(force, tolerance=1e-6)` checks whether a given force vector lies
inside the friction cone:

1. Decompose the force into normal and tangential components:

$$
f_n = \mathbf{f} \cdot \hat{\mathbf{n}}, \qquad
\mathbf{f}_t = \mathbf{f} - f_n \hat{\mathbf{n}}
$$

2. Check the unilateral constraint: if $f_n < -\epsilon$, the force is tensile
   (pulling the surfaces apart), which violates the contact model. Return
   `False`.

3. Check the friction constraint: return
   $\|\mathbf{f}_t\| \leq \mu \cdot f_n + \epsilon$.

```python
cone = FrictionCone(mu=0.5, normal=np.array([0, 0, 1]))
force = np.array([3.0, 0.0, 10.0])
assert cone.contains(force)  # 3.0 <= 0.5 * 10.0 = 5.0, within cone
```

### 13.2.3 The `get_generators()` Method

`get_generators()` returns a polyhedral approximation of the friction cone as a
set of $N$ edge directions (generators), where $N$ = `num_sides`. The
generators are arranged in a circular pattern around the contact normal.

**Algorithm.** Given the contact normal $\hat{\mathbf{n}}$, the code first
constructs an orthonormal tangent basis $\{\hat{\mathbf{t}}_1, \hat{\mathbf{t}}_2\}$
using Gram-Schmidt orthogonalization. The initial vector is chosen to avoid
near-parallel alignment with the normal:

$$
\mathbf{v}_0 =
\begin{cases}
[1, 0, 0]^T & \text{if } |n_x| < 0.9 \\
[0, 1, 0]^T & \text{otherwise}
\end{cases}
$$

Then:

$$
\hat{\mathbf{t}}_1 = \frac{\mathbf{v}_0 - (\mathbf{v}_0 \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}}
  {\|\mathbf{v}_0 - (\mathbf{v}_0 \cdot \hat{\mathbf{n}})\hat{\mathbf{n}}\|},
\qquad
\hat{\mathbf{t}}_2 = \hat{\mathbf{n}} \times \hat{\mathbf{t}}_1
$$

For $i = 0, 1, \ldots, N-1$ the angular position around the cone is:

$$
\theta_i = \frac{2\pi i}{N}
$$

and the $i$-th generator direction is:

$$
\mathbf{g}_i = \hat{\mathbf{n}} + \mu \bigl(\cos\theta_i \cdot \hat{\mathbf{t}}_1
  + \sin\theta_i \cdot \hat{\mathbf{t}}_2\bigr)
$$

Each generator lies on the surface of the friction cone. The returned matrix has
shape `(3, num_sides)`, with each column being one generator.

**Preconditions** (enforced by `@precondition` decorators on the internal
`_compute_cone_generators` function):

- $\mu \geq 0$
- `num_sides >= 3`

**Postcondition:** the result has at least 3 columns.

**Example:**

```python
cone = FrictionCone(mu=0.7, normal=np.array([0, 0, 1]), num_sides=8)
G = cone.get_generators()  # shape (3, 8)
# Each column G[:, i] is a generator on the cone surface
```

### 13.2.4 Linearizing the Friction Cone

The function `linearize_friction_cone(mu, normal, num_faces=8)` converts the
nonlinear second-order cone constraint into a polyhedral (linear) approximation
of the form:

$$
A \mathbf{f} \leq \mathbf{b}
$$

This approximation is essential for QP-based controllers that require linear
constraints.

**Derivation.** The exact Coulomb constraint is:

$$
\|\mathbf{f}_t\| \leq \mu f_n
$$

This is approximated by $N$ half-space constraints, one for each face of the
inscribed polyhedron. For face $i$, define the tangent direction:

$$
\mathbf{d}_i = \cos\theta_i \cdot \hat{\mathbf{t}}_1 + \sin\theta_i \cdot \hat{\mathbf{t}}_2,
\qquad \theta_i = \frac{2\pi i}{N}
$$

The $i$-th linear constraint is:

$$
\mathbf{d}_i^T \mathbf{f} \leq \mu \, \hat{\mathbf{n}}^T \mathbf{f}
$$

which rearranges to:

$$
(\mathbf{d}_i - \mu \hat{\mathbf{n}})^T \mathbf{f} \leq 0
$$

The function returns `(A, b)` where $A$ has shape `(num_faces, 3)` and $b$ is a
zero vector of length `num_faces`. Each row of $A$ is $\mathbf{d}_i - \mu \hat{\mathbf{n}}$.

**Accuracy vs. performance trade-off.** The polyhedral approximation is
conservative (inscribed): every force satisfying the linear constraints also
satisfies the true cone constraint. The approximation tightens as `num_faces`
increases. Common choices:

| `num_faces` | Approximation quality  | Typical use case       |
| ----------- | ---------------------- | ---------------------- |
| 4           | Coarse pyramid         | Real-time MPC          |
| 8           | Good balance (default) | Whole-body control     |
| 16          | High fidelity          | Grasp quality analysis |

The `FrictionConeType` enum in `src/robotics/core/types.py` provides named
presets: `EXACT`, `LINEARIZED_4`, `LINEARIZED_8`, `LINEARIZED_16`.

### 13.2.5 The `compute_friction_cone_constraint()` Function

`compute_friction_cone_constraint(contact_normal, contact_position,
friction_coeff, num_faces=8)` bundles the linearization with the non-negative
normal force constraint and the cone generators into a single dictionary for
direct consumption by optimization-based controllers:

```python
result = compute_friction_cone_constraint(
    contact_normal=np.array([0, 0, 1]),
    contact_position=np.array([0.1, 0.0, 0.0]),
    friction_coeff=0.5,
    num_faces=8,
)
# result['A']          shape (num_faces + 1, 3)
# result['b']          shape (num_faces + 1,)
# result['normal']     shape (3,)
# result['generators'] shape (3, num_faces)
```

The extra row in `A` encodes the non-negative normal force constraint
$-\hat{\mathbf{n}}^T \mathbf{f} \leq 0$, i.e., $f_n \geq 0$.

### 13.2.6 Projecting Onto the Friction Cone

`project_to_friction_cone(force, cone)` finds the closest point inside the
friction cone to a given force vector. This is useful for enforcing friction
constraints in force control or for "clamping" forces computed by unconstrained
solvers.

**Algorithm.** The projection considers three cases:

**Case 1: Force is inside the cone.** If `cone.contains(force)` returns `True`,
return the force unchanged.

**Case 2: Normal force is negative** ($f_n < 0$). The force is pulling the
surfaces apart. Two sub-cases arise:

- If the tangential magnitude is small ($\|\mathbf{f}_t\| \leq -\mu f_n$), the
  force lies in the dual cone; the closest point on the friction cone is the
  origin (zero force).

- Otherwise, project onto the cone surface. The cone edge direction in the
  tangent plane of the force is:

$$
\hat{\mathbf{e}} = \cos\alpha \cdot \hat{\mathbf{n}} + \sin\alpha \cdot \hat{\mathbf{t}}_f
$$

where $\alpha = \arctan(\mu)$ and $\hat{\mathbf{t}}_f = \mathbf{f}_t / \|\mathbf{f}_t\|$.
The projection length along this edge is:

$$
\ell = f_n \cos\alpha + \|\mathbf{f}_t\| \sin\alpha
$$

If $\ell \leq 0$, project to the origin. Otherwise, the projected force is
$\ell \cdot \hat{\mathbf{e}}$.

**Case 3: Tangential force exceeds friction limit** ($\|\mathbf{f}_t\| > \mu f_n$,
with $f_n \geq 0$). Scale the tangential component down to the friction limit
while preserving direction:

$$
\mathbf{f}_t \leftarrow \mathbf{f}_t \cdot \frac{\mu f_n}{\|\mathbf{f}_t\|}
$$

The projected force is $f_n \hat{\mathbf{n}} + \mathbf{f}_t^{\text{proj}}$.

```python
from src.robotics.contact.friction_cone import project_to_friction_cone, FrictionCone

cone = FrictionCone(mu=0.5, normal=np.array([0.0, 0.0, 1.0]))
raw_force = np.array([10.0, 0.0, 5.0])  # tangential exceeds limit
projected = project_to_friction_cone(raw_force, cone)
# projected tangential magnitude will be 0.5 * 5.0 = 2.5
```

---

## 13.3 Contact Management

The `ContactManager` class (`src/robotics/contact/contact_manager.py`) acts as
the central coordinator for contact detection, caching, Jacobian computation,
and support polygon analysis. It inherits from `ContractChecker` to gain
automatic invariant verification.

### 13.3.1 Construction and Invariants

```python
manager = ContactManager(engine, default_friction=0.5)
```

The constructor verifies that `engine` implements `RoboticsCapable` (raising
`TypeError` otherwise) and probes whether it also implements `ContactCapable`.
Two invariants are enforced throughout the object's lifetime:

1. `default_friction >= 0` -- the fallback friction coefficient is non-negative.
2. All contact IDs in the cache are unique -- no two `ContactState` objects
   share the same `contact_id`.

The class maintains an auto-incrementing counter (`_next_contact_id`) that
assigns monotonically increasing IDs to newly created contacts. This guarantees
uniqueness even across successive calls to `detect_contacts`.

### 13.3.2 Detecting Contacts

```python
contacts = manager.detect_contacts(q=None)
```

If a configuration `q` is provided, the engine state is temporarily set to that
configuration (preserving the current velocity). The method then delegates to the
engine's contact detection:

1. Query `engine.get_contact_count()` for the number of active contacts.
2. For each index $i \in [0, n)$, call `engine.get_contact_info(i)` to retrieve
   the raw contact data dictionary.
3. Decompose the reported force into normal and tangential components:

$$
f_n = \max\bigl(0,\; \mathbf{f} \cdot \hat{\mathbf{n}}\bigr),
\qquad
\mathbf{f}_t = \mathbf{f} - f_n \hat{\mathbf{n}}
$$

4. Construct a `ContactState` with a fresh unique ID.

If the engine does not implement `ContactCapable`, an empty list is returned.

**Postcondition** (enforced by `@postcondition`): every element of the returned
list is an instance of `ContactState`.

If `get_contact_info` raises for any index, the error is wrapped in a
`ContactError` with the offending `contact_id` attached.

### 13.3.3 Contact Jacobians

**Single contact:**

```python
J = manager.get_contact_jacobian(contact)  # (3, n_v) or (6, n_v) or None
```

The method looks up the contact's position in the internal cache by matching
`contact_id`, then delegates to `engine.get_contact_jacobian(idx)`. The Jacobian
maps joint velocities to contact-point velocities:

$$
\dot{\mathbf{p}}_c = J_c \dot{\mathbf{q}}
$$

A `(3, n_v)` Jacobian captures linear velocity only (point contacts). A
`(6, n_v)` Jacobian includes angular velocity as well (patch or wrench contacts).

**Stacked Jacobians:**

```python
J_stack = manager.get_contact_jacobian_stack(contacts)  # (3*n_c, n_v) or None
```

For $n_c$ contacts, the method vertically stacks the linear parts of all contact
Jacobians. If a `(6, n_v)` Jacobian is returned by the engine, only the first 3
rows (linear portion) are retained. The result has shape $(3 n_c, n_v)$:

$$
J_{\text{stack}} =
\begin{bmatrix}
J_{c,1} \\ J_{c,2} \\ \vdots \\ J_{c,n_c}
\end{bmatrix}
$$

**Postcondition:** the result is either `None` or a 2D matrix.

### 13.3.4 Computing the Support Polygon

```python
polygon = manager.compute_support_polygon(contacts)  # (m, 2) or None
```

The support polygon is the convex hull of the ground-plane projections of all
active contact points. It is the region within which the zero-moment point (ZMP)
must lie for static stability.

**Algorithm.** The method projects each contact's 3D position onto the $xy$-plane
by taking `position[:2]`. It then computes the 2D convex hull using one of two
strategies:

1. **SciPy path** (preferred): `scipy.spatial.ConvexHull` is used when available.
   Vertices are returned in counter-clockwise order.

2. **Graham scan fallback**: when SciPy is not installed, a manual implementation
   is used. The Graham scan algorithm proceeds as follows:

   **Step 1 -- Pivot selection.** Find the point with the smallest $y$-coordinate
   (breaking ties by smallest $x$). This point is guaranteed to be on the hull.

   **Step 2 -- Polar angle sorting.** Sort all remaining points by their polar
   angle relative to the pivot. Ties in angle are broken by distance from the
   pivot (closer points first).

   **Step 3 -- Stack-based construction.** Initialize a stack with the pivot.
   For each sorted point $p$:

   - While the stack has at least 2 elements and the cross product of
     $\overrightarrow{s_{-2} s_{-1}}$ and $\overrightarrow{s_{-1} p}$ is
     negative (clockwise turn), pop the stack.
   - Push $p$.

   The cross product test is:

   $$
   \text{cross}(\mathbf{o}, \mathbf{a}, \mathbf{b})
   = (a_x - o_x)(b_y - o_y) - (a_y - o_y)(b_x - o_x)
   $$

   Positive values indicate a counter-clockwise turn; negative values indicate
   clockwise. Points causing clockwise turns are discarded.

The function returns `None` if fewer than 3 contacts are provided, since a
polygon requires at least 3 vertices.

### 13.3.5 Point-in-Support-Polygon Test

```python
inside = manager.point_in_support_polygon(point, contacts)
```

This method first computes the support polygon (Section 13.3.4), then checks
whether the given point lies inside it using the **ray casting algorithm**.

**Ray casting algorithm.** Cast a horizontal ray from the test point toward
positive $x$-infinity. Count the number of polygon edges that the ray crosses.
If the count is odd, the point is inside; if even, outside.

For each edge from vertex $(x_i, y_i)$ to $(x_j, y_j)$:

1. Check if the ray's $y$-coordinate is between $y_i$ and $y_j$ (exclusive on
   one side to handle corner cases).
2. Compute the $x$-coordinate of the intersection:
   $x_{\text{int}} = x_i + (y_p - y_i) \cdot (x_j - x_i) / (y_j - y_i)$.
3. If $x_p < x_{\text{int}}$, toggle the inside flag.

The method accepts 2D or 3D input points; the $z$-component is ignored if present.

---

## 13.4 Grasp Analysis

The grasp analysis module (`src/robotics/contact/grasp_analysis.py`) provides
tools for evaluating multi-fingered grasps in terms of force closure, quality
metrics, and wrench-space coverage.

### 13.4.1 The Grasp Matrix

`compute_grasp_matrix(contacts, object_frame=None)` constructs the matrix $G$
that maps stacked contact forces to the resultant object wrench:

$$
\mathbf{w} = G \mathbf{f}
$$

where $\mathbf{w} \in \mathbb{R}^6$ is the wrench (force + torque) applied to
the object at its reference frame, and $\mathbf{f} \in \mathbb{R}^{3 n_c}$ is
the concatenation of all contact forces.

**Structure.** For $n_c$ point contacts, the grasp matrix has shape
$(6, 3 n_c)$:

$$
G = \begin{bmatrix}
I_3 & I_3 & \cdots & I_3 \\
[\mathbf{r}_1]_\times & [\mathbf{r}_2]_\times & \cdots & [\mathbf{r}_{n_c}]_\times
\end{bmatrix}
$$

where $\mathbf{r}_i = \mathbf{p}_i - \mathbf{p}_{\text{obj}}$ is the vector
from the object frame to the $i$-th contact point, and $[\mathbf{r}]_\times$ is
the skew-symmetric matrix:

$$
[\mathbf{r}]_\times =
\begin{bmatrix}
0 & -r_z & r_y \\
r_z & 0 & -r_x \\
-r_y & r_x & 0
\end{bmatrix}
$$

This matrix satisfies the identity $[\mathbf{r}]_\times \mathbf{f} = \mathbf{r} \times \mathbf{f}$.

The top block of $G$ sums all contact forces to produce the net force. The
bottom block computes the net torque by summing the moment arm contributions
$\mathbf{r}_i \times \mathbf{f}_i$.

If `object_frame` is `None`, the centroid of all contact positions is used as
the reference:

$$
\mathbf{p}_{\text{obj}} = \frac{1}{n_c} \sum_{i=1}^{n_c} \mathbf{p}_i
$$

**Precondition:** at least 1 contact must be provided.

**Postcondition:** the result has shape `(6, 3 * len(contacts))`.

```python
from src.robotics.contact.grasp_analysis import compute_grasp_matrix

G = compute_grasp_matrix(contacts, object_frame=np.array([0, 0, 0.05]))
# G.shape == (6, 3 * len(contacts))
```

### 13.4.2 Force Closure Check

`check_force_closure(contacts, num_cone_faces=8)` determines whether a grasp
can resist arbitrary external wrenches. A grasp has **force closure** if and only
if the origin of wrench space lies strictly inside the convex hull of the
achievable wrenches (the contact wrench cone).

**Algorithm.** The check proceeds in three stages:

1. **Build wrench generators.** For each contact, create a `FrictionCone` with
   the contact's friction coefficient and normal, then compute the cone
   generators. Each force generator $\mathbf{g}_j$ at contact $i$ produces a
   wrench generator:

$$
\mathbf{w}_{ij} = \begin{bmatrix} \mathbf{g}_j \\ [\mathbf{r}_i]_\times \mathbf{g}_j \end{bmatrix}
$$

The full set of wrench generators is stored as columns of a $(6, N)$ matrix,
where $N = n_c \cdot n_{\text{faces}}$.

2. **Check generator span.** If there are fewer than 6 generators, the wrench
   space cannot be spanned and force closure is impossible.

3. **Linear programming feasibility test.** Solve the LP:

$$
\min \; \mathbf{0}^T \boldsymbol{\alpha}
\quad \text{s.t.} \quad
\sum_{i=1}^{N} \alpha_i \mathbf{w}_i = \mathbf{0}, \quad
\sum_{i=1}^{N} \alpha_i = 1, \quad
\alpha_i \geq 0
$$

This checks whether the origin can be expressed as a convex combination of
the wrench generators. If the LP is feasible, the grasp has force closure.
The quality margin is reported as the minimum $\alpha_i$ in the solution --
a larger value indicates the origin is deeper inside the convex hull.

The implementation uses `scipy.optimize.linprog` with the HiGHS solver. If
SciPy is unavailable, a heuristic fallback checks the rank of the generator
matrix via SVD: if all 6 singular values exceed $10^{-6}$, force closure is
assumed likely.

**Precondition:** at least 2 contacts are required.

```python
from src.robotics.contact.grasp_analysis import check_force_closure

has_closure, margin = check_force_closure(contacts, num_cone_faces=8)
if has_closure:
    print(f"Force closure achieved, quality margin: {margin:.4f}")
```

### 13.4.3 Grasp Quality Metrics

`compute_grasp_quality(contacts, metric="min_singular_value")` quantifies the
grasp's ability to transmit forces to the object using the singular values of the
grasp matrix.

The grasp matrix $G$ is decomposed via SVD:

$$
G = U \Sigma V^T
$$

where $\Sigma = \text{diag}(\sigma_1, \sigma_2, \ldots, \sigma_k)$ with
$\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_k > 0$ (singular values
exceeding $10^{-10}$ are retained).

Three metrics are available:

| Metric                 | Formula                         | Interpretation                                                                             |
| ---------------------- | ------------------------------- | ------------------------------------------------------------------------------------------ |
| `"min_singular_value"` | $\sigma_{\min}(G)$              | Worst-case force transmission. Larger is better.                                           |
| `"volume"`             | $\prod_{i=1}^{k} \sigma_i$      | Volume of the grasp wrench ellipsoid. Larger indicates broader wrench coverage.            |
| `"isotropy"`           | $\sigma_{\min} / \sigma_{\max}$ | Uniformity of force transmission across directions. A value of 1.0 is perfectly isotropic. |

A degenerate grasp (all singular values below threshold) returns quality 0.0.

```python
from src.robotics.contact.grasp_analysis import compute_grasp_quality

q_min = compute_grasp_quality(contacts, metric="min_singular_value")
q_vol = compute_grasp_quality(contacts, metric="volume")
q_iso = compute_grasp_quality(contacts, metric="isotropy")
```

### 13.4.4 Contact Wrench Cone

`compute_contact_wrench_cone(contacts, num_faces=8)` returns the full set of
wrench generators without performing the force-closure check. The result is a
$(6, N)$ matrix where each column is a wrench generator:

$$
\mathbf{w}_{ij} =
\begin{bmatrix}
\mathbf{g}_j \\
\mathbf{r}_i \times \mathbf{g}_j
\end{bmatrix}
$$

This is useful for custom wrench-space analyses beyond the built-in metrics.

### 13.4.5 Required Contact Forces

`required_contact_forces(contacts, desired_wrench)` solves for the minimum-norm
contact forces that produce a specified object wrench:

$$
\min_{\mathbf{f}} \; \|\mathbf{f}\|^2
\quad \text{s.t.} \quad G \mathbf{f} = \mathbf{w}_d
$$

The implementation uses `scipy.optimize.minimize` with the SLSQP method. The
initial guess is the least-squares pseudoinverse solution
$\mathbf{f}_0 = G^+ \mathbf{w}_d$. Box bounds approximate friction-cone
feasibility by limiting each force component to
$\pm f_n (1 + \mu)$.

Returns the optimal force vector $(3 n_c,)$ or `None` if the problem is
infeasible.

```python
from src.robotics.contact.grasp_analysis import required_contact_forces

desired_wrench = np.array([0, 0, -9.81 * mass, 0, 0, 0])
forces = required_contact_forces(contacts, desired_wrench)
if forces is not None:
    print(f"Required forces: {forces}")
```

---
