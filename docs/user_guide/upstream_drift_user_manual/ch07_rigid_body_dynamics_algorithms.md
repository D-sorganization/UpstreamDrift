# Chapter 7: Rigid Body Dynamics Algorithms

## 7.1 Overview

The `rigid_body_dynamics/` subpackage provides pure-Python implementations of
the three foundational algorithms of multibody dynamics, following Featherstone's
formulation with 6D spatial algebra:

```
rigid_body_dynamics/
    rnea.py               -- Recursive Newton-Euler Algorithm (inverse dynamics)
    crba.py               -- Composite Rigid Body Algorithm (mass matrix)
    aba.py                -- Articulated Body Algorithm (forward dynamics)
    induced_acceleration.py -- Induced acceleration analysis
    common.py             -- Shared constants and utilities
```

All algorithms operate on a **model dictionary** with the following fields:

| Key       | Type               | Description                                       |
| --------- | ------------------ | ------------------------------------------------- |
| `NB`      | `int`              | Number of bodies                                  |
| `parent`  | `np.ndarray`       | Parent body indices ($-1$ for root)               |
| `jtype`   | `list[str]`        | Joint types (`"Rx"`, `"Ry"`, `"Rz"`, `"Px"`, ...) |
| `Xtree`   | `list[np.ndarray]` | Tree transforms ($6 \times 6$ spatial)            |
| `I`       | `list[np.ndarray]` | Spatial inertias ($6 \times 6$)                   |
| `gravity` | `np.ndarray`       | 6D spatial gravity vector                         |

---

## 7.2 Recursive Newton-Euler Algorithm (RNEA)

**File:** `rnea.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 5, Algorithm 5.1.

### 7.2.1 Problem Statement

Given joint positions $q$, velocities $\dot{q}$, and accelerations $\ddot{q}$,
compute the required joint torques:

$$\tau = \text{RNEA}(q, \dot{q}, \ddot{q}) = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q)$$

### 7.2.2 Algorithm

```python
def rnea(model, q, qd, qdd, f_ext=None) -> np.ndarray:
```

**Pass 1 -- Forward Kinematics** (root to leaves):

For each body $i$ with parent $p(i)$:

1. Compute joint transform $X_J$ and motion subspace $S_i$ from `jcalc(jtype, q[i])`.
2. Combined transform: $X_{\text{up},i} = X_J \cdot X_{\text{tree},i}$.
3. Spatial velocity:

$$v_i = X_{\text{up},i} \, v_{p(i)} + S_i \dot{q}_i$$

4. Spatial acceleration:

$$a_i = X_{\text{up},i} \, a_{p(i)} + S_i \ddot{q}_i + v_i \times^m (S_i \dot{q}_i)$$

For the root body ($p(i) = -1$), $v_i = S_i \dot{q}_i$ and the base acceleration
incorporates gravity: $a_i = X_J(-a_g) + S_i \ddot{q}_i$.

**Pass 2 -- Backward Dynamics** (leaves to root):

For each body $i$ (in reverse order):

1. Compute the net spatial force:

$$f_i = I_i a_i + v_i \times^f (I_i v_i) - f_{\text{ext},i}$$

2. Project to joint torque:

$$\tau_i = S_i^T f_i$$

3. Propagate force to parent:

$$f_{p(i)} \mathrel{+}= X_{\text{up},i}^T f_i$$

### 7.2.3 Complexity

RNEA runs in $O(n)$ time where $n$ is the number of bodies. The implementation
uses several optimizations:

- **Pre-allocated Fortran-ordered arrays** for cache-friendly column access.
- **Sparse motion subspace exploitation**: for standard revolute/prismatic joints,
  $S_i$ has a single non-zero element, reducing 6-element dot products to scalar
  reads.
- **Cached dictionary lookups** to avoid repeated hashing in tight loops.

---

## 7.3 Composite Rigid Body Algorithm (CRBA)

**File:** `crba.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 6, Algorithm 6.1.

### 7.3.1 Problem Statement

Given joint positions $q$, compute the joint-space mass matrix $H(q)$:

$$H(q) = \sum_{i=1}^{n} S_i^T \, I_i^c \, S_i$$

where $I_i^c$ is the **composite rigid body inertia** of the subtree rooted at
body $i$.

```python
def crba(model, q) -> np.ndarray:
```

### 7.3.2 Algorithm

**Forward Pass** -- compute joint transforms and motion subspaces for each body.

**Backward Pass** -- composite inertia accumulation (leaves to root):

$$I_{p(i)}^c \mathrel{+}= X_{\text{up},i}^T \, I_i^c \, X_{\text{up},i}$$

**Mass Matrix Assembly:**

For each body $i$:

1. Compute $f = I_i^c S_i$ (force due to unit acceleration at joint $i$).
2. Diagonal element: $H_{ii} = S_i^T f$.
3. Propagate $f$ up the tree. For each ancestor $j$:

$$H_{ij} = H_{ji} = S_j^T f_{\text{transformed}}$$

### 7.3.3 Complexity

CRBA runs in $O(n^2)$ time due to the off-diagonal element computation.
Optimizations include:

- **Pre-computed contiguous transposes** of $X_{\text{up}}$ for efficient `np.dot`.
- **Sparse motion subspace**: column extraction replaces full matrix-vector
  products ($36 \to 6$ operations).
- **Buffer swapping** (instead of copying) in the ancestor propagation loop.

### 7.3.4 Properties

The returned mass matrix satisfies:

- **Symmetry:** $H = H^T$ (enforced by simultaneous assignment of $H_{ij}$ and $H_{ji}$).
- **Positive definiteness:** All eigenvalues are positive (guaranteed by construction
  from positive-definite spatial inertias).

---

## 7.4 Articulated Body Algorithm (ABA)

**File:** `aba.py`

**Reference:** Featherstone, R. (2008). _Rigid Body Dynamics Algorithms_.
Chapter 7, Algorithm 7.1.

### 7.4.1 Problem Statement

Given joint positions $q$, velocities $\dot{q}$, and applied torques $\tau$,
compute the resulting joint accelerations:

$$\ddot{q} = \text{ABA}(q, \dot{q}, \tau)$$

This is equivalent to solving $M(q)\ddot{q} = \tau - C(q, \dot{q})\dot{q} - g(q)$,
but in $O(n)$ time without forming or inverting the mass matrix.

```python
def aba(model, q, qd, tau, f_ext=None) -> np.ndarray:
```

### 7.4.2 Algorithm

**Pass 1 -- Forward Kinematics** (root to leaves):

Same as RNEA Pass 1, computing velocities $v_i$ and bias accelerations $c_i$
(velocity-product terms).

Additionally, the bias force is computed:

$$p_i^A = v_i \times^f (I_i v_i) - f_{\text{ext},i}$$

**Pass 2 -- Backward Recursion** (leaves to root):

For each body $i$, compute the articulated-body inertia $I_i^A$ and bias force
$p_i^A$:

1. $U_i = I_i^A S_i$ (projected inertia along joint axis).
2. $d_i = S_i^T U_i$ (scalar joint-space inertia for 1-DOF joints).
3. $u_i = \tau_i - S_i^T p_i^A$ (net joint-space force).
4. Update parent articulated inertia:

$$I_{p(i)}^A \mathrel{+}= X_{\text{up},i}^T \left( I_i^A - \frac{U_i U_i^T}{d_i} \right) X_{\text{up},i}$$

5. Update parent bias force:

$$p_{p(i)}^A \mathrel{+}= X_{\text{up},i}^T \left( p_i^A + I_i^A c_i + \frac{u_i}{d_i} U_i \right)$$

**Pass 3 -- Forward Recursion** (root to leaves):

Compute spatial and joint accelerations:

$$\ddot{q}_i = \frac{u_i - U_i^T a_i}{d_i}$$

$$a_i = X_{\text{up},i} a_{p(i)} + c_i + S_i \ddot{q}_i$$

For root bodies, $a_i = X_{\text{up},i}(-a_g) + c_i$.

### 7.4.3 Complexity

ABA runs in $O(n)$ time with three passes. The implementation features:

- **Rank-1 update avoidance** of outer products via careful buffer management.
- **Numerical tolerance** $\epsilon = 10^{-10}$ to prevent division by zero in
  degenerate configurations.
- **Batch pre-allocation** of all work arrays at the start.

### 7.4.4 Relationship to Other Algorithms

| Algorithm | Solves                                   | Complexity | Notes            |
| --------- | ---------------------------------------- | ---------- | ---------------- |
| RNEA      | $\tau = \text{ID}(q, \dot{q}, \ddot{q})$ | $O(n)$     | Inverse dynamics |
| CRBA      | $M(q)$                                   | $O(n^2)$   | Mass matrix only |
| ABA       | $\ddot{q} = \text{FD}(q, \dot{q}, \tau)$ | $O(n)$     | Forward dynamics |

ABA is strictly more efficient than the na\"ive approach of computing $M$ via CRBA
and then solving $M\ddot{q} = \tau - b$, which requires $O(n^2)$ for CRBA plus
$O(n^2)$ or $O(n^3)$ for the solve.

---

## 7.5 Induced Acceleration Analysis

**File:** `induced_acceleration.py`

### 7.5.1 `MuJoCoInducedAccelerationAnalyzer`

This module decomposes the total joint acceleration into contributions from
individual force sources.

```python
class MuJoCoInducedAccelerationAnalyzer:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData): ...
```

### 7.5.2 Mathematical Formulation

Starting from the Euler-Lagrange equation:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = \tau + J^T f_c$$

The total acceleration decomposes as:

$$\ddot{q} = \underbrace{M^{-1}(-g)}_{\ddot{q}_g} + \underbrace{M^{-1}(-C\dot{q})}_{\ddot{q}_c} + \underbrace{M^{-1}\tau}_{\ddot{q}_\tau} + \underbrace{M^{-1}J^T f_c}_{\ddot{q}_{cn}}$$

### 7.5.3 Efficient Batch Solve

Rather than solving four separate linear systems, the implementation stacks
all right-hand sides and performs a single batch solve:

```python
rhs_stack = np.column_stack((-term_G, -term_C, tau_vec, data.qfrc_constraint))
results = np.linalg.solve(M, rhs_stack)  # One LU decomposition for all 4
```

### 7.5.4 Result Structure

```python
class InducedAccelerationResult(TypedDict):
    gravity:    np.ndarray   # M^{-1}(-g)
    velocity:   np.ndarray   # M^{-1}(-C*v)
    control:    np.ndarray   # M^{-1}(tau)
    constraint: np.ndarray   # M^{-1}(J^T f_c)
    total:      np.ndarray   # Sum of all components
```

### 7.5.5 Task-Space Decomposition

The `compute_task_space_components()` method maps joint-space induced
accelerations to Cartesian-space (world frame) accelerations for a specific body:

$$\ddot{x}_{\text{source}} = J \ddot{q}_{\text{source}}$$

The velocity-dependent term additionally includes the bias acceleration
$\dot{J}\dot{q}$:

$$\ddot{x}_{\text{velocity}} = J \ddot{q}_c + \underbrace{(\ddot{x}_{\text{total}} - J \ddot{q}_{\text{total}})}_{\dot{J}\dot{q}}$$

This decomposition reveals, for example, how much of the club head acceleration
at impact is due to gravity versus active muscle torques.

---

---
