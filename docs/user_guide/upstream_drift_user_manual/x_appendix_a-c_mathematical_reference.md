# UpstreamDrift User Manual -- Part 7: Appendices

---

## Appendix A: Mathematical Reference

This appendix consolidates the mathematical notation, equations, and derivations
referenced throughout the UpstreamDrift (Golf Modeling Suite) documentation.
All notation follows standard conventions from robotics, biomechanics, and
computational mechanics literature.

---

### A.1 Notation Conventions

The following symbols are used consistently across the codebase, API
documentation, and this manual.

#### A.1.1 State Variables

| Symbol                          | Domain              | Description                                         |
| ------------------------------- | ------------------- | --------------------------------------------------- |
| $q \in \mathbb{R}^{n_q}$        | Configuration space | Generalized coordinates (joint angles, positions)   |
| $\dot{q} \in \mathbb{R}^{n_v}$  | Tangent space       | Generalized velocities (joint angular/linear rates) |
| $\ddot{q} \in \mathbb{R}^{n_v}$ | Acceleration space  | Generalized accelerations                           |
| $\tau \in \mathbb{R}^{n_u}$     | Actuation space     | Applied torques and forces                          |
| $t \in \mathbb{R}_{\geq 0}$     | Time domain         | Simulation time in seconds                          |

Note: In general $n_q = n_v$ for systems without quaternion joints. For models
using quaternion orientation representation, $n_q = n_v + 1$ per quaternion
joint because the quaternion has four components but only three degrees of
freedom.

#### A.1.2 Dynamic Matrices

| Symbol                                          | Dimensions  | Description                                        |
| ----------------------------------------------- | ----------- | -------------------------------------------------- |
| $M(q) \in \mathbb{R}^{n_v \times n_v}$          | Square, SPD | Mass (inertia) matrix; symmetric positive-definite |
| $C(q, \dot{q}) \in \mathbb{R}^{n_v \times n_v}$ | Square      | Coriolis and centrifugal effects matrix            |
| $g(q) \in \mathbb{R}^{n_v}$                     | Vector      | Gravitational forces vector                        |
| $B \in \mathbb{R}^{n_v \times n_u}$             | Rectangular | Actuation selection matrix                         |
| $J(q) \in \mathbb{R}^{m \times n_v}$            | Rectangular | Task-space Jacobian                                |
| $J_c(q) \in \mathbb{R}^{n_c \times n_v}$        | Rectangular | Contact Jacobian                                   |
| $\Lambda(q) \in \mathbb{R}^{m \times m}$        | Square, SPD | Operational space inertia matrix                   |

The operational space inertia is defined as:

$$\Lambda(q) = \left( J(q) \, M(q)^{-1} \, J(q)^T \right)^{-1}$$

#### A.1.3 Task-Space Variables

| Symbol                         | Domain                   | Description                              |
| ------------------------------ | ------------------------ | ---------------------------------------- |
| $\mathbf{x} \in SE(3)$         | Special Euclidean group  | Task-space pose (position + orientation) |
| $\mathbf{p} \in \mathbb{R}^3$  | Euclidean space          | Cartesian position                       |
| $R \in SO(3)$                  | Special orthogonal group | Rotation matrix                          |
| $\mathcal{V} \in \mathbb{R}^6$ | Spatial velocity         | Twist (angular + linear velocity)        |
| $\mathcal{F} \in \mathbb{R}^6$ | Spatial force            | Wrench (moment + force)                  |
| $f_c \in \mathbb{R}^{n_c}$     | Contact forces           | Contact force magnitudes                 |

#### A.1.4 Muscle Model Variables

| Symbol                        | Domain        | Description                                       |
| ----------------------------- | ------------- | ------------------------------------------------- |
| $a \in [0, 1]$                | Bounded       | Muscle activation level                           |
| $u \in [0, 1]$                | Bounded       | Neural excitation signal                          |
| $l_M \in \mathbb{R}_{>0}$     | Positive real | Muscle fiber length                               |
| $l_T \in \mathbb{R}_{>0}$     | Positive real | Tendon length                                     |
| $l_{MT} \in \mathbb{R}_{>0}$  | Positive real | Musculotendon path length                         |
| $F_M \in \mathbb{R}_{\geq 0}$ | Non-negative  | Total muscle force                                |
| $F_0 \in \mathbb{R}_{>0}$     | Positive      | Maximum isometric muscle force                    |
| $\tilde{l}_M$                 | Normalized    | Normalized fiber length $= l_M / l_M^{opt}$       |
| $\tilde{v}_M$                 | Normalized    | Normalized fiber velocity $= \dot{l}_M / v_{max}$ |
| $r_j$                         | Real          | Muscle moment arm about joint $j$                 |

#### A.1.5 Ball Flight Variables

| Symbol                           | Domain        | Description                                             |
| -------------------------------- | ------------- | ------------------------------------------------------- |
| $\mathbf{r} \in \mathbb{R}^3$    | Position      | Ball position vector (x=downrange, y=lateral, z=height) |
| $\mathbf{v} \in \mathbb{R}^3$    | Velocity      | Ball velocity vector                                    |
| $\omega \in \mathbb{R}_{\geq 0}$ | Scalar        | Ball spin rate (rad/s)                                  |
| $\hat{s} \in \mathbb{R}^3$       | Unit vector   | Spin axis direction                                     |
| $C_D$                            | Dimensionless | Drag coefficient                                        |
| $C_L$                            | Dimensionless | Lift (Magnus) coefficient                               |
| $\rho$                           | kg/m^3        | Air density                                             |
| $A$                              | m^2           | Ball cross-sectional area                               |
| $e$                              | Dimensionless | Coefficient of restitution                              |

#### A.1.6 Subscript and Superscript Conventions

| Notation          | Meaning                                    |
| ----------------- | ------------------------------------------ |
| $(\cdot)_k$       | Value at time step $k$                     |
| $(\cdot)^{opt}$   | Optimal or reference value                 |
| $(\cdot)^{slack}$ | Slack (rest) length                        |
| $(\cdot)^{max}$   | Maximum value                              |
| $[\cdot]_\times$  | Skew-symmetric (cross-product) matrix form |
| $\hat{(\cdot)}$   | Unit vector or bias-corrected estimate     |
| $\tilde{(\cdot)}$ | Normalized quantity                        |

---

### A.2 Equations of Motion

#### A.2.1 Manipulator Equation

The fundamental equation governing the rigid-body dynamics of an articulated
system with $n_v$ velocity degrees of freedom:

$$M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q) = B\tau + J_c(q)^T f_c$$

where:

- $M(q)\ddot{q}$ represents inertial forces,
- $C(q, \dot{q})\dot{q}$ represents velocity-dependent (Coriolis and centrifugal) forces,
- $g(q)$ represents gravitational forces,
- $B\tau$ represents applied actuator forces/torques,
- $J_c(q)^T f_c$ represents contact forces mapped to joint space.

**Properties of $M(q)$:**

1. Symmetric: $M(q) = M(q)^T$
2. Positive-definite: $x^T M(q) x > 0$ for all $x \neq 0$
3. Bounded: $\lambda_{min} I \preceq M(q) \preceq \lambda_{max} I$ for finite $\lambda_{min}, \lambda_{max} > 0$

**Property of $\dot{M} - 2C$:**
The matrix $\dot{M}(q) - 2C(q, \dot{q})$ is skew-symmetric when $C$ is computed
using Christoffel symbols:

$$C_{ij} = \sum_k \frac{1}{2}\left( \frac{\partial M_{ij}}{\partial q_k} + \frac{\partial M_{ik}}{\partial q_j} - \frac{\partial M_{jk}}{\partial q_i} \right) \dot{q}_k$$

This property is essential for passivity-based control and energy analysis.

#### A.2.2 Forward Dynamics

Given $q, \dot{q}, \tau$, compute $\ddot{q}$:

$$\ddot{q} = M(q)^{-1}\left( B\tau + J_c^T f_c - C(q, \dot{q})\dot{q} - g(q) \right)$$

In practice, rather than explicitly forming $M^{-1}$, one solves the linear
system $M(q)\ddot{q} = b$ using Cholesky decomposition ($O(n_v^2)$ for
dense, $O(n_v)$ for sparse/articulated using the Articulated Body Algorithm).

#### A.2.3 Inverse Dynamics

Given $q, \dot{q}, \ddot{q}$, compute $\tau$:

$$\tau = M(q)\ddot{q} + C(q, \dot{q})\dot{q} + g(q)$$

Efficiently computed using the Recursive Newton-Euler Algorithm (RNEA) in
$O(n_v)$ time.

#### A.2.4 Articulated Body Algorithm (ABA)

The ABA computes forward dynamics in $O(n_v)$ for tree-structured kinematic
chains. The algorithm proceeds in three passes:

**Pass 1 (outward):** Compute velocities and bias forces for each body $i$:

$$v_i = {}^i X_{\lambda(i)} \, v_{\lambda(i)} + S_i \dot{q}_i$$

$$a_i^{bias} = {}^i X_{\lambda(i)} \, a_{\lambda(i)}^{bias} + S_i c_i + v_i \times S_i \dot{q}_i$$

**Pass 2 (inward):** Compute articulated body inertias $I_i^A$ and bias forces $p_i^A$:

$$I_i^A = I_i + \sum_{j \in \text{children}(i)} I_j^a$$

$$p_i^A = p_i^{bias} + \sum_{j \in \text{children}(i)} p_j^a$$

where the projected quantities are:

$$U_i = I_i^A S_i, \quad D_i = S_i^T U_i, \quad u_i = \tau_i - S_i^T p_i^A$$

**Pass 3 (outward):** Compute accelerations:

$$\ddot{q}_i = D_i^{-1}(u_i - U_i^T a_{\lambda(i)})$$

$$a_i = a_{\lambda(i)}^{bias} + S_i \ddot{q}_i$$

#### A.2.5 Composite Rigid Body Algorithm (CRBA)

The CRBA computes the mass matrix $M(q)$ in $O(n_v^2)$. This is used
when the full mass matrix is needed (e.g., for operational space control):

$$M_{ij} = S_i^T \, I_i^c \, S_j$$

where $I_i^c$ is the composite rigid body inertia of body $i$ and all its
descendants.

---

### A.3 Spatial Algebra

Spatial algebra provides a compact, coordinate-invariant notation for
rigid body dynamics. UpstreamDrift uses spatial algebra internally in its
Pinocchio and Drake engine backends.

#### A.3.1 Spatial Velocity (Twist)

A spatial velocity combines angular and linear velocity into a single 6D vector:

$$\mathcal{V} = \begin{bmatrix} \omega \\ v \end{bmatrix} \in \mathbb{R}^6$$

where $\omega \in \mathbb{R}^3$ is the angular velocity and $v \in \mathbb{R}^3$
is the linear velocity of the body-fixed frame origin.

#### A.3.2 Spatial Force (Wrench)

A spatial force combines moment and force:

$$\mathcal{F} = \begin{bmatrix} n \\ f \end{bmatrix} \in \mathbb{R}^6$$

where $n \in \mathbb{R}^3$ is the moment (torque) and $f \in \mathbb{R}^3$
is the linear force. The power generated by a wrench acting through a twist is:

$$P = \mathcal{F}^T \mathcal{V} = n^T \omega + f^T v$$

#### A.3.3 Spatial Inertia

The $6 \times 6$ spatial inertia matrix of a rigid body:

$$\mathcal{I} = \begin{bmatrix} I + m[c]_\times [c]_\times^T & m[c]_\times \\ m[c]_\times^T & mE_3 \end{bmatrix}$$

where:

- $m$ is the body mass,
- $c \in \mathbb{R}^3$ is the center of mass offset from the frame origin,
- $I \in \mathbb{R}^{3 \times 3}$ is the rotational inertia about the center of mass,
- $E_3$ is the $3 \times 3$ identity matrix,
- $[c]_\times$ is the skew-symmetric matrix corresponding to cross product with $c$.

**Properties:**

- $\mathcal{I}$ is symmetric positive-definite (for $m > 0$)
- $\mathcal{I}$ has exactly 10 independent parameters: $m$, $c$ (3), $I$ (6 due to symmetry)

#### A.3.4 Spatial Cross Product Operators

**Motion cross product:**

$$\mathcal{V} \times = \begin{bmatrix} [\omega]_\times & 0_{3 \times 3} \\ [v]_\times & [\omega]_\times \end{bmatrix}$$

**Force cross product (dual):**

$$\mathcal{V} \times^* = -(\mathcal{V} \times)^T = \begin{bmatrix} [\omega]_\times^T & [v]_\times^T \\ 0_{3 \times 3} & [\omega]_\times^T \end{bmatrix}$$

#### A.3.5 Skew-Symmetric Matrix

For a vector $a = [a_1, a_2, a_3]^T$, the skew-symmetric form is:

$$[a]_\times = \begin{bmatrix} 0 & -a_3 & a_2 \\ a_3 & 0 & -a_1 \\ -a_2 & a_1 & 0 \end{bmatrix}$$

such that $[a]_\times b = a \times b$ for any $b \in \mathbb{R}^3$.

#### A.3.6 Plucker Transforms

A Plucker transform ${}^A X_B$ maps spatial quantities from frame $B$ to
frame $A$. Given rotation $R \in SO(3)$ and translation $p \in \mathbb{R}^3$:

**Motion transform:**

$${}^A X_B = \begin{bmatrix} R & 0 \\ [p]_\times R & R \end{bmatrix}$$

**Force transform:**

$${}^A X_B^{*} = \begin{bmatrix} R & [p]_\times R \\ 0 & R \end{bmatrix}$$

These satisfy ${}^A X_B^{*} = ({}^A X_B)^{-T}$.

---

### A.4 Rotation Representations

UpstreamDrift supports multiple rotation representations. Each has
trade-offs between compactness, singularity avoidance, and computational
efficiency.

#### A.4.1 Rotation Matrix

A rotation matrix $R \in SO(3)$ satisfies:

$$R^T R = I, \quad \det(R) = 1$$

**Properties:**

- 9 parameters with 6 constraints (3 DOF)
- No singularities
- Composition: $R_{AC} = R_{AB} \, R_{BC}$
- Inverse: $R^{-1} = R^T$

#### A.4.2 Euler Angles (ZYX Convention)

UpstreamDrift uses the ZYX (yaw-pitch-roll) convention by default:

$$R = R_z(\psi) \, R_y(\theta) \, R_x(\phi)$$

where:

$$R_x(\phi) = \begin{bmatrix} 1 & 0 & 0 \\ 0 & \cos\phi & -\sin\phi \\ 0 & \sin\phi & \cos\phi \end{bmatrix}$$

$$R_y(\theta) = \begin{bmatrix} \cos\theta & 0 & \sin\theta \\ 0 & 1 & 0 \\ -\sin\theta & 0 & \cos\theta \end{bmatrix}$$

$$R_z(\psi) = \begin{bmatrix} \cos\psi & -\sin\psi & 0 \\ \sin\psi & \cos\psi & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Gimbal lock** occurs when $\theta = \pm \pi/2$, causing loss of one degree
of freedom. UpstreamDrift detects gimbal lock proximity and issues warnings
when $|\theta - \pi/2| < 0.01$ rad.

#### A.4.3 Axis-Angle (Rodrigues' Formula)

A rotation of angle $\theta$ about unit axis $\hat{k}$:

$$R = I + \sin\theta \, [\hat{k}]_\times + (1 - \cos\theta) \, [\hat{k}]_\times^2$$

This is the matrix exponential of the skew-symmetric matrix:

$$R = \exp(\theta \, [\hat{k}]_\times) = \exp([\omega]_\times)$$

where $\omega = \theta \hat{k}$ is the rotation vector. The inverse mapping
(logarithm) is:

$$\theta = \arccos\!\left(\frac{\text{tr}(R) - 1}{2}\right), \quad [\hat{k}]_\times = \frac{R - R^T}{2\sin\theta}$$

#### A.4.4 Unit Quaternion

A unit quaternion $\mathbf{q} = [w, x, y, z]^T$ with $\|\mathbf{q}\| = 1$ represents
a rotation. Writing $\mathbf{q} = [w, \mathbf{v}]$ where $\mathbf{v} = [x, y, z]^T$:

**Quaternion multiplication:**

$$\mathbf{q}_1 \otimes \mathbf{q}_2 = \begin{bmatrix} w_1 w_2 - \mathbf{v}_1 \cdot \mathbf{v}_2 \\ w_1 \mathbf{v}_2 + w_2 \mathbf{v}_1 + \mathbf{v}_1 \times \mathbf{v}_2 \end{bmatrix}$$

**Rotating a point:**

$$\mathbf{p}' = \mathbf{q} \otimes [0, \mathbf{p}]^T \otimes \mathbf{q}^{-1}$$

where $\mathbf{q}^{-1} = [w, -\mathbf{v}]^T$ for unit quaternions (conjugate).

**Quaternion to rotation matrix:**

$$R = \begin{bmatrix} 1-2(y^2+z^2) & 2(xy-wz) & 2(xz+wy) \\ 2(xy+wz) & 1-2(x^2+z^2) & 2(yz-wx) \\ 2(xz-wy) & 2(yz+wx) & 1-2(x^2+y^2) \end{bmatrix}$$

**Quaternion integration (angular velocity $\omega$ over timestep $\Delta t$):**

$$\Delta \mathbf{q} = \begin{bmatrix} \cos(\|\omega\| \Delta t / 2) \\ \sin(\|\omega\| \Delta t / 2) \, \hat{\omega} \end{bmatrix}$$

$$\mathbf{q}_{k+1} = \mathbf{q}_k \otimes \Delta \mathbf{q}$$

**Advantages over Euler angles:**

- No gimbal lock
- Smooth interpolation via SLERP
- Compact (4 parameters, 1 constraint)
- Efficient composition

**Quaternion SLERP (Spherical Linear Interpolation):**

$$\text{slerp}(\mathbf{q}_0, \mathbf{q}_1, t) = \frac{\sin((1-t)\Omega)}{\sin\Omega} \mathbf{q}_0 + \frac{\sin(t\Omega)}{\sin\Omega} \mathbf{q}_1$$

where $\cos\Omega = \mathbf{q}_0 \cdot \mathbf{q}_1$ and $t \in [0, 1]$.

---

### A.5 Optimization Methods

UpstreamDrift employs several optimization methods for trajectory
optimization, parameter identification, and motion planning.

#### A.5.1 SLSQP (Sequential Least Squares Programming)

Used for small-to-medium scale constrained optimization. Solves a sequence of
quadratic programming subproblems:

$$\min_{d} \quad \frac{1}{2} d^T H_k d + \nabla f(x_k)^T d$$

$$\text{s.t.} \quad \nabla c_i(x_k)^T d + c_i(x_k) = 0, \quad i \in \mathcal{E}$$

$$\quad\quad\; \nabla c_i(x_k)^T d + c_i(x_k) \geq 0, \quad i \in \mathcal{I}$$

where $H_k$ is a BFGS approximation to the Hessian of the Lagrangian.

UpstreamDrift uses SciPy's `minimize(method='SLSQP')` for problems with
fewer than approximately 100 decision variables.

#### A.5.2 Adam Optimizer

Used for neural network training in reinforcement learning policies:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$$

$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$$

Bias-corrected estimates:

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

Parameter update:

$$\theta_{t+1} = \theta_t - \frac{\eta \, \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

**Default hyperparameters in UpstreamDrift:**

- Learning rate: $\eta = 10^{-3}$
- First moment decay: $\beta_1 = 0.9$
- Second moment decay: $\beta_2 = 0.999$
- Numerical stability: $\epsilon = 10^{-8}$

#### A.5.3 SNOPT and IPOPT

For large-scale nonlinear programming (trajectory optimization with Drake):

$$\min_{x} \quad f(x)$$

$$\text{s.t.} \quad c_L \leq c(x) \leq c_U$$

$$\quad\quad\; x_L \leq x \leq x_U$$

**SNOPT** uses a sparse sequential quadratic programming (SQP) method with
limited-memory quasi-Newton Hessian updates. Preferred when the constraint
Jacobian is sparse.

**IPOPT** uses a primal-dual interior-point method with line-search or
trust-region globalization. Preferred for problems with many inequality
constraints.

#### A.5.4 Direct Collocation

Trajectory optimization by transcribing continuous dynamics into a finite-
dimensional nonlinear program. UpstreamDrift supports Hermite-Simpson
collocation.

**Decision variables:** States $x_k$ and controls $u_k$ at $N$ knot points.

**Hermite-Simpson collocation defect constraints:**

Midpoint state interpolation:

$$x_{k+1/2} = \frac{x_k + x_{k+1}}{2} + \frac{h}{8}\left(f_k - f_{k+1}\right)$$

Simpson quadrature constraint:

$$x_{k+1} = x_k + \frac{h}{6}\left(f_k + 4 f_{k+1/2} + f_{k+1}\right)$$

where $f_k = f(x_k, u_k)$ is the dynamics evaluated at knot point $k$,
$f_{k+1/2} = f(x_{k+1/2}, u_{k+1/2})$ is the dynamics at the midpoint,
and $h$ is the time step between knot points.

**Typical problem sizes in UpstreamDrift:**

- Swing optimization (3 DOF, 100 knot points): ~600 decision variables, ~300 constraints
- Full-body swing (15 DOF, 200 knot points): ~6000 decision variables, ~3000 constraints

#### A.5.5 CMA-ES (Covariance Matrix Adaptation Evolution Strategy)

Used for derivative-free optimization of swing parameters:

$$\mathbf{x}_k^{(i)} \sim \mathcal{N}\!\left(\mathbf{m}_k, \sigma_k^2 \, C_k\right), \quad i = 1, \ldots, \lambda$$

where $\mathbf{m}_k$ is the distribution mean, $\sigma_k$ is the step size,
and $C_k$ is the covariance matrix, all adapted over generations $k$.

---

### A.6 Hill-Type Muscle Model -- Complete Derivation

UpstreamDrift implements Hill-type muscle models for musculoskeletal
simulation via the MyoSuite integration and the `activation_dynamics` module.

#### A.6.1 Musculotendon Architecture

A musculotendon unit consists of:

1. **Contractile Element (CE):** Active force generation
2. **Parallel Elastic Element (PE):** Passive stiffness of muscle fiber
3. **Series Elastic Element (SE/Tendon):** Tendon compliance

The total musculotendon length:

$$l_{MT} = l_T + l_M \cos\alpha$$

where $\alpha$ is the pennation angle, $l_T$ is the tendon length, and
$l_M$ is the muscle fiber length.

#### A.6.2 Active Muscle Force

The contractile element force depends on activation, force-length, and
force-velocity relationships:

$$F_{CE} = a \cdot F_0 \cdot f_L(\tilde{l}_M) \cdot f_V(\tilde{v}_M)$$

where:

- $a \in [0, 1]$ is the activation level,
- $F_0$ is the maximum isometric force,
- $\tilde{l}_M = l_M / l_M^{opt}$ is the normalized fiber length,
- $\tilde{v}_M = \dot{l}_M / v_{max}$ is the normalized fiber velocity.

#### A.6.3 Force-Length Relationship

The active force-length curve is modeled as a Gaussian:

$$f_L(\tilde{l}) = \exp\!\left( -\frac{(\tilde{l} - 1)^2}{\gamma} \right)$$

where $\gamma \approx 0.45$ controls the width of the bell curve. The muscle
generates peak force at its optimal length ($\tilde{l} = 1$) and diminishing
force at shorter or longer lengths.

#### A.6.4 Force-Velocity Relationship

The force-velocity relationship is asymmetric, with different behavior for
concentric (shortening) and eccentric (lengthening) contractions:

$$f_V(\tilde{v}) = \begin{cases} \displaystyle \frac{1 + \tilde{v}/\tilde{v}_{max}}{1 - \tilde{v}/(K \cdot \tilde{v}_{max})} & \tilde{v} \leq 0 \;\text{(concentric)} \\[10pt] \displaystyle N + (N - 1)\frac{\tilde{v}_{max} + \tilde{v}}{7.56 K \tilde{v} - \tilde{v}_{max}} & \tilde{v} > 0 \;\text{(eccentric)} \end{cases}$$

**Typical parameters:**

- $K \approx 0.25$: Shape factor for concentric curve
- $N \approx 1.5$: Eccentric force enhancement ratio
- $\tilde{v}_{max} \approx 10$: Maximum normalized shortening velocity

#### A.6.5 Passive Elastic Force

The parallel elastic element produces force when the muscle is stretched
beyond its optimal length:

$$F_{PE}(\tilde{l}_M) = k_{PE} \cdot \left[\max(0, \, \tilde{l}_M - 1)\right]^2 \cdot F_0$$

where $k_{PE} \approx 5.0$ is the passive stiffness coefficient.

#### A.6.6 Total Muscle Force

The total muscle force is:

$$F_M = F_{CE} + F_{PE} = a \cdot F_0 \cdot f_L(\tilde{l}_M) \cdot f_V(\tilde{v}_M) + F_{PE}(\tilde{l}_M)$$

#### A.6.7 Tendon Force

The tendon is modeled as a nonlinear elastic element:

$$F_T = F_T^{max} \cdot f_T(\epsilon_T)$$

where the tendon strain is:

$$\epsilon_T = \frac{l_T - l_T^{slack}}{l_T^{slack}}$$

and $f_T(\epsilon_T)$ is an exponential-linear tendon curve:

$$f_T(\epsilon) = \begin{cases} 0 & \epsilon < 0 \\ c_1 \exp(k_T \epsilon) - c_2 & 0 \leq \epsilon < \epsilon_{toe} \\ k_{lin}(\epsilon - \epsilon_{toe}) + F_{toe} & \epsilon \geq \epsilon_{toe} \end{cases}$$

with $\epsilon_{toe} \approx 0.033$ (3.3% strain at the toe region transition).

#### A.6.8 Activation Dynamics

The time delay between neural excitation and muscle activation is modeled
as a first-order differential equation (implemented in
`src/shared/python/activation_dynamics.py`):

$$\dot{a} = \frac{u - a}{\tau(u, a)}$$

where the time constant $\tau$ depends on whether the muscle is activating
or deactivating:

$$\tau = \begin{cases} \tau_{act} \cdot (0.5 + 1.5a) & \text{if } u > a \;\text{(activation)} \\ \tau_{deact} / (0.5 + 1.5a) & \text{if } u \leq a \;\text{(deactivation)} \end{cases}$$

**Typical values:**

- $\tau_{act} \approx 10 \text{ ms}$: Fast activation (calcium release)
- $\tau_{deact} \approx 40 \text{ ms}$: Slower deactivation (calcium reuptake)

This asymmetry is physiologically realistic: calcium release from the
sarcoplasmic reticulum is faster than reuptake by the calcium pump (SERCA).

#### A.6.9 Moment Arm and Joint Torque

The muscle moment arm about joint $j$ is the negative partial derivative of
the musculotendon path length with respect to the joint angle:

$$r_j = -\frac{\partial l_{MT}}{\partial q_j}$$

The total joint torque from all muscles crossing joint $j$:

$$\tau_j = \sum_{m=1}^{N_m} r_{j,m} \cdot F_M^m$$

where $N_m$ is the number of muscles crossing joint $j$.

#### A.6.10 Equilibrium Condition

At each instant, the force in the tendon must equal the component of muscle
force along the tendon direction:

$$F_T = F_M \cos\alpha$$

This equilibrium condition is solved implicitly at each time step to
determine the fiber length and tendon length partition.

---

### A.7 Numerical Integration Methods

#### A.7.1 Runge-Kutta `4th` Order (RK4)

The primary integrator for ball flight simulation
(`src/shared/python/ball_flight_physics.py`):

$$k_1 = f(t_n, y_n)$$

$$k_2 = f(t_n + h/2, \; y_n + h k_1 / 2)$$

$$k_3 = f(t_n + h/2, \; y_n + h k_2 / 2)$$

$$k_4 = f(t_n + h, \; y_n + h k_3)$$

$$y_{n+1} = y_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

Local truncation error: $O(h^5)$. Global error: $O(h^4)$.

#### A.7.2 Semi-Implicit Euler (MuJoCo Default)

MuJoCo uses a semi-implicit Euler scheme:

$$v_{n+1} = v_n + h \, M^{-1}(f_n + J_c^T \lambda)$$

$$q_{n+1} = q_n + h \, v_{n+1}$$

This is first-order but symplectic, preserving energy behavior over long
simulations. MuJoCo compensates with small time steps ($h \leq 0.002$ s).

#### A.7.3 Implicit Midpoint Rule

For stiff systems and contact-rich simulation:

$$q_{n+1} = q_n + h \, f\!\left(\frac{q_n + q_{n+1}}{2}\right)$$

Requires solving a nonlinear system at each step (Newton iterations).

---

## Appendix B: Model Specifications

This appendix details the biomechanical models available in UpstreamDrift,
ordered from simplest to most complex.

---

### B.1 Double Pendulum (2 DOF)

The simplest golf swing model, treating the arm-club system as a planar
double pendulum.

| Property                 | Segment 1 (Upper Arm)                      | Segment 2 (Forearm + Club)                 |
| ------------------------ | ------------------------------------------ | ------------------------------------------ |
| Length                   | 0.30 m                                     | 0.60 m                                     |
| Mass                     | 2.0 kg                                     | 1.5 kg                                     |
| MOI (about proximal end) | $\frac{1}{3} m l^2 = 0.060 \text{ kg m}^2$ | $\frac{1}{3} m l^2 = 0.180 \text{ kg m}^2$ |
| COM location             | 0.15 m from shoulder                       | 0.30 m from elbow                          |

**Joints:**

- Shoulder: Revolute, range $[-\pi, \pi]$ rad
- Elbow: Revolute, range $[0, \pi]$ rad (only flexion/extension)

**Degrees of freedom:** 2 ($n_q = n_v = 2$)

**Use cases:** Basic swing mechanics, wrist-lag analysis, teaching demonstrations.

**Configuration:** `q = [\theta_{shoulder}, \theta_{elbow}]^T$

---

### B.2 Triple Pendulum (3 DOF)

Extends the double pendulum with a wrist joint for studying club release
mechanics.

| Property   | Upper Arm     | Forearm    | Hand + Club            |
| ---------- | ------------- | ---------- | ---------------------- |
| Length     | 0.30 m        | 0.27 m     | 0.19 m + 1.14 m (club) |
| Mass       | 2.5 kg        | 1.5 kg     | 0.4 kg + 0.315 kg      |
| Joint      | Shoulder      | Elbow      | Wrist                  |
| Joint type | Revolute      | Revolute   | Revolute               |
| Range      | $[-\pi, \pi]$ | $[0, \pi]$ | $[-\pi/2, \pi/3]$      |

**Degrees of freedom:** 3 ($n_q = n_v = 3$)

**Configuration:** $q = [\theta_{shoulder}, \theta_{elbow}, \theta_{wrist}]^T$

**Use cases:** Wrist cock/release timing, kinematic sequence analysis, club
lag optimization.

---

### B.3 Upper Body Model (10 DOF)

A more realistic model including bilateral arm kinematics and trunk rotation.

| Body Segment   | DOF | Joint Types                            |
| -------------- | --- | -------------------------------------- |
| Torso          | 2   | Axial rotation, lateral bend           |
| Left shoulder  | 2   | Flexion/extension, abduction/adduction |
| Right shoulder | 2   | Flexion/extension, abduction/adduction |
| Left elbow     | 1   | Flexion/extension                      |
| Right elbow    | 1   | Flexion/extension                      |
| Left wrist     | 1   | Flexion/extension                      |
| Right wrist    | 1   | Flexion/extension                      |

**Total degrees of freedom:** 10 ($n_q = n_v = 10$)

**Segment masses (default, kg):**

- Torso: 35.0 (from `TORSO_MASS_KG`)
- Upper arm: 2.5 each (from `UPPER_ARM_MASS_KG`)
- Forearm: 1.5 each (from `FOREARM_MASS_KG`)
- Hand: 0.4 each (from `HAND_MASS_KG`)

**Use cases:** Bilateral coordination, trunk rotation contribution, X-factor
analysis, lead arm vs. trail arm mechanics.

---

### B.4 Full Body Model (15 DOF)

Adds lower body kinematics for ground reaction force modeling and
kinetic chain analysis.

| Body Segment | DOF | Joint Types                 |
| ------------ | --- | --------------------------- |
| Upper body   | 10  | (as in B.3)                 |
| Left hip     | 1   | Internal/external rotation  |
| Right hip    | 1   | Internal/external rotation  |
| Left knee    | 1   | Flexion/extension           |
| Right knee   | 1   | Flexion/extension           |
| Left ankle   | 1   | Dorsiflexion/plantarflexion |

**Total degrees of freedom:** 15 ($n_q = n_v = 15$)

**Additional segment masses (default, kg):**

- Thigh: 7.5 each
- Shank: 3.5 each
- Foot: 1.0 each

**Use cases:** Full kinetic chain, ground reaction forces, weight transfer,
lower body contribution to clubhead speed.

---

### B.5 Advanced Model (28 DOF)

High-fidelity model with scapular kinematics, 3-DOF shoulders, and
flexible club shaft.

| Body Segment   | DOF | Joint Types                                                        |
| -------------- | --- | ------------------------------------------------------------------ |
| Spine          | 4   | Flexion, extension, lateral bend, axial rotation                   |
| Left scapula   | 2   | Protraction/retraction, elevation/depression                       |
| Right scapula  | 2   | Protraction/retraction, elevation/depression                       |
| Left shoulder  | 3   | Flexion/extension, abduction/adduction, internal/external rotation |
| Right shoulder | 3   | Flexion/extension, abduction/adduction, internal/external rotation |
| Left elbow     | 2   | Flexion/extension, pronation/supination                            |
| Right elbow    | 2   | Flexion/extension, pronation/supination                            |
| Left wrist     | 2   | Flexion/extension, radial/ulnar deviation                          |
| Right wrist    | 2   | Flexion/extension, radial/ulnar deviation                          |
| Hips           | 2   | Internal/external rotation (each)                                  |
| Knees          | 2   | Flexion/extension (each)                                           |

**Club shaft flexibility:** Additional DOF from modal or finite element
shaft model (see `src/shared/python/flexible_shaft.py`):

- Modal model: 2-4 mode shapes (bending modes)
- FE model: $2 \times n_{nodes}$ DOF (deflection + rotation per node)

**Total degrees of freedom:** 28 (rigid body) + shaft DOF

**Use cases:** Scapular plane analysis, shaft flexibility effects, pronation/
supination contribution, detailed shoulder mechanics, club path optimization.

---

### B.6 MyoSuite Full Body (52 DOF, 290 Muscles)

Complete musculoskeletal model integrated through the MyoSuite/OpenSim
interface.

| System                 | DOF | Muscles                                                  |
| ---------------------- | --- | -------------------------------------------------------- |
| Lumbar spine           | 3   | 36 (erector spinae, obliques, rectus abdominis, etc.)    |
| Thoracic spine         | 3   | 24                                                       |
| Neck                   | 3   | 16                                                       |
| Left shoulder complex  | 5   | 22 (deltoid, rotator cuff, pectoralis, latissimus, etc.) |
| Right shoulder complex | 5   | 22                                                       |
| Left elbow/forearm     | 2   | 18 (biceps, triceps, brachialis, pronators, supinators)  |
| Right elbow/forearm    | 2   | 18                                                       |
| Left wrist/hand        | 4   | 20 (flexors, extensors, intrinsic hand muscles)          |
| Right wrist/hand       | 4   | 20                                                       |
| Left hip               | 3   | 22 (gluteals, iliopsoas, adductors, etc.)                |
| Right hip              | 3   | 22                                                       |
| Left knee              | 1   | 12 (quadriceps, hamstrings, gastrocnemius)               |
| Right knee             | 1   | 12                                                       |
| Left ankle/foot        | 3   | 14 (tibialis anterior/posterior, peroneals, intrinsic)   |
| Right ankle/foot       | 3   | 14                                                       |
| Subtalar joints        | 2   | 6                                                        |
| Metatarsophalangeal    | 4   | 12                                                       |

**Total:** 52 DOF, 290 muscles

**Muscle properties for each muscle include:**

- Maximum isometric force $F_0$
- Optimal fiber length $l_M^{opt}$
- Tendon slack length $l_T^{slack}$
- Pennation angle $\alpha_0$
- Maximum shortening velocity $v_{max}$
- Activation/deactivation time constants $\tau_{act}, \tau_{deact}$

**Use cases:** Detailed muscle coordination, co-contraction analysis,
injury risk assessment, EMG-driven simulation, metabolic cost estimation,
muscle-level optimization of swing technique.

---

### B.7 Standard Humanoid (66 DOF)

Derived from the human-gazebo `humanSubject06_66dof` model (robotology
project). This high-fidelity model includes full-body articulation with
mesh-based collision and visualization geometries.

**Key properties:**

- Height: 1.75 m
- Mass: 75.0 kg
- Format: URDF with STL meshes
- License: CC-BY-SA 2.0
- Cross-engine compatible: MuJoCo, Drake, Pinocchio

---

## Appendix C: Golf Ball Physics

This appendix documents the ball flight model implemented in
`src/shared/python/ball_flight_physics.py` and the aerodynamics module.

---

### C.1 USGA Ball Specifications

All ball parameters conform to USGA and R&A equipment rules:

| Parameter                | Value                                      | Rule Reference    |
| ------------------------ | ------------------------------------------ | ----------------- |
| Maximum mass             | 45.93 g (1.620 oz)                         | USGA Rule 5-1     |
| Minimum diameter         | 42.67 mm (1.680 in)                        | USGA Rule 5-2     |
| Maximum COR              | 0.830 at 48.8 m/s (109 mph)                | USGA Appendix III |
| Maximum initial velocity | 76.2 m/s (250 ft/s) + 2% tolerance         | USGA Appendix III |
| Symmetry                 | Must not deviate > 0.75% between poles     | USGA Appendix III |
| Overall distance         | Not to exceed 317 yards + 3 yard tolerance | USGA Appendix III |

**Dimple specifications:**

- Typical count: 300-500 dimples
- Dimple depth: 0.127-0.178 mm (0.005-0.007 in)
- Dimple diameter: 3.56-4.45 mm (0.140-0.175 in)
- Coverage: approximately 75-80% of ball surface
- Patterns: icosahedral, octahedral, or proprietary

**Default ball properties in UpstreamDrift** (from `BallProperties` dataclass):

| Property                     | Symbol   | Default Value                          |
| ---------------------------- | -------- | -------------------------------------- |
| Mass                         | $m$      | 0.04593 kg                             |
| Diameter                     | $d$      | 0.04267 m                              |
| Radius                       | $r$      | 0.021335 m                             |
| Cross-sectional area         | $A$      | $\pi r^2 = 1.430 \times 10^{-3}$ m$^2$ |
| Drag coefficient (base)      | $C_{D0}$ | 0.21                                   |
| Drag coefficient (linear)    | $C_{D1}$ | 0.05                                   |
| Drag coefficient (quadratic) | $C_{D2}$ | 0.02                                   |
| Lift coefficient (base)      | $C_{L0}$ | 0.00                                   |
| Lift coefficient (linear)    | $C_{L1}$ | 0.38                                   |
| Lift coefficient (quadratic) | $C_{L2}$ | 0.08                                   |

---

### C.2 Ball Flight Model

#### C.2.1 Governing Equation

The ball is treated as a point mass with aerodynamic forces:

$$m \ddot{\mathbf{r}} = \mathbf{F}_g + \mathbf{F}_D + \mathbf{F}_L$$

where the three force components are gravity, drag, and lift (Magnus).

**Coordinate system:**

- $x$: downrange (positive toward target)
- $y$: lateral (positive to the right from behind the ball)
- $z$: vertical (positive upward)

#### C.2.2 Gravitational Force

$$\mathbf{F}_g = \begin{bmatrix} 0 \\ 0 \\ -mg \end{bmatrix}$$

where $g = 9.80665$ m/s$^2$ (from `GRAVITY_M_S2`).

#### C.2.3 Aerodynamic Drag

$$\mathbf{F}_D = -\frac{1}{2} \rho \, v_{rel} \, C_D \, A \, \mathbf{v}_{rel}$$

where:

- $\rho = 1.225$ kg/m$^3$ is air density at sea level and 15 C (from `AIR_DENSITY_SEA_LEVEL_KG_M3`),
- $\mathbf{v}_{rel} = \mathbf{v} - \mathbf{v}_{wind}$ is the velocity relative to the air,
- $v_{rel} = \|\mathbf{v}_{rel}\|$ is the relative speed,
- $A = \pi r^2$ is the cross-sectional area.

The drag coefficient depends on the spin ratio $S = \omega r / v_{rel}$:

$$C_D(S) = C_{D0} + S \cdot C_{D1} + S^2 \cdot C_{D2}$$

For a typical driver shot ($v = 70$ m/s, $\omega = 280$ rad/s, $r = 0.021$ m):
$S \approx 0.084$, $C_D \approx 0.214$.

#### C.2.4 Magnus Force (Lift)

The Magnus effect creates a force perpendicular to both the velocity and spin
axis:

$$\mathbf{F}_L = \frac{1}{2} \rho \, v_{rel}^2 \, C_L \, A \, \frac{\hat{s} \times \hat{v}_{rel}}{\|\hat{s} \times \hat{v}_{rel}\|}$$

where $\hat{s}$ is the unit spin axis vector and $C_L$ depends on the spin ratio:

$$C_L(S) = \min\!\left(C_{L,max}, \; C_{L0} + S \cdot C_{L1} + S^2 \cdot C_{L2}\right)$$

The maximum lift coefficient is capped at $C_{L,max} = 0.25$ to prevent
unphysical behavior at extreme spin rates.

**Backspin** (spin axis pointing left from behind) creates upward lift,
increasing carry distance and landing angle.

**Sidespin** creates lateral drift (draw/fade/hook/slice).

#### C.2.5 Spin Decay

Ball spin decays exponentially due to air resistance:

$$\omega(t) = \omega_0 \, e^{-t / \tau_{spin}}$$

where $\tau_{spin} \approx 20$ s is the spin decay time constant
(from `SPIN_DECAY_RATE_S = 0.05` 1/s, so $\tau_{spin} = 1/0.05 = 20$ s).

In the `EnhancedBallFlightSimulator`, spin decay is computed per time step:

$$\omega_{n+1} = \omega_n \cdot e^{-\Delta t / \tau_{spin}}$$

#### C.2.6 Environmental Effects

**Altitude correction:** Air density decreases with altitude:

$$\rho(h) = \rho_0 \exp\!\left(-\frac{Mgh}{RT}\right)$$

where $M = 0.029$ kg/mol is the molar mass of air, $R = 8.314$ J/(mol K), and
$T$ is the absolute temperature.

**Temperature correction:** Air density is inversely proportional to temperature:

$$\rho(T) = \rho_0 \cdot \frac{T_0}{T}$$

**Humidity:** Moist air is less dense than dry air (water vapor is lighter
than N$_2$/O$_2$), reducing drag.

**Wind model** (from `aerodynamics.py`):

- Base wind velocity (constant component)
- Gust model (random impulses with configurable frequency and magnitude)
- Turbulence model (Dryden or von Karman spectral models)

---

### C.3 Impact Model

The club-ball impact determines launch conditions from swing delivery
parameters.

#### C.3.1 Coefficient of Restitution

The COR relates separation and approach speeds:

$$e = -\frac{v_{sep}}{v_{approach}} = -\frac{v_{ball}' - v_{club}'}{v_{ball} - v_{club}}$$

For a stationary ball ($v_{ball} = 0$):

$$e = \frac{v_{ball}' - v_{club}'}{v_{club}}$$

USGA maximum COR: $e = 0.830$ (for drivers at standard test speed).

In practice, COR varies with:

- Impact speed (COR decreases at very high speeds due to material nonlinearity)
- Impact location on the face (lower at heel/toe due to reduced face flex)
- Club type (irons have lower COR than drivers)

#### C.3.2 Ball Speed

From conservation of momentum and the COR definition:

$$v_{ball} = \frac{(1 + e) \, m_{club}}{m_{club} + m_{ball}} \cdot v_{club}$$

For typical values ($e = 0.83$, $m_{club} = 0.200$ kg, $m_{ball} = 0.046$ kg):

$$\frac{v_{ball}}{v_{club}} = \frac{(1 + 0.83) \times 0.200}{0.200 + 0.046} = \frac{0.366}{0.246} \approx 1.49$$

This "smash factor" of ~1.49 is consistent with measured driver performance.

#### C.3.3 Launch Angle

The launch angle depends on the club loft $\alpha_{loft}$, the angle of
attack $\alpha_{AoA}$ (dynamic loft), and the impact location:

$$\alpha_{launch} \approx k_1 \cdot \alpha_{dynamic\,loft} + k_2 \cdot \alpha_{AoA}$$

where $k_1 \approx 0.85$ and $k_2 \approx 0.15$ for a driver. The dynamic
loft includes shaft lean and face angle at impact.

#### C.3.4 Spin Generation

Ball spin arises from two mechanisms:

**Loft spin (backspin):** Friction between the ball and the lofted club face:

$$\omega_{back} \propto \frac{\mu \, F_N \, \Delta t}{I_{ball}} \cdot \sin(\alpha_{loft})$$

**Gear effect spin:** For off-center impacts, the club head rotates about its
center of gravity, imparting additional spin to the ball in the opposite
direction. For toe hits on a driver, this produces draw spin (right-to-left
for a right-handed golfer).

**Total spin axis tilt** determines the ratio of backspin to sidespin:

$$\omega_{total} = \sqrt{\omega_{back}^2 + \omega_{side}^2}$$

$$\text{Axis tilt} = \arctan\!\left(\frac{\omega_{side}}{\omega_{back}}\right)$$

---

### C.4 Typical Ball Flight Parameters

Reference values for a well-struck driver shot by a professional golfer:

| Parameter      | Value                     |
| -------------- | ------------------------- |
| Clubhead speed | 50-56 m/s (112-125 mph)   |
| Ball speed     | 74-83 m/s (165-186 mph)   |
| Smash factor   | 1.48-1.51                 |
| Launch angle   | 10-14 degrees             |
| Backspin rate  | 2200-2800 rpm             |
| Carry distance | 255-290 m (280-317 yards) |
| Maximum height | 28-35 m (92-115 ft)       |
| Flight time    | 5.5-6.5 s                 |
| Landing angle  | 38-45 degrees             |

---
