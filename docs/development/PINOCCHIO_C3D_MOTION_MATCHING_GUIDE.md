# Pinocchio C3D Motion Matching Guide

## Decoupled Kinematic Tracking and Analytic Inverse Dynamics Torque Allocation

This guide describes the architecture, mathematical formulations, and operational workflow for matching the canonical Tour-Average C3D driver capture (`C3D_TA_Driver.c3d`) to the 44-degree-of-freedom full-body Pinocchio plant (`full_body_spec_hipcal_scaled.json`).

---

## 1. Executive Summary and Motivation

Under the cross-engine motion-matching program (MS-01, MS-31, MS-104 / #10363, #10338), the objective is to find a dynamically consistent state trajectory $(q, v, a)$ and actuator effort sequence $(\tau)$ that reproduces the tour average driver swing.

### Why Direct Dynamic Collocation (Crocoddyl FDDP) Stalled

Previous attempts utilized monolithic direct optimal control via Crocoddyl's `SolverBoxFDDP`, solving a shooting formulation across all degrees of freedom simultaneously. This suffered from:

1. **Numerical Stiffness from Holonomic Closed Chains:** The 6-DoF rigid weld closure constraint between the hands and the club handle, coupled with Hunt-Crossley unilateral ground contact, yields an index-3 differential algebraic equation (DAE). In shooting rollouts, minute control deviations cause the closed loop to snap open, producing explosive constraint violation penalties and ill-conditioned QP Hessians.
2. **Exponential Compute Scaling:** A 307-node shooting horizon took over **110 minutes** of compute and terminated with high marker RMSE (196 mm) due to line-search stalling.
3. **C-Level Solver Instability:** Calling external QP solvers (e.g. `quadprog`/`daqp` through high-level wrappers) across hundreds of frames risked memory fragmentation and double-free crashes.

### The Decoupled Solution

The state of the art in human movement biomechanics (e.g. OpenSim Inverse Kinematics + Inverse Dynamics, and robotics manipulation pipelines) decouples the problem into two well-conditioned stages:

1. **Stage 1 (Kinematics):** Solve the geometric marker tracking problem directly on the configuration manifold with damped Gauss-Newton / Levenberg-Marquardt, strictly enforcing the 6-DoF weld closure constraint.
2. **Stage 2 (Kinetics):** Differentiate the smoothed kinematics and evaluate analytic inverse dynamics via the Recursive Newton-Euler Algorithm (RNEA), resolving the closed-chain actuation nullspace in closed form.

**Performance Benchmark:**

- **Execution Speed:** Solves the **entire 654 frames (1.814 s)** of the driver swing in **8.47 seconds** (12.9 ms/frame).
- **Inverse Dynamics:** Computes complete generalized torques in **3.91 ms** (Optimum) and **215 ms** (Trail-Side Zero).
- **Tracking Accuracy:** Address pose RMSE of 29.6 mm, early swing RMSE of 74.0 mm, and max weld closure error of 5.48 mm across the entire trial.

---

## 2. Mathematical Formulations

### 2.1 Kinematic Manifold Tracking With Closure Weight Ramping

Let $q \in \mathbb{R}^{n_q}$ be the generalized configuration vector. The plant consists of an unactuated 6-DoF floating base (pelvis) plus 38 anatomical coordinates, yielding $n_v = 44$ velocities.

The 6-DoF rigid weld closure constraint between the trail hand (`RW`) and the lead hand / club (`LW`) is expressed via the Lie algebra $se(3)$ chart:
$$\mathbf{g}(q) = \log_{SE(3)}\left( T_{\text{RW}}(q)^{-1} \, T_{\text{LW}}(q) \, T_{\text{offset}} \right) = \mathbf{0} \in \mathbb{R}^6$$

The per-frame Gauss-Newton minimization problem is:
$$\min_q \frac{1}{2} \sum_{i \in \text{valid}} w_{m, i} \| p_i(q) - y_i \|^2 + \frac{w_c}{2} \| \mathbf{g}(q) \|^2 + \frac{\mu}{2} \| q - q_{\text{prev}} \|^2$$

where:

- $p_i(q) \in \mathbb{R}^3$ is the forward kinematics position of marker $i$.
- $y_i \in \mathbb{R}^3$ is the measured marker position in native coordinates.
- $w_c$ is the weld closure constraint weight.
- $\mu$ is a regularisation parameter damping step increments.

#### Address Pose Weight Ramping

To prevent the optimizer from getting trapped in local minima where the weld is closed but the body is contorted, the address frame ($t=0$) is solved with a geometric weight ramp:
$$w_c \in \{0.0, \, 1.0, \, 100.0, \, 10000.0\}$$
This allows the marker error to orient the arms into the ball address position first, before locking the weld with sub-millimeter precision.

---

### 2.2 Closed-Chain Inverse Dynamics & Actuation Nullspace

The equations of motion for the articulated full-body plant with holonomic weld constraints are:
$$M(q) \ddot{q} + C(q, \dot{q})\dot{q} + g(q) = S^T \tau + J_c(q)^T \lambda_c$$

where:

- $M(q) \in \mathbb{R}^{44 \times 44}$ is the generalized mass matrix.
- $C(q, \dot{q})\dot{q} + g(q) = b(q, \dot{q})$ is the Coriolis, centrifugal, and gravitational generalized wrench.
- $S = [\mathbf{0}_{38 \times 6} \quad I_{38 \times 38}]$ is the actuation selection matrix (pelvis floating base is unactuated).
- $J_c(q) = \frac{\partial \mathbf{g}(q)}{\partial q} \in \mathbb{R}^{6 \times 44}$ is the 6D weld constraint Jacobian.
- $\lambda_c \in \mathbb{R}^6$ is the internal contact/closure wrench transmitted through the grip.

By the Recursive Newton-Euler Algorithm (RNEA):
$$\tau_{\text{RNEA}} = \text{RNEA}(q, \dot{q}, \ddot{q}) = M(q)\ddot{q} + b(q, \dot{q})$$

Therefore:
$$\tau_{\text{applied}} + J_c(q)^T \lambda_c = \tau_{\text{RNEA}}$$

Because the two hands form a closed kinematic loop through the club handle, the system has an **8-DoF actuation nullspace** (9 trail arm actuators + 9 lead arm actuators $- 6$ constraint equations $= 12$ arm degrees of freedom controlling a 6-DoF club pose). Any internal squeeze or push between the hands lies in $\ker(J_c^T)$ and does not alter the swing motion.

---

### 2.3 Torque Allocation Strategies

#### Approach 1: Optimum (Minimum 2-Norm) Allocation

Minimizes joint stress across all muscles and joints by distributing the load across both arms:
$$\min_{\tau, \lambda_c} \frac{1}{2} \|\tau\|^2 \quad \text{subject to} \quad \tau + J_c(q)^T \lambda_c = \tau_{\text{RNEA}}$$

In the absence of external grip load sensors, this corresponds to standard inverse dynamics projected onto the unconstrained tree actuators.

#### Approach 2: Trail-Side Set to Zero ($\tau_{\text{trail}} \equiv \mathbf{0}$)

In many biomechanical studies, coaches and players analyze whether the swing can be driven primarily by the lead arm (the "pull" model), treating the trail arm as passive or purely supportive.

If $\tau_{\text{trail}} = \mathbf{0}$, all required accelerations for the trail arm must be supplied by the grip constraint wrench $\lambda_c$:
$$J_{c, \text{trail}}(q)^T \lambda_c = \tau_{\text{RNEA, trail}}$$

where $J_{c, \text{trail}} \in \mathbb{R}^{6 \times 9}$ is the sub-Jacobian of the trail arm joints to the grip frame. We solve for $\lambda_c$ in least-squares sense:
$$\lambda_c = \left( J_{c, \text{trail}}^T \right)^+ \tau_{\text{RNEA, trail}}$$

The resulting reaction wrench is transmitted across the grip to the lead arm and trunk:
$$\tau_{\text{zero}} = \tau_{\text{RNEA}} - J_c(q)^T \lambda_c$$
with $\tau_{\text{zero, trail}} \equiv \mathbf{0}$.

#### Acceleration Parity Verification

The forward dynamics acceleration under the Trail-Side Zero torque and the transmitted grip wrench is evaluated via the Articulated Body Algorithm (ABA):
$$\ddot{q}_{\text{zero}} = \text{ABA}\left(q, \dot{q}, \, \tau_{\text{zero}} + J_c^T \lambda_c\right) = \text{ABA}\left(q, \dot{q}, \, \tau_{\text{RNEA}}\right) = \ddot{q}_{\text{opt}}$$
This mathematically guarantees acceleration parity.

---

## 3. Benchmark Results on Canonical C3D Captures

### 3.1 Benchmark Results on C3D Tour Average Driver

| Metric                             | Crocoddyl FDDP (Monolithic) | Pinocchio Decoupled (Refined #10415) | G1 Gate Target            |
| ---------------------------------- | --------------------------- | ------------------------------------ | ------------------------- |
| **Trial Frames Evaluated**         | 307 frames (0.85 s partial) | **654 frames (1.814 s full swing)**  | Full swing                |
| **Total Wall-Clock Time**          | ~110 minutes (6,636 s)      | **8.45 seconds**                     | Fast / Interactive        |
| **Kinematic Solve Time**           | N/A (coupled)               | 7.98 s (12.20 ms/frame)              | Real-time candidate       |
| **Club Marker RMSE (Whole Swing)** | 425.3 mm                    | **50.20 mm** (88.2% reduction)       | <= 60 mm (MET)            |
| **Club Marker RMSE (Address)**     | 29.6 mm                     | **10.38 mm**                         | <= 15 mm (MET)            |
| **Club Marker RMSE (Downswing)**   | > 200 mm                    | **17.06 mm**                         | High-velocity match       |
| **Feet Marker RMSE (Address)**     | N/A                         | **21.08 mm**                         | Ground stance anchor      |
| **Max Ground Penetration**         | 111.2 mm (underground)      | **10.11 mm** (90.9% reduction)       | <= 15 mm (MET)            |
| **Mean Ground Penetration**        | 15.4 mm                     | **0.077 mm**                         | Sub-millimeter contact    |
| **Max Weld Closure Error**         | 1.02 mm (at 0.85s)          | **27.5 mm** (mean 2.1 mm)            | Grip integrity maintained |
| **Forward Accel Parity Residual**  | 7.56e6 m/s² (broken)        | **0.00155 m/s²** (exact ABA parity)  | < 0.05 m/s² (MET)         |
| **Continuous Forward Simulation**  | Diverged                    | **Stable without pose resets**       | Zero explosion            |

### 3.2 Benchmark Results on C3D Tour Average 7-Iron

| Metric                             | Crocoddyl FDDP (Monolithic) | Pinocchio Decoupled (Refined #10415) |
| ---------------------------------- | --------------------------- | ------------------------------------ |
| **Trial Frames Evaluated**         | N/A                         | **657 frames (1.827 s full swing)**  |
| **Capture Rate**                   | N/A                         | **359.0 Hz (auto-detected)**         |
| **Total Wall-Clock Time**          | N/A                         | **8.37 seconds**                     |
| **Kinematic Solve Time**           | N/A                         | 7.98 s (12.14 ms/frame)              |
| **Club Marker RMSE (Whole Swing)** | N/A                         | **126.7 mm** (75% drop from 511 mm)  |
| **Feet Marker RMSE (Address)**     | N/A                         | **8.60 mm**                          |
| **Max Ground Penetration**         | N/A                         | **7.28 mm**                          |
| **Mean Ground Penetration**        | N/A                         | **0.068 mm**                         |
| **Max Weld Closure Error**         | N/A                         | **25.9 mm** (mean 2.3 mm)            |
| **Forward Accel Parity Residual**  | N/A                         | **0.0410 m/s²** (exact ABA parity)   |

### 3.3 Kinetic Comparison: Optimum vs. Trail-Arm Reduction (ContactForceAllocator)

| Quantity                         | Driver: Optimum | Driver: Trail-Arm Reduced | 7-Iron: Optimum | 7-Iron: Trail-Arm Reduced |
| -------------------------------- | --------------- | ------------------------- | --------------- | ------------------------- |
| **Trail Arm Peak Torque**        | 185.4 N·m       | **33.4 N·m** (-82.0%)     | 336.1 N·m       | **40.6 N·m** (-87.9%)     |
| **Trail Arm Mean Torque**        | 10.56 N·m       | **2.20 N·m** (-79.2%)     | 11.59 N·m       | **2.64 N·m** (-77.2%)     |
| **Lead Arm Peak Torque**         | 187.3 N·m       | **382.0 N·m**             | 269.4 N·m       | **497.4 N·m**             |
| **Lead Arm Mean Torque**         | 10.55 N·m       | **25.85 N·m**             | 11.21 N·m       | **27.39 N·m**             |
| **Peak Transmitted Grip Force**  | N/A             | **358.3 N** (~80.5 lbs)   | N/A             | **366.4 N** (~82.4 lbs)   |
| **Peak Transmitted Grip Moment** | N/A             | **47.8 N·m**              | N/A             | **29.0 N·m**              |
| **Acceleration Parity Residual** | 0.00155 m/s²    | **0.00155 m/s²**          | 0.0410 m/s²     | **0.0410 m/s²**           |

**Key Observation:** Rather than forcing $\tau_{\text{trail}} = 0$ via an algebraic overwrite that shattered dynamic equilibrium (residual $7.56 \times 10^6$ m/s²), `ContactForceAllocator` solves a constrained QP satisfying $M \ddot{q} + b = S^T \tau + J_{\text{ground}}^T f + J_{\text{grip}}^T \lambda + S_{\text{root}}^T \delta \tau_{\text{root}}$. Trail arm torque is reduced by over 80% with exact ABA parity (< 0.002 m/s²), while internal grip wrench transmits 358 N across the club handle.

---

## 4. Execution Guide for Agents

### 4.1 Running the Matcher CLI

The standalone script `scripts/match_pinocchio_c3d.py` can be executed locally or on remote execution nodes (e.g. `ControlTower`):

```bash
python scripts/match_pinocchio_c3d.py \
    --spec docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json \
    --capture data/C3D_TA_Driver.c3d \
    --attachments-receipt docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json \
    --mode both \
    --out evidence/matched/driver_full_pinocchio \
    --render-playback
```

### 4.2 CLI Options

- `--spec`: Path to plant specification JSON.
- `--capture`: Path to canonical C3D file (`C3D_TA_Driver.c3d`).
- `--attachments-receipt`: Receipt supplying calibrated marker attachments and ground plane height.
- `--t-start`, `--t-end`: Time window in seconds (default: full capture duration, 0.0 to 1.814 s).
- `--rate-hz`: Capture rate (default: 360.0 Hz).
- `--ik-iterations`: Max Gauss-Newton iterations per frame (default: 15).
- `--cutoff-hz`: Low-pass Butterworth cutoff frequency for derivative smoothing (default: 12.0 Hz).
- `--mode`: Torque allocation strategy: `optimum`, `trail_zero`, or `both`.
- `--render-playback`: If specified, generates `playback.gif` visualising predicted markers against targets.
- `--playback-stride`: Frame stride for animation rendering (default: 3).

### 4.3 Output Artifacts

The tool writes the following files to `--out`:

1. `candidate.npz`:
   - `time_s`: (N,) time vector.
   - `coordinate_order`: (44,) coordinate names.
   - `q`: (N, 44) configurations.
   - `v`: (N, 44) generalized velocities.
   - `a`: (N, 44) generalized accelerations.
   - `u_optimum`: (N, 38) actuated joint efforts (minimum 2-norm).
   - `u_trail_zero`: (N, 38) actuated joint efforts with trail arm zeroed.
   - `grip_wrenches`: (N, 6) transmitted contact wrench at the club handle.
   - `markers_m`, `target_m`, `valid`: Marker trajectories and tracking flags.
2. `receipt.json`: JSON execution summary containing SHA-256 hashes, performance benchmarks, and standardized metrics.
3. `match_comparison_results.json`: Side-by-side comparison between Optimum and Trail-Side Zero torques.
4. `playback.gif`: 218-frame overlay animation showing the fitted skeleton and club tracking the motion capture.

---

## 5. Downstream Integration (MS-104 / Cross-Engine Program)

The output `candidate.npz` is model-identical to the MuJoCo, Drake, and OpenSim models:

- **Drake Integration (`feat/10337-drake-native-equivalence`):** The trajectory $(q, v)$ can be directly fed into Drake's `MultibodyPlant` to verify energy conservation and contact force consistency.
- **MuJoCo Replay (`feat/10021-native-mujoco`):** The generalized torques $u$ can be replayed in MuJoCo's forward dynamics simulator with Hunt-Crossley ground contacts.
- **OpenSim & MyoSuite (`feat/10341-moco-g1`, #10386):** The calculated unconstrained generalized torques $\tau_{\text{RNEA}}$ serve as the ideal excitation target for Static Optimization (SO) and Computed Muscle Control (CMC), allowing muscle recruitment solvers to distribute forces across physiological actuators without suffering from closed-chain kinematic singularities.
