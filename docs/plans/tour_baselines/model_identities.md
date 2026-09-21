# Model Identities, Topologies, and Degrees of Freedom

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Governing Issue: [#10585](https://github.com/D-sorganization/UpstreamDrift/issues/10585) (TB-00)

---

## 1. Taxonomic Classification

All models operating across the UpstreamDrift and Tools ecosystems are cataloged by their topological structure, physical constraints, degrees of freedom, and club representation.

```
                    ┌───────────────────────────────────────────────┐
                    │               Golf Models Taxonomy            │
                    └───────────────────────┬───────────────────────┘
                                            │
        ┌───────────────────┬───────────────┴───────────────┬───────────────────┐
        ▼                   ▼                               ▼                   ▼
┌───────────────┐   ┌───────────────┐               ┌───────────────┐   ┌───────────────┐
│ Kinematic     │   │ Planar Driven │               │ Constrained   │   │ Full-Body     │
│ Reconstruct   │   │ Pendulums     │               │ Upper Body    │   │ Multibody     │
│ (No Club)     │   │ (Sim Club)    │               │ (Closed Loop) │   │ (Dual Club)   │
└───────────────┘   └───────────────┘               └───────────────┘   └───────────────┘
```

---

## 2. Model Enumeration and Specification

### 2.1 Kinematic Reconstruction Models

Defined in `src/motion_capture/reconstruct/model/registry.py` and `golfer.py`. These models solve inverse kinematics for motion capture marker clouds without dynamic equations of motion, mass properties, or applied joint torques.

- `reconstruction_golfer`:

  - **Source File:** `src/motion_capture/reconstruct/model/golfer.py`
  - **Topology:** Open-chain kinematic tree with 17 anatomical joints (scapula, torso, limbs).
  - **Degrees of Freedom:** 17 rotational joints.
  - **Club Representation:** **None**.
  - **Source Owner:** UpstreamDrift.
  - **Aliases:** `golfer` (in `reconstruction` context).

- `reconstruction_double_pendulum`:

  - **Source File:** `src/motion_capture/reconstruct/model/registry.py`
  - **Topology:** Pivot $\rightarrow$ Arm $\rightarrow$ Hands.
  - **Degrees of Freedom:** 4 coordinates (3-DOF pivot frame rotation $\mathbf{R}_{xyz}$ selecting the swing plane, 1 planar hinge about normal).
  - **Club Representation:** **None** (hands represent the mean midpoint of left and right wrists).
  - **Source Owner:** UpstreamDrift.
  - **Aliases:** `double_pendulum` (in `reconstruction` context), `double-pendulum/1.0`.

- `reconstruction_triple_pendulum`:
  - **Source File:** `src/motion_capture/reconstruct/model/registry.py`
  - **Topology:** Pivot $\rightarrow$ Upper Arm $\rightarrow$ Forearm $\rightarrow$ Hands.
  - **Degrees of Freedom:** 5 coordinates (3-DOF pivot frame rotation, 2 planar hinges for elbow and wrist).
  - **Club Representation:** **None**.
  - **Source Owner:** UpstreamDrift.
  - **Aliases:** `triple_pendulum` (in `reconstruction` context), `triple-pendulum/1.0`.

---

### 2.2 Planar Driven Pendulums

Defined in `src/shared/python/pendulum_simulator/` and the shipped `double_pendulum_golf` package (`Tools` repository). These are forward-dynamics Lagrangian models with mass matrices, Coriolis/centrifugal forces, gravitational potential, and applied joint torque functions.

- `driven_double_pendulum`:

  - **Source File:** `src/shared/python/pendulum_simulator/physics.py` / `src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py`
  - **Topology:** Planar 2-link mechanism constrained to an inclined plane ($\beta$).
  - **Generalized Coordinates:** $\mathbf{q} = [\theta_1, \theta_2]^T$ (shoulder angle $\theta_1$, wrist angle $\theta_2$).
  - **Degrees of Freedom:** 2 independent DOFs (State dimension 4: $[\theta_1, \theta_2, \dot{\theta}_1, \dot{\theta}_2]$).
  - **Club Representation:** **Simulated composite golf club** (shaft mass, shaft center-of-mass ratio, clubhead point mass, length, rotational inertia).
  - **Source Owner:** `Tools` repository.
  - **Aliases:** `double`, `double_pendulum` (in `pendulum_simulator` context), `double_pendulum_golf`, `double_pendulum_analytical`.

- `driven_triple_pendulum`:
  - **Source File:** `src/shared/python/pendulum_simulator/physics_triple.py`
  - **Topology:** Planar 3-link mechanism with moving hub pivot.
  - **Generalized Coordinates:** $\mathbf{q} = [\theta_0, \theta_1, \theta_2]^T$ (hub rotation $\theta_0$, arm angle $\theta_1$, wrist/club angle $\theta_2$).
  - **Degrees of Freedom:** 3 independent DOFs (State dimension 6).
  - **Club Representation:** **Simulated golf club**.
  - **Source Owner:** `Tools` repository.
  - **Aliases:** `triple`, `triple_pendulum` (in `pendulum_simulator` context), `triple_pendulum_golf`.

---

### 2.3 Constrained Upper-Body Golfer (Closed Kinematic Loop)

Defined in `src/shared/python/pendulum_simulator/physics_golfer.py` and `golfer_constraints.py`.

- `constrained_upper_body_golfer`:
  - **Source File:** `src/shared/python/pendulum_simulator/physics_golfer.py`
  - **Topology:** Fixed origin $\rightarrow$ massless standoff $\rightarrow$ hub bar $\rightarrow$ bilateral shoulder joints (RS, LS) $\rightarrow$ bilateral arm chains (elbows RE/LE, wrists RH/LH) $\rightarrow$ shared rigid club gripped by both hands.
  - **Generalized Coordinates (8 Coordinates):**
    $$\mathbf{q} = [\theta_{\text{hub}}, \alpha_{\text{rs}}, \alpha_{\text{re}}, \alpha_{\text{rh}}, \alpha_{\text{ls}}, \alpha_{\text{le}}, \alpha_{\text{lh}}, \theta_{\text{club}}]^T$$
  - **Holonomic Loop Closure Constraints (4 Equations):**
    $$\boldsymbol{\Phi}(\mathbf{q}) = \begin{bmatrix} \mathbf{p}_{\text{LH}}(\mathbf{q}) - \mathbf{p}_{\text{grip\_left}}(\mathbf{q}) \\ (\mathbf{p}_{\text{LH}} - \mathbf{p}_{\text{RH}}) \cdot \hat{\mathbf{u}}_{\perp}(\theta_{\text{club}}) \\ (\mathbf{p}_{\text{LH}} - \mathbf{p}_{\text{RH}}) \cdot \hat{\mathbf{u}}_{\parallel}(\theta_{\text{club}}) - d_{\text{grip}} \end{bmatrix} = \mathbf{0} \in \mathbb{R}^4$$
  - **Constraint Rank and Independent DOFs:**
    Analytical evaluation and SVD of the constraint Jacobian $\mathbf{J}_c = \frac{\partial \boldsymbol{\Phi}}{\partial \mathbf{q}} \in \mathbb{R}^{4 \times 8}$ demonstrates that row 3 and row 4 are collinear projections of the 2D vector position closure. The algebraic rank of $\mathbf{J}_c$ is **3**.
    $$\text{Independent DOFs} = N_{\text{coords}} - \text{rank}(\mathbf{J}_c) = 8 - 3 = \mathbf{5}$$
  - **Club Representation:** Rigid club shaft and tip clubhead gripped at offset positions by both hands.
  - **Source Owner:** `Tools` repository.
  - **Aliases:** `golfer` (in `pendulum_simulator` context), `golfer_upper_body`, `upper_body_golfer`, `golfer_sim`.

---

### 2.4 Flagship Full-Body Engine Models

Governed by the Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363), [#10378](https://github.com/D-sorganization/UpstreamDrift/issues/10378)).

| Model ID              | Backend               | DOFs | Constraints                            | Club Representation                    | Source Owner  | Governing Issue |
| --------------------- | --------------------- | ---- | -------------------------------------- | -------------------------------------- | ------------- | --------------- |
| `full_body_mujoco`    | MuJoCo                | 38   | Foot-ground contact, bilateral grip    | Driver & 7-Iron CAD/inertial meshes    | UpstreamDrift | #10378          |
| `full_body_pinocchio` | Pinocchio / Crocoddyl | 46   | Finite weld Jacobian, contact cone     | Rigid club bodies with inertia tensors | UpstreamDrift | #10377, #10378  |
| `full_body_drake`     | Drake MultibodyPlant  | 40   | Hydroelastic contact, bilateral grip   | Multibody club element                 | UpstreamDrift | #10375, #10378  |
| `full_body_opensim`   | OpenSim Moco          | 35   | Coordinate couplers, muscle dynamics   | Rigid club attached to hand bodies     | UpstreamDrift | #10376, #10414  |
| `full_body_simscape`  | Simscape Multibody    | 26   | Physical joints, mechanical hard stops | Flexible/rigid shaft + head block      | UpstreamDrift | #9921           |
| `full_body_myosuite`  | MyoSuite Neural       | 30   | Neural activation (fail-closed)        | MyoSuite asset (pending retarget)      | UpstreamDrift | #9478           |

---

### 2.5 Catalog Reference URDF / MJCF Presets (#9914)

Bundled native reference models evaluated in historical epic #9914:

- `reference_pinocchio_urdf` (`pinocchio_golfer`): 46 DOF full body URDF.
- `reference_pinocchio_urdf_ik` (`pinocchio_golfer_ik`): 46 DOF IK-optimized URDF.
- `reference_drake_urdf` (`drake_golfer`): 40 DOF Drake URDF.
- `reference_simple_humanoid` (`simple_humanoid`): 28 DOF simplified humanoid URDF (no club).
- `reference_human_subject` (`human_subject`): 32 DOF anthropomorphic mesh URDF (no club).
- `reference_mujoco_humanoid` (`mujoco_humanoid`): 21 DOF bundled MJCF humanoid (no club).
- `myosuite_body`: Unavailable placeholder asset.
- `opensim_golfer`: Unavailable placeholder model requiring OpenSim native adapter.
