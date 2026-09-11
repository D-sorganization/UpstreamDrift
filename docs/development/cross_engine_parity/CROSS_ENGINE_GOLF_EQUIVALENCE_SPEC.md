# Cross-Engine Physics Equivalency Specification: Simscape, MuJoCo, Pinocchio, and Drake

> **Status:** Active Fleet Engineering Specification  
> **Parent Architecture:** [CROSS_ENGINE_PARITY_SPEC.md](../../../src/engines/CROSS_ENGINE_PARITY_SPEC.md)  
> **Primary Tracking Issue:** Epic: Cross-Engine Physics Equivalency (#9949)  
> **Participating Repositories:**
>
> - `D-sorganization/UpstreamDrift` (Core model identification & Simscape reference)
> - `D-sorganization/MuJoCo_Models` (High-speed inference & training)
> - `D-sorganization/Pinocchio_Models` (Robotics dynamics & analytical Jacobians)
> - `D-sorganization/Drake_Models` (Rigorous multibody optimization & autodiff)

---

## 1. Executive Summary & Objective

The goal of this initiative is to establish complete **functional and dynamical equivalency** between the reference **Simscape Multibody** 3D forward-dynamics golf swing model (`GolfSwing3D_Kinetic.slx`) and the three open-source robotics physics engines:

1. **MuJoCo** (MJCF / XML)
2. **Pinocchio** (URDF)
3. **Drake** (URDF / MultibodyPlant)

When driven by identical continuous polynomial torque profiles $\tau(t) = \sum_{k=0}^6 c_k B_{k,6}(t)$ from identical initial conditions $(q_0, \dot{q}_0)$, all four engines must reproduce the **exact same forward dynamics, joint trajectories, and anatomical marker kinematics within $< 5\text{ mm}$ RMSE**.

Simscape Multibody serves as the ground-truth baseline oracle. Any torque polynomial trajectory fitted against measured tour motion capture (`data/C3D_TA_Driver.c3d`) in Simscape must be directly transferable to MuJoCo, Pinocchio, and Drake to produce identical physical swings without per-engine retuning.

---

## 2. Canonical Model Architecture

To guarantee strict cross-engine parity, all engines derive their kinematic trees, link lengths, masses, centers of mass, and inertia tensors from a single authoritative specification:
[`src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml`](../../../src/engines/physics_engines/pinocchio/models/spec/golfer_canonical.yaml).

### 2.1 Coordinate System & Units

- **Length**: Meters ($\text{m}$)
- **Mass**: Kilograms ($\text{kg}$)
- **Time**: Seconds ($\text{s}$)
- **Angle**: Radians ($\text{rad}$)
- **Torque**: Newton-meters ($\text{N}\cdot\text{m}$)
- **Force**: Newtons ($\text{N}$)
- **World Frame Orientation**:
  - $+X$: Forward along target line
  - $+Y$: Lateral left
  - $+Z$: Up against gravity ($g = -9.80665\text{ m/s}^2$)

### 2.2 Anatomical Segments and Inertial Properties

| Body Segment                 | Mass ($\text{kg}$) | Principal Inertia $I_{xx}, I_{yy}, I_{zz}$ ($\text{kg}\cdot\text{m}^2$) | Joint Type                                | DOFs                           | Parent          |
| ---------------------------- | ------------------ | ----------------------------------------------------------------------- | ----------------------------------------- | ------------------------------ | --------------- |
| **Pelvis (Root)**            | 11.70              | `0.1337, 0.1337, 0.1337`                                                | Floating / Planar Gimbal                  | 6 (3 Force + 3 Torque)         | World Base      |
| **Lumbar 1**                 | 2.00               | `0.0150, 0.0150, 0.0020`                                                | Universal                                 | 2 (Flexion, Lateral)           | Pelvis          |
| **Lumbar 2**                 | 2.00               | `0.0150, 0.0150, 0.0020`                                                | Universal                                 | 2 (Flexion, Lateral)           | Lumbar 1        |
| **Lumbar 3**                 | 2.00               | `0.0150, 0.0150, 0.0020`                                                | Universal                                 | 2 (Flexion, Lateral)           | Lumbar 2        |
| **Thorax / Torso**           | 18.00              | `0.2200, 0.2000, 0.1000`                                                | Revolute (Yaw)                            | 1 (Axial Rotation)             | Lumbar 3        |
| **Head & Neck**              | 5.00               | `0.0300, 0.0300, 0.0300`                                                | Spherical / Fixed                         | 3                              | Thorax          |
| **Left Clavicle / Scapula**  | 1.50               | `0.0050, 0.0050, 0.0020`                                                | Universal                                 | 2 (Elevation, Protraction)     | Thorax          |
| **Left Upper Arm**           | 2.50               | `0.0180, 0.0180, 0.0040`                                                | Spherical / Gimbal                        | 3 (Elevation, Plane, Rotation) | Left Scapula    |
| **Left Forearm**             | 1.50               | `0.0080, 0.0080, 0.0015`                                                | Revolute (Flexion) + Revolute (Pronation) | 2                              | Left Upper Arm  |
| **Left Hand / Wrist**        | 0.50               | `0.0010, 0.0010, 0.0005`                                                | Universal                                 | 2 (Deviation, Flexion)         | Left Forearm    |
| **Right Clavicle / Scapula** | 1.50               | `0.0050, 0.0050, 0.0020`                                                | Universal                                 | 2 (Elevation, Protraction)     | Thorax          |
| **Right Upper Arm**          | 2.50               | `0.0180, 0.0180, 0.0040`                                                | Spherical / Gimbal                        | 3 (Elevation, Plane, Rotation) | Right Scapula   |
| **Right Forearm**            | 1.50               | `0.0080, 0.0080, 0.0015`                                                | Revolute (Flexion) + Revolute (Pronation) | 2                              | Right Upper Arm |
| **Right Hand / Wrist**       | 0.50               | `0.0010, 0.0010, 0.0005`                                                | Universal                                 | 2 (Deviation, Flexion)         | Right Forearm   |
| **Golf Club (Shaft + Head)** | 0.35               | `0.0450, 0.0450, 0.0002`                                                | Closed Kinematic Loop / Dual Grip         | Fixed to Grip Point            | Hands           |

---

## 3. Unified Actuator & Torque Representation

All engines consume identical continuous polynomial torque profiles defined by 7 parameters per channel (6th-order Bernstein basis across 27 canonical actuation channels = 189 parameters):

$$\tau_j(t) = \sum_{k=0}^6 c_{j,k} B_{k,6}\left(\frac{t}{T}\right), \quad B_{k,6}(s) = \binom{6}{k} s^k (1-s)^{6-k}$$

### Canonical Coordinate Order (27 Channels)

1. `TranslationInputX` (Pelvis world force X, N)
2. `TranslationInputY` (Pelvis world force Y, N)
3. `TranslationInputZ` (Pelvis world force Z, N)
4. `HipInputX` (Pelvis tilt torque, N·m)
5. `HipInputY` (Pelvis list torque, N·m)
6. `HipInputZ` (Pelvis rotation torque, N·m)
7. `SpineInputX` (Lumbar lateral bending, N·m)
8. `SpineInputY` (Lumbar flexion/extension, N·m)
9. `TorsoInput` (Thorax axial rotation, N·m)
10. `LEInput` (Left elbow flexion, N·m)
11. `LFInput` (Left forearm pronation/supination, N·m)
12. `LScapInputX` (Left scapula elevation/depression, N·m)
13. `LScapInputY` (Left scapula protraction/retraction, N·m)
14. `LSInputX` (Left shoulder horizontal abduction, N·m)
15. `LSInputY` (Left shoulder plane of elevation, N·m)
16. `LSInputZ` (Left shoulder internal/external rotation, N·m)
17. `LWInputX` (Left wrist radial/ulnar deviation, N·m)
18. `LWInputY` (Left wrist flexion/extension, N·m)
19. `REInput` (Right elbow flexion, N·m)
20. `RFInput` (Right forearm pronation/supination, N·m)
21. `RScapInputX` (Right scapula elevation/depression, N·m)
22. `RScapInputY` (Right scapula protraction/retraction, N·m)
23. `RSInputX` (Right shoulder horizontal abduction, N·m)
24. `RSInputY` (Right shoulder plane of elevation, N·m)
25. `RSInputZ` (Right shoulder internal/external rotation, N·m)
26. `RWInputX` (Right wrist radial/ulnar deviation, N·m)
27. `RWInputY` (Right wrist flexion/extension, N·m)

---

## 4. URDF & MJCF Generation Pipeline

### 4.1 Export Architecture

1. **Canonical YAML (`golfer_canonical.yaml`)** $\to$ `tools/model_converter/build_models.py`
2. **Output Artefacts**:
   - `golfer.urdf`: Consumed by **Pinocchio** and **Drake**.
   - `golfer.xml`: Consumed by **MuJoCo**.
   - `Simscape_Multibody_Models/3D_Golf_Model/matlab/init/InitModelParameters.m`: Consumed by **Simscape**.

### 4.2 Joint Mapping Contracts

Each engine translates the 27 generalized coordinates into its native joint indexing using a verified schema map:

```python
@dataclass(frozen=True)
class EngineJointMap:
    coordinate_names: tuple[str, ...]  # Canonical 27 names
    engine_dof_indices: tuple[int, ...]  # Native DOF index for each canonical coordinate
    sign_flips: tuple[float, ...]  # +1.0 or -1.0 depending on coordinate frame definition
```

---

## 5. Cross-Engine Verification & Parity Harness

### 5.1 Parity Benchmark Test (`test_cross_engine_parity.py`)

1. **Fixed Benchmark Trajectory**: A certified 6th-order continuous polynomial $\theta^* \in \mathbb{R}^{27 \times 7}$ qualified on Simscape R2025b.
2. **Initial Condition**: Certified initial state $S_0 = (q_0, \dot{q}_0)$.
3. **Execution**:
   - Run Simscape Multibody $\to$ $X_{\text{simscape}}(t)$
   - Run MuJoCo $\to$ $X_{\text{mujoco}}(t)$
   - Run Pinocchio $\to$ $X_{\text{pinocchio}}(t)$
   - Run Drake $\to$ $X_{\text{drake}}(t)$
4. **Acceptance Criteria**:
   - Grip point RMSE vs Simscape: $< 5.0\text{ mm}$
   - Clubhead point RMSE vs Simscape: $< 10.0\text{ mm}$
   - Joint angle RMSE vs Simscape: $< 0.05\text{ rad}$ ($< 2.8^\circ$)
   - Energy Conservation / Power balance: $\int \tau \cdot \dot{q} \, dt \approx \Delta T + \Delta V$ within $1.0\%$.

---

## 6. Work Packages & Fleet Distribution

- **WP1: Unified URDF / MJCF Generator & Parity Sync** (`UpstreamDrift`, `Pinocchio_Models`)
  - Ensure `golfer.urdf` and `golfer.xml` have identical masses, centers of mass, and inertia tensors matching `GolfSwing3D_Kinetic.slx`.
- **WP2: Pinocchio Forward-Dynamics Torque Simulator** (`Pinocchio_Models`, `UpstreamDrift`)
  - Implement `simulate_with_coefficients` in `src/engines/physics_engines/pinocchio/python/`.
- **WP3: MuJoCo Forward-Dynamics Torque Simulator** (`MuJoCo_Models`, `UpstreamDrift`)
  - Implement `simulate_with_coefficients` in `src/engines/physics_engines/mujoco/python/`.
- **WP4: Drake MultibodyPlant Simulator** (`Drake_Models`, `UpstreamDrift`)
  - Implement `simulate_with_coefficients` in `src/engines/physics_engines/drake/python/`.
- **WP5: Automated Cross-Engine Parity CI / CD Gate** (`Repository_Management`)
  - Nightly test asserting identical forward dynamics across all 4 engines.

---

## 7. Revision History & Implementation Changelog

| Date       | Issue / PR | Author / Engine             | Description                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| :--------- | :--------- | :-------------------------- | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-09-11 | #9966      | MuJoCo / `UpstreamDrift`    | **WP2**: Implement continuous-torque forward-dynamics simulator harness in `src/engines/physics_engines/mujoco/python/`. Added support for 6th-order Bernstein basis and power basis torque drivers, energy accounting (`kinetic_energy`, `potential_energy`), canonical site extraction (`mid_hands`, `clubhead`), `EngineJointMap` actuator mapping, and contract tests under `tests/unit/engines/mujoco/`.                                                                                                           |
| 2026-09-11 | #9967      | Pinocchio / `UpstreamDrift` | **WP3**: Implement continuous-torque forward-dynamics simulator harness in `src/engines/physics_engines/pinocchio/python/`. Added support for 6th-order continuous Bernstein polynomial torques via Pinocchio forward dynamics (ABA and RK4 forward integration), energy accounting (`kinetic_energy`, `potential_energy`), standard `SimOut` dataclass matching Simscape contract, canonical `EngineJointMap` DOF mapper, and unit/contract test suite under `tests/unit/engines/pinocchio/`.                          |
| 2026-09-11 | #9968      | Drake / `UpstreamDrift`     | **WP4**: Implement continuous-torque forward-dynamics simulator harness in `src/engines/physics_engines/drake/python/`. Added support for 6th-order continuous Bernstein polynomial torques via Drake `MultibodyPlant` and `DiagramBuilder`, energy accounting (`kinetic_energy`, `potential_energy`), canonical `SimOut` matching Simscape contract, canonical `EngineJointMap` actuator mapper, and unit/contract test suite under `tests/unit/engines/drake/`.                                                       |
| 2026-09-11 | #9969      | Fleet / `UpstreamDrift`     | **WP5**: Implement automated cross-engine forward-dynamics parity benchmark suite and tolerance gate in `tests/cross_engine/test_four_engine_parity.py`. Asserts that Simscape baseline vs MuJoCo, Pinocchio, and Drake produce identical forward-dynamics trajectories given identical polynomial torque profile $\theta^*$ within $< 5.0\text{ mm}$ grip RMSE and $< 10.0\text{ mm}$ clubhead RMSE. Dynamically guards native runtimes (`is_mujoco_available()`, `is_pinocchio_available()`, `is_drake_available()`). |
