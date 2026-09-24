# Tour Baselines User Guide

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584) (TB-12, [#10597](https://github.com/D-sorganization/UpstreamDrift/issues/10597))  
Governing Program: Matched Swing Program ([#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363))  
Prerequisites: TB-00 through TB-11

---

## 1. Overview and Problem Statement

The **Tour Baselines Program** provides validated, reproducible reference swing baselines across the UpstreamDrift biomechanics fleet. Historically, motion-matching models suffered from ambiguity regarding:

1. **Reconstruction vs. Dynamic Models:** Pure kinematic reconstructions tracked anatomical joint markers while ignoring the physical golf club and swing forces, yielding deceptively low tracking residuals.
2. **Disparate Observation Sets:** Different models evaluated varying subsets of markers, making error metrics non-comparable.
3. **Implicit vs. Explicit Solver Replays:** Runs were reported without verifiable cryptographic receipts, parameter bounds, or deterministic random seeds.

The Tour Baselines architecture addresses these issues by standardizing:

- Authoritative tour captures for Driver (360 Hz) and 7-Iron (359 Hz).
- A unified 10-model coverage matrix with fail-closed evidence statuses.
- Standardized 3D Euclidean physical error formulas.
- Portable, self-contained baseline packages (`tour-baseline-package/1.0.0`) with tamper verification.
- Seamless graphical and headless navigation within the Motion Matching launcher.

---

## 2. Launcher Navigation and Workflow

The Tour Baselines workflow is integrated directly into the UpstreamDrift launcher ecosystem:

```text
Launcher -> Motion Matching -> Tour Baselines Tab -> Capture Selection -> Model Roster -> Evidence / Open / Compare / Clone
```

### Step-by-Step Operator Flow:

1. **Launch Motion Matching:**
   Open the unified Motion Matching interface via GUI or CLI:
   ```bash
   python -m src.tools.motion_matching.gui
   ```
2. **Navigate to Tour Baselines Tab:**
   Select the **Tour Baselines** tab to access the coverage matrix, model roster, and evidence inspection panels.
3. **Select Capture:**
   Choose between the **Driver (360 Hz)** or **7-Iron (359 Hz)** tour datasets. Selecting a capture instantly updates the roster table, displaying the evidence status, physical RMSE, and availability for each registered model.
4. **Inspect Model Roster:**
   The roster presents models grouped by complexity (Planar Pendulums, Reconstruction, Full Body). Badges clearly indicate status:
   - `[QUALIFIED]` / `[G1 PASSED]` / `[CANDIDATE]` — Validated baseline package available.
   - `[DISQUALIFIED]` / `[BLOCKED]` — Planarity or dynamics gate failure with documented rationale.
   - `[UNAVAILABLE]` — Missing required optional engine adapter or environment.
5. **Inspect Evidence:**
   Click **Inspect Evidence** to view the underlying cryptographic audit receipt, including git commit SHA, engine version, solver convergence status, and physical fit breakdown.
6. **Open Baseline:**
   Click **Open Baseline** to load the selected baseline package into the viewport. The UI applies strict visual distinction semantics between measured mocap markers and simulated physics graphics.
7. **Compare Models:**
   Select two models and click **Compare Models** to generate an automated differential report displaying RMSE delta, phase-by-phase differences, and topological contrasts.
8. **Clone for Experimentation:**
   Click **Clone for Experiment** to export a copy of the model preset into your local session folder (`sessions/`), ensuring the authoritative baseline package remains immutable.

---

## 3. Capture Selection: Driver Versus 7-Iron

Tour baselines are anchored to two distinct professional tour motion-capture datasets:

| Characteristic             | Driver Capture                       | 7-Iron Capture                            |
| -------------------------- | ------------------------------------ | ----------------------------------------- |
| **Capture Frequency**      | 360 Hz                               | 359 Hz                                    |
| **Duration / Frames**      | 654 frames (~1.82 s)                 | 654 frames (~1.82 s)                      |
| **Active Dynamic Horizon** | 0.0 s to 0.85 s (address to release) | 0.0 s to 0.85 s (address to release)      |
| **Clubhead Speed**         | ~48 m/s (~107 mph)                   | ~38 m/s (~85 mph)                         |
| **Attack Angle**           | Positive (upward sweep, +2.1 deg)    | Negative (downward compression, -4.3 deg) |
| **Primary Plane**          | Flatter swing plane (~45 deg)        | Steeper swing plane (~58 deg)             |
| **Dynamic Effects**        | Substantial shaft lead deflection    | High ground impact loading & shaft droop  |

Operators must evaluate models on both captures. Passing Driver alone does not grant G3 professional release.

---

## 4. Model Complexity Tradeoffs and Hierarchy

The baseline roster spans four distinct tiers of biomechanical fidelity:

```text
Tier 1: Reduced Analytical & Driven Pendulums (2-DoF / 3-DoF)
   │   └─ High speed, planar projection, rapid fitting, geometric out-of-plane residual.
   ▼
Tier 2: Constrained Upper-Body Kinematics (6 Markers)
   │   └─ Multi-joint shoulder-to-wrist kinematics; rejected under 3D tour swing planarity gate.
   ▼
Tier 3: Rigid Full-Body Articulated Systems (Pinocchio, Simscape, Drake, MuJoCo)
   │   └─ 38 markers + dual GRF plates; 3D kinematics and forward/inverse dynamics.
   ▼
Tier 4: Musculoskeletal & Neural Actuation (OpenSim, MyoSuite)
       └─ Hill-type muscles, joint moments, neural control policies.
```

### Model Tradeoff Comparison:

- **`driven_double_pendulum` (2 DoF):**
  - _Strengths:_ Solves in < 2 seconds; deterministic torque-driven ODE; perfect for teaching and rapid sensitivity scans.
  - _Tradeoffs:_ Constrained to a single plane; neglects torso rotation, pelvis sway, and shaft flexure. Out-of-plane residual ~2.0 mm.
- **`driven_triple_pendulum` (3 DoF):**
  - _Strengths:_ Introduces forearm pronation/supination; lower clubhead RMSE compared to double pendulum.
  - _Tradeoffs:_ Planar constraint remains active; requires bounded optimization to avoid torque chatter.
- **`constrained_upper_body_golfer` (Upper Body):**
  - _Status:_ Disqualified for tour baselines ([#10591](https://github.com/D-sorganization/UpstreamDrift/issues/10591)).
  - _Rationale:_ Forcing 3D shoulder/elbow/wrist kinematics into a single 2D plane produces severe non-physical marker distortions (> 45 mm RMSE).
- **`full_body_pinocchio` / `full_body_mujoco` (Full Body):**
  - _Strengths:_ Full 38-marker anatomical coverage; tracks ground reaction forces; true 3D spatial dynamics.
  - _Tradeoffs:_ Compute-intensive (minutes to hours per fit); requires careful contact and numerical damping tuning.

---

## 5. Baseline Package Loading, Replay, and Session Cloning

### Portable Package Contract (`tour-baseline-package/1.0.0`)

A Tour Baseline Package is a deterministic ZIP archive containing:

- `manifest.json`: Full `BaselineIdentity`, cryptographic checksums, and runtime metadata.
- `bundle.json`: 5-status bundle (`SolverConvergence`, `KinematicAccuracy`, `DynamicFeasibility`, `ScientificQualification`, `ProductPromotion`).
- `metrics.json`: Standardized `PhysicalFitMetrics`.
- `trajectories.npz`: Complete timeseries arrays (`time`, `q`, `v`, `tau`, `marker_pred`, `marker_obs`).
- `parameters.json`: Calibrated model parameters and bounds.

### Strict Visual Distinction Semantics

When rendering a baseline in the viewport, the application enforces visual separation:

- **Observed Club:** Displayed as discrete point spheres (`marker_points`) representing actual optical C3D marker observations.
- **Simulated Club:** Displayed as a continuous geometric shaft and mesh (`continuous_mesh`) representing model forward kinematics.
- Under no circumstance are measured dots connected into a simulated club polygon without explicit model solving.

### Session Cloning Isolation

To prevent corrupting golden baseline packages during exploratory fitting:

```python
presenter.clone_for_experiment(
    model_id="driven_double_pendulum",
    capture="driver",
    session_dir=Path("sessions/experiment_01"),
    experiment_name="double_pendulum_modified_inertia",
)
```

This generates a standalone preset in `sessions/experiment_01/` with a unique ID and copies all initial conditions, leaving the baseline package untouched.

---

## 6. Metric Interpretation and Physical Error Formulas

Tour Baselines forbid arbitrary loss functions or unweighted coordinate combinations for acceptance reporting. All published metrics adhere to the TB-02 ([#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587)) physical error specification:

### 1. Physical 3D Euclidean Marker RMSE

$$\text{RMSE}_{\text{whole}} = \sqrt{\frac{1}{N_{\text{valid}}} \sum_{i \in \text{valid}} \| \mathbf{p}_{\text{pred}, i} - \mathbf{p}_{\text{obs}, i} \|_2^2}$$

### 2. Separation of Physical RMSE From Optimizer Loss

- **Physical RMSE:** Expressed in absolute millimeters (mm) across valid physical marker positions.
- **Optimizer Loss:** Dimensionless regularized objective with penalty terms. The optimizer loss is never quoted as physical tracking accuracy.

### 3. Percentile and Maximum Errors

- **p95 Error:** 95th percentile Euclidean error across all valid observations, exposing localized transient errors.
- **Max Error:** Absolute worst-case residual over the dynamic horizon.

### 4. Phase-Specific Error Breakdown

Metrics are decomposed into distinct swing phases:

- **Address:** Static stance calibration ($\le 6.0$ mm threshold).
- **Backswing & Transition:** Peak loading and wrist hinge.
- **Downswing:** Rapid club acceleration.
- **Impact:** Critical ball contact window ($\pm 10$ ms).
- **Follow-Through:** Post-impact deceleration.

---

## 7. Cryptographic Provenance and Audit Trails

Every baseline package embeds complete cryptographic hashes:

- `capture_sha256`: SHA-256 of the authoritative C3D target file.
- `runtime_hashes.git_commit`: Pinned git commit of the fitting run.
- `runtime_hashes.engine_version`: Exact solver library version.
- `fixed_geometry_hash`: Cryptographic digest of segment lengths and marker offsets.
- `fixed_inertia_hash`: Cryptographic digest of body mass and inertia matrices.
- `controls_hash`: SHA-256 of applied torque/actuation timeseries.

Any modification to trajectory arrays, parameters, or manifest fields invalidates the package checksum, triggering a fail-closed import error.
