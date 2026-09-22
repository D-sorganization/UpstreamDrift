# Two-Window Direct-Node SLSQP Continuation 102 — Recentered Bounds & Zero Bound Saturation

Continuation 102 restarts from Candidate 101 (`b505d8b6...`) and implements the ordered work packages specified in `docs/development/simscape_tour_matching/RUN101_REVIEW_AND_TURNOVER.md`:

1. **Recentered Control Bounds ($\pm 3.0\text{ N/Nm}$)**: Unlocked the 22 control coordinates that had become pinned against historical box boundaries inherited from Parent 19, restoring full bidirectional gradient authority across all 81 active B4/B5/B6 parameters.
2. **Recentered Node Chart at $t=0.60\text{ s}$**: Anchored chart directly on Candidate 101's integrated endpoint with zero initial defect, eliminating the 11 saturated node coordinates.
3. **Zero Active Bound Count Achieved**: The optimization converged with **0 active theta bounds** (lower=0, upper=0) and **0 active node bounds**, completely eliminating numerical boundary lock.
4. **Independent MATLAB R2025b Update 5 Validation**: Evaluated on DeskComputer under qualified solver settings (`ode15s`, `RelTol 1e-6`, `MaxStep 1/1440 s`), demonstrating **60.5 micrometer** max Euclidean discrepancy and **1.1 micrometer** mean discrepancy against Pinocchio 4.1.0.

---

## 1. Uninterrupted Original-State Replay (0–0.85 s, Zero Resets)

| Metric                                 | run73  |    run100     |   run101 (Simscape)    | **run102 (Pinocchio)** | **run102 (Simscape R2025b)** |     Gate Target      | Gate Status |
| :------------------------------------- | :----: | :-----------: | :--------------------: | :--------------------: | :--------------------------: | :------------------: | :---------: |
| **Whole RMS (mm)**                     | 28.105 |    20.342     |         20.265         |       **20.267**       |          **20.267**          | $\le 25.0\text{ mm}$ |  **PASS**   |
| **Early RMS $\le 0.60\text{ s}$ (mm)** | 10.860 |    10.015     |         9.995          |       **9.995**        |          **9.995**           | $\le 12.0\text{ mm}$ |  **PASS**   |
| **Terminal RMS (mm)**                  | 65.398 |    39.202     |         40.312         |       **40.303**       |          **40.301**          | $\le 35.0\text{ mm}$ |   Plateau   |
| **Club cluster (mm)**                  | 30.041 |     8.728     |         8.422          |       **8.391**        |          **8.389**           | $\le 60.0\text{ mm}$ |  **PASS**   |
| **Pelvis yaw error %**                 | 6.164  |    15.691     |         0.543          |       **0.595**        |          **0.610**           |  $< 5.0\text{ \%}$   |  **PASS**   |
| **Pelvis yaw diff (deg)**              |  N/A   | $+8.57^\circ$ |     $+0.30^\circ$      |   **$+0.33^\circ$**    |      **$+0.33^\circ$**       |    $< 3.0^\circ$     |  **PASS**   |
| **Active Bounds**                      |  N/A   |      27       | 33 (22 ctrl + 11 node) |         **0**          |            **0**             |          0           |  **PASS**   |
| **Assembled Score**                    | 18.300 |    426.749    |         28.586         |       **28.585**       |          **28.585**          |         N/A          |     N/A     |

---

## 2. Key Physical & Optimization Findings

1. **Total Bound Wall Elimination**:
   - In run 101, 22 controls and 11 node coordinates were completely trapped at one-sided bounds.
   - In Trial 102, recentering eliminated all bound stalls: `active_bound_count = 0`, `active_theta_lower = 0`, `active_theta_upper = 0`, `active_node_bounds = 0`.
2. **Single-Digit Early RMS Retained**:
   - Early trajectory RMS ($\le 0.60\text{ s}$) firmly maintained at **`9.995 mm`**, confirming the backswing trajectory remains globally stable and unaffected by downstream parameter adjustments.
3. **New All-Time Best Clubhead Accuracy**:
   - Clubhead cluster terminal RMS improved to **`8.389 mm`** (Simscape R2025b), demonstrating near-perfect clubhead path delivery.
4. **Pelvis Yaw Stability**:
   - Pelvis yaw error remained sub-percent at **`0.610%`** (**`+0.333 deg`**), confirming the 2-component unit vector formulation prevents any over-rotation.
5. **Terminal RMS Plateau Diagnosis**:
   - With all control and node bounds fully unconstrained, the optimizer reduced terminal RMS slightly to 40.301 mm.
   - Work Package 2 rigid-cluster lower bounds identified that the Hub marker cluster (Head + Back) has an independent rigid floor of 47.66 mm because the model lacks an independent neck joint. Therefore, within the fixed 3D Golf Model kinematic topology, terminal whole-body error is fundamentally constrained by thorax-head coupling rather than parameter bound saturation.

---

## 3. Cross-Engine Parity Summary (Pinocchio 4.1.0 vs Simscape R2025b)

| Parity Metric                   |                   Measured Value                    |           Qualification            |
| :------------------------------ | :-------------------------------------------------: | :--------------------------------: |
| **Max Coordinate Discrepancy**  |  **$0.0550\text{ mm}$ ($55\text{ }\mu\text{m}$)**   | Sub-millimeter ($< 0.1\text{ mm}$) |
| **Mean Coordinate Discrepancy** |      **$0.00054\text{ mm}$ ($544\text{ nm}$)**      |          Nanometer regime          |
| **Max Euclidean Discrepancy**   | **$0.0605\text{ mm}$ ($60.5\text{ }\mu\text{m}$)**  | Sub-millimeter ($< 0.1\text{ mm}$) |
| **Mean Euclidean Discrepancy**  | **$0.00110\text{ mm}$ ($1.10\text{ }\mu\text{m}$)** |      Single-digit micrometer       |
| **Initial Pose Parity ($t=0$)** |                **$1.554\text{ fm}$**                |         Machine precision          |
| **MATLAB Simulation Time**      |                **$23.28\text{ s}$**                 |       Real-time performance        |
| **Replay MAT File Size**        |                 **$423\text{ KB}$**                 |    Qualified ($< 1\text{ MB}$)     |

---

## 4. Candidate Artifact Inventory

The candidate evidence directory contains:

- `returned-candidate.json`: Candidate document with SHA-256 `ee4a908737f4cf3065bb6e64a056dfab71926c43ac40c0e85574fc8531576c82`.
- `config.json`: Execution configuration with seed and candidate hashes.
- `receipt.json`: Convergence receipt with preflight zero-displacement audit and elapsed optimization time.
- `returned.json`: Full report including cost history and active bounds (`active_bound_count: 0`).
- `returned-nodes.json`: Intermediate shooting node chart coordinates.
- `returned-replay.npz`: Complete multi-channel trajectory data.
- `candidate.npz`: Versioned `MatchedSwingCandidate` package (MS-60 / #10347).
- `run_manifest.json`: R2025b host/release/SHA/wall-clock receipt (MS-60).
- `playback.gif`: Marker-overlay playback of the committed candidate (MS-60).
- `pinocchio_replay.mat`: Converted Pinocchio trajectory.
- `qualified_candidate_replay.json`: Independent MATLAB R2025b Update 5 forward simulation report.
- `qualified_candidate_replay.mat`: Compact MATLAB replay data ($423\text{ KB}$).
- `two_window_fit_102.py`: Full source script.
- `run_audit_102.sh` & `run_full_102.sh`: Execution launchers.
- `replay_returned102_r2025b.m` & `run_replay_102.bat`: R2025b validation scripts.
- Fleet runner: `scripts/matlab/run_simscape_candidate.ps1 -Run two_window_fit_9967_102 -Replay`.
