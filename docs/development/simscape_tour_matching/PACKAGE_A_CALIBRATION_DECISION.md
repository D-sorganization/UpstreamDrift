# Package A: Matching Diagnosis and Calibration Decision Report

**Issue Context:** #9921 / Epic #10003 / #9967  
**Date:** 2026-09-17  
**Agent:** Antigravity (Pair Programming Session `9efc50d9-1e1a-4d78-a883-cf8c5e9f7a59`)  
**Worktree:** `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-pinocchio-native`  
**Branch:** `feat/10285-native-saved-replay` (PR #10287)  
**Deliverable:** Bounded diagnostic report resolving Steps 1–5 of `CHEAPER_AGENT_WORK_PACKAGES.md`.

---

## Executive Summary and Binding Decision

1. **Fixed-Offset Calibration Evaluation (Offset-Only, Unchanged Topology):**

   - On the unchanged baseline Simscape model topology (where `Head` and `Back` markers share the single rigid `Hub` body), re-optimizing fixed 3D marker attachments across the training interval ($t \in [0.0, 0.85]\text{ s}$) yields:
     - **Training Aggregate RMS** ($0.0 \le t \le 0.85\text{ s}$, all 25 markers): drops by $2.62\text{ mm}$ (from $10.28\text{ mm}$ to $7.66\text{ mm}$).
     - **Held-Out Validation Aggregate RMS** ($0.85 < t \le 1.814\text{ s}$, all 25 markers): increases slightly by $+0.18\text{ mm}$ (from $21.92\text{ mm}$ to $22.09\text{ mm}$).
     - **Full-Capture Aggregate RMS** (all 25 markers): slight reduction of $0.53\text{ mm}$ (from $17.39\text{ mm}$ to $16.86\text{ mm}$).
     - **Hub Cluster RMS Alone** (6 markers: `BackTop`, `BackLeft`, `BackRight`, `HeadTop`, `HeadFront`, `HeadSide`): held-out error remains at $43.65\text{ mm}$ (vs original $43.07\text{ mm}$).
   - **Conclusion:** Parameter adjustment of fixed marker offsets alone under the single-Hub assumption does not resolve the held-out validation discrepancy. However, this offset-only trial does not evaluate segment-length calibration (e.g. torso length, shoulder span) or articulated solutions.

2. **Cluster Deformation vs. Independent-Body Lower Bounds:**

   - In optical capture `driver_marker_payload_9967.json`, the head and back markers deform relative to each other by up to **$109.6\text{ mm}$** across the swing:
     - `BackLeft`–`HeadSide`: model distance $= 421.8\text{ mm}$; observed range $= 349.7\text{ mm}$ to $459.3\text{ mm}$ ($\Delta = 109.6\text{ mm}$).
     - `BackRight`–`HeadSide`: model distance $= 464.0\text{ mm}$; observed range $= 395.7\text{ mm}$ to $464.6\text{ mm}$ ($\Delta = 68.9\text{ mm}$).
     - `BackTop`–`HeadSide`: model distance $= 269.3\text{ mm}$; observed range $= 202.3\text{ mm}$ to $293.1\text{ mm}$ ($\Delta = 90.8\text{ mm}$).
     - `BackLeft`–`HeadFront`: model distance $= 514.0\text{ mm}$; observed range $= 424.9\text{ mm}$ to $522.5\text{ mm}$ ($\Delta = 97.6\text{ mm}$).
   - In any rigid body, pairwise distance between attached points is an invariant under proper $\mathrm{SE}(3)$ motions.
   - When Head and Back are evaluated as **independent rigid bodies (unconstrained 6D rigid relaxation)** without kinematic joint or weld constraints:
     - **Head cluster independent lower bound** (`HeadTop`, `HeadFront`, `HeadSide`): **$0.16\text{ mm}$ mean RMS** across all 654 frames (max $0.55\text{ mm}$).
     - **Back cluster independent lower bound** (`BackTop`, `BackLeft`, `BackRight`): **$5.05\text{ mm}$ mean RMS** across all 654 frames (max $11.36\text{ mm}$).
     - **Total unconstrained rigid lower bound** across all 10 bodies drops from **$14.59\text{ mm}$ to $3.72\text{ mm}$**.
   - These independent-body lower bounds are theoretical geometric limits under complete mechanical disconnection; they do not certify that a connected, kinematically articulated model can achieve $3.72\text{ mm}$.

3. **Analysis of Flagged 1.25–1.55 s Local Articulated Solves:**

   - The $84.03\text{ mm}$ aggregate error spike at $1.25\text{ s}$ (Frame 450) was caused by **missing target data interacting with the penalty formulation and numerical search bounds**, not an intrinsic kinematic limitation:
     - At $1.25\text{ s}$, `WaistRight` is unobserved (`valid = False`, target coordinates `[0.0, 0.0, 0.0]`).
     - In the earlier diagnostic search formulation, the target yaw vector was computed between `WaistRight` and `WaistLeft`. When `WaistRight` defaulted to origin, the artificial target yaw vector pointed across the room ($172.3^\circ$), while the physical pelvis yaw was $\approx -40^\circ$.
     - Solves 0–5 enforced this spurious yaw constraint, forcing `TorsoInput` and `HipInputZ` to pin against their local numerical bounds ($\pm 1.0\text{ rad}$ and $\pm 0.25\text{ m}$).
     - When evaluated without this corrupted yaw constraint (Solves 6 and 7), the local articulated solution achieved **$31.08\text{ mm}$ aggregate RMS with 0 active bounds**.
     - Solves at $1.30\text{ s}$ ($32.09\text{ mm}$), $1.40\text{ s}$ ($35.50\text{ mm}$), and $1.45\text{ s}$ ($38.40\text{ mm}$) similarly reflect local bound encounters in the sampled continuation rather than dynamic failure.

4. **Recommendations & Next Technical Steps:**
   - **Do NOT conclude that an offset-only experiment rules out calibration:** Evaluate permitted segment-length adjustments alongside marker placements under held-out validation before modifying model topology.
   - **Preserve baseline Simscape reference model:** Do not make unauthorized modifications to the reference Simscape model.
   - **Hypothesis Qualification:**
     - Quantify whether a 3-DOF rotational articulation (cervical spine / neck joint) with a fixed joint center explains the relative Head/Back motion across the swing before introducing topology changes.
     - Prepare any candidate model as a separately versioned proposal with rigorous mass/inertia consistency.
     - Advance toward a reproducible, independently validated 0.90 s candidate.

---

## Detailed Audit Findings

### 1. Head/Back Cluster Decomposition and Rigidity Analysis

Across all 654 frames ($1.81389\text{ s}$ @ 360 Hz):

| Subsystem / Cluster                         | Markers Included                                           |      Baseline Combined Rigid Lower Bound      |     Split Independent Rigid Lower Bound      | Lower Bound Reduction  |
| :------------------------------------------ | :--------------------------------------------------------- | :-------------------------------------------: | :------------------------------------------: | :--------------------: |
| **Hub Combined Cluster**                    | BackTop, BackLeft, BackRight, HeadTop, HeadFront, HeadSide | **$28.72\text{ mm}$** (max $55.83\text{ mm}$) |                      —                       |           —            |
| **Back Cluster Alone**                      | BackTop, BackLeft, BackRight                               |                       —                       | **$5.05\text{ mm}$** (max $11.36\text{ mm}$) | **$-23.67\text{ mm}$** |
| **Head Cluster Alone**                      | HeadTop, HeadFront, HeadSide                               |                       —                       | **$0.16\text{ mm}$** (max $0.55\text{ mm}$)  | **$-28.56\text{ mm}$** |
| **Full Capture Aggregate (All 25 Markers)** | All 10 bodies / 25 markers                                 | **$14.59\text{ mm}$** (max $27.97\text{ mm}$) | **$3.72\text{ mm}$** (max $9.18\text{ mm}$)  | **$-10.87\text{ mm}$** |

Key frame progression (Cluster RMS on Hub vs Split Lower Bounds):

- At Address ($t = 0.000\text{ s}$): Hub Combined $= 0.00\text{ mm}$, Back $= 0.00\text{ mm}$, Head $= 0.00\text{ mm}$.
- At Mid-Backswing ($t = 0.600\text{ s}$): Hub Combined $= 21.82\text{ mm}$, Back $= 2.87\text{ mm}$, Head $= 0.16\text{ mm}$.
- At Top of Backswing / Transition ($t = 0.850\text{ s}$): Hub Combined $= 47.66\text{ mm}$, Back $= 7.18\text{ mm}$, Head $= 0.07\text{ mm}$.
- At Delivery ($t = 1.150\text{ s}$): Hub Combined $= 49.88\text{ mm}$, Back $= 9.44\text{ mm}$, Head $= 0.08\text{ mm}$.
- At Impact ($t = 1.233\text{ s}$): Hub Combined $= 26.41\text{ mm}$, Back $= 0.82\text{ mm}$, Head $= 0.03\text{ mm}$.

**Interpretation:** During the backswing and transition, the golfer's cervical spine flexes, rotates, and tilts relative to the thoracic spine. Because `Hub` holds both clusters rigidly, any rotation of the torso to fit the back markers drags the head markers away from their observed optical targets by $50\text{–}80\text{ mm}$.

### 2. Audit of Flagged 1.25–1.55 s Local Articulated Solves

| Time (s) | Frame | Valid Markers |  Missing Markers  | Solve Type         | Aggregate Error (RMS) | Active Bounds | Note / Root Cause                                              |
| :------: | :---: | :-----------: | :---------------: | :----------------- | :-------------------: | :-----------: | :------------------------------------------------------------- |
|  1.200   |  432  |    25 / 25    |       None        | Yaw-constrained    |   $40.08\text{ mm}$   |       0       | Normal baseline trajectory                                     |
|  1.250   |  450  |    24 / 25    |   `WaistRight`    | Solve 0 (Yaw 0%)   |   $84.03\text{ mm}$   |       2       | `HipInputZ`, `TorsoInput` at bound due to corrupted target yaw |
|  1.250   |  450  |    24 / 25    |   `WaistRight`    | Solve 6 (Yaw Free) | **$31.08\text{ mm}$** |     **0**     | **Relieved bounds: error drops below normal trajectory**       |
|  1.300   |  468  |    25 / 25    |       None        | Best passing       |   $32.09\text{ mm}$   |       1       | 1 active coordinate bound ($\Delta q = 1.0\text{ rad}$)        |
|  1.350   |  486  |    25 / 25    |       None        | Best passing       |   $40.25\text{ mm}$   |       3       | Continuation step size limit                                   |
|  1.400   |  504  |    25 / 25    |       None        | Best passing       |   $35.50\text{ mm}$   |       2       | Rapid post-impact recoil                                       |
|  1.450   |  522  |    22 / 25    | 3 markers missing | Best passing       |   $38.40\text{ mm}$   |       1       | Occlusions during follow-through                               |
|  1.550   |  558  |    22 / 25    | 3 markers missing | Best passing       |   $40.06\text{ mm}$   |       1       | Bounded follow-through posture                                 |

**Conclusion on Step 2:** The $84.03\text{ mm}$ spike at $1.25\text{ s}$ is an artifact of the marker dropout interacting with the pelvis yaw penalty and local numerical box width ($\pm 0.25\text{ m}, \pm 1.0\text{ rad}$). When evaluated without the corrupted yaw constraint, the kinematic fit achieves $31.08\text{ mm}$ with 0 active bounds.

---

## Verification & Test Additions

In accordance with TDD, DbC, and LoD:

1. `tests/motion_matching/test_rigidity.py`:
   - Added `test_cluster_separation_preserves_internal_rigidity()`: Verifies that decomposing coupled clusters into independent rigid bodies diagnoses relative deformation while preserving near-zero internal residual floors.
2. `tests/unit/motion_matching/test_marker_calibration.py`:
   - Added `test_alternating_calibration_held_out_validation()`: Verifies that alternating calibration on training frames generalizes to held-out validation frames under true rigid conditions, ensuring that training improvements do not mask held-out overfitting or non-rigid deformation.
3. Quality Checks:
   - `python -m pytest tests/motion_matching/test_rigidity.py tests/unit/motion_matching/test_marker_calibration.py tests/unit/motion_matching/test_constrained_marker_pose.py -q --no-cov`: **17 passed**.
   - `python -m ruff check tests/motion_matching/test_rigidity.py tests/unit/motion_matching/test_marker_calibration.py`: **Clean (0 errors)**.
   - `python -m ruff format --check tests/motion_matching/test_rigidity.py tests/unit/motion_matching/test_marker_calibration.py`: **Clean (0 diffs)**.
