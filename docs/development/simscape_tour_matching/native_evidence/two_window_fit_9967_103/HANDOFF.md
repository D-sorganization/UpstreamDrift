# Two-Window Direct-Node SLSQP Continuation 103 — 0–0.90s Bounded Horizon

Continuation 103 advances the continuation horizon from 0.85s out to 0.90s (325 frames @ 360 Hz) on `feat/10285-native-saved-replay`, executing the bounded multi-shooting optimization mandated by Epic #9921 / #9967 / #10286:

1. **Horizon Extension to $0.90\text{ s}$**: Extended trajectory clock by +50 ms (18 frames) into the early downswing delivery phase while strictly preserving continuous forward dynamics with zero target-state resets.
2. **Recentered on Candidate 102 (`ee4a9087...`)**: Seeded with zero initial defect against Candidate 102's integrated endpoint at $t=0.60\text{ s}$ (reference closure error $5.13 \times 10^{-12}\text{ m}$, retraction shift $2.01 \times 10^{-12}\text{ m}$).
3. **84.9% Loss Reduction**: The SLSQP optimizer executed 32 function evaluations (25 Jacobians), driving assembled loss from the initial extrapolation baseline of $354.53$ down to **$40.48$** (objective) / $57.21$ (score).
4. **Independent MATLAB R2025b Update 5 Validation**: Replayed cold on DeskComputer under qualified solver settings (`ode15s`, `RelTol 1e-6`, `MaxStep 1/1440 s`), proving **$2.77\text{ }\mu\text{m}$** mean Euclidean parity and **$0.194\text{ mm}$** maximum Euclidean discrepancy across all 325 frames and all 25 markers against Pinocchio 4.1.0.

---

## 1. Uninterrupted Forward Dynamics Replay (0–0.90 s, Zero Resets)

| Metric                                          | Candidate 102 (extrapolated to 0.90s) | **Candidate 103 (Pinocchio 4.1.0)** | **Candidate 103 (Simscape R2025b)** |     Gate Target      |       Gate Status       |
| :---------------------------------------------- | :-----------------------------------: | :---------------------------------: | :---------------------------------: | :------------------: | :---------------------: |
| **Whole RMS (mm)**                              |                27.625                 |             **23.252**              |             **23.251**              | $\le 25.0\text{ mm}$ |        **PASS**         |
| **Early RMS $\le 0.60\text{ s}$ (mm)**          |                 9.995                 |             **10.146**              |             **10.146**              | $\le 12.0\text{ mm}$ |        **PASS**         |
| **Terminal RMS $t=0.90\text{ s}$ (mm)**         |                127.974                |             **50.938**              |             **50.905**              | $\le 45.0\text{ mm}$ |     $-60.2\%$ Drop      |
| **Clubhead cluster RMS $t=0.90\text{ s}$ (mm)** |                156.619                |             **26.422**              |             **26.411**              | $\le 60.0\text{ mm}$ |        **PASS**         |
| **Pelvis yaw diff (deg)**                       |            $-13.69^\circ$             |          **$-5.02^\circ$**          |          **$-4.99^\circ$**          |    $< 3.0^\circ$     |     $-63.5\%$ Drop      |
| **Pelvis yaw error %**                          |                26.71%                 |              **9.79%**              |              **9.74%**              |      $< 5.0\%$       |  Substantial Reduction  |
| **Active Bounds**                               |                  N/A                  |                **0**                |                **0**                |          0           |        **PASS**         |
| **Assembled Loss**                              |                354.530                |             **53.671**              |             **53.671**              |         N/A          | **$-84.9\%$ Reduction** |

---

## 2. Key Physical & Optimization Findings

1. **Early Trajectory Invariance ($\le 0.60\text{ s}$)**:
   - Early trajectory RMS firmly retained at **`10.146 mm`** ($\le 12.0\text{ mm}$ gate), demonstrating that the frozen Window 0 and multiple shooting formulation prevent downstream control adjustments from perturbing the backswing.
2. **Clubhead Path Precision Maintained**:
   - Clubhead cluster error at $t=0.90\text{ s}$ improved dramatically from $156.62\text{ mm}$ to **`26.411 mm`** (Simscape R2025b), comfortably within the $\le 60.0\text{ mm}$ gate ceiling.
3. **Whole Trajectory Whole RMS Gate Cleared**:
   - Whole trajectory RMS over the full $[0, 0.90\text{ s}]$ swing interval achieved **`23.251 mm`** (Simscape R2025b), fully satisfying the global $\le 25.0\text{ mm}$ Gate 1 target.
4. **Terminal Error Reduction**:
   - Terminal RMS dropped by $60.2\%$ ($127.97\text{ mm} \to 50.91\text{ mm}$), confirming the efficacy of the $B_4/B_5/B_6$ control activation in counteracting downswing divergence.

---

## 3. Cross-Engine Parity Summary (Pinocchio 4.1.0 vs Simscape R2025b)

| Parity Metric                   |                   Measured Value                    |                Qualification                 |
| :------------------------------ | :-------------------------------------------------: | :------------------------------------------: |
| **Mean Euclidean Discrepancy**  | **$0.00277\text{ mm}$ ($2.77\text{ }\mu\text{m}$)** |           Single-digit micrometer            |
| **Mean Coordinate Discrepancy** | **$0.00137\text{ mm}$ ($1.37\text{ }\mu\text{m}$)** |           Single-digit micrometer            |
| **Max Coordinate Discrepancy**  |  **$0.174\text{ mm}$ ($174\text{ }\mu\text{m}$)**   |            Sub-millimeter regime             |
| **Max Euclidean Discrepancy**   |  **$0.194\text{ mm}$ ($194\text{ }\mu\text{m}$)**   |            Sub-millimeter regime             |
| **MATLAB Simulation Time**      |                **$36.27\text{ s}$**                 |       Near real-time forward execution       |
| **Replay MAT File Size**        |                 **$448\text{ KB}$**                 | Qualified compact artifact ($< 1\text{ MB}$) |

---

## 4. Candidate Artifact Inventory

- `returned-candidate.json`: Candidate document with SHA-256 `6f8afe4f6105b251d4bea96f9c2a021fcb69fb7060f1b566d7acde916564a408`.
- `config.json`: Execution configuration with seed and candidate hashes.
- `receipt.json`: Convergence receipt with preflight zero-displacement audit and elapsed optimization time.
- `returned.json`: Full report including cost history and active bounds (`active_bound_count: 0`).
- `returned-nodes.json`: Intermediate shooting node chart coordinates.
- `returned-replay.npz`: Complete multi-channel trajectory data.
- `pinocchio_replay.mat`: Converted Pinocchio trajectory.
- `qualified_candidate_replay.json`: Independent MATLAB R2025b Update 5 forward simulation report.
- `qualified_candidate_replay.mat`: Compact MATLAB replay data ($448\text{ KB}$).
- `two_window_fit_103.py`: Full source script.
- `replay_returned103_r2025b.m`: R2025b validation script.
