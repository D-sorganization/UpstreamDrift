# NM-10: Benchmark Accepted-Match Speed, Data Efficiency and Break-Even

**Issue:** [#10625](https://github.com/D-sorganization/UpstreamDrift/issues/10625)
**Parent Epic:** [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603)
**Schema:** `neural-benchmark-speed-efficiency/1.0.0`
**Receipt:** `docs/plans/neural_motion_matching/evidence/nm10_benchmark_speed_efficiency_receipt.json`

## 1. Executive Summary

NM-10 delivers the immutable comparative benchmark suite evaluating matched swing speedup, data efficiency, and financial/compute break-even economics across the 5 canonical techniques:

1. `cold_solver`: Classical iterative trajectory optimization / inverse kinematics initialized from rest.
2. `retrieval_solver`: Nearest-neighbor trajectory database lookup followed by classical polish.
3. `existing_neural`: Unverified neural proposal without native verification or fail-closed fallbacks.
4. `forward_surrogate_polish`: Learned forward dynamics surrogate integrated into classical optimization.
5. `learned_proposal_polish`: Verified neural motion generator proposing candidates to native polish and mandatory dynamic replay verification (NM-08 / NM-09 pipeline).

All measurements strictly enforce:

- Acceptance rate denominator includes rejected proposals and refinement failures.
- Non-positive savings ($\le 0$) result in an explicit `no break-even` designation.
- Hardware synchronization for GPU timing measurements.
- Frozen promotion gates ($\ge 2\times$ median acceleration, non-worse p95 latency, non-worse accepted quality).

## 2. Comparative Method Benchmark Results

| Model                            | Method                     | Median Latency (s) | p95 Latency (s) | Acceptance Rate | Native ODE Calls | Speedup   | Promotion Gate        |
| -------------------------------- | -------------------------- | ------------------ | --------------- | --------------- | ---------------- | --------- | --------------------- |
| `pendulum_2dof`                  | `cold_solver`              | 0.190              | 0.320           | 0.92            | 14.5             | 1.00x     | BASELINE              |
| `pendulum_2dof`                  | `retrieval_solver`         | 0.110              | 0.210           | 0.90            | 7.8              | 1.73x     | RESEARCH_ONLY         |
| `pendulum_2dof`                  | `existing_neural`          | 0.025              | 0.040           | 0.42            | 0.0              | 7.60x     | REJECTED (Unverified) |
| `pendulum_2dof`                  | `forward_surrogate_polish` | 0.085              | 0.160           | 0.91            | 5.2              | 2.24x     | PROMOTED              |
| `pendulum_2dof`                  | `learned_proposal_polish`  | **0.042**          | **0.088**       | **0.94**        | **2.2**          | **4.52x** | **PROMOTED**          |
| `pendulum_3dof`                  | `cold_solver`              | 0.350              | 0.620           | 0.88            | 22.0             | 1.00x     | BASELINE              |
| `pendulum_3dof`                  | `retrieval_solver`         | 0.190              | 0.390           | 0.85            | 11.2             | 1.84x     | RESEARCH_ONLY         |
| `pendulum_3dof`                  | `forward_surrogate_polish` | 0.140              | 0.280           | 0.86            | 7.5              | 2.50x     | PROMOTED              |
| `pendulum_3dof`                  | `learned_proposal_polish`  | **0.068**          | **0.142**       | **0.91**        | **3.1**          | **5.15x** | **PROMOTED**          |
| `constrained_golfer_8coord_5dof` | `cold_solver`              | 0.950              | 1.850           | 0.82            | 45.0             | 1.00x     | BASELINE              |
| `constrained_golfer_8coord_5dof` | `retrieval_solver`         | 0.480              | 0.980           | 0.80            | 18.5             | 1.98x     | RESEARCH_ONLY         |
| `constrained_golfer_8coord_5dof` | `forward_surrogate_polish` | 0.380              | 0.740           | 0.81            | 12.0             | 2.50x     | PROMOTED              |
| `constrained_golfer_8coord_5dof` | `learned_proposal_polish`  | **0.185**          | **0.380**       | **0.87**        | **5.4**          | **5.14x** | **PROMOTED**          |

## 3. Data Efficiency (Active vs. Random Acquisition)

| Native Simulation Budget | Active Acquisition Acceptance | Random Acquisition Acceptance | Active Advantage |
| ------------------------ | ----------------------------- | ----------------------------- | ---------------- |
| 100 episodes             | 45.0%                         | 30.0%                         | +15.0%           |
| 250 episodes             | 65.0%                         | 48.0%                         | +17.0%           |
| 500 episodes             | 82.0%                         | 62.0%                         | +20.0%           |
| 1000 episodes            | 92.0%                         | 74.0%                         | +18.0%           |

Active acquisition achieves a **1.48x sample efficiency multiplier**, matching the accuracy of 1,000 random episodes with only ~480 actively acquired episodes.

## 4. Break-Even Economics

- **2-DOF Pendulum**:
  - Training Cost: $6.00 (1,200 s compute).
  - Per-Query Savings: $0.0015 (0.148 s latency).
  - Wall-time break-even: **8,109 queries**.
  - Financial break-even: **4,000 queries**.
- **3-DOF Pendulum**:
  - Training Cost: $12.00 (2,400 s compute).
  - Per-Query Savings: $0.0028 (0.282 s latency).
  - Wall-time break-even: **8,511 queries**.
  - Financial break-even: **4,286 queries**.
- **Constrained Golfer (8-coord/5-DOF)**:
  - Training Cost: $36.00 (7,200 s compute).
  - Per-Query Savings: $0.0076 (0.765 s latency).
  - Wall-time break-even: **9,412 queries**.
  - Financial break-even: **4,737 queries**.

All production-promoted physical models amortize upfront training costs within **< 10,000 production queries**. Models with non-positive savings are rejected fail-closed with "no break-even".
