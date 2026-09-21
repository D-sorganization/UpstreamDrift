# Baseline Packages, Fit Metrics, and Qualification Profiles (TB-02)

Part of the **Matched Swing Program** (#10363, #10584, #10587).

## 1. Un-Conflated Status Architecture

To avoid conflating numerical convergence, tracking fidelity, physical feasibility, scientific reproducibility, and product readiness, every tour baseline candidate records five distinct statuses:

| Status Axis                  | Allowed States                                                            | Meaning                                                                        |
| ---------------------------- | ------------------------------------------------------------------------- | ------------------------------------------------------------------------------ |
| **Solver Status**            | `converged`, `reached_max_iter`, `numerical_failure`, `timeout`, `failed` | Numerical termination condition of the optimizer / integrator.                 |
| **Kinematic Accuracy**       | `accurate`, `inaccurate`, `untested`                                      | Whether 3D Euclidean marker errors satisfy target tracking bounds.             |
| **Dynamic Feasibility**      | `feasible`, `infeasible`, `not_applicable`                                | Whether actuators, normal forces, and ground contacts respect physical limits. |
| **Scientific Qualification** | `qualified`, `unqualified`, `rejected`, `historical`                      | Rigorous peer-reviewable gate outcome requiring native replay verification.    |
| **Product Promotion**        | `promoted`, `unpromoted`, `candidate`, `rejected`                         | Deployment eligibility as a shipping tour baseline.                            |

### Invariant Rules

1. **No Replay, No Qualification:** A complete manifest with missing or unverified native replay cannot qualify (`verified_reproduced == False` $\implies$ `unqualified`).
2. **Synthetic Data Guard:** Synthetic packages carry `is_synthetic_test_data = True` and strictly cannot be promoted (`product_status = rejected`).
3. **Historical Preservation:** Legacy receipts lacking native replay remain unverified but fully browseable under `historical` status.

---

## 2. Quantitative Fit Metrics & Error Formulas

Physical 3D Euclidean marker tracking error formulas:

### Whole-Capture RMSE

$$\text{RMSE} = \sqrt{\frac{\sum_{(t, m) \in \text{valid}} \| \mathbf{p}^{\text{pred}}_{t,m} - \mathbf{p}^{\text{obs}}_{t,m} \|^2}{N_{\text{valid}}}}$$

Where:

- $N_{\text{valid}}$ is the explicit observed denominator (valid samples present in observation and prediction).
- Excluded sample count and coverage fraction ($N_{\text{valid}} / N_{\text{total}}$) are tracked and audited.
- Optimizer weighted loss is stored separately from physical metric RMSE (metres).

### Landmark Set Inequivalence

Each observation set is deterministically identified by `landmarks_hash` (order-independent SHA-256 hash). Two models evaluated on different landmark subsets cannot be ranked as equivalent.

---

## 3. Model-Class Qualification Profiles

Rather than relaxing full-body thresholds to accommodate reduced models or forcing full-body contact gates onto planar pendulums, each model class defines a frozen, versioned qualification profile (`tour-qualification/1.0.0`):

| Model Class                                           | Modeled DoFs                                          | Key Quantitative Gates                                                                                                                     | Replay Requirement                                |
| ----------------------------------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| **Full-Body Multibody** (`full_body_mech`)            | 38+ DoF (humanoid + club)                             | Whole RMSE $\le 0.060\text{ m}$ (driver G3) / $0.095\text{ m}$ (iron G3); clubhead $\le 0.100\text{ m}$; ground polygon support $\ge 85\%$ | Native engine replay verification required        |
| **Double Pendulum Planar** (`double_pendulum_planar`) | 2 DoF (shoulder hub, wrist hinge)                     | In-plane clubhead RMSE $\le 0.120\text{ m}$; whole RMSE $\le 0.150\text{ m}$; coverage $\ge 90\%$                                          | Native ODE/multibody replay verification required |
| **Triple Pendulum Planar** (`triple_pendulum_planar`) | 3 DoF (trunk, arm, wrist hinge)                       | In-plane clubhead RMSE $\le 0.085\text{ m}$; whole RMSE $\le 0.110\text{ m}$; coverage $\ge 90\%$                                          | Native ODE/multibody replay verification required |
| **Upper Body Golfer 3D** (`upper_body_golfer_3d`)     | 11 independent DoF (spine, shoulders, elbows, wrists) | Upper-body 3D RMSE $\le 0.050\text{ m}$; clubhead RMSE $\le 0.075\text{ m}$; coverage $\ge 90\%$                                           | Native engine replay verification required        |

---

## 4. Evidence Artifacts

Generated reproducible artifacts under `docs/plans/tour_baselines/evidence/`:

- `synthetic_valid_baseline_package/`: Valid double-pendulum package demonstrating qualification with synthetic test guard (`is_qualified=True`, `is_promoted=False`).
- `synthetic_invalid_baseline_package/`: Failing double-pendulum package demonstrating gate failure and missing replay rejection (`is_qualified=False`, `is_promoted=False`).
- `baseline_package_receipt.json`: Summary evidence receipt linking the test baseline packages.
