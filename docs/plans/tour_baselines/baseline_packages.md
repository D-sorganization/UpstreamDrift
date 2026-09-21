# Baseline Packages, Manifest Identities, and Status Bundles

Parent Epic: [#10584](https://github.com/D-sorganization/UpstreamDrift/issues/10584)  
Work Package: [#10587](https://github.com/D-sorganization/UpstreamDrift/issues/10587) (TB-02)  
Governing Epic: [#10363](https://github.com/D-sorganization/UpstreamDrift/issues/10363) (Matched Swing Program)

---

## 1. Overview and Architecture

The **Baseline Package Contract** (`tour-baseline-package/1.0.0`) extends the portable results framework established under #10379 and #10334 without creating a parallel baseline store. It provides deterministic, self-contained packaging for physical motion matching candidates across all registered golf model topologies.

### Core Guarantees

1. **Deterministic Identity:** Every baseline package is bound to a cryptographic `BaselineIdentity` linking capture target, model topology, numerical backend, solver configuration, random seed, computational budgets, and runtime environment hashes.
2. **Orthogonal Status Bundle:** Five separate dimensions are tracked independently:
   - **Solver Convergence:** Numerical optimization termination (`converged`, `max_iterations`, `infeasible`, `diverged`, `unsolved`).
   - **Kinematic Accuracy:** Geometric marker tracking error against declared profile thresholds (`within_tolerance`, `exceeds_threshold`, `unevaluated`).
   - **Dynamic Feasibility:** Physical validity including ground contact forces, friction cone limits, joint limits, and weld closures (`physically_feasible`, `infeasible`, `kinematic_only`, `unevaluated`).
   - **Scientific Qualification:** Authoritative peer-reviewed scientific gate decision (`qualified`, `disqualified`, `unverified`, `pending_review`).
   - **Product Promotion:** Readiness for customer-facing application shells (`promoted`, `exploratory`, `rejected`, `demoted`).
3. **Fail-Closed Native Replay Rule:** A complete candidate manifest lacking native simulation replay evidence **cannot** qualify scientifically (`scientific_qualification != qualified`).
4. **Synthetic Test Data Guard:** Synthetic fixtures explicitly declare `is_synthetic = True`, which permanently bars them from product promotion (`product_promotion != promoted`).

---

## 2. Baseline Identity Schema

The `BaselineIdentity` immutable dataclass encapsulates all generation parameters:

| Field                     | Type              | Description                                                                    |
| ------------------------- | ----------------- | ------------------------------------------------------------------------------ | --------------------------------------------------- |
| `model_id`                | `str`             | Canonical golf model identifier from registry                                  |
| `topology`                | `ModelTopology`   | Mechanical topology (full-body, planar, upper-body, kinematic)                 |
| `backend`                 | `BackendType`     | Simulation engine (`mujoco`, `pinocchio`, `drake`, etc.)                       |
| `provider_pin`            | `str`             | Commit hash or pinned version of provider                                      |
| `fit_mode`                | `FitMode`         | Trajectory nature (`kinematic_pose`, `prescribed_trajectory`, `torque_driven`) |
| `capture`                 | `str`             | Canonical capture trial (`driver` or `iron`)                                   |
| `capture_sha256`          | `str`             | Audited SHA-256 hash of C3D optical capture                                    |
| `horizon`                 | `str`             | Evaluation horizon (`G1`, `G2`, `G3`)                                          |
| `frame_convention`        | `str`             | Coordinate convention (default: `z_up_y_forward`)                              |
| `plane_convention`        | `str`             | Functional plane layout (`transverse_sagittal_frontal`)                        |
| `measurement_map_version` | `str`             | Semantics version (`tour-measurement-map/1.0.0`)                               |
| `fixed_geometry_hash`     | `str`             | SHA-256 of immutable link lengths and segment dimensions                       |
| `fixed_inertia_hash`      | `str`             | SHA-256 of segment mass and rotational inertia tensor                          |
| `solver_name`             | `str`             | Optimization algorithm (e.g. `ipopt`, `sqp`, `slsqp`)                          |
| `solver_config`           | `dict`            | Solver parameters (`max_iter`, tolerances, barrier weights)                    |
| `integrator`              | `str`             | Numerical integrator (`implicit_euler`, `rk4`)                                 |
| `seed`                    | `int              | None`                                                                          | Deterministic pseudo-random seed                    |
| `wall_clock_budget_s`     | `float            | None`                                                                          | Computation wall-clock budget in seconds            |
| `max_evaluations_budget`  | `int              | None`                                                                          | Computation budget in function/Jacobian evaluations |
| `candidate_ancestry`      | `tuple[str, ...]` | Lineage hashes of predecessor candidates                                       |
| `runtime_hashes`          | `dict[str, str]`  | Environment dependencies (`python`, `numpy`, compiler)                         |
| `file_hashes`             | `dict[str, str]`  | SHA-256 of model definition and configuration files                            |

---

## 3. Physical Fit Metrics Formulation

Physical marker tracking error is formulated strictly using 3D Euclidean distances on observed valid data:

$$\text{RMSE} = \sqrt{\frac{1}{N_{\text{valid}}} \sum_{(t,m) \in \mathcal{V}} \|\mathbf{p}^{\text{pred}}_{t,m} - \mathbf{p}^{\text{obs}}_{t,m}\|_2^2}$$

Where:

- $\mathcal{V} = \{(t,m) \mid \text{valid}[t,m] = \text{True}\}$ is the set of valid observations.
- $N_{\text{valid}} = |\mathcal{V}|$ is the observed denominator.
- Empty valid sets ($N_{\text{valid}} = 0$) are rejected with `ValueError`, never reported as zero error.
- Non-finite coordinates ($\text{NaN}, \infty$) at valid indices are rejected with `ValueError`.
- Optimizer weighted loss is recorded distinctly under `optimizer_weighted_loss`, ensuring unweighted physical millimetres are never conflated with penalty terms.
- The `landmark_set_signature` computes a SHA-256 hash over the sorted landmark labels, guaranteeing that disparate marker subsets cannot be ranked as equivalent.

---

## 4. Serialization and Clean-Machine Portability

Baseline packages are exported to `.npz` container files with strict security and portability standards:

- **No Pickling:** `allow_pickle = False` is enforced for both export and import.
- **Array Checksums:** Every array stored in the container (`q`, `v`, `tau`, `model_markers_m`, etc.) has its SHA-256 hash verified against the package manifest.
- **Tamper Detection:** Tampered arrays or unauthorized extra arrays are immediately detected and rejected.
- **Clean-Machine Re-basing:** Path references are relative to the package root, ensuring relocatable storage across developer workstations and automated CI runners.
