# Assumptions — Option 1

Everything Option 1 takes as given. If any of these change, the fit is no longer guaranteed to work and the relevant tests must be rewritten.

Cross-cutting assumptions are also in the shared specs and are linked back here. Option 1 does not relax any of them.

## 1. Forward simulator is fixed

- The simulator is [GolfSwing3D_Kinetic.slx](../../src/model/GolfSwing3D_Kinetic.slx). Option 1 does **not** modify the .slx, the body parameters, or the integrator settings.
- Simulation duration is fixed at the default `T ≈ 0.3 s`; the optimizer is not allowed to choose `T`.
- Simulator sample rate is `1000 Hz` (default in [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md#defaults)).
- The simulator is invoked through `simulate_with_coefficients(theta)` (issue #018). Option 1 treats it as a black box.
- See [README — Hard constraints](../README.md#hard-constraints-assumptions-all-four-options-must-respect).

## 2. Decision variables: polynomial coefficients

- `theta` is a flat vector of length `n_joints × 7`. The 7 coefficients per joint are `A B C D E F G` corresponding to powers `t^6 ... t^0`.
- The list of joints comes from [getPolynomialParameterInfo.m](../../src/functions/dataset_generator/getPolynomialParameterInfo.m). Option 1 calls it once at the start of a fit and caches the result for the duration of the run.
- The flatten/unflatten convention is **joint-major, coefficient-minor** (`theta(7j+k+1)` is coefficient `k` of joint `j`, both 0-indexed inside the math, 1-indexed in MATLAB).
- See [APPROACH.md — Decision variable layout](APPROACH.md#decision-variable-layout).

## 3. Bounds

From [generateRandomCoefficients.m](../../src/functions/dataset_generator/generateRandomCoefficients.m):

| Coefficient | Power | Lower | Upper |
| ----------- | ----- | ----- | ----- |
| A           | t^6   | −1000 | +1000 |
| B           | t^5   | −1000 | +1000 |
| C           | t^4   | −500  | +500  |
| D           | t^3   | −500  | +500  |
| E           | t^2   | −100  | +100  |
| F           | t^1   | −100  | +100  |
| G           | t^0   | −25   | +25   |

These are **outer bounds**. The optimizer is free to converge inside them. Option 1 builds `lb`, `ub` vectors of length `n_joints × 7` from this table at the start of every fit.

## 4. Observation: club only (Phase 1)

- The cost function consumes butt position, clubhead position, and club orientation only. No body markers. No torque measurements. No EMG.
- This makes the fit **under-determined**: many torque histories produce the same club trajectory.
- We disambiguate with a regularizer (see assumption 5).
- See [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).

## 5. Cost function: `compute_cost.m` (issue #015), defined in [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md)

- Position term + orientation term + impact-anchor term + regularizer.
- Regularizer defaults to `total_work` (see [shared/COST_FUNCTION_SPEC.md — Regularizer](../shared/COST_FUNCTION_SPEC.md#regularizer-minimum-total-mechanical-work)).
- Default weights: `w_position=1`, `w_orientation=0.1`, `w_anchor_impact=10`, `lambda=1e-4`.
- Option 1 does **not** change the cost spec. It only schedules `w_anchor_impact` over iterations (see [APPROACH.md — Impact-anchor schedule](APPROACH.md#impact-anchor-schedule)).

## 6. Time alignment is the cost function's responsibility

- The `target` struct already lives on the simulation timegrid (see [shared/CLUB_IK_SPEC.md — Time alignment](../shared/CLUB_IK_SPEC.md#time-alignment)).
- Option 1 does not re-time, re-sample, or re-window the target. If the target is mis-aligned, it is fixed in the loader, not here.

## 7. Gradient is **not** available analytically

- Simscape does not expose adjoints. Option 1 treats the cost as a black box.
- `fmincon` is run with **finite-difference** gradients (`'FiniteDifferenceType' = 'forward'` by default; `'central'` if `options.fd_central = true`). This makes one gradient evaluation cost `n_joints × 7 + 1` simulations.
- Hessian is BFGS (the default for `fmincon-sqp`).
- Where finite differences are too expensive (`n_joints × 7` simulations per iteration), `surrogateopt` is preferred.
- See [APPROACH.md — Solver strategies](APPROACH.md#solver-strategies).

## 8. Cost is deterministic given `theta`

- The Simscape integrator is deterministic with fixed solver settings. We assume `simulate_with_coefficients(theta)` returns bit-identical output for identical input within a single MATLAB session.
- If determinism is not preserved across MATLAB sessions (e.g. different BLAS), the cache (assumption 11) is keyed by MATLAB version as well as `theta`.
- The optimizer's stochastic re-runs (e.g. `MultiStart` random starts) are seeded for reproducibility — see assumption 11.

## 9. Parallelism

- `MultiStart` and `surrogateopt` use `parpool` when `options.use_parallel = true` (default true if `Parallel Computing Toolbox` is licensed).
- The number of workers is `options.num_workers` (default = `feature('numcores')`).
- `parsim` is preferred over `parfor` for batched Simulink runs because it manages model loading per worker. Fallback to `parfor` if `parsim` errors.
- Workers do **not** share the cache; the cache is in-process. Cross-worker caching is an explicit non-goal for v1.

## 10. Hard constraints: bounds only (no equality or nonlinear constraints)

- Option 1 imposes only the box bounds from assumption 3. No equality constraints. No nonlinear constraints.
- Future work may add a constraint that the club position at `t = 0` matches the address pose; for v1 the address pose is enforced by initial conditions in the .slx, not by the optimizer.

## 11. Reproducibility / caching

- Every fit produces the result struct in [shared/CODING_STANDARDS.md — Provenance](../shared/CODING_STANDARDS.md#provenance-and-reproducibility).
- The result cache is keyed by `sha256(target) || sha256(options) || git_commit || matlab_version`. Cache hit returns the prior result struct unchanged.
- Random starts in `MultiStart` and `surrogateopt` are seeded by `options.rng_seed` (default = `42`). Reseeding is explicit; the optimizer does not pull from the global stream.

## 12. Failure mode: simulator-side errors

- If `simulate_with_coefficients(theta)` errors (e.g. integrator divergence) the cost evaluator returns a large finite penalty (`options.penalty_on_sim_failure`, default `1e9`) — **not** `NaN` or `Inf`. This keeps the optimizer convergent. The penalty is logged.
- If a single fit experiences `> options.max_sim_failures` (default 50) the fit is aborted with a clear error.

## 13. Coordinate convention

- Position in metres, time in seconds, orientation as unit quaternion `[w x y z]` (see [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md)).
- The Excel mocap loader is responsible for converting from inches; Option 1 receives metric data only.

## 14. Determinism of the cache hash

- `target_hash` is computed from `[time, butt, clubhead, club_quat, impact_idx]` only — **not** from `target.source`. Two targets that differ only in `source` provenance hit the same cache entry.

---

### Cross-cutting links (do not duplicate)

- Cost function spec: [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md).
- Club IK / target schema: [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).
- Coding standards (TDD, DbC, DRY, LOD, file size): [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md).
- Visualization spec (three required views): [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md).
