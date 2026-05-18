# Approach — Option 1

The algorithm in detail. Lower-level agents implement to this spec; do not deviate without an issue and a doc PR.

## Problem statement

Find `theta* ∈ ℝ^d` minimizing the scalar cost `J(theta)` defined in [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md):

```
minimize    J(theta) = w_p · J_pos(theta) + w_o · J_ori(theta)
                       + w_a(k) · J_anchor(theta) + lambda · W_total(theta)
subject to  lb <= theta <= ub
```

- **Decision variables.** `theta`, `d = n_joints × 7`. From [getPolynomialParameterInfo.m](../../src/functions/dataset_generator/getPolynomialParameterInfo.m). Typical `d ≈ 200` (28 joints × 7) but Option 1 must read the count at runtime.
- **Bounds.** `lb`, `ub` derived from [generateRandomCoefficients.m](../../src/functions/dataset_generator/generateRandomCoefficients.m); see [ASSUMPTIONS.md — Bounds](ASSUMPTIONS.md#3-bounds).
- **Objective.** Black box. `J` requires a Simscape forward sim. No analytic gradient.
- **Equality / nonlinear constraints.** None in v1.

### Decision variable layout

```
theta = [ A_1, B_1, C_1, D_1, E_1, F_1, G_1,    % joint 1
          A_2, B_2, C_2, D_2, E_2, F_2, G_2,    % joint 2
          ...
          A_J, B_J, C_J, D_J, E_J, F_J, G_J ]   % joint J
```

Helpers (in `private/`):

- `flatten_coefficients(theta_struct) -> theta_vec`
- `unflatten_coefficients(theta_vec, joint_names) -> theta_struct`

These are the **only** places the layout convention is encoded. Every other function uses them. (DRY.)

## Solver strategies (in increasing cost / coverage)

### (a) `fmincon-sqp` single start

- **When.** You have a warm start (e.g. a previous fit, a related swing's coefficients, or the polynomial-fit-to-mocap-derivatives prior).
- **Algorithm.** `fmincon` with `'algorithm' = 'sqp'`, finite-difference forward gradient.
- **Cost.** ~`(d + 1)` simulations per iteration; typically 30–80 iterations to local optimum on a moderate problem.
- **API.** `fit_swing_fmincon(target, options)` — see [INTERFACES.md](INTERFACES.md#fit_swing_fmincon).
- **Caveat.** Finds the **nearest** local minimum to the start. On a cold start the swing is almost certainly multimodal — use (b) or (c).

### (b) `MultiStart` with N starting points

- **When.** Cold start. Default N = 8 (configurable via `options.multistart_n`).
- **Algorithm.** `MultiStart` from the Global Optimization Toolbox, wrapping `fmincon-sqp`. Starting points are drawn:
  - 1 from the current best cached solution if any (warm start).
  - 1 from the **lowest-cost member of the random sweep** in the parquet dataset (see [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md)) if available.
  - The remainder from the bound-constrained Sobol sequence (`sobolset(d)`, scrambled, `'Skip'=1024`).
- **Parallelism.** `'UseParallel' = true` when the `Parallel Computing Toolbox` is licensed. Per-worker `parsim` (preferred) or `parfor`. Each worker has its own loaded model; loading is amortized once per worker.
- **Cost.** N independent `fmincon` runs. Wall clock ≈ `(N / num_workers) × per-fmincon-cost`.
- **API.** `fit_swing_multistart(target, options)` — see [INTERFACES.md](INTERFACES.md#fit_swing_multistart).

### (c) `surrogateopt` global gradient-free

- **When.** No good warm start, very rough cost surface, or `d` is small enough (`d < 100`) that the surrogate fits make sense.
- **Algorithm.** `surrogateopt` from the Global Optimization Toolbox. Builds a radial-basis surrogate over the bound box; samples adaptively.
- **Cost.** No gradient computations; one cost eval per surrogate query. Default budget: `options.max_function_evals = 500`.
- **Parallelism.** `'UseParallel' = true`.
- **API.** `fit_swing_surrogateopt(target, options)` — see [INTERFACES.md](INTERFACES.md#fit_swing_surrogateopt).
- **Caveat.** `surrogateopt` quality drops as `d` grows above ~50. For full `d ≈ 200` the **hybrid** is the recommended default.

### Hybrid (the recommended default): `surrogateopt → fmincon` polish

- **Stage 1.** `surrogateopt` for `0.7 × budget` evals. Returns `theta_global`.
- **Stage 2.** `fmincon-sqp` from `theta_global` for the remaining budget. Returns `theta_local`.
- The hybrid is exposed as `fit_swing_hybrid(target, options)` — see [INTERFACES.md](INTERFACES.md#fit_swing_hybrid).
- An equivalent `particleswarm → fmincon` variant is offered behind `options.global_stage = "particleswarm"`. Choose `"surrogateopt"` (default) when each cost eval is expensive (which it is — minutes); choose `"particleswarm"` only when the cost is cheap and the surface is suspected smooth.

## Cold-start strategy

When no warm start is available:

1. If the parquet random-sweep dataset (see [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md)) is on disk, evaluate the cost (cheaply, by re-using the precomputed kinematics if the schema permits) on every member, take the K=8 best as the starting points for `MultiStart`. **No Simscape calls** for this step — it's a table lookup.
2. Otherwise, draw K=8 Sobol-spaced points across the bounds and evaluate `J` on each (`K` simulations). Take the lowest as the warm start.
3. If even (2) is too expensive, fall back to `theta = zeros(d, 1)` (the model's nominal). This is rarely good but it always converges to _something_.

The cold-start logic lives in `private/cold_start_seed.m` and is selected by `options.cold_start_strategy ∈ {"dataset","sobol","zeros"}` (default `"dataset"` if dataset is available, else `"sobol"`).

## Multi-stage schedule

Two schedules are supported and exposed via `options.schedule`:

### Schedule A — flat (default for `fit_swing_fmincon`)

Single solver, single set of weights, no schedule.

### Schedule B — staged (default for `fit_swing_hybrid`)

| Stage        | Solver                              | Iterations / evals                  | `w_anchor_impact`                               | `lambda`                            |
| ------------ | ----------------------------------- | ----------------------------------- | ----------------------------------------------- | ----------------------------------- |
| 0. Warm seed | (cold-start lookup)                 | —                                   | —                                               | —                                   |
| 1. Global    | `surrogateopt` (or `particleswarm`) | 70% of `options.max_function_evals` | `5 × w_p`                                       | `0`                                 |
| 2. Polish    | `fmincon-sqp`                       | 30% remaining                       | ramps `5 × w_p → 0` over the first 25% of iters | ramps `0 → options.lambda` over 50% |

Rationale: in stage 1 the optimizer needs the anchor to get into the right basin (otherwise it drifts into low-work, low-fidelity solutions). In stage 2 we want the unbiased cost so the final answer is **the** answer, not the answer-with-anchor. The regularizer is annealed in only after the position term is small enough that `lambda · W_total` doesn't dominate.

### Impact-anchor schedule

The impact-anchor weight `w_a` ramps according to:

```
w_a(k) = w_a_start + (w_a_end - w_a_start) · clamp((k - k0) / (k1 - k0), 0, 1)
```

Defaults:

- Stage 1 (`surrogateopt`): `w_a` is **constant** at `5 × w_p` (no schedule — surrogateopt doesn't expose iteration count cleanly).
- Stage 2 (`fmincon`): `w_a_start = 5 × w_p`, `w_a_end = 0`, `k0 = 0`, `k1 = 0.25 × max_iter`.

The schedule is implemented in `private/anchor_weight_schedule.m` and consumed by the cost wrapper that `fmincon` actually sees (`private/wrap_cost_with_schedule.m`).

## Caching: target hash → result struct

To avoid re-running an expensive fit:

1. Compute `target_hash = sha256([time; butt; clubhead; club_quat; impact_idx])`.
2. Compute `options_hash = sha256(jsonencode(options))`.
3. The cache key is `cache_key = sha256(target_hash || options_hash || git_commit || matlab_version)`.
4. Cache directory: `motion_matching/results/cache/<cache_key[1:2]>/<cache_key>.mat`.
5. On cache hit, return the stored `result` struct unchanged. On miss, run the fit and write the cache atomically (`save` to a tempfile, `movefile`).
6. Cache invalidation is **manual** — the user deletes the directory.

The cache lives in `private/result_cache.m` (read/write) and is consulted by every entry-point function before doing real work. Cache is opt-out via `options.use_cache = false`.

## Output structure

Every fit returns a `result` struct conforming to [shared/CODING_STANDARDS.md — Provenance](../shared/CODING_STANDARDS.md#provenance-and-reproducibility), plus Option-1-specific fields:

```
result.coefficients          (d × 1 double)
result.final_rmse_m          (scalar)
result.final_total_work_J    (scalar)
result.final_cost_terms      (struct: position, orientation, impact_anchor, regularizer, total)
result.solver                (string)
result.solver_options        (struct)
result.target_hash           (string)
result.git_commit            (string)
result.matlab_version        (string)
result.duration_s            (scalar)
result.timestamp_utc         (string ISO-8601)
%% Option-1-specific:
result.iter_history          (table: iter, fval, step_norm, grad_norm, w_anchor, lambda)
result.exitflag              (integer from solver)
result.output                (raw output struct from solver)
result.start_points          (d × N double, only set for MultiStart)
result.start_costs           (1 × N double, only set for MultiStart)
result.cache_hit             (logical)
```

The exact contract is in [INTERFACES.md — result struct](INTERFACES.md#result-struct-contract).

## What lives in `private/`

The implementer fills these in. They are not directly tested (LOD: tests go through the public API), but they are documented here so the implementer knows the shape:

- `flatten_coefficients.m`, `unflatten_coefficients.m` — layout helpers.
- `cold_start_seed.m` — picks the warm start.
- `wrap_cost_with_schedule.m` — wraps `compute_cost` with the iteration-dependent `w_anchor`.
- `anchor_weight_schedule.m` — the ramp.
- `result_cache.m` — cache read/write.
- `make_fmincon_options.m`, `make_surrogate_options.m` — build the `optimoptions` structs from the canonical options.
- `assemble_result_struct.m` — gathers provenance + solver output into the canonical `result`.

## What lives at the package root

- The four `fit_swing_*.m` entry points (the only public callable functions).
- `default_option1_options.m`.
- `OptimizationProgressDashboard.m` (handle class).
- `fmincon_output_fcn.m` — the output callback. Streams to the dashboard.

## Pseudocode for `fit_swing_hybrid` (the headline path)

```matlab
function result = fit_swing_hybrid(target, options)
    % 1. Cache check
    key = compute_cache_key(target, options);
    if options.use_cache && cache_has(key)
        result = cache_get(key);
        result.cache_hit = true;
        return
    end

    % 2. Cold-start seed
    [theta0, sweep_costs] = cold_start_seed(target, options);

    % 3. Stage 1: global
    cost_stage1 = wrap_cost_with_schedule(target, options, "stage1");
    [theta_global, info1] = surrogateopt(cost_stage1, lb, ub, surrogate_opts);

    % 4. Stage 2: polish
    cost_stage2 = wrap_cost_with_schedule(target, options, "stage2");
    [theta_local, fval, exitflag, output] = fmincon(cost_stage2, theta_global, ...
        [], [], [], [], lb, ub, [], fmincon_opts);

    % 5. Assemble + cache + return
    result = assemble_result_struct(theta_local, target, options, ...
                                    info1, output, exitflag);
    if options.use_cache, cache_put(key, result); end
end
```
