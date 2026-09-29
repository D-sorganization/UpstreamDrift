# Interfaces — Option 1

Every public function this option ships, with full signatures, contracts, and post-conditions. Lower-level agents implement to this; the tests in [TESTING.md](TESTING.md) bind to these signatures and **must not** be relaxed without a doc PR.

All preconditions are MATLAB `arguments` blocks (R2019b+) using `mustBe*` validators per [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md#dbc-design-by-contract). All post-conditions are `assert`s at the end of the function body.

Custom validators live in `motion_matching/shared/+validators/`. New ones for Option 1 (e.g. `mustBeOption1Options`) live in `option1_direct_optimization/private/+validators/`.

## `default_option1_options() -> options`

Returns the canonical options struct. Every other Option 1 entry point accepts `options` as its second argument and merges user overrides on top of this default.

```matlab
function options = default_option1_options()
%DEFAULT_OPTION1_OPTIONS  Canonical options for Option 1 fits.
%
%   OPTIONS = DEFAULT_OPTION1_OPTIONS() returns the default options struct
%   for every fit_swing_* function. Override individual fields and pass to
%   the fit_swing_* function. Bad fields produce a clean error from the
%   validator, NOT a silent override.
%
%   Postconditions:
%     - All fields documented below are present.
%     - Field types and ranges match the validator in
%       option1_direct_optimization/private/+validators/mustBeOption1Options.m.
end
```

Required fields and defaults:

| Field                    | Type    | Default                       | Notes                                                                                                                                        |
| ------------------------ | ------- | ----------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `solver`                 | string  | `"hybrid"`                    | One of `"fmincon", "multistart", "surrogateopt", "hybrid"`. Selecting via this is equivalent to calling the matching `fit_swing_*` directly. |
| `schedule`               | string  | `"staged"`                    | `"flat" \| "staged"` (see [APPROACH.md](APPROACH.md#multi-stage-schedule)).                                                                  |
| `cost`                   | struct  | from `default_cost_options()` | Cost-fn weights — see [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md#defaults).                                              |
| `cold_start_strategy`    | string  | `"dataset"`                   | `"dataset" \| "sobol" \| "zeros"`. See [APPROACH.md — Cold-start](APPROACH.md#cold-start-strategy).                                          |
| `multistart_n`           | uint32  | 8                             | N for `MultiStart`.                                                                                                                          |
| `max_iterations`         | uint32  | 200                           | `fmincon` cap.                                                                                                                               |
| `max_function_evals`     | uint32  | 500                           | `surrogateopt` cap and `fmincon`'s eval cap.                                                                                                 |
| `fd_central`             | logical | false                         | `true` ⇒ central diffs. ~2× cost, more accurate.                                                                                             |
| `tol_fun`                | double  | 1e-6                          | `OptimalityTolerance`.                                                                                                                       |
| `tol_x`                  | double  | 1e-8                          | `StepTolerance`.                                                                                                                             |
| `use_parallel`           | logical | depends on PCT license        | Parallel pool for MultiStart / surrogateopt.                                                                                                 |
| `num_workers`            | uint32  | `feature('numcores')`         | Pool size.                                                                                                                                   |
| `rng_seed`               | uint32  | 42                            | Reproducibility seed.                                                                                                                        |
| `use_cache`              | logical | true                          | Result-cache opt-out.                                                                                                                        |
| `cache_dir`              | string  | `"results/cache"`             | Path relative to `motion_matching/`.                                                                                                         |
| `verbosity`              | string  | `"Normal"`                    | `"Silent" \| "Normal" \| "Verbose" \| "Debug"`.                                                                                              |
| `dashboard`              | logical | true                          | Live `OptimizationProgressDashboard`.                                                                                                        |
| `dashboard_refresh_hz`   | double  | 5                             | Cap per [shared/VISUALIZATION_SPEC.md — Live updates](../shared/VISUALIZATION_SPEC.md#live-updates).                                         |
| `penalty_on_sim_failure` | double  | 1e9                           | Per [ASSUMPTIONS.md — Failure mode](ASSUMPTIONS.md#12-failure-mode-simulator-side-errors).                                                   |
| `max_sim_failures`       | uint32  | 50                            | Abort threshold.                                                                                                                             |
| `global_stage`           | string  | `"surrogateopt"`              | For `fit_swing_hybrid`. `"surrogateopt" \| "particleswarm"`.                                                                                 |

The validator `mustBeOption1Options(s)` enforces field presence + types and **errors** on unknown fields (a typo like `options.lamda = 1e-3` must not silently fail).

## `fit_swing_fmincon(target, options) -> result`

Single-start gradient (SQP) fit.

```matlab
function result = fit_swing_fmincon(target, options)
%FIT_SWING_FMINCON  Local SQP fit of polynomial torque coefficients.
%
%   RESULT = FIT_SWING_FMINCON(TARGET, OPTIONS) fits a coefficient vector
%   THETA such that running the Simscape forward simulator with THETA
%   reproduces TARGET in the sense of COST_FUNCTION_SPEC.md.
%
%   TARGET is the canonical struct from CLUB_IK_SPEC.md.
%   OPTIONS is from default_option1_options() with optional overrides.
%
%   Algorithm: fmincon('algorithm','sqp') with finite-difference gradient.
%   Warm start: options.cold_start_strategy.
%
%   Preconditions (DbC, validated by `arguments`):
%     - TARGET has fields {time, butt, clubhead, club_quat, impact_idx, source}.
%     - OPTIONS satisfies mustBeOption1Options.
%
%   Postconditions:
%     - result.solver == "fmincon"
%     - 0 <= result.final_rmse_m < Inf
%     - result.coefficients lies inside the bounds (lb,ub).
%     - result.iter_history has at least 1 row.
%     - result.target_hash matches sha256 of TARGET (without source).
%
%   Issue: #024.
    arguments
        target (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end
    error("not implemented");
end
```

## `fit_swing_multistart(target, options) -> result`

`MultiStart` over N random starts; parallel by default.

```matlab
function result = fit_swing_multistart(target, options)
%FIT_SWING_MULTISTART  MultiStart fit over N=options.multistart_n starts.
%
%   RESULT = FIT_SWING_MULTISTART(TARGET, OPTIONS) runs the Global
%   Optimization Toolbox MultiStart class wrapping fmincon-sqp. Starts are
%   selected per APPROACH.md "Cold-start strategy" (Sobol-spaced, optionally
%   seeded by the random-sweep parquet's lowest-cost members).
%
%   Parallelizes via parsim (preferred) or parfor when options.use_parallel
%   is true and the Parallel Computing Toolbox is licensed.
%
%   Preconditions: as fit_swing_fmincon, plus options.multistart_n >= 1.
%
%   Postconditions:
%     - result.solver == "multistart"
%     - size(result.start_points) == [d, options.multistart_n]
%     - result.coefficients == result.start_points(:, k*) for some k* whose
%       polish achieved the lowest fval.
%
%   Issue: #025.
    arguments
        target (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end
    error("not implemented");
end
```

## `fit_swing_surrogateopt(target, options) -> result`

`surrogateopt` global gradient-free.

```matlab
function result = fit_swing_surrogateopt(target, options)
%FIT_SWING_SURROGATEOPT  Global gradient-free fit using surrogateopt.
%
%   RESULT = FIT_SWING_SURROGATEOPT(TARGET, OPTIONS) runs the Global
%   Optimization Toolbox surrogateopt over the bound box. Uses
%   options.max_function_evals as the budget. Parallel evaluation when
%   options.use_parallel is true.
%
%   Preconditions: as fit_swing_fmincon.
%
%   Postconditions:
%     - result.solver == "surrogateopt"
%     - result.iter_history has options.max_function_evals rows
%       (or fewer if surrogateopt converged early).
%
%   Issue: #026.
    arguments
        target (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end
    error("not implemented");
end
```

## `fit_swing_hybrid(target, options) -> result`

Stage 1 (`surrogateopt` or `particleswarm`) → Stage 2 (`fmincon`) polish. **The recommended default.**

```matlab
function result = fit_swing_hybrid(target, options)
%FIT_SWING_HYBRID  Two-stage fit: global gradient-free then SQP polish.
%
%   RESULT = FIT_SWING_HYBRID(TARGET, OPTIONS) is the recommended default.
%   Stage 1 spends 70% of options.max_function_evals on
%   options.global_stage ("surrogateopt" by default; "particleswarm"
%   alternative). Stage 2 polishes from the global optimum with fmincon-sqp
%   for the remaining budget.
%
%   The impact-anchor weight and the regularizer lambda are scheduled
%   per APPROACH.md "Multi-stage schedule".
%
%   Preconditions: as fit_swing_fmincon.
%
%   Postconditions:
%     - result.solver == "hybrid"
%     - result.iter_history has a "stage" column with values {"global","polish"}.
%     - The polished theta has cost <= the global stage's best (assert).
%
%   Issue: #026.
    arguments
        target (1,1) struct {validators.mustBeClubTarget}
        options (1,1) struct {validators.mustBeOption1Options} = default_option1_options()
    end
    error("not implemented");
end
```

## `OptimizationProgressDashboard` (handle class)

Live dashboard during a fit. One instance per fit. The fit creates it (when `options.dashboard == true`), the `OutputFcn` pushes data to it at iteration boundaries, an internal timer drives refresh at `options.dashboard_refresh_hz` Hz (capped at 5 per [VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md#live-updates)).

```matlab
classdef OptimizationProgressDashboard < handle
%OPTIMIZATIONPROGRESSDASHBOARD  Live optimizer dashboard for Option 1 fits.
%
%   The dashboard renders four panels:
%     1. Cost vs iteration (semilogy)
%     2. |grad J| vs iteration (semilogy)
%     3. Step size ‖theta_{k+1} - theta_k‖ vs iteration (semilogy)
%     4. Current best theta as a horizontal-bar overlay on the bounds box
%
%   See VISUALIZATION.md and shared/VISUALIZATION_SPEC.md.
%
%   The OutputFcn callback pushes one row per iteration; rendering is
%   throttled by an internal timer at refresh_hz (default 5 Hz).
%
%   Issue: #027.
    properties (SetAccess = private)
        Figure          matlab.ui.Figure
        Target          struct                       % the target trajectory
        Bounds          struct                       % .lb, .ub
        RefreshHz       double = 5
        History         table   % iter, fval, grad_norm, step_norm, w_anchor, lambda, theta
        BestTheta       double  % d x 1 current best
        BestCost        double  % scalar
    end

    methods
        function obj = OptimizationProgressDashboard(target, bounds, refreshHz)
            arguments
                target (1,1) struct {validators.mustBeClubTarget}
                bounds (1,1) struct {validators.mustBeBoundsStruct}
                refreshHz (1,1) double {mustBePositive, mustBeLessThanOrEqual(refreshHz, 5)} = 5
            end
            error("not implemented");
        end

        function push(obj, row)
            %PUSH  Append one iteration's state. Called from the OutputFcn.
            arguments
                obj
                row (1,1) struct {validators.mustBeIterRow}
            end
            error("not implemented");
        end

        function close(obj)
            %CLOSE  Tear down the figure and timer. Idempotent.
            error("not implemented");
        end
    end
end
```

## `fmincon_output_fcn(theta, optimValues, state, dashboard, scheduleCtx) -> stop`

Output callback installed in `optimoptions('fmincon', 'OutputFcn', @(...) fmincon_output_fcn(...))`. Streams to the dashboard and updates the schedule context.

```matlab
function stop = fmincon_output_fcn(theta, optimValues, state, dashboard, scheduleCtx)
%FMINCON_OUTPUT_FCN  fmincon OutputFcn that streams to the live dashboard.
%
%   STOP = FMINCON_OUTPUT_FCN(THETA, OPTIMVALUES, STATE, DASHBOARD, SCHEDULECTX)
%   returns false (don't stop) unless the dashboard's user closed the figure.
%   Pushes one iteration row to DASHBOARD.push() per call. Updates
%   SCHEDULECTX (anchor-weight schedule cursor) so the cost wrapper sees
%   the right w_a on the next call.
%
%   Preconditions:
%     - state in {"init","iter","interrupt","done"}.
%
%   Postconditions:
%     - On state == "done", dashboard contains one row per iteration.
%
%   Issue: #027.
    arguments
        theta (:,1) double {mustBeFinite}
        optimValues (1,1) struct
        state (1,1) string {mustBeMember(state, ["init","iter","interrupt","done"])}
        dashboard
        scheduleCtx (1,1) struct
    end
    error("not implemented");
end
```

## Result struct contract

Every `fit_swing_*` function returns a struct with these fields (per [APPROACH.md — Output structure](APPROACH.md#output-structure)). The exact list is fixed; tests assert presence (`test_result_struct_contains_all_provenance_fields` in [TESTING.md](TESTING.md)).

| Field                | Type                   | Source                             |
| -------------------- | ---------------------- | ---------------------------------- |
| `coefficients`       | double, `d × 1`        | optimizer                          |
| `final_rmse_m`       | scalar double          | derived from `terms.position`      |
| `final_total_work_J` | scalar double          | derived from regularizer           |
| `final_cost_terms`   | struct                 | breakdown from `compute_cost`      |
| `solver`             | string                 | constant per fn                    |
| `solver_options`     | struct                 | full `optimoptions` used           |
| `target_hash`        | string (hex)           | `sha256(target without .source)`   |
| `git_commit`         | string                 | `git rev-parse HEAD` (best-effort) |
| `matlab_version`     | string                 | `version()`                        |
| `duration_s`         | scalar double          | wall clock                         |
| `timestamp_utc`      | string (ISO-8601)      | `datetime("now","TimeZone","UTC")` |
| `iter_history`       | table                  | per-iter rows                      |
| `exitflag`           | int                    | from solver                        |
| `output`             | struct                 | raw solver output                  |
| `start_points`       | `d × N` double or `[]` | only for MultiStart                |
| `start_costs`        | `1 × N` double or `[]` | only for MultiStart                |
| `cache_hit`          | logical                | always present                     |

## Validators (shared)

These validators must exist before any `fit_swing_*` function compiles. They are documented here so the implementer files them with the right signatures. Implementation issues are listed.

| Validator                    | File                                                                     | Issue |
| ---------------------------- | ------------------------------------------------------------------------ | ----- |
| `mustBeClubTarget(target)`   | `motion_matching/shared/+validators/mustBeClubTarget.m`                  | #013  |
| `mustBeOption1Options(opts)` | `option1_direct_optimization/private/+validators/mustBeOption1Options.m` | #024  |
| `mustBeBoundsStruct(b)`      | `option1_direct_optimization/private/+validators/mustBeBoundsStruct.m`   | #024  |
| `mustBeIterRow(r)`           | `option1_direct_optimization/private/+validators/mustBeIterRow.m`        | #027  |
| `mustHaveFields(s, names)`   | `motion_matching/shared/+validators/mustHaveFields.m`                    | #015  |
