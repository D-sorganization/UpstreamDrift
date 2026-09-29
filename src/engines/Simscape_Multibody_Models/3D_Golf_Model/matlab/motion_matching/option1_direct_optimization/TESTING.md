# Testing — Option 1

**TDD is non-negotiable.** Per [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md#tdd-test-driven-development), tests are written **first** and committed **first** when feasible. Every PR includes the tests for the code it ships.

This file lists the concrete tests the lower-level agent must write first. Each test is a method on a `matlab.unittest.TestCase` subclass. Test files live in `option1_direct_optimization/tests/`.

Run from the repo root:

```matlab
results = runtests("motion_matching/option1_direct_optimization/tests", ...
                   "IncludeSubfolders", true);
```

## Test fixtures (write these first; everything else depends on them)

A small fixture class avoids re-loading the simulator on every test:

```
tests/
├── fixtures/
│   ├── Option1Fixture.m         (TestClassSetup: parpool warmup, model load)
│   ├── known_theta.mat          (a fixed, small theta to use in oracle tests)
│   └── synthetic_target_cache/  (cached synthesized targets, .gitignore'd)
```

`Option1Fixture` should:

- Open a tiny `parpool` (1–2 workers) for the suite, close at teardown.
- Pre-load the `.slx` once; share the model handle across tests via the fixture.
- Provide `target_synthetic = synthesize_target_from_coefficients(theta_known)` lazily and cache to disk.

## Required tests (Issues #024, #025, #026, #027)

### `test_options_struct_contract` — Issue #024

Asserts the options-struct DbC.

```matlab
function rejects_unknown_field(testCase)
    opts = default_option1_options();
    opts.lamda = 1e-3;  % typo of lambda
    testCase.verifyError( ...
        @() fit_swing_fmincon(testCase.target_synthetic, opts), ...
        "validator:unknownField");
end

function rejects_negative_lambda(testCase)
    opts = default_option1_options();
    opts.cost.lambda = -1;
    testCase.verifyError( ...
        @() fit_swing_fmincon(testCase.target_synthetic, opts), ...
        "validator:badRange");
end

function accepts_default_options(testCase)
    opts = default_option1_options();
    testCase.verifyTrue(isstruct(opts));
    testCase.verifyEqual(opts.solver, "hybrid");
end
```

Bad fields **must error cleanly** with a documented identifier — silent override is forbidden by [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md#dbc-design-by-contract).

### `test_fits_synthetic_to_within_1mm` — Issue #024

The trivial-fit oracle from [shared/CLUB_IK_SPEC.md — Synthetic](../shared/CLUB_IK_SPEC.md#3-synthetic-for-testing). If the optimizer cannot match the trajectory it itself produced, **the optimizer is broken**.

```matlab
function fmincon_fits_synthetic_to_within_1mm(testCase)
    target = synthesize_target_from_coefficients(testCase.theta_known);
    opts = default_option1_options();
    opts.cost.lambda = 0;       % no regularizer for the oracle test
    opts.dashboard = false;     % no UI in CI
    result = fit_swing_fmincon(target, opts);
    testCase.verifyLessThan(result.final_rmse_m, 1e-3);   % 1 mm
end
```

This test runs on **every** `fit_swing_*` function (parameterized via `TestParameterDefinition`). If a solver can't pass it, that solver is not shipping.

### `test_recovers_known_coefficients_within_5pct` — Issue #024

Stronger than the trajectory-recovery test: the **coefficients** themselves must come back. This is only possible with a regularizer (otherwise the under-determined fit can recover the trajectory with the wrong coefficients).

```matlab
function recovers_theta_within_5pct_with_work_regularizer(testCase)
    theta_truth = testCase.theta_known;
    target = synthesize_target_from_coefficients(theta_truth);
    opts = default_option1_options();
    opts.cost.lambda = 1e-4;
    opts.dashboard = false;
    result = fit_swing_hybrid(target, opts);

    rel_err = norm(result.coefficients - theta_truth) / norm(theta_truth);
    testCase.verifyLessThan(rel_err, 0.05);
end
```

Run on `fit_swing_hybrid` only (a single-start `fmincon` is unlikely to pass and is allowed to fail this test). On `fit_swing_multistart` the threshold is the same; on `fit_swing_surrogateopt` it's 0.10 (less stringent).

### `test_total_work_regularizer_reduces_torque_magnitude` — Issue #024

Asserts the regularizer does what it says. Two fits on the same target — one with `lambda = 0`, one with `lambda = 1e-4` — and the latter has lower total work.

```matlab
function lambda_positive_reduces_total_work(testCase)
    target = synthesize_target_from_coefficients(testCase.theta_known);
    opts0 = default_option1_options();   opts0.cost.lambda = 0;     opts0.dashboard = false;
    opts1 = default_option1_options();   opts1.cost.lambda = 1e-4;  opts1.dashboard = false;

    r0 = fit_swing_fmincon(target, opts0);
    r1 = fit_swing_fmincon(target, opts1);

    testCase.verifyLessThan(r1.final_total_work_J, r0.final_total_work_J);
    testCase.verifyLessThan(r1.final_rmse_m, 5e-3);  % still a reasonable fit
end
```

### `test_multistart_outperforms_single_start_on_known_multimodal_target` — Issue #025

Build a target whose cost surface has two basins (e.g. by mixing two synthesised swings or by using a known-multimodal `theta_pair`). Show `MultiStart(N=8)` finds the deeper basin where the single start does not.

```matlab
function multistart_finds_deeper_basin(testCase)
    target = testCase.multimodal_target;   % fixture that builds the multimodal case
    opts = default_option1_options();   opts.dashboard = false;
    opts.multistart_n = 8;

    r_single = fit_swing_fmincon(target, opts);
    r_multi  = fit_swing_multistart(target, opts);

    testCase.verifyLessThan(r_multi.final_rmse_m, 0.5 * r_single.final_rmse_m);
end
```

### `test_surrogateopt_completes_in_under_5min_on_30coeff_problem` — Issue #026

Bound the wall-clock so we don't ship a slow surrogate.

```matlab
function surrogateopt_completes_in_under_5min(testCase)
    target = testCase.target_synthetic_small;   % 30-coefficient subset
    opts = default_option1_options();
    opts.max_function_evals = 200;
    opts.dashboard = false;
    t0 = tic;
    result = fit_swing_surrogateopt(target, opts);
    elapsed = toc(t0);
    testCase.verifyLessThan(elapsed, 300);   % 5 minutes
    testCase.verifyLessThan(result.final_rmse_m, 5e-3);
end
```

This test is marked `slow` and excluded from the fast CI pass per [CLAUDE.md — Test markers](../../../../../../CLAUDE.md). It runs on the nightly.

### `test_result_struct_contains_all_provenance_fields` — Issue #024

Asserts the contract from [INTERFACES.md — Result struct contract](INTERFACES.md#result-struct-contract).

```matlab
function result_has_all_required_fields(testCase, fitFn)
    result = fitFn(testCase.target_synthetic, default_option1_options());

    required = ["coefficients", "final_rmse_m", "final_total_work_J", ...
                "final_cost_terms", "solver", "solver_options", ...
                "target_hash", "git_commit", "matlab_version", ...
                "duration_s", "timestamp_utc", "iter_history", ...
                "exitflag", "output", "cache_hit"];
    for f = required
        testCase.verifyTrue(isfield(result, f), ...
            sprintf("Missing required result field: %s", f));
    end

    testCase.verifyMatches(result.target_hash, "^[0-9a-f]{64}$");
    testCase.verifyClass(result.iter_history, "table");
end
```

Parameterize `fitFn` over all four `fit_swing_*` functions. The MultiStart variant additionally asserts `result.start_points` and `result.start_costs` are non-empty.

## Additional tests (high-value, non-required-by-issue)

These are not strictly required by the four issues but the agent **should** write them; reviewers will look for them.

### `test_cache_hit_returns_identical_result`

```matlab
function cache_hit_skips_recomputation(testCase)
    opts = default_option1_options();   opts.dashboard = false;   opts.use_cache = true;
    r1 = fit_swing_fmincon(testCase.target_synthetic, opts);
    t0 = tic;
    r2 = fit_swing_fmincon(testCase.target_synthetic, opts);
    elapsed = toc(t0);
    testCase.verifyEqual(r2.coefficients, r1.coefficients);
    testCase.verifyTrue(r2.cache_hit);
    testCase.verifyLessThan(elapsed, 1);   % cache hit < 1 s
end
```

### `test_sim_failure_yields_finite_penalty_not_NaN`

Force a `theta` that diverges the integrator and assert the penalty path is hit.

### `test_options_struct_is_dry`

Asserts every `fit_swing_*` function uses the **same** options struct produced by `default_option1_options()`. (DRY, per [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md#dry).) Implementation: introspect the validators each function calls.

### `test_fmincon_output_fcn_pushes_one_row_per_iteration`

Drives a small fit and asserts `dashboard.History` has `result.output.iterations + 1` rows (counting `"init"`).

### `test_no_method_chains_deeper_than_two`

Static lint over the option's `.m` files using a small regex helper in `motion_matching/shared/static_lint/`. Per [shared/CODING_STANDARDS.md — LOD](../shared/CODING_STANDARDS.md#lod-law-of-demeter).

## What the agent runs locally before opening a PR

```matlab
results = runtests("motion_matching/option1_direct_optimization/tests", ...
                   "IncludeSubfolders", true);
disp(table(results));
assert(all([results.Passed]), "tests failed; do not push");
```

And on the Python side (for the file-size budget and TODO/FIXME checks per [CLAUDE.md — CI Requirements](../../../../../../CLAUDE.md#ci-requirements-all-must-pass)):

```bash
python3 scripts/ci/check_file_size_budget.py
python3 -m ruff check src/  # if any Python was added
```

## Test categorization

Per the markers in [CLAUDE.md — Test markers](../../../../../../CLAUDE.md):

| Test                                                                  | Marker                           | Where it runs  |
| --------------------------------------------------------------------- | -------------------------------- | -------------- |
| `test_options_struct_contract`                                        | `unit`                           | Every CI run   |
| `test_fits_synthetic_to_within_1mm`                                   | `integration`, `live_simulation` | Hourly nightly |
| `test_recovers_known_coefficients_within_5pct`                        | `integration`, `slow`            | Nightly        |
| `test_total_work_regularizer_reduces_torque_magnitude`                | `integration`                    | Every CI run   |
| `test_multistart_outperforms_single_start_on_known_multimodal_target` | `slow`, `live_simulation`        | Nightly        |
| `test_surrogateopt_completes_in_under_5min_on_30coeff_problem`        | `slow`, `benchmark`              | Nightly        |
| `test_result_struct_contains_all_provenance_fields`                   | `unit`                           | Every CI run   |
| `test_cache_hit_returns_identical_result`                             | `unit`                           | Every CI run   |

The `unit`-marked subset must complete in under 60 s to fit the [CLAUDE.md pytest timeout](../../../../../../CLAUDE.md#development-commands).
