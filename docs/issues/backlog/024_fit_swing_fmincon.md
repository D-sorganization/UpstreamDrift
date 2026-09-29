# Issue: Implement fit_swing_fmincon.m (Option 1 Single-Start fmincon-sqp)

## Summary

Implement Option 1's single-start direct optimizer: an `fmincon`-sqp call over
the polynomial coefficients minimising the cost function from #015. This is the
simplest, most direct path to a fit and serves as the reference oracle for the
other three options.

## Motivation

See `motion_matching/README.md` "Why four options in parallel". Option 1 is what
ships a working fit on a real swing within ~2 weeks. `fmincon` with
`Algorithm='sqp'` handles the bound constraints from
`generateRandomCoefficients.m` and tolerates the noisy black-box gradient
implied by Simscape forward sims.

## Dependencies

- #013 (`load_club_target_excel.m`) — provides input target.
- #015 (`compute_cost.m`) — the J(θ) being minimised.
- #018 (`simulate_with_coefficients.m`) — forward call.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\fit_swing_fmincon.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\default_fmincon_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\private\build_result_struct.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\tests\test_fit_swing_fmincon.m`

## Public API

```matlab
function result = fit_swing_fmincon(target, options)
%FIT_SWING_FMINCON  Single-start fmincon-sqp swing fit.
%
%   result = FIT_SWING_FMINCON(TARGET, OPTIONS) optimizes polynomial
%   coefficients to minimise compute_cost(theta, TARGET, simulate_with_coefficients,
%   OPTIONS.cost_opts) using fmincon with Algorithm='sqp'.
%
%   TARGET is a struct conforming to CLUB_IK_SPEC.md.
%
%   OPTIONS is the result of default_fmincon_options() with optional overrides.
%
%   RESULT is the canonical provenance struct from CODING_STANDARDS.md §
%   "Provenance and reproducibility".
%
%   Preconditions:
%     - TARGET satisfies CLUB_IK_SPEC.md validation rules.
%     - OPTIONS.x0 is a finite vector of length n_joints*7 within bounds.
%
%   Postconditions:
%     - result.final_rmse_m is finite and non-negative.
%     - result.coefficients is within coefficient bounds.

function options = default_fmincon_options()
    options = struct();
    options.x0                  = "nominal";  % "nominal" | "random" | numeric
    options.algorithm           = "sqp";
    options.MaxIterations       = 200;
    options.MaxFunctionEvaluations = 5000;
    options.OptimalityTolerance = 1e-6;
    options.StepTolerance       = 1e-8;
    options.UseParallel         = false;
    options.Display             = "iter-detailed";
    options.cost_opts           = default_cost_options();
    options.sim_opts            = default_sim_options();
    options.output_fcn          = [];  % attached by #027 dashboard if used
    options.results_dir         = "";  % auto-named if empty
end
```

## Required tests (TDD)

- `test_fits_synthetic_swing_to_within_1mm_rmse_when_x0_equals_truth_minus_5pct`
- `test_fits_synthetic_swing_to_within_5mm_rmse_when_x0_equals_nominal`
- `test_rejects_target_with_wrong_dimensions`
- `test_rejects_x0_outside_coefficient_bounds`
- `test_returns_result_with_all_provenance_fields_per_coding_standards`
- `test_result_target_hash_matches_sha256_of_input_target`
- `test_result_git_commit_field_populated`
- `test_result_solver_field_equals_fmincon`
- `test_result_solver_options_field_contains_full_options_struct`
- `test_result_duration_s_is_positive`
- `test_x0_random_uses_generateRandomCoefficients_and_is_seed_reproducible`
- `test_uses_simulate_with_coefficients_not_a_separate_simscape_call`
- `test_output_fcn_callback_invoked_at_each_iteration_when_provided`

## DbC contract

Preconditions (`arguments` block):

- `target (1,1) struct {validators.mustHaveFields(target, ["time","butt","clubhead","club_quat","impact_idx"])}`
- `options (1,1) struct = default_fmincon_options()`
- If `options.x0` is numeric: finite vector of length `n_joints*7` within bounds.

Postconditions (`assert(...)` after solve):

- `result.final_rmse_m >= 0` and finite.
- `result.coefficients` within bounds from `generateRandomCoefficients.m`.
- All required provenance fields present (per `CODING_STANDARDS.md`).
- `result.target_hash` matches sha256 of input target.

## Acceptance Criteria

- [ ] `fit_swing_fmincon.m` calls `compute_cost` and `simulate_with_coefficients`
      directly; does not duplicate cost or forward logic.
- [ ] All listed tests pass; synthetic round-trip recovers RMSE < 1 mm.
- [ ] Result struct conforms to `CODING_STANDARDS.md` provenance schema.
- [ ] `arguments` block enforces preconditions; `assert(...)` checks postconditions.
- [ ] Result `.mat` written to `motion_matching/results/<timestamp>_<swing_id>_fmincon.mat`.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option1`, `matlab`, `tdd`, `dbc`

## Effort estimate

M (1-3 days). Most time goes into tuning bounds/scaling so `fmincon` doesn't
get stuck on the noisy Simscape gradient.
