# Issue: Implement fit_swing_surrogateopt.m (Option 1 surrogateopt + fmincon Polish)

## Summary

Implement Option 1's hybrid global solver: `surrogateopt` for global exploration
(no gradient required), followed by `fmincon` polish from the best `surrogateopt`
incumbent. This handles cost surfaces where multistart can't find the global basin.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" and
`VISUALIZATION_SPEC.md` §"Optimizer progress dashboard". `surrogateopt` was added
in R2018b and is purpose-built for expensive black-box objectives like Simscape
forward sims; it is faster than multistart for moderately-rugged surfaces.

## Dependencies

- #018 (`simulate_with_coefficients.m`).
- #024 (`fit_swing_fmincon.m`) — used as the polishing step.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\fit_swing_surrogateopt.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\default_surrogateopt_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\tests\test_fit_swing_surrogateopt.m`

## Public API

```matlab
function result = fit_swing_surrogateopt(target, options)
%FIT_SWING_SURROGATEOPT  Hybrid surrogateopt + fmincon polish swing fit.
%
%   result = FIT_SWING_SURROGATEOPT(TARGET, OPTIONS) runs MATLAB's
%   surrogateopt for OPTIONS.MaxFunctionEvaluations evaluations, then
%   warm-starts fit_swing_fmincon from the surrogateopt incumbent for the
%   final polish.

function options = default_surrogateopt_options()
    options = struct();
    options.MaxFunctionEvaluations  = 500;
    options.MinSurrogatePoints      = 50;
    options.MinSampleDistance       = 1e-3;
    options.UseParallel             = true;
    options.PlotFcn                 = "surrogateoptplot";
    options.checkpoint_file         = "";   % auto-named if empty
    options.skip_polish             = false; % set true to compare without fmincon polish
    options.fmincon_options         = default_fmincon_options();
    options.cost_opts               = default_cost_options();
    options.sim_opts                = default_sim_options();
    options.results_dir             = "";
end
```

## Required tests (TDD)

- `test_surrogateopt_returns_result_with_global_phase_and_polish_phase_telemetry`
- `test_surrogateopt_polished_rmse_is_less_than_or_equal_to_global_rmse`
- `test_skip_polish_true_returns_global_phase_result_only`
- `test_global_phase_records_all_evaluated_coefficients_in_history`
- `test_uses_min_surrogate_points_50_by_default`
- `test_use_parallel_true_engages_parsim_in_global_phase`
- `test_checkpoint_file_resumes_after_interruption`
- `test_fits_synthetic_swing_to_within_2mm_rmse_with_500_evaluations`
- `test_outperforms_multistart_8_on_handcrafted_multimodal_synthetic_target`
- `test_rejects_target_with_wrong_dimensions`
- `test_polish_phase_uses_fit_swing_fmincon_not_a_separate_fmincon_call`

## DbC contract

Preconditions:

- `target` per `CLUB_IK_SPEC.md`.
- `options.MaxFunctionEvaluations >= options.MinSurrogatePoints`.

Postconditions:

- When `skip_polish == false`, `result.final_rmse_m <= result.global_phase.final_rmse_m`.
- All required provenance fields present.
- `result.solver` == "surrogateopt+fmincon" (or "surrogateopt" when `skip_polish==true`).

## Acceptance Criteria

- [ ] `fit_swing_surrogateopt.m` runs the global phase, then warm-starts `fit_swing_fmincon`.
- [ ] All listed tests pass.
- [ ] `arguments` block enforces preconditions; `assert(...)` checks postconditions.
- [ ] Checkpointing verified by simulating an interruption mid-run.
- [ ] Result `.mat` written to `motion_matching/results/<timestamp>_<swing_id>_surrogateopt.mat`.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option1`, `matlab`, `tdd`, `dbc`

## Effort estimate

M (1-3 days).
