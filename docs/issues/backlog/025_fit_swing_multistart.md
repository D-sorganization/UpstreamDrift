# Issue: Implement fit_swing_multistart.m (Option 1 MultiStart + parsim/parfor)

## Summary

Wrap `fit_swing_fmincon` in a `MultiStart` driver with `parsim`/`parfor`
parallelism: launch N starting points in parallel, return the best fit, and
record per-start telemetry for the parallel-coords visualization in
`VISUALIZATION_SPEC.md`.

## Motivation

See `motion_matching/README.md` "Why four options in parallel" — Option 1's
multistart is what produces robust fits on real swings whose cost surface has
many local minima. The parallelism is necessary because each Simscape forward
sim is in the seconds-to-tens-of-seconds range; serial multistart is impractical.

## Dependencies

- #024 (`fit_swing_fmincon.m`) — single-start solver wrapped here.
- #018 (`simulate_with_coefficients.m`) with `parallel_safe=true` mode tested.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\fit_swing_multistart.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\default_multistart_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\private\sample_starting_points.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\private\plot_multistart_parallel_coords.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\tests\test_fit_swing_multistart.m`

## Public API

```matlab
function result = fit_swing_multistart(target, options)
%FIT_SWING_MULTISTART  Parallel multi-start fmincon-sqp swing fit.
%
%   result = FIT_SWING_MULTISTART(TARGET, OPTIONS) launches OPTIONS.n_starts
%   parallel runs of fit_swing_fmincon and returns the best result, augmented
%   with .all_starts (cell array of per-start results) for parallel-coords plotting.

function options = default_multistart_options()
    options = struct();
    options.n_starts            = 8;
    options.starting_strategy   = "latin_hypercube";  % "random" | "latin_hypercube" | "sobol"
    options.parallel_pool_size  = "auto";               % "auto" | numeric
    options.parallel_method     = "parsim";             % "parsim" | "parfor"
    options.seed                = 42;
    options.fmincon_options     = default_fmincon_options();
    options.results_dir         = "";
end
```

## Required tests (TDD)

- `test_multistart_returns_result_with_all_starts_cell_array_of_length_n_starts`
- `test_multistart_returned_best_has_minimum_final_rmse_m_across_all_starts`
- `test_starting_strategy_latin_hypercube_covers_bound_box`
- `test_starting_strategy_random_is_seed_reproducible`
- `test_starting_strategy_sobol_covers_bound_box_with_lower_discrepancy_than_random`
- `test_parsim_method_runs_n_starts_in_parallel_with_simulink_pool`
- `test_parfor_method_runs_n_starts_in_parallel_with_local_pool`
- `test_handles_failed_start_with_warning_does_not_crash_overall_run`
- `test_fits_synthetic_swing_to_within_0_5mm_rmse_with_8_starts`
- `test_result_includes_per_start_wall_times_for_load_balancing_diagnostics`
- `test_plot_multistart_parallel_coords_renders_one_line_per_start_colored_by_final_cost`

## DbC contract

Preconditions:

- `target` per `CLUB_IK_SPEC.md`.
- `options.n_starts >= 1`.
- `options.parallel_method in {"parsim","parfor"}`.

Postconditions:

- `result.final_rmse_m == min([result.all_starts{:}.final_rmse_m])`.
- `length(result.all_starts) == options.n_starts`.
- All starting points within coefficient bounds.

## Acceptance Criteria

- [ ] `fit_swing_multistart.m` parallelises across N starts and returns the best.
- [ ] All listed tests pass.
- [ ] Parallel-coords figure renders correctly for `result.all_starts`.
- [ ] Both `parsim` and `parfor` methods supported and verified.
- [ ] `arguments` block enforces preconditions; `assert(...)` checks postconditions.
- [ ] Result `.mat` written to `motion_matching/results/<timestamp>_<swing_id>_multistart.mat`.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option1`, `matlab`, `tdd`, `dbc`

## Effort estimate

M (1-3 days). The unknowns are the parsim setup quirks for the Simscape model.
