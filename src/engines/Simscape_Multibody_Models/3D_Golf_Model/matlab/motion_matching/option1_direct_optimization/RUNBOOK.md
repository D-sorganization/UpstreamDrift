# Runbook — Option 1

How a human (or a follow-up agent) actually runs Option 1. Literal MATLAB command sequences for the four standard tasks.

These commands assume the working directory is the repo root (`UpstreamDrift/`). Adjust paths if you've `cd`-ed into a subfolder. All paths are absolute or repo-relative.

## 0. One-time setup (per machine)

In MATLAB:

```matlab
% Add motion_matching paths (recursive)
mm_root = fullfile(pwd, "src", "engines", "Simscape_Multibody_Models", ...
                   "3D_Golf_Model", "matlab", "motion_matching");
addpath(genpath(mm_root));

% Add the simulator paths
mdl_root = fullfile(pwd, "src", "engines", "Simscape_Multibody_Models", ...
                    "3D_Golf_Model", "matlab", "src");
addpath(genpath(mdl_root));

% Confirm toolbox availability
ver_info = ver;
required = ["Optimization Toolbox", "Global Optimization Toolbox", ...
            "Simulink", "Simscape Multibody"];
have    = string({ver_info.Name});
missing = setdiff(required, have);
if ~isempty(missing)
    error("Missing toolboxes: %s", strjoin(missing, ", "));
end

% Optional: warm a parallel pool
if license("test","Distrib_Computing_Toolbox") && isempty(gcp("nocreate"))
    parpool("Processes", feature("numcores"));
end
```

Save the above as `motion_matching/option1_direct_optimization/private/option1_setup.m` and run it once at the start of each session.

## 1. Fresh fit on a synthetic target (the smoke test)

This is the test that **must work** before you trust any other run. If this fails, the optimizer or the simulator is broken — not the data.

```matlab
option1_setup();

% Build a synthetic target from a known coefficient vector
theta_truth = generateRandomCoefficients(numel(getPolynomialParameterInfo().joint_names) * 7);
target      = synthesize_target_from_coefficients(theta_truth);

% Default options, no regularizer for the oracle test
opts                  = default_option1_options();
opts.cost.lambda      = 0;
opts.dashboard        = true;
opts.verbosity        = "Normal";

% Fit — uses the recommended hybrid solver
result = fit_swing_hybrid(target, opts);

% Inspect
fprintf("Final RMSE:        %.3f mm\n", result.final_rmse_m * 1e3);
fprintf("Total work:        %.1f J\n",  result.final_total_work_J);
fprintf("Wall clock:        %.1f s\n",  result.duration_s);
fprintf("Coefficient err:   %.2f%%\n",  ...
        100 * norm(result.coefficients - theta_truth) / norm(theta_truth));

% Render the canonical three views
plot_trajectory_overlay(result, target);
plot_error_timecourse(result, target);
plot_fit_quality_card(result, target);
```

**Expected:** RMSE < 1 mm, wall clock 2–10 minutes depending on `n_joints`.

## 2. Fresh fit on TW_ProV1 from the Excel file

```matlab
option1_setup();

xlsx_path = fullfile(pwd, "src", "apps", "golf_gui", ...
                     "Motion Capture Plotter", ...
                     "Wiffle_ProV1_club_3D_data.xlsx");

target = load_club_target_excel(xlsx_path, "TW_ProV1", default_align_options());

opts                       = default_option1_options();
opts.cost.lambda           = 1e-4;       % regularizer ON for real swings
opts.cost.w_anchor_impact  = 10.0;
opts.solver                = "hybrid";   % surrogateopt -> fmincon
opts.multistart_n          = 8;
opts.max_function_evals    = 500;
opts.use_parallel          = true;
opts.dashboard             = true;

result = fit_swing_hybrid(target, opts);

% Save and visualize
run_dir = fullfile("motion_matching", "results", ...
                   datetime("now","Format","yyyyMMdd'T'HHmmss'Z'", ...
                            "TimeZone","UTC") + "_TW_ProV1_hybrid");
mkdir(run_dir);
save(fullfile(run_dir, "result.mat"), "result", "target", "opts");
exportgraphics(plot_trajectory_overlay(result, target),     fullfile(run_dir, "trajectory_overlay.png"),  "Resolution", 200);
exportgraphics(plot_error_timecourse(result, target),       fullfile(run_dir, "error_timecourse.png"),    "Resolution", 200);
exportgraphics(plot_fit_quality_card(result, target),       fullfile(run_dir, "fit_quality_card.png"),    "Resolution", 200);
```

**Expected wall clock:** ~5–15 minutes on a 16-core workstation with `parsim` enabled.

**Tuning notes:**

- If RMSE plateaus above ~5 mm, increase `opts.multistart_n` to 16 or extend `opts.max_function_evals`.
- If the optimized swing has implausibly high peak power (> 2 kW per joint), increase `opts.cost.lambda` to `5e-4` and re-run.
- If the orientation error is large but position error is small, increase `opts.cost.w_orientation` from 0.1 to 0.5.

## 3. Resuming a checkpointed fit

`fit_swing_*` consults the cache by default. Re-running the **same** target + options returns the prior result instantly:

```matlab
% First run (cold)
result1 = fit_swing_hybrid(target, opts);    % ~10 minutes

% Second run (cache hit)
result2 = fit_swing_hybrid(target, opts);    % < 1 second
assert(result2.cache_hit);
assert(isequal(result1.coefficients, result2.coefficients));
```

To **continue** an in-progress fit (warm-start `fmincon` from the previous best `theta`):

```matlab
% Inspect the previous result
prev = load(fullfile("motion_matching","results","20260505T120000Z_TW_ProV1_hybrid","result.mat"));

% Override the cold-start strategy: use the prior theta as the warm start
opts = default_option1_options();
opts.cold_start_strategy = "zeros";          % skip the dataset/sobol seed
opts.warm_start_theta    = prev.result.coefficients;   % new option, optional
opts.solver              = "fmincon";        % skip the global stage
opts.use_cache           = false;            % don't return the prior

result_continued = fit_swing_fmincon(target, opts);
```

The `options.warm_start_theta` field is allowed but not required by the validator (per [INTERFACES.md](INTERFACES.md#default_option1_options---options) — it is an optional override; a missing value triggers the default cold-start strategy).

## 4. Comparing two solvers on the same target

```matlab
option1_setup();

% Same target, same regularizer, two solvers
target = load_club_target_excel(xlsx_path, "TW_ProV1", default_align_options());
opts   = default_option1_options();
opts.cost.lambda = 1e-4;
opts.dashboard   = false;
opts.use_parallel = true;

r_fmincon  = fit_swing_fmincon(target, opts);
r_multi    = fit_swing_multistart(target, opts);
r_surr     = fit_swing_surrogateopt(target, opts);
r_hybrid   = fit_swing_hybrid(target, opts);

% Tabulate
T = table(...
    ["fmincon";"multistart";"surrogateopt";"hybrid"], ...
    [r_fmincon.final_rmse_m; r_multi.final_rmse_m; r_surr.final_rmse_m; r_hybrid.final_rmse_m] * 1e3, ...
    [r_fmincon.final_total_work_J; r_multi.final_total_work_J; r_surr.final_total_work_J; r_hybrid.final_total_work_J], ...
    [r_fmincon.duration_s; r_multi.duration_s; r_surr.duration_s; r_hybrid.duration_s], ...
    'VariableNames', ["solver", "rmse_mm", "total_work_J", "wall_s"]);
disp(T);

% Cross-render: every option's results plot through the same shared viz
figure("Name","Comparison");
tiledlayout(2,2);
nexttile; plot_trajectory_overlay(r_fmincon,  target);  title("fmincon");
nexttile; plot_trajectory_overlay(r_multi,    target);  title("multistart");
nexttile; plot_trajectory_overlay(r_surr,     target);  title("surrogateopt");
nexttile; plot_trajectory_overlay(r_hybrid,   target);  title("hybrid");
```

The shared leaderboard (`motion_matching/shared/leaderboard.m`, issue #023) scans `motion_matching/results/` and produces this table automatically across **all four options** when those exist.

## 5. Troubleshooting

| Symptom                                 | Likely cause                                   | Fix                                                                                                                                                          |
| --------------------------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `error("validator:unknownField",...)`   | Typo in `opts` field                           | Inspect `default_option1_options()`; bad fields must error per [TESTING.md](TESTING.md#test_options_struct_contract).                                        |
| RMSE never goes below ~5 cm             | Time alignment off                             | Inspect `target.time` and `target.impact_idx`; re-load with `time_alignment = "address"`.                                                                    |
| One coefficient pinned to bound         | Bounds too tight, or regularizer wrong         | Check the bottom-right panel of the `OptimizationProgressDashboard`.                                                                                         |
| `fmincon` stalls at the start           | `w_anchor_impact` too low; wrong basin         | Increase `opts.cost.w_anchor_impact` to 50 for the first 25% of iterations (see [APPROACH.md — Impact-anchor schedule](APPROACH.md#impact-anchor-schedule)). |
| `surrogateopt` runs but doesn't improve | `d` too high for `surrogateopt`                | Switch to `fit_swing_hybrid` with `global_stage = "particleswarm"`.                                                                                          |
| Sim diverges (penalty path hit)         | `theta` outside the integrator's stable region | Check `result.iter_history.fval == options.penalty_on_sim_failure`; widen bounds or restart.                                                                 |
| Cache stale                             | New simulator commit                           | Delete `motion_matching/results/cache/`. Cache is keyed by git commit but if you committed without re-running, manually clear it.                            |

## 6. Verification before opening a PR

```matlab
% Run the unit suite
results = runtests("motion_matching/option1_direct_optimization/tests", ...
                   "IncludeSubfolders", true);
disp(table(results));
assert(all([results.Passed]), "tests must pass before PR");
```

```bash
# From the repo root
python3 scripts/ci/check_file_size_budget.py
python3 -m ruff check src/  # only if you touched Python
```

Per [CLAUDE.md — CI Requirements](../../../../../../CLAUDE.md#ci-requirements-all-must-pass).
