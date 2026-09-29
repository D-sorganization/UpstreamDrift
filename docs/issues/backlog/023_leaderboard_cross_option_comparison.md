# Issue: Implement leaderboard.m for Cross-Option Comparison

## Summary

Implement `motion_matching/shared/leaderboard.m`: scans `motion_matching/results/`
for result structs and emits a comparison table across the four options for the
same swing target.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"Comparison across options".
The whole point of running four options in parallel is comparing them on the
same swing with the same cost function. Without this leaderboard, the
parallel-options approach has no payoff.

## Dependencies

- #015 (`compute_cost.m`) — ensures every option's `final_rmse_m` is computed
  with the same metric.
- #018 — every option's result has consistent `sim_out` schema.
- #022 — leaderboard rows can link to per-fit quality cards.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\leaderboard.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\scan_results_directory.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\verify_result_provenance.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_leaderboard.m`

## Public API

```matlab
function [tbl, fig] = leaderboard(results_dir, opts)
    arguments
        results_dir (1,1) string = fullfile( ...
            "src","engines","Simscape_Multibody_Models","3D_Golf_Model", ...
            "matlab","motion_matching","results")
        opts (1,1) struct = default_leaderboard_options()
    end
    % Scans results_dir for *.mat files containing result structs (per
    % CODING_STANDARDS.md provenance schema). Returns a sorted table:
    %
    %   swing_id   option   solver           rmse_mm   work_J   wall_s   commit
    %   TW_ProV1   1        fmincon+ms8      2.3       284      252      7a3f
    %   TW_ProV1   2        nn-surrogate     3.7       301        4      9b1e
    %   TW_ProV1   3        inverse-cvae     5.1       312       <1      9b1e
    %   TW_ProV1   4        bridge-fmincon   2.4       286      378      7a3f
    %
    % FIG is a figure with a uitable rendering of the same data, plus a bar
    % chart of rmse_mm grouped by swing_id.
end

function opts = default_leaderboard_options()
    opts = struct();
    opts.sort_by         = "rmse_mm";  % or "wall_s" or "work_J"
    opts.filter_swing_id = "";          % empty = no filter
    opts.write_csv       = true;
    opts.csv_path        = "leaderboard.csv";
end
```

## Required tests (TDD)

- `test_leaderboard_scans_results_dir_and_returns_table_with_required_columns`
- `test_leaderboard_columns_are_swing_id_option_solver_rmse_mm_work_J_wall_s_commit`
- `test_leaderboard_sorts_by_rmse_mm_ascending_by_default`
- `test_leaderboard_filter_swing_id_returns_only_matching_rows`
- `test_leaderboard_skips_result_structs_missing_provenance_fields_with_warning`
- `test_leaderboard_converts_final_rmse_m_to_rmse_mm_in_table`
- `test_leaderboard_writes_csv_to_csv_path_when_write_csv_true`
- `test_leaderboard_returns_figure_with_uitable_and_bar_chart`
- `test_leaderboard_handles_empty_results_dir_returns_empty_table_no_error`
- `test_leaderboard_recomputes_final_rmse_m_via_compute_cost_when_recheck_true`

## DbC contract

Preconditions:

- `results_dir` exists.

Postconditions:

- Returned table has the documented columns.
- Every row's `result` struct passed `verify_result_provenance` (had all
  fields per `CODING_STANDARDS.md` §"Provenance").
- When `opts.recheck == true`, `rmse_mm` matches the value from re-running
  `compute_cost` on the saved coefficients to within `1e-6`.

## Acceptance Criteria

- [ ] `leaderboard.m` produces both a sorted table and a comparison figure.
- [ ] All listed tests pass.
- [ ] Skips/warns on malformed result structs without crashing.
- [ ] CSV export verified against fixture results.
- [ ] `arguments` block enforces preconditions.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `viz`, `tdd`

## Effort estimate

S (≤1 day).
