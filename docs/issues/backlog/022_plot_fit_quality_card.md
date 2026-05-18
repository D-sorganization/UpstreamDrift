# Issue: Implement plot_fit_quality_card.m (View 3)

## Summary

Implement View 3 from `VISUALIZATION_SPEC.md`: a single-figure fit-quality
summary card that's safe to drop into a PR description — final RMSEs, total
work, peak power, plus thumbnails of Views 1 and 2. Saved as both `.png` and `.fig`.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"View 3 — Fit quality summary
card". This is the artefact reviewers look at first; it must be terse,
self-contained, and reproducible across all four options.

## Dependencies

- #020 (View 1) — generates trajectory-overlay thumbnail.
- #021 (View 2) — generates error-timecourse thumbnail.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\plot_fit_quality_card.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\format_quality_metrics.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_plot_fit_quality_card.m`

## Public API

```matlab
function fig = plot_fit_quality_card(result, target, opts)
    arguments
        result (1,1) struct
        target (1,1) struct
        opts (1,1) struct = default_viz_options()
    end
    % Single-figure summary:
    %   - Header: swing id, solver, iterations, wall clock.
    %   - Metrics block: clubhead RMSE (mm), butt RMSE (mm), mean orientation
    %     error (deg), clubhead speed at impact (mph, sim vs meas).
    %   - Regularizer block: total work (J), peak joint power (kW + joint name).
    %   - Two thumbnails: View 1 still + View 2 stack.
    %   - Footer: target hash (short), branch name, git commit (short).
    %
    % Saves both .png (DPI 200) and .fig next to opts.output_dir/<swing>.png/.fig
    % when opts.save_to_disk is true.
end
```

## Required tests (TDD)

- `test_card_renders_into_single_figure_with_header_metrics_thumbnails_footer`
- `test_card_clubhead_rmse_displayed_in_mm_to_one_decimal`
- `test_card_butt_rmse_displayed_in_mm_to_one_decimal`
- `test_card_orientation_error_displayed_in_degrees_to_two_decimals`
- `test_card_clubhead_speed_at_impact_shows_both_meas_and_sim_in_mph`
- `test_card_total_work_displayed_in_joules`
- `test_card_peak_power_displayed_in_kW_with_joint_name`
- `test_card_writes_png_at_dpi_200_when_save_to_disk_true`
- `test_card_writes_fig_alongside_png_when_save_to_disk_true`
- `test_card_includes_short_git_commit_in_footer`
- `test_card_includes_short_target_hash_in_footer`
- `test_card_includes_branch_name_in_footer`
- `test_card_renders_correctly_for_each_option_result_schema`
- `test_card_runs_in_headless_mode_with_visible_off`

## DbC contract

Preconditions:

- `result` has fields `final_rmse_m`, `final_total_work_J`, `solver`,
  `solver_options`, `target_hash`, `git_commit`, `duration_s`,
  `timestamp_utc` (per `CODING_STANDARDS.md` §"Provenance and reproducibility").
- `target` conforms to `CLUB_IK_SPEC.md`.

Postconditions:

- `fig` is a valid figure handle.
- When `opts.save_to_disk == true`, the `.png` and `.fig` files exist on disk.

## Acceptance Criteria

- [ ] Function implemented per spec; uses Views 1 and 2 as inset thumbnails.
- [ ] All listed tests pass (use `tempdir` for save-to-disk tests).
- [ ] Output PNG renders at DPI 200; figure uses `tightInset`.
- [ ] No emoji in figure text.
- [ ] `arguments` blocks enforce preconditions.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `viz`, `tdd`

## Effort estimate

S (≤1 day) once #020 and #021 land.
