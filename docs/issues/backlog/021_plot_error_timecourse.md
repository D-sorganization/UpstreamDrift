# Issue: Implement plot_error_timecourse.m (View 2)

## Summary

Implement View 2 from `VISUALIZATION_SPEC.md`: a four-panel stacked plot of
position error (mm), orientation error (deg), clubhead speed (mph), and joint
torques (N·m) versus simulation time, with a vertical impact line across all panels.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"View 2 — Error timecourse".
This is the diagnostic view that says **where in the swing** the fit is failing
— for example, at impact vs late follow-through. Indispensable when debugging
optimizer drift.

## Dependencies

- #013 — `target` struct schema.
- #018 — `sim_out` struct embedded in `result`.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\plot_error_timecourse.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\compute_per_frame_errors.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_plot_error_timecourse.m`

## Public API

```matlab
function fig = plot_error_timecourse(result, target, opts)
    arguments
        result (1,1) struct
        target (1,1) struct
        opts (1,1) struct = default_viz_options()
    end
    % Four stacked panels vs simulation time:
    %   1) Position error (mm) — butt (blue) and clubhead (orange), shaded ±1σ.
    %   2) Orientation error (deg) — d_geo(R_sim, R_meas) per frame.
    %   3) Clubhead speed (mph) — measured (solid) vs simulated (dashed).
    %   4) Joint torques (N*m) — one trace per joint, colororder lines palette.
    % Vertical line at impact frame across all panels.
end
```

## Required tests (TDD)

- `test_returns_figure_with_exactly_four_stacked_axes`
- `test_panel_1_plots_butt_position_error_in_millimetres`
- `test_panel_1_plots_clubhead_position_error_in_millimetres`
- `test_panel_1_uses_blue_for_butt_orange_for_clubhead`
- `test_panel_2_plots_orientation_error_in_degrees_using_d_geo_with_abs`
- `test_panel_3_plots_clubhead_speed_in_mph_with_solid_meas_dashed_sim`
- `test_panel_4_plots_one_torque_trace_per_joint_using_colororder`
- `test_vertical_impact_line_present_on_all_four_panels`
- `test_panel_1_shaded_band_uses_one_sigma_from_sample_rate_noise`
- `test_compute_per_frame_errors_returns_position_orientation_per_frame_arrays`
- `test_compute_per_frame_errors_orientation_uses_2_acos_abs_dot_product`
- `test_plot_runs_in_headless_mode_using_visible_off_for_ci`
- `test_plot_rejects_target_missing_clubhead_field_with_validator_error`

## DbC contract

Preconditions (`arguments` block + `validators.mustHaveFields`):

- `result.sim_out` has all fields per #018.
- `target` conforms to `CLUB_IK_SPEC.md`.

Postconditions:

- `fig` has exactly four axes.
- Per-frame error arrays are non-negative.

## Acceptance Criteria

- [ ] Function implemented per `VISUALIZATION_SPEC.md` styling rules.
- [ ] All listed tests pass.
- [ ] Output uses `exportgraphics` with `dpi=200`; `tightInset` applied.
- [ ] Vertical impact line drawn via `xline` across all subplots.
- [ ] `compute_per_frame_errors.m` shared helper used by both this view and the
      fit-quality card (#022) to avoid duplicated math.
- [ ] No emoji in figure text.
- [ ] `arguments` blocks enforce preconditions.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `viz`, `tdd`

## Effort estimate

S (≤1 day) once #013 and #018 land.
