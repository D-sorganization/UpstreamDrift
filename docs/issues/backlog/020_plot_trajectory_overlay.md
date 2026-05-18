# Issue: Implement plot_trajectory_overlay.m and animate_trajectory_overlay.m (View 1)

## Summary

Implement View 1 from `VISUALIZATION_SPEC.md`: two side-by-side 3D plots (measured
vs simulated club skeleton) with a shared time slider, plus an animated variant
that returns a `VideoWriter` handle.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"View 1 — Trajectory overlay".
The user explicitly asked for "great visuals of matching quality". This is the
money-shot view that makes optimizer drift legible at a glance, and it must work
for **any** option's `result` struct so a different option can render Option 1's
output with no glue code.

## Dependencies

- #013 (`load_club_target_excel.m`) — provides the `target` schema.
- #018 (`simulate_with_coefficients.m`) — provides the `sim_out` schema embedded
  in the `result` struct.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\plot_trajectory_overlay.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\animate_trajectory_overlay.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\default_viz_options.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\private\draw_club_skeleton.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_plot_trajectory_overlay.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\shared\tests\test_animate_trajectory_overlay.m`

## Public API

Verbatim from `VISUALIZATION_SPEC.md`:

```matlab
function fig = plot_trajectory_overlay(result, target, opts)
    arguments
        result (1,1) struct
        target (1,1) struct
        opts (1,1) struct = default_viz_options()
    end
    % Side-by-side 3D plots:
    %   Left  — measured club skeleton from target, full clubhead trace.
    %   Right — simulated club skeleton from result.sim_out, full clubhead trace.
    % Inset shows per-frame error vector (sim → meas) at the impact frame.
end

function vh = animate_trajectory_overlay(result, target, opts)
    arguments
        result (1,1) struct
        target (1,1) struct
        opts (1,1) struct = default_viz_options()
    end
    % Returns a VideoWriter handle. Caller is responsible for vh.close().
    % Tied time slider drives both subplots from the same time index.
end

function opts = default_viz_options()
    opts = struct();
    opts.color_measured   = "#1f77b4";
    opts.color_simulated  = "#d62728";
    opts.color_error      = "#7f7f7f";
    opts.dpi              = 200;
    opts.fontsize_axes    = 11;
    opts.fontsize_title   = 13;
    opts.video_fps        = 30;
    opts.video_format     = "MPEG-4";
    opts.show_impact_marker = true;
    opts.tight_inset      = true;
end
```

## Required tests (TDD)

- `test_plot_returns_figure_handle_with_two_axes_arranged_side_by_side`
- `test_plot_uses_measured_color_1f77b4_and_simulated_color_d62728`
- `test_plot_axes_share_camera_view_after_link_axes_call`
- `test_plot_inset_shows_error_vector_at_impact_frame`
- `test_plot_runs_in_headless_mode_using_visible_off_for_ci`
- `test_animate_returns_videowriter_handle`
- `test_animate_writes_at_30_fps_by_default`
- `test_animate_uses_mpeg4_format_by_default`
- `test_animate_time_slider_drives_both_subplots_consistently`
- `test_plot_renders_correctly_for_option1_result_struct`
- `test_plot_renders_correctly_for_option2_result_struct`
- `test_plot_rejects_target_missing_butt_field_with_validator_error`

## DbC contract

Preconditions (`arguments` block):

- `result` has fields `sim_out` (per #018 schema) and `coefficients`.
- `target` conforms to `CLUB_IK_SPEC.md` schema.
- `opts` is a `default_viz_options()` struct (or override).

Postconditions:

- `fig` is a valid figure handle (or for `animate_*`, a valid `VideoWriter` handle).
- Two axes children present in `fig.Children` for `plot_trajectory_overlay`.

## Acceptance Criteria

- [ ] Both functions implemented per `VISUALIZATION_SPEC.md` styling rules.
- [ ] All listed tests pass (use `Visible='off'` for CI).
- [ ] Render verified for at least two different option-result schemas (use a
      stub Option-2-style result in tests).
- [ ] Output uses `exportgraphics` not `saveas`; `tightInset` applied.
- [ ] No emoji in figure text.
- [ ] `arguments` blocks enforce preconditions.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `matlab`, `viz`, `tdd`

## Effort estimate

M (1-3 days). The animation timing/sync is the time sink.
