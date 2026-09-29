# Issue: Implement Optimization Progress Dashboard (Option 1 Live Viz Handle Class)

## Summary

Implement a MATLAB `handle` class that subscribes to optimizer iterations via
`OutputFcn` and refreshes a live dashboard at 5 Hz: cost vs iteration, gradient
norm, step size, and a thumbnail trajectory overlay. The class is generic across
`fmincon`, `MultiStart`, and `surrogateopt`.

## Motivation

See `motion_matching/shared/VISUALIZATION_SPEC.md` §"Optimizer progress dashboard"
and §"Live updates". The user wants visibility into long-running fits. The
critical design constraint is: **do not redraw on every iteration** — that
throttles `fmincon`. Use a thread-safe queue and a timer-driven refresh.

## Dependencies

- #020 (`plot_trajectory_overlay.m`) — used as the thumbnail.
- #024, #025, #026 — three optimizers whose `OutputFcn` this class hooks.

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\OptimizationProgressDashboard.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\private\IterationQueue.m`
- `C:\Users\diete\Repositories\UpstreamDrift\src\engines\Simscape_Multibody_Models\3D_Golf_Model\matlab\motion_matching\option1_direct_optimization\tests\test_OptimizationProgressDashboard.m`

## Public API

```matlab
classdef OptimizationProgressDashboard < handle
%OPTIMIZATIONPROGRESSDASHBOARD  Live-updating dashboard for fmincon/surrogateopt.
%
%   dash = OptimizationProgressDashboard(target, opts) creates the figure and
%   starts a timer that refreshes panels at opts.refresh_hz (default 5).
%
%   stop = dash.outputFcn();  % returns a function handle suitable for
%                              % options.OutputFcn = stop;
%   dash.close();             % stops the timer and closes the figure.

    properties (SetAccess = private)
        Figure
        Axes
        Target
        Options
        IterationQueue
        RefreshTimer
        IterationCount
    end

    methods
        function obj = OptimizationProgressDashboard(target, opts)
            arguments
                target (1,1) struct
                opts (1,1) struct = default_dashboard_options()
            end
        end

        function fcn = outputFcn(obj)
            % Returns a function handle compatible with fmincon's OutputFcn
            % API: stop = fcn(x, optimValues, state)
        end

        function close(obj)
            % Stops the timer and closes the figure.
        end
    end
end

function opts = default_dashboard_options()
    opts = struct();
    opts.refresh_hz   = 5;
    opts.show_thumbnail_trajectory = true;
    opts.history_limit = 1000;  % cap on retained iterations to avoid memory growth
end
```

## Required tests (TDD)

- `test_dashboard_constructs_figure_with_four_panels_cost_grad_step_thumbnail`
- `test_dashboard_outputFcn_pushes_iteration_into_queue_without_blocking_optimizer`
- `test_dashboard_refresh_timer_runs_at_5_hz_by_default`
- `test_dashboard_does_not_redraw_on_every_iteration`
- `test_dashboard_history_limit_bounds_memory_when_exceeded`
- `test_dashboard_outputFcn_compatible_with_fmincon_signature`
- `test_dashboard_outputFcn_compatible_with_multistart_signature`
- `test_dashboard_outputFcn_compatible_with_surrogateopt_signature`
- `test_dashboard_close_stops_timer_and_releases_figure`
- `test_dashboard_thumbnail_panel_calls_plot_trajectory_overlay_with_current_best`
- `test_dashboard_runs_in_headless_mode_without_visible_figure`

## DbC contract

Preconditions:

- `target` per `CLUB_IK_SPEC.md`.
- `opts.refresh_hz > 0`.

Postconditions:

- After construction: `Figure` is valid; `RefreshTimer` is running.
- After `close()`: `RefreshTimer` is stopped; figure is invalid.
- `outputFcn` returns a function handle that is non-blocking.

## Acceptance Criteria

- [ ] `OptimizationProgressDashboard` is a `handle` class (not a value class).
- [ ] All listed tests pass.
- [ ] Verified to work with all three Option 1 solvers (#024, #025, #026).
- [ ] Refresh demonstrably bounded at 5 Hz; iteration push is O(1).
- [ ] `close()` is idempotent and safe to call after the figure is already gone.
- [ ] No file exceeds 1200 lines.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `option1`, `matlab`, `viz`, `tdd`

## Effort estimate

M (1-3 days). Timer + thread-safe-queue plumbing in MATLAB has known footguns.
