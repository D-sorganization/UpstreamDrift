# Visualization — Option 1

This file specifies what Option 1 ships **on top of** the three required views in [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md). The shared three views (trajectory overlay, error timecourse, fit-quality summary card) are mandatory and not re-specified here.

The three Option-1-specific views below are: **OptimizationProgressDashboard**, **MultiStartParallelCoords**, and **live trajectory overlay**.

All three respect the styling rules in [shared/VISUALIZATION_SPEC.md — Styling](../shared/VISUALIZATION_SPEC.md#styling) (colour palette, fonts, no emoji, `exportgraphics`).

## File layout

```
option1_direct_optimization/visualization/
├── OptimizationProgressDashboard.m   (also exposed at the package root — handle class)
├── MultiStartParallelCoords.m        (handle class, post-fit)
├── plot_live_trajectory_overlay.m    (timer-driven, runs during a fit)
└── private/
    ├── render_dashboard_panel.m      (one render, called by timer)
    ├── push_to_history.m
    └── ...
```

The shared three views (`plot_trajectory_overlay`, `plot_error_timecourse`, `plot_fit_quality_card`) live in `motion_matching/shared/visualization/` (issue #023) and Option 1 calls them by name.

## View 1 — `OptimizationProgressDashboard`

A live, four-panel dashboard refreshed at `options.dashboard_refresh_hz` (default and cap = 5 Hz).

### Panels

```
┌─────────────────────────────────────────────────────────┐
│  Optimization Progress — TW_ProV1 — fit_swing_hybrid    │
├──────────────────────────┬──────────────────────────────┤
│  Cost vs iteration       │  |grad J| vs iteration       │
│  (semilogy)              │  (semilogy)                  │
│  red dots: stage 1 evals │  flat dashed at tol_fun      │
│  blue line: stage 2      │                              │
├──────────────────────────┼──────────────────────────────┤
│  Step size vs iteration  │  Current best theta          │
│  (semilogy)              │  (horizontal bar overlay     │
│  flat dashed at tol_x    │   on lb/ub box)              │
└──────────────────────────┴──────────────────────────────┘
```

- **Top-left — Cost vs iteration.** `semilogy` of `iter_history.fval`. Stage 1 (`surrogateopt` evals) is plotted as red dots; stage 2 (`fmincon` iterations) as a blue line. A vertical separator marks the stage boundary.
- **Top-right — `|grad J|` vs iteration.** `semilogy` of `iter_history.grad_norm` (only available when finite-difference gradient is computed). Horizontal dashed line at `options.tol_fun`. Empty in the `surrogateopt` stage.
- **Bottom-left — Step size vs iteration.** `semilogy` of `‖theta_{k+1} - theta_k‖`. Horizontal dashed line at `options.tol_x`.
- **Bottom-right — Current best `theta` overlay.** Horizontal bars: one row per coefficient (`d` rows), x-axis from `lb(i)` to `ub(i)`, a tick mark at `theta_best(i)`. Coefficients are grouped by joint (alternating row shading) and labelled. This makes it instantly visible which coefficients are pinned to a bound (a sign the bounds are wrong, or the regularizer is wrong).

### Refresh policy

Per [shared/VISUALIZATION_SPEC.md — Live updates](../shared/VISUALIZATION_SPEC.md#live-updates):

- Do **not** redraw on every iteration — that throttles the optimizer.
- The `OutputFcn` (`fmincon_output_fcn` from [INTERFACES.md](INTERFACES.md#fmincon_output_fcntheta-optimvalues-state-dashboard-schedulectx--stop)) calls `dashboard.push(row)` on every iteration. This appends to `History` only — it does not draw.
- An internal `timer` calls a private render function at `RefreshHz` Hz (default 5 Hz, max 5 Hz).
- On `state == "done"` the timer is stopped, a final draw is forced, and the figure is left open for inspection.

### `OptimizationProgressDashboard.close()`

Idempotent. Stops the timer, deletes the figure if it is still open, releases any held resources. Called from the fit's `cleanupObj`/`onCleanup` so the dashboard is never orphaned even if the fit errors.

## View 2 — `MultiStartParallelCoords`

A post-fit parallel-coordinate plot for `fit_swing_multistart` (and the global stage of `fit_swing_hybrid`).

### Layout

- X-axis: coefficient index `1..d`, grouped by joint with separator lines.
- Y-axis: coefficient value, normalized to `[0,1]` per `(lb, ub)` so all bars share a scale.
- One polyline per starting point (`N = options.multistart_n` lines).
- **Colour:** mapped from final cost (the cost at the polished optimum reached from that start). Use `parula` colormap clipped to the cost range (low cost = blue, high cost = yellow). The map is shown in a colorbar.
- The lowest-cost line is drawn last, in heavy red, on top.

### Use

This plot answers "did the multistart actually explore different basins?" If all polylines collapse to the same shape, `N` is too small or the bounds are too tight. If they fan out and only one or two converge to low cost, the surface is multimodal and `MultiStart` was justified.

### API

```matlab
fig = MultiStartParallelCoords.plot(result, opts)
```

Where `result.start_points` is the `d × N` matrix of starting points and the polish-end `theta` is plotted as a heavy red line. `opts` is the canonical viz options struct from [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md).

## View 3 — Live trajectory overlay

While the fit is running, the user sees the simulated club catching up to the measured club. This is the **single most reassuring visual** during a long fit — it tells the user the optimizer is doing something useful before any numerical metric has converged.

### Behaviour

- Every 5 seconds (`options.live_overlay_refresh_s = 5`), in a separate timer-driven background task:
  1. Take a **snapshot** of `dashboard.BestTheta` (atomic; the dashboard's `History` lock).
  2. Run `simulate_with_coefficients(theta_best)` — one full sim. **This is the only Simscape call made by the visualization.**
  3. Re-render `plot_trajectory_overlay(synthetic_result, target, viz_opts)` (the shared view).
- The overlay re-render runs in a separate worker (`parfeval`) so the optimizer is not blocked. If the previous re-render is still running the new request is dropped (no queue pile-up).
- On final convergence the live overlay is replaced by the canonical static trajectory overlay (full quality, all frames rendered).

### Cost

- One Simscape sim every 5 s. On the assumed ~5–30 s/sim cost this means at most 6–12 extra sims per minute of optimization. The user can disable it via `options.live_overlay = false`.

### API

```matlab
function handle = plot_live_trajectory_overlay(dashboard, target, options)
%PLOT_LIVE_TRAJECTORY_OVERLAY  Spawn a timer that re-renders the trajectory
%   overlay every options.live_overlay_refresh_s seconds using the current
%   dashboard.BestTheta. Returns a handle whose close() stops the timer.
%
%   Issue: #027.
end
```

## Output artifacts

At the end of every fit, regardless of solver, write to `motion_matching/results/<run_id>/`:

| File                                   | Source                                                      |
| -------------------------------------- | ----------------------------------------------------------- |
| `dashboard.png`                        | `OptimizationProgressDashboard` final render at 1.5× retina |
| `dashboard.fig`                        | Same, MATLAB-native for interactive inspection              |
| `multistart_parallel_coords.{png,fig}` | Only for MultiStart / hybrid                                |
| `trajectory_overlay.{png,fig}`         | Shared view 1, final render                                 |
| `error_timecourse.{png,fig}`           | Shared view 2, final render                                 |
| `fit_quality_card.{png,fig}`           | Shared view 3, final render                                 |
| `result.mat`                           | The full `result` struct                                    |

The run ID is `result.timestamp_utc` rendered as `yyyyMMddTHHmmssZ`.
