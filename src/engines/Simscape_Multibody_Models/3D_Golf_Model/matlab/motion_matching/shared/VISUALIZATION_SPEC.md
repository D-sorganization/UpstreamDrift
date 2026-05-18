# Visualization Specification

The user explicitly asked for **great visuals of matching quality**. This document specifies what those look like so any agent picking up a viz issue produces something consistent across the four options. The goal is to make optimizer progress and final fit quality immediately legible at a glance.

## Three required views

Every option must produce these three views at the end of every fit, plus stream the first and second live during the fit when computationally cheap.

### View 1 — Trajectory overlay (the money shot)

Two side-by-side 3D plots:

- **Left:** measured club skeleton (butt → clubhead), animated through time, plus a faint trace of the clubhead path.
- **Right:** simulated club skeleton from the current/final coefficients, same animation, same camera.

A shared playback control. The two views are tied to the same time slider so the eye sees them tracking (or drifting).

A small inset shows the per-frame error vector (drawn from the simulated to the measured clubhead) so you can see _where_ the fit is failing.

### View 2 — Error timecourse

Stacked plots versus simulation time:

```
┌────────────────────────────────────────────┐
│  Position error (mm)                       │   butt (blue) and clubhead (orange)
│                                            │   shaded ±1σ from sample-rate noise
│  ──────────────────────────────────────────│
│  Orientation error (deg)                   │   d_geo(R_sim, R_meas)
│                                            │
│  ──────────────────────────────────────────│
│  Clubhead speed (mph)                      │   measured (solid) vs simulated (dashed)
│                                            │
│  ──────────────────────────────────────────│
│  Joint torques (N·m)                       │   one trace per joint
│                                            │   highlights the impact frame
└────────────────────────────────────────────┘
```

A vertical line at the impact frame across all four panels.

### View 3 — Fit quality summary card

A single-figure summary that's safe to drop into a PR or status update:

```
┌───────────────────────────────────────────────────┐
│  Swing: TW_ProV1 (TaylorMade w/ ProV1, 2024-03-14)│
│  Solver: fmincon-sqp + multistart(8)              │
│  Iterations: 247   Wall clock: 4m 12s             │
│                                                   │
│  Final RMSE — clubhead position:   2.3 mm         │
│  Final RMSE — butt position:       1.8 mm         │
│  Final mean orientation error:     0.41°          │
│  Final clubhead speed at impact:   112 mph (meas: 111)│
│                                                   │
│  Total work (regularized):         284 J          │
│  Peak joint power:                 1.2 kW (LE)    │
│                                                   │
│  [ Trajectory overlay thumbnail ]                 │
│  [ Error timecourse thumbnail ]                   │
│                                                   │
│  Hash: 7a3f...     Branch: feat/motion-matching/o1│
└───────────────────────────────────────────────────┘
```

Saved as both `.png` (for PRs) and `.fig` (for interactive inspection).

## Optional / option-specific views

| View                             | Owner    | Purpose                                                                                |
| -------------------------------- | -------- | -------------------------------------------------------------------------------------- |
| **Optimizer progress dashboard** | Option 1 | `fmincon` iterations: cost vs iter, gradient norm, step size; updated live             |
| **Multi-start parallel-coords**  | Option 1 | One line per starting point; coloured by final cost; reveals which basins are explored |
| **Surrogate training curves**    | Option 2 | Train/val loss; learning rate; validation RMSE on held-out trials                      |
| **Surrogate-vs-truth residuals** | Option 2 | Histogram of `f_θ(coeffs) - truth_kinematics` across the validation set                |
| **Latent space projection**      | Option 3 | t-SNE/UMAP of the inverse model's latent — useful for spotting mode collapse           |
| **Round-trip residuals**         | Option 3 | `‖truth - sim(g_φ(truth))‖` per trial — the only honest measure of an inverse model    |
| **Engine-side adapter trace**    | Option 4 | Per-call latency; correctness vs MATLAB-direct on a fixed regression suite             |

## Styling

Make plots presentable, not just legible. Specifics:

- **Colour palette:** use the `colororder` lines palette (R2019b+) for joint traces; explicit hex for measured-vs-simulated comparisons (measured = `#1f77b4` blue, simulated = `#d62728` red, error = `#7f7f7f` grey).
- **Fonts:** default MATLAB sans-serif; size 11 axes, 13 titles.
- **Output:** save at 1.5× retina (DPI 200) for PNG. Always `tightInset` and `exportgraphics` rather than `saveas`.
- **No emoji in figure text.** Plain unicode only.

## File and function naming

Every option's viz code lives under `optionN_*/visualization/` with the following standard entry points:

```
<option>/visualization/
├── plot_trajectory_overlay.m         (View 1 still)
├── animate_trajectory_overlay.m      (View 1 animated, returns VideoWriter handle)
├── plot_error_timecourse.m           (View 2)
├── plot_fit_quality_card.m           (View 3)
└── (option-specific extras)
```

All four entry points take the same first two arguments — the result struct from a fit (per CODING_STANDARDS.md) and the target struct (per CLUB_IK_SPEC.md):

```matlab
function fig = plot_trajectory_overlay(result, target, opts)
    arguments
        result (1,1) struct
        target (1,1) struct
        opts (1,1) struct = default_viz_options()
    end
    % ...
end
```

This means **a different option can render Option 1's results** with no glue code — useful for cross-option comparison plots.

## Live updates

For `fmincon`/`surrogateopt`, install an `OutputFcn` that pushes the current best coefficients into a thread-safe queue, with a separate timer-driven plot refresh at ~5 Hz. Do **not** redraw on every iteration — that throttles the optimizer.

For Python (Options 2/3), use `tensorboard` if available, else live `matplotlib` with `FuncAnimation`. Same 5 Hz cap.

## Comparison across options

A small leaderboard helper, `motion_matching/shared/leaderboard.m`, scans `motion_matching/results/` for result structs and emits a comparison table:

```
swing_id   option   solver           rmse_mm   work_J   wall_s   commit
TW_ProV1   1        fmincon+ms8      2.3       284      252      7a3f
TW_ProV1   2        nn-surrogate     3.7       301        4      9b1e
TW_ProV1   3        inverse-cvae     5.1       312       <1      9b1e
TW_ProV1   4        bridge-fmincon   2.4       286      378      7a3f
```

This is what makes the four-option-parallel approach pay off — same target, same cost function, four answers, clear comparison.
