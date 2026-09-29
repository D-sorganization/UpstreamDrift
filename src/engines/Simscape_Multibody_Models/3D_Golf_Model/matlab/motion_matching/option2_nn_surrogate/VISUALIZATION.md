# Option 2 — Visualization

The three required views (trajectory overlay, error timecourse, fit quality card) come from [shared/VISUALIZATION_SPEC.md § Three required views](../shared/VISUALIZATION_SPEC.md#three-required-views) and are produced by the shared visualization entry points. Option 2 ships **option-specific** views on top of those.

All option-specific viz lives under `option2_nn_surrogate/visualization/` and follows the styling rules in [shared/VISUALIZATION_SPEC.md § Styling](../shared/VISUALIZATION_SPEC.md#styling).

## V1 — Training curves

Live during training; saved as `.png` and `.fig`-equivalent (matplotlib `pickle`) at end.

Stacked panels vs training step:

```
┌──────────────────────────────────────────────────┐
│  Training loss (total)                           │
│  ─────────────────────────────────────────────── │
│  Validation loss (total)                         │
│  ─────────────────────────────────────────────── │
│  Validation RMSE (mm) — butt, clubhead           │
│  ─────────────────────────────────────────────── │
│  Validation orientation error (deg)              │
│  ─────────────────────────────────────────────── │
│  Learning rate                                   │
└──────────────────────────────────────────────────┘
```

A vertical dashed line at the best-on-val step (the saved `best.pt`).

Implementation:

- **Default:** TensorBoard. Scalars under `train/loss`, `train/lr`, `val/loss`, `val/rmse_butt_mm`, `val/rmse_clubhead_mm`, `val/orient_deg`.
- **Fallback (no TensorBoard installed):** matplotlib + a JSON-Lines log at `models/<run_id>/train_log.jsonl`. A separate script `plot_training_curves.py` reads the JSONL and emits a 5-panel PNG.
- Update cadence: every `eval_every` steps (default 1000), capped at 5 Hz.

Entry point: `option2_nn_surrogate/visualization/plot_training_curves.py`.

## V2 — Validation residual histogram

Per-component, after best-on-val checkpoint is selected, evaluated over the full **test** split.

Three side-by-side histograms:

```
┌────────────┬────────────┬───────────────┐
│  r_butt    │  r_clubhead│  q_club       │
│  (mm)      │  (mm)      │  (deg)        │
│            │            │               │
│   median   │   median   │   median      │
│   p95      │   p95      │   p95         │
└────────────┴────────────┴───────────────┘
```

Vertical lines at median and p95 with the numeric value annotated. The legend cites the test-split size and the dataset run id.

This is the headline plot for "is the surrogate good enough" — it goes in the PR description on issue #028.

Entry point: `option2_nn_surrogate/visualization/plot_residual_histogram.py`.

## V3 — Surrogate-vs-truth side-by-side animation (held-out trial)

The same layout as the shared **View 1** trajectory overlay, but the "left" panel is the **dataset-truth** trajectory (from Simscape, stored in the parquet) and the "right" panel is the **surrogate prediction** for the same `trial_id`. This is **not** the measured-vs-fitted view — it's a surrogate sanity check.

Pick the held-out trial whose clubhead RMSE is the **median** in the test set, not the best — median is honest, best is a glamour shot.

Entry point: `option2_nn_surrogate/visualization/animate_surrogate_vs_truth.py`. Outputs MP4 + a static-frame PNG (impact frame).

## V4 — Inversion progress

For each fit (each `fit_swing_via_surrogate` call), produce:

```
┌──────────────────────────────────────────────────┐
│  Adam loss vs iteration (one line per restart)   │
│  thicker line = best-so-far                       │
│  ─────────────────────────────────────────────── │
│  Best-so-far surrogate RMSE (mm) vs iteration    │
│  ─────────────────────────────────────────────── │
│  Coefficient distance from start (L2) vs iter     │
│   one trace per restart, coloured by final loss  │
└──────────────────────────────────────────────────┘
```

Plus a small inset showing the trajectory overlay at three iterations: `0`, `iters/2`, `iters` — the eye sees the predicted clubhead path morphing toward the target. The shared `plot_trajectory_overlay` is reused for the inset; passing `result.intermediate_states` (recorded by the inversion loop when `opts.record_progress=True`).

Entry point: `option2_nn_surrogate/visualization/plot_inversion_progress.py`.

`opts.record_progress` is opt-in because it costs O(K × max_iters × N) memory; turn on only when debugging or producing publication figures.

## V5 — (Optional) Surrogate gradient field slice

Pick two coefficient axes (e.g., `joint_0_A`, `joint_0_B`), hold the others at their nominal, and plot the surrogate's clubhead-RMSE-against-target as a heatmap over a 50×50 grid in `[bounds_low, bounds_high]^2`. Overlay the Adam path from V4. This is the "is the loss surface well-shaped?" diagnostic; if it's full of axis-aligned ridges, FiLM-MLP is broken and we should retry with the 1D-CNN.

Entry point: `option2_nn_surrogate/visualization/plot_loss_surface_2d.py`. Marked **optional**; ship if there's bandwidth on issue #031, otherwise punt to v2.

## Cross-references and reuse

- All four required-shared views (trajectory overlay, error timecourse, fit quality card, leaderboard) are produced by the **shared** entry points called with Option 2's `result` struct as input. No Option-2-specific glue needed.
- The Option 1 `OptimizationProgressDashboard` and Option 2's V4 `plot_inversion_progress` are conceptually the same — different optimizers, same idea. They do **not** share code yet because the data sources are different (MATLAB `OutputFcn` vs Python iteration log). Consolidating them is a v2 concern.

## File and function naming

```
option2_nn_surrogate/visualization/
├── plot_training_curves.py
├── plot_residual_histogram.py
├── animate_surrogate_vs_truth.py
├── plot_inversion_progress.py
└── plot_loss_surface_2d.py        (optional)
```

Each entry point takes a serializable input (a path to the run dir, or a `FitResult`/`ValidationReport`/`TrainedSurrogate`) and emits files into `<output_dir>/figures/`. Naming: `<view>_<run_id>_<timestamp>.png`. No view writes outside its `output_dir` — keeps results auditable.

## Live updates during training

Per [shared/VISUALIZATION_SPEC.md § Live updates](../shared/VISUALIZATION_SPEC.md#live-updates), Python uses TensorBoard if available else matplotlib `FuncAnimation`, capped at 5 Hz. **Do not redraw on every training step** — that throttles the GPU.

## What we don't visualize

- **Per-weight histograms.** TensorBoard does this for free; we don't add a separate view.
- **Activation distributions.** Same — TensorBoard's default suffices.
- **t-SNE/UMAP of the latent.** That's an Option 3 (inverse model) concern, not Option 2.
