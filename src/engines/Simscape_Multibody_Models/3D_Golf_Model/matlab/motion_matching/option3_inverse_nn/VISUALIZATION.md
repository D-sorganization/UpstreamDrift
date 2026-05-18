# Option 3 — Visualization

> Option 3 produces the three required views from [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md) (trajectory overlay, error timecourse, fit-quality card) **plus** the four option-specific views below. The shared views are produced by the standard entry points in [shared/VISUALIZATION_SPEC.md §File and function naming](../shared/VISUALIZATION_SPEC.md#file-and-function-naming) — this folder does not duplicate them. Option-3-specific code lives under `option3_inverse_nn/visualization/`.

## V1 — Latent-space projection (UMAP / t-SNE)

**Purpose.** Detect mode collapse and visualize where a new prediction sits relative to the training distribution.

**Layout.** A 2-D scatter:

- One point per **trial** in the validation set, coloured by `total_work_J` (continuous viridis).
- The 2-D coordinate is `UMAP(μ_φ(x_i, θ_i))` — the posterior **mean** for each trial. UMAP fit is computed once and saved with the model.
- A new prediction overlays as `K=32` red dots — the latent samples drawn for the new target's prediction.
- A black star marks the encoder's posterior mean for the new target (for in-distribution targets — for out-of-distribution targets we use only `N(0,I)` samples and there is no star).

**Diagnostic value.**

- **Collapsed latent → all training points pile on one or two clusters.** Fail.
- **Healthy latent → diffuse cloud roughly Gaussian-shaped with smooth `total_work` gradient.**
- **Out-of-distribution prediction → red dots fall in empty regions of the cloud.**

```python
def plot_latent_projection(model, val_dataset, prediction_result, *, method="umap"):
    """Save a PNG and a .pkl of the projector for reuse."""
    ...
```

Stored: `option3_inverse_nn/visualization/latent_umap.{png,pkl}`. The pkl is the fitted UMAP/TSNE for fast overlay of subsequent predictions.

## V2 — Sample diversity (animated multi-overlay)

**Purpose.** Show that 16 latent samples produce 16 distinct but valid swings.

**Layout.** A single 3-D plot animated through the swing:

- Solid blue line: measured club skeleton.
- 16 translucent red lines: simulated club skeletons from `θ̂_1, …, θ̂_16`.
- Time slider shared with the trajectory-overlay view.

**Diagnostic value.**

- All 16 traces collapse onto each other → mode collapse, even if KL test passed (latent is being decoded near-deterministically). Investigate.
- 16 traces fan out wildly → the rejection-sampling stage is doing real work; the variance is the option's "uncertainty quantification."
- 16 traces cluster around 2–3 distinct paths → multi-modal coverage. Best case.

```python
def animate_sample_diversity(target, samples_theta, sim_fn, *, n_show=16):
    """Returns a matplotlib FuncAnimation. Save with anim.save('diversity.mp4')."""
    ...
```

Stored: `option3_inverse_nn/visualization/sample_diversity.mp4`.

## V3 — Round-trip residual histogram (Option 3 vs Option 2)

**Purpose.** Make the cost of multi-modality explicit. Option 3 has wider residuals than Option 2 by construction; this plot quantifies it on the same held-out set.

**Layout.** Overlapping histograms of `final_rmse_m` (clubhead, mm):

- Blue: Option 2 (one prediction per target, deterministic).
- Red: Option 3 with `n_samples=1` (single CVAE sample, no validation).
- Green: Option 3 with `n_samples=32` and round-trip validation (best of 32).
- Vertical dashed line at the 10 mm acceptance threshold.

A short table beside the histogram lists median, p90, p99 for each.

**Diagnostic value.** Validates the rejection-sampling step pays off: green should be left of red and ideally close to blue.

```python
def plot_round_trip_residuals_compare(option2_results, option3_single, option3_validated):
    ...
```

Stored: `option3_inverse_nn/visualization/round_trip_compare.png`.

## V4 — "Where does the inverse fail" coverage map

**Purpose.** Surface dataset-coverage gaps. Where in coefficient/kinematic space does Option 3 produce high round-trip residuals?

**Layout.** A 2-D plot:

- Same UMAP projection as V1 (so the plots overlay nicely).
- Each held-out trial coloured by its **round-trip RMSE** in mm (sequential YlOrRd colormap).
- Trials above the 10 mm threshold get a black ring.

**Diagnostic value.** Clusters of high-error points = under-sampled regions of the dataset. These are the **training targets for the next data-generation run** (active learning per [APPROACH.md §Future work](APPROACH.md#future-work)).

```python
def plot_coverage_failure_map(model, held_out_results):
    ...
```

Stored: `option3_inverse_nn/visualization/coverage_map.png`.

## Live training curves

For the training run itself (Issue #032), expose:

- Train and val loss (total + KL + MSE + work-reg components separately).
- KL value vs the annealing schedule (so we can see if it's actually being respected).
- Validation round-trip RMSE every `eval_every_epochs` (default 5).

Use TensorBoard if available; else `matplotlib` `FuncAnimation` capped at 5 Hz per [shared/VISUALIZATION_SPEC.md §Live updates](../shared/VISUALIZATION_SPEC.md#live-updates).

## Styling

Inherits from [shared/VISUALIZATION_SPEC.md §Styling](../shared/VISUALIZATION_SPEC.md#styling):

- Measured = `#1f77b4` blue.
- Simulated = `#d62728` red. (V2's 16 samples are this colour at α=0.3.)
- Error / annotation = `#7f7f7f` grey.
- Round-trip-pass = `#2ca02c` green; round-trip-fail = `#000000` black ring.
- DPI 200 PNG, `.fig` not applicable (Python plots).
- No emoji in figure text.

## Entry points

```
option3_inverse_nn/visualization/
├── plot_latent_projection.py
├── animate_sample_diversity.py
├── plot_round_trip_residuals_compare.py
├── plot_coverage_failure_map.py
└── training_curves.py
```

The shared three views (trajectory overlay, error timecourse, fit-quality card) are rendered by `motion_matching/shared/visualization/*` and consume `InverseFitResult` directly because it has the same shape as Option 1's and Option 2's result structs (per [shared/VISUALIZATION_SPEC.md §File and function naming](../shared/VISUALIZATION_SPEC.md#file-and-function-naming)).
