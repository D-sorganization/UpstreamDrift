# Option 2 — Data

The full schema and validation rules live in [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md). This file documents the **Option 2-specific** preprocessing, splits, augmentation, and the protocol for adding new dataset runs.

## Status

The parquet dataset is **not yet in the repo**. The user has stated they will copy it in. Until it lands, every `integration`-marked test in [TESTING.md](TESTING.md) is skipped, and the surrogate cannot be trained.

When the dataset arrives, the loader at issue #019 (`load_sweep_dataset`) ingests it. Option 2 wraps that loader with a PyTorch `Dataset` (see `INTERFACES.md § dataset.py`) — it does **not** re-implement parquet I/O.

## Where it lives

```
data/sweep/
└── <dataset_run_id>/                      # e.g. 20251030
    ├── trials.parquet
    ├── timesteps.parquet
    └── manifest.json                      # bounds, joint_names, generator git_commit, seed range
```

`<dataset_run_id>` is whatever string `runSimulation.m` (the dataset generator) writes; the user's most-recent run id is `20251030`. The loader uses the manifest to cross-check the sweep bounds against `generateRandomCoefficients.m`.

The `data/` folder is `.gitignore`'d; only the symlink target on the user's machine is real. **Confirm before training.** See [shared/DATASET_SCHEMA.md § Open questions](../shared/DATASET_SCHEMA.md#open-questions-for-the-user).

## Preprocessing

Applied inside `SweepDatasetTorch` on `__getitem__`, **not** stored on disk. (Storing preprocessed tensors makes the cache stale every time we tweak normalization.)

### Per-feature normalization

Computed on the **train split only** — never on the full dataset, never on val/test. Bake the stats into the checkpoint so inference reproduces them exactly.

| Feature             | Normalization   | Notes                                                   |
| ------------------- | --------------- | ------------------------------------------------------- |
| `coeffs` (D,)       | z-score per dim | mean and std over train trials                          |
| `r_butt` (N, 3)     | z-score per dim | mean and std over (train trials × timesteps)            |
| `r_clubhead` (N, 3) | z-score per dim | same                                                    |
| `q_club` (N, 4)     | none            | already unit-norm; canonicalize sign on load (`w >= 0`) |

Stats live in `NormalizationStats` (see [INTERFACES.md](INTERFACES.md)) and are saved as `norm_stats.npz` next to the checkpoint.

### Quaternion canonicalization

For every quaternion in every trial, on load: if `w < 0`, flip sign. This must happen at the **dataset boundary**, not inside the surrogate, because it's a pre-supervision step. The model learns one branch of the double cover.

### Solver-failure exclusion

Trials with `trials.solver_status != "success"` are excluded by default. Override only for debugging: `SweepDatasetTorch(..., include_failures=True)` — does not affect training, throws a warning.

### Time grid alignment

Per [ASSUMPTIONS.md § A4](ASSUMPTIONS.md#a4-time-grid-is-fixed-across-the-dataset), every trial is on the canonical `(T = 0.3 s, sample_rate = 1000 Hz, N = 300)` grid. If a trial in the parquet has a different grid (sample-rate drift, simulation aborted early), the loader **resamples** linearly in position and SLERP in orientation. Trials with `t[-1] < 0.95 × T` are excluded as truncated.

## Splits

### 80 / 10 / 10 by `trial_id`

```python
splits = {"train": 0.8, "val": 0.1, "test": 0.1}
```

Splitting **by trial**, never by timestep. Splitting by timestep would leak future timesteps from a trial into validation, which is meaningless for a sequence model. The split is deterministic given the seed in `TrainConfig.seed`.

### Stratification

Stratify by `trials.clubhead_speed_max_mph` quintile. This keeps the speed distribution balanced across folds and prevents a "we trained on slow swings, tested on fast" failure mode. The stratifier is `sklearn.model_selection.StratifiedShuffleSplit`-equivalent; we don't pull in scikit-learn just for this — implement inline (it's ~30 lines).

### Reproducibility

Splits are recomputed deterministically from `(dataset_sha256, seed)` at every load. We do **not** persist a split file — the seed is the persistent artifact.

## Augmentation

### v1: none

We do not augment in v1. The dataset is already a Latin-hypercube-style random sweep over the coefficient space; further augmentation risks drifting outside the bounds and contaminating training with extrapolation samples.

### Future: rotation augmentation

A natural augmentation: rotate the entire trial about the world `z` axis (gravity axis) by a random angle, applied identically to butt position, clubhead position, and club quaternion. The coefficients themselves do **not** rotate (they are torques in joint frame, not world frame), so this only augments the output side — which means it's better cast as a **post-hoc invariance constraint** rather than augmentation.

Document in v2: add a `rotation_aug_strength` flag to `TrainConfig`; train with `r_world ← R_z(α) · r_world` for `α ∼ U[-π, π]`. Skip in v1.

### What we will not augment

- **Coefficient noise.** Adding gaussian noise to coefficients is a data-augmentation hammer that here is just smearing the supervision signal — the dataset already covers the coefficient space densely.
- **Time warping.** Breaks the fixed-grid assumption (A4).
- **Joint-angle noise.** v1 has no joint-angle supervision.

## Adding new dataset runs

When the user generates `data/sweep/20260101/` (or whatever):

1. **Drop in.** Copy the new `<run_id>` folder under `data/sweep/`.
2. **Validate.** Run the loader's manifest cross-check:
   ```bash
   python3 -m src.engines.Simscape_Multibody_Models.\
   3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.dataset \
       --inspect data/sweep/20260101
   ```
   This prints: trial count, joint names, simulation duration, sample rate, coefficient bounds. Compare against the manifest from the old run to catch silent schema drift.
3. **Confirm bounds match.** If `generateRandomCoefficients.m` has changed, the surrogate trained on the old run will be invalid for the new run — surface this loudly. The loader compares manifest bounds with `generateRandomCoefficients.m` constants and refuses to load if they disagree (issue #019 acceptance).
4. **Re-train.** Run `train.py` per [RUNBOOK.md § 1](RUNBOOK.md#1-train-the-surrogate-from-a-fresh-parquet) with `--dataset-path data/sweep/20260101`.
5. **Re-validate the prior fits.** Any fit produced against the old surrogate references an old `checkpoint_id`; the leaderboard auto-greys those rows. Re-fit the canonical swings against the new surrogate to bring them current.

## Open questions for the human

- **Format.** Parquet vs HDF5 — see open question in [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md#open-questions-for-the-user). If HDF5, this doc and `dataset.py` need to switch backend; the public Python API does not change.
- **Trial count for `20251030`.** Influences the smoke-test heuristics in [RUNBOOK.md § Smoke test](RUNBOOK.md#smoke-test-no-gpu-no-real-dataset).
- **Multi-subject.** Does any dataset run mix subjects? If so, surface the subject id in the schema and revisit the v1 single-subject assumption ([ASSUMPTIONS.md § A11](ASSUMPTIONS.md#a11-the-surrogate-is-single-subject-for-v1)).
- **`git_commit` of the generator.** Is it captured per-trial or per-run? Per-run is fine; per-trial is overkill.
