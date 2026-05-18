# Option 2 — Runbook

Literal Python and MATLAB commands for the four operational tasks: train, fit, validate, hybrid.

> **Status.** Most commands below are not yet executable — the implementation lands in issues #028–#031. This runbook is the spec the agents implement against. All paths are absolute or repo-relative as documented.

## Conventions

- Run all Python commands from the **repo root** (`C:\Users\diete\Repositories\UpstreamDrift`).
- Always `python3`, never `python`, per [`CLAUDE.md`](../../../../../../../CLAUDE.md).
- All output paths default to `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/models/<run_id>/` — `.gitignore`'d.
- `<run_id>` defaults to a UTC timestamp; override with `--run-id`.

## 0. Prerequisites

```bash
# Verify Python env has torch.
python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Verify the parquet dataset is present.
ls data/sweep/  # or whatever path DATA.md settles on; expect trials.parquet, timesteps.parquet
```

If the dataset is not present, **stop**. See [DATA.md](DATA.md).

## 1. Train the surrogate from a fresh parquet

```bash
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.train \
    --dataset-path data/sweep/20251030 \
    --output-dir src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/models/20251030_run1 \
    --architecture film_mlp \
    --hidden-dim 256 \
    --n-layers 4 \
    --batch-size 32 \
    --max-steps 50000 \
    --lr 3e-4 \
    --seed 0xC0FFEE
```

**What it does**

1. Loads `trials.parquet` + `timesteps.parquet` via `load_sweep_dataset` (issue #019).
2. Splits 80/10/10 by `trial_id` per [APPROACH.md § Data split](APPROACH.md#data-split).
3. Computes per-feature normalization stats from the train split only.
4. Trains the surrogate with AdamW + cosine schedule + mixed precision.
5. Logs to TensorBoard if installed; always to JSONL at `<output_dir>/train_log.jsonl`.
6. Saves `<output_dir>/best.pt` (best-on-val) and `<output_dir>/last.pt` (final step).
7. Saves `<output_dir>/config.json` (a `TrainConfig` dump including `git_commit`).
8. Saves `<output_dir>/norm_stats.npz`.

**Wall-clock estimate**

- ~5k trials × 300 timesteps × 50k steps = ~6 hours on a single mid-range GPU; ~2 days on CPU.
- Use `--max-steps 5000` for a smoke test.

**Smoke test (no GPU, no real dataset)**

```bash
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.train \
    --dataset-path tests/_fixtures/tiny_sweep \
    --output-dir /tmp/option2_smoke \
    --max-steps 200 \
    --batch-size 4
```

Expected: completes in < 60 s; `best.pt` exists; final val RMSE is **not** required to be small.

## 2. Fit a measured swing (Python entry-point)

```bash
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.invert \
    --checkpoint src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/models/20251030_run1/best.pt \
    --target-source excel \
    --target-path src/apps/golf_gui/Motion\ Capture\ Plotter/Wiffle_ProV1_club_3D_data.xlsx \
    --target-sheet TW_ProV1 \
    --n-restarts 8 \
    --max-iters 200 \
    --invert-lr 1e-2 \
    --output-dir src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/results/option2/TW_ProV1_<timestamp>
```

**What it does**

1. Loads the trained surrogate from the checkpoint (`load_trained_surrogate`).
2. Loads the target via `load_club_target_excel` (or `--target-source c3d|synthetic`).
3. Runs Adam on the input coefficients with bound projection and K-restart per [APPROACH.md § Inversion](APPROACH.md#inversion).
4. Saves `<output_dir>/fit_result.json` (a `FitResult` dump) and `<output_dir>/coefficients.npy`.
5. Renders V4 (inversion progress) into `<output_dir>/figures/`.

`simscape_rmse_m` is `null` in the result — populate it via step 3.

## 3. Validate the fit against Simscape (round-trip)

There are two paths: Python-only (uses the simulate-from-Python adapter from Option 4 if present) or MATLAB-driven (uses `simulate_with_coefficients.m`).

### 3a. MATLAB-driven (canonical path)

From MATLAB, with the repo on the path:

```matlab
% Load the Python-side fit result.
result_py = jsondecode(fileread( ...
    'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/results/option2/TW_ProV1_<timestamp>/fit_result.json'));

% Round-trip through Simscape.
sim_out = simulate_with_coefficients(result_py.coefficients);

% Compute the same cost the leaderboard uses.
target = load_club_target_excel( ...
    'src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx', ...
    'TW_ProV1');
[J, terms] = compute_cost(result_py.coefficients, target, ...
    @simulate_with_coefficients, default_cost_options());

% Persist the validation report.
report = struct( ...
    "simscape_rmse_m", terms.position_rmse_m, ...
    "surrogate_rmse_m", result_py.surrogate_rmse_m, ...
    "extrapolation_ratio", terms.position_rmse_m / result_py.surrogate_rmse_m, ...
    "is_extrapolation", (terms.position_rmse_m / result_py.surrogate_rmse_m) > 2.0 ...
);
save("validation_report.mat", "-struct", "report");
```

### 3b. Python-only (only when Option 4 bridge is up)

```bash
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.validate \
    --fit-result <fit_dir>/fit_result.json \
    --extrapolation-factor 2.0
```

This requires `option4_python_bridge` to be operational; if not, use 3a.

## 4. Hybrid: surrogate warm-start → fmincon polish

The recommended production pattern. Driven from MATLAB.

```matlab
target = load_club_target_excel( ...
    'src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx', ...
    'TW_ProV1');

% Step A: surrogate warm-start (calls Python under the hood).
opts2 = struct();
opts2.checkpoint_path = fullfile( ...
    'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching', ...
    'option2_nn_surrogate/models/20251030_run1/best.pt');
opts2.n_restarts = 8;
opts2.max_iters = 200;
opts2.polish    = true;        % triggers the fmincon handoff inside the shim
warm_then_polished = fit_swing_surrogate(target, opts2);

% Result is in the same format every other option emits, per
% shared/CODING_STANDARDS.md § Provenance.
disp(warm_then_polished.solver);          % "nn-surrogate+fmincon"
disp(warm_then_polished.final_rmse_m);
```

If you'd prefer to wire the handoff manually (more control, more code):

```matlab
warm = fit_swing_surrogate(target, struct( ...
    "checkpoint_path", "...best.pt", "polish", false));
opts1 = default_option1_options();
opts1.initial_coefficients = warm.coefficients;   % from the surrogate fit
polished = fit_swing_fmincon(target, opts1);
```

The hybrid handoff is issue #030.

## 5. Resume training from a checkpoint

```bash
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.train \
    --resume-from src/engines/.../models/20251030_run1/last.pt \
    --max-steps 75000
```

Resume rebuilds optimizer + scheduler state from the checkpoint; the `git_commit` recorded with the checkpoint is compared against `HEAD` and a warning printed if they disagree (does not block).

## 6. Promote a checkpoint to "blessed"

Once a checkpoint has passed:

- Held-out RMSE < 5 mm (`test_surrogate_held_out_rmse_under_5mm`).
- All 6 required tests in [TESTING.md](TESTING.md).
- A round-trip validation on the canonical `TW_ProV1` swing.

…tag it:

```bash
# Copy (do not move — keeps the run id traceable).
cp src/engines/.../models/20251030_run1/best.pt \
   src/engines/.../models/blessed/v1.pt
# Record provenance.
python3 -m src.engines.Simscape_Multibody_Models.\
3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.bless \
    --src models/20251030_run1/best.pt \
    --as v1
```

The `bless` tool writes `models/blessed/v1.metadata.json` with `(run_id, dataset_run_id, git_commit, val_rmse_m, blessed_by, blessed_at_utc)`. The MATLAB shim defaults `checkpoint_path` to `models/blessed/v1.pt` if not specified.

## 7. Troubleshooting

| Symptom                                              | Likely cause                                    | Fix                                                                              |
| ---------------------------------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------- |
| Training loss is NaN at step 0                       | unnormalized inputs, `coeffs` not z-scored      | check `norm_stats.npz` exists and is loaded                                      |
| Val RMSE plateaus at ~50 mm                          | quaternion sign-flip during data loading        | confirm canonicalize-on-load in `dataset.py`                                     |
| `pyrunfile` errors on import                         | wrong `pyenv()`                                 | run `pyenv("Version", "...")` in MATLAB pointing at `requirements.lock`'s python |
| Extrapolation flagged on every fit                   | bounds in inversion don't match training bounds | re-run inversion with the bounds stored in the checkpoint metadata               |
| Adam diverges                                        | `invert_lr` too high for this checkpoint        | lower to `1e-3`                                                                  |
| MATLAB shim returns surrogate RMSE not Simscape RMSE | round-trip step skipped                         | set `opts.validate = true` (default)                                             |

## 8. CI hooks

The full Option 2 pipeline is **not** run on every PR — too slow. CI runs:

- All `unit`-marked tests in `option2_nn_surrogate/tests/` (millisecond budget).
- `test_surrogate_predicts_training_trial_within_2mm` (`integration`, ~30 s).
- `test_surrogate_gradient_finite` (`unit`, milliseconds).

The `integration + slow` tests (full training, round-trip) run on a nightly schedule with the parquet dataset mounted. See `.github/workflows/` once issue #028 lands.
