# Option 3 — Runbook

> Literal commands. Run from the **repo root** (`C:\Users\diete\Repositories\UpstreamDrift\`) unless noted. Paths use forward slashes; works in both bash and PowerShell.

## 0. Prerequisites

- Python 3.10+, PyTorch 2.x with CUDA if available (CPU works for inference).
- The random-sweep parquet dataset at `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/data/sweep/<run_id>/` with both `trials.parquet` and `timesteps.parquet` (see [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md)).
- For round-trip validation: MATLAB R2022b+ with the Simscape Multibody toolbox **or** Option 4's `SimscapeAdapter` running.
- Option 2's data loader importable. If Option 2 has not yet promoted its loader to `src/shared/python/motion_matching/`, set:
  ```bash
  export PYTHONPATH="$PYTHONPATH:src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate"
  ```

## 1. One-time setup

```bash
# Install the project in editable mode (pulls torch, polars, etc.)
python3 -m pip install -e .

# Sanity check the dataset loads:
python3 -c "
from src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.data import load_sweep_dataset
from pathlib import Path
ds = load_sweep_dataset(Path('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/data/sweep/20251030'))
print('trials:', len(ds.trials), 'joints:', ds.joint_names)
"
```

If that fails, fix Option 2's loader (Issue #019) before continuing.

## 2. Train the inverse CVAE

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.train \
  --dataset src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/data/sweep/20251030 \
  --checkpoint-dir src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/models \
  --epochs 200 \
  --batch-size 64 \
  --d-z 32 \
  --kl-warmup-epochs 20 \
  --lambda-work 1e-3 \
  --device cuda \
  --seed 0 \
  --log-dir runs/option3/$(date +%Y%m%d-%H%M%S)
```

**Expected wall clock:** ~2–6 hours on a single 12 GB GPU for `~20k` trials. CPU training is feasible but slow (~24 h); use only for smoke tests.

**TensorBoard:**

```bash
tensorboard --logdir runs/option3
```

Watch for: KL ramping smoothly to ~1, val MSE on θ decreasing, val round-trip RMSE (logged every 5 epochs) stabilizing under 10 mm.

The run produces a checkpoint under `models/inverse_cvae_<git_sha>_<timestamp>.pt` plus a sibling `<...>.config.json` and `<...>.norm_stats.npz`. Promote one to `models/inverse_cvae_latest.pt` (symlink) for downstream use.

## 3. Smoke-test the trained model

```bash
python3 -m pytest \
  src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/tests \
  -m "unit or integration" -n auto --timeout=60
```

If `test_kl_does_not_collapse` fails, the model is degenerate even if the loss looked fine — re-train with a longer KL warmup or smaller `d_z`. Do not ship.

## 4. Predict for a single target (no validation, fast path)

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.predict \
  --model src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/models/inverse_cvae_latest.pt \
  --target src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/data/Wiffle_ProV1_club_3D_data.xlsx \
  --target-sheet TW_ProV1 \
  --n-samples 1 \
  --no-validate \
  --output results/option3/TW_ProV1_fast.json
```

Wall clock: < 100 ms. Output is an `InverseFitResult` JSON. **Not safe to ship downstream — see ASSUMPTIONS.md §A4.** Use this mode only as a warm start for Option 1.

## 5. Predict with round-trip validation (recommended)

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.predict \
  --model .../models/inverse_cvae_latest.pt \
  --target .../data/Wiffle_ProV1_club_3D_data.xlsx \
  --target-sheet TW_ProV1 \
  --n-samples 32 \
  --validate \
  --sim-backend matlab-engine \
  --rmse-threshold-m 0.010 \
  --surrogate-prefilter .../option2_nn_surrogate/models/surrogate_latest.pt \
  --output results/option3/TW_ProV1_validated.json
```

Wall clock with surrogate prefilter: ~5–10 s. Without prefilter: ~30 s for 32 Simscape calls.

The result struct is canonical (per [shared/CODING_STANDARDS.md §Provenance](../shared/CODING_STANDARDS.md#provenance-and-reproducibility)) and the shared visualization helpers consume it directly:

```bash
python3 -c "
from motion_matching.shared.visualization import render_all
render_all('results/option3/TW_ProV1_validated.json',
           'src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/data/Wiffle_ProV1_club_3D_data.xlsx',
           'TW_ProV1',
           out_dir='results/option3/figures/TW_ProV1/')
"
```

## 6. Round-trip-validate an existing result file

For offline analysis (e.g. to re-score after a Simscape model change):

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.validate_offline \
  --result results/option3/TW_ProV1_validated.json \
  --sim-backend matlab-engine \
  --output results/option3/TW_ProV1_revalidated.json
```

## 7. Build the under-determined synthetic test fixture

This is a one-time offline job that produces the test data for `test_multiple_samples_produce_distinct_coefficients_for_under_determined_target`:

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.notebooks.build_under_determined_fixture \
  --dataset .../data/sweep/20251030 \
  --n-cases 64 \
  --tol-mm 1.0 \
  --output src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/tests/data/synthetic_under_determined.pt
```

Wall clock: ~30 min (brute-force perturbation per [APPROACH.md §Mode-coverage diagnostic](APPROACH.md#mode-coverage-diagnostic)). Re-run only when the Simscape model or coefficient bounds change.

## 8. Hybrid handoff to Option 1 (`fmincon` warm start)

```bash
# Step A: get an Option 3 warm start (fast, no validate)
python3 -m ...option3_inverse_nn.predict \
  --model .../models/inverse_cvae_latest.pt \
  --target .../data/Wiffle_ProV1_club_3D_data.xlsx \
  --target-sheet TW_ProV1 \
  --n-samples 1 --no-validate \
  --output results/hybrid/TW_ProV1_warmstart.json

# Step B: feed θ̂ to Option 1's fmincon as x0 (MATLAB)
matlab -batch "
  addpath(genpath('src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching'));
  warm = loadjson('results/hybrid/TW_ProV1_warmstart.json');
  target = load_club_target_excel('.../data/Wiffle_ProV1_club_3D_data.xlsx', 'TW_ProV1');
  opts = default_options(); opts.x0 = warm.coefficients;
  result = fit_swing_fmincon(target, opts);
  save('results/hybrid/TW_ProV1_final.mat', 'result');
"
```

Expected wall clock: cold-start `fmincon` is ~3–5 min; warm-started from a healthy CVAE prediction is ~30–60 s.

## 9. Evaluate on the held-out test split (full report)

```bash
python3 -m src.engines.Simscape_Multibody_Models._3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.evaluate \
  --model .../models/inverse_cvae_latest.pt \
  --dataset .../data/sweep/20251030 \
  --split test \
  --n-samples 32 \
  --validate \
  --output results/option3/eval_test.json
```

Produces all four option-specific visualizations plus the comparison table feeding `motion_matching/shared/leaderboard.m`.

## 10. Common gotchas

- **`KL == 0` after training.** Mode collapse. See [TESTING.md §test_kl_does_not_collapse](TESTING.md#test_kl_does_not_collapse). Fix: longer KL warmup, smaller `d_z`, free-bits trick.
- **Round-trip RMSE > 100 mm on every sample.** The decoder is producing out-of-bound coefficients OR Simscape is failing silently. Inspect `result.solver_options` and the `solver_status` field on the simulator return.
- **Predict is slower than 50 ms.** GPU contention or the model loaded onto CPU vs GPU mismatch. Force `--device cpu` for the latency target; GPU for batch inference.
- **Surrogate prefilter ranks samples disagree wildly with Simscape.** Option 2's surrogate is out-of-date or its normalization stats differ. Re-export them together.
