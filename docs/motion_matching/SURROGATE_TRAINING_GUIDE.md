# Option-2 NN Surrogate Training Guide (Issue #4075)

This document provides a complete guide to training and evaluating the FiLM-MLP surrogate model for motion-matching Option 2.

## Overview

The Option-2 surrogate is a neural network that learns to map from polynomial torque coefficients to club kinematic trajectories:

```
f_θ : coefficients (D,) → kinematic_trajectory (N, 10)
```

where:

- `D = n_joints × 7` (polynomial coefficients)
- `N = 300` (timesteps at 1 kHz over 0.3 s)
- Output per timestep: `[r_butt(3), r_clubhead(3), q_club(4)]`

## Architecture

The model uses a **FiLM-conditioned MLP**:

1. **Coefficient encoder**: MLP that processes input coefficients → latent vector z
2. **FiLM head**: Projects z to per-layer affine modulation parameters (γ, β)
3. **Backbone**: Stack of MLP layers with FiLM modulation, conditioned by sinusoidal time embedding
4. **Output heads**: Separate heads for butt position, clubhead position, club quaternion, and auxiliary joint angles

See `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/APPROACH.md` for detailed architecture and algorithm description.

## Default Configuration

```
n_joints: 14
coeffs_per_joint: 7
seq_len: 300 timesteps
hidden_dim: 256
n_layers: 3 (FiLM-modulated backbone)
time_embed_dim: 64
encoder_layers: 3
dropout: 0.0

Total parameters: ~12M (float32: ~48 MB)
```

### Training Hyperparameters

| Parameter     | Default | Notes                                        |
| ------------- | ------- | -------------------------------------------- |
| n_epochs      | 50      | Can be reduced for faster iteration          |
| batch_size    | 32      | Trials per batch; adjust based on GPU memory |
| learning_rate | 3.0e-4  | AdamW initial learning rate                  |
| weight_decay  | 1.0e-4  | L2 regularization strength                   |
| grad_clip     | 1.0     | Global gradient norm clipping                |
| w_butt        | 1.0     | Weight on butt position MSE                  |
| w_clubhead    | 1.0     | Weight on clubhead position MSE              |
| w_quat        | 0.1     | Weight on quaternion loss                    |
| w_aux         | 0.1     | Weight on auxiliary joint angles             |

### Loss Function

```
L = w_butt · MSE(butt_pred, butt_true)
  + w_clubhead · MSE(clubhead_pred, clubhead_true)
  + w_quat · (1 − ⟨q_pred, q_true⟩²)        # sign-invariant quaternion loss
  + w_aux · MSE(q_joints_pred, q_joints_true)
```

The quaternion loss is sign-invariant: `(1 - dot²(q, q*))` instead of `‖q - q*‖²`, which handles the double cover of SO(3).

## Dataset

### Loading the Real Dataset

Place the 10k parquet dataset at:

```
data/sweep/20251030/
├── trials.parquet        # Per-trial metadata and coefficients
├── timesteps.parquet     # Per-timestep kinematics
└── manifest.json         # Bounds, joint names, generator commit
```

The dataset contract is defined in:

```
src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/DATASET_SCHEMA.md
```

### Synthetic Dataset (Testing)

For testing without the real dataset:

```python
from src.shared.python.motion_matching.dataset import make_synthetic_sweep
make_synthetic_sweep("data/sweep_test", n_trials=100, n_joints=14)
```

### Data Split Strategy

- **80/10/10 split by trial_id** — never split timesteps within a trial (temporal leakage)
- **Stratified by clubhead_speed_max_mph** — keeps speed distribution balanced across folds
- **Deterministic from seed** — splits are recomputed at every load, seed is the persistent artifact
- **Preprocessing**:
  - Coefficients: z-scored per dimension (fitted on train split only)
  - Positions: z-scored per dimension
  - Quaternions: unit-normalized, w ≥ 0 canonicalization

## Training

### Option 1: Command Line

```bash
python3 src/shared/python/motion_matching/surrogate/train_10k.py \
    --dataset-path data/sweep/20251030 \
    --output-dir models/surrogates \
    --n-epochs 50 \
    --batch-size 32 \
    --device auto
```

### Option 2: Python API

```python
from src.shared.python.motion_matching.dataset import load_sweep_dataset
from src.shared.python.motion_matching.surrogate import train_surrogate, TrainConfig

dataset = load_sweep_dataset("data/sweep/20251030")
cfg = TrainConfig(n_epochs=50, batch_size=32, device="auto")
trained = train_surrogate(dataset, cfg)

# Save model
import torch
torch.save(trained.model.state_dict(), "model.pt")
```

### Training Output

The training loop logs:

- Epoch-by-epoch training loss
- Validation loss and clubhead RMSE (mm)
- Model parameters and architecture summary
- Training time and throughput

Example output:

```
epoch 1/50 train=0.482391 val=0.421920 rmse_m=0.0142
epoch 2/50 train=0.325841 val=0.298412 rmse_m=0.0091
...
epoch 50/50 train=0.048291 val=0.052318 rmse_m=0.0034

Total parameters: 12,234,567
Model size (fp32): 46.72 MB
Training time: 1234.5 seconds
```

## Evaluation

### Run the Evaluation Notebook

```bash
jupyter notebook notebooks/evaluate_surrogate.ipynb
```

The notebook covers:

1. Load dataset and create test split
2. Train or load a trained model
3. Model architecture summary and parameter count
4. Inference speed benchmark (target: ≤ 1 ms per sample)
5. Test set accuracy: RMSE on clubhead, butt, quaternion
6. Training curves visualization
7. Per-timestep accuracy analysis
8. Integration checklist for MATLAB

### Inference Speed Benchmark

Expected on CPU (batch of 32, fp32):

- **~30 ms per batch** → ~1 ms per sample ✓

Expected on GPU (batch of 32, fp16):

- **~5 ms per batch** → ~0.15 ms per sample ✓

The target is **≤ 1 ms per prediction** for real-time fitting.

### Test Set Accuracy

Grade: clubhead position RMSE on holdout test set:

- **Excellent**: < 5 mm
- **Good**: 5-10 mm
- **Acceptable**: 10-20 mm
- **Poor**: > 20 mm

## Integration with fit_swing_surrogate.m

Once the model is trained and saved:

1. **Export to ONNX or TorchScript** (optional, for MATLAB integration):

   ```python
   import torch
   model = trained.model
   dummy = torch.randn(1, 98)  # 1 sample, 98 coefficients
   traced = torch.jit.trace(model, dummy)
   torch.jit.save(traced, "surrogate.pt")
   ```

2. **Implement coefficient inversion** in `fit_swing_surrogate.m`:

   - Load trained surrogate weights
   - Gradient descent on coefficients to minimize clubhead position error
   - Apply bound projection (hard clamp, not soft penalty)
   - K-restart strategy (default 8 restarts) to avoid local minima

3. **Round-trip validation** (mandatory):

   - Once surrogate fit finds `coeffs*`, run through full Simscape simulator
   - If `RMSE_simscape > 2× RMSE_surrogate`, flag as extrapolation
   - Use Simscape RMSE in leaderboard, never surrogate RMSE

4. **Hybrid warm-start** (recommended production pattern):

   ```matlab
   % Warm-start with surrogate
   coeffs_warm = fit_via_surrogate_python(target, opts);

   % Polish with fmincon (Option 1)
   result = fit_swing_fmincon(target, ...
       default_option1_options().with_initial(coeffs_warm));
   ```

## Troubleshooting

### Issue: "FileNotFoundError: sweep dataset folder not found"

**Solution**: Place the real 10k parquet at `data/sweep/20251030/` (or specify `--dataset-path`). For testing, use `--use-synthetic`.

### Issue: "Inference time > 1 ms"

**Possible causes**:

- Using fp32 instead of fp16 (consider `torch.float16`)
- Batch size too small (inference is more efficient with batch_size ≥ 8)
- CPU is throttling (check system load)
- Using older GPU with poor tensor core support

**Solutions**:

1. Enable mixed precision: `torch.amp.autocast("cuda")`
2. Use larger batch sizes for amortized latency
3. Profile with `torch.profiler` to identify bottlenecks

### Issue: "Validation RMSE > 5 mm"

**Possible causes**:

- Dataset is too small (need at least 500+ trials)
- Hyperparameters not tuned (try `lr=1e-3`, `hidden_dim=512`)
- Insufficient training (try `n_epochs=100`)

**Solutions**:

1. Verify dataset size: `dataset.n_trials()`
2. Increase `hidden_dim` or `n_layers`
3. Try longer training with early stopping

### Issue: "Round-trip validation flags extrapolation"

**Meaning**: Surrogate prediction disagrees significantly with Simscape ground truth.

**Solution**:

- Do **not** trust this fit; re-run inversion with stricter bounds
- Check if coefficients are at the boundary of the training distribution
- Consider hybrid approach: use surrogate fit as warm-start only, always polish with Simscape

## Files and Structure

```
src/shared/python/motion_matching/surrogate/
├── __init__.py                 # Public API exports
├── model.py                    # SwingSurrogate (FiLM-MLP architecture)
├── train.py                    # Training loop (train_surrogate entry point)
├── train_10k.py                # CLI entry point for 10k dataset
├── invert.py                   # Coefficient inversion (gradient descent)
├── validate.py                 # Round-trip validation against Simscape
├── _normalize.py               # Normalization statistics
├── _quaternion_loss.py         # Sign-invariant quaternion loss
├── _bounds.py                  # Coefficient bounds
└── perstep/
    └── extract_dataset.py      # Extract dataset from full simulator

notebooks/
├── evaluate_surrogate.ipynb    # Comprehensive evaluation notebook

tests/unit/motion_matching/
└── test_surrogate_train_10k.py # Unit tests for training and inference

scripts/
└── validate_surrogate_structure.py  # Sanity check for implementation
```

## References

- **Approach**: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/APPROACH.md`
- **Dataset schema**: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/DATASET_SCHEMA.md`
- **Cost function spec**: `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/shared/COST_FUNCTION_SPEC.md`
- **Issue tracker**: #4075 (Train Option-2 NN surrogate on 10k parquet)
