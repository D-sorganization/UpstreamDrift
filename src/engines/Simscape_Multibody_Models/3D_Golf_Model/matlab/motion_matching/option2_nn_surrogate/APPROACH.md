# Option 2 — Approach

End-to-end algorithm for the differentiable forward surrogate and the gradient-based inversion that fits a measured swing.

## Problem statement

Given:

- A trained surrogate `f_θ : coefficients (D,) → kinematic_trajectory (N, 10)` where `D = n_joints × 7`, `N = 300` (default), and the 10-vector per timestep is `[r_butt(3), r_clubhead(3), q_club(4)]` per [ASSUMPTIONS.md § A3](ASSUMPTIONS.md#a3-output-kinematic-representation-matches-the-cost-function).
- A measured target trajectory `y* ∈ ℝ^(N, 10)` per [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).

Find:

```
coeffs* = argmin_{coeffs ∈ Bounds}  L(f_θ(coeffs), y*)
```

with `L` the same weighted MSE used by [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md), augmented with a quaternion-aware orientation term.

## Architecture: candidates and recommendation

Two reasonable architectures. **Recommended: FiLM-conditioned MLP.**

### Candidate A — 1D-CNN over time + MLP encoder

```
coeffs (D,)   ──►  MLP encoder  ──►  z (latent, e.g. 256)
                                       │
                                       ▼
                       Broadcast to (N, 256), concat with sinusoidal time embedding
                                       │
                                       ▼
                          1D-CNN stack (residual, dilated)
                                       │
                                       ▼
                            Heads:
                              ─ butt    (N, 3)
                              ─ clubhead (N, 3)
                              ─ q_club  (N, 4) → unit-normalize
                              ─ q_aux   (N, n_joints)  [debug only]
```

Pros: temporal locality; convolutions natively handle the sequence. Cons: more parameters to tune (kernel sizes, dilation schedule), bigger memory footprint at `N=300`.

### Candidate B — FiLM-conditioned MLP (recommended)

```
coeffs (D,)   ──►  MLP encoder  ──►  (γ, β)  per layer (FiLM)
                                       │
                                       ▼
sinusoidal_time_embedding(t)  ──► MLP backbone with FiLM modulation per layer
                                       │
                                       ▼
                            Heads (same 4 as above)
```

The trick: the time embedding is the **input** to a per-timestep MLP whose weights are unconditional but whose layer-wise affine modulation `(γ, β)` is conditioned on the coefficient encoding. Each timestep is decoded independently, so the model is naively parallel across `N` and trivially differentiable through the FiLM path.

**Why FiLM wins for v1.**

1. The forward map `coefficients → club_state(t)` is closer to "evaluate a polynomial-driven dynamical system at time `t`" than "denoise a sequence". A point-wise decoder conditioned on a global coefficient embedding matches the physics better than a convolutional sequence model.
2. Memory cost is `O(N × hidden)` not `O(N × hidden × kernel × layers)`. With `N=300`, batch_size=32, that's the difference between 4 GB and 16 GB.
3. Gradient w.r.t. coefficients flows through a small encoder MLP — a very clean, well-conditioned path. (See `test_surrogate_gradient_finite` in [TESTING.md](TESTING.md).)
4. FiLM is a 2018 trick, well-understood, easy to implement in PyTorch in <100 lines.

**Fallback to 1D-CNN** if held-out RMSE plateaus above 5 mm with FiLM. Architecture choice is configurable in `SurrogateConfig.architecture = "film_mlp" | "cnn1d"`.

### Sketch of FiLM-MLP forward (PyTorch-ish pseudocode)

```python
# In surrogate.py — implementation skeleton only; real bodies land in #028.
def forward(self, coeffs):              # coeffs: (B, D)
    z = self.coeff_encoder(coeffs)      # (B, K)
    gammas, betas = self.film_head(z)   # each: list of (B, hidden) per layer

    t_emb = self.time_embed(self.time_grid)  # (N, T_emb)
    t_emb = t_emb.unsqueeze(0).expand(B, -1, -1)  # (B, N, T_emb)

    h = self.input_proj(t_emb)          # (B, N, hidden)
    for layer, gamma, beta in zip(self.backbone, gammas, betas):
        h = layer(h)
        h = gamma.unsqueeze(1) * h + beta.unsqueeze(1)   # FiLM
        h = F.gelu(h)

    butt     = self.butt_head(h)         # (B, N, 3)
    clubhead = self.clubhead_head(h)     # (B, N, 3)
    q_raw    = self.quat_head(h)         # (B, N, 4)
    q_unit   = q_raw / q_raw.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    q_unit   = canonicalize_quaternion_sign(q_unit)      # w >= 0
    q_aux    = self.aux_joint_head(h)    # (B, N, n_joints), debug only

    return ClubTrajectory(butt=butt, clubhead=clubhead, q_club=q_unit, q_joints=q_aux)
```

## Loss

Weighted MSE with a quaternion-aware orientation term and an auxiliary joint-angle term:

```
L = w_butt · MSE(butt_pred, butt_true)
  + w_clubhead · MSE(clubhead_pred, clubhead_true)
  + w_quat · (1 − ⟨q_pred, q_true⟩²)            # accounts for q ↔ −q
  + w_aux · MSE(q_joints_pred, q_joints_true)   # auxiliary head, debug
```

Defaults (in `TrainConfig`):

| Weight       | Value | Rationale                                                            |
| ------------ | ----- | -------------------------------------------------------------------- |
| `w_butt`     | `1.0` | Position in metres²; sets the scale                                  |
| `w_clubhead` | `1.0` | Same scale as butt                                                   |
| `w_quat`     | `0.1` | Same orientation/position ratio as `compute_cost`                    |
| `w_aux`      | `0.1` | Auxiliary; small enough to stay a regularizer, large enough to learn |

The quaternion term uses `(1 − ⟨q,q*⟩²)` rather than `‖q − q*‖²` because the latter is double-valued for sign-flipped quaternions. The `⟨·,·⟩²` form is smooth, sign-invariant, and equals `sin²(θ/2)·cos²(θ/2)` in the angle θ between rotations — small near zero, monotonic in θ on `[0, π/2]`. This is the standard quaternion-supervision trick and is what passes `test_surrogate_gradient_finite`.

## Training procedure

### Data split

- **80/10/10 by `trial_id`.** Never split inside a trial — that leaks future timesteps into validation.
- Stratify by `clubhead_speed_max_mph` to keep the speed distribution balanced across folds. Cite `trials.clubhead_speed_max_mph` from [DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md).
- Random seed is fixed in `TrainConfig.seed` (default `0xC0FFEE`).

### Optimizer

- **AdamW**, `lr=3e-4`, `weight_decay=1e-4`, `betas=(0.9, 0.999)`.
- **Cosine schedule** with `warmup_steps=500`, `max_steps=50_000`, `min_lr_ratio=0.01`.
- **Gradient clipping** at `‖g‖₂ = 1.0`.
- **Mixed precision** via `torch.amp.autocast` + `GradScaler` when CUDA is available; falls back to fp32 on CPU.

### Batch shape

- `batch_size = 32` trials × `N = 300` timesteps. With `D ≈ 7 × n_joints` and a 4-layer 256-hidden FiLM-MLP, that's ~12 M parameters and ~3 GB GPU memory at fp16.

### Stopping

- **Primary stop:** validation `RMSE_clubhead < 5 mm` and not improving for 5 consecutive evals (eval every 1000 steps).
- **Secondary stop:** `max_steps` exhausted.
- Best-on-val checkpoint is saved to `models/<run_id>/best.pt`. Last checkpoint also saved (`last.pt`).

### Per-feature normalization

- Inputs: `coeffs` z-scored per dimension using stats computed on the **training split only**.
- Outputs: `(r_butt, r_clubhead)` z-scored; quaternions left raw (already unit-norm).
- Normalization stats are saved alongside the checkpoint and applied at inference time.

See [DATA.md § Preprocessing](DATA.md#preprocessing).

## Inversion

Once the surrogate is trained, fit a measured swing by gradient descent on the input.

### Algorithm

```
function fit_swing_via_surrogate(target, surrogate, opts):
    best_result = None
    for k in 1..opts.n_restarts:
        coeffs = sample_uniform(bounds_low, bounds_high)   # cold start
        opt = Adam([coeffs], lr=opts.invert_lr)            # default 1e-2
        for it in 1..opts.max_iters:                       # default 200
            pred = surrogate(coeffs)
            loss = weighted_mse(pred, target)
            loss.backward()
            opt.step()
            coeffs.data.clamp_(bounds_low, bounds_high)    # bound projection
            opt.zero_grad()
        result_k = (coeffs.detach(), final_loss)
        if best_result is None or result_k.loss < best_result.loss:
            best_result = result_k
    return best_result
```

### Why bound projection (not penalty)

A bound penalty (`λ · max(0, x − x_max)²`) softens the bounds and lets the surrogate be queried outside its training distribution — that's the failure mode we want to avoid (see [ASSUMPTIONS.md § A1](ASSUMPTIONS.md#a1-surrogate-validity-is-bounded-by-the-training-distribution)). Hard projection (`clamp_`) keeps every iterate inside the trust region.

### Why K-restart

The forward map is multi-modal; multiple coefficient regions can produce the same club trajectory because of the under-determined club-only observation (see [shared/README.md § Hard constraints (6)](../shared/README.md#hard-constraints-assumptions-all-four-options-must-respect)). Adam from a single random start often finds a local mode that fits the surrogate well but fails round-trip validation. Default `n_restarts = 8`, parallelized across the batch dim so it costs ~1 forward pass.

### Default `InvertOptions`

| Option            | Default  | Notes                                                             |
| ----------------- | -------- | ----------------------------------------------------------------- |
| `n_restarts`      | `8`      | K random starts                                                   |
| `max_iters`       | `200`    | per restart                                                       |
| `invert_lr`       | `1e-2`   | Adam learning rate on coefficients                                |
| `early_stop_loss` | `1e-6`   | bail early if loss falls below this                               |
| `regularizer`     | `"none"` | optional `‖coeffs‖²` (`"coeff_l2"`) — see `COST_FUNCTION_SPEC.md` |
| `lambda`          | `0.0`    | regularizer strength                                              |
| `seed`            | `None`   | reproducibility                                                   |

### Wall-clock budget

Per fit: 8 restarts × 200 iterations × ~5 ms/iter = **~8 seconds** on CPU. On GPU, ~1 second. Both well inside the "seconds per fit" promise.

`test_fit_completes_under_30s_for_typical_target` asserts this. See [TESTING.md](TESTING.md).

## Validation: round-trip every fit

**Mandatory.** Once `fit_swing_via_surrogate` returns `coeffs*`, immediately run them through Simscape:

```
sim_truth = simulate_with_coefficients(coeffs*)         # the ground truth
rmse_simscape = compute_cost(coeffs*, target, sim_fn).final_rmse_m
rmse_surrogate = ‖f_θ(coeffs*) − target‖              # what the surrogate thinks

if rmse_simscape > extrapolation_factor × rmse_surrogate:
    flag "extrapolation"; the fit is not trustworthy
```

`extrapolation_factor` defaults to `2.0`. This is the test `test_round_trip_validation_catches_extrapolation` in [TESTING.md](TESTING.md).

The `final_rmse_m` reported in the leaderboard (per [shared/CODING_STANDARDS.md § Provenance](../shared/CODING_STANDARDS.md#provenance-and-reproducibility)) is **always** the Simscape round-trip RMSE, never the surrogate-predicted RMSE. See [ASSUMPTIONS.md § A12](ASSUMPTIONS.md#a12-cost-function-parity-with-option-1).

## Hybrid: handoff to Option 1

The recommended production pattern: **use Option 2 to warm-start Option 1.**

```
coeffs_warm = fit_swing_via_surrogate(target, surrogate, opts).coefficients
result = fit_swing_fmincon(target, options.with_initial(coeffs_warm))
```

Why this is good:

- Option 2 lands you in the right basin in seconds.
- Option 1's `fmincon` polish (~1–2 minutes from a warm start, vs ~10 minutes cold) closes the surrogate-truth gap by running on the true Simscape forward.
- The leaderboard column shows both: _"surrogate seed → fmincon polish"_.

The MATLAB-side glue:

```matlab
% In option2_nn_surrogate/fit_swing_surrogate.m, optional `polish` mode.
if opts.polish
    warm = fit_via_surrogate_python(target, opts);
    result = fit_swing_fmincon(target, ...
        default_option1_options().with_initial(warm.coefficients));
else
    result = fit_via_surrogate_python(target, opts);
end
```

This is issue #030.

## What we are explicitly not doing in v1

- **Variable-length sequences.** Fixed `N=300`.
- **Subject conditioning.** One surrogate per dataset run.
- **Augmentation.** No rotation/translation augmentation in v1; flagged as future work in [DATA.md](DATA.md).
- **Probabilistic surrogate.** Deterministic regression only; no MC-dropout or ensemble. (Round-trip validation gives us the uncertainty signal we actually need.)
- **End-to-end joint training of the surrogate with downstream tasks.** Surrogate is trained once, frozen, and inverted.
