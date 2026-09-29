# Option 3 — Approach

## Why CVAE over a deterministic regressor

The naive choice is a sequence-to-vector regressor:

```
g_φ : x = (butt, clubhead, club_quat over time) ─MLP/Transformer─► θ ∈ ℝ^(n_joints·7)
loss = MSE(θ̂, θ_truth)
```

This **silently fails** on the under-determined inverse:

- For any club trajectory `x*`, there are many coefficient vectors `Θ(x*) = {θ : sim(θ) ≈ x*}`.
- The dataset contains samples `(x_i, θ_i)` where different `θ_i` produced _similar_ `x_i`.
- The MSE-optimal estimator is `g_φ(x*) = E[θ | x*]` — the **mean** of the manifold `Θ(x*)`.
- The mean of a manifold is generally **not on the manifold**. The output is a fictional "average swing" that, when fed back through Simscape, produces an unphysical trajectory unrelated to `x*`.

A CVAE solves this by modelling `p(θ | x)` as a distribution rather than a point, parameterized so each draw is _one mode_ of the manifold:

```
encoder:   q_φ(z | x, θ_truth) = N(μ_φ(x, θ), σ_φ(x, θ))     # only used during training
decoder:   p_ψ(θ | z, x)        = N(θ̂_ψ(z, x), Σ)             # used at inference
prior:     p(z) = N(0, I)
```

At inference, `θ̂ = decode(z, x)` for `z ~ p(z)` is one sample of one mode. Different `z`s give different valid coefficient vectors. We pick the best with [round-trip validation](#inference).

## Architecture

```
                       ┌───────────────────────────────────┐
   x = ClubTarget      │  1D Transformer encoder           │
   (N×12 sequence:     │  - learned positional embedding   │   h_x ∈ ℝ^d_ctx
    butt, clubhead,    │  - 4 layers, 8 heads, d_model=256 │ ─────┐
    quat)              │  - mean-pool over time            │      │
                       └───────────────────────────────────┘      │
                                                                  ▼
                       ┌─────────────────────────┐         ┌──────────────────────┐
   θ_truth (training)─►│ θ_embed: Linear         │─h_θ─┬──►│ q(z|x,θ): MLP        │──► (μ, log σ)
                       └─────────────────────────┘     │   │  - input: [h_x, h_θ] │
                                                       │   │  - output: 2×d_z     │
                                                       │   └──────────────────────┘
                                                       │              │
                                                       │              ▼ reparam
                                                       │             z ∈ ℝ^d_z
                                                       │              │
                                                       │              ▼
                       ┌──────────────────────────────────────────────────────────┐
                       │  Decoder MLP                                             │
                       │  input:  [z, h_x]                                        │
                       │  output: θ̂ ∈ ℝ^(n_joints·7)                              │  ─► θ̂
                       │  activation: per-coefficient scaled tanh to enforce      │
                       │              bounds (A,B ±1000 ; C,D ±500 ; ...)         │
                       └──────────────────────────────────────────────────────────┘
```

**Sizes (defaults; tune per Issue #032).**

| Component           | Param                                              |
| ------------------- | -------------------------------------------------- |
| `d_model` (encoder) | 256                                                |
| Encoder depth       | 4 layers, 8 heads                                  |
| `d_ctx`             | 256                                                |
| `d_z` (latent)      | 32                                                 |
| Decoder             | MLP `[d_z + d_ctx → 512 → 512 → n_joints·7]`, GELU |
| θ embed             | Linear(`n_joints·7 → 128`)                         |
| Posterior MLP       | `[d_ctx + 128 → 256 → 2·d_z]`                      |

Total parameters: ~6–10 M. Trains on a single 12 GB GPU.

### Why a transformer encoder

The club kinematic trajectory has long-range dependencies (impact-frame information must reach the address-frame attention head). A 1D conv would also work; pick transformer for:

- Better long-range mixing on `N ≈ 300` timesteps.
- Cheap to integrate with future body-marker channels (just extend the input embedding).
- Training stability with KL annealing is well-documented for transformer-encoder VAEs.

A bidirectional GRU is acceptable as a simpler v1 if the team prefers; document the choice in Issue #032.

## Loss function

```
L = β·KL(q(z|x,θ) ‖ N(0,I))           # latent regularization
  + λ_θ·MSE(θ̂, θ_truth)               # reconstruct the demonstrated coefficients
  + λ_W·|Ŵ(θ̂) − W_trial|              # match the dataset's logged total_work_J
```

| Term             | Default weight                          | Purpose                                                             |
| ---------------- | --------------------------------------- | ------------------------------------------------------------------- |
| KL               | `β`, KL-annealed `0 → 1` over 20 epochs | Forces latent toward the prior; prevents collapse                   |
| MSE on `θ`       | `λ_θ = 1.0`                             | Anchors the decoder to the demonstrated mode for this `(x, θ)` pair |
| Work regularizer | `λ_W = 1e-3`                            | Biases the decoder toward modes with realistic total work           |

### Why work regularization on top of MSE

MSE on `θ` already pushes the decoder toward the demonstrated coefficients. The work term adds a **physics-aware** signal: even if `θ̂` is far from `θ_truth` (a different mode), if its total work matches the dataset, we treat it as plausible. This is what makes the latent represent "which physical strategy" rather than "which arbitrary point in coefficient space."

`Ŵ(θ̂)` requires either (a) a forward pass through Simscape (too slow for in-loop training), (b) a learned approximator (Option 2's surrogate, if available), or (c) a closed-form approximation from the polynomial torque profiles assuming nominal joint velocities. **v1: use Option 2's forward surrogate if it has trained, else use (c)**. See Issue #032 for the choice point.

### KL annealing schedule

```
β(epoch) = clamp((epoch - 0) / 20, 0.0, 1.0)
```

Standard linear warmup. Saved alongside checkpoints so a fine-tune resumes at the right `β`.

## Inference

```
function predict_coefficients(x: ClubTarget, model, n_samples=32, validate=True):
    samples = []
    for k in 1..n_samples:
        z_k ~ N(0, I)
        θ̂_k = decode(z_k, encode(x))
        samples.append(θ̂_k)
    if not validate:
        return InverseFitResult(coefficients=samples[0], samples=samples, validated=False)
    # Round-trip validation
    best = None
    for θ̂ in samples:
        rmse = sim_fn(θ̂).rmse_against(x)
        if best is None or rmse < best.rmse: best = (θ̂, rmse)
    return InverseFitResult(coefficients=best.θ̂, final_rmse_m=best.rmse, ...)
```

### Rejection-sampling budget

| Mode               | `n_samples` | Wall clock (CPU + Simscape) | Use                              |
| ------------------ | ----------- | --------------------------- | -------------------------------- |
| Fast (no validate) | 1           | < 50 ms                     | Initial seed for Option 1 hybrid |
| Default            | 32          | ~30 s (32 × ~1 s Simscape)  | Standard fit                     |
| Thorough           | 128         | ~2 min                      | Diagnostic / leaderboard         |

The Simscape forward call is the cost driver. When Option 2's surrogate is trustworthy, a **two-stage validation** is preferred: rank all 32 samples by _surrogate_ RMSE, run real Simscape only on the top 4. This brings default-mode wall clock to ~5 s.

Acceptance threshold: round-trip club RMSE `< 10 mm` (looser than Option 2; see [ASSUMPTIONS.md §A1–A2](ASSUMPTIONS.md)). If no sample meets the threshold, return the best one and flag `validated=False, threshold_met=False` in the result struct.

## Mode-coverage diagnostic

Validating an inverse model on real targets is fundamentally limited: we don't know how many modes exist for a given `x_meas`. So we **construct** synthetic targets where multiple modes are guaranteed:

1. Pick a coefficient vector `θ_a` from the held-out set.
2. Compute the **null space of the club Jacobian** at a sequence of representative timesteps. Coefficients `θ_a + δ` for `δ` in this null space produce (approximately) the same club trajectory.
3. Generate a second `θ_b = θ_a + δ_null` and synthesize the trajectory `x*` from `θ_a`. By construction, both `θ_a` and `θ_b` are valid inverses.
4. Sample `θ̂_1, …, θ̂_K` from `g_φ(x*)`. Test passes if the samples cluster around at least two distinct points and at least one cluster is near `θ_a` and one is near `θ_b`.

Implementation note: computing the null space of a Simscape model is non-trivial; an approximation is **brute-force perturbation** — sample `θ_a + ε·v` for random `v`, run forward, keep those that produce `x*` within tolerance. Build a library of `(x*, [θ_1, θ_2, …, θ_M])` cases via this technique once and reuse for every test run.

This goes in [TESTING.md §test_multiple_samples_produce_distinct_coefficients_for_under_determined_target](TESTING.md#test_multiple_samples_produce_distinct_coefficients_for_under_determined_target).

## Hybrid with Option 1

The fastest path to a high-quality fit on a real swing:

```
x_meas → g_φ(x_meas) → θ̂₀ (CVAE warm start, ~ms)
       → fmincon(cost, x0=θ̂₀)            (~10× faster than cold start)
       → θ̂_final
```

Why it works: CVAE puts you on (or near) the right mode of the manifold. `fmincon` then polishes locally instead of wandering between basins. Empirically (anticipated): **~10× wall-clock reduction** on Option 1 with no quality loss. Implementation lives in Issue #035 and is a thin adapter — Option 1's existing `fit_swing_fmincon` already accepts `x0`.

## Future work / alternatives

### v2: Normalizing flow for exact density

A CVAE gives a learned approximate posterior. A **conditional normalizing flow** `θ = T_φ(z; x)` with `z ~ N(0,I)` gives an _exact_ density `log p_φ(θ|x)`, which lets us:

- Score multiple candidates by likelihood, not just round-trip RMSE.
- Detect out-of-distribution targets (low likelihood under all modes) cleanly.
- Combine with explicit physics priors via tempered sampling.

Cost: more training engineering, slightly more compute. Defer to v2 — the CVAE is sufficient for the four-option comparison.

### v2: Body-kinematics conditioning

When body markers come online ([CLUB_IK_SPEC.md §Status](../shared/CLUB_IK_SPEC.md)), the encoder input grows from 12 channels (club only) to ~36 channels (club + shoulder/elbow/wrist). Under-determination shrinks dramatically; the CVAE may collapse to a near-deterministic decoder, which is fine — the latent dimension can be reduced.

### v2: Active learning

Once the model is in production, log targets that fail round-trip validation. These are the **edges of the dataset's coverage**. Generate new random-sweep trials biased toward those regions, retrain. This closes the loop on [ASSUMPTIONS.md §A3](ASSUMPTIONS.md#a3-the-datasets-coefficient-distribution-is-the-de-facto-prior-over-swings).

### v2: Diffusion model

A conditional diffusion model on `θ` would also handle multi-modality and is the current SOTA for similar inverse problems in graphics and robotics. Higher inference cost (multi-step denoising) than a CVAE. Worth considering only if v1 CVAE saturates a metric we care about.
