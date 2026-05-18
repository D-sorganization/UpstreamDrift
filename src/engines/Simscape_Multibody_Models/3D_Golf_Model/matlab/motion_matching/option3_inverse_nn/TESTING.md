# Option 3 — Testing Plan

> Tests are written **first** per [shared/CODING_STANDARDS.md §TDD](../shared/CODING_STANDARDS.md#tdd-test-driven-development). Every test below has an issue label and an acceptance criterion. CI invokes `pytest -m "unit or integration" -n auto --timeout=60` against `tests/` in this folder.

## Test taxonomy

| Marker            | Speed     | When                               |
| ----------------- | --------- | ---------------------------------- |
| `unit`            | < 1 s     | Every push                         |
| `integration`     | < 60 s    | Every push                         |
| `slow`            | < 10 min  | Nightly + pre-merge to `main`      |
| `live_simulation` | unbounded | Pre-merge only; uses MATLAB Engine |

## Required tests

### `test_cvae_overfits_single_trial`

**Marker:** `unit`. **Issue:** #032.

Take **one** `(x, θ)` pair from the dataset, train for 500 iterations on it alone (effectively memorize), and assert:

- Reconstruction MSE on `θ` is near zero (< 1e-3 in normalized units).
- The model can generate `θ` to within `1e-3` from a fixed `z = μ`.

**Why.** Smoke test the gradient flow and the bound-enforcing tanh. If this fails, nothing else matters.

```python
@pytest.mark.unit
def test_cvae_overfits_single_trial(single_trial_fixture, default_cvae_config):
    model = SwingInverseCVAE(default_cvae_config)
    overfit(model, single_trial_fixture, steps=500)
    out = model(single_trial_fixture.x, single_trial_fixture.theta)
    assert torch.mean((out["theta_hat"] - single_trial_fixture.theta) ** 2) < 1e-3
```

---

### `test_held_out_round_trip_rmse_under_10mm`

**Marker:** `slow`, `live_simulation`. **Issue:** #033.

On 16 trials from the held-out test split, run `predict_coefficients(..., validate=True)` and assert the **median** round-trip clubhead RMSE is `< 10 mm`. Mean is too easy to game; median is the bar.

This will be **looser than Option 2** (Option 2 targets ~5 mm). That gap is expected because the inverse map is multi-valued; if the test passes at 10 mm we are inside the regime where the latent prior is doing its job.

```python
@pytest.mark.slow
@pytest.mark.live_simulation
def test_held_out_round_trip_rmse_under_10mm(trained_model, held_out_test_set, sim_fn):
    rmses = []
    for target, _ in held_out_test_set[:16]:
        r = predict_coefficients(target, trained_model, n_samples=32, sim_fn=sim_fn)
        rmses.append(r.final_rmse_m)
    assert np.median(rmses) < 0.010
```

---

### `test_kl_does_not_collapse`

**Marker:** `integration`. **Issue:** #032.

After full training, on a held-out batch of 64 trials, assert:

- Mean `KL(q(z|x,θ) ‖ N(0,I))` over the batch is `> 0.5 nats`.
- Per-sample KL is `> 0.05 nats` for `> 95%` of samples (no collapsed individuals).

**Why.** Mode collapse is the dominant CVAE failure (see [ASSUMPTIONS.md §A5](ASSUMPTIONS.md#a5-mode-collapse-is-the-dominant-failure-mode)). A model that fails this is still a useful regressor but is not the option we asked for, and downstream mode-coverage tests will silently pass on degenerate decoders. Block the build here.

```python
@pytest.mark.integration
def test_kl_does_not_collapse(trained_model, held_out_batch):
    out = trained_model.model(held_out_batch.x, held_out_batch.theta)
    kl = kl_divergence_diagonal(out["mu"], out["log_sigma"])  # per-sample
    assert kl.mean().item() > 0.5
    assert (kl > 0.05).float().mean().item() > 0.95
```

---

### `test_multiple_samples_produce_distinct_coefficients_for_under_determined_target`

**Marker:** `integration`. **Issue:** #034.

Use a **synthetic under-determined target** — built per [APPROACH.md §Mode-coverage diagnostic](APPROACH.md#mode-coverage-diagnostic) — that we know admits at least two coefficient vectors `θ_a`, `θ_b` producing the same club trajectory.

Sample 32 coefficient vectors from `g_φ`. Assert:

- Sample standard deviation across samples (per-coefficient, then averaged) is `> 0.05` in normalized units. A deterministic regressor would score ~0 here.
- At least two samples lie within `0.1` of `θ_a` and at least two within `0.1` of `θ_b` — the model is covering both modes.

If the second assertion is too strict in practice (mode-imbalanced datasets), relax to "the samples form ≥ 2 clusters under k-means with silhouette > 0.4". Document the relaxation in the issue.

```python
@pytest.mark.integration
def test_multiple_samples_produce_distinct_coefficients_for_under_determined_target(
    trained_model, under_determined_synthetic_target,
):
    x_target, theta_a, theta_b = under_determined_synthetic_target
    samples = trained_model.model.sample_coefficients(x_target, n_samples=32)
    samples = samples.squeeze(1).cpu().numpy()  # (32, n_joints*7)
    assert samples.std(axis=0).mean() > 0.05
    near_a = ((samples - theta_a) ** 2).sum(axis=1) ** 0.5
    near_b = ((samples - theta_b) ** 2).sum(axis=1) ** 0.5
    assert (near_a < 0.1).sum() >= 2
    assert (near_b < 0.1).sum() >= 2
```

---

### `test_predict_under_50ms_for_single_target`

**Marker:** `unit`. **Issue:** #032.

Cold start (model loaded), no validation, `n_samples=1`: `predict_coefficients` returns in `< 50 ms` on a CPU. This is the latency budget for the **fast** mode (the warm-start path into Option 1's `fmincon`). Validation passes are obviously slower; that's a different test.

```python
@pytest.mark.unit
def test_predict_under_50ms_for_single_target(trained_model_cpu, one_target):
    # Warm up
    predict_coefficients(one_target, trained_model_cpu, n_samples=1, validate=False)
    t0 = time.perf_counter()
    predict_coefficients(one_target, trained_model_cpu, n_samples=1, validate=False)
    t1 = time.perf_counter()
    assert (t1 - t0) < 0.050
```

---

### `test_round_trip_validation_filters_invalid_samples`

**Marker:** `integration`. **Issue:** #033.

Construct a deliberately-bad sample — e.g. inject a noise vector into `model.sample_coefficients` output — and verify `validate_round_trip` ranks it last. Then verify `predict_coefficients(..., validate=True)` does **not** return it.

```python
@pytest.mark.integration
def test_round_trip_validation_filters_invalid_samples(trained_model, one_target, sim_fn, monkeypatch):
    real_sample = trained_model.model.sample_coefficients
    def poisoned(*a, **kw):
        s = real_sample(*a, **kw)
        s[0] += 50.0  # garbage first sample
        return s
    monkeypatch.setattr(trained_model.model, "sample_coefficients", poisoned)
    r = predict_coefficients(one_target, trained_model, n_samples=8, sim_fn=sim_fn)
    assert r.accepted_index != 0
    assert r.final_rmse_m < r.samples[0].sim(sim_fn).rmse  # garbage sample is worse
```

## Fixtures

Live in `tests/conftest.py`. Required fixtures:

| Fixture                             | Scope                           | Description                                                               |
| ----------------------------------- | ------------------------------- | ------------------------------------------------------------------------- |
| `single_trial_fixture`              | session                         | One `(x, θ)` pair from the dataset for overfit tests                      |
| `default_cvae_config`               | session                         | `CVAEConfig` with `n_joints` matching the test dataset                    |
| `held_out_test_set`                 | session                         | Iterable of `(ClubTarget, theta)` from the held-out split                 |
| `held_out_batch`                    | session                         | Tensor batch of 64 held-out samples for KL test                           |
| `trained_model`                     | session (slow), function (fast) | Real or stub `TrainedInverseCVAE`                                         |
| `trained_model_cpu`                 | session                         | `TrainedInverseCVAE` forced to CPU for latency test                       |
| `under_determined_synthetic_target` | session                         | Triple `(x, θ_a, θ_b)` produced offline; cached                           |
| `sim_fn`                            | session                         | Simscape callback. Stub for fast tests; real Engine for `live_simulation` |
| `one_target`                        | function                        | Any `ClubTarget` for ad-hoc tests                                         |

The `under_determined_synthetic_target` fixture's offline construction is itself a script under `notebooks/` (per the deliverable list). Output is committed under `tests/data/synthetic_under_determined.pt` (~MB-scale). Documented in [RUNBOOK.md](RUNBOOK.md).

## CI invocation

From the repo root:

```bash
# Fast tests every push:
python3 -m pytest \
  src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/tests \
  -m "unit or integration" -n auto --timeout=60

# Slow / live tests pre-merge:
python3 -m pytest \
  src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option3_inverse_nn/tests \
  -m "slow or live_simulation" --timeout=600
```

Coverage is reported with `--cov=src/.../option3_inverse_nn`. Per [shared/CODING_STANDARDS.md §TDD](../shared/CODING_STANDARDS.md#tdd-test-driven-development), coverage must not decrease.

## What is **not** tested here

- Numeric agreement of the cost function with the MATLAB implementation — that lives in Issue #016 (cross-language fixture). Option 3 just calls the canonical Python cost.
- The parquet schema validation — owned by Option 2's loader (Issue #019).
- The Simscape forward simulator itself — owned by Option 1 / Option 4 (Issues #018 and #027).
