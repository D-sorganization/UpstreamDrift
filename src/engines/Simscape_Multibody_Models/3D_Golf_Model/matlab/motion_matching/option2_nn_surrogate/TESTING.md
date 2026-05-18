# Option 2 — Testing

TDD plan. Tests are written first; implementation lands in the same PR per [shared/CODING_STANDARDS.md § TDD](../shared/CODING_STANDARDS.md#tdd-test-driven-development). All tests are `pytest` under `option2_nn_surrogate/tests/` (or the canonical mirror at `tests/motion_matching/option2/`, depending on which the repo conventions land on at issue #028 time — confirm before merging).

Run from repo root:

```bash
python3 -m pytest src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/option2_nn_surrogate/tests -n auto --timeout=60
```

Markers (see [`CLAUDE.md`](../../../../../../../CLAUDE.md)): `unit`, `integration`, `slow`, `requires_gl`, `live_simulation`. Inversion + Simscape round-trip tests are `integration` and may be `slow`.

## Required tests

Each test below maps to an acceptance bullet on issues #028–#031. **No PR merges without all of these green.**

### `test_surrogate_predicts_training_trial_within_2mm`

```python
@pytest.mark.unit
def test_surrogate_predicts_training_trial_within_2mm(tiny_dataset, trained_overfit_surrogate):
    """A surrogate trained on N=8 trials must overfit them trivially.

    If this fails, the architecture, the loss, or the data pipeline is broken —
    independent of any held-out generalization concern.
    """
    trial = tiny_dataset.get_train_trial(0)
    pred = trained_overfit_surrogate(trial.coeffs.unsqueeze(0))
    butt_rmse = ((pred.butt - trial.butt_true) ** 2).mean().sqrt()
    clubhead_rmse = ((pred.clubhead - trial.clubhead_true) ** 2).mean().sqrt()
    assert butt_rmse.item() < 2e-3, f"butt RMSE {butt_rmse:.4f} m > 2 mm"
    assert clubhead_rmse.item() < 2e-3, f"clubhead RMSE {clubhead_rmse:.4f} m > 2 mm"
```

Fixture: `trained_overfit_surrogate` trains for ~2000 steps on 8 trials with no regularization. Should hit < 1 mm easily; the 2 mm threshold gives slack.

### `test_surrogate_held_out_rmse_under_5mm`

```python
@pytest.mark.integration
@pytest.mark.slow
def test_surrogate_held_out_rmse_under_5mm(full_dataset, trained_full_surrogate):
    """Held-out RMSE must clear the production threshold (5 mm clubhead).

    Threshold matches APPROACH.md § Stopping and is the issue #028 acceptance bar.
    """
    metrics = evaluate_surrogate(trained_full_surrogate, full_dataset.test_split)
    assert metrics.clubhead_rmse_m < 5e-3
    assert metrics.butt_rmse_m < 5e-3
    assert metrics.quat_geo_deg < 2.0
```

### `test_surrogate_gradient_finite`

```python
@pytest.mark.unit
def test_surrogate_gradient_finite(trained_overfit_surrogate):
    """∂(loss)/∂coeffs must be finite and non-vanishing.

    This is the load-bearing assumption for inversion. Any architectural choice
    that breaks differentiability (hard clip, argmax, non-differentiable
    preprocess) gets caught here, not three weeks later in production.
    """
    coeffs = torch.randn(1, trained_overfit_surrogate.cfg.n_joints * 7, requires_grad=True)
    pred = trained_overfit_surrogate(coeffs)
    loss = pred.butt.pow(2).mean() + pred.clubhead.pow(2).mean()
    loss.backward()
    g = coeffs.grad
    assert torch.isfinite(g).all(), "gradients contain NaN or Inf"
    assert g.abs().max().item() > 1e-10, "gradient is degenerate (all near-zero)"
    # Also verify the quaternion path: q gradient should not be zero.
    coeffs2 = torch.randn(1, trained_overfit_surrogate.cfg.n_joints * 7, requires_grad=True)
    pred2 = trained_overfit_surrogate(coeffs2)
    q_loss = (1.0 - pred2.q_club[..., 0]).mean()
    q_loss.backward()
    assert torch.isfinite(coeffs2.grad).all()
    assert coeffs2.grad.abs().max().item() > 1e-10
```

### `test_inversion_recovers_known_coeffs_on_synthetic_target`

```python
@pytest.mark.integration
def test_inversion_recovers_known_coeffs_on_synthetic_target(
    trained_full_surrogate, synth_target_factory
):
    """fit(synthesize(θ)) ≈ θ in surrogate-loss space.

    We synthesize a target *from the surrogate itself* with known θ_truth, then
    invert. This isolates the inversion algorithm from surrogate generalization
    error — round-trip through Simscape is a separate test.
    """
    coeffs_truth = sample_inside_bounds(seed=0)
    target = synth_target_factory.from_surrogate(trained_full_surrogate, coeffs_truth)
    result = fit_swing_via_surrogate(target, trained_full_surrogate, InvertOptions(seed=0))
    err = np.linalg.norm(result.coefficients - coeffs_truth) / np.linalg.norm(coeffs_truth)
    assert err < 0.05, f"relative coefficient error {err:.3f} > 5%"
    assert result.surrogate_rmse_m < 1e-3
```

Note: with K-restart=8 and a multi-modal forward map, this can find a different basin that produces equivalent kinematics. We allow that — the assertion is on **kinematic** match (`surrogate_rmse_m < 1 mm`), with the coefficient-error bound as a softer secondary check that we relax if the multi-modality is real.

### `test_round_trip_validation_catches_extrapolation`

```python
@pytest.mark.integration
@pytest.mark.live_simulation
def test_round_trip_validation_catches_extrapolation(
    trained_full_surrogate, simscape_sim_fn, adversarial_target_factory
):
    """When the surrogate is extrapolating, validate_against_simscape must flag it.

    The adversarial_target_factory builds a target whose best-fitting coefficients
    lie just outside the random-sweep bounds (or in a low-density region). The
    surrogate happily fits it; Simscape disagrees.
    """
    adv_target = adversarial_target_factory.build()
    fit = fit_swing_via_surrogate(adv_target, trained_full_surrogate, InvertOptions())
    report = validate_against_simscape(
        fit, simscape_sim_fn, extrapolation_factor=2.0
    )
    assert report.is_extrapolation is True
    assert report.flag == "extrapolation"
    assert report.simscape_rmse_m > 2.0 * report.surrogate_rmse_m
```

This is the **honest** test of Option 2's safety. Without it, the surrogate is unfalsifiable.

### `test_fit_completes_under_30s_for_typical_target`

```python
@pytest.mark.integration
def test_fit_completes_under_30s_for_typical_target(trained_full_surrogate, typical_target):
    """A typical fit (8 restarts × 200 iters) finishes well under 30 seconds.

    The promise of Option 2 is seconds-per-fit. This test enforces it.
    """
    t0 = time.perf_counter()
    fit_swing_via_surrogate(trained_full_surrogate, typical_target, InvertOptions())
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"fit took {elapsed:.1f}s, budget is 30s"
```

CPU CI slack: bumped to 30 s rather than 10 s to accommodate runners without GPU. APPROACH.md targets ~8 s on CPU; the test threshold is 4× that to absorb noise.

## Recommended additional tests

Not required for issue acceptance but expected before declaring v1 done.

| Test                                            | Purpose                                                 | Marker            |
| ----------------------------------------------- | ------------------------------------------------------- | ----------------- |
| `test_normalization_stats_round_trip`           | denormalize(normalize(x)) == x                          | unit              |
| `test_quaternion_canonicalization_idempotent`   | canonicalize(canonicalize(q)) == canonicalize(q)        | unit              |
| `test_bound_projection_keeps_iterates_inside`   | clamped Adam step never exits the box                   | unit              |
| `test_dataset_split_no_trial_id_leakage`        | train ∩ val ∩ test = ∅                                  | unit              |
| `test_checkpoint_round_trip`                    | train → save → load → predict identical                 | unit              |
| `test_polish_handoff_matches_option1_format`    | hybrid result struct conforms to CODING_STANDARDS.md    | integration       |
| `test_matlab_shim_produces_valid_result_struct` | `fit_swing_surrogate.m` round-trips through `pyrunfile` | integration, slow |

## Fixtures

Lives in `tests/conftest.py`:

- `tiny_dataset` — 8 trials, in-memory, deterministic. Fast.
- `full_dataset` — points at the real parquet (skipped if not present).
- `trained_overfit_surrogate` — session-scoped, 2000-step train on `tiny_dataset`.
- `trained_full_surrogate` — session-scoped, **expensive**: trains for `early_stop_patience` evals or 50k steps on `full_dataset`. Cached to `tests/_cache/` keyed by `(dataset_sha, train_config_hash, code_sha)`.
- `synth_target_factory` — builds `ClubTarget` from a coeffs vector via the surrogate (fast) or via Simscape (slow, `live_simulation`).
- `adversarial_target_factory` — builds targets at the dataset boundary; see APPROACH.md.
- `simscape_sim_fn` — `live_simulation`-marked, requires MATLAB. Skipped on CI.
- `typical_target` — a held-out trial converted to `ClubTarget`. Used as the realistic benchmark.

## Skip policy

If the parquet dataset is not present in `data/sweep/`, skip every `integration` test with a clear message. Unit tests (gradient, normalization, contracts) must pass even without the dataset.

```python
@pytest.fixture(scope="session")
def full_dataset():
    if not DATASET_PATH.exists():
        pytest.skip(f"Sweep dataset not found at {DATASET_PATH}; see option2/DATA.md")
    return load_sweep_dataset(DATASET_PATH)
```

## Coverage

Targets from [`CLAUDE.md`](../../../../../../../CLAUDE.md): coverage must not decrease. Option 2 ships with branch coverage on `surrogate.py`, `train.py`, `invert.py`, `validate.py`, `dataset.py`. The MATLAB shim is exercised by an `integration` test that runs through `pyrunfile`; if MATLAB isn't available on a runner, the test skips with `pytest.importorskip` semantics on a `matlab_engine` shim.

## What NOT to test (yet)

- **Surrogate generalization across subjects.** v1 is single-subject; cross-subject is a v2 concern.
- **GPU vs CPU numerical parity beyond fp32 tolerance.** We pin a tolerance; we do not chase byte-for-byte determinism across devices.
- **MATLAB-side performance.** The shim's job is correctness, not speed; the Python side owns wall-clock.
