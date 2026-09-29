# Option 2 — Assumptions

Explicit list. Every item below is a load-bearing assumption; if it breaks, the option breaks. Cross-cutting assumptions are in [shared/README.md](../shared/README.md).

## A1. Surrogate validity is bounded by the training distribution

**Statement.** `f_θ(coeffs)` is only trustworthy when `coeffs` lies inside the random-sweep coefficient envelope used to generate the parquet dataset.

The random-sweep bounds are defined by [generateRandomCoefficients.m](../../src/functions/dataset_generator/generateRandomCoefficients.m): `A,B ∈ ±1000; C,D ∈ ±500; E,F ∈ ±100; G ∈ ±25`. Anything outside is **extrapolation** and the surrogate's predictions are undefined.

**Consequences.**

- The Adam inversion in `fit_swing_via_surrogate` **must project** onto these bounds at every step. See [APPROACH.md § Inversion](APPROACH.md#inversion).
- Round-trip validation (run the fitted coefficients through Simscape) is **mandatory** before publishing a fit. See [TESTING.md § round-trip](TESTING.md#round-trip-validation).
- A measured swing whose physiologically-plausible coefficients lie _outside_ the random-sweep envelope **cannot be fit by Option 2 alone** — fall back to Option 1 or expand the envelope and regenerate the dataset.

## A2. The surrogate is differentiable w.r.t. its input

**Statement.** `∂f_θ(coeffs) / ∂coeffs` exists, is finite, and is informative (not vanishing) over the working range.

This is non-negotiable: it is the entire point of Option 2. Architectural choices that violate this (e.g., a hard-clipped output head, `argmax` in the body, a non-differentiable preprocessing step) are rejected.

**Validated by.** `test_surrogate_gradient_finite` in [TESTING.md](TESTING.md).

## A3. Output kinematic representation matches the cost function

**Statement.** The surrogate emits exactly what `compute_cost` consumes per [shared/CLUB_IK_SPEC.md § Output schema](../shared/CLUB_IK_SPEC.md#output-schema-canonical-target-struct):

- `time` — fixed simulation timegrid, `T = 0.3 s`, default `sample_rate = 1000 Hz` → `N = 300` samples (or 301 if endpoint-inclusive; pick one and stick with it; default is 300).
- `r_butt(t) ∈ ℝ³` (metres, world frame).
- `r_clubhead(t) ∈ ℝ³` (metres, world frame).
- `q_club(t) ∈ S³` (unit quaternion, `[w, x, y, z]`, sign-normalized so `w ≥ 0`).

The surrogate **does not** emit joint angles `q(t)`, joint velocities `qd(t)`, or torques `τ(t)` from its primary head. Those are produced by an **auxiliary head** (see [APPROACH.md § Loss](APPROACH.md#loss)) and are used only for debugging — **never** consumed by the cost function.

## A4. Time grid is fixed across the dataset

**Statement.** Every trial in the parquet dataset uses the same `(T, sample_rate, N)`. The surrogate models a fixed-length sequence per trial; variable-length is not supported in v1.

If the dataset contains heterogeneous time grids, the loader resamples to the canonical grid before training. See [DATA.md § Preprocessing](DATA.md#preprocessing).

## A5. Quaternion handling is sign-normalized

**Statement.** All quaternions (target, surrogate output, dataset) follow the convention from `CLUB_IK_SPEC.md`: `[w, x, y, z]`, unit-norm, `w ≥ 0`. The surrogate's quaternion head **must** project to this canonical sign before computing loss; the quaternion-aware loss must use `1 − |q_pred · q_true|²` so the `q ↔ −q` ambiguity does not produce a phantom gradient.

## A6. The dataset's coefficient ordering matches the model's joint ordering

**Statement.** `coefficients` in `trials.parquet` is a flat vector of length `n_joints × 7`, ordered `[joint_0_A, joint_0_B, ..., joint_0_G, joint_1_A, ...]` matching `joint_names × [A,B,C,D,E,F,G]`.

This must agree with what `getPolynomialParameterInfo.m` emits. **The loader cross-checks** (issue #019); if there is a mismatch, training is aborted.

## A7. No solver-failed trials in training

**Statement.** Trials with `solver_status != "success"` are excluded from training. The surrogate is trained only on physically-realized swings; including failed trials biases the model toward NaN-flavored outputs.

The dataset loader filters these by default; can be overridden for debugging.

## A8. Float32 is sufficient for training; float64 for inversion

**Statement.** Training uses `float32` (mixed precision where available). The inversion loop in `fit_swing_via_surrogate` uses `float32` for the surrogate forward and `float64` for the optimization state. Loss is reported in metres (the user-meaningful unit).

The 1 mm noise floor cited in [shared/COST_FUNCTION_SPEC.md § Numerical considerations](../shared/COST_FUNCTION_SPEC.md#numerical-considerations) is well within `float32` precision; we are not chasing below that.

## A9. The Simscape forward simulator is the ground truth

**Statement.** `simulate_with_coefficients` (issue #018) is the oracle. If the surrogate disagrees with Simscape on a trial, **Simscape wins**. The surrogate is an approximation by design.

Round-trip validation (Adam-fit using the surrogate, then run the resulting coefficients through Simscape) is the only honest measure of fit quality and is **required** before any result is published.

## A10. PyTorch is the framework

Per repo convention and `requirements.lock`. JAX and TensorFlow are out of scope for v1.

## A11. The surrogate is single-subject for v1

**Statement.** v1 trains one surrogate per subject (or one global surrogate if the dataset is single-subject). Subject-conditioning (e.g., a subject-id embedding concatenated with the coefficients) is a v2 feature.

This matches the user's stated framing and avoids over-engineering.

## A12. Cost function parity with Option 1

**Statement.** When Option 2 reports `final_rmse_m` it uses **the same `compute_cost` implementation** Option 1 uses, called on the round-tripped Simscape result — not on the surrogate's prediction. This is the only way the leaderboard in [shared/VISUALIZATION_SPEC.md § Comparison across options](../shared/VISUALIZATION_SPEC.md#comparison-across-options) compares apples to apples.

## A13. Reproducibility

**Statement.** A trained surrogate is reproducible from `(dataset_run_id, train_config, seed, git_commit)`. The training pipeline records all four in the checkpoint metadata. A fit is reproducible from `(checkpoint_id, target_hash, invert_options, seed)`.

This matches the provenance requirements in [shared/CODING_STANDARDS.md § Provenance and reproducibility](../shared/CODING_STANDARDS.md#provenance-and-reproducibility).

## Open questions for the human

- A1 — confirm the random-sweep bounds in `generateRandomCoefficients.m` are still authoritative; if the dataset was generated with different bounds, document the actual envelope here.
- A4 — confirm `(T, sample_rate, N) = (0.3 s, 1000 Hz, 300)`.
- A5 — confirm the quaternion sign convention. The Excel mocap has its own convention; see `mocap_data_loader.py`.
- A11 — confirm v1 is single-subject; if not, surface the subject-id schema before training.
