# Option 3 — Direct Inverse Neural Network (CVAE)

> **Read first**: [PROJECT_SPEC.md](../../../PROJECT_SPEC.md), [MATLAB_GOLF_MODEL_GUIDE.md](../../MATLAB_GOLF_MODEL_GUIDE.md), [GRIP_FIT_PLAYBOOK.md](../shared/GRIP_FIT_PLAYBOOK.md).

> **One-line.** Train `g_φ : club_kinematic_trajectory → torque_coefficients` end-to-end on the random-sweep parquet dataset; one forward pass produces a candidate fit in milliseconds. Because the inverse map is **one-to-many** when only the club is observed, `g_φ` is implemented as a **Conditional Variational Autoencoder (CVAE)** whose latent disambiguates between the multiple coefficient vectors that produce the same club trajectory.

## Status

Greenfield. This folder contains documentation and a single skeleton `.py` only. Implementation is delegated to issues [#032, #033, #034, #035](#github-issues). No production logic should land in this folder until the issues open.

## When to use it

- You already have Option 2 trained (its parquet preprocessing is reused — see [Dependency on Option 2](#dependency-on-option-2)).
- You need the **lowest possible per-fit latency** (~ms / target).
- You can afford a Simscape round-trip _after_ the prediction to validate it.
- The target swing falls inside the dataset's coverage (random sweep around the model's nominal coefficients). Out-of-distribution targets fall back to Option 1 or Option 2.

## When NOT to use it

- You have a single high-stakes fit and need provable optimality → use Option 1.
- The dataset doesn't yet exist or is too small (< ~10k trials) → wait, or use Option 2 with stronger regularization.
- You cannot afford a Simscape round-trip on the prediction → don't ship raw `g_φ` output to downstream consumers; the inverse model is **not** trustworthy without round-trip validation. See [ASSUMPTIONS.md](ASSUMPTIONS.md).

## What this option does, mechanically

1. Load the random-sweep dataset (Option 2's loader; see [DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md)).
2. Train a CVAE: encoder consumes the club kinematic sequence and emits `q(z | x)`; decoder consumes `(z, x)` and emits coefficient vector `θ̂`.
3. At inference: sample `z ~ N(0, I)` (or `q(z|x)` if the target is in-distribution), decode, **round-trip the prediction through Simscape**, accept if club RMSE < threshold, else resample. See [APPROACH.md §Inference](APPROACH.md#inference) for the budget.
4. Optionally: hand the accepted `θ̂` to Option 1's `fmincon` as a warm start.

## Dependency on Option 2

Option 3 **shares Option 2's preprocessing pipeline** for the parquet dataset:

- `SweepDataset` loader (see [DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md), Issue #019).
- Per-trial sequence batching, normalization statistics, and train/val/test split logic.
- The same `joint_names` ordering and the same coefficient-vector flattening (`n_joints × 7`).

**Do not duplicate the loader.** Option 3 imports it from `option2_nn_surrogate/data/` (or, if it has been promoted, from `motion_matching/shared/python/`). See [INTERFACES.md](INTERFACES.md) for the import surface.

If Option 2 changes its preprocessing, Option 3 must be retrained — that is the price of the dependency, and we accept it.

## Files in this folder

```
option3_inverse_nn/
├── README.md            (this file)
├── ASSUMPTIONS.md       (what we are betting on; what breaks the option)
├── APPROACH.md          (CVAE rationale, architecture, loss, inference, alternatives)
├── INTERFACES.md        (Python signatures with @precondition / @postcondition)
├── TESTING.md           (TDD plan; concrete test names)
├── VISUALIZATION.md     (option-specific views beyond the shared three)
├── RUNBOOK.md           (literal commands for train / predict / round-trip)
├── inverse_cvae.py      (skeleton ONLY — class signature + docstrings, no bodies)
├── tests/.gitkeep
├── models/.gitkeep
└── notebooks/.gitkeep
```

The skeleton `.py` is the **only** code allowed to land in this folder before the issues open.

## Relationship to the other options

| Direction                | Detail                                                                                                                                                     |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **From Option 1**        | Option 1 is the validation oracle. Use Option 1 to fit the same target; if Option 3's RMSE is within ~5x of Option 1's RMSE, the inverse model is healthy. |
| **From Option 2**        | Reuses the dataset loader, normalization, and train/val/test split. Cannot start until Option 2's dataset pipeline is green (Issue #019, #024).            |
| **To Option 1 (hybrid)** | `g_φ` output as `x0` for `fmincon` cuts Option 1's wall-clock by ~10x on average. See [APPROACH.md §Hybrid](APPROACH.md#hybrid-with-option-1).             |
| **To Option 4**          | Once Option 4's `SimscapeAdapter` is up, the round-trip validation step in Option 3 should call it instead of MATLAB Engine directly.                      |

## Acceptance bar (for "Option 3 is done")

1. Held-out round-trip RMSE on club position is **< 10 mm** on the validation split (looser than Option 2 — see [ASSUMPTIONS.md](ASSUMPTIONS.md)).
2. Single-target prediction (no round-trip) runs in **< 50 ms** on a CPU.
3. Latent space does not collapse (`KL(q(z|x) ‖ N(0,I)) > 0.5 nats` on a held-out batch).
4. Multiple samples for the same target produce **distinct** coefficient vectors, demonstrating mode coverage.
5. All three shared views (trajectory overlay, error timecourse, fit quality card) render from the result struct without glue code.

## GitHub issues

Reserved for this option:

| #    | Title                                                        | Scope                                                                   |
| ---- | ------------------------------------------------------------ | ----------------------------------------------------------------------- |
| #032 | Option 3 — CVAE architecture and training loop               | `inverse_cvae.py`, `train_inverse_cvae`, training tests                 |
| #033 | Option 3 — Round-trip validation and rejection sampling      | `predict_coefficients`, `validate_round_trip`, Simscape callback wiring |
| #034 | Option 3 — Latent diagnostics and mode-coverage tests        | t-SNE/UMAP plots, null-space synthetic targets, mode-coverage test      |
| #035 | Option 3 — Hybrid handoff to Option 1 (`fmincon` warm start) | Adapter from `InverseFitResult.coefficients` to Option 1's `x0`         |

Each issue's acceptance criteria reference specific tests in [TESTING.md](TESTING.md).

## Open questions for the user

- The user has stated each timestep is treated as a separate `(kinematics, torques)` sample. Option 3 instead treats **a full trial as one sample** because the inverse map at trial scope is what matters for swing-matching. Confirm both views can coexist: Option 3 may need a different parquet read pattern than Option 2's per-timestep one.
- How many trials in the random-sweep dataset? CVAE training typically wants `≥ 20k` trials before mode coverage is reasonable. If we have less, the latent prior fights the data and rejection sampling carries more weight.
- Should the encoder also consume the body kinematics (Phase 2) or only club (Phase 1)? Phase 1 is harder (more under-determined) but matches the cost function spec. Default: Phase 1, with a config flag to add body channels later.
