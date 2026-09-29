# Option 2 — Differentiable NN Forward Surrogate

> **Read first**: [PROJECT_SPEC.md](../../../PROJECT_SPEC.md), [MATLAB_GOLF_MODEL_GUIDE.md](../../MATLAB_GOLF_MODEL_GUIDE.md), [GRIP_FIT_PLAYBOOK.md](../shared/GRIP_FIT_PLAYBOOK.md).

> **What.** Train a neural network `f_θ : coefficients → club_kinematic_trajectory` on the random-sweep parquet dataset, then fit a measured swing by gradient descent on the input coefficients (Adam) with the surrogate weights frozen.
>
> **Why.** Per-fit cost drops from minutes (Option 1) to **seconds**. Setup cost is medium-to-high — needs a trained surrogate, which needs the parquet dataset.

## Status

Greenfield. Docs scaffolded; implementation delegated to issues #028–#031. **Blocked on the parquet dataset landing in the repo** — see [Dependencies](#dependencies).

## When to use this option

- High-volume fitting after the surrogate is trained (sweep many swings, many subjects).
- Warm-starting Option 1 (`fmincon` polish) — see [APPROACH.md § Hybrid](APPROACH.md#hybrid-handoff-to-option-1).
- Sensitivity analysis: differentiable `∂kinematics/∂coefficients` is useful in its own right.

When **not** to use it:

- A single one-off fit on a swing far outside the training distribution — Option 1 is safer.
- Before the surrogate has been validated against Simscape on held-out trials. See [TESTING.md](TESTING.md).

## What it ships

| File                    | Purpose                                                                                                                    |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `surrogate.py`          | `SwingSurrogate(nn.Module)` — the differentiable forward model. Skeleton lives in this folder; agents fill in.             |
| `train.py`              | `train_surrogate(dataset, config) -> TrainedSurrogate`. AdamW + cosine schedule + mixed precision.                         |
| `invert.py`             | `fit_swing_via_surrogate(target, surrogate, opts) -> FitResult`. Adam over coefficients with bound projection + restarts.  |
| `validate.py`           | `validate_against_simscape(result, sim_fn) -> ValidationReport`. Round-trips every fit through the true forward simulator. |
| `dataset.py`            | Dataset adapter wrapping `shared/load_sweep_dataset` (#019) into a PyTorch `Dataset`.                                      |
| `config.py`             | `TrainConfig`, `InvertOptions`, `SurrogateConfig` dataclasses.                                                             |
| `fit_swing_surrogate.m` | MATLAB shim that calls the Python entry-point via `pyrunfile`.                                                             |
| `tests/`                | `pytest` suite per [TESTING.md](TESTING.md).                                                                               |
| `models/`               | Trained checkpoint store. Contents `.gitignore`'d (only `.gitkeep` is tracked).                                            |
| `notebooks/`            | Exploration notebooks. Contents `.gitignore`'d.                                                                            |
| `visualization/`        | Option-specific viz (training curves, residual histograms, inversion progress).                                            |

The contracts are in [INTERFACES.md](INTERFACES.md). The algorithm is in [APPROACH.md](APPROACH.md). The assumptions are in [ASSUMPTIONS.md](ASSUMPTIONS.md). How to run it is in [RUNBOOK.md](RUNBOOK.md). What to test is in [TESTING.md](TESTING.md). The dataset dependency is in [DATA.md](DATA.md).

## Dependencies

### Hard dependency: parquet dataset

Option 2 cannot start training until the random-sweep parquet dataset is in the repo. The proposed schema is in [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md). The user has stated they will copy it in; until then this folder is documentation only.

**Flag for the human.** Confirm:

1. The dataset matches the schema in `DATASET_SCHEMA.md` (or update the schema with deltas).
2. How many trials are in the latest run (`20251030`).
3. Is it actually parquet, or HDF5? See open questions in `DATASET_SCHEMA.md`.

### Python toolchain

- **Python 3.10+** (per repo [`CLAUDE.md`](../../../../../../../CLAUDE.md)).
- **PyTorch** — already pinned in `requirements.lock`. CUDA-optional; CPU works for ≤ 5k trials but is slow.
- **polars** or **pandas** + **pyarrow** for parquet I/O (already in the repo).
- **TensorBoard** (optional, for training curves; matplotlib fallback documented in [VISUALIZATION.md](VISUALIZATION.md)).

### MATLAB ↔ Python bridge

`fit_swing_surrogate.m` calls Python via `pyrunfile`. The MATLAB-side prerequisites:

- MATLAB R2021b+ (for `pyrunfile`).
- The Python environment from `requirements.lock` must be active in MATLAB's `pyenv()`.

### Shared dependencies (issues #013–#023)

Option 2 consumes — does not re-implement:

- `compute_cost.m` / `cost.py` (issue #015) — see [shared/COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md).
- `load_club_target_*` (issues #013, #014, #017) — see [shared/CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).
- `simulate_with_coefficients.m` (issue #018) — the Simscape forward callback used for round-trip validation.
- `load_sweep_dataset` (issue #019) — see [shared/DATASET_SCHEMA.md](../shared/DATASET_SCHEMA.md).
- Visualization entry points (issue #023) — see [shared/VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md).

## GitHub issues for Option 2

| #        | Title                                                          | Notes                                                                                                                     |
| -------- | -------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **#028** | `SwingSurrogate` model + `train_surrogate`                     | Architecture + training loop. Acceptance: held-out RMSE < 5 mm. See [APPROACH.md](APPROACH.md).                           |
| **#029** | `fit_swing_via_surrogate` (Adam over coefficients)             | Differentiable inversion with bound projection + K-restart. Acceptance: recovers known coefficients on synthetic targets. |
| **#030** | `validate_against_simscape` + hybrid Option-2→Option-1 handoff | Round-trip oracle and warm-start to `fmincon`.                                                                            |
| **#031** | `fit_swing_surrogate.m` MATLAB shim + viz                      | `pyrunfile` integration; training curves; residual hists; inversion progress plot.                                        |

Shared infrastructure consumed by Option 2: issues #013, #014, #015, #017, #018, #019, #023.

## Open questions for the human

- See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full list. Headlines:
  - **Dataset format.** Parquet vs HDF5 — confirm what's actually shipping.
  - **Coefficient bounds.** Surrogate is only valid inside the random-sweep bounds. Confirm the bounds in `generateRandomCoefficients.m` match the dataset's actual sampling envelope.
  - **Architecture.** [APPROACH.md](APPROACH.md) recommends FiLM-conditioned MLP over 1D-CNN — confirm before agents start training.
  - **Surrogate-truth gap budget.** What `surrogate_rmse / simscape_rmse` ratio constitutes "extrapolation rejected"? Default is `2.0×`; see [TESTING.md § round-trip](TESTING.md#round-trip-validation).
