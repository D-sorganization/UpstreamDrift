# Option 1 — Direct Optimization in MATLAB

> **Read first**: [PROJECT_SPEC.md](../../../PROJECT_SPEC.md), [MATLAB_GOLF_MODEL_GUIDE.md](../../MATLAB_GOLF_MODEL_GUIDE.md), [GRIP_FIT_PLAYBOOK.md](../shared/GRIP_FIT_PLAYBOOK.md).

> **What.** Fit polynomial torque coefficients to a measured club trajectory by repeatedly running the Simscape forward simulator from a MATLAB optimizer (`fmincon`, `surrogateopt`, `MultiStart`, `particleswarm`, `ga`).
>
> **Why first.** Per-fit cost is high (minutes), per-fit setup is low (days). This is the option we ship first and the **validation oracle** the other three options must reproduce.

## Status

Greenfield. Docs scaffolded; implementation delegated to the issues listed below.

## When to use this option

- First-time fits on a new swing or new model variant.
- Ground-truthing the surrogate (Option 2) and inverse network (Option 3).
- One-off, high-fidelity fits where minutes-per-fit is acceptable.
- Debugging cost-function or club-IK regressions — this option has the fewest moving parts.

When you need to fit **hundreds of swings** in an evening, switch to Option 2 or 3 once they are trained. Option 1 stays the reference.

## What it ships

| File                              | Purpose                                                                                                        |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| `fit_swing_fmincon.m`             | Single-start gradient (SQP) fit. Cheapest, locally optimal.                                                    |
| `fit_swing_multistart.m`          | `MultiStart` over N random starts, parallelized via `parsim`/`parfor`.                                         |
| `fit_swing_surrogateopt.m`        | Global gradient-free fit using `surrogateopt`.                                                                 |
| `fit_swing_hybrid.m`              | `surrogateopt` (or `particleswarm`) → `fmincon` polish. The recommended default.                               |
| `default_option1_options.m`       | Returns the canonical options struct; everything overrides from here.                                          |
| `OptimizationProgressDashboard.m` | Handle class, live dashboard during fits.                                                                      |
| `private/`                        | Internal helpers (not callable from outside this folder).                                                      |
| `tests/`                          | `matlab.unittest.TestCase` suite. Tests-first per [shared/CODING_STANDARDS.md](../shared/CODING_STANDARDS.md). |
| `visualization/`                  | Option-specific viz (see [VISUALIZATION.md](VISUALIZATION.md)).                                                |

The contracts are in [INTERFACES.md](INTERFACES.md). The algorithm is in [APPROACH.md](APPROACH.md). The assumptions are in [ASSUMPTIONS.md](ASSUMPTIONS.md). How to run it is in [RUNBOOK.md](RUNBOOK.md). What to test is in [TESTING.md](TESTING.md).

## Dependencies (toolboxes)

- **MATLAB R2019b+** (for `arguments` blocks).
- **Simulink + Simscape Multibody** (the forward simulator: [GolfSwing3D_Kinetic.slx](../../src/model/GolfSwing3D_Kinetic.slx)).
- **Optimization Toolbox** — `fmincon`.
- **Global Optimization Toolbox** — `MultiStart`, `surrogateopt`, `particleswarm`, `ga`.
- **Parallel Computing Toolbox** (recommended) — `parsim`, `parfor`, `parpool` for `MultiStart` and `surrogateopt`'s `'UseParallel'` mode.

If a required toolbox is missing the entry-point function must error cleanly with a pointer to the install instructions; see [INTERFACES.md](INTERFACES.md).

## Shared dependencies (issues #013–#023)

This option **does not** re-implement these — it consumes them from `motion_matching/shared/`.

- `compute_cost.m` (issue #015) — see [COST_FUNCTION_SPEC.md](../shared/COST_FUNCTION_SPEC.md).
- `load_club_target_excel.m` / `load_club_target_c3d.m` / `synthesize_target_from_coefficients.m` (issues #013, #014) — see [CLUB_IK_SPEC.md](../shared/CLUB_IK_SPEC.md).
- `simulate_with_coefficients.m` (issue #018) — the Simscape forward callback.
- `getPolynomialParameterInfo.m` — already exists at [matlab/src/functions/dataset_generator/getPolynomialParameterInfo.m](../../src/functions/dataset_generator/getPolynomialParameterInfo.m).
- `generateRandomCoefficients.m` — already exists at [matlab/src/functions/dataset_generator/generateRandomCoefficients.m](../../src/functions/dataset_generator/generateRandomCoefficients.m).
- Visualization entry points (issue #023) — see [VISUALIZATION_SPEC.md](../shared/VISUALIZATION_SPEC.md).

## GitHub issues for Option 1

| #        | Title                                   | Notes                                                                       |
| -------- | --------------------------------------- | --------------------------------------------------------------------------- |
| **#024** | `fit_swing_fmincon`                     | Single-start SQP. Acceptance: passes `test_fits_synthetic_to_within_1mm`.   |
| **#025** | `fit_swing_multistart`                  | `MultiStart` + parallel. Acceptance: outperforms #024 on multimodal target. |
| **#026** | `fit_swing_surrogateopt` and hybrid     | `surrogateopt` standalone + the `surrogateopt → fmincon` polish.            |
| **#027** | `OptimizationProgressDashboard` and viz | Live dashboard + `MultiStartParallelCoords` + final summary card.           |

Shared infrastructure consumed by Option 1: issues #013 (C3D loader), #014 (synthetic target), #015 (`compute_cost`), #018 (`simulate_with_coefficients`), #023 (shared visualization).

## Open questions for the human (flagged by the docs in this folder)

- See [ASSUMPTIONS.md](ASSUMPTIONS.md) for the full list. Headline items:
  - Confirm the parallel pool size policy (per-host vs CI-bounded).
  - Confirm `lambda` default of `1e-4` produces physiologically plausible swings on `TW_ProV1`. The agent will need to sweep `lambda` once a real fit is available.
  - Confirm the impact-anchor weight schedule (`w_a`: ramp from 0 → 10·w_p over the first 25% of iterations) — see [APPROACH.md](APPROACH.md#impact-anchor-schedule).
