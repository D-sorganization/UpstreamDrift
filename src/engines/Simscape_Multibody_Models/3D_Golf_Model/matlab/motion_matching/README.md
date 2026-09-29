# Motion Matching for the 3D Simscape Kinetic Golf Model

> **Goal.** Given a measured club trajectory (and optionally body kinematics), find polynomial torque coefficients such that, when fed to `GolfSwing3D_Kinetic.slx`, the simulated club trajectory reproduces the measurement. The problem is under-determined when only the club is observed; we resolve this by adding a regularizer (typically minimum total mechanical work).

## Status

Greenfield. Nothing in this folder has executable code yet. Folder structure, contracts, and tests are scaffolded; implementation is delegated to parallel agents working off the GitHub issues listed at the bottom of this README.

## Why four options in parallel

The user has explicitly asked for four parallel pathways so the team can compare what works. They are not redundant — each makes different assumptions, has different cost/risk/iteration-speed profiles, and is useful in different regimes:

| #   | Option                                                                                            | Per-fit cost            | Setup cost                                 | Best for                                                      |
| --- | ------------------------------------------------------------------------------------------------- | ----------------------- | ------------------------------------------ | ------------------------------------------------------------- |
| 1   | Direct MATLAB optimization (`fmincon` / `surrogateopt` / `MultiStart`) over Simscape forward sims | High (~minutes per fit) | Low (~days)                                | First fits, ground-truth, validation oracle                   |
| 2   | NN forward surrogate `f_θ: coeffs → kinematics`, then differentiable inversion                    | Low (~seconds per fit)  | Medium (~weeks; needs the parquet dataset) | High-volume fitting once trained                              |
| 3   | NN inverse model `g_φ: kinematics → coeffs`                                                       | Lowest (~ms per fit)    | Medium-high (multi-modal, needs CVAE)      | Once enough demonstrations exist; fast bulk inference         |
| 4   | `SimscapeAdapter(PhysicsEngineProtocol)` Python ↔ Simscape bridge                                | Medium-high             | High (MATLAB Engine for Python plumbing)   | Reuses existing `system_identification`, RL, retargeter stack |

**Recommended sequencing.** Build Option 1 to ship a fit on a real swing within ~2 weeks. Build Option 2 in parallel on the random-sweep parquet dataset; expect it to mature ~1 week behind Option 1. Option 3 follows Option 2's data pipeline. Option 4 is independent and high-value but high-cost; defer until 1+2 close the loop.

## Folder layout

```
motion_matching/
├── README.md                       (this file)
├── shared/                         (contracts every option consumes)
│   ├── README.md
│   ├── CODING_STANDARDS.md         (TDD, DbC, DRY, LOD for MATLAB and Python)
│   ├── COST_FUNCTION_SPEC.md       (J = w_pos·‖x_sim - x_meas‖² + w_ori·... + λ·W_total)
│   ├── CLUB_IK_SPEC.md             (mocap → club 6-DOF on simulation timegrid)
│   ├── DATASET_SCHEMA.md           (parquet schema for the random-sweep training data)
│   └── VISUALIZATION_SPEC.md       (live overlay plots, error timecourse, parallel coords)
├── option1_direct_optimization/    (MATLAB; fmincon / surrogateopt / MultiStart)
├── option2_nn_surrogate/           (Python+MATLAB; differentiable forward surrogate)
├── option3_inverse_nn/             (Python; sequence encoder → coefficient decoder)
├── option4_python_bridge/          (Python adapter implementing PhysicsEngineProtocol)
├── data/                           (mocap captures + dataset symlinks; .gitignore'd contents)
└── results/                        (optimizer logs, fitted coefficients, .gitignore'd contents)
```

## Hard constraints (assumptions all four options must respect)

1. **Forward simulator.** [GolfSwing3D_Kinetic.slx](../src/model/GolfSwing3D_Kinetic.slx). Treat the model as fixed for now; do not modify the .slx in this work.
2. **Decision variables.** Polynomial coefficients per joint, **7 per joint** named A (t^6) through G (t^0). The full set is loaded from [PolynomialInputValues.mat](../src/model/inputs/PolynomialInputValues.mat) by [getPolynomialParameterInfo.m](../src/functions/dataset_generator/getPolynomialParameterInfo.m). Total dim = `n_joints × 7`.
3. **Bounds.** From [generateRandomCoefficients.m](../src/functions/dataset_generator/generateRandomCoefficients.m): A,B ∈ ±1000; C,D ∈ ±500; E,F ∈ ±100; G ∈ ±25. Treat these as outer bounds; the optimizer is free to converge inside them.
4. **Observation set (Phase 1).** Club only — butt position (3D), clubhead position (3D), and club orientation (3×3 rotation or quaternion) over time. Sampled from [Wiffle_ProV1_club_3D_data.xlsx](../src/apps/golf_gui/Motion%20Capture%20Plotter/) for the first round; one C3D file is available in [Data/Mocap C3D Files/](../Data/Mocap%20C3D%20Files/) but is **untested and may not parse cleanly** — Issue #013 covers reading it.
5. **Simulation duration.** ~0.3 s. Time-align the measured swing to the simulation timegrid (re-sample to the simulation `sample_rate`, default 1000 Hz; trim to the same window).
6. **Under-determined fit.** With club-only observation, the joint torques are not unique. We disambiguate by adding a regularizer: **minimize total mechanical work** (or peak power) of the swing. This biases the solver toward physiologically reasonable solutions.

## Coding standards (enforced by CI on PRs)

See [shared/CODING_STANDARDS.md](shared/CODING_STANDARDS.md). Summary:

- **TDD:** Every public function has a unit test in the same PR. Coverage must not decrease.
- **DbC:** MATLAB uses `arguments` blocks for preconditions and `assert(...)` for postconditions; Python uses `@precondition` / `@postcondition` / `@invariant` decorators from `src.shared.python.core.contracts`.
- **DRY:** No duplicated logic blocks > 5 lines. Shared utilities go under `shared/`.
- **LOD:** No method chains > 2 levels (`a.b.c.d()` is a violation; add a delegating method).
- **File size:** 1200 lines max per .m or .py file (matches the existing CLAUDE.md policy).

## How an agent picks up work

1. Pick an issue from the list below (or `gh issue list --label motion-matching`).
2. Create a branch `feat/motion-matching/<short-name>` off main.
3. Implement against the contracts in [shared/](shared/). Tests-first.
4. Open a PR targeting `main`. The PR description should reference the issue.
5. CI must pass; the issue's acceptance criteria must be checked off.

## GitHub issues

Issues are filed on the repo and also stored as numbered markdown under [docs/issues/backlog/](../../../../../../docs/issues/backlog/) (013–028). See each option folder's `README.md` for the issue list specific to that option.

## Open questions for the user

- The parquet dataset isn't in the repo yet. The proposed schema is in [shared/DATASET_SCHEMA.md](shared/DATASET_SCHEMA.md) — please confirm or correct after you copy the file in.
- The C3D capture in `Data/Mocap C3D Files/` has not been read end-to-end. Issue #013 will validate it; please confirm which file is the canonical one.
- Confirm coordinate convention: the model uses meters and radians; the Excel mocap is in inches per [mocap_data_loader.py](../src/apps/golf_gui/Motion%20Capture%20Plotter/mocap_data_loader.py). The shared loader will convert; please flag if there's a different ground-truth source.
