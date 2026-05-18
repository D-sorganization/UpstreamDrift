# Shared utilities for motion matching

> **Read first**: [PROJECT_SPEC.md](../../../PROJECT_SPEC.md), [MATLAB_GOLF_MODEL_GUIDE.md](../../MATLAB_GOLF_MODEL_GUIDE.md), [GRIP_FIT_PLAYBOOK.md](GRIP_FIT_PLAYBOOK.md).

The four options consume the same contracts. This folder holds the specs (so any agent can implement to them) and will hold the reference implementations once the corresponding issues land.

## Specs (read these first)

- [CODING_STANDARDS.md](CODING_STANDARDS.md) — TDD, DbC, DRY, LOD rules
- [COST_FUNCTION_SPEC.md](COST_FUNCTION_SPEC.md) — the J(θ) every option minimises
- [CLUB_IK_SPEC.md](CLUB_IK_SPEC.md) — measured-swing → canonical `target` struct
- [DATASET_SCHEMA.md](DATASET_SCHEMA.md) — parquet schema for the random-sweep training data
- [VISUALIZATION_SPEC.md](VISUALIZATION_SPEC.md) — required views + styling

## Code that will live here (one PR per file, all tracked by issues)

| File                                    | Issue      | Description                                                         |
| --------------------------------------- | ---------- | ------------------------------------------------------------------- | --- | --- |
| `compute_cost.m` / `cost.py`            | #015, #016 | Reference cost function (MATLAB and Python; numerically equivalent) |
| `compute_total_work.m`                  | #015       | Regularizer; integrates `Σ                                          | τ·ω | `   |
| `load_club_target_excel.m`              | #013       | Wiffle xlsx → `target` struct                                       |
| `load_club_target_c3d.m`                | #013       | C3D → `target` struct (validates the one untested capture)          |
| `synthesize_target_from_coefficients.m` | #014       | TDD oracle: known θ → target                                        |
| `simulate_with_coefficients.m`          | #018       | The single Simscape-call wrapper. Every option uses this — DRY      |
| `load_sweep_dataset.py`                 | #019       | Parquet loader for the random-sweep dataset                         |
| `+validators/`                          | #015       | `mustHaveFields`, `mustBeFiniteVector`, etc.                        |
| `plot_trajectory_overlay.m`             | #020       | View 1                                                              |
| `animate_trajectory_overlay.m`          | #020       | View 1 animated                                                     |
| `plot_error_timecourse.m`               | #021       | View 2                                                              |
| `plot_fit_quality_card.m`               | #022       | View 3                                                              |
| `leaderboard.m`                         | #023       | Cross-option comparison table                                       |

## How the contracts compose

```
                ┌─────────────────────────────┐
                │  load_club_target_*         │  → target struct (CLUB_IK_SPEC)
                └──────────────┬──────────────┘
                               │
                               ▼
            ┌──────────────────────────────────────┐
            │  optimizer (option-specific)         │
            │     decides θ ∈ ℝ^(n_joints×7)       │
            └──┬─────────────────────────┬─────────┘
               │                         │
               ▼                         ▼
   ┌────────────────────────┐   ┌────────────────────────┐
   │ simulate_with_         │   │ compute_cost(θ,        │
   │ coefficients(θ)        │──▶│   target, sim_fn, opts)│ → scalar J
   └────────────────────────┘   └────────────────────────┘
               │
               ▼
   ┌────────────────────────────────────────┐
   │  result struct: coefficients, RMSE,    │
   │  total work, solver, provenance        │  (CODING_STANDARDS § Provenance)
   └─────────────────┬──────────────────────┘
                     │
                     ▼
   ┌────────────────────────────────────────┐
   │  visualization (VIEW 1, 2, 3) +        │
   │  leaderboard.m                         │
   └────────────────────────────────────────┘
```

This is the only place `simulate_with_coefficients` is allowed to be implemented. If an option needs a slightly different forward call, it parameterises this one rather than forking it.

## Stage-1 starting-pose solver

[`solve_starting_pose.m`](solve_starting_pose.m) implements the warm-start
recipe described in [GRIP_FIT_PLAYBOOK.md §"Stage 1 — initial pose"](GRIP_FIT_PLAYBOOK.md).
It produces the `input_overrides` struct that Stage-2 (`fit_swing_fmincon`)
consumes via `opts.sim.input_overrides`, layering small perturbations on
top of a base per-pose MAT (e.g. `3DModelInputs_Impact.mat`) so the model's
mid-hands grip lands at the measured grip pose at the swing's address frame.
