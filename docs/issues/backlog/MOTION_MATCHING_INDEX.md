# Motion Matching Issue Index (#013–#040)

Table of contents and dependency DAG for the 28 motion-matching issues that
scaffold the four parallel pathways. See
[`src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/README.md`](../../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/motion_matching/README.md)
for context.

## Issues by group

### Shared infrastructure (013–023)

| #   | File                                                                                     | Title                                                            | Effort | Depends on       |
| --- | ---------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ------ | ---------------- |
| 013 | [013_load_club_target_excel_and_c3d.md](013_load_club_target_excel_and_c3d.md)           | Implement load_club_target_excel.m and load_club_target_c3d.m    | M      | none             |
| 014 | [014_synthesize_target_from_coefficients.md](014_synthesize_target_from_coefficients.md) | Implement synthesize_target_from_coefficients.m TDD oracle       | S      | #018             |
| 015 | [015_compute_cost_matlab.md](015_compute_cost_matlab.md)                                 | Implement compute_cost.m, compute_total_work.m, +validators      | M      | none             |
| 016 | [016_compute_cost_python.md](016_compute_cost_python.md)                                 | Implement Python cost.py with cross-check vs MATLAB              | M      | #015             |
| 017 | [017_load_club_target_python.md](017_load_club_target_python.md)                         | Implement Python ClubTarget + Excel/C3D loaders                  | M      | none             |
| 018 | [018_simulate_with_coefficients_matlab.md](018_simulate_with_coefficients_matlab.md)     | Implement simulate_with_coefficients.m — single Simscape wrapper | L      | none             |
| 019 | [019_load_sweep_dataset.md](019_load_sweep_dataset.md)                                   | Implement load_sweep_dataset parquet loader                      | M      | none             |
| 020 | [020_plot_trajectory_overlay.md](020_plot_trajectory_overlay.md)                         | Implement plot*trajectory_overlay.m + animate*\* (View 1)        | M      | #013, #018       |
| 021 | [021_plot_error_timecourse.md](021_plot_error_timecourse.md)                             | Implement plot_error_timecourse.m (View 2)                       | S      | #013, #018       |
| 022 | [022_plot_fit_quality_card.md](022_plot_fit_quality_card.md)                             | Implement plot_fit_quality_card.m (View 3)                       | S      | #020, #021       |
| 023 | [023_leaderboard_cross_option_comparison.md](023_leaderboard_cross_option_comparison.md) | Implement leaderboard.m for cross-option comparison              | S      | #015, #018, #022 |

### Option 1 — Direct optimization (024–027)

| #   | File                                                                             | Title                                                | Effort | Depends on             |
| --- | -------------------------------------------------------------------------------- | ---------------------------------------------------- | ------ | ---------------------- |
| 024 | [024_fit_swing_fmincon.md](024_fit_swing_fmincon.md)                             | Implement fit_swing_fmincon.m (single-start sqp)     | M      | #013, #015, #018       |
| 025 | [025_fit_swing_multistart.md](025_fit_swing_multistart.md)                       | Implement fit_swing_multistart.m (parsim/parfor)     | M      | #018, #024             |
| 026 | [026_fit_swing_surrogateopt.md](026_fit_swing_surrogateopt.md)                   | Implement fit_swing_surrogateopt.m (global + polish) | M      | #018, #024             |
| 027 | [027_optimization_progress_dashboard.md](027_optimization_progress_dashboard.md) | Implement OptimizationProgressDashboard handle class | M      | #020, #024, #025, #026 |

### Option 2 — NN forward surrogate (028–031)

| #   | File                                                                                           | Title                                           | Effort | Depends on       |
| --- | ---------------------------------------------------------------------------------------------- | ----------------------------------------------- | ------ | ---------------- |
| 028 | [028_swing_surrogate_pytorch.md](028_swing_surrogate_pytorch.md)                               | Implement SwingSurrogate nn.Module + train loop | L      | #019             |
| 029 | [029_invert_via_surrogate.md](029_invert_via_surrogate.md)                                     | Implement Adam-on-coefficients inversion        | M      | #016, #017, #028 |
| 030 | [030_round_trip_validation_against_simscape.md](030_round_trip_validation_against_simscape.md) | Round-trip validation vs Simscape               | M      | #018, #028, #029 |
| 031 | [031_hybrid_surrogate_then_fmincon.md](031_hybrid_surrogate_then_fmincon.md)                   | Hybrid surrogate-then-fmincon polish            | M      | #024, #029, #030 |

### Option 3 — Inverse CVAE (032–035)

| #   | File                                                                                 | Title                                           | Effort | Depends on             |
| --- | ------------------------------------------------------------------------------------ | ----------------------------------------------- | ------ | ---------------------- |
| 032 | [032_swing_inverse_cvae_pytorch.md](032_swing_inverse_cvae_pytorch.md)               | Implement SwingInverseCVAE encoder/decoder/CVAE | L      | #019                   |
| 033 | [033_train_inverse_cvae.md](033_train_inverse_cvae.md)                               | Training pipeline with KL annealing             | L      | #019, #032             |
| 034 | [034_predict_with_rejection_sampling.md](034_predict_with_rejection_sampling.md)     | Sample-and-validate inference                   | M      | #016, #018, #028, #032 |
| 035 | [035_inverse_diagnostics_mode_coverage.md](035_inverse_diagnostics_mode_coverage.md) | UMAP latent + diversity diagnostics             | M      | #019, #032, #034       |

### Option 4 — Python ↔ Simscape bridge (036–040)

| #   | File                                                                                   | Title                                                        | Effort | Depends on             |
| --- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------ | ------ | ---------------------- |
| 036 | [036_simscape_adapter_protocol_skeleton.md](036_simscape_adapter_protocol_skeleton.md) | SimscapeAdapter protocol skeleton                            | M      | none                   |
| 037 | [037_simscape_adapter_simulate.md](037_simscape_adapter_simulate.md)                   | SimscapeAdapter.simulate_with_coefficients via MATLAB Engine | L      | #018, #036             |
| 038 | [038_register_matlab_3d_loader.md](038_register_matlab_3d_loader.md)                   | Register load_matlab_3d_engine in loaders.py                 | S      | #036, #037             |
| 039 | [039_simscape_adapter_pool.md](039_simscape_adapter_pool.md)                           | SimscapeAdapterPool for parallel inference                   | L      | #036, #037             |
| 040 | [040_integration_system_identification.md](040_integration_system_identification.md)   | Integration test — system_identification.py                  | M      | #036, #037, #038, #039 |

## Dependency DAG

```
                          (none)
                            │
              ┌─────────────┼──────────────┬────────────┬────────┐
              ▼             ▼              ▼            ▼        ▼
            #013          #015           #017         #018     #019
              │            │               │            │        │
              │            ▼               │      ┌─────┼────┬───┴──────┬─────────┐
              │          #016              │      ▼     ▼    ▼          ▼         ▼
              │            │               │    #014  #024  #036       #028     #032
              │            │               │            │     │         │         │
              │            └────────┐      │            │     ├──┬──┐   │      ┌──┴──┐
              │                     │      │            │     │  │  │   │      ▼     │
              │                     │      │            │     ▼  ▼  ▼   │    #033    │
              │                     │      │            │   #038│#039 │ │      │     │
              │                     ▼      ▼            ▼      ▼  │  ▼ ▼      │     ▼
              ▼                   #029 ◀────                #037 #040 #030  #034 ◀──┘
            #020─────────┐         │                                  │      │
              │           ▼        │                                  ▼      │
              ▼         #022       │                               #031     ▼
            #021──┐       │        │                                       #035
              │   ▼       ▼        ▼
              │ #023 (used by all options' results)
              ▼
            #025, #026 (depend on #024 and #018)
              │
              ▼
            #027 (depends on #020, #024, #025, #026)
```

## Recommended starting order (unblock fan-out)

These are the issues to tackle first; they unblock the largest number of
downstream issues and have no upstream dependencies:

1. **#018** — `simulate_with_coefficients.m`. Six issues block on this
   (#014, #024, #025, #026, #030, #037). Start here.
2. **#015** — `compute_cost.m` + `+validators/`. Three downstream issues
   (#016, #024, #023) plus every other option's J evaluation.
3. **#013** — `load_club_target_excel.m` + `load_club_target_c3d.m`. Five
   downstream (#020, #021, #024, #017 partly, #014).
4. **#019** — `load_sweep_dataset`. Required for Options 2 and 3 training
   pipelines (#028, #032, #033).
5. **#017** — Python `ClubTarget` + loaders. Required for Options 2, 3, 4.
6. **#036** — `SimscapeAdapter` skeleton. Cleanly parallelizable from day 0.

After those six, the rest can be picked up in dependency order. Within each
option group, the listed dependencies inside each issue file determine the
intra-option sequence.

## Issue-to-spec cross-reference

| Spec doc                              | Issues that consume it                      |
| ------------------------------------- | ------------------------------------------- |
| `shared/COST_FUNCTION_SPEC.md`        | #015, #016, #024, #025, #026, #029, #034    |
| `shared/CLUB_IK_SPEC.md`              | #013, #014, #017, #020, #021, #022          |
| `shared/DATASET_SCHEMA.md`            | #019, #028, #032, #033                      |
| `shared/VISUALIZATION_SPEC.md`        | #020, #021, #022, #023, #027, #035          |
| `shared/CODING_STANDARDS.md`          | every issue (TDD, DbC, DRY, LOD, file size) |
| `option4_python_bridge/INTERFACES.md` | #036, #037, #038, #039                      |
| `option4_python_bridge/RUNBOOK.md`    | #037, #039, #040                            |

## Labels by group

- All issues: `motion-matching`
- Issues 013–023: `shared`
- Issues 024–027: `option1`
- Issues 028–031: `option2`
- Issues 032–035: `option3`
- Issues 036–040: `option4`
- MATLAB-only: `matlab` (013–015, 018, 020–027)
- Python-only: `python` (016, 017, 028–037, 039, 040)
- Both: `matlab` + `python` (019, 031, 038)
- Visualization: `viz` (020–023, 027, 035)
- Infrastructure (foundational): `infra` (013–019, 038, 040)

Every issue carries `tdd` and (where applicable) `dbc`.
