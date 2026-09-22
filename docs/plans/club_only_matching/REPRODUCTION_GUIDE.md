# Club-Only Reproduction Guide

Governing issue: [CO-10 #10614](https://github.com/D-sorganization/UpstreamDrift/issues/10614).
Parent epic: [#10602](https://github.com/D-sorganization/UpstreamDrift/issues/10602).
Schema: `club-only-reproduction/1.0.0`.

## Purpose

Final operator/reproduction turnover for classical club-only matching.
Use this guide to replay saved jobs, verify workbook provenance, select
candidates under declared assumptions, and export/import portable packages
in a clean environment. Software-contract GREEN is not native Fit/G1.

## Trial and Model Roster

- Trials (4): TW_wiffle, TW_ProV1, GW_wiffle, GW_ProV11
- Models (20): constrained_upper_body_golfer, driven_double_pendulum, driven_triple_pendulum, full_body_drake, full_body_mujoco, full_body_myosuite, full_body_opensim, full_body_pinocchio, full_body_simscape, myosuite_body, opensim_golfer, reconstruction_double_pendulum, reconstruction_golfer, reconstruction_triple_pendulum, reference_drake_urdf, reference_human_subject, reference_mujoco_humanoid, reference_pinocchio_urdf, reference_pinocchio_urdf_ik, reference_simple_humanoid
- Matrix cells: 80 (scored=12, unresolved=68)

## Raw-Source Provenance

- `data/Club_Data.xlsx` — SHA-256 `5d9183e1d01ea7c6f9c162375dd6855c076ee26a96b76c90e59e9cf2679dde25` (CO-00 workbook identity / PR #10628 review)
- `src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/Wiffle_ProV1_club_3D_data.xlsx` — SHA-256 `88d3eb31541d886031f6c5ad7c82493b0e3ee4e3f73a8652372277cc14d02234` (CO-00 workbook identity / PR #10628 review)

## Assumptions

- Workbook Definitions declare inches; reviewed SI authority is centimetres (to_meters_scale=0.01) per CO-00 unit contract.
- Body motion is a plausible inferred candidate under GolfPlausibilityPriors and geometry choices — it was not measured in the club-only Excel source.
- Priors, handedness, and grip/geometry edits are operator assumptions, not measurements; changing them invalidates prior candidate hashes.
- Only two orientation axes are independently populated; the third is derived with derivation metadata and must not be scored as measured.
- Filtering Experiments is an alias of TW_ProV1 and is listed once among the four unique trials.
- Software-contract fixtures and UI preview statuses are not native Fit/G1 evidence and do not satisfy full-body G3.

## Candidate Selection

- Strategy: feasibility-first pruning then bounded Pareto diversity (CO-07)
- Preview vs verified: FAST_PREVIEW is display-only; verified-fit requires a continuous independent replay without measured-state resets (CO-06/CO-09)
- Ranking: lexicographic/Pareto measured residual then prior then runtime; visual attractiveness never overrides physical failure (CO-02/CO-08)
- Neural slot: Optional empty neural proposal slot may exist; neural proposals are not native evidence and do not inherit club-only matrix success

## Exact Saved-Job Commands

### `workbook_identity`

Verify frozen workbook SHA-256 and four-trial lineage.

```bash
python -m pytest tests/unit/motion_matching/test_club_workbook_identity.py -q -n 0 --no-cov --timeout=60
```

### `matrix_qualification`

Reconcile four-trial × #10585 roster matrix (software contract).

```bash
python -m pytest tests/unit/motion_matching/test_club_matrix_qualification.py -q -n 0 --no-cov --timeout=60
```

### `ui_integration`

Exercise FitSwingProvider/pipeline/ledger Club-Only UI contracts.

```bash
python -m pytest tests/unit/motion_matching/test_club_ui_integration.py -q -n 0 --no-cov --timeout=90
```

### `fast_preview_match`

Run a software-contract FAST_PREVIEW club-only match for TW_wiffle.

```bash
python -c from src.shared.python.motion_matching.club_only.ui_integration import create_club_only_session, run_club_only_ui_match; from src.shared.python.motion_matching.club_only.fast_matching import MatchPreset; s=create_club_only_session(trial_id='TW_wiffle', model_id='driven_double_pendulum', preset=MatchPreset.FAST_PREVIEW); r=run_club_only_ui_match(s); assert r.display_status.value!='verified' or r.match.native_g1_pass is False; print(r.display_status, r.match.qualification_blockers)
```

### `portable_package_roundtrip`

Export/import an MS-105 portable package without a second scheduler.

```bash
python -c from pathlib import Path; from src.shared.python.motion_matching.jobs import (MatchingJobSpec, JobStage, HashBundle, export_portable_package, import_portable_package, JOBS_SCHEMA); print(JOBS_SCHEMA); print('export_portable_package', export_portable_package.__name__); print('import_portable_package', import_portable_package.__name__); print('MatchingJobSpec', MatchingJobSpec.__name__, JobStage.FIT_REPLAY)
```

### `reproduction_freshness`

Docs/context/parity freshness for the CO-10 reproduction guide.

```bash
python -m pytest tests/unit/motion_matching/test_club_reproduction_turnover.py -q -n 0 --no-cov --timeout=60
```

## Clean-Environment Portable Replay

- Requires clean venv: `True`
- Portable package schema: `motion-matching-jobs/1.0.0`
- MATLAB release for native work: `R2025b`
- Rejects measured-state resets: `True`

Export:

```bash
python -c "from pathlib import Path; from src.shared.python.motion_matching.jobs import export_portable_package; export_portable_package(Path('runs/<run_id>'), Path('packages/<run_id>'))"
```

Import:

```bash
python -c "from pathlib import Path; from src.shared.python.motion_matching.jobs import import_portable_package; pkg=import_portable_package(Path('packages/<run_id>')); print(pkg.run_id, pkg.status)"
```

## Qualification Blockers and Epic Closure

- `epic_closure_allowed`: `False`
- `native_g1_pass`: `False`
- `claims_native_qualification`: `False`

- native_g1_qualification_requires_desk_native_receipt
- software_contract_reproduction_guide_is_not_native_evidence
- mandatory_native_fits_remain_open_for_unqualified_matrix_cells
- native_g1_qualification_requires_desk_native_receipt
- software_contract_matrix_is_not_native_evidence

## Program Separation

- Club-only profiles do not satisfy full-body G3 acceptance (native program #10363).
- Neural speed program (#10603) does not inherit success from club-only classical matching (#10602).
- Do not close epic #10602 while mandatory native fits remain missing for unqualified / missing_runtime / unsupported matrix cells.

## Evidence Links

- [evidence/club_workbook_identity.json](evidence/club_workbook_identity.json)
- [evidence/club_observation_contracts.json](evidence/club_observation_contracts.json)
- [evidence/club_plausibility_acceptance.json](evidence/club_plausibility_acceptance.json)
- [evidence/club_starting_guesses.json](evidence/club_starting_guesses.json)
- [evidence/club_pendulum_match.json](evidence/club_pendulum_match.json)
- [evidence/club_body_candidates.json](evidence/club_body_candidates.json)
- [evidence/club_control_replay.json](evidence/club_control_replay.json)
- [evidence/club_fast_matching.json](evidence/club_fast_matching.json)
- [evidence/club_matrix_qualification.json](evidence/club_matrix_qualification.json)
- [evidence/club_ui_integration.json](evidence/club_ui_integration.json)
- [evidence/club_reproduction_turnover.json](evidence/club_reproduction_turnover.json)

## Unresolved Matrix Cells (Executable Next Steps)

<!-- prettier-ignore-start -->
| Model | Trial | Status | Owner | Next Step |
| --- | --- | --- | --- | --- |
| `full_body_drake` | `TW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_drake×TW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_drake` | `TW_ProV1` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_drake×TW_ProV1 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_drake` | `GW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_drake×GW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_drake` | `GW_ProV11` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_drake×GW_ProV11 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_mujoco` | `TW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_mujoco×TW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_mujoco` | `TW_ProV1` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_mujoco×TW_ProV1 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_mujoco` | `GW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_mujoco×GW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_mujoco` | `GW_ProV11` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_mujoco×GW_ProV11 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_myosuite` | `TW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_myosuite, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_myosuite` | `TW_ProV1` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_myosuite, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_myosuite` | `GW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_myosuite, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_myosuite` | `GW_ProV11` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_myosuite, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_opensim` | `TW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_opensim, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_opensim` | `TW_ProV1` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_opensim, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_opensim` | `GW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_opensim, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_opensim` | `GW_ProV11` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_opensim, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_pinocchio` | `TW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_pinocchio×TW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_pinocchio` | `TW_ProV1` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_pinocchio×TW_ProV1 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_pinocchio` | `GW_wiffle` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_pinocchio×GW_wiffle and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_pinocchio` | `GW_ProV11` | `unqualified` | desk-native / runtime owner | On DeskComputer with MATLAB R2025b, run native Fit/G1 for full_body_pinocchio×GW_ProV11 and attach a receipt under docs/plans/club_only_matching/evidence/; do not invent native_g1_pass |
| `full_body_simscape` | `TW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_simscape, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_simscape` | `TW_ProV1` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_simscape, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_simscape` | `GW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_simscape, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `full_body_simscape` | `GW_ProV11` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for full_body_simscape, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `myosuite_body` | `TW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for myosuite_body, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `myosuite_body` | `TW_ProV1` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for myosuite_body, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `myosuite_body` | `GW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for myosuite_body, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `myosuite_body` | `GW_ProV11` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for myosuite_body, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `opensim_golfer` | `TW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for opensim_golfer, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `opensim_golfer` | `TW_ProV1` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for opensim_golfer, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `opensim_golfer` | `GW_wiffle` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for opensim_golfer, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `opensim_golfer` | `GW_ProV11` | `missing_runtime` | desk-native / runtime owner | On DeskComputer, install/enable the missing runtime for opensim_golfer, then python -c "from src.shared.python.motion_matching.club_only.matrix_qualification import build_matrix_qualification_report; build_matrix_qualification_report()" and commit a fresh evidence receipt |
| `reconstruction_double_pendulum` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_double_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_double_pendulum` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_double_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_double_pendulum` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_double_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_double_pendulum` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_double_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_golfer` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_golfer out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_golfer` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_golfer out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_golfer` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_golfer out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_golfer` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_golfer out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_triple_pendulum` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_triple_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_triple_pendulum` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_triple_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_triple_pendulum` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_triple_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reconstruction_triple_pendulum` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reconstruction_triple_pendulum out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_drake_urdf` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_drake_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_drake_urdf` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_drake_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_drake_urdf` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_drake_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_drake_urdf` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_drake_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_human_subject` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_human_subject out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_human_subject` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_human_subject out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_human_subject` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_human_subject out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_human_subject` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_human_subject out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_mujoco_humanoid` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_mujoco_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_mujoco_humanoid` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_mujoco_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_mujoco_humanoid` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_mujoco_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_mujoco_humanoid` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_mujoco_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf_ik` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf_ik out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf_ik` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf_ik out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf_ik` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf_ik out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_pinocchio_urdf_ik` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_pinocchio_urdf_ik out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_simple_humanoid` | `TW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_simple_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_simple_humanoid` | `TW_ProV1` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_simple_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_simple_humanoid` | `GW_wiffle` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_simple_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
| `reference_simple_humanoid` | `GW_ProV11` | `unsupported` | model-baseline (#10585) | On DeskComputer, keep reference_simple_humanoid out of club-only promotion; reconcile via #10585 ownership / topology before adding a matrix cell campaign |
<!-- prettier-ignore-end -->

## Limitations

- Software-contract reproduction guide only; desk-native receipts required before scientific promotion.
- Epic #10602 remains open while mandatory native fits are missing.
- UI/docs GREEN does not close unqualified matrix cells.
