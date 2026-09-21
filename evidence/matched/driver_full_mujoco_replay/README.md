# Pinocchio Driver Controls Replayed in MuJoCo

Issue #10336; source merged in #10411 / #10425. This is a diagnostic replay,
not an accepted match. Read `receipt.json` for measured results and exact hashes.

## Recorded Result

MuJoCo completed 571 of 654 requested frames (0–1.583333 s) before exhausting
200,000 right-hand-side evaluations. The G1 window itself completed, but its
whole-marker RMS was 0.940137 m, club RMS 0.741646 m and pelvis yaw RMS
1.760140 rad. Refining the integrator changed G1 markers by up to 3.087162 m
(limit 0.00001 m): numerical convergence failed. Same-state FK marker error
against the saved source markers was 2.314597e-15 m. These are different tests;
excellent FK agreement does not rescue the rejected dynamics.

## Configuration and Qualification

The shared model, calibrated attachments, ground height and Hunt–Crossley /
regularized Coulomb contact law are reused. MuJoCo stock contact stays disabled.
The rigid KKT grip solver consumes 38 saved efforts with six zero root efforts.
DOP853 carries state continuously from the original q0/v0; controls are held
over each source frame interval. There is no feedback or per-frame pose reset.

The 0.005 kg·m² non-root armature follows the MS-31 turnover instruction.
However, `scripts/match_pinocchio_c3d.py` never calls `apply_armature`, and the
merged source receipt omits armature, contact configuration, interpolation and
candidate hash. Thus the declared diagnostic configuration is explicit but
**source-identical configuration is unverified**. Missing root-assistance
history and independent forward replay also block G1. QP ground/grip wrenches
are not injected as forces: the actual shared contact and rigid grip law
determine forces at the evolving replay state.

`returned-replay.npz` contains only completed native replay frames in the
existing viewer format. `playback.gif` visualizes that rejected replay.
`kinematic-playback.gif` shows all 654 source poses evaluated through MuJoCo FK,
explicitly labelled IK playback only. Same-state marker agreement is not
dynamic parity. G1 metrics on an incomplete replay describe its completed
prefix and cannot pass the complete-horizon gate. No thresholds were relaxed.

## Reproduction

Run from the repository root with MuJoCo, NumPy, SciPy, ezc3d, Matplotlib and
imageio installed; initialize the pinned `vendor/ud-tools` submodule first.

```powershell
$candidateHash = (Get-FileHash evidence/matched/driver_full_pinocchio/candidate.npz -Algorithm SHA256).Hash.ToLower()
python3 -m scripts.replay_pinocchio_in_mujoco `
  --candidate evidence/matched/driver_full_pinocchio/candidate.npz `
  --source-receipt evidence/matched/driver_full_pinocchio/receipt.json `
  --document docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json `
  --capture data/C3D_TA_Driver.c3d `
  --attachments docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json `
  --output evidence/matched/driver_full_mujoco_replay `
  --candidate-sha256 $candidateHash --armature 0.005 `
  --max-evaluations 200000 --diagnostic
```

The expected CLI exit is nonzero (2) for rejected acceptance. Without
`--diagnostic`, missing source provenance raises before replay. A changed
candidate requires an intentional new expected hash; a conflicting hash in
the source receipt always rejects, even in diagnostic mode.

## Validation

```powershell
python3 -m pytest tests/unit/motion_matching/test_mujoco_candidate_replay.py tests/unit/motion_matching/test_acceptance.py tests/unit/motion_matching/pipeline/test_receipt_schema.py tests/unit/motion_matching/test_full_body_mujoco.py tests/unit/motion_matching/test_contact_law.py -q --no-cov -o addopts=''
python3 scripts/ci/check_architecture_budget.py
python3 -m agent_context --root . check
```

The tests cover hashes, coordinate/actuation identity, exact G1 boundaries,
missing/nonfinite metrics, empty marker populations, missing source dynamics,
integration without resets, explicit budget failure, armature mass-matrix
deltas, unchanged shared contact parameters, disabled stock contacts, and
receipt/artifact integrity. Issue #10336 remains open for qualified G1/G2/G3
driver and iron evidence.
