# Pinocchio Crocoddyl B100 Driver Controls Replayed in MuJoCo

Issue #10336 (MS-21); source candidate merged in #10491 (MS-107). This is a diagnostic
replay of the barrier-reduced `b100` G1 continuation candidate, not an accepted match.
Read `receipt.json` for measured results and exact hashes.

## Recorded Result

MuJoCo completed 307 of 307 requested frames (0–0.85 s) without numerical fault.
The G1 whole-marker RMS was 0.350012 m, club RMS 0.464922 m, and pelvis yaw RMS
0.988470 rad. Same-state FK marker agreement against the saved Crocoddyl candidate
markers was 2.514600e-15 m. While kinematic agreement is near machine epsilon,
open-loop forward integration of the source controls diverged dynamically, leading
to a rejected G1 acceptance status under fail-closed MS-100 gates.

## Configuration and Qualification

The shared anthropometric model, calibrated attachments, ground height and
Hunt–Crossley / regularized Coulomb contact law are reused. MuJoCo stock contact
stays disabled. The rigid KKT grip solver consumes 38 saved joint efforts with six
zero root efforts. DOP853 carries state continuously from the original q0/v0; controls
are held over each source frame interval. There is no feedback or per-frame pose reset.

The 0.005 kg·m² non-root armature follows the MS-31 turnover instruction.
`returned-replay.npz` contains all 307 completed native replay frames in the viewer format.
`playback.gif` visualizes that rejected dynamic replay.
`kinematic-playback.gif` shows all 307 source poses evaluated through MuJoCo FK,
explicitly labelled IK playback only.

## Reproduction

Run from the repository root with MuJoCo, NumPy, SciPy, ezc3d, Matplotlib and imageio installed:

```powershell
$candidateHash = (Get-FileHash evidence/matched/driver_g1_crocoddyl_rk45_b100/candidate.npz -Algorithm SHA256).Hash.ToLower()
python3 -m scripts.replay_pinocchio_in_mujoco `
  --candidate evidence/matched/driver_g1_crocoddyl_rk45_b100/candidate.npz `
  --source-receipt evidence/matched/driver_g1_crocoddyl_rk45_b100/receipt.json `
  --document docs/development/full_body_models/evidence/ground_support/anthro_driver/full_body_spec_hipcal_scaled.json `
  --capture data/C3D_TA_Driver.c3d `
  --attachments docs/development/full_body_models/evidence/ground_support/anthro_driver_shoot_g025/receipt.json `
  --output evidence/matched/driver_g1_crocoddyl_rk45_b100_mujoco_replay `
  --candidate-sha256 $candidateHash --armature 0.005 `
  --max-evaluations 200000 --diagnostic
```
