# Pinocchio/Crocoddyl Continuation Handoff

Reviewed 2026-09-18 at 01:45 UTC. Governing continuation: #10381 (MS-107),
then #10385 (G2/G3); parent #10363. Historical implementation issue #10338
is closed. This is a timestamped operational snapshot, not acceptance.

## Start Here

A native G1 job was still running on ControlTower. Do not relaunch, overwrite
its output, change its runtime files, or kill it. Check its process and outputs
at the next checkpoint. Sparse quiet-mode logs alone do not prove a hang.
The first safe task for a cheaper agent is retrieval and qualification reporting,
not another solver design or parameter sweep.

Read the [program restart prompt](AGENT_CONTINUATION_PROMPT.md), then this file.
The existing main-branch `full_body_fit.py` is a different implementation from
Claude's native lane. Do not run a command against main merely because the
module name exists.

## Exact Source and Runtime

- Windows source: `C:/Users/diete/Repositories/_wt_claude_10338`.
- Published source branch: `feat/10338-crocoddyl-native-fit`.
- Source commit: `43da8dfa20415b07f1f7a94e3aeacd4ef95b955a`.
- Native source is not integrated into reviewed main `94d593cf1`.
- ControlTower SSH host: `controltower`; Windows SSH shell; WSL distro:
  `ControlTower-Runner`.
- Running checkout: `/home/dieterolson/ms31-crocoddyl`; reported Git HEAD
  `f356382a233a68cafe5e4975bea88f2eeaba3f76` plus modified/untracked files.
  **HEAD alone does not describe the running source.** All compared deployed
  Pinocchio Python files and the native integration test match the Windows
  source commit after CRLF-to-LF normalization. Raw and normalized hashes are
  in [the manifest](evidence/continuation_20260918/matching-source-manifest.json).
- Runtime: `/home/dieterolson/bin/micromamba`, root
  `/home/dieterolson/mm-root`, environment `upstream-motion-runtime`.
  Prior runtime pin: Pinocchio 4.1.0, Crocoddyl 3.2.1, Pink 4.4.0. Re-probe
  versions before a new accepted run; this review did not reinstall anything.
- Source launcher: `scripts/matched_swing/run_crocoddyl_fit.sh` on the pinned
  branch. It uses `nohup`, not systemd; do not pretend this existing job has a
  systemd unit. Future new runners should use the approved fleet lifecycle.

## Running Job and Recovery Files

At 01:45 UTC, native Python PID 3104452 was active (PID is historical; verify
command and start time before acting). Actual log:
`/home/dieterolson/fit_driver_g1.log`, NOT the launcher's default `fits/*.log`.
Output: `/home/dieterolson/fits/driver_g1`.

Observed command parameters: horizon 0.85 s; final iterations 120; warm start
`/home/dieterolson/fits/w030r/candidate.npz`; continuation `0.30,0.45,0.60`;
30 iterations per stage; RK45 rtol `1e-4`; quiet mode. Document is
`anthro_driver/full_body_spec_hipcal_scaled.json`; attachment/ground receipt
is `anthro_driver_shoot_g025/receipt.json` under ground-support evidence.

Saved checkpoints: `stage_0.30s.npz`, `stage_0.45s.npz`, `stage_0.60s.npz`.
No final `solution.npz` or G1 `receipt.json` existed at the snapshot.
After solving, code writes `solution.npz` and `solver_stages.json` before
post-processing, then `candidate.npz` and `receipt.json`. A prior attempt lost
its solution in diagnostics; the current checkpoint commit fixes that order.

The [recovery archive](evidence/continuation_20260918/matching-continuation-snapshot-20260918.tar.gz)
contains selected deployed source, the completed w030r receipt/candidate and
available G1 stage checkpoints. It is a recovery subset, not a complete runtime
or proof of current job completion. Verify its manifest before extracting to
an empty owned directory; never extract over a live checkout.

## What the Completed Result Actually Shows

Source: `/home/dieterolson/fits/w030r/receipt.json`, copied into the recovery
archive and represented in [the raw snapshot](evidence/continuation_20260918/matching-handoff-snapshot.json).
Read `metrics.replay.shared`, not the similarly named `replay_five` metrics,
which have different terminal/club definitions.

| Quantity                | Measured Value     | Interpretation                       |
| ----------------------- | ------------------ | ------------------------------------ |
| Horizon                 | 0.30 s             | Not 0.85 s G1                        |
| Whole/Early RMS         | 19.7258 mm         | Early exceeds current 12 mm G1 limit |
| Terminal RMS            | 14.2255 mm         | Prefix terminal only                 |
| Club RMS                | 9.7531 mm          | Shared metric definition             |
| Pelvis Yaw RMS          | 0.0235665 rad      | About 1.35 degrees                   |
| Maximum Penetration     | 17.1946 mm         | Exceeds current 10 mm limit          |
| Peak Normal Force       | 1.415 body weights | Not a full physical audit by itself  |
| Minimum Weight Fraction | 0.20959            | Other gates/evidence still required  |

FDDP rollout and replay metric values agree in this receipt; that is not an
independent cross-engine replay or derivative/convergence qualification.
The implicit-Euler diagnostic gives 602 mm whole error. Earlier w030c gave
15.9 mm solver error but 192 mm replay error and must remain rejected.
No accepted G1 exists in the evidence inspected by this review.

## Bounded Next Actions

1. Inspect the existing process and completed artifacts once. Do not duplicate
   a still-running fit. Copy newly completed outputs to a new timestamped
   evidence directory; retain the log, source hashes and original failed runs.
2. Validate NPZ keys/shapes/time coverage and finite states/controls; compare
   `metrics.fddp_rollout.shared` to `metrics.replay.shared`, then inspect every
   physical gate. Never fabricate missing support/rotation/coverage evidence.
3. Use current `acceptance.py` as an implementation input and #10374 as the
   outstanding qualification contract. The old note that acceptance.py does
   not exist is obsolete. G1 is a dynamic horizon, not the tracker's 30 mm IK
   milestone. Gate documentation drift is flagged in GATES.md.
4. If processing failed after solve, preserve `solution.npz` and stage files;
   recover post-processing first. If the process exited before a final solution,
   verify the latest complete checkpoint and resume only in a NEW output path.
   Do not make another 2-hour run merely to regenerate a GIF.
5. Before accepted replay, verify derivatives of the actual RK45 rollout versus
   the implicit-Euler approximation, integrator refinement and the 0.005 kg m^2
   non-root armature. Apply the same armature/contact model in MuJoCo; no hidden
   model changes. These are expert review checkpoints under #10381/#10352.
6. Submit native source integration separately from evidence/docs. Compare the
   two existing `full_body_fit.py` APIs and tests; do not blindly cherry-pick
   over the newer main implementation. Preserve both histories and choose the
   program's one supported entry point under MS-12.

Conditional resume command, only from the pinned native lane after verifying
no active job owns the output and checkpoint/model hashes match:

```bash
cd /home/dieterolson/ms31-crocoddyl
bash scripts/matched_swing/run_crocoddyl_fit.sh driver_g1_resume_review 0.85 120 \
  --quiet --warm-start-candidate /home/dieterolson/fits/driver_g1/stage_0.60s.npz \
  --continuation 0.60 --stage-iterations 30 --rk45-rtol 1e-4
```

This is a proposed bounded continuation, not a command executed by the review.
Confirm the output name is unused and inspect `--help` on the pinned code first.
Do not relax gates or change contact/armature to make a receipt green.

## Validation and Escalation

On the Windows source commit, 14 tests passed:

```text
python -m pytest tests/unit/motion_matching/test_crocoddyl_problem.py tests/unit/motion_matching/test_crocoddyl_action.py -q -o addopts=''
```

These tests do not qualify native physics. A cheaper agent can retrieve artifacts,
run established tests, package results and perform one justified bounded resume.
Escalate derivative inconsistency, contact/armature changes, model/API integration
conflicts or a failed physical gate requiring a modeling decision. Do not repeatedly
sweep parameters or claim that a converged partial fit finishes #10381.
