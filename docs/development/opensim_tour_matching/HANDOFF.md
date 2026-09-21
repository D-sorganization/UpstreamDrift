# OpenSim Motion-Matching Continuation Handoff

Reviewed 2026-09-18 at 01:45 UTC. Current issue #10341 (MS-42), epic #10363.
OS-0..6 are historical delivered infrastructure; they are not an accepted
full-swing match. The OS-6 shared metrics were unverified. The following
continuation replaces the old restart instructions for completed OS-2b/3b work.

## Exact Source and Runtime

- Local worktree: `C:/Users/diete/Repositories/_wt_claude_10341`.
- Local branch: `feat/10341-moco-g1`, commit `6914383f5`.
- Original source branch was absent on origin and had no PR found by the exact
  source-branch query. This review published the unchanged commit as
  `handoff/10341-opensim-20260918`; fetch that recovery ref on a new host.
  Publication preserves work, but is not integration into main or acceptance.
- Local uncommitted `docs/development/DEVELOPMENT_LOG.md` and untracked
  `evidence/os7_moco_g1/rungs/0850ms_diag_1iter/receipt.json` belong to the
  prior agent; do not reset/stash them or add them wholesale to another PR.
- SSH `controltower`, WSL `ControlTower-Runner`; runtime Python
  `/home/dieterolson/opensim-10003/bin/python` (prior probe OpenSim 4.6/Moco).
- Current deployed tree: `/home/dieterolson/opensim-10341-runtime`, not the
  older `opensim-10003-runtime`. It is a copied tree with no reported Git HEAD.
- Deployed moco_tracking.py, moco_g1.py and os7_moco_g1_driver.py match the local
  commit after line-ending normalization. See the shared
  [source manifest](../matched_swing_program/evidence/continuation_20260918/matching-source-manifest.json).
- Model: `/mnt/c/Users/diete/opensim-10003/os3b-scale-ik/golf_humanoid_scaled_tour_markers.osim`.
- IK: `/mnt/c/Users/diete/opensim-10003/ik_states_full.sto`.
- TRC: `docs/development/opensim_tour_matching/evidence/tour_average_tracked.trc`
  beneath the deployed tree. Pinned model/IK copies also exist in the local
  branch's `evidence/os7_moco_g1/inputs/`; verify hashes before substitution.

## Live Job Snapshot

PID 3082595 was running at the recorded time. Verify current process identity
before acting. Log `/home/dieterolson/os7_g1f.log`; output
`/home/dieterolson/os7_g1f`. No completed receipt existed there at the snapshot.
Latest sampled iteration 236 showed large primal infeasibility (~4.69e4);
this is not convergence. Do not terminate or duplicate the owner's job.

Observed arguments: `--max-iterations 600 --replay-timeout-s 1800 --rungs 0.85
1.813888889 --warm-start /home/dieterolson/os7_g1b/rungs/0600ms
--continue-on-failure --ik-smooth-frames 15 --inverse-warm-start`.

`os7_g1f` is the inverse-warm-start attempt, not the older raw/smoothed-IK jobs.
`os7_g1e.log` reports a MocoCasADiSolver success; that alone is not a full G1
tracking/replay receipt. Do not use it to label the live f run accepted.

## Evidence and Interpretation

Raw receipts/log tails are preserved in the
[review snapshot](../matched_swing_program/evidence/continuation_20260918/matching-handoff-snapshot.json).
Local branch evidence holds completed runs under `evidence/os7_moco_g1/`.

| Run                 | Horizon             | Collocation Whole RMS | Replay Whole RMS | Verdict                                  |
| ------------------- | ------------------- | --------------------- | ---------------- | ---------------------------------------- |
| os7_g1              | 0.10/0.30 s         | About 41 mm           | About 41 mm      | Fails early/whole/yaw requirements       |
| os7_g1, Capped      | 0.60 s              | 41.6 mm               | 257.7 mm         | Iteration cap; rejected                  |
| os7_g1b             | 0.60 s              | 41.5 mm               | 81.4 mm          | Solver converged; replay still rejected  |
| os7_g1d, Diagnostic | 0.85 s              | 473.4 mm              | 902.6 mm         | One-iteration diagnostic; not a solution |
| os7_g1f             | 0.85/Full Requested | No Final Receipt      | No Final Receipt | Running at snapshot                      |

The 0.60 s run-2 replay has terminal 204.1 mm, club 97.2 mm, yaw 33.56 degrees;
it leaves the 60 mm error band around 0.417 s. Residual vertical support RMS
is about 758 N for a ~782 N body weight. This native phase-A model has no
qualified shared foot contact, uses pelvis residual actuators, attaches head
markers to the torso and welds the club only to the right hand. It cannot pass
the unassisted full-body physical contract regardless of solver status.

Do not describe all horizon replays as identical to collocation. They agree
approximately only on the short runs. Calibration floor, mesh/defect errors,
initial states, control interpolation and open-loop sensitivity must be tested
separately; the existing receipts do not prove one unique cause of divergence.

## Next Bounded Work for a Cheaper Agent

1. Inspect the existing f job once and collect its completed receipt/states/
   controls/replay or explicit failure. Preserve raw files and source/runtime
   hashes before doing any merging or rerun. Do not overwrite the b warm start.
2. Run the 27 pure ladder tests and receipt validation on the pinned source.
   Report requested versus actually replayed horizon, frame coverage, original
   validity mask, sparse-marker exclusions and smoothing. Trailing-marker gap
   trimming cannot satisfy full-capture G3; keep that row unverified.
3. Use `os7_merge_receipts.py` only after inspecting its CLI and validating each
   source run. Retain per-run provenance and distinguish best converged
   collocation from best physically accepted replay. A top-level receipt is
   not a substitute for individual rung evidence.
4. If f fails, do not immediately buy another 600-iteration solve. Inspect
   `inverse_warm_start`, state/control naming/scaling, initial speeds and the
   mesh/replay discrepancy using a bounded short-window diagnostic. Compare
   one change at a time against a frozen baseline, with a budget and receipt.
5. Phase A may continue using its own model/IK; it does not wait for MuJoCo.
   Phase B depends on #10339/#10340 shared model/markers and #10352 contact/
   closure conformance, then #10374 acceptance. Keep residual support and
   native-model scope visible; do not remove diagnostics to manufacture a pass.
6. Publish/integrate this branch separately from the handoff PR, after comparing
   current main changes and preserving the two local dirty items. Re-run native
   tests on the exact integrated source before claiming its old receipts apply.

The following is the recorded command, **not an instruction to launch a duplicate**:

```bash
cd /home/dieterolson/opensim-10341-runtime
/home/dieterolson/opensim-10003/bin/python \
  docs/development/opensim_tour_matching/os7_moco_g1_driver.py \
  --model /mnt/c/Users/diete/opensim-10003/os3b-scale-ik/golf_humanoid_scaled_tour_markers.osim \
  --trc docs/development/opensim_tour_matching/evidence/tour_average_tracked.trc \
  --ik-states /mnt/c/Users/diete/opensim-10003/ik_states_full.sto \
  --outdir /home/dieterolson/os7_g1f --max-iterations 600 \
  --replay-timeout-s 1800 --rungs 0.85 1.813888889 \
  --warm-start /home/dieterolson/os7_g1b/rungs/0600ms \
  --continue-on-failure --ik-smooth-frames 15 --inverse-warm-start
```

## Validation and Escalation

Executed on the local source commit: 27 tests passed with
`python -m pytest tests/opensim/test_moco_g1_ladder.py -q -o addopts=''`.
This is pure-data validation, not a new OpenSim native solve or acceptance.

Escalate contact/model topology, physiological actuation, changed initial-state
assumptions, cross-engine mapping or unresolved high infeasibility. A cheaper
agent can inventory, validate, package and run a single bounded diagnostic;
it should not independently waive gates or redesign the physical model.

Read [NEXT_AGENT_PROMPT.md](NEXT_AGENT_PROMPT.md) for the copy-ready assignment.

## Anatomical Playback and Muscle Scope

Owner clarification on 2026-09-18: the product must show an anatomical golfer
performing the complete swing, with explicit joint and muscle coverage. A
marker-only animation does not satisfy this user-facing deliverable.

Direct XML inspection of the preserved `os7_moco_g1/golf_humanoid_scaled_tour_markers_moco.osim`
found 23 bodies (including Club), 23 joints, 39 coordinates and 39 coordinate
actuators, with **zero muscle actuators**. Bone meshes include skull, ribcage,
spine, pelvis, arms, hands, legs and feet. Skull geometry belongs to the torso;
finger bones are visual meshes rather than independently articulated fingers.
Anatomical appearance does not establish anatomical completeness or muscle forces.

Local OpenSim 4.5 successfully loaded that model, its `rungs/0600ms/replay.mot`
and `inputs/ik_states_full.sto`; playback was exercised on both. The 0.60 s file
is a rejected dynamic replay. The 1.813888889 s IK file is the full kinematic
fit, not a validated muscle-driven forward simulation. Its unnamed storage
header causes a blank motion label in the GUI: package a clearly named copy
without altering raw evidence. Inspect apparent shoulder/arm mesh gaps and
club visibility in the GUI before shipping; do not infer the cause from a
screenshot or silently change physical geometry.

Required follow-up: qualify a distributable anatomical viewing package with
all referenced meshes, labeled IK versus dynamic motions, capture overlay,
full-duration playback and a reproducible video. Inventory modeled versus
omitted joints/muscles. Qualify a separate muscle-actuated OpenSim variant
with muscle paths, scaling, force capacity, activation/tendon dynamics and
native replay validation before advertising muscle-driven motion or muscle
loads. Coordinate torques are not muscle activations. Preserve the current
residual-actuated model as a diagnostic baseline. This modeling/validation
work requires expert review; a cheaper agent can first inventory, package,
verify rendering and collect evidence.

Implementation follow-up: #10394 (anatomical playback and muscle-actuated
model qualification), retained under epic #10363. Handoff PR: #10393.

## Golf Model Improvement Turnover

For the owner-requested anatomy, visible club and address corrections, follow
[EPIC_GOLF_MODEL.md](EPIC_GOLF_MODEL.md) (epic #10394, children #10395–#10403)
and [GOLF_MODEL_AGENT_PROMPT.md](GOLF_MODEL_AGENT_PROMPT.md). Start #10395,
then #10397. This is separate from resuming the prior Moco job. Preserve
that job and its evidence; corrected physical models invalidate old receipts.
