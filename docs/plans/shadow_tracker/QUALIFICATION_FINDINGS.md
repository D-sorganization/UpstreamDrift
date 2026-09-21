# Model Qualification Findings

## Verdict

**The existing MuJoCo model executes, but is not qualified for Shadow Tracker.**
ST-01 scientific acceptance remains incomplete regardless of issue state.
Source/mask/camera bookkeeping can proceed under
[Contract Freeze](CONTRACT_FREEZE.md); fitting and scientific claims stay blocked.

## Measured Evidence

The [10-Frame Receipt](evidence/model_probe_10_frames.json) records three fresh
passive rollouts over 0.025 s using the existing measured C3D, stored initial IK
pose, zero initial velocity and zero control coefficients. These assumptions
make a reproducibility probe, not a reconstruction of a golfer's actual motion.
The model has 24 bodies, 23 joints and 41 scalar coordinates (`nq=nv=41`).

- All three integrations returned the requested times and finite states.
- Repeated nominal q/v traces were identical in this environment.
- Ground calibration stayed fixed across each run.
- Tighter integration tolerances changed translation by about 2.0e-10 m and
  rotation by about 7.3e-8 rad over this short window.
- Historical scrambled baseline (commit `9814c348`): reported **1.266 m of grip
  displacement and 2.350 rad of grip rotation** due to legacy `qpos[i] = q_arr[i]`
  sequential copying in `MujocoFullBodyIK` (#10068).
- Coordinate mapping fix (#10140, PR #10164): aligned `MujocoFullBodyIK` with native
  `qpos` DOF addresses via precomputed `_qpos_indices`, establishing exact FK parity.
- Regenerated evidence (#10167):
  - Calibrated marker offsets re-solved on 34 markers (subsample RMS: 145.2 mm vs legacy 174.5 mm).
  - 654-frame full trajectory IK re-solved (mean frame RMS: 133.7 mm vs legacy 157.7 mm;
    total RMS: 138.1 mm vs legacy 159.5 mm).
  - Initial grip translation error dropped from **1.265687 m down to 0.013733 m (13.7 mm)**,
    matching exactly across both IK and forward dynamics paths.
  - Initial grip rotation error improved from 2.350 rad to 1.657 rad.
  - Historical evidence is preserved in `docs/development/full_body_models/evidence/fb4_calibration/mujoco/historical/`
    and `docs/plans/shadow_tracker/evidence/model_probe_10_frames_historical_scrambled.json`.
- Physical acceptance status: Despite the 92x improvement in translation closure, the rollout
  **remains scientifically unqualified** (`scientifically_qualified: false`). The 13.7 mm translation
  residual exceeds the 5 mm physical tolerance, and rotation residual (1.657 rad) exceeds the 0.05 rad
  tolerance. `ForwardRolloutResult.is_closure_accepted()` fails closed as required.
- Production closure units separation was implemented in #10141 (PR #10165):
  `ForwardRolloutResult` and `EngineReplayOutcome` record `max_closure_translation_m`
  and `max_closure_rotation_rad` separately and enforce physical tolerance checks via
  `is_accepted()`. Under these physical criteria, the rollout is rejected despite solver success.

The probe records initial IK and dynamics closure separately to make this
disagreement reproducible. It does not fix the existing engine or validate the
canonical-v2 adapter. Native scalar RPY coordinates cannot be copied into the
canonical quaternion state (`nq=nv+1`).

## Legacy Receipt Interpretation

The existing FB-5 receipt says PASSED while reporting 0.110 m penetration and a
`max_closure_residual_m` near 2.67. Source inspection shows the latter is the norm
of a six-vector combining xyz displacement in metres and rotation in radians.
It is **not a physical distance of 2.67 m**. The probe splits the quantities.
Do not change or overwrite the historical receipt; preserve its source version
and record corrected evidence separately after the adapter fix.

Additional capability limits: HeadFront/HeadSide/HeadTop bind to rigid Hub; the
current visual skeleton uses generic capsules/spheres, not a subject silhouette;
the rollout API requires marker capture; failure output can contain zero-valued
diagnostics, so status must always gate their interpretation.

## Prerequisites for Dynamics Work

1. Fix named-coordinate IK packing; establish independent FK parity with the
   dynamics path; audit saved trajectory order and regenerate IK/offset receipts.
2. Separate translation and rotation closure metrics in production reports and
   require actual numeric acceptance, rather than solver-success labels.
3. Implement/test the full-body native-to-canonical state and velocity mapping,
   including base frame/rotation offsets and the RPY Jacobian. Generic free-joint
   adapters are insufficient evidence for this scalar-joint model.
4. Decouple rollout from marker scoring without fabricating `TourCapture`.
5. Qualify morphology, head/hand/club geometry and contacts, then preregister
   physically justified G2–G5 profiles and ground-truth holdouts.

All original scientific acceptance requirements remain. No human-accuracy,
club-impact, whole-swing, contact-force or uncertainty-coverage gate is passed
by the 25 ms experiment. The current development target table stays provisional;
there is insufficient evidence to freeze it as a release profile.

## Reproduction and Test Evidence

```bash
python3 -m scripts.shadow_tracker.model_probe --frames 10 --output receipt.json
python3 -m pytest tests/unit/shadow_tracker tests/integration/shadow_tracker/test_model_probe.py -m "unit or live_simulation" --no-cov --timeout=60
python3 -m ruff check scripts/shadow_tracker tests/unit/shadow_tracker tests/integration/shadow_tracker
python3 -m ruff format --check scripts/shadow_tracker tests/unit/shadow_tracker tests/integration/shadow_tracker
python3 -m mypy scripts/shadow_tracker --follow-imports=silent --ignore-missing-imports
```

Initialize the exact Tools pin before running. This worktree used the already
initialized sibling UpstreamDrift vendor at
`e83bd2e4a7a29a2dcd8145ef2d1efa07123324f0`, verified equal to its gitlink.
For pytest, set `TOOLS_REPO_PATH` to that provider. For the probe process,
`PYTHONPATH` included its `src/shared/python`, `src`, and `src/python/src` roots,
matching the repository's vendor search order. No provider files were modified.
The CLI records engine/environment, model/assets and probe hashes. Time measurements
exclude general Python process/import startup and do not imply production latency.
The committed receipt was generated before the probe's first commit: its Git
field identifies the base checkout, while `source_sha256` and `probe_sha256`
identify the exact experimental source files. The handoff PR supplies those
files; do not assume the base commit alone contains the probe.

TDD: the metrics suite first failed importing missing `pilot_metrics`, then 13
cases passed. The probe test first failed importing missing `model_probe`, then
passed; the added native-order evidence test failed with `KeyError` before the
diagnostic was implemented. The CLI source-hash test likewise failed before the
receipt metadata was added. The combined focused run passed 16 tests, including
real MuJoCo replay and CLI persistence. Known unrelated import deprecation
warnings remain.
