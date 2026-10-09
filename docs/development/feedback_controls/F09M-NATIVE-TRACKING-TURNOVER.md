# F09m Native Tracking Objective Turnover

Issue #12023; parent #11793; epic #11784. Branch
`feat/f09m-native-tracking-objective-12023` starts at F09l commit
`d5f003614426aef0ec6fdf2c0ed203942e25101c`, draft PR #12017.

## Objective and Authority

`myosuite_project_tracking.py` adds frozen native reference and physical-scale
objects plus a source-bound `ProjectTaskTrackingObjective`. It uses owned scratch
MjData and native `mj_differentiatePos` to compare positions on the configuration
manifold. It never forwards, integrates, resets or commands the live task.
Velocity residuals compare native qvel coordinates without transporting twists.
Activation and exact post-mapping command increments retain native units.
Positive physical scales make each squared residual dimensionless. These costs
are discrete sums on one frozen grid, not timestep-invariant integrals. Slew
cost is a soft penalty; the optimizer must separately enforce hard slew bounds.

Freeze absolute reference clocks, reference provenance declaration, all scales,
complete initial integration state, previous native controls, ordered actuators,
compiled/source/resource-closure identities and implementation hashes. Reject
changed parameters, live anchors, clocks, command bounds and model/source/SDK/
producer lineage. Preserve bytes-backed immutable reference arrays. A digest
binds a declaration; it does not authenticate anatomy, measured subject identity
or arbitrary provisional arrays. Callers serialize live task mutation.

Use the same scorer for provisional native predictions and fully guarded SDK
histories. `history_cost` is admissible as the existing promotion objective:
source/full-state live guards can nest, while the scorer owns its own scratch
lock. Guarded promotion recomputes commands and independently replays complete
serialized/reloaded T01 histories. Numerical costs cannot override hard physical
admission. A validated future plan is separate from an executed live prefix.

## TDD and Validation

Workspace planning retains actual MyoSuite 3.0/MuJoCo 3.6 evidence:

- `f09m-native-api-red.xml`: ten API-absence failures before implementation.
- `f09m-native-first.xml`: ten passes in 18.92 seconds.
- `f09m-lineage-red.xml`: first adversarial run includes a test-construction
  integrity rejection; preserve it rather than counting it as behavioral RED.
- `f09m-lineage-red-corrected.xml`: a genuine rehashed changed-source T01 bundle
  and changed SDK producer lineage each fail to reject; stale anchor and changed
  parameters already reject. Both lineage checks are then implemented.
- `f09m-tracking-green.xml`: sixteen passes in 56.23 seconds, including original
  driver and iron twenty-step cost agreement with full guarded promotion and
  independent native replay. These finite-state test criteria do not establish
  contact feasibility or motion matching.
- `f09m-native-full-campaign.xml`: final seven-module campaign passes 146 tests
  with zero failures, errors or skips in 152.81 seconds. All sixteen executed
  source/test hashes match (`f09m-native-source-hashes-final.json`). The first
  post-test hash collector failed to decode an older JSON file; a separate
  explicit-list collector recovered hashes without editing executed source.

Run the prior six-module F09l native campaign plus
`tests/unit/engines/myosuite/test_project_task_tracking.py` using the existing
pinned SDK environment and `--noconftest -o addopts=`. Retain combined JUnit and
all sixteen executed source/test hashes. Historical checkpoints bind their own
source, not a later edited module. Portable SDK skips are not native evidence.

Local corrected pre-PR checks use the checkout root as MYPYPATH. The first
attempt encountered duplicate `engines`/`src.engines` discovery. An unrestricted
transitive mypy run found one existing nullable multiplication error in
`validation_pkg/kaggle_validation.py:327`; no tracking-file error remained.
Keep these diagnostics separate from scoped checks and actual SDK execution.

## Next Integration and Scientific Gates

Reuse F02's bounded candidate-search callback and map its decisions to exact
mixed native controls. Keep motor torques distinct from muscle excitations;
never invent a constant torque map for state-dependent muscle channels. Supply
full-horizon signed hard margins, per-channel slew, safe-fallback admission,
explicit budgets and failure receipts before promoting selected commands.

Current references in these tests are declared native hold targets. Private
capture/body attachments, ground calibration, contact/grip, anatomy, muscle-only
assistance and physiological acceptance remain open. Keep all seventeen model
rows and six ecosystems in scope. The final endpoint remains a muscular OpenSim
private-motion reference match and independent full-state/full-horizon replay
of saved excitations with the model's own contacts. Preserve rejected candidates.
