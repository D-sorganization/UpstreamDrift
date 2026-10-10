# F09l Persistent Native Search Turnover

Issue #12014; parent #11793; epic #11784. Branch
`feat/f09l-persistent-native-search-12014` starts at published F09k
`d50db51b603e967c9e09a4b3c31f7f97a0fdaaac` / draft PR #12011. Parent integration
and the exact merged Tools pin remain separate acceptance gates.

## Authority and Ownership

`myosuite_project_native_search.py` owns one native context admitted from a
genuine independently verified seed. Reuse existing full-state restoration,
T01/profile admission and `native_direct_model_replay._execute_commands`.
The live task and SDK forecaster remain caller-owned. Serial live mutation is
the caller's responsibility. No new wire contract or native stepping loop.

Search model/source guards operate at invocation boundaries. This deliberately
weaker temporal policy makes all predictions provisional. Retain immutable
copied arrays and the existing genuine planned T01 bundle. Planned commands
are post-mapping native controls, bounded by explicit `max_steps` and native
limits; no implicit clipping. Preserve complete current epoch/activation/ctrl/
warmstart. Clear warnings only in owned reset; failures never return a partial
successful record. Reentrancy and stale snapshots reject.

Promotion recomputes selected commands through the existing fully guarded SDK
forecaster. Numerical callbacks receive bytes-backed immutable history arrays.
Require exact selected-command correspondence, strict Boolean hard admission
and finite objective. The existing freezer independently verifies every native
transition through canonical serialized/reloaded T01. A validated full future
plan remains distinct from an executed live prefix. Objective provenance is a
future optimizer responsibility, not attested by this callback interface.

## TDD and Reproduction

Workspace planning retains actual MyoSuite 3.0/MuJoCo 3.6 reports:

- `f09l-native-api-red.xml`: seven API-absence failures, no skips.
- `f09l-native-first.xml`: seven passes in 23.22 seconds.
- `f09l-callback-red.xml`: two callback array-writeability regressions fail.
- `f09l-native-second.xml`: twelve passes in 28.89 seconds after immutable copies.
- `f09l-native-full-campaign.xml`: 128 passes in 97.16 seconds, including real
  partial-owned-step recovery; this predates the two production search cases.
- `f09l-native-production.xml`: both original driver/iron twenty-step search,
  repeated-candidate and guarded-promotion cases pass in 51.94 seconds.
- `f09l-native-full-campaign-final.xml`: 130 passes, zero failures/errors/skips,
  in 126.06 seconds. All fourteen executed source/test hashes match the final
  campaign checkout (`f09l-native-source-hashes-final.json`).

The expanded suite adds unsupported-load, horizon, closure, model mutation,
reentrancy, nonfinite objective and real partial-native-execution recovery cases.
Run chapter 38's prior five-module actual SDK campaign plus
`tests/unit/engines/myosuite/test_project_task_native_search.py` with
`--noconftest -o addopts=` in the existing pinned SDK environment. Retain final
combined JUnit and all fourteen executed source/test hashes. Portable SDK skips
do not establish native availability. The shared forecast admission extraction
changes its source hash; historical F09k proof remains bound to its own head.

## Performance and Remaining Work

Construction/attempt/promotion timers include guards and failure overhead.
Separate four-history twenty-step prototype receipts retain observed guard,
restore, kernel, SDK and replay costs. They are one uncontrolled offline probe;
do not promise online deadlines or generalize its speed ratio.

Next reuse the existing bounded NMPC optimizer kernel with explicit actuator
command bounds, physical cost scales, reference/objective/criterion provenance,
warm starts and failure receipts. No invented constant-torque actuator mapping
for mixed muscle/motor channels. Full capture mapping, independent holdout,
anatomy, contact/grip and physiological acceptance remain open. All seventeen
production rows, six ecosystems and the final muscular OpenSim private-mocap
fit with independent full-state/full-horizon saved-excitation replay remain
required. Preserve all rejected scientific candidates and historical receipts.
