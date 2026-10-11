# F07 Passive-Readiness Turnover

Child #11939, branch `feat/feedback-passive-readiness-11939`, builds on the public
native constraint observer from #11923/PR #11931. Parent #11791/#11792 and epic
#11784 remain open. Chapter 33 is the canonical calculation reference.

The new native observer and enforcing entrypoint require explicit model-bound,
per-muscle sourced limits and runtime compliance/activation flags. They retain
separate elastic, damping, net passive, tendon and along-tendon force evidence.
No source parameters, native laws or existing exploratory APIs are changed.
Passing this necessary gate never grants muscle-matching qualification.

TDD: missing module RED; native equilibrium alone incorrectly satisfied passive
readiness; runtime flag mutations bypassed XML identity (two RED cases); and a
native Millard nonsteady cancellation fixture bypassed the initial net-only gate
(RED). Separate elastic-force admission passes all regressions. Current actual
Python 3.12/OpenSim 4.6 suite: 22 passive plus 18 constrained-state tests pass.
The existing constrained-source subprocess test now declares its repository cwd,
fixing a reproduced import failure under the shared cwd-changing test fixture.
The queued #11931 branch was not edited.

The unchanged 520-muscle source preserves all 1348 named values and source/loaded
hashes. Its new local receipt is under the fleet workspace at
`docs/development/feedback_controls_planning/native_model_evaluation/native_passive_readiness_11939.json`.
All 520 muscles are observed; missing physiological policy remains unavailable.
MF_m5_laminar elastic/Fmax is 62.37697; IL_L4 fiber-axis passive is 48406.73 N
versus tendon/along-tendon 48066.55 N, reflecting force direction/pennation, not
damping. The separate synthetic Millard cancellation is recorded in chapter 33.
No new equilibrium, source modification, geometry download or capture access
was required. Exact torso donor ancestry, state/resource closure, locks/couplers,
assistance, anatomy, grip/contact and capture qualification remain blocked.

Next: obtain source-backed per-muscle limits and donor assembly/neutral-posture
provenance; integrate the necessary gate when a full-source matching consumer
exists, retaining all independent acceptance gates. Local default-runtime skips
must never be reported as native passes. All five central pre-PR gates pass,
including changed-source mypy and default-runtime mapped tests (6 passed,
34 native skips, with separate actual-native evidence above). Ruff, fragment
validation, title checks and manual governance pass; 11 governance tests pass.
Normal hooks and protected CI remain to be recorded after execution. The fleet
inbox returned incomplete evidence with its known malformed/page-limit warnings;
direct root coordination preserves the ownership partition. Generated manual
release remains blocked.
