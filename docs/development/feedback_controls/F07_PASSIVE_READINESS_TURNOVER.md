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

## Main Integration After the Constraint Observer Merged

The prerequisite constrained-state observer #11931 merged to main as
`2e3c79b4808f334b06dab5828db0d22dd7f2f80a` on 2026-10-09. This branch
genuinely merged that main commit. Conflicts were limited to the pre-existing
constrained-state turnover, the calculation registry, the manual chapter index,
and the subprocess observer test: the resolution retained main's prior native
torque and geometry entries, this branch's passive chapter and registry blocker,
main's observer integration history, and the explicit repository cwd in the
subprocess regression. The observer source blob is identical in parent and
main (`589c0108a2995d7f118fe686c585180ba8b2c473`). The pinned Tools
submodule remains `2e7665111b06f92ffbfe178b92d74d6a81c95388`.

On the merged tree, 60 actual OpenSim 4.6 observer, passive-readiness and native
marker-geometry tests pass. The OpenSim environment lacks `cv2`; that native
test process appended the existing Python 3.12 host site-packages only after
loading its own NumPy 2.5.3. No package or physics provider was changed.
Tests remain software/native-boundary evidence; physiological policy and
capture matching are still unavailable. Eleven manual governance tests and all
five scoped central pre-PR gates passed on the merged tree. Normal hooks,
protected CI and final PR disposition follow the integration commit.
