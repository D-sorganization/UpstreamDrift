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

## Issue #12342: Persist the Observed Audit

Native preparation now atomically stores the same `NativePassiveReadinessAudit`
object returned by its single observation call. The receipt binds request and
source identity, loaded model and runtime, prepared-state observation digest,
observer implementation, explicit optional policy identity, and the complete
muscle inventory. The solve entrypoint verifies receipt bytes and these expected
identities before proceeding; a missing or stale receipt blocks admission. This
is an observation receipt, not a complete restart-state artifact and not a
physical qualification. Policy absence remains an explicit blocker, and the
existing unqualified status is unchanged.

This source-specific OpenSim receipt changes no qualification status in the
17-row model inventory and makes no implementation or parity claim for other
engines.

Validation on the supported Python 3.12 QA environment:

```text
python -m pytest -o 'addopts=' -q -rA tests/opensim/test_native_passive_receipt.py tests/opensim/test_native_passive_readiness.py tests/opensim/test_native_moco_runner.py tests/opensim/test_native_moco_diagnostic.py tests/unit/engines/opensim/test_native_moco_chart_bounds.py
```

Result: 41 passed, 66 skipped because this QA environment has no OpenSim module,
and 2 warnings. The full required pre_pr completed with exit 1 at the existing Fleet Policy
Gate; the other required gates passed. The fleet findings include missing verifying
SHAs for entries #8869 and #9193, missing State and Owner for #9349, 144 active
entries against a 75-entry ceiling, and an oversized Development Log. These
findings are existing policy debt and are not reported as passing.

The durable pre_pr log is
`C:\Users\diete\Repositories\_temp\passive_receipt_pre_pr_final_20261012.log`
(SHA-256
`af6c98b0bb766dfd9ebf6f78b34b3526bff4b88b0a37f6d0432561cb55ea1030`).
An earlier pre_pr attempt was interrupted during consistency tests after an
idle owned pytest process; that run is retained as incomplete, not counted as
a pass. The later full pre_pr run completed the consistency gate successfully.
The receipt tests cover complete audit roundtrip, one-observation propagation
through preparation, consumer-side tamper and
stale-identity rejection for opt-in and default modes, malformed payload and
inventory refusal, and atomic-write failure through preparation. No native
execution or solve was run for this receipt change. The fragment validator
accepted the new #12342 fragment. Design-manual governance verified the 62 QMD
sources and retained `release=blocked-inventory-required`; calculation/manual
release remains blocked.

### Receipt Validation Maintenance (2026-10-12)

The passive receipt validator now delegates native-state and muscle-coverage
checks to `_validate_passive_muscle_inventory`, keeping the public validation
sequence and fail-closed errors unchanged while satisfying the 100-line
function budget. The targeted receipt module passed 27 tests; architecture and
file-size budget checks passed. Agent-context render/check passed, and its
12-query evaluation passed after registering the receipt source and test.
