# F09j State-Dependent SDK Feedback Turnover

Issue #12003; parent #11793 and epic #11784. Branch
`feat/f09j-project-feedback-recording-12003` starts at F09i
`976feaf0513f55c0112e8b9c9f3390bcbab55f3e`. Parent and Tools feature dependencies
remain unmerged; this does not establish main authority.

## Implementation and Contracts

`myosuite_project_feedback.py` exposes immutable native observations, declared
policy provenance and `record_project_task_feedback`. Fixed tables and feedback
share the existing producer's admitted loop. No new simulator, replay loop,
wire format or optimization algorithm is added. Read canonical chapter 38 and
the project-task native replay boundary before extending the pathway.

The callback receives copies without task/model/data handles. Immutable
bytes-backed arrays reject write access. Native state changes, model changes,
callback installation, implementation changes, source changes and invalid
commands fail before the next integration. Exceptions return no successful
partial recording. Numeric action conversion runs inside the state/source guard.
Completed steps remain on failure; no rollback or retry occurs. The callback is
ordinary Python, not a sandbox; source and
parameter declarations are not signatures or recursive dependency attestation.

Retain `ProjectFeedbackRecording` provenance alongside T01 inputs. Its applied
input array digest binds actual post-mapping command bytes. The explicitly named
`initial_state_array_sha256` and `applied_command_array_sha256` are raw-array
hashes, distinct from canonical schema/channel/clock-bound T01 digests. The
feedback adapter source digest remains separate from declared policy source.
Native replay provider identity
does not become controller identity. Reference-derived initialization and
optimizer-assembled initialization remain distinct from source defaults.

## TDD and Actual Evidence

Local actual MuJoCo 3.8 snapshot/budget checks: five API RED cases, then five
passes; six SDK tests explicitly skipped locally. The remote staging initially
lacked test package markers and failed collection; unchanged repository package
markers were copied before the actual API RED run. Actual MyoSuite 3.0/MuJoCo
3.6: eleven API RED cases, then eleven passes. The combined unchanged-production,
resource/export and feedback campaign initially passed 87 tests with zero skips.
Astra then reproduced a state mutation during deferred numeric command conversion
in an actual native RED regression. Moving conversion inside the guard resolved
it. The final actual SDK campaign passes 89 tests, including 13 feedback cases,
with zero failures, errors or skips in 32.265 seconds. All ten executed source
and test hashes match this checkout. Historical RED and 87-pass reports remain
separate from `f09j-native-full-campaign-final.xml` and
`f09j-native-source-hashes-final.json` in workspace planning evidence.

The positive fixture records four state-dependent actions and independently
replays complete states with feedback disabled. Negative cases alter plant
state/model, install a native callback or emit invalid actions; integration time
does not advance. Receipts and executed source hashes are retained in the
workspace feedback-controls planning directory. Reproduce with the reviewed SDK
pair and exact Tools pin using `--noconftest -o addopts=` and the four
`test_project_task_*`/`test_native_model_resource_closure.py` modules listed in
chapter 38 and planning receipts.

## Remaining Acceptance

The new policy is a fixture controller, not a fitted swing controller. Sol's
separate revision-2 bounded experiment independently admits an optimizer-derived
iron initial pose; driver remains rejected by contact limits. This is no capture,
stable-stance, physiology or full-horizon qualification. Qualified marker
correspondence, gain/control optimization, muscle allocation and physiological
source readiness remain open. Keep all 17 rows and six ecosystems; full-horizon
muscle-driven OpenSim matching and independent excitation replay under its own
contact remain the final endpoint.
