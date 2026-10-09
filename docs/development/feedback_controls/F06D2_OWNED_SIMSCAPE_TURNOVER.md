# Owned Native Simscape Replay Turnover

## Ownership and Current State

Issue11942; parent11921; branch `feat/feedback-simscape-owned-native-11942`
reuses `Worktrees/feedback-opensim-bundle-11908` after prerequisite PR11938 was
published. No new clone or runtime. Canonical chapter34 is the design authority.
The merged Tools T04 dependency 89415ee859 is pinned consistently in the
gitlink, Cargo manifest and Python requirements. Its complete tree matches the
f52a856e0 source used for actual native execution. Tools5472 separately repairs
the canonical-manual count gate; its local commit awaits network recovery.

## TDD and Limits

The missing-module RED preceded implementation. Fifteen synthetic byte tests
pass, as do focused Ruff and source mypy. They establish immutable model/blob
ownership, integrity and supported input/policy admission, absolute native time
export, unavailable-capability refusal and scoped cleanup. The initial fixture
also correctly failed Tools' force-frame requirement before its explicit axis
declaration was added. These are unit contracts, not native replay evidence.

The actual R2025b diagnostic from11933/PR11938 remains separately hash-bound in
fleet staging and Desktop previews. Its trusted fixture loader is not the new
native envelope decoder. No arbitrary MAT safety or production-model coverage
is asserted. Original producer input prehistory must stay distinct from future
replay inputs; do not constrain all future controls to the captured history.

## Actual Native Execution

R2025b Update 5 executes the owned envelope from0.2 through0.4s. Same and
deliberately changed future-force histories pass independent full-run comparison:
position errors7.4593e-17m and1.27566e-7m, below1e-5m; discrete errors1.11e-16,
below1e-12; actual executed-force error0N. Replay receives only suffix inputs,
without prehistory integration or native clock retiming. Different future
inputs retain the original producing prehistory identity.

Actual configuration/explicit simulation compilation tests pass, including
strict error diagnostics (interface/contents/release), repeatability, changed
MaxStep, existing input restoration and compile termination. Native request
digest/configuration negatives prove failure cleanup. Independent full-run and
replay tests use conflicting base/model-workspace input sentinels; explicit
model-workspace SimulationInput overrides select the saved values. The consumer
checks complete output clocks and rehashes all owned inputs after execution.

The source-hashed native record is `F06D2_NATIVE_OWNED_RECEIPT.json`. Actual
MAT/SLX and series remain in fleet staging `simscape_owned_11942` and the owned
Desk task folder; no capture or binary model is committed. The bounded MATLAB
provider hashes seven actual executed sources; Python admission/harness and
test hashes are recorded separately. It is not transitive-library attestation.
The public `prepare_f06d2_native_fixture.py` reproduces both exact request hashes
from the native producer, matching the originally executed archives.

Canonical chapter34 contains exact commands. Central gates and normal hooks
remain required before publication. The merged dependency is pinned; native receipt hashes distinguish
the executed MATLAB provider from later Python admission-only rejection guards.
Both request hashes are reproduced after those guards. Parent11921 and the entire
production/capture/parity denominator remain open.

The inspected preview is saved on the local Desktop in
`Motion_Matching_Previews/simscape_owned_replay_11942/native_owned_replay.mp4`.
The PNG, six original CSV files and hash-bound receipts accompany it; together
they occupy under0.4MB. The plot retains the native source-input/position sign
convention and labels both traces as synthetic diagnostic continuations.

Astra review reproduced capture-wall-clock admission and missing full envelope
retention before fixes. Native admission now requires simulation_absolute and
retains the complete validated envelope with its file hash, alongside the
compact request. Loaded-model identity, solver version and declared frame and
contact/load policies survive materialization for actual decoder comparison.

Production model/variant mislabeling produced two meaningful failing tests
before the diagnostic inventory guard. The guard changes no accepted request
bytes; it prevents the synthetic fixture from claiming driver/iron coverage.
