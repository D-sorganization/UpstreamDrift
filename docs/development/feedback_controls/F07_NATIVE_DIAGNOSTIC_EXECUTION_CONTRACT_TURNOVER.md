# F07 Native Diagnostic Execution Contract Turnover (#12236)

## Scope

The shared OpenSim diagnostic contract separates typed software admission,
the viability of one identified native runtime attempt, and physical
qualification. The last value remains `not-established` or `unqualified`; no
constructor, replacement operation, successful solve, finite replay, or score
can promote it.

Requests may provide a strict source-bound diagnostic opt-in and an optional
`NativeQualificationContext`. The context accepts only the registered
conditional-anatomy, support/contact/grip, and provisional-marker-registration
codes. It binds each finding to an evidence SHA-256 and provenance references.
It is absent by default and does not assert a fact about other models. JSON
cannot provide the positive audit findings needed for optimizer admission;
those are derived from native preparation checks.

## Runtime Evidence

The native short capability probe records the exact OpenSim runtime and native
binary identity, request/model hashes, initial named and discrete state,
native default actuator controls and controller-free protocol, native joint
transforms (including CustomJoint evaluation through OpenSim), position,
velocity and dynamics realization, finite state derivatives, muscle path
lengths and actuation, declared constraint residuals where applicable, and the
exact time interval reached. It delegates mechanics to the installed OpenSim
runtime. The reported interval is the actual probe interval, which is bounded
by the request and can be shorter than its requested horizon. A private failure
receipt preserves stage, exception class, bounded message, attempted horizon,
and actually reached time; it does not infer a stage when none was recorded.
Successful probe evidence uses the
`native-dynamics-closed-source-v1` protocol identifier and binds the probe
implementation bytes. The earlier failed moment-arm assay remains a separate
historical attempt with its original scope and is not relabeled as this probe.

An optional exact-source `NativeMechanicsExecutionProfile` lists serialized
`CustomJoint` and `MovingPathPoint` identities evaluated by OpenSim when their
functions exceed the legacy analytic subset. It is limited to constrained
muscle execution under a typed diagnostic policy. It does not waive source,
controller, force, state, constraint, or qualification requirements; wrapped
path closure remains separately declared.

Preparation requires generated positive audit findings for source identity,
state reconstruction, constraints, input and marker bindings, physical body
inertia, mechanics evaluation and short-probe completion. The initial profile
fails closed on source controllers and unclassified or unreproduced
non-muscle forces. Broader own-support/contact mechanics require an explicit
inventoried, reproducible profile; hidden actuation remains disallowed.

The assessment accompanies preparation and solve, is retained in the export
receipt, and is carried forward as a source-bound full-replay assessment in
replay and score receipts when those stages complete. Runtime phase changes
from short probe to optimization to full replay; findings and qualification
effects do not change.

The complex end-to-end profile fixture has not completed a Moco solve. Its
three-knot, 2 ms request passed preparation and entered `MocoStudy.solve()`;
the native process then remained at the `optimizing` stage with effectively no
CPU progress. An independently bounded 180 s investigation preserved the
progress sidecar and was terminated without inferring a solver cause. The
regression fixture now runs in a separate process with a 30 s cap, verifies
that it reaches the same recorded stage, and captures the timeout receipt.
This is a retained failure, not evidence of solve, export, or replay success.

## Validation

At final integrated head `3957ac912d90144d274b9615480d015cab0be783`, system
Python 3.13 central pre-PR validation passed all five gates: lint/format, diff
mypy (18 source files), affected tests, Semgrep/import policy, and policy/fragment
validators. The affected selection reported 291 passed and 241 skipped. Native
tests skipped only because the system interpreter does not have OpenSim; the
OpenSim-specific selection below used the pinned SDK.

The focused actual OpenSim 4.6 request/preparation/contact/wrapped/profile/runner
selection produced 134 passing tests. Its first launch omitted `CASADIPATH` and
10 solve-dependent nodes reported CasADi Ipopt DLL load errors (Win32 126). The
DLL was present at the pinned SDK package path; rerunning only those 10 exact
nodes with `CASADIPATH` set there and OMP/OpenBLAS/MKL thread counts at one passed
all 10. These are 144 distinct passing SDK tests, not a claim that the broader
suite passed. The bounded three-knot profile lifecycle node is a timeout-aware
regression; its pass does not promote the retained Moco solve/export/replay
claim. The prior broader 289-test SDK selection hit its independent 180-second
process-tree cap at roughly 65% progress without a pytest summary and is retained
as incomplete, not green. Path-work helper/native tests added in the child merge
passed 24 tests under the pinned SDK. Post-merge ledger/context/monolith
freshness tests passed 19 tests.

The SDK interpreter was
`C:\Users\diete\Repositories\.venv-feedback-opensim\Scripts\python.exe`,
with `CASADIPATH=C:\Users\diete\Repositories\.venv-feedback-opensim\Lib\site-packages\opensim`
and BLAS/OpenMP worker counts set to one. Normal merge hooks passed. The bounded
557-component preparation/probe remains runtime-viable only over `[0, 0.001] s`;
no optimizer or replay ran for that private source. The full-capture source,
physical qualification, and full-horizon replay remain blocked.

This test evidence covers the fixtures only. It does not qualify measured
anatomy, passive physiology, support/grip, full-capture matching, all-model
parity, or publication. Private source/evidence hashes and private capture
paths remain in their restricted receipts.
