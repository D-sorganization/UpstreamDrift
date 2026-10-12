# Native Path-Work Diagnostic Turnover

## Scope and Ownership

F07 child #11856, branch `feat/feedback-path-work-11856`, isolates the same-body
moving-point concern from #11834 without modifying donor or runtime code.
Canonical calculation source is chapter 24; no second editable manual is made.
Discovery found the existing public path derivative and consistency providers;
this change reuses them rather than adding a duplicate admission or model schema.

## Evidence and Validation

TDD first produced 9 failures and 2 passes: NaN/Inf lengths, tolerances,
coordinates and unresolvable stencils could pass, and the new probe was absent.
Explicit finite and numerical-resolution guards now reject these cases even
when optional contract diagnostics are disabled. Original native fixed/moving
fixtures verify actual unit-tension acceleration, native moment arms, path
lengthening speed, achieved coordinates, and finite-difference refinement.
The moving case retains a 0.03 m discrepancy; this child detects the problem,
it does not claim the native implementation has been repaired.

Initial fixture construction required all three MovingPathPoint coordinate
sockets even for constant functions. Setting native controls also requires
realization through velocity after coordinates are changed. Both were corrected
in the original fixture. A 2e-9 m coarse-stencil assertion was too tight for the
observed central-difference truncation error (~1e-8 m); tests retain the declared
1e-7 m diagnostic threshold and independently require 1e-10 m agreement with
the known discrepancy at the finest stencil. This is a numerical fixture
correction, not a relaxed donor acceptance threshold.

Native provider: Python 3.12, OpenSim `4.6-2026-06-22-85aaf64`. Native test
commands are serial with `-o addopts=''`. Latest scoped result: 56 tests pass
across the new diagnostic, existing muscle audits and manual governance;
3 opt-in mocap fixture tests are deselected. Repository gates are recorded
in the PR handoff after execution.
The original fixture receipt is generated under local planning staging, not
committed as a fabricated qualification receipt. Exact official source hashes
and actual runtime extension hashes are recorded separately; complete binary
reproducibility and dependent-DLL closure remain unverified.

## Blockers and Next Steps

Primary-source interpretation correction: Sherman/Seth/Delp (2013),
https://pmc.ncbi.nlm.nih.gov/articles/PMC4404026/, and the official
FunctionBasedPath reference explicitly restrict tendon-excursion equality to
workless constraints. MovingPathPoints can introduce unaccounted guide work.
The numerical observations remain unchanged, but they alone do not prove an
OpenSim defect. Receipts now name rejection of the workless identity rather
than a runtime inconsistency. Retain force-defined native moment arms; an
explicit, physically qualified guide-work/power budget is an alternative to
path replacement. Do not prescribe a runtime or donor rewrite by default.
The old local receipt remains historical; the separate semantics-revision
receipt binds the clarified provider and labels. No tolerance was weakened.

No donor rewrite, runtime patch, upstream message, physiological fitting,
capture use, contact qualification or F07 closure is included. Review actual
force-law alternatives before admitting affected moving paths; retain donor
FPL/ECRL and coordinate-domain gates. A geometric-gradient substitution in
the optimizer cannot repair a different native force plant.

The sparse Tools checkout borrows its objects from the retained native-admission
worktree. Do not delete that object store until this borrower is repacked and
verified independent. Public fixture XML requires no foreign asset download.

## Child #12233: Source-Bound Chart Survey

`native_path_work_audit.py` adds a typed diagnostic contract over exact source
bytes, a loaded native component-path inventory, path/coordinate/path-point
identities, and the intersection of declared chart, conditional activity, and
function domains. It reuses the existing finite-difference provider and records
achieved coordinate spans, full path-length derivatives, native moment arms,
and independently force-derived moment arms for every requested stencil.
Out-of-domain samples, mutated source bytes, component identities not in the
loaded inventory, and unresolved native states fail closed. Its status is
sampled-domain evidence only; it does not change any replay admission or expand
the production MovingPathPoint law policy.

The fixed and moving synthetic native `PathActuator` fixtures are consumers of
the same audit. Native unit-tension acceleration independently reproduces the
native moment arm on both fixtures; the fixed-path full derivative passes the
1e-7 m/rad diagnostic threshold, while the moving-guide difference remains
0.03 m/rad at every tested stencil. The result classifies the workless identity
only and does not establish a runtime defect or donor acceptance. Dedicated
OpenSim 4.6 validation ran
`$env:CASADIPATH='C:\Users\diete\Repositories\.venv-feedback-opensim\Lib\site-packages\opensim'; & 'C:\Users\diete\Repositories\.venv-feedback-opensim\Scripts\python.exe' -m pytest -q -o addopts= tests/unit/engines/opensim/test_native_path_work_audit.py tests/opensim/test_native_path_work.py`:
24 tests passed (7 generic contract tests and 17 native test cases). The two
fixture cases each sample three coordinates with two finite-difference steps,
for 12 force-derived arm observations total. Each uses native unit-tension
acceleration and the independent identity `q_acceleration * inertia /
actuator_force`; this reproduces the native moment arm on the fixed and moving
fixtures. Full source-chart or wrapped-path qualification remains outside this
child. The ordinary pre-PR affected-test gate uses the system Python, so its
native OpenSim cases skip; the dedicated SDK command above is the native
validation evidence.

## Private Source-Bound Survey (Not a Replay Gate)

The private full-chart report is bound to mesh-free derivative model
`80a01950aaa9eb6d8b8cb00446caefe1fdf35b5784421866971db29bc692a243`, with
state-audit bytes `77d3023f13c0f050db1cab1cad235f2b3882ea334e45d58ab8aaff43e211ec6f`
whose parent source is `7090a6bd577745a2d514579c0650bb0d73241511f715ea6273b428b649abb93b`.
The report digest is
`a0a6f82ee151c9a0c873a9d04c922126bcf241d0f90dcc923c704045cf81ce91`. It
contains 240 sampled comparisons across active/inactive conditional-path
regions, transition sides, and three stencils. Two bilateral vastus
intermedius coarse `1e-3` rad samples exceed the fixed `1e-7` m/rad threshold
(maximum `2.25495e-7`); both pass at `1e-4` and `1e-5` rad. These sampled
observations do not establish a global bound, full wrapped-path behavior, or
replay qualification. Model and captured-state assets remain in their private
repository; this summary does not represent direct execution on the parent
7090 model.

## Current Parent Integration

Normal merge `ec9e786e1fd73e68a53e20fdfc4d8d658933e7ea` preserves the
current feasibility parent `2ee82c02c4c239ae5090a5397b4c8e9af56a825a`. The
dedicated OpenSim 4.6 audit suite passed all 24 tests again with zero skips.
Agent-context check and all five central pre-PR gates passed; that central
invocation mapped no tests, so the dedicated SDK run supplies native evidence.
The existing byte-bound closed-loop neck receipt is explicitly excluded from
Prettier to preserve its provenance during normal hooks. PR #12239 remains
a diagnostic audit and requires its parent integration before final merge.

### Optimizer-Domain Enforcement (Issue #12233)

Preparation now inspects every source `MovingPathPoint` under each loaded
`GeometryPath` before it can run the native capability probe. For each attached
coordinate, the allowed Moco state-value box must fit the intersection of the
native coordinate range, the request's source-bound declared chart, and every
attached reviewed function domain. The audit reuses the existing function-domain
reader: SimmSpline knot limits and the reviewed LinearFunction/Constant rules;
unknown functions, missing charts, incomplete point ownership, empty
intersections, and optimizer boxes outside the result produce a typed hard
execution blocker (`native_path_optimizer_domain_invalid`).

A preceding `ConditionalPathPoint` active range remains a local route-regime
switch and is not folded into a global optimizer box. This guard does not prove
that every internal function evaluation stays within domain between sampled
states, and it does not change guide-work qualification, physical qualification,
or the supported source mechanics. Unit regressions cover chart/function
intersection independently of CPP activity, unsupported/missing-domain refusal,
and refusal before a capability probe. A small OpenSim fixture assertion checks
the loaded GeometryPath/MovingPathPoint inventory when the optional SDK is
available; it is not a full-model replay claim.

## Native Model Connection Finalization

The optimizer-domain audit now runs after one Model.finalizeConnections() call
at the native-preparation boundary. This ensures MovingPathPoint axes are
connected before the source-domain reader inspects them. If finalization fails,
preparation returns the existing hard domain blocker immediately; it does not continue
into component/state/passive audits or the capability probe. No initSystem or
integration was added.

An isolated source-bound diagnostic observed that the model constructor alone
left moving-point coordinate associations unresolved. After one connection
finalization, the audit resolved all requested axes and the resulting knee
intervals matched the request bounds exactly. This is lifecycle and source-domain
evidence only. It does not qualify a full-model preparation or optimization;
the independent passive-policy blocker remains, and all 17 registered
model/variant rows remain unqualified.

The orchestration regressions exercise both success ordering and fail-closed
behavior. On a finalization exception, spies for component, state, domain,
passive-policy, reference, and capability-probe work remain untouched.
