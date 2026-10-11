# F07 Native Contact Source Closure Turnover

## Scope

Issue #12244 adds a typed exact-source closure for the existing native sphere/
half-space contact path used by constrained muscle replay. It reuses
`OpenSimForceTorqueSource` and the shared wrench sampler. It does not add
contact to the exact 557-muscle private source or authorize capture contact.
LunaTools owns request/runner serialization and lifecycle wiring; this branch
contains provider and constrained replay support only.

## Identity and Admission

`NativeContactExecutionDeclaration` binds the model SHA-256, exact supported
force paths/classes/native serializations, and the complete supported contact
geometry inventory. The executed-policy identity binds the declaration and
the contents of both `native_contact_policy.py` and
`opensim_force_torque.py`. Constrained bundle provider identity also includes
the wrench-provider bytes, so implementation changes invalidate existing
policy identity. Contact remains rejected without a valid exact-source
declaration; source or component changes invalidate it.

Supported force types are `HuntCrossleyForce` and
`SmoothSphereHalfSpaceForce`; supported geometry is `ContactSphere` and
`ContactHalfSpace`. The provider delegates wrench sampling to the existing
OpenSim force/torque implementation rather than defining another law.

## Validation and Limits

OpenSim 4.6 focused checks: 73 passed and 2 skipped in the combined contact,
legacy muscle replay, and mechanics-profile slice. The dedicated contact file
passes 8 tests. Both declared force types and sphere/half-space component
closures are exercised. Smooth contact builds and freshly replays through the
constrained path with finite native wrench samples. Legacy native replay
continues to exercise both laws through the shared sampler. Undeclared contact
is rejected for both laws; the generated HuntCrossley fixture also hits the
strict self-contained XML rejection before component-level constrained replay.
The skipped cases are inapplicable legacy extra-geometry checks.

Commit `316b011324` adds provider-implementation identity and Chapter 58;
`b48f7b053b` adds both-law default rejection coverage and clarifies the
HuntCrossley limitation. Normal hooks passed on both commits. The parent-
relative central gate is still pending.

No test establishes calibrated capture contact, human support or grip,
physiological validity, full-body Moco solve, full-horizon replay, or
all-model parity. Chapter 58, SPEC, registry, and this turnover keep those
qualification claims open.
