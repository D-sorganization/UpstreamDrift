# F07 Native Contact Source Closure Turnover

## Scope

Issue #12244 adds a typed exact-source closure for the existing native sphere/
half-space contact path used by constrained muscle replay. It reuses
`OpenSimForceTorqueSource` and the shared wrench sampler. The #12236 request
integration serializes and revalidates the declaration through preparation,
export identity, and fresh replay. It does not add contact to the exact
557-muscle private source or authorize capture contact.

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

OpenSim 4.6 focused contact checks pass 16 tests. Both declared force types
and sphere/half-space component closures are exercised. Smooth and
Hunt-Crossley contact each build and freshly replay through the constrained
path with finite native wrench samples. A Hunt-Crossley `NativeMocoRequest`
passes the runner's component audit only with the exact declaration; a
mutated force digest blocks. JSON digest and unknown-field negatives pass.
Legacy native replay continues to exercise both laws through the shared
sampler. Undeclared contact is rejected for both laws.
The skipped cases are inapplicable legacy extra-geometry checks.

The provider/parser merge `c6824895092e81c4859bc0c8a6cbc273cde22ca6` retains
the native C++-qualified XML parser fix and its security regressions. The
request/export/replay integration is committed in `7f47f3aaf1`; the normal
merge commit is `3a18a64c2c`. Full Moco solve/export/replay success remains
unproven for the complex source profile, and no optimizer qualification is
claimed by the bounded native capability probe.

No test establishes calibrated capture contact, human support or grip,
physiological validity, full-body Moco solve, full-horizon replay, or
all-model parity. Chapter 58, SPEC, registry, and this turnover keep those
qualification claims open.
