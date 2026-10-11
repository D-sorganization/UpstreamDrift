# F07 Native Moco Physical-Body Preflight Turnover

Issue #12205 makes the existing #12191 body audit a necessary condition for
`prepare_native_moco` and `solve_native_moco`. Preparation writes
`physical_body_inertia.json` with the full native audit, including source,
adapter, shared validator and OpenSim runtime identities. The preparation
report records the sidecar digest and admitted status. A physical failure is
retained alongside other independent preparation blockers; read-only audits
remain available.

The solve/export/replay input guard requires the original sidecar and performs
a fresh full source/native audit before proceeding. It compares the complete
audit bytes, not only a model hash or caller assertion. Missing old receipts,
modified sidecars, validator drift and native invalid inertia reject. No
tensor law or tolerance was duplicated in the runner; it calls
`audit_native_body_inertia` and its shared principal-moment validator.

TDD first showed four missing-boundary failures. The native Millard and Thelen
regressions now exercise valid admission, rotated physically invalid inertia,
malformed massless source, old/missing audit, sidecar tampering and validator
drift. Canonical chapter53 and the calculation inventory remain provisional.
The exact 557-muscle source still has four invalid positive-mass tensors.
Passing this necessary gate on another source would not establish anatomy,
passive-force, contact, marker or full-horizon qualification. Preserve all 17
variants and six ecosystems in the acceptance denominator.
