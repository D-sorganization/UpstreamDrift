# Necromatcher Native Resource Procedure

## Verified Resource Boundary

`workspace.load_native_fit_binding(library, fit_id)` opens hash-checked fit,
model and capture versions, reproduces the saved XML from the recorded native
definition, compiles the native plant and compares its declared order and units.
MuJoCo units come from each named compiled joint: scalar slides use m and hinges
use rad. Missing capabilities and unit disagreement fail instead of assuming
rotational coordinates. Other engines remain unsupported by this boundary.

The returned `NativeFitBinding` provides one plant for projection, fitting and
future downstream execution. `project(frame_index)` preserves source identity
and research qualification; `review_inputs()` reuses canonical camera validation.
The refit worker compiles once and reuses that binding instead of compiling a
second plant after projection validation. This reduces duplicate setup; it is
not a measured throughput claim.

## Authored Control Boundary

Load an authored profile with `library.load_effort_profile(profile_id, model_id)`.
Call `binding.efforts(profile, operator_time_s)` to obtain native coordinate-name
to generalized-effort commands. This checks exact model and fit identities and
hashes, coordinate order, compiled coordinate units and conjugate N/N\*m effort
units before evaluating the canonical polynomial. Operator time must be finite
and within the authored interval. The binding retains model identity independently
of mutable detached research samples; editing sample metadata cannot relabel
controls for another compiled model.

This maps authored forces and torques; it does not compute measured efforts.
Root forces can represent artificial actuation and must not be reported as
contact reactions. Source PTS does not become a verified physical clock.

## Runtime and Qualification

Use the existing clean-interpreter worker when invoked from a Qt host. Direct
SDK imports into an already initialized Qt host remain inappropriate on Windows;
NativeFitProjectionProcess continues to own source-frame review outside Qt.

Native resource and unit verification does not qualify player anthropometry,
camera calibration, grip closure, anatomical ranges or forward dynamics. Keep
saved historical fits as monocular research hypotheses and handoffs as draft,
with scientific qualification false. Independent forward replay, impact/whole
analysis consumers and AffineDrift integration remain open.

## Verification

Eighteen new native-boundary cases cover compiled units with reversed coordinate
order, one compilation reused for projection, model/unit mismatch, missing native
capability, malformed camera records, semantic JSON key ordering, exact control
bindings and immutable model identity. Red-first tests reproduced missing native
units, camera mapping errors, absent control mapping and mutable identity bypass.
The combined profile/library/fit/refit/handoff/native suite passes 96 tests;
one real-Drake check skips because the installed runtime is mocked. Native and
source-clock qualification remain separate from these software-contract checks.

## Actual Saved Player Resources

The [Native Resource Receipt](historical_capture/native-unit-resource-receipt-v1.json)
records execution from exact Python source commit `54d8f6fb136fcdab2ed7349df1c78217369df34b` with runtime/source
fingerprints. Hogan's 750-frame v5 fit and Tiger's 210-frame v5 fit bind their
saved native XML and research definitions. Each compiles with three translation
and 41 rotation units. Start/middle/end review returns thirteen native attachments
per player. The same compiled binding is reused across these three projections.

No historical controls were evaluated in this resource check. Physical timing,
dynamics replay, anatomy and independent motion acceptance remain unqualified.
The full source-file receipt is retained outside Git with the analysis artifacts.
