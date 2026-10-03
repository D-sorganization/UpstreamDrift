# Necromatcher Authored Effort Profiles

## Purpose and Qualification

Use `necromatcher/effort-profile/2` for models with translation and rotation
coordinates. A translation in `m` receives generalized force in `N`; a rotation
in `rad` receives generalized torque in `N*m`. These are authored research
controls. Root forces may represent artificial actuation; they do not establish
contact reactions, measured human effort or a physically feasible swing.

## Required Fields

The JSON object contains exactly `schema_version`, `model_id`, `model_hash`,
`fit_id`, `fit_hash`, `dofs`, `coordinate_units`, `effort_units`, `timebase`,
`provenance` and `segments`. Obtain identity, hash and coordinate arrays from a
verified library model and `load_fit`; preserve their exact order. Both parents
must belong to the target swing, and the fit must bind that exact model version.
Coordinate units remain producer declarations until independently checked
against the compiled native model.

Set `timebase` to `physical_seconds` only for explicitly authored operator timing.
Do not copy video presentation timestamps and call them a calibrated motion clock.
Set provenance `kind` to `authored` and describe the control assumptions in a
nonempty `description`. Preserve the saved fit's rejected research qualification.

Each segment contains exactly `start_s`, `end_s`, `coefficients` and
`is_bernstein`. Coefficients are a finite numeric matrix with one row per DOF;
columns are Bernstein control points or ascending powers of normalized segment
time. Basis selection is a JSON boolean. Segments must be consecutive and share
channel count. Empty, string, boolean and nonfinite coefficients are rejected.

## Import, Recall and Export Procedure

1. Save the checked JSON outside the repository's raw-media boundaries.
2. Import through the existing native/web profile action or local route
   `POST /necromatcher/swings/{swing_id}/profiles` with `id` and `source_path`.
3. Recall with `library.load_effort_profile(profile_id, model_id)`. The returned
   object retains exact parent hashes, ordered DOFs and coordinate/effort units.
4. Evaluate with `controls.evaluate(operator_time_s)`. Out-of-interval,
   nonfinite and boolean times are rejected. Numeric evaluation overflow is
   rejected. Coefficients cannot be mutated through recalled arrays.
5. Export with `library.export_swing`. It revalidates profile bindings and units
   before publishing the package; an incompatible profile blocks the export.
6. Before downstream execution, compile the exact native model, independently
   verify DOF order and native units, distinguish root actuation from reactions,
   and retain an independently checked replay receipt. These steps remain open.

The canonical WorkspaceHandoff schema registry accepts v2 driving-profile
references. A red-first save/recall regression checks asset hashes, ordered units,
draft status and `qualification.passed=False`; 31 effort/handoff checks pass.
This establishes metadata transport, not an engine simulation adapter.

The existing `load_torque` refuses v2 profiles so consumers cannot silently lose
mixed units. Update consumers to the typed effort facade before simulation.

## Legacy Migration

Legacy `necromatcher/torque-profile/1` retains its all-rotation assumption only
when no verified bound fit establishes translations. It remains unqualified.
Import, recall and export reject a legacy profile for any model with a verified
fit declaring `m` coordinates. This also prevents export of an older profile after
new unit evidence is saved. Author a new version with explicit N/N\*m channels and
exact fit/model hashes; do not relabel numerical torque coefficients as forces.

## Verification and Remaining Work

TDD reproduced unsupported mixed-profile import and unsafe legacy acceptance,
then a separate finite-coefficient overflow test failed before repair. Regression
coverage checks immutable coefficients, ordered units, parent hashes, malformed
segments, cross-swing identities, API import and export refusal. Reuse the
canonical PiecewisePolynomialTorque evaluator; no second polynomial engine is
introduced. Historical controls, native unit qualification, independent replay,
impact/analysis consumers and AffineDrift integration remain pending.
