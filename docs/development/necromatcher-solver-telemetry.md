# Authoritative Solver Counts and Scoped Timing

Tracked under #11388, #11235 and #11284. Measurements describe computation,
never scientific acceptance. Existing historical records remain unchanged.

## Typed Measurements

The SDK-free public estimation facade exposes frozen `SolverTelemetry` and
`SolverBackend`. Counts `nfev` and `njev` are actual backend-reported nonnegative
native Python integers: booleans, floats and strings are rejected. Unsupported
counts remain null. The single-trial MAP `n_iterations` compatibility field
continues to mean SciPy `nfev`; this does not relabel other solvers' iterations.
Backend name, effective method and installed version accompany new records.
LM requests constrained to a finite domain record the effective TRF method.
Post-solve residual diagnostics are outside backend counts.

Complete new records require every measurement key and their schema/scope
constants; malformed or partial new records fail. Only absent legacy telemetry
decodes as `legacy_record_has_no_telemetry`. No requested budget, filesystem time,
UTC stamp or polling frequency can fill missing measured values.

## Separate Monotonic Scopes

- `solver_elapsed_s` measures only the backend invocation with `perf_counter`.
  Packing, identifiability, post-solve diagnostics and serialization are excluded.
- `worker_elapsed_s` measures child entry through the computed fit payload or
  caught computation failure, including request decode, native binding, fitting
  and dense diagnostics. Interpreter imports/startup, telemetry publication,
  final transport serialization, queue wait, network and UI time are excluded.
- Parent timeout/cancellation cannot supply a missing child duration or count.
  No parent process-wait interval is presented as worker or solver time.

The all-fixed MAP path reports zero residual evaluations, missing Jacobian count,
backend `canonical_fixed_decisions` and no fictional solver-call duration. An
unoptimized initialization has `optimizer_not_run`, not a fabricated optimizer
termination. Available solver data can coexist with unknown worker transport
data; `unavailable_reason` identifies the missing measurement scope.

## Owned Terminal Persistence

New fit evidence preserves `solver_telemetry` under `original_fit`. Canonical
fit import/recall validates it without rewriting stored bytes. An optional
`worker-telemetry.json` sidecar belongs only to the existing canonical run root;
it binds run identity, exact raw request SHA, source fit identity/hash and source
and runtime digests. Source request paths must be regular and link-free. No
user-selected telemetry path or second result store is introduced.

The owner compares the live queue recipe with the stored request in the finite
JSON domain: typed tuple sequences and JSON arrays represent the same ordered
values. Mapping keys must be strings, and Boolean, integer and floating-point
values retain distinct encodings. Changed nested values and nonfinite numbers
are rejected. This comparison does not replace the exact raw request SHA in
the sidecar binding; even a semantically equal byte rewrite invalidates an
existing receipt. It does not rewrite historical failed runs or publish their
unpublished results.

Child success and named failure can publish measured worker duration. A killed
child without a terminal receipt leaves all missing child measurements null and
the parent records `child_terminal_receipt_unavailable`. Generic job manifests
and diagnostics remain the status authority. Valid sidecars are checked against
candidate telemetry before publication; foreign or malformed bindings fail
closed on reading. Optional sidecar publication faults are logged and do not
convert a successful computation into failure or replace its original fault.
The successful fit can retain actual solver counts with unknown worker duration
and `worker_telemetry_publication_failed`. No cancellation publishes a fit.

## Repeatability and Limits

Red-first tests cover analytical early stopping under a larger requested budget,
all-fixed bypass, strict schema/count/duration validation, fake separate clocks,
exact request bindings, pre-solver failure, parent timeout, sidecar transport
failure, immutable legacy bytes and successful recall. These software fixtures
are not new historical Tiger/Hogan executions. Keep old solver-budget reports
immutable with unavailable actual counts; do not backfill them after this feature.
Telemetry cannot promote convergence, historical anatomy, physical clock,
contact/dynamics qualification, public publication or goal completion.
