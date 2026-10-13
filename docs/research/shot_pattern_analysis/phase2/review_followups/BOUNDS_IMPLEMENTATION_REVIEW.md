# Bounds Implementation Review

Reviewed isolated commit `926dc6bada3d218e4099df7b73ad536314135dc7` in the
`shot-pattern-bounds` worktree. Main frozen scientific sources and active jobs
were not changed during this review.

The new request fields preserve legacy defaults: reference face ±3° and speed
factors [0.75, 1.25]. `bounds_for()` delegates validation to the existing
`MatchBounds`; `calibrate_patterns()` still calls the same public optimizer.
The solver's numerical equations and tolerance are unchanged. Three explicit
supplement selectors delegate to the standard matrix/export pipeline, preserving
recorded inputs, source snapshots, and manifest checks. The 18/4/4 inventory,
fixed targets, and separate driver speed factors match the reviewed plan.
Four alias candidates are declarations only, correctly withholding equivalence
until independent source/input verification.

The focused isolated unit/integration command passed 34 selected tests. The
native-only command passed both tests: exact legacy-bound nominal parity and
wide-box PW matching. It is recorded separately so default slow-test selection
is not mistaken for executed flight parity.

## Open Acceptance Finding

The existing final acceptance validator binds software execution and hashed
review evidence, but does not bind which supplemental protocol/cell inventory
was covered by its pilot. Consequently, a same-source receipt could authorize
an unreviewed protocol. Sol and root were asked to add a test-first explicit
supplemental inventory/protocol binding before final execution, preserving the
original pathway. This review is not final supplemental run authorization.

Independent source-projection review, nominal fine replay, the 32-shot pilot,
strict saved-data/alias checks, and immutable final acceptance remain required.
The separate symmetry review also forbids arbitrary draw/fade symmetry claims.

## Acceptance and GUI Follow-Up

Independently inspected isolated commit `22747504f3f529a0a9b46460a6c319674484028b`. The final gate now rejects custom selected cells or bounds without an explicit supplemental protocol. It binds all three exact current-source final catalogs, source identity, and three hashed pilot plans whose only configuration difference is 32 rather than 10,000 shots. This addresses the previously reported protocol authorization gap. Independent supplemental/GUI verification passed 24 tests (offscreen), recorded in `/tmp/astra-bounds-followup-tests.log`; no supplemental Monte Carlo was executed.

This is code-level acceptance only. Real pilot audits, numerical refinement, source transition, and a new immutable supplemental final acceptance remain required. The auditor's protocol schema agrees with the execution gate. GUI explicit bounds preserve legacy defaults and reject nonrepresentable imported values instead of silently clamping them.

The isolated plot typing changes in `ecc82796ce` were checked against the original frozen files and the recorded new files: the before/after hashes and reproducer hash agree. The recorded 84 exact density comparisons support the explicit float64 construction change; formatting changes have matching normalized ASTs. This limited equivalence evidence must be included in the eventual full source transition, not used to rewrite original acceptance.
