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

No donor rewrite, runtime patch, upstream message, physiological fitting,
capture use, contact qualification or F07 closure is included. Review actual
force-law alternatives before admitting affected moving paths; retain donor
FPL/ECRL and coordinate-domain gates. A geometric-gradient substitution in
the optimizer cannot repair a different native force plant.

The sparse Tools checkout borrows its objects from the retained native-admission
worktree. Do not delete that object store until this borrower is repacked and
verified independent. Public fixture XML requires no foreign asset download.
