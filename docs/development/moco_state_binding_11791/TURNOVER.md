# Native Moco State-Binding Turnover

## Ownership and Publication State

Root/codex owns the implementation and delegated publication to Astra on branch
`feat/feedback-moco-state-binding-11791`. Child #11949 was created before the
11:54 UTC quota halt; that first claim/lease attempt failed. Recovery was confirmed
at 12:54:19 UTC. Root's lease succeeded at 12:57:48 UTC through 14:57:48 UTC
(comment 6081353033, session `01a11e27-3343-7950-8a97-82d4d606e66d`).
Astra registered scoped publication presence; the inbox read remains incomplete
because of malformed/pagination-limited board evidence, not a claim of no peers.

Publication targets a ready stacked PR based on
`feat/feedback-opensim-bundle-11908` / #11917 at
`72db775afff0d265014ee7ce67f9c020b1adc452`. Keep the feature-base PR UNARMED.
The PR body supplies the exact publication identity and current hook results.
Parent #11791 and all scientific acceptance gates remain open. The native source
and immutable experiment evidence are preserved during publication.

## Implemented Boundary and Native Evidence

The public `MocoInitialBindings` boundary applies caller-declared complete named
continuous-state and scalar-control bounds before Moco solver initialization and
guess creation. Initial values remain fixed even when guesses disagree.
Existing controllers/PositionMotion and multi-control actuators require separate
policies. Native state serialization, model physiology and broader control
completeness are not claimed.

The implementation owner reports 31 tests passing with actual OpenSim 4.6,
including state/general/control readback, malformed and unknown same-cardinality
maps, extra control, boolean/nonfinite values, fixed bounds, real
`PrescribedController` early refusal and native six-control `BodyActuator`
refusal. The earlier missing-fiber-bounds and DLL-plugin-search failures are
retained as exact archived evidence. The documentation worker did not rerun
optimization or alter source/tests.

The implementation owner reports the expanded actual OpenSim regression at
**165 passed, 2 inapplicable contact-fixture skips**, separate from the 31 binding
tests. Root reports all five central pre-PR gates passed at 11:52 UTC after the
first recovery. After recovery, publication validation reran all five central gates successfully.
Its Python 3.13 test lane passed 17 tests and skipped 14 SDK-dependent tests; a
separate explicit OpenSim 4.6/Python 3.12 run passed all 31 scoped tests without
SDK skips. Context, fragment and manual-governance checks passed. Normal hooks
are recorded in the PR body; historical checks do not replace current validation.

Four corrected public-API experiments use 10/20/40/80 intervals. Every actual
native solve succeeds and independent F06 replay repeats exactly, after T01
serialization and reloading through its authority. Dense diagnostics retain all
original control knots, use 101/121/161/241 union samples and score the same 101
physical-time probes. Activation discrepancy improves to .0000929729 at 80
intervals, but position error and marker RMSE are not monotone. Teacher activation
mismatch at the final 80-mesh time is a separate .01802976. No scientific gate is
closed by these observations.

The original 21-file archive is preserved, including valid original-grid state
errors. Its dense diagnostics are explicitly superseded: the old plain 101-point
grid could omit native control knots. The corrected script/receipts/logs/bundles
and outputs add 33 exact files in `evidence/grid-preserving/`. All 54 manifest
entries retain actual byte identities; no old entry or receipt is rewritten.

The prior local preview remains at
`C:/Users/diete/Desktop/Motion_Matching_Previews/moco_binding_11791/`.
Its PNG/MP4/receipt bind the historical 20-mesh receipt. It was inspected but
must not be relabeled as corrected evidence. Its .1 s horizon plays over 5 s;
this is not real-time performance.

The corrected 80-mesh preview is separately retained at
`C:/Users/diete/Desktop/Motion_Matching_Previews/moco_binding_11949/` with
`native_state_comparison.png`, `.mp4` and `preview-receipt.json`. Its source receipt
hash is `da3c944109785a288aa9bffa9312a714d5658ef0207288f94dcd1e866861bcec`.
The documentation worker verified both artifact hashes/sizes (82,955-byte PNG,
179,067-byte MP4) and visually inspected the PNG's four state overlays. It
clearly preserves the teacher activation difference despite close marker/state
agreement. Playback remains 5 s for .1 s native time; this is a local synthetic
preview, not a released manual artifact or a real-time performance claim.

## Durable Files and Integrity

- Canonical chapter: `manuals/upstreamdrift/chapters/35-native-moco-initial-bindings.qmd`.
- Manual index and parent-owned calculation inventory blocker are updated.
- `DESIGN.md` describes API ordering, historical evidence and reproduction.
- `evidence/MANIFEST.json` hashes the byte-preserved scripts/receipts/logs and
  small synthetic model/TRC/STO/NPZ outputs.

Archived `.py.txt` files are historical execution evidence, not maintained
executable source. Do not refactor or format them while retaining the old script
hash. A new executable harness, altered checkout path, source change or runtime
change requires a new evidence identity. Raw receipt `.json.txt` and log suffixes
likewise preserve hashes; the maintained manifest can be formatted normally.
Evidence-local Git attributes preserve original CRLF bytes during staging and
checkout and mark immutable raw evidence as binary-style diffs; the manifest and
attribute rules remain ordinary reviewable text. This prevents whitespace
normalization from corrupting archived receipts. All 54 staged evidence payloads
were verified byte-identical to their manifest. Do not replace archived receipt
bytes with a normalized equivalent.

## Checks and Next Handoff

Pre-edit design-manual governance passed with 14 QMD sources and release blocked.
Post-edit documentation validation passed:

- Design-manual governance: 15 canonical QMD sources, zero admitted calculations,
  release still blocked.
- Documentation governance, size budget and catalog checks.
- Scoped title-case audit of the three new authored documents.
- Manual-governance contract tests: 11 passed.
- `git diff --check`, all 54 archived file hashes and the final receipts' artifact
  hashes. A Git clean-filter comparison confirms archived log bytes remain exact.

The full tracked title audit did not pass: it reported inherited title violations
in agent/skill documents, then stopped while printing a Unicode character under
Windows cp1252. The scoped new documents have no title violations. No unrelated
documents or title checker were changed. No local Prettier executable was
available; the documentation worker did not download one or run commit hooks.
No optimization/native experiment was rerun during this documentation task.

The real child fragment `changes/11949-native-moco-initial-bindings.md` carries
SPEC change-log/development-log collation; shared log rows are not manually
rewritten. SPEC's interface narrative records the new optional public boundary.
The curated context catalog has no registered Moco/tour-matching boundary to
renew; source review and native tests supply this change's evidence. Publication
runs current context/governance checks without fabricating a boundary review.

Astra owns normal commit/push and ready stacked PR creation under root's lease.
Preserve parents #11791/#11792 and epic #11784. Retarget only after the actual
parent merge graph is inspected and checks rerun; never arm a feature-base PR.
No successful documentation or hook check closes scientific acceptance.
Preserve all model variants and all six engines. Next scientific work needs
declared state tolerances, further mesh/native feasibility evidence, production
anatomy/passive readiness, actual capture bindings and training-only calibration,
native contact/grip, complete native state and independent full-horizon replay.
