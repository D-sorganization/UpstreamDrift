# Academic Density Child Publication Plan

This is a design review, not implementation or acceptance. The live parent must finish and pass its original frozen verifier before this procedure begins. Preserve its directory and receipt unchanged.

## Existing Boundaries

- `matched_supplement_pipeline.load_verified_supplement_artifacts` validates eight report artifacts, exact current reporting-source hashes, original/supplement source cells and transition evidence, every figure, and post-read hashes. Retain this contract unchanged.
- `matched_supplement_figures._render_pair_figures` already reconstructs sorted common indices across six populations, prepares point sets and creates academic density ledger entries. `_write_pair_density_figure` owns the offending layout. Reuse this pathway; do not duplicate mask/label/bin logic in a repair script. A small public academic-density-only entry point should delegate to shared preparation and writer logic, with the ordinary renderer using the same preparation. It must not call the statistical pair analyzer or bootstrap.
- `build_final_report.load_supplement_at_stable_receipt` and `assert_publication_inputs_unchanged` hardcode the parent loader and receipt name. `final_report_inputs.assemble_interpretation` also hashes that filename. These are the narrow report-consumer adaptation points.

## Separate Child Contract

Use a distinct `REPORT_SUPPLEMENT_PUBLICATION_REVISION.json`, schema `matched-supplement-publication-revision/1`, with a closed operation value `academic_pair_density_row_gutter`. Never emit a replacement pipeline receipt claiming all assets were generated under new sources.

Required fields:

1. Parent root and exact parent receipt SHA; parent eight-artifact hash map; frozen parent verifier source map, interpreter/runtime/native identity and successful verification receipt SHA.
2. Revised renderer source map and exact reviewed source delta/evidence hashes; revision tool/loader hashes; unchanged numerical/source-input identities.
3. Child artifact map: seven non-figure-ledger report artifacts copied byte-for-byte, plus the child ledger hash. The numerical comparisons, aliases and inventories remain identical bytes.
4. One origin record per graphic, keyed by the exact parent relative path: `reused` requires equal parent/child hashes; `academic_density_rerender` requires parent and child hashes, ordered pair identity, source protocols, target, common-six count, source CSV/manifest hashes and revised renderer identity. No other operation/path is allowed.
5. Expected and observed counts and a final before/after source/input verification result. Publish the receipt last, atomically, only after all checks and visual qualification succeed.

The expected complete inventory is 6,456 graphics: 6,144 pair assets plus 312 cell assets. Exactly 1,152 academic pair-density assets (384 comparisons × PNG/PDF/SVG) are revised; 5,304 assets are reused unchanged. Derive this closed set from the verified parent ledger and 384 comparison identities, then assert these totals. Reject missing/extra/duplicate entries and unavailable rows that claim files. Preserve every original figure-ledger metadata field except revised hashes; lineage lives in the separate revision receipt.

A new explicit child loader should invoke the **unchanged** parent public loader in a subprocess whose import root and full bound source map match the frozen parent. Do not dynamically import stale modules into the revised process, trust an old success Boolean, or disable the parent's current-source guard. Bind and recheck the parent verifier's source/runtime before and after invocation. Then verify the closed child contract and every child byte, and recheck parent, child and evidence after reading. If the frozen verifier is unavailable or differs, fail closed. A preserved source checkout is thus an explicit dependency, not a silent fallback.

Return the existing report/comparisons/figure-ledger shape plus an explicit publication identity containing both parent pipeline receipt and child revision receipt hashes. The final report should consume this identity at initial load and final endguard, report both hashes, and link only child graphics. Its numerical provenance remains the parent pipeline. Keep ordinary parent loading available with exactly its existing behavior; dispatch only on an explicit requested publication type, never on a failed parent validation.

## Failing Tests Before Implementation

- Parent receipt missing, non-final, altered, or verified under changed source/runtime must fail, even with self-consistent child hashes.
- Any numeric JSON/CSV/Markdown/alias byte differs from parent: fail.
- Missing/extra/duplicate origin, a changed reused graphic, density-social/overhead/cell path marked revised, or wrong extension: fail.
- Pair order/protocol/target/common count or source CSV changes: fail; preserve verified alias identities.
- Preserve numerical renderer inputs exactly: ordered indices, point arrays, histogram edges/densities/HDR thresholds, axis limits, fonts and dot selection; only decorated row gutter changes.
- Mutate parent input, receipt, child graphic, renderer or evidence during rendering/loading/report assembly: fail before final receipt/report publication.
- Correct full synthetic inventory passes; one revised academic density is selected by the real report figure selector with parent and child identities retained. Ordinary parent-loader tests remain unchanged and pass.
- Actual all-384 child output inventory/hash reconciliation and visual review; no bootstrap or simulation calls permitted by the revision driver.

## Scope and Cost

This approach rerenders 384 academic figures, not 384 statistics or all 2,408 PNGs. It adds one purpose-specific publication revision and verifier, not a generic cache. The numerical parent remains independently reproducible. Source hashes alone do not establish that a change is layout-only: numerical input/plot-array parity and the exact reviewed source delta are required evidence.

## Portable Command and Single Consumer Interface

Implement one purpose-specific command (proposed `python -m scripts.research.shot_pattern_analysis.revise_supplement_density`) with parent/output/review-evidence paths and explicit `--parent-source-root` and `--parent-python`. Resolve publication/evidence paths relative to a declared repository root; store relative references plus immutable Git commit and source hashes, not an irreproducible absolute worktree path. Reproduction may reconstruct the pinned parent commit in a fresh worktree, initialize the pinned vendor dependency, then verify the complete parent source/runtime/native map before use. Record the actual resolved execution path as diagnostic provenance only.

Require an absolute resolved interpreter executable, its byte hash and the qualified runtime/dependency identity; sanitize import-path overrides and run the fixed verifier module with the pinned checkout as its import root. Caller-supplied source paths must never authorize arbitrary replacement source. A verified source commit plus exact file hashes, executable/runtime/native checks and subprocess origin checks form the contract. No credentials or installation changes are involved.

Expose a single publication loader dispatch (for example `load_verified_supplement_publication`) outside the unchanged pipeline verifier. Return a stable publication identity with receipt filename/path/hash, parent pipeline receipt hash, and publication kind. Both initial report loading and final endguard consume this interface and identity; `assemble_interpretation` uses the supplied identity rather than choosing schema filenames itself. This avoids scattering parent/child schema conditionals through the builder. Dispatch is explicit and fail-closed; it never converts a failed ordinary parent validation into child acceptance.
