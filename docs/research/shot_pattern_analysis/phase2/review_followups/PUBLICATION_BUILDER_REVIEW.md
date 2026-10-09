# Publication Builder Review

Reviewed report builder `f4cc243f06` and figure builder `f86caf2389` against the
saved-output contracts. This is publication/software review, not final-data or
empirical acceptance. Final numerical workers remained untouched.

## Corrected Figure Findings

1. Detail metadata called static club loft “nominal delivered loft.” Renamed
   this to static loft and reference speed to distinguish reference controls
   from the pattern-specific nominal solve.
2. The final figure pathway accepted any non-pilot stage and arbitrary matrix
   size/population. It now requires final stage, 324 within-lean or 72 fixed-target
   cells according to output kind, and 10,000 planned attempts per pattern.
3. Saved CSV/flight bytes were checked before rendering but reread afterward
   without a publication-time check. The builder now revalidates every
   manifest-listed input and compares plan/manifest identities before writing
   its success manifest. Changed bytes are rejected even if their manifest is
   rewritten consistently. A failed build can leave unaccepted image files;
   consumers must require the final success manifest. This is a quiescent-source
   contract, not an operating-system snapshot against adversarial concurrent writes.

Five new test cases failed before fixes. The final tests additionally exercise
real saved-byte mutation with both unchanged and rewritten manifests. No
numerical sources, scoring equations, or accepted receipts changed.

## Scientific and Presentation Review

Direct SD/curve comparison pairs hold club, profile, seed, covariance, target
protocol, and other recorded controls fixed. The plotting API uses shared
histogram axes and density scale within each comparison. Different standalone
figures should not be assumed to share axes. Raw overhead paths and landing
densities are separately indexed in `facebook_overhead.csv` and
`facebook_ready.csv`; unavailable comparisons retain explicit ledger rows.
Pitching-wedge overhead fallbacks name the changed lean rather than silently
substituting it in matched density comparisons.

Report prose correctly identifies supported-shot conditioning, paired support
intersections, rematched speed/face/path in lean comparisons, Monte Carlo-only
intervals, and hypothetical scoring limitations. Two report findings were sent
to the report owner: the direction-count “Unavailable” column excluded whole
unavailable cells, and unavailable inventory rows used planned sample count as
`total_n`, potentially implying shots had been attempted. Both were corrected in `2de9960b77` and independently verified: the
interval table explicitly covers completed cells, and inventory columns
distinguish planned from attempted shots (zero attempts for unavailable cells).

Final acceptance still requires complete data audit, rebuilt artifacts, and
visual inspection of the actual final plots/PDF. This source review does not
claim those pending gates passed or that GUI layout has been visually accepted.

## Follow-Up Software Review

The combined report/figure test suite passes all 20 tests. Targeted Mypy found
three typing errors in the figure builder: an unannotated endpoint dictionary
and a local name reused for list and dictionary results. An explicit annotation
and local rename correct these without changing runtime calculations.

Both publication builders now pass targeted Mypy under Python 3.12 and Ruff.
The typing-only figure changes have identical normalized calculation/control-flow
AST to the preceding revision. No new concrete scientific defect remains from
this source review; full-data and final-render acceptance remain pending.

## Issue #11988: Matched-Run Figure Metadata

The saved phase-two `summary.json` config has no `delivery_mode` field. The
figure builder had displayed `unspecified` despite the matched-run contract.
Commit `fc5435a951` now labels the mode `Matched Raw Landing` (the runner's
raw matched landing contract, with no post-simulation aim rotation) and adds
each pattern's saved nominal face, path and speed to the figure metadata.
The focused figure suite passes 14 tests; targeted Mypy under Python 3.12 and
Ruff pass. Pilot detail figures were rerendered and visually checked by two
reviewers; the captions fit, and titles, axes, legends and caveats remain
unclipped. Hashes and paths are in
[`PRESENTATION_PREVIEW_V8_REVIEW.json`](PRESENTATION_PREVIEW_V8_REVIEW.json).
This review covers pilot presentation only, not final numerical acceptance.
