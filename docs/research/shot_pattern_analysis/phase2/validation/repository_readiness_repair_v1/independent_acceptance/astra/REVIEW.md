# Independent Validation Helper Repair Review

## Verdict

The current18-file candidate is qualified for the scoped helper/type/lint repair, not for whole publication acceptance. The original17 executed files and their historical receipts remain preserved. No article, scientific module, producer source, raw data, existing receipt or generated graphic was changed by this review. Historical scripts with top-level writes were not executed.

## Findings and Corrections

The AGY candidate's green test count overstated meaningful contract coverage: most new tests inspected spellings, annotations or variable names. Those tests were removed. Isolated executable AST extraction remains solely a safe way to run actual functions/guard blocks without historical top-level writes; assertions now test behavior, not source spelling.

Three article helpers rejected nonfinite displayed tokens but not nonfinite expected values. In particular, infinite expected values could produce infinite error and tolerance and be treated as matching, while NaN could evade a greater-than comparison. Independent tests reproduced9 failures (three values across three sites), then explicit finite-expected-value guards fixed them. Existing finite tolerances and precision boundaries were retained. Tests cover finite signed/scientific/zero inputs, nonfinite tokens and expected values, mismatch rejection and half-display-unit boundaries.

New E402/T201/BLE001 suppression comments violated the no-suppression instruction. They were removed, with ordinary delayed public imports after checkout path selection, stdout writes preserving print separators/newlines/flush, and named-logger exception reporting. There are no new checking exclusions or renamed-to-escape runnable scripts.

Scoped savefig interception is exercised on success and exception: the original method is called once with exact arguments/return and restored afterward. Dataclass instance/class behavior, absent/present scrollbar, import specification/loader, and numeric-reference guards are executed independently. Existing22 terminal/asset/caption harness tests and25 supplemental-audit tests remain green.

## Diff Review

All17 original-to-current diffs were reviewed through normalized AST output (normalized_diff.txt), with annotations removed for comparison. This is not a claim of whole-file AST equality. Changes fall into formatting/import sorting, type annotations, variable isolation, finite/optional input guards, scoped interception, ordinary output/logging, and strict ordered-pair zip validation. Numerical constants, formulas, display tolerance coefficients, source selectors, producer CLI, terminal boundary and actual target/figure arguments remain unchanged. The capture TypedDict and parameter names do not change saved figure arguments. Pure formatting/type changes do not renew historical acceptance.

The original archive remains exactSHA b8492d68ee960d84a71a3f30b019e10359da00e851011a3859a02ecc221f1642 with17 verified members. Before and after final tests, frozen producer source/input/native hashes matched the authoritative v6 start; tracked edits stayed within the owned scope. Current helper hashes belong to this NEW receipt, while old receipts retain their original executed-source hashes.

## Evidence and Limits

- AGY supplied static-check red logs but no pytest red log. Test-first chronology for its initial tests is unverified; it is not invented here.
- Independent first run:11 failures, of which two were an extraction-harness mistake selecting an unrelated file-open context. Preserved red.txt.
- Corrected isolated harness, unchanged candidate:9 genuine nonfinite-expected failures and51 passing cases; red_corrected_harness.txt.
- After repairs and additional valid precision/spec cases:114 passed,7 existing warnings; final_tests.txt. This is67 behavioral cases plus22 harness and25 audit tests.
- Scoped Mypy18 files passes; Ruff check and format check pass; all files remain under1200 lines; diff-check clean.
- Full pre-PR gate remains root's responsibility. The22 historical review harness tests still report missing suite markers under the existing report-only marker policy; this repair does not conceal that warning.

No simulation, bootstrap, production figure export, article compilation or top-level historical evidence writer was run during this helper review. The current operational review harness is newly source-bound; the original /tmp qualified harness and historical receipts are not silently rewritten.
