# PR #12278 Validation Follow-Up

This record supplements the original failed gate and incomplete rerun; it does
not replace them. The current CI unit job reports 13 failures and 25,132 passes.
Local failing-first repairs cover missing launcher help, companion registration,
isolated frozen-source fixture imports, and monolith-register freshness. A separate
peer-handoff split preserves all original bytes when its four parts are joined
with one newline between parts. No frozen report, simulation, manifest, article,
or publication receipt changed.

The focused suites pass 168 distinct tests, including 71 launcher/atlas boundary
tests after governed atlas regeneration and explicit contract review. The two peer tests were rerun after
formatting and remain included in that total. See `receipt.json` and compressed
logs for exact scopes. These results were collected at the recorded local base
with the scoped uncommitted changes; integration against the newer remote PR
head still requires validation.

Repository acceptance remains blocked by architecture budgets, the frozen
report size, owner-confirmed development-log debt, and unrelated unit failures.
The built-in LaTeX defect #12120 and deferred empirical qualification #11984
remain separate. No budget, check, origin guard, or branch protection was relaxed.

## Remaining Architecture Work

Sol classified the 62 study findings across 42 files: 16 findings in 15
historical audit/review scripts, 11 in five report/vector builders, and 35
in 22 live tool modules. Five additional findings concern shared physics.
The existing check explicitly includes Python under `docs/`; moving or
renaming those scripts would not constitute a source repair.

A lawful continuation can refactor current Python without changing the
original producer commit or frozen artifacts. Add preservation tests first;
then extract parser/audit/serialization helpers, report assembly helpers, and
GUI/plotting/scoring responsibilities. Use temporary saved-row fixtures and
mocked physics rather than rerunning the study or invoking publication CLIs.
Record separate transition evidence; never rehash historical receipts or
claim the refactored source produced the published release.
