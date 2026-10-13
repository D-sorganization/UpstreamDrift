# Course Scoring Refactor Review

Independently reviewed isolated commits `9d7035968d` and `6cea552a1f6cbe40a0cb99e0cbbcdeda8b98f21b`. The public report now delegates endpoint geometry, lie selection, and source-backed scoring to `course_score_vectors.py`. Interaction reports can consume the same per-shot vectors without copying the strokes-gained formula or course classification. The existing public Tools API knot cache remains the scoring authority.

## Exact Characterization

The independent reproducer loads the historical `scenario_scoring.py` directly from `6cea552a^`, verifies its source hash against the saved characterization fixture, and compares it with the refactored public function. All report fields, numerical values, bootstrap intervals, provenance, and wording match exactly for six saved reports and 30 additional seeded randomized reports. These cover driver, 7-iron, and pitching wedge with varying hole length, fairway width, and green radius. All three saved invalid/unsupported-input error messages also match exactly. This verifies the retained behavior on the tested domain; it is not an empirical qualification or proof over every possible input.

The source inspection confirms the original validation, boundary inclusion (`<=`), distance units, lie assignment, cache invocation, and per-pattern row order were moved intact. The new frozen result container holds the score arrays and associated course/baseline metadata. No flight was executed.

## Reproduction and Provenance

From the isolated bounds worktree (or a later worktree containing both commits):

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. /tmp/shot-pattern-validation-venv-20261009/bin/python /home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift/docs/research/shot_pattern_analysis/phase2/review_followups/check_course_scoring_refactor.py > NEW_RECEIPT.json
```

`course_scoring_refactor_review.json` binds the checked old source, new scorer/helper/provenance sources, fixture, and reproducer. Ruff and scoped Mypy pass for the reproducer. The canonical source inventory now covers the helper and legacy gear utility. The latter is a coverage extension with a separately reviewed behavior repair; it must not be represented as globally unchanged physics. No original final-run acceptance is replaced by this scoring-only evidence.
